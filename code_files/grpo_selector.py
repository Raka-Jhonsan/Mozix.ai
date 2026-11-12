"""
grpo_selector.py
----------------
Drop-in replacement for the SMI frame selection step in QueryAwarePipeline.

Architecture:
  CLIP ViT-B/32 embeddings (512-dim, frozen)  ← you already compute these
       ↓
  Tiny MLP  (512 → 256 → 64 → 1)             ← ~170K trainable params
       ↓
  Per-frame keep-probabilities  pt ∈ (0,1)
       ↓
  Top-k sweep → K candidate 8-frame subsets
       ↓
  SmolVLM2-500M (frozen) scores each subset via MCQ accuracy
       ↓
  GRPO loss → update MLP only

Usage
-----
TRAINING:
    from grpo_selector import GRPOFrameSelector, train_grpo

    selector = GRPOFrameSelector()
    train_grpo(
        selector       = selector,
        smolvlm_model  = model,        # your existing frozen SmolVLM2
        smolvlm_proc   = processor,    # your existing processor
        clip_model     = clip_model,   # your existing CLIP model
        clip_processor = clip_processor,
        dataset        = list_of_dicts,  # [{"video_path":..,"question":..,"candidates":..,"answer":..}]
        epochs         = 3,
        save_path      = "grpo_mlp.pt",
    )

INFERENCE (inside QueryAwarePipeline.run_inference, replace SMI block):
    selector = GRPOFrameSelector(checkpoint="grpo_mlp.pt")
    top_k_indices = selector.select(frame_embeddings, k=8)
    selected_frames = [frames[i] for i in top_k_indices]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional
import time


# ---------------------------------------------------------------------------
# 1.  MLP Policy  (the only trainable component)
# ---------------------------------------------------------------------------

class FrameSelectorMLP(nn.Module):
    """
    Maps a CLIP frame embedding (512-dim) to a keep-probability in (0,1).
    Three linear layers with GELU activations, sigmoid output.
    Total parameters: ~170 K  (negligible for mobile deployment)
    """

    def __init__(self, input_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (T, 512)  per-frame CLIP embeddings
        returns: (T,)  keep probabilities
        """
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# 2.  GRPOFrameSelector  (wraps the MLP + candidate generation + inference)
# ---------------------------------------------------------------------------

class GRPOFrameSelector:
    """
    Wraps FrameSelectorMLP with:
      - top-k sweep candidate generation  (like HORNet)
      - greedy inference (single forward pass, pick top-k by probability)
      - GRPO training logic
    """

    def __init__(
        self,
        input_dim: int = 512,
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mlp = FrameSelectorMLP(input_dim).to(self.device)

        if checkpoint:
            state = torch.load(checkpoint, map_location=self.device)
            self.mlp.load_state_dict(state)
            print(f"[GRPOFrameSelector] Loaded checkpoint: {checkpoint}")

        total_params = sum(p.numel() for p in self.mlp.parameters())
        print(f"[GRPOFrameSelector] MLP parameters: {total_params:,}")

    # ------------------------------------------------------------------ #
    #  Candidate generation  (top-k sweep, same strategy as HORNet)       #
    # ------------------------------------------------------------------ #

    def generate_candidates(
        self,
        probs: torch.Tensor,   # (T,)
        k: int = 8,
        num_candidates: int = 8,
    ) -> List[List[int]]:
        """
        Produce `num_candidates` candidate subsets of size `k`.

        Strategy (mirrors HORNet):
          - Sort frames by probability (descending)
          - Candidate i selects frames ranked [0 .. k + step_i - 1]
            then takes the top-k from that window.
          - Last candidate is a stochastic Bernoulli draw for exploration.

        Returns list of lists of frame indices.
        """
        T = probs.shape[0]
        k = min(k, T)
        sorted_indices = torch.argsort(probs, descending=True).tolist()

        candidates = []
        step = max(1, (T - k) // max(num_candidates - 1, 1))

        for i in range(num_candidates - 1):
            window_size = k + i * step
            window_size = min(window_size, T)
            # take top-k from this window
            candidate = sorted(sorted_indices[:window_size][:k])
            if candidate not in candidates:
                candidates.append(candidate)

        # stochastic candidate for exploration
        with torch.no_grad():
            bernoulli_mask = torch.bernoulli(probs).bool()
            stoch_indices = bernoulli_mask.nonzero(as_tuple=True)[0].tolist()
            if len(stoch_indices) < k:
                # fill with top-prob frames
                extra = [i for i in sorted_indices if i not in stoch_indices]
                stoch_indices = sorted(stoch_indices + extra[: k - len(stoch_indices)])
            else:
                stoch_indices = sorted(stoch_indices[:k])
            if stoch_indices not in candidates:
                candidates.append(stoch_indices)

        # always include deterministic top-k as first candidate
        topk = sorted(sorted_indices[:k])
        if topk not in candidates:
            candidates.insert(0, topk)

        return candidates[:num_candidates]

    # ------------------------------------------------------------------ #
    #  Inference: single forward pass, return top-k indices               #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def select(
        self,
        frame_embeddings: np.ndarray,   # (T, 512)  from CLIP
        k: int = 8,
    ) -> List[int]:
        """
        Given CLIP frame embeddings, return indices of top-k frames.
        Drop-in replacement for SMI selection.
        """
        feats = torch.tensor(frame_embeddings, dtype=torch.float32).to(self.device)
        probs = self.mlp(feats)                          # (T,)
        topk_indices = torch.topk(probs, min(k, len(probs))).indices.tolist()
        return sorted(topk_indices)


# ---------------------------------------------------------------------------
# 3.  Reward function  (MCQ accuracy, same as HORNet stage-2 reward)
# ---------------------------------------------------------------------------

def compute_mcq_reward(prediction: str, ground_truth: str) -> float:
    """
    Binary MCQ reward: 1.0 if ground_truth string appears in prediction, else 0.0.
    Simple and directly comparable to HORNet's selection-accuracy reward.
    """
    return 1.0 if ground_truth.lower() in prediction.lower() else 0.0


def smolvlm_answer(
    frames,           # list of PIL Images (selected subset)
    question: str,
    candidates: List[str],
    smolvlm_model,
    smolvlm_processor,
    device: str,
) -> str:
    """
    Run SmolVLM2-500M on a set of selected frames and return the answer string.
    Mirrors your existing QueryAwarePipeline inference block exactly.
    """
    options_str = ", ".join(candidates)
    prompt = (
        f"Question: {question}\n"
        f"Options: {options_str}\n"
        f"Answer with the correct option directly."
    )
    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": frame} for frame in frames],
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = smolvlm_processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        do_image_splitting=False,
    ).to(smolvlm_model.device)

    with torch.no_grad():
        generated_ids = smolvlm_model.generate(**inputs, max_new_tokens=50)

    prediction = smolvlm_processor.batch_decode(
        generated_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )[0].strip()

    del inputs, generated_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return prediction


# ---------------------------------------------------------------------------
# 4.  GRPO training loop
# ---------------------------------------------------------------------------

def grpo_loss(
    probs: torch.Tensor,          # (T,)  current policy probabilities
    candidates: List[List[int]],  # K candidate subsets (lists of indices)
    rewards: List[float],         # K scalar rewards
    T: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    GRPO objective (eq. 7-9 from HORNet / DeepSeekMath).

    log π(b|F) = Σ_t [ b_t * log(p_t) + (1-b_t) * log(1-p_t) ]
    A_i = (r_i - mean(r)) / (std(r) + ε)
    L = -1/K * Σ_i A_i * log π(b_i|F)
    """
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    r_mean = rewards_tensor.mean()
    r_std  = rewards_tensor.std() + eps
    advantages = (rewards_tensor - r_mean) / r_std   # (K,)

    loss = torch.tensor(0.0, requires_grad=True)

    for i, mask_indices in enumerate(candidates):
        # Build binary mask
        b = torch.zeros(T, dtype=torch.float32)
        b[mask_indices] = 1.0

        # Log-probability of this mask under current policy
        log_p = (
            b * torch.log(probs + eps) +
            (1 - b) * torch.log(1 - probs + eps)
        ).sum()

        loss = loss + (-advantages[i] * log_p)

    return loss / len(candidates)


def train_grpo(
    selector: GRPOFrameSelector,
    smolvlm_model,
    smolvlm_processor,
    clip_model,
    clip_processor,
    dataset: List[Dict],          # list of {video_path, question, candidates, answer}
    epochs: int = 3,
    top_k: int = 8,
    num_candidates: int = 8,
    lr: float = 1e-4,
    save_path: str = "grpo_mlp.pt",
    log_every: int = 50,
):
    """
    Train the MLP policy with GRPO on your existing MVBench / VLNqa data.

    The SmolVLM2 and CLIP models remain completely frozen.
    Only selector.mlp is updated.

    Expected dataset format (same keys your benchmark already uses):
        [
          {
            "video_path": "/path/to/video.mp4",
            "question":   "What is the next action?",
            "candidates": ["Move forward", "Stop", ...],
            "answer":     "Stop",
          },
          ...
        ]
    """
    import cv2
    from PIL import Image
    import torch.nn.functional as F_nn

    device = selector.device
    optimizer = torch.optim.Adam(selector.mlp.parameters(), lr=lr)

    # ── helper: extract CLIP embeddings for a video ── #
    def get_clip_embeddings(video_path, n_frames=32):
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        actual = min(n_frames, total)
        indices = [int(round(i * (total - 1) / (actual - 1))) for i in range(actual)] if actual > 1 else [total // 2]
        frames_pil = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames_pil.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()

        if not frames_pil:
            return None, None

        clip_dev = next(clip_model.parameters()).device
        img_inputs = clip_processor(images=frames_pil, return_tensors="pt").to(clip_dev)
        with torch.no_grad():
            img_out   = clip_model.vision_model(**img_inputs)
            img_feats = img_out.pooler_output
            img_feats = clip_model.visual_projection(img_feats)
            img_feats = F_nn.normalize(img_feats, dim=-1)   # (T, 512)
        return frames_pil, img_feats.cpu().numpy()

    selector.mlp.train()
    total_samples = len(dataset)

    for epoch in range(1, epochs + 1):
        epoch_loss     = 0.0
        epoch_reward   = 0.0
        epoch_accuracy = 0.0

        for step, sample in enumerate(dataset, 1):
            video_path = sample["video_path"]
            question   = sample["question"]
            candidates_list = sample["candidates"]
            ground_truth    = sample["answer"]

            # ── 1. Extract frames + CLIP embeddings ── #
            frames_pil, embeddings = get_clip_embeddings(video_path)
            if embeddings is None:
                continue

            T = len(frames_pil)
            feats = torch.tensor(embeddings, dtype=torch.float32).to(device)

            # ── 2. Forward pass through MLP ── #
            probs = selector.mlp(feats)   # (T,) — keeps grad

            # ── 3. Generate K candidate subsets ── #
            with torch.no_grad():
                cands = selector.generate_candidates(probs.detach(), k=top_k, num_candidates=num_candidates)

            # ── 4. Score each candidate with frozen SmolVLM2 ── #
            rewards = []
            best_reward = 0.0
            for mask_indices in cands:
                selected_frames = [frames_pil[i] for i in mask_indices]
                pred = smolvlm_answer(
                    selected_frames, question, candidates_list,
                    smolvlm_model, smolvlm_processor, device,
                )
                r = compute_mcq_reward(pred, ground_truth)
                rewards.append(r)
                best_reward = max(best_reward, r)

            # ── 5. GRPO loss + backward ── #
            optimizer.zero_grad()
            loss = grpo_loss(probs, cands, rewards, T)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(selector.mlp.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss     += loss.item()
            epoch_reward   += float(np.mean(rewards))
            epoch_accuracy += best_reward

            if step % log_every == 0:
                avg_loss   = epoch_loss / step
                avg_reward = epoch_reward / step
                avg_acc    = epoch_accuracy / step
                print(
                    f"[Epoch {epoch}/{epochs}] Step {step}/{total_samples} | "
                    f"Loss: {avg_loss:.4f} | Avg Reward: {avg_reward:.3f} | "
                    f"Best-of-K Acc: {avg_acc:.3f}"
                )

        print(
            f"\n[Epoch {epoch} complete] "
            f"Loss: {epoch_loss/total_samples:.4f} | "
            f"Avg Reward: {epoch_reward/total_samples:.3f}\n"
        )
        # Save checkpoint after each epoch
        torch.save(selector.mlp.state_dict(), save_path.replace(".pt", f"_ep{epoch}.pt"))

    torch.save(selector.mlp.state_dict(), save_path)
    print(f"[Training complete] Final checkpoint saved to: {save_path}")
    selector.mlp.eval()
    return selector


# ---------------------------------------------------------------------------
# 5.  GRPOQueryAwarePipeline  — drop-in replacement for QueryAwarePipeline
# ---------------------------------------------------------------------------

class GRPOQueryAwarePipeline:
    """
    Identical interface to your existing QueryAwarePipeline.run_inference().
    Replaces the SMI selection step with the trained GRPO MLP selector.

    If no checkpoint is provided (or before training), it falls back to
    top-k by raw CLIP similarity — same as your current behaviour.
    """

    NUM_FRAMES = 32   # reduced from 50 (HORNet uses 32; saves ~36% CLIP time)
    TOP_K      = 8
    CLIP_PATH  = "/DATA/teaching/DL_group45/models/CLIP-ViT-B-32-Local"

    def __init__(
        self,
        smolvlm_model,
        smolvlm_processor,
        checkpoint: Optional[str] = None,
        num_frames: int = None,
        top_k: int = None,
    ):
        import cv2
        from transformers import CLIPProcessor, CLIPModel

        self.smolvlm_model     = smolvlm_model
        self.smolvlm_processor = smolvlm_processor
        self.num_frames        = num_frames or self.NUM_FRAMES
        self.top_k             = top_k or self.TOP_K
        self.device            = "cuda" if torch.cuda.is_available() else "cpu"

        # Load CLIP (same as your existing pipeline)
        self.clip_processor = CLIPProcessor.from_pretrained(self.CLIP_PATH, local_files_only=True)
        self.clip_model     = CLIPModel.from_pretrained(self.CLIP_PATH, local_files_only=True).to(self.device)
        self.clip_model.eval()

        # Load (or create untrained) GRPO selector
        self.selector = GRPOFrameSelector(input_dim=512, checkpoint=checkpoint, device=self.device)
        self.selector.mlp.eval()

        print(
            f"[GRPOQueryAwarePipeline] "
            f"pool={self.num_frames} → top-k={self.top_k} | "
            f"checkpoint={'loaded' if checkpoint else 'none (untrained)'}"
        )

    def _get_clip_embeddings(self, frames):
        """Encode PIL frames with CLIP. Returns (T, 512) numpy array."""
        import torch.nn.functional as F_nn
        img_inputs = self.clip_processor(images=frames, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out   = self.clip_model.vision_model(**img_inputs)
            feats = self.clip_model.visual_projection(out.pooler_output)
            feats = F_nn.normalize(feats, dim=-1)
        return feats.cpu().numpy()

    def _sample_frames(self, video_path):
        """Uniform frame sampling — identical to your FrameSampler."""
        import cv2
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS)
        dur   = total / fps if fps > 0 else 0
        actual = min(self.num_frames, total)
        indices = (
            [int(round(i * (total - 1) / (actual - 1))) for i in range(actual)]
            if actual > 1 else [total // 2]
        )
        frames, timestamps = [], []
        from PIL import Image
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                timestamps.append(idx / fps if fps > 0 else 0)
        cap.release()
        return frames, timestamps, total, fps, dur

    def run_inference(self, video_path, question, candidates, ground_truth):
        """
        Same return-dict structure as your existing QueryAwarePipeline.run_inference().
        """
        try:
            t0 = time.time()

            # Step 1: Sample frames
            frames, timestamps, total_frames, fps, duration = self._sample_frames(video_path)
            if not frames:
                return None

            # Step 2: CLIP embeddings
            embeddings = self._get_clip_embeddings(frames)   # (T, 512)

            # Step 3: GRPO MLP → top-k selection
            t_sel = time.time()
            top_k_indices = self.selector.select(embeddings, k=self.top_k)
            sel_time = time.time() - t_sel

            selected_frames = [frames[i] for i in top_k_indices]

            # Step 4: Build prompt + run SmolVLM2  (identical to your existing code)
            options_str = ", ".join(candidates)
            prompt = (
                f"Question: {question}\n"
                f"Options: {options_str}\n"
                f"Answer with the correct option directly."
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": f} for f in selected_frames],
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            t_tok = time.time()
            inputs = self.smolvlm_processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                do_image_splitting=False,
            ).to(self.smolvlm_model.device)
            tok_time = time.time() - t_tok

            num_input_tokens = inputs.input_ids.shape[1]

            t_gen = time.time()
            with torch.no_grad():
                generated_ids = self.smolvlm_model.generate(**inputs, max_new_tokens=50)
            gen_time = time.time() - t_gen

            prediction = self.smolvlm_processor.batch_decode(
                generated_ids[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )[0].strip()

            num_output_tokens = generated_ids.shape[1] - inputs.input_ids.shape[1]

            del inputs, generated_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            is_correct = ground_truth.lower() in prediction.lower()
            total_time = time.time() - t0

            return {
                "question":       question,
                "ground_truth":   ground_truth,
                "prediction":     prediction,
                "is_correct":     is_correct,
                "accuracy":       1 if is_correct else 0,

                "num_frames_pool":     len(frames),
                "num_frames_selected": len(top_k_indices),
                "top_k_indices":       top_k_indices,
                "total_video_frames":  total_frames,
                "video_duration_sec":  round(duration, 2),

                "num_input_tokens":  num_input_tokens,
                "num_output_tokens": num_output_tokens,
                "total_tokens":      num_input_tokens + num_output_tokens,

                "frame_selection_time_sec": round(sel_time, 3),
                "tokenization_time_sec":    round(tok_time, 3),
                "generation_time_sec":      round(gen_time, 3),
                "total_inference_time_sec": round(total_time, 3),

                "candidates": candidates,
                "approach":   "GRPO-MLP (pool={}, top-k={})".format(self.num_frames, self.top_k),
            }

        except Exception as e:
            print(f"Error in GRPOQueryAwarePipeline: {e}")
            import traceback; traceback.print_exc()
            return None
