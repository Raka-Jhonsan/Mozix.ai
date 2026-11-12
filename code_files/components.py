"""
components.py
==============
App-aligned building blocks for training and exporting Mozix.ai's GRPO
frame selector.

The Android app consumes one 513-dimensional feature vector per sampled frame:
512 L2-normalized CLIP visual features followed by one normalized motion score.
Its selector applies greedy temporal diversity with min_distance=max(1, N // 10).
This module mirrors that feature layout and selection rule.

SmolVLM2 and CLIP are frozen. Only FrameSelectorMLP is trainable.
"""

from __future__ import annotations

import gc
import re
from typing import List, Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

SMOLVLM_PATH = "/DATA/teaching/DL_group45/models/SmolVLM2-500M"
CLIP_PATH = "/DATA/teaching/DL_group45/models/CLIP-ViT-B-32-Local"

smolvlm_processor = None
smolvlm_model = None
clip_processor = None
clip_model = None

FEATURE_DIM = 513
CLIP_FEATURE_DIM = 512
APP_IMAGE_SIZE = 512


def load_models(
    smolvlm_path: str = SMOLVLM_PATH,
    clip_path: str = CLIP_PATH,
) -> None:
    """Load and freeze the local SmolVLM2-500M and CLIP ViT-B/32 models."""
    global smolvlm_processor, smolvlm_model, clip_processor, clip_model

    try:
        from transformers import AutoProcessor, AutoModelForImageTextToText
    except ImportError:
        from transformers import AutoProcessor, AutoModel as AutoModelForImageTextToText
    from transformers import CLIPProcessor, CLIPModel

    smolvlm_processor = AutoProcessor.from_pretrained(
        smolvlm_path, local_files_only=True, trust_remote_code=True
    )
    smolvlm_model = AutoModelForImageTextToText.from_pretrained(
        smolvlm_path,
        torch_dtype=DTYPE,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
    )
    smolvlm_model.eval()
    for parameter in smolvlm_model.parameters():
        parameter.requires_grad_(False)

    clip_processor = CLIPProcessor.from_pretrained(
        clip_path, local_files_only=True
    )
    clip_model = CLIPModel.from_pretrained(
        clip_path, local_files_only=True
    ).to(DEVICE)
    clip_model.eval()
    for parameter in clip_model.parameters():
        parameter.requires_grad_(False)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("SmolVLM2-500M loaded and frozen")
    print("CLIP ViT-B/32 loaded and frozen")


class FrameSampler:
    """
    Uniformly sample and square-resize video frames as the Android app does.

    Android samples N timestamps over [0, duration), then resizes each frame
    to 512x512. OpenCV frame-index sampling is an approximation of Android's
    nearest-sync-frame timestamp sampling.
    """

    NUM_FRAMES = 32
    IMAGE_SIZE = APP_IMAGE_SIZE

    @staticmethod
    def sample_frames_uniform(
        video_path: str,
        num_frames: Optional[int] = None,
    ) -> Optional[dict]:
        if num_frames is None:
            num_frames = FrameSampler.NUM_FRAMES
        if num_frames < 1:
            raise ValueError("num_frames must be at least 1")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            print(f"FrameSampler: cannot open video: {video_path}")
            return None

        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if total <= 0:
                return None

            actual = min(int(num_frames), total)
            # Mirrors Android's i * duration / N timestamps (does not force
            # the final sample to the video's last frame).
            indices = [
                min(total - 1, int(round(i * total / actual)))
                for i in range(actual)
            ]

            frames, timestamps, sampled = [], [], []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                square = cv2.resize(
                    rgb,
                    (FrameSampler.IMAGE_SIZE, FrameSampler.IMAGE_SIZE),
                    interpolation=cv2.INTER_LINEAR,
                )
                frames.append(Image.fromarray(square))
                timestamps.append(idx / fps if fps > 0 else 0.0)
                sampled.append(idx)

            if not frames:
                return None

            return {
                "frames": frames,
                "timestamps": timestamps,
                "frame_indices": sampled,
                "num_frames": len(frames),
                "total_frames": total,
                "fps": fps,
                "video_duration": total / fps if fps > 0 else 0.0,
            }
        except Exception as exc:
            print(f"FrameSampler error: {exc}")
            return None
        finally:
            cap.release()


def compute_motion_scores(frames: Sequence[Image.Image]) -> np.ndarray:
    """
    Match SmolVLMEngine.computeMotionScores:
    nearest-neighbour resize to at most 64x64, RGB absolute difference,
    mean over pixels/channels, then max-normalize to [0, 1].
    Frame zero receives a score of zero.
    """
    if not frames:
        return np.zeros((0,), dtype=np.float32)

    scores = np.zeros((len(frames),), dtype=np.float32)
    for index in range(1, len(frames)):
        width = min(frames[index].width, 64)
        height = min(frames[index].height, 64)
        size = (width, height)

        previous = np.asarray(
            frames[index - 1].resize(size, Image.Resampling.NEAREST).convert("RGB"),
            dtype=np.int16,
        )
        current = np.asarray(
            frames[index].resize(size, Image.Resampling.NEAREST).convert("RGB"),
            dtype=np.int16,
        )
        scores[index] = np.abs(current - previous).mean(dtype=np.float64)

    maximum = float(scores.max()) if scores.size else 0.0
    if maximum > 0.0:
        scores /= maximum
    return scores


def get_clip_embeddings(frames: Sequence[Image.Image]) -> np.ndarray:
    """Return L2-normalized (T, 512) CLIP visual-projection features."""
    if clip_model is None or clip_processor is None:
        raise RuntimeError("Call load_models() before extracting CLIP features.")

    device = next(clip_model.parameters()).device
    inputs = clip_processor(images=list(frames), return_tensors="pt").to(device)
    with torch.inference_mode():
        vision_output = clip_model.vision_model(**inputs)
        features = clip_model.visual_projection(vision_output.pooler_output)
        features = F.normalize(features, dim=-1)
    return features.float().cpu().numpy()


def get_clip_embeddings_with_motion(
    frames: Sequence[Image.Image],
) -> np.ndarray:
    """Return the app's (T, 513) feature layout: normalized CLIP + motion."""
    clip_features = get_clip_embeddings(frames)
    motion = compute_motion_scores(frames)
    if len(clip_features) != len(motion):
        raise RuntimeError("CLIP and motion feature counts do not match.")
    return np.concatenate(
        [clip_features.astype(np.float32), motion[:, None]], axis=1
    )


class FrameSelectorMLP(nn.Module):
    """The 513 -> 256 -> 64 -> 1 GELU MLP used by the app's selector."""

    def __init__(self, input_dim: int = FEATURE_DIM):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def select_diverse_indices(
    scores: Sequence[float] | np.ndarray | torch.Tensor,
    k: int,
    min_distance: Optional[int] = None,
) -> List[int]:
    """
    Mirror SmolVLMEngine.selectFrames: greedily rank by score, reject frames
    closer than min_distance, then fill any remaining slots without the filter.
    """
    if torch.is_tensor(scores):
        values = scores.detach().float().cpu().reshape(-1).tolist()
    else:
        values = np.asarray(scores, dtype=np.float32).reshape(-1).tolist()

    count = len(values)
    if count == 0 or k <= 0:
        return []
    k = min(int(k), count)
    if min_distance is None:
        min_distance = max(1, count // 10)

    ranked = sorted(range(count), key=lambda idx: (-float(values[idx]), idx))
    selected: List[int] = []

    for index in ranked:
        if len(selected) >= k:
            break
        if all(abs(previous - index) >= min_distance for previous in selected):
            selected.append(index)

    if len(selected) < k:
        for index in ranked:
            if index not in selected:
                selected.append(index)
                if len(selected) >= k:
                    break

    return sorted(selected)


class GRPOFrameSelector:
    """513-feature policy, app-matched diverse inference, and GRPO candidates."""

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.device = device or DEVICE
        self.input_dim = int(input_dim)
        self.mlp = FrameSelectorMLP(self.input_dim).to(self.device)

        if checkpoint:
            state = torch.load(
                checkpoint, map_location=self.device, weights_only=True
            )
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.mlp.load_state_dict(state)

        parameter_count = sum(p.numel() for p in self.mlp.parameters())
        print(f"[GRPOFrameSelector] input_dim={self.input_dim}, params={parameter_count:,}")

    def generate_candidates(
        self,
        probs: torch.Tensor,
        k: int = 8,
        num_candidates: int = 8,
    ) -> List[List[int]]:
        """
        Generate distinct fixed-size candidates using Bernoulli exploration.
        Every candidate uses the same temporal-diversity rule as Android.
        """
        probabilities = probs.detach().float().cpu().reshape(-1)
        frame_count = int(probabilities.numel())
        if frame_count == 0 or num_candidates <= 0:
            return []

        k = min(int(k), frame_count)
        if k == frame_count:
            return [list(range(frame_count))]

        candidates: List[List[int]] = [
            select_diverse_indices(probabilities, k)
        ]
        sampling_probs = probabilities.clamp(0.05, 0.95)

        attempts = max(32, num_candidates * 16)
        for _ in range(attempts):
            sampled_mask = torch.bernoulli(sampling_probs).bool()
            random_tiebreak = torch.rand_like(probabilities) * 1e-3
            sampled_scores = torch.where(
                sampled_mask,
                probabilities + random_tiebreak,
                torch.full_like(probabilities, -1.0),
            )
            candidate = select_diverse_indices(sampled_scores, k)
            if candidate not in candidates:
                candidates.append(candidate)
            if len(candidates) >= num_candidates:
                break

        # If Bernoulli samples repeat (e.g. a nearly saturated policy), use
        # random rankings to fill the requested group with unique subsets.
        for _ in range(attempts):
            if len(candidates) >= num_candidates:
                break
            candidate = select_diverse_indices(torch.rand_like(probabilities), k)
            if candidate not in candidates:
                candidates.append(candidate)

        return candidates[:num_candidates]

    @torch.inference_mode()
    def select(self, features: np.ndarray, k: int = 8) -> List[int]:
        """Run the policy and return the app-compatible diverse Top-K indices."""
        array = np.asarray(features, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected features with shape (T, {self.input_dim}); "
                f"received {array.shape}."
            )
        if len(array) == 0:
            return []

        tensor = torch.as_tensor(array, dtype=torch.float32, device=self.device)
        probabilities = self.mlp(tensor)
        return select_diverse_indices(probabilities, k)


def smolvlm_answer(
    frames: Sequence[Image.Image],
    question: str,
    candidates: Sequence[str],
) -> str:
    """Run the frozen SmolVLM2 model on the supplied candidate frames."""
    if smolvlm_model is None or smolvlm_processor is None:
        raise RuntimeError("Call load_models() before running SmolVLM2.")

    options = ", ".join(str(option) for option in candidates)
    prompt = (
        f"Question: {question}\n"
        f"Options: {options}\n"
        "Answer with the correct option directly."
    )
    messages = [{
        "role": "user",
        "content": [
            *[{"type": "image", "image": frame} for frame in frames],
            {"type": "text", "text": prompt},
        ],
    }]

    inputs = smolvlm_processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        do_image_splitting=False,
    ).to(smolvlm_model.device)

    with torch.inference_mode():
        generated = smolvlm_model.generate(**inputs, max_new_tokens=50)
    prediction = smolvlm_processor.batch_decode(
        generated[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )[0].strip()

    del inputs, generated
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return prediction


def _normalise_answer(value: str) -> str:
    value = str(value).casefold()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def compute_mcq_reward(prediction: str, ground_truth: str) -> float:
    """
    Binary MCQ reward with token/phrase boundaries, avoiding substring matches
    such as treating answer 'a' as present in the word 'walking'.
    """
    predicted = _normalise_answer(prediction)
    expected = _normalise_answer(ground_truth)
    if not predicted or not expected:
        return 0.0

    pattern = r"(?<![a-z0-9])" + re.escape(expected).replace(r"\ ", r"\s+") + r"(?![a-z0-9])"
    return 1.0 if re.search(pattern, predicted) else 0.0


def grpo_loss(
    probs: torch.Tensor,
    candidates: Sequence[Sequence[int]],
    rewards: Sequence[float],
    T: Optional[int] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    GRPO Bernoulli-mask objective. All tensors are created on probs.device;
    population standard deviation avoids NaNs for small candidate groups.
    """
    if not candidates or len(candidates) != len(rewards):
        return probs.sum() * 0.0

    reward_tensor = torch.as_tensor(
        rewards, dtype=probs.dtype, device=probs.device
    ).reshape(-1)
    if reward_tensor.numel() < 2:
        return probs.sum() * 0.0

    reward_std = reward_tensor.std(unbiased=False)
    if not torch.isfinite(reward_std) or reward_std.item() <= eps:
        return probs.sum() * 0.0

    advantages = (reward_tensor - reward_tensor.mean()) / (reward_std + eps)
    frame_count = int(T if T is not None else probs.numel())
    if frame_count != probs.numel():
        raise ValueError("T must equal the number of frame probabilities.")

    safe_probs = probs.clamp(eps, 1.0 - eps)
    log_keep = torch.log(safe_probs)
    log_drop = torch.log1p(-safe_probs)
    loss = probs.new_zeros(())

    for advantage, indices in zip(advantages, candidates):
        mask = torch.zeros_like(probs)
        if indices:
            mask[list(indices)] = 1.0
        log_probability = (
            mask * log_keep + (1.0 - mask) * log_drop
        ).sum()
        loss = loss - advantage * log_probability

    return loss / len(candidates)


__all__ = [
    "FEATURE_DIM",
    "CLIP_FEATURE_DIM",
    "DEVICE",
    "DTYPE",
    "FrameSampler",
    "FrameSelectorMLP",
    "GRPOFrameSelector",
    "compute_mcq_reward",
    "compute_motion_scores",
    "get_clip_embeddings",
    "get_clip_embeddings_with_motion",
    "grpo_loss",
    "load_models",
    "select_diverse_indices",
    "smolvlm_answer",
]
