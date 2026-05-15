# Mozix.ai
**On-Device AI Visual Assistant for Blind & Low-Vision Users**

> Real-time scene understanding, obstacle detection, and voice-driven answers — fully offline, fully private, on any modern Android device.

[![Platform](https://img.shields.io/badge/platform-Android-green)](https://android.com)
[![Language](https://img.shields.io/badge/language-Kotlin-blue)](https://kotlinlang.org)
[![Model](https://img.shields.io/badge/model-SmolVLM2--500M-purple)](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M)
[![Runtime](https://img.shields.io/badge/runtime-ONNX%20Runtime%20Mobile-orange)](https://onnxruntime.ai)
[![Latency](https://img.shields.io/badge/latency-1.83s-brightgreen)]()
[![Speedup](https://img.shields.io/badge/speedup-3.07×-brightgreen)]()

---

## Overview

Mozix.ai is a production-oriented Android application that runs a full Vision-Language Model (VLM) pipeline entirely on-device — no internet connection, no cloud server, no data leaving the phone.

Built for **Blind & Low-Vision (BLV) users**, the app lets users point their camera at any scene, ask a question by voice, and hear a spoken answer within ~20-30 seconds on a flagship Android device. It achieves this through a novel **GRPO-trained MLP frame selector** that reduces input tokens by 78.3% before the expensive VLM inference step, delivering a **3.07× latency speedup** over naive 50-frame baselines while losing less than 3% accuracy on MVBench benchmarks.

---

## Key Results(Compared on GPU)

| Metric | Baseline (50 frames) | Mozix Pipeline | Improvement |
|---|---|---|---|
| Avg tokens | 2,932 | 593 | 78.3% reduction |
| Avg latency | 4.32s | 1.83s | 3.07× speedup |
| Accuracy loss | — | ≤3% | on action tasks |
| Frames used | 50 | 8 (Top-K) | 84% fewer |
| Active RAM | — | 325 MB | on-device heap |
| Model footprint | — | ~720 MB | on-disk total |

---

## Features

### Home Mode
- **Gallery & Camera input** — pick photos/videos from gallery or capture directly in-app
- **4-tier Quality Mode selector** — Fast / Balanced / Accurate / Ultra, controlling sampled frames (N), selected frames (Top-K), and max output tokens
- **Voice question input** — tap microphone to speak any question
- **Quick-tap question chips** — preset prompts: "What is happening?", "Describe scene", "Identify objects"
- **Live streaming output** — tokens appear word-by-word as the model generates
- **Stochastic multi-turn sampling** — randomised frame sampling across follow-up questions for broader scene coverage
- **TTS auto-playback** — every answer is automatically spoken aloud via TextToSpeech as it is generated

### Live Navigation Mode
- **CameraX real-time feed** — continuous live camera stream with torch and zoom controls
- **Proactive narrator loop** — automatically describes surroundings every ~8.5 seconds without any user input; continues until navigation is turned off
- **Rotating 5-prompt cycle** — cycles through: describe environment, identify obstacles, read text, detect people, describe objects
- **Ultra-low latency K=2 selection** — live mode uses only Top-2 frames for minimum possible inference time
- **Voice interrupt override** — user can ask a specific question at any time, pausing the narrator loop

### History Screen
- Searchable Q&A log of all past sessions
- Bookmark important sessions
- Session replay and export

### Community Screen
- Local volunteer network across 5 assistance categories
- Request help nearby with accessibility focus

### Settings Screen
- Theme selection
- Answer length tiers
- Default quality mode
- User guide and onboarding

---

## Pipeline Architecture

The full inference pipeline has 6 stages. Stages 1–3 are lightweight and run on all N frames. Stages 4–6 are compute-heavy and run only on the Top-K selected frames.

```
Stage 1 — Frame Extraction (OpenCV)
  Video → N uniformly spaced frames (e.g. 16 for Balanced mode)
  Each frame resized to 512×512

Stage 2 — CLIP Encoding (clip_vision.onnx, FP32, ~300 MB)
  Each frame downscaled to 224×224
  Cut into 49 patches of 32×32 pixels
  Passed through ViT-B/32 (12 transformer blocks)
  Output: 512-dim embedding per frame
  Also encodes query text → 512-dim vector
  Cosine similarity gives per-frame relevance score

Stage 3 — GRPO MLP Frame Selector (frame_selector.onnx, FP32, <1 MB)
  Input per frame: 512-dim CLIP embedding + 1 motion score = 513-dim
  MLP architecture: 513 → 256 → 64 → 1 (ReLU activations, 148K params)
  Trained via Group Relative Policy Optimization on MVBench (action_antonym + nav)
  Reward signal: VQA accuracy on held-out questions
  Selection: greedy with redundancy penalty (λ=0.5) → diverse + relevant Top-K frames

Stage 4 — SmolVLM2 Vision Encoder (vision_encoder_q4.onnx, INT4, ~85 MB)
  Input: Top-K frames at 512×512
  256 patches per frame → projected to 960-dim (LLM hidden size)
  Pixel shuffle merges 2×2 patch groups → 64 tokens per frame
  Output: K × 64 × 960-dim visual tokens

Stage 5 — SmolVLM2 Language Decoder (decoder_merged_q4.onnx, INT4, ~240 MB)
  32 transformer layers, 5 attention heads (GQA), 960-dim hidden size
  Input: system prompt + visual tokens + user question (~420 tokens total)
  KV-cached greedy decode with repetition penalty
  Streams tokens live to UI

Stage 6 — Answer + TTS
  Text output displayed word-by-word
  TextToSpeech speaks answer aloud automatically
```

### Quality Modes

| Mode | N (sampled) | Top-K | Max Tokens | Use case |
|---|---|---|---|---|
| Fast | 12 | 4 | 80 | Budget phones, quick answers |
| Balanced | 16 | 6 | 120 | Default, most devices |
| Accurate | 24 | 8 | 200 | Flagship devices |
| Ultra | 32 | 12 | 300 | Best quality, slowest |

---

## Model Assets

| File | Format | Size | Purpose |
|---|---|---|---|
| `clip_vision.onnx` | FP32 | ~300 MB | Frame relevance scoring (ViT-B/32) |
| `frame_selector.onnx` | FP32 | <1 MB | GRPO MLP Top-K frame selection |
| `vision_encoder_q4.onnx` | INT4 | ~85 MB | SmolVLM2 vision encoder, 64 tokens/frame |
| `decoder_merged_q4.onnx` | INT4 | ~240 MB | SmolVLM2 language decoder, 32 layers |
| `embed_tokens.bin` | FP32 | ~189 MB | Token embedding table (memory-mapped) |
| `tokenizer.json` | JSON | ~0.5 MB | Pure Kotlin BPE tokenizer |

**Total on-disk: ~720 MB · Active heap during inference: ~325 MB**

`embed_tokens.bin` is memory-mapped (mmap) — the OS loads only accessed pages on demand, keeping heap usage low even on 4 GB RAM devices.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Kotlin |
| UI framework | Jetpack Compose |
| Architecture | MVVM + StateFlow |
| Camera | CameraX |
| Inference runtime | ONNX Runtime Mobile |
| Vision model | SmolVLM2-500M (INT4 quantized) |
| Frame scoring | CLIP ViT-B/32 (FP32) |
| Frame selection | GRPO-trained MLP |
| Voice input | RecognizerIntent |
| Voice output | Android TextToSpeech |
| Haptics | VibrationEffect |
| Large file storage | Git LFS |
| Build | Gradle (Kotlin DSL) |

---

## Repository Structure

```text
Mozix.ai/
├─ app/
│  ├─ src/
│  │  └─ main/
│  │     ├─ AndroidManifest.xml
│  │     ├─ java/com/group45/smolvlm/
│  │     │  ├─ MainActivity.kt          # UI, Jetpack Compose screens, navigation
│  │     │  ├─ InferenceViewModel.kt    # Inference lifecycle, state, quality modes
│  │     │  └─ SmolVLMEngine.kt         # ONNX pipeline, CLIP, GRPO, VLM decode
│  │     └─ assets/
│  │        ├─ clip_vision.onnx
│  │        ├─ frame_selector.onnx
│  │        ├─ vision_encoder_q4.onnx
│  │        ├─ decoder_merged_q4.onnx
│  │        ├─ embed_tokens.bin
│  │        └─ tokenizer.json
│  └─ build.gradle.kts
├─ gradle/
├─ settings.gradle.kts
└─ README.md
```

---

## Getting Started

### Prerequisites

- Android Studio (latest stable)
- JDK 17+
- Git + Git LFS
- Physical Android device recommended (Snapdragon 8 Gen 2 or newer for best performance)

### Clone

```bash
git clone https://github.com/Raka-Jhonsan/SmolVLM-Android.git
cd SmolVLM-Android
```

### Install LFS Assets

```bash
git lfs install
git lfs pull
```

### Open and Run

1. Open project in Android Studio
2. Allow Gradle sync to complete
3. Connect a physical Android device
4. Run from Android Studio

---

## Model Asset Management

Model files live under `app/src/main/assets/`. GitHub blocks files >100 MB in regular Git — all `.onnx` and `.bin` files must be tracked via Git LFS.

**Add new model assets:**
```bash
git lfs track "*.onnx" "*.bin"
git add .gitattributes
git add app/src/main/assets/*.onnx app/src/main/assets/*.bin
git commit -m "Add/update model assets via Git LFS"
git push
```

**Verify tracked files:**
```bash
git lfs ls-files
```

---

## Performance Notes

- **Flagship (Snapdragon 8 Gen 3):** ~1.6–2.0s end-to-end
- **Mid-range (Snapdragon 7 Gen 1):** ~3.5–4.5s
- **Budget (2021 device):** ~8–15s

Optimisations already applied:
- INT4 quantization on vision encoder and decoder (4× size reduction vs FP32)
- Memory-mapped token embeddings (189 MB file, minimal heap impact)
- GRPO frame selection cuts prefill attention from ~8.6B to ~56M operations
- KV-cache reuse across decode steps
- Quality modes let users trade speed for accuracy based on device capability

---

## Development Workflow

```bash
# Create feature branch
git checkout -b feature/<name>

# Commit source changes
git add app/src/main/java app/src/main/AndroidManifest.xml
git commit -m "Describe feature/fix"

# Commit model asset changes separately
git add .gitattributes app/src/main/assets/*.onnx app/src/main/assets/*.bin
git commit -m "Update model assets"

# Push and open PR
git push -u origin feature/<name>
```

---

## Troubleshooting

**Push rejected — large file (GH001)**
File >100 MB pushed through regular Git. Track it with Git LFS and recommit.

**LFS clean filter failed (Windows)**
Close Android Studio, ensure disk space, pause antivirus/OneDrive temporarily, then:
```bash
git lfs prune
git gc --prune=now
```

**App crashes mid-inference on low-RAM device**
The OS killed the process due to memory pressure. Use Fast quality mode to reduce active heap. Ensure no other heavy apps are running.

**Non-fast-forward push rejection**
```bash
git fetch origin
git rebase origin/main
git push
```

---

## Security and Privacy

- **100% offline** — no data ever leaves the device
- No API keys, no cloud endpoints, no telemetry
- All inference runs locally via ONNX Runtime Mobile
- Do not commit signing keys, secrets, or local config files

---

## Roadmap

- [ ] Expand to full 20 MVBench task categories
- [ ] LoRA fine-tuning of SmolVLM2 on accessibility-specific data
- [ ] Dynamic K selection based on query complexity
- [ ] Real-time temporal segment detection
- [ ] iOS deployment via CoreML / ONNX Mobile
- [ ] Community volunteer matching with LLM routing
- [ ] Reproducible benchmark suite across target devices
- [ ] CI checks for build and code quality

---

## Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit clear, scoped changes
4. Open a pull request with a change summary, test notes, and screenshots/logs if UI or runtime behaviour changed

Please keep PRs focused and include only relevant files.

---

## License

MIT License — see `LICENSE` file for details.

---

## Acknowledgements

- [SmolVLM2](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M) by HuggingFace
- [CLIP](https://github.com/openai/CLIP) by OpenAI
- [ONNX Runtime Mobile](https://onnxruntime.ai) by Microsoft
- [MVBench](https://github.com/OpenGVLab/Ask-Anything) benchmark dataset
- Android, Kotlin, and Jetpack Compose ecosystems
- CS-671 Deep Learning, IIT Mandi — Group 45
- Mentor: Sushovan
