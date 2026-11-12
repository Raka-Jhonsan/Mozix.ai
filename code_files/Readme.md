# Code Files – Mozix.ai

This directory contains the research, training, and model-export code used in the **Mozix.ai** project. It implements a Group Relative Policy Optimization (GRPO)-based frame selection mechanism designed to improve the efficiency of video understanding using Vision-Language Models (VLMs).

The primary objective of these scripts is to select the most relevant video frames before passing them to the SmolVLM2 model, reducing computational overhead while maintaining answer accuracy.

## 1. Overview

Mozix.ai uses a lightweight Multi-Layer Perceptron (MLP) trained with GRPO to identify informative video frames.

The frame selection pipeline combines:

- **CLIP ViT-B/32:** Extracts visual embeddings from video frames.
- **Motion features:** Provide additional temporal information for frame selection.
- **GRPO-trained MLP:** Estimates the relevance of individual frames.
- **SmolVLM2-500M:** Evaluates selected frames and generates answers to visual questions.

The trained frame selector helps reduce the number of frames processed by the vision-language model, improving inference efficiency for on-device deployment.

## 2. Directory Structure

```text
code_files/
│
├── export_all.py
│
├── grpo_selector.py
│
├── nb2_training.ipynb
│
├── grpo_mlp_final.pt
│
└── Readme.md
```

## 3. File Descriptions

### 3.1 `grpo_selector.py`

This file implements the GRPO-based frame selection architecture.

It contains:

- **FrameSelectorMLP:** A lightweight neural network that predicts frame-selection probabilities.
- **GRPOFrameSelector:** Handles candidate generation, frame ranking, and selection.
- **generate_candidates():** Generates candidate frame subsets for policy optimization.
- **select():** Selects the top-K frames based on predicted probabilities.
- **compute_mcq_reward():** Calculates the reward based on multiple-choice question accuracy.
- **grpo_loss():** Implements the GRPO policy optimization objective.
- **train_grpo():** Trains the frame selection network using feedback from SmolVLM2.
- **GRPOQueryAwarePipeline:** Provides a query-aware inference pipeline.

The MLP uses a three-layer architecture:

```text
Input Features
      |
      v
Linear Layer
512 → 256
      |
     GELU
      |
      v
Linear Layer
256 → 64
      |
     GELU
      |
      v
Linear Layer
64 → 1
      |
    Sigmoid
      |
      v
Frame Selection Probability
```

The standalone implementation defaults to 512-dimensional CLIP embeddings. The training notebook also contains a 513-dimensional feature configuration that incorporates a motion score.

### 3.2 `nb2_training.ipynb`

This notebook implements the two-stage training procedure for the GRPO-based frame selector.

It uses the MVBench dataset and a frozen SmolVLM2 model to provide reward signals during training.

**Stage 1 – Action Recognition**

- Dataset: MVBench `action_antonym`
- Focus: Identifying action-relevant frames in short video clips.
- Training epochs: 2
- Learning rate: 1e-4

**Stage 2 – Navigation Fine-tuning**

- Dataset: MVBench `egocentric_navigation`
- Focus: Improving frame selection for navigation and temporal reasoning tasks.
- Training epochs: 3
- Learning rate: 5e-5

**Feature Extraction Process**

1. Uniformly sample 32 frames from each video.
2. Apply perceptual hash (pHash) deduplication.
3. Extract 512-dimensional CLIP visual embeddings.
4. Append a motion score to obtain 513-dimensional feature vectors.
5. Generate candidate frame subsets.
6. Evaluate candidates using SmolVLM2.
7. Calculate rewards using multiple-choice answer accuracy.
8. Update the MLP policy using the GRPO loss.

The final trained model is saved as:

```text
grpo_mlp_final.pt
```

### 3.3 `grpo_mlp_final.pt`

This file contains the trained PyTorch weights of the GRPO frame selection MLP.

It is used to initialize the trained frame selector during inference and ONNX model export.

The checkpoint corresponds to the final training stage and should be loaded using a compatible MLP architecture.

### 3.4 `export_all.py`

This script exports the trained models into ONNX format for deployment.

It exports three components:

| Exported Model | Output File | Purpose |
|---|---|---|
| CLIP Vision Encoder | `clip_vision.onnx` | Visual feature extraction |
| GRPO MLP Selector | `frame_selector.onnx` | Frame relevance scoring |
| SmolVLM2 Language Model | `smolvlm2_lm.onnx` | Language generation |

The script performs the following operations:

1. Loads the locally stored CLIP model.
2. Exports the CLIP vision encoder.
3. Loads the trained GRPO MLP checkpoint.
4. Exports the frame selection network.
5. Loads SmolVLM2 and exports its language model decoder.
6. Validates the exported models using ONNX Runtime.

The exported files are stored in:

```text
onnx_exports/
```

ONNX export uses opset version 14.

## 4. Technologies and Dependencies

The codebase uses the following technologies:

| Technology | Purpose |
|---|---|
| Python | Implementation and training |
| PyTorch | Neural network development and training |
| NumPy | Numerical operations |
| Hugging Face Transformers | CLIP and SmolVLM2 models |
| OpenCV | Video frame processing |
| Pillow | Image processing |
| ONNX | Model export |
| ONNX Runtime | Export validation and inference |

Install the main Python dependencies using:

```bash
pip install torch numpy transformers opencv-python pillow onnx onnxruntime
```

Use a compatible PyTorch installation for your hardware and CUDA environment.

## 5. Training Requirements

Before running the training notebook, ensure that the following resources are available:

- MVBench dataset.
- `action_antonym.json` and `egocentric_navigation.json`.
- Corresponding video files.
- Locally downloaded CLIP ViT-B/32 model.
- Locally downloaded SmolVLM2-500M model.
- The project-specific `components.py` module and model-loading utilities used by the notebook.

The notebook uses local dataset and model paths. Update these paths according to your environment before execution.

## 6. Running the Training Pipeline

Open the notebook:

```text
nb2_training.ipynb
```

Execute the notebook cells sequentially.

The training process first trains the frame selector on action recognition data and subsequently fine-tunes it on navigation data.

The final checkpoint is generated as:

```text
grpo_mlp_final.pt
```

## 7. Exporting Models to ONNX

After training, configure the model and checkpoint paths in `export_all.py`.

The script expects the patched `components.py` module and the required local model checkpoints to be available.

Run:

```bash
python export_all.py
```

After successful execution, the following files should be generated:

```text
onnx_exports/
│
├── clip_vision.onnx
├── frame_selector.onnx
└── smolvlm2_lm.onnx
```

The script also validates the exported models using ONNX Runtime.

## 8. Role in Mozix.ai

These files support the machine learning component of Mozix.ai.

The GRPO-based frame selector acts as an optimization layer between video frame extraction and the Vision-Language Model.

By selecting a smaller set of informative frames, the system aims to:

- Reduce the number of visual tokens processed by the VLM.
- Improve inference efficiency.
- Reduce computational requirements.
- Support deployment on resource-constrained Android devices.
- Maintain strong visual question-answering performance.

The trained selector and exported ONNX components are intended to support the broader on-device inference pipeline implemented in the main Mozix.ai application.

---

**Note:** Model checkpoints, datasets, and local model directories must be configured separately. The training notebook and export script depend on project-specific components and model assets that are not all included in this folder.
