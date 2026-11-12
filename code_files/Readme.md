# Mozix.ai — GRPO Training Code

This folder contains the Python training and selector-export code for the frame-selection component used by the Mozix.ai Android app.

The Android implementation in \`app/src/main/java/com/group45/smolvlm/SmolVLMEngine.kt\` is the deployment reference. The Python feature pipeline and selector are designed to match that contract; the Android Kotlin files and VLM ONNX assets are not modified by the training workflow.

## Files

\`\`\`text
code_files/
├── components.py          # Shared sampling, features, MLP, GRPO loss and reward
├── nb2_training.ipynb     # Two-stage MVBench training and optional evaluation
├── grpo_mlp_final.pt      # Existing trained selector checkpoint
├── export_selector.py     # Exports only frame_selector.onnx
├── requirements.txt       # Python dependencies
└── Readme.md
\`\`\`

The old \`grpo_selector.py\` duplicated the selector with a 512-dimensional input, which did not match the Android app. The old \`export_all.py\` exported an incomplete SmolVLM2 graph and could overwrite assets whose interfaces are specific to the Android runtime. Both have been removed. The app's existing model assets are kept separately under \`app/src/main/assets/\`.

## Android compatibility contract

The app processes video frames in this order:

1. Sample a configurable number of frames uniformly from the video.
2. Resize sampled frames to 512 × 512.
3. Extract a 512-dimensional, L2-normalized CLIP visual-projection feature for each frame.
4. Compute one normalized RGB frame-difference motion score per frame.
5. Concatenate the CLIP feature and motion score into a **513-dimensional** feature vector.
6. Run the MLP and select Top-K frames using greedy temporal diversity.

The selector architecture is:

\`\`\`text
513 → 256 → GELU → 64 → GELU → 1 → Sigmoid
\`\`\`

It has 148,097 trainable parameters. During training, CLIP and SmolVLM2 are frozen; only the MLP is optimized.

### Feature details

- **CLIP:** \`CLIPModel.vision_model(...).pooler_output\` followed by \`visual_projection\` and L2 normalization. Output shape: \`(T, 512)\`.
- **Motion:** RGB absolute pixel difference between adjacent frames after nearest-neighbour resizing to at most 64 × 64; frame 0 receives 0; scores are normalized by the maximum. Output shape: \`(T,)\`.
- **Combined features:** \`[CLIP_512, motion_1]\`. Output shape: \`(T, 513)\`.

Python samples 32 frames by default and square-resizes them to 512 × 512, like the app's default Ultra pool. OpenCV frame-index sampling approximates Android's timestamp/nearest-sync-frame extraction; exact decoded frames may differ slightly between platforms.

### Frame-selection rule

The app sorts frames by selector score, greedily accepts a frame only when it is at least \`max(1, N // 10)\` indices away from previously selected frames, then fills any remaining slots from the ranked list. \`components.select_diverse_indices()\` mirrors this rule. GRPO training candidates use the same temporal-diversity constraint.

## Training

### 1. Prepare the environment

Use a Python environment with a PyTorch build appropriate for your CPU/GPU. Install the listed packages:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

The training machine must also have local copies of:

- SmolVLM2-500M
- CLIP ViT-B/32
- MVBench JSON files and the corresponding video folders

### 2. Configure paths

Open \`nb2_training.ipynb\` and update the paths in its configuration cell:

- \`MODEL_PATH\`
- \`CLIP_PATH\`
- \`DATA_ROOT\`
- Action JSON/video paths
- Navigation JSON/video paths

The defaults reflect the original project workstation layout and will need changing on another machine.

### 3. Run the notebook

Run the notebook cells in order. It:

1. Loads and freezes SmolVLM2 and CLIP once.
2. Loads the \`action_antonym\` and \`egocentric_navigation\` MVBench samples.
3. Samples frames and extracts app-aligned 513-dimensional features.
4. Trains on action-antonym (default: 2 epochs, learning rate 1e-4).
5. Fine-tunes the same selector on navigation (default: 3 epochs, learning rate 5e-5).
6. Saves stage and final PyTorch checkpoints.

Expected outputs:

\`\`\`text
grpo_mlp_stage1_ep1.pt
grpo_mlp_stage1_ep2.pt
grpo_mlp_stage1.pt
grpo_mlp_final_ep1.pt
...
grpo_mlp_final.pt
\`\`\`

The final checkpoint is a PyTorch state dict for \`FrameSelectorMLP(input_dim=513)\`. The notebook includes an optional single-pipeline evaluation cell. A baseline/SMI comparison is not included here because the corresponding \`query_and_baseline_clean.py\` pipelines are not part of this repository.

### Training notes

- The notebook no longer applies pHash deduplication: the Android app does not perform that preprocessing, so retaining it would change the temporal positions and feature distribution seen by the selector.
- Training samples use a 32-frame pool by default. The Android app's runtime quality modes use different frame counts and Top-K budgets.
- The reward is binary multiple-choice correctness. When an answer is an A/B/C/D label, the reward helper resolves it against the supplied candidate list before comparing it with the model output.
- Samples whose candidate group has no reward variation are skipped because their group-relative advantage is zero.
- Training runtime depends heavily on video I/O, GPU, and the number of VLM calls. No fixed runtime is claimed.

## Export the Android frame selector

After training (or when using the compatible checkpoint), export only the MLP:

\`\`\`bash
python export_selector.py --checkpoint grpo_mlp_final.pt
\`\`\`

The default output is:

\`\`\`text
onnx_exports/frame_selector.onnx
\`\`\`

The exporter checks that the ONNX interface matches the Android engine:

- Input name: \`features\`
- Input shape: \`[num_frames, 513]\`
- Output name: \`scores\`

It also compares ONNX output against PyTorch output on a test batch.

To copy the validated selector to the Android assets folder:

\`\`\`bash
python export_selector.py --checkpoint grpo_mlp_final.pt --copy-to-app
\`\`\`

**This command changes only \`app/src/main/assets/frame_selector.onnx\`.** It does not export or replace \`clip_vision.onnx\`, \`vision_encoder_q4.onnx\`, \`decoder_model_merged_q4.onnx\`, or \`embed_tokens.bin\`. Those files are part of the app-specific ONNX/tokenizer runtime and must be exported or prepared using the matching SmolVLM2 Android conversion process.

The repository currently tracks large Android model assets with Git LFS. Make sure Git LFS is installed and that the LFS objects have been pulled before building the app.

## Important app integration check

\`SmolVLMEngine.kt\` loads \`tokenizer.json\` from \`app/src/main/assets/\`. The current GitHub tree listing did not show that file, although the root README describes it. Verify that the tokenizer asset is present in your local checkout / release build before packaging the app. This training folder does not generate the tokenizer.

## Dependencies

See \`requirements.txt\`. ONNX export validation requires \`onnx\` and \`onnxruntime\`.

## What is intentionally not included

- MVBench videos or JSON data
- Local SmolVLM2 / CLIP weight directories
- Android app source (it remains under \`app/\`)
- The app's full vision encoder, language decoder, or token-embedding export pipeline
- The previous 512-feature selector implementation and its incompatible exporter

