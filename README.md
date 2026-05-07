# Mozix.ai 
Production-oriented Android implementation of a lightweight Vision-Language Model (VLM) pipeline using ONNX assets and Kotlin.
---
## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Clone](#clone)
  - [Install LFS Assets](#install-lfs-assets)
  - [Open and Run](#open-and-run)
- [Model Asset Management](#model-asset-management)
- [Configuration Notes](#configuration-notes)
- [Development Workflow](#development-workflow)
- [Troubleshooting](#troubleshooting)
- [Performance Notes](#performance-notes)
- [Security and Privacy](#security-and-privacy)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
---
## Overview
`Mozix` is an Android application that integrates on-device vision-language inference components.  
The app is designed to:
- load model assets from the app package,
- run inference through Kotlin-side orchestration,
- expose results through a responsive Android UI.
This repository is suitable for research prototypes, demos, and incremental optimization toward real-time mobile AI features.
---
## Key Features
- **On-device inference workflow** (no mandatory cloud dependency for core flow)
- **Kotlin + Android app architecture** with clear separation of UI and inference orchestration
- **ONNX model asset support** via `app/src/main/assets/`
- **ViewModel-driven state management** for inference lifecycle
- **Extensible engine layer** to evolve runtime behavior and optimization strategies
---
## Architecture
High-level module responsibilities:
- **`MainActivity`**
  - Entry point for user interactions and UI navigation
  - Binds UI actions to inference state/events
- **`InferenceViewModel`**
  - Owns inference state and lifecycle orchestration
  - Coordinates calls to the engine and updates UI-observable state
- **`SmolVLMEngine`**
  - Core model integration layer
  - Handles model initialization, execution flow, and output handling
- **`assets/`**
  - ONNX and related binary resources required by runtime
---
## Tech Stack
- **Language:** Kotlin
- **Platform:** Android
- **Build:** Gradle
- **Model format:** ONNX (`.onnx`) and auxiliary binaries (`.bin`)
- **Version control for large files:** Git LFS
---
## Repository Structure
```text
Mozix.ai/
├─ app/
│  ├─ src/
│  │  └─ main/
│  │     ├─ AndroidManifest.xml
│  │     ├─ java/com/group45/smolvlm/
│  │     │  ├─ MainActivity.kt
│  │     │  ├─ InferenceViewModel.kt
│  │     │  └─ SmolVLMEngine.kt
│  │     └─ assets/
│  │        ├─ *.onnx
│  │        └─ *.bin
│  └─ build.gradle(.kts)
├─ gradle/
├─ settings.gradle(.kts)
└─ README.md
Actual files may evolve over time; the layout above reflects the current core structure.

Getting Started
Prerequisites
Install the following before setup:

Android Studio (latest stable recommended)
JDK 17+ (or version compatible with your Android Gradle Plugin)
Git
Git LFS (required for model assets)
Clone
git clone https://github.com/Raka-Jhonsan/SmolVLM-Android.git
cd SmolVLM-Android
Install LFS Assets
git lfs install
git lfs pull
If you are setting up tracking for new large assets:

git lfs track "*.onnx" "*.bin"
git add .gitattributes
Open and Run
Open project in Android Studio.
Allow Gradle sync to complete.
Connect a physical Android device (recommended for performance testing) or start an emulator.
Run the app from Android Studio.
Model Asset Management
Model files are expected under:

app/src/main/assets/*.onnx
app/src/main/assets/*.bin
Because GitHub blocks files larger than 100 MB in regular Git, keep these tracked via Git LFS.

Add new model assets
git lfs track "*.onnx" "*.bin"
git add .gitattributes
git add app/src/main/assets/*.onnx app/src/main/assets/*.bin
git commit -m "Add/update model assets via Git LFS"
git push
Verify tracked LFS files
git lfs ls-files
Configuration Notes
Ensure Android SDK and build tools are correctly installed in Android Studio.
Keep local-only IDE/system files out of commits using .gitignore, especially:
.idea/
.gradle/
**/build/
If model files are environment-specific, document expected checksums/versions for reproducibility.
Development Workflow
Recommended branch flow:

Create feature branch:

git checkout -b feature/<name>
Commit source updates:

git add app/src/main/java app/src/main/AndroidManifest.xml
git commit -m "Describe feature/fix"
If model assets changed, commit LFS changes separately:

git add .gitattributes app/src/main/assets/*.onnx app/src/main/assets/*.bin
git commit -m "Update model assets"
Push and open PR:

git push -u origin feature/<name>
Troubleshooting
Push rejected due to large file size (GH001)
A file >100 MB is being pushed through regular Git. Fix: Track it with Git LFS and recommit that file.

error: cannot pull with rebase: You have unstaged changes
Working directory is dirty. Fix: stash or commit changes before pulling/rebasing.

LFS clean filter failed / Unable to move ... .git\lfs\tmp ...
Common on Windows due to lock/permission/AV interference.

Try:

Close Android Studio and terminals using that file.
Ensure sufficient disk space.
Pause antivirus/OneDrive for repo folder temporarily.
Retry git add.
Run:
git lfs prune
git gc --prune=now
Non-fast-forward push rejection
Remote contains commits your local branch does not. Fix: fetch and rebase/cherry-pick onto origin/main, then push.

Performance Notes
For better mobile runtime performance:

Prefer smaller/quantized model variants where quality permits
Cache heavy initialization steps where safe
Avoid repeated asset reloading
Minimize UI-thread work during inference
Add timing logs around preprocessing, inference, and postprocessing
Security and Privacy
Review data handling policy before processing sensitive user media.
Avoid committing secrets, API keys, signing keys, or local config files.
Keep dependencies and tooling updated to latest stable secure versions.
Roadmap

 Improve model load and warm-up latency

 Add robust error/reporting around inference pipeline

 Add reproducible benchmark suite on target devices

 Expand UI for richer prompt/response workflows

 Add CI checks for build and code quality
Contributing
Contributions are welcome.

Fork repository
Create feature branch
Commit clear, scoped changes
Open pull request with:
change summary
test notes
screenshots/logs if UI/runtime behavior changed
Please keep PRs focused and include only relevant files.

License
Choose and add one license file (recommended: MIT or Apache-2.0).

Example:

MIT License
Copyright (c) 2026 ...
Until a license is added, all rights remain with the repository owner.

Acknowledgements
Android and Kotlin ecosystems
ONNX tooling/runtime ecosystem
Community contributors and testers
