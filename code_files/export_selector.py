"""
export_selector.py
==================
Export ONLY the GRPO MLP used by the Android app.

This intentionally does not export or replace CLIP, the SmolVLM2 vision encoder,
the decoder, or token embeddings. Those assets have separate Android-specific
interfaces and must not be overwritten by this selector exporter.

Example:
    python export_selector.py --checkpoint grpo_mlp_final.pt
    python export_selector.py --checkpoint grpo_mlp_final.pt --copy-to-app
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from components import FEATURE_DIM, FrameSelectorMLP


def load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError("Expected a PyTorch state_dict checkpoint.")
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("grpo_mlp_final.pt"))
    parser.add_argument("--output", type=Path, default=Path("onnx_exports/frame_selector.onnx"))
    parser.add_argument(
        "--copy-to-app",
        action="store_true",
        help="Copy the validated selector ONNX to app/src/main/assets/frame_selector.onnx.",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    model = FrameSelectorMLP(input_dim=FEATURE_DIM).to(device)
    model.load_state_dict(load_checkpoint(args.checkpoint, device))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(8, FEATURE_DIM, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        str(args.output),
        input_names=["features"],
        output_names=["scores"],
        dynamic_axes={"features": {0: "num_frames"}, "scores": {0: "num_frames"}},
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,
    )

    # Verify the exact input/output contract expected by SmolVLMEngine.kt.
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("Install onnxruntime to validate the exported model.") from exc

    session = ort.InferenceSession(
        str(args.output), providers=["CPUExecutionProvider"]
    )
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    if input_meta.name != "features" or output_meta.name != "scores":
        raise RuntimeError(
            f"Unexpected ONNX names: input={input_meta.name}, output={output_meta.name}"
        )
    if len(input_meta.shape) != 2 or input_meta.shape[1] != FEATURE_DIM:
        raise RuntimeError(f"Unexpected ONNX input shape: {input_meta.shape}")

    test_input = np.random.default_rng(7).standard_normal(
        (5, FEATURE_DIM), dtype=np.float32
    )
    onnx_scores = session.run(["scores"], {"features": test_input})[0]
    with torch.inference_mode():
        torch_scores = model(torch.from_numpy(test_input)).numpy()
    if not np.allclose(onnx_scores, torch_scores, rtol=1e-4, atol=1e-5):
        raise RuntimeError("ONNX output does not match the PyTorch selector.")

    print(f"Validated selector: {args.output}")
    print(f"Input: {input_meta.name} {input_meta.shape}")
    print(f"Output: {output_meta.name} {output_meta.shape}")
    print(f"Size: {args.output.stat().st_size / (1024 * 1024):.2f} MiB")

    if args.copy_to_app:
        app_asset = (
            Path(__file__).resolve().parent.parent
            / "app/src/main/assets/frame_selector.onnx"
        )
        app_asset.parent.mkdir(parents=True, exist_ok=True)
        app_asset.write_bytes(args.output.read_bytes())
        print(f"Copied validated selector to Android asset: {app_asset}")
        print("No other Android model asset was changed.")


if __name__ == "__main__":
    main()
