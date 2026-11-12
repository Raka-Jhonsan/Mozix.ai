"""
export_all.py
=============
Exports CLIP vision encoder, GRPO MLP selector, and SmolVLM2 language model
to ONNX format for Android deployment.

Run from the directory containing your patched components.py:
    python export_all.py
"""

import sys, os, gc
import torch
import numpy as np

# ── Make sure we use the local patched components.py ──────────────────────
# Works in both notebooks and scripts
try:
    _dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _dir = os.getcwd()   # notebook: use current working directory
sys.path.insert(0, _dir)
print(f'Using directory: {_dir}')

SMOLVLM_PATH = '/DATA/teaching/DL_group45/models/SmolVLM2-500M'
CLIP_PATH    = '/DATA/teaching/DL_group45/models/CLIP-ViT-B-32-Local'
MLP_CKPT     = '/DATA/teaching/DL_group45/Temp/grpo_mlp_final.pt'
OUT_DIR      = './onnx_exports'
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = 'cpu'  # always export on CPU for ONNX compatibility

# ═══════════════════════════════════════════════════════════════════════════
# 1. Export CLIP vision encoder
# ═══════════════════════════════════════════════════════════════════════════
print('\n── 1. Exporting CLIP vision encoder ──')
from transformers import CLIPModel, CLIPProcessor

clip = CLIPModel.from_pretrained(CLIP_PATH, local_files_only=True).to(DEVICE)
clip.eval()
for p in clip.parameters():
    p.requires_grad_(False)

dummy_img = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    clip.vision_model,
    dummy_img,
    f'{OUT_DIR}/clip_vision.onnx',
    input_names=['pixel_values'],
    output_names=['last_hidden_state', 'pooler_output'],
    dynamic_axes={'pixel_values': {0: 'batch'}},
    opset_version=14,
    do_constant_folding=True,
)
print(f'CLIP exported → {OUT_DIR}/clip_vision.onnx ✓')

del clip
gc.collect()

# ═══════════════════════════════════════════════════════════════════════════
# 2. Export MLP frame selector
# ═══════════════════════════════════════════════════════════════════════════
print('\n── 2. Exporting MLP frame selector ──')
from components import FrameSelectorMLP

mlp = FrameSelectorMLP(input_dim=513).to(DEVICE)

# Load your trained weights
ckpt = torch.load(MLP_CKPT, map_location=DEVICE)

# Handle if checkpoint saved the full GRPOFrameSelector or just the MLP state dict
if any(k.startswith('net.') for k in ckpt.keys()):
    mlp.load_state_dict(ckpt)
elif any(k.startswith('mlp.net.') for k in ckpt.keys()):
    # strip 'mlp.' prefix if saved from GRPOFrameSelector
    new_ckpt = {k.replace('mlp.', ''): v for k, v in ckpt.items()}
    mlp.load_state_dict(new_ckpt)
else:
    # try loading directly
    mlp.load_state_dict(ckpt)

mlp.eval()
for p in mlp.parameters():
    p.requires_grad_(False)

dummy_feat = torch.randn(8, 513)  # 8 frames, 513 features each

torch.onnx.export(
    mlp,
    dummy_feat,
    f'{OUT_DIR}/frame_selector.onnx',
    input_names=['features'],
    output_names=['scores'],
    dynamic_axes={'features': {0: 'num_frames'}},
    opset_version=14,
    do_constant_folding=True,
)
print(f'MLP exported → {OUT_DIR}/frame_selector.onnx ✓')

del mlp
gc.collect()

# ═══════════════════════════════════════════════════════════════════════════
# 3. Export SmolVLM2 language model decoder
# ═══════════════════════════════════════════════════════════════════════════
print('\n── 3. Exporting SmolVLM2 language model ──')
print('    (this may take a few minutes on CPU...)')

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    from transformers import AutoModel as AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained(
    SMOLVLM_PATH,
    torch_dtype=torch.float32,   # float32 for ONNX export
    device_map='cpu',
    local_files_only=True,
    trust_remote_code=True,
)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

# Export the language model decoder (text generation component)
lm = model.language_model
dummy_ids  = torch.zeros(1, 10, dtype=torch.long)
dummy_attn = torch.ones(1, 10,  dtype=torch.long)

torch.onnx.export(
    lm,
    (dummy_ids, dummy_attn),
    f'{OUT_DIR}/smolvlm2_lm.onnx',
    input_names=['input_ids', 'attention_mask'],
    output_names=['logits'],
    dynamic_axes={
        'input_ids':      {0: 'batch', 1: 'seq_len'},
        'attention_mask': {0: 'batch', 1: 'seq_len'},
    },
    opset_version=14,
    do_constant_folding=True,
)
print(f'SmolVLM2 LM exported → {OUT_DIR}/smolvlm2_lm.onnx ✓')

del model, lm
gc.collect()

# ═══════════════════════════════════════════════════════════════════════════
# 4. Validate all exports
# ═══════════════════════════════════════════════════════════════════════════
print('\n── 4. Validating exports ──')
import onnxruntime as rt

checks = [
    ('CLIP',
     f'{OUT_DIR}/clip_vision.onnx',
     {'pixel_values': np.random.randn(1, 3, 224, 224).astype(np.float32)}),
    ('MLP',
     f'{OUT_DIR}/frame_selector.onnx',
     {'features': np.random.randn(8, 513).astype(np.float32)}),
    ('SmolVLM2 LM',
     f'{OUT_DIR}/smolvlm2_lm.onnx',
     {'input_ids':      np.zeros((1, 10), dtype=np.int64),
      'attention_mask': np.ones((1,  10), dtype=np.int64)}),
]

all_ok = True
for name, path, inputs in checks:
    try:
        sess = rt.InferenceSession(path, providers=['CPUExecutionProvider'])
        out  = sess.run(None, inputs)
        print(f'  {name:<14} ✓  output shape: {out[0].shape}')
    except Exception as e:
        print(f'  {name:<14} ✗  ERROR: {e}')
        all_ok = False

print()
if all_ok:
    # Print file sizes
    for name, path, _ in checks:
        size_mb = os.path.getsize(path) / 1024**2
        print(f'  {name:<14} {size_mb:.1f} MB  →  {path}')
    print('\n✓ All exports validated — ready to copy to Android assets/')
else:
    print('✗ Some exports failed — check errors above')
