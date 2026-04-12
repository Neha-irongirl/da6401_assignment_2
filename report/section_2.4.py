import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import wandb
from PIL import Image

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_DIR, 'data')
CKPT_DIR = os.path.join(REPO_DIR, 'checkpoints')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

from data.pets_dataset import PetsDataset
from models.vgg11 import VGG11Encoder

# ══════════════════════════════════════════════════════════════
# Load trained model
# ══════════════════════════════════════════════════════════════
model = VGG11Encoder(num_classes=37).to(DEVICE)
ckpt  = torch.load(
    os.path.join(CKPT_DIR, 'classifier.pth'),
    map_location=DEVICE)

# Load weights
feat_w = {k.replace('features.', ''): v
          for k, v in ckpt.items()
          if k.startswith('features.')}
model.features.load_state_dict(feat_w, strict=False)
model.eval()
print("✅ Model loaded")

# ══════════════════════════════════════════════════════════════
# Get one dog image from dataset
# ══════════════════════════════════════════════════════════════
ds = PetsDataset(DATA_DIR, split='val')

# Find a dog image — species=2 means dog in Oxford Pets
# We just take first image for simplicity
sample = ds[0]
image_tensor = sample['image'].unsqueeze(0).to(DEVICE)
fname        = sample['fname']
print(f"Using image: {fname}")

# Also load original image for display
img_path = os.path.join(DATA_DIR, 'images', fname + '.jpg')
orig_img = np.array(Image.open(img_path).convert('RGB'))

# ══════════════════════════════════════════════════════════════
# Extract feature maps using hooks
# First conv  = features[0]
# Last conv before final pool = features[27]
# (Block 5 has: Conv[22],BN[23],ReLU[24],Conv[25],BN[26],ReLU[27],Pool[28])
# ══════════════════════════════════════════════════════════════
feature_maps = {}

def get_hook(name):
    def hook(module, input, output):
        feature_maps[name] = output.detach().cpu()
    return hook

# Register hooks
model.features[0].register_forward_hook(
    get_hook('first_conv'))   # first conv layer
model.features[27].register_forward_hook(
    get_hook('last_conv'))    # last conv before final pool

with torch.no_grad():
    _ = model(image_tensor)

first_fm = feature_maps['first_conv']  # (1, 64, 224, 224)
last_fm  = feature_maps['last_conv']   # (1, 512, 14, 14)

print(f"First conv feature map shape: {first_fm.shape}")
print(f"Last conv feature map shape : {last_fm.shape}")

# ══════════════════════════════════════════════════════════════
# Log to WandB
# ══════════════════════════════════════════════════════════════
wandb.init(
    project='da6401-assignment2',
    name='2.4-feature-map-visualization',
    config={
        'section'    : '2.4',
        'image'      : fname,
        'first_layer': 'features[0] - Conv2d(3,64)',
        'last_layer' : 'features[27] - ReLU after last Conv',
    }
)

# ── Plot 0: Original image ─────────────────────────────────────
fig0, ax0 = plt.subplots(figsize=(4, 4))
ax0.imshow(orig_img)
ax0.set_title(f'Input Image\n{fname}', fontsize=10)
ax0.axis('off')
plt.tight_layout()
wandb.log({
    'section_2_4/input_image': wandb.Image(
        fig0, caption=f'Input: {fname}')
})
plt.close()

# ── Plot 1: First conv layer — 16 filters ─────────────────────
fig1, axes = plt.subplots(4, 4, figsize=(12, 12))
fig1.suptitle(
    'Section 2.4 — First Conv Layer Feature Maps\n'
    'features[0]: Conv2d(3→64) — Detects edges & colors',
    fontsize=12, fontweight='bold'
)

for i in range(16):
    row, col = i // 4, i % 4
    fm = first_fm[0, i].numpy()
    # Normalize to 0-1 for display
    fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-8)
    axes[row, col].imshow(fm, cmap='viridis')
    axes[row, col].set_title(f'Filter {i+1}', fontsize=8)
    axes[row, col].axis('off')

plt.tight_layout()
wandb.log({
    'section_2_4/first_conv_feature_maps': wandb.Image(
        fig1,
        caption='First Conv Layer: 16 of 64 filters shown'
    )
})
plt.close()

# ── Plot 2: Last conv layer — 16 filters ──────────────────────
fig2, axes2 = plt.subplots(4, 4, figsize=(12, 12))
fig2.suptitle(
    'Section 2.4 — Last Conv Layer Feature Maps\n'
    'features[27]: ReLU after Conv2d(512→512) — '
    'Detects high-level semantic shapes',
    fontsize=12, fontweight='bold'
)

for i in range(16):
    row, col = i // 4, i % 4
    fm = last_fm[0, i].numpy()
    fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-8)
    axes2[row, col].imshow(fm, cmap='viridis')
    axes2[row, col].set_title(f'Filter {i+1}', fontsize=8)
    axes2[row, col].axis('off')

plt.tight_layout()
wandb.log({
    'section_2_4/last_conv_feature_maps': wandb.Image(
        fig2,
        caption='Last Conv Layer: 16 of 512 filters shown'
    )
})
plt.close()

# ── Plot 3: Side by side comparison ───────────────────────────
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))

# Original
axes3[0].imshow(orig_img)
axes3[0].set_title('Original Image', fontsize=11)
axes3[0].axis('off')

# First conv — average of all 64 filters
first_avg = first_fm[0].mean(dim=0).numpy()
first_avg = (first_avg - first_avg.min()) / \
            (first_avg.max() - first_avg.min() + 1e-8)
axes3[1].imshow(first_avg, cmap='viridis')
axes3[1].set_title(
    'First Conv Layer\n(avg of 64 filters)\nEdges & Textures',
    fontsize=11)
axes3[1].axis('off')

# Last conv — average of all 512 filters
last_avg = last_fm[0].mean(dim=0).numpy()
last_avg = (last_avg - last_avg.min()) / \
           (last_avg.max() - last_avg.min() + 1e-8)
axes3[2].imshow(last_avg, cmap='viridis')
axes3[2].set_title(
    'Last Conv Layer\n(avg of 512 filters)\nSemantic Shapes',
    fontsize=11)
axes3[2].axis('off')

plt.suptitle(
    'Section 2.4 — Feature Map Comparison',
    fontsize=13, fontweight='bold')
plt.tight_layout()
wandb.log({
    'section_2_4/comparison': wandb.Image(
        fig3,
        caption='Original vs First Conv vs Last Conv'
    )
})
plt.close()

# ── Log stats ──────────────────────────────────────────────────
wandb.log({
    'section_2_4/first_conv_mean': float(
        first_fm.mean()),
    'section_2_4/first_conv_std' : float(
        first_fm.std()),
    'section_2_4/last_conv_mean' : float(
        last_fm.mean()),
    'section_2_4/last_conv_std'  : float(
        last_fm.std()),
})

wandb.finish()
print("✅ Section 2.4 fully logged to WandB!")