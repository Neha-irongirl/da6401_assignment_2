import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as npcd
import matplotlib.pyplot as plt
import wandb

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_DIR, 'data')
CKPT_DIR = os.path.join(REPO_DIR, 'checkpoints')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# ── Load one batch of real images ──────────────────────────────
from data.pets_dataset import PetsDataset
from torch.utils.data import DataLoader

ds     = PetsDataset(DATA_DIR, split='val')
dl     = DataLoader(ds, batch_size=32, shuffle=False)
batch  = next(iter(dl))
images = batch['image'].to(DEVICE)
print(f"Loaded {images.shape[0]} images")

# ══════════════════════════════════════════════════════════════
# Model WITH BatchNorm — your trained VGG11Encoder
# ══════════════════════════════════════════════════════════════
from models.vgg11 import VGG11Encoder

model_bn  = VGG11Encoder(num_classes=37).to(DEVICE)
ckpt      = torch.load(
    os.path.join(CKPT_DIR, 'classifier.pth'),
    map_location=DEVICE)
feat_w    = {k.replace('features.', ''): v
             for k, v in ckpt.items()
             if k.startswith('features.')}
model_bn.features.load_state_dict(feat_w, strict=False)
model_bn.eval()
print(" Model WITH BatchNorm loaded")

# ══════════════════════════════════════════════════════════════
# Model WITHOUT BatchNorm
# ══════════════════════════════════════════════════════════════
class VGG11NoBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 3rd conv layer — index 6
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.features(x)

model_nobn = VGG11NoBN().to(DEVICE)
model_nobn.eval()
print(" Model WITHOUT BatchNorm created")

# ══════════════════════════════════════════════════════════════
# Capture activations via hooks
# WITH BN    : features[8] = 3rd Conv2d
# WITHOUT BN : features[6] = 3rd Conv2d
# ══════════════════════════════════════════════════════════════
activations = {}

def get_hook(name):
    def hook(module, input, output):
        activations[name] = output.detach().cpu().numpy().flatten()
    return hook

model_bn.features[8].register_forward_hook(get_hook('with_bn'))
model_nobn.features[6].register_forward_hook(get_hook('without_bn'))

with torch.no_grad():
    _ = model_bn(images)
    _ = model_nobn(images)

act_bn   = activations['with_bn']
act_nobn = activations['without_bn']

print(f"With BN    — mean:{act_bn.mean():.4f} "
      f"std:{act_bn.std():.4f}")
print(f"Without BN — mean:{act_nobn.mean():.4f} "
      f"std:{act_nobn.std():.4f}")

# ══════════════════════════════════════════════════════════════
# Log EVERYTHING to WandB only
# ══════════════════════════════════════════════════════════════
wandb.init(
    project='da6401-assignment2',
    name='2.1-batchnorm-activation-distribution',
    config={
        'section'   : '2.1',
        'layer'     : '3rd Conv Layer',
        'batch_size': 32,
        'split'     : 'val',
    }
)

# ── Plot 1: Side by side histogram ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    'Section 2.1 — 3rd Conv Layer Activation Distribution',
    fontsize=13, fontweight='bold'
)

axes[0].hist(act_bn, bins=100,
             color='steelblue', alpha=0.85)
axes[0].axvline(act_bn.mean(), color='red',
                linestyle='--', linewidth=2, label='mean')
axes[0].set_title(
    f'WITH BatchNorm\n'
    f'mean={act_bn.mean():.3f}  std={act_bn.std():.3f}')
axes[0].set_xlabel('Activation Value')
axes[0].set_ylabel('Frequency')
axes[0].legend()

axes[1].hist(act_nobn, bins=100,
             color='coral', alpha=0.85)
axes[1].axvline(act_nobn.mean(), color='red',
                linestyle='--', linewidth=2, label='mean')
axes[1].set_title(
    f'WITHOUT BatchNorm\n'
    f'mean={act_nobn.mean():.3f}  std={act_nobn.std():.3f}')
axes[1].set_xlabel('Activation Value')
axes[1].set_ylabel('Frequency')
axes[1].legend()

plt.tight_layout()
wandb.log({
    'section_2_1/activation_histogram': wandb.Image(
        fig,
        caption='3rd Conv Layer: With BN vs Without BN'
    )
})
plt.close()

# ── Plot 2: Overlaid histogram ─────────────────────────────────
fig2, ax = plt.subplots(figsize=(10, 5))
ax.hist(act_bn, bins=100, color='steelblue',
        alpha=0.6, label='With BatchNorm')
ax.hist(act_nobn, bins=100, color='coral',
        alpha=0.6, label='Without BatchNorm')
ax.axvline(act_bn.mean(), color='blue',
           linestyle='--', linewidth=2,
           label=f'BN mean={act_bn.mean():.3f}')
ax.axvline(act_nobn.mean(), color='red',
           linestyle='--', linewidth=2,
           label=f'No BN mean={act_nobn.mean():.3f}')
ax.set_title('Overlaid Activation Distributions — 3rd Conv Layer')
ax.set_xlabel('Activation Value')
ax.set_ylabel('Frequency')
ax.legend()
plt.tight_layout()

wandb.log({
    'section_2_1/activation_overlay': wandb.Image(
        fig2,
        caption='Overlaid: With BN vs Without BN'
    )
})
plt.close()

# ── Log scalar stats ───────────────────────────────────────────
wandb.log({
    'section_2_1/with_bn_mean'   : float(act_bn.mean()),
    'section_2_1/with_bn_std'    : float(act_bn.std()),
    'section_2_1/with_bn_max'    : float(act_bn.max()),
    'section_2_1/with_bn_min'    : float(act_bn.min()),
    'section_2_1/without_bn_mean': float(act_nobn.mean()),
    'section_2_1/without_bn_std' : float(act_nobn.std()),
    'section_2_1/without_bn_max' : float(act_nobn.max()),
    'section_2_1/without_bn_min' : float(act_nobn.min()),
})

# ── Log WandB table with stats ─────────────────────────────────
table = wandb.Table(
    columns=['Model', 'Mean', 'Std', 'Min', 'Max'],
    data=[
        ['With BatchNorm',
         round(float(act_bn.mean()), 4),
         round(float(act_bn.std()),  4),
         round(float(act_bn.min()),  4),
         round(float(act_bn.max()),  4)],
        ['Without BatchNorm',
         round(float(act_nobn.mean()), 4),
         round(float(act_nobn.std()),  4),
         round(float(act_nobn.min()),  4),
         round(float(act_nobn.max()),  4)],
    ]
)
wandb.log({'section_2_1/activation_stats_table': table})

wandb.finish()
print(" Section 2.1 fully logged to WandB!")