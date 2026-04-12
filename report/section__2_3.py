import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import wandb
from torch.utils.data import DataLoader

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_DIR, 'data')
CKPT_DIR = os.path.join(REPO_DIR, 'checkpoints')

DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS     = 10
BATCH_SIZE = 16
LR         = 1e-4
print(f"Using device: {DEVICE}")

from data.pets_dataset import PetsDataset
from models.segmentation import VGG11UNet

# ── Data ───────────────────────────────────────────────────────
train_ds = PetsDataset(DATA_DIR, split='train',
                        task='segmentation')
val_ds   = PetsDataset(DATA_DIR, split='val',
                        task='segmentation')
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                      shuffle=True,  num_workers=0)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                      shuffle=False, num_workers=0)

# ══════════════════════════════════════════════════════════════
# Dice Score helper
# ══════════════════════════════════════════════════════════════
def dice_score(pred_logits, true_mask,
               num_classes=3, eps=1e-6):
    pred_mask = pred_logits.argmax(dim=1)
    dice = 0.0
    for c in range(num_classes):
        pred_c = (pred_mask == c).float()
        true_c = (true_mask == c).float()
        inter  = (pred_c * true_c).sum()
        dice  += (2 * inter + eps) / \
                 (pred_c.sum() + true_c.sum() + eps)
    return (dice / num_classes).item()

# ══════════════════════════════════════════════════════════════
# Train one strategy
# ══════════════════════════════════════════════════════════════
def train_one_strategy(freeze_mode, strategy_name):
    print(f"\n{'='*50}")
    print(f"Strategy: {strategy_name}")
    print(f"{'='*50}")

    model = VGG11UNet(num_classes=3).to(DEVICE)

    # Load pretrained backbone
    ckpt_path = os.path.join(CKPT_DIR, 'classifier.pth')
    if os.path.exists(ckpt_path):
        model.load_pretrained_backbone(
            ckpt_path, freeze=freeze_mode)
    else:
        print("⚠️  No classifier checkpoint found")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad,
               model.parameters()),
        lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses, val_dices = [], [], []

    for epoch in range(EPOCHS):

        # ── Train ──
        model.train()
        train_loss = 0.0
        for batch in train_dl:
            imgs  = batch['image'].to(DEVICE)
            masks = batch['mask'].to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for batch in val_dl:
                imgs   = batch['image'].to(DEVICE)
                masks  = batch['mask'].to(DEVICE)
                logits = model(imgs)
                val_loss += criterion(logits, masks).item()
                val_dice += dice_score(
                    logits.cpu(), masks.cpu())

        tl = train_loss / len(train_dl)
        vl = val_loss   / len(val_dl)
        vd = val_dice   / len(val_dl)

        train_losses.append(tl)
        val_losses.append(vl)
        val_dices.append(vd)

        print(f"  Epoch {epoch+1:2d}/{EPOCHS} | "
              f"Train:{tl:.4f} Val:{vl:.4f} "
              f"Dice:{vd:.4f}")

    return train_losses, val_losses, val_dices


# ══════════════════════════════════════════════════════════════
# Run all 3 strategies
# ══════════════════════════════════════════════════════════════
strategies = [
    ('all',     'Frozen Backbone'),
    ('partial', 'Partial Fine-Tuning'),
    ('none',    'Full Fine-Tuning'),
]

all_results = {}
for freeze_mode, name in strategies:
    tl, vl, vd = train_one_strategy(freeze_mode, name)
    all_results[name] = {
        'train_loss': tl,
        'val_loss'  : vl,
        'val_dice'  : vd,
        'freeze'    : freeze_mode,
    }

# ══════════════════════════════════════════════════════════════
# Log to WandB
# ══════════════════════════════════════════════════════════════
wandb.init(
    project='da6401-assignment2',
    name='2.3-transfer-learning-showdown',
    config={
        'section'   : '2.3',
        'epochs'    : EPOCHS,
        'lr'        : LR,
        'batch_size': BATCH_SIZE,
    }
)

colors = {
    'Frozen Backbone'    : ('navy',      'skyblue'),
    'Partial Fine-Tuning': ('darkgreen', 'lightgreen'),
    'Full Fine-Tuning'   : ('darkred',   'salmon'),
}

epochs = range(1, EPOCHS + 1)

# ── Plot 1: Val Loss curves ────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
for name, res in all_results.items():
    c_train, c_val = colors[name]
    ax.plot(epochs, res['train_loss'],
            color=c_train, linewidth=2,
            linestyle='-',
            label=f'{name} — Train')
    ax.plot(epochs, res['val_loss'],
            color=c_val, linewidth=2,
            linestyle='--',
            label=f'{name} — Val')
ax.set_title(
    'Section 2.3 — Transfer Learning Strategies\n'
    'Train vs Val Loss',
    fontsize=12, fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Cross Entropy Loss')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
wandb.log({
    'section_2_3/loss_curves': wandb.Image(
        fig,
        caption='Train vs Val Loss: 3 Transfer Strategies'
    )
})
plt.close()

# ── Plot 2: Val Dice Score ─────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(11, 5))
for name, res in all_results.items():
    c_train, _ = colors[name]
    ax2.plot(epochs, res['val_dice'],
             color=c_train, linewidth=2.5,
             marker='o', markersize=4,
             label=f'{name} '
                   f'(best={max(res["val_dice"]):.4f})')
ax2.set_title(
    'Section 2.3 — Validation Dice Score\n'
    'Frozen vs Partial vs Full Fine-Tuning',
    fontsize=12, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Dice Score')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
wandb.log({
    'section_2_3/dice_curves': wandb.Image(
        fig2,
        caption='Val Dice Score: 3 Transfer Strategies'
    )
})
plt.close()

# ── Log per epoch metrics ──────────────────────────────────────
for epoch in range(EPOCHS):
    log_dict = {'epoch': epoch + 1}
    for name, res in all_results.items():
        safe = name.replace(' ', '_').replace('-', '_')
        log_dict[f'train_loss/{safe}'] = \
            res['train_loss'][epoch]
        log_dict[f'val_loss/{safe}']   = \
            res['val_loss'][epoch]
        log_dict[f'val_dice/{safe}']   = \
            res['val_dice'][epoch]
    wandb.log(log_dict)

# ── Summary table ──────────────────────────────────────────────
table = wandb.Table(
    columns=['Strategy', 'Freeze Mode',
             'Final Train Loss', 'Final Val Loss',
             'Best Dice', 'Final Dice']
)
for name, res in all_results.items():
    table.add_data(
        name,
        res['freeze'],
        round(res['train_loss'][-1], 4),
        round(res['val_loss'][-1],   4),
        round(max(res['val_dice']),  4),
        round(res['val_dice'][-1],   4),
    )
wandb.log({'section_2_3/summary_table': table})

wandb.finish()
print("\n Section 2.3 fully logged to WandB!")

# ── Print summary ──────────────────────────────────────────────
print(f"\n{'Strategy':<22} {'Train':>8} "
      f"{'Val':>8} {'BestDice':>10}")
print("-" * 52)
for name, res in all_results.items():
    print(f"{name:<22} "
          f"{res['train_loss'][-1]:>8.4f} "
          f"{res['val_loss'][-1]:>8.4f} "
          f"{max(res['val_dice']):>10.4f}")