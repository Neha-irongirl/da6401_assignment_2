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
EPOCHS     = 5       # short run — just to show the effect
BATCH_SIZE = 32
LR         = 1e-3
print(f"Using device: {DEVICE}")

from data.pets_dataset import PetsDataset
from models.classification import VGG11Classifier

# ── Data ───────────────────────────────────────────────────────
train_ds = PetsDataset(DATA_DIR, split='train')
val_ds   = PetsDataset(DATA_DIR, split='val')
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                      shuffle=True,  num_workers=0)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                      shuffle=False, num_workers=0)

# ══════════════════════════════════════════════════════════════
# Train one run for given dropout probability
# ══════════════════════════════════════════════════════════════
def train_one_run(dropout_p):
    print(f"\n--- Training with dropout_p={dropout_p} ---")

    # Load pretrained weights so training starts from
    # a good point — only dropout behaviour differs
    model = VGG11Classifier(num_classes=37,
                             dropout_p=dropout_p).to(DEVICE)

    ckpt  = torch.load(
        os.path.join(CKPT_DIR, 'classifier.pth'),
        map_location=DEVICE)
    model.load_state_dict(ckpt, strict=False)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):

        # Train
        model.train()
        train_loss = 0.0
        for batch in train_dl:
            imgs = batch['image'].to(DEVICE)
            lbls = batch['label'].to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                imgs = batch['image'].to(DEVICE)
                lbls = batch['label'].to(DEVICE)
                val_loss += criterion(model(imgs), lbls).item()

        tl = train_loss / len(train_dl)
        vl = val_loss   / len(val_dl)
        train_losses.append(tl)
        val_losses.append(vl)

        print(f"  Epoch {epoch+1}/{EPOCHS} | "
              f"Train:{tl:.4f} Val:{vl:.4f} "
              f"Gap:{vl-tl:.4f}")

    return train_losses, val_losses


# ══════════════════════════════════════════════════════════════
# Run all 3 conditions
# ══════════════════════════════════════════════════════════════
conditions = [
    (0.0, 'No_Dropout_p0.0'),
    (0.2, 'Dropout_p0.2'),
    (0.5, 'Dropout_p0.5'),
]

all_results = {}
for p, name in conditions:
    tl, vl = train_one_run(p)
    all_results[name] = {'train': tl, 'val': vl, 'p': p}

# ══════════════════════════════════════════════════════════════
# Log everything to WandB
# ══════════════════════════════════════════════════════════════
wandb.init(
    project='da6401-assignment2',
    name='2.2-dropout-generalization-gap',
    config={
        'section'   : '2.2',
        'epochs'    : EPOCHS,
        'lr'        : LR,
        'batch_size': BATCH_SIZE,
    }
)

colors = {
    'No_Dropout_p0.0': ('royalblue',  'skyblue'),
    'Dropout_p0.2'   : ('darkgreen',  'lightgreen'),
    'Dropout_p0.5'   : ('darkred',    'salmon'),
}

# ── Plot 1: Train vs Val Loss overlaid ─────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
epochs  = range(1, EPOCHS + 1)

for name, res in all_results.items():
    c_train, c_val = colors[name]
    ax.plot(epochs, res['train'],
            color=c_train, linewidth=2.5,
            marker='o', markersize=4,
            label=f'{name} — Train')
    ax.plot(epochs, res['val'],
            color=c_val, linewidth=2.5,
            marker='s', markersize=4,
            linestyle='--',
            label=f'{name} — Val')

ax.set_title(
    'Section 2.2 — Train vs Validation Loss\n'
    'No Dropout vs Dropout p=0.2 vs Dropout p=0.5',
    fontsize=12, fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Cross Entropy Loss')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()

wandb.log({
    'section_2_2/train_val_loss_curves': wandb.Image(
        fig,
        caption='Train vs Val Loss across Dropout conditions'
    )
})
plt.close()

# ── Plot 2: Generalization Gap ─────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 5))

for name, res in all_results.items():
    c_train, _ = colors[name]
    gap = [v - t for t, v in
           zip(res['train'], res['val'])]
    ax2.plot(epochs, gap,
             color=c_train, linewidth=2.5,
             marker='o', markersize=5,
             label=f'{name}  '
                   f'(final gap={gap[-1]:.4f})')

ax2.axhline(0, color='black', linestyle='-',
            linewidth=1, alpha=0.4,
            label='Zero gap (perfect generalization)')
ax2.set_title(
    'Section 2.2 — Generalization Gap\n'
    '(Val Loss - Train Loss)',
    fontsize=12, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Gap (Val Loss - Train Loss)')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
plt.tight_layout()

wandb.log({
    'section_2_2/generalization_gap': wandb.Image(
        fig2,
        caption='Generalization Gap: lower = less overfitting'
    )
})
plt.close()

# ── Log per epoch metrics ──────────────────────────────────────
for epoch in range(EPOCHS):
    log_dict = {'epoch': epoch + 1}
    for name, res in all_results.items():
        log_dict[f'train_loss/{name}'] = res['train'][epoch]
        log_dict[f'val_loss/{name}']   = res['val'][epoch]
        log_dict[f'gap/{name}']        = (res['val'][epoch]
                                          - res['train'][epoch])
    wandb.log(log_dict)

# ── Summary table ──────────────────────────────────────────────
table = wandb.Table(
    columns=['Condition', 'p',
             'Final Train Loss',
             'Final Val Loss',
             'Final Gap']
)
for name, res in all_results.items():
    table.add_data(
        name,
        res['p'],
        round(res['train'][-1], 4),
        round(res['val'][-1],   4),
        round(res['val'][-1] - res['train'][-1], 4)
    )
wandb.log({'section_2_2/summary_table': table})

wandb.finish()
print("\n Section 2.2 fully logged to WandB!")

# ── Print summary ──────────────────────────────────────────────
print(f"\n{'Condition':<20} {'Train':>8} "
      f"{'Val':>8} {'Gap':>8}")
print("-" * 48)
for name, res in all_results.items():
    t = res['train'][-1]
    v = res['val'][-1]
    print(f"{name:<20} {t:>8.4f} {v:>8.4f} {v-t:>8.4f}")