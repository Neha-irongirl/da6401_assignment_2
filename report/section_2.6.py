import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from torch.utils.data import DataLoader
from PIL import Image

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_DIR, 'data')
CKPT_DIR = os.path.join(REPO_DIR, 'checkpoints')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

from data.pets_dataset import PetsDataset
from models.segmentation import VGG11UNet

# ══════════════════════════════════════════════════════════════
# Load segmentation model
# ══════════════════════════════════════════════════════════════
model = VGG11UNet(num_classes=3).to(DEVICE)
ckpt  = torch.load(
    os.path.join(CKPT_DIR, 'unet.pth'),
    map_location=DEVICE)
model.load_state_dict(ckpt, strict=False)
model.eval()
print("✅ UNet loaded")

# ══════════════════════════════════════════════════════════════
# Load 5 test images
# ══════════════════════════════════════════════════════════════
test_ds = PetsDataset(DATA_DIR, split='test')
test_dl = DataLoader(test_ds, batch_size=1,
                     shuffle=False, num_workers=0)

# ══════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════
def dice_score_per_class(pred_mask, true_mask,
                          num_classes=3, eps=1e-6):
    """Returns dice score for each class separately."""
    scores = []
    for c in range(num_classes):
        pred_c = (pred_mask == c).float()
        true_c = (true_mask == c).float()
        inter  = (pred_c * true_c).sum()
        score  = (2 * inter + eps) / \
                 (pred_c.sum() + true_c.sum() + eps)
        scores.append(score.item())
    return scores

def pixel_accuracy(pred_mask, true_mask):
    """Percentage of correctly classified pixels."""
    correct = (pred_mask == true_mask).sum().item()
    total   = true_mask.numel()
    return correct / total

def mask_to_color(mask):
    """
    Convert class mask to RGB color image.
    0=foreground → green
    1=background → black
    2=boundary   → red
    """
    h, w   = mask.shape
    color  = np.zeros((h, w, 3), dtype=np.uint8)
    color[mask == 0] = [0,   200, 0]    # green = foreground
    color[mask == 1] = [0,   0,   0]    # black = background
    color[mask == 2] = [200, 0,   0]    # red   = boundary
    return color

# ══════════════════════════════════════════════════════════════
# Run inference on 5 images
# ══════════════════════════════════════════════════════════════
results  = []
count    = 0

for batch in test_dl:
    if count >= 5:
        break

    imgs  = batch['image'].to(DEVICE)
    masks = batch['mask'][0]             # (224, 224)
    fname = batch['fname'][0]

    with torch.no_grad():
        logits    = model(imgs)          # (1, 3, 224, 224)
        pred_mask = logits.argmax(dim=1)[0].cpu()  # (224, 224)

    # Metrics
    dice_scores = dice_score_per_class(pred_mask, masks)
    mean_dice   = np.mean(dice_scores)
    pix_acc     = pixel_accuracy(pred_mask, masks)

    results.append({
        'fname'      : fname,
        'image'      : imgs[0].cpu(),
        'gt_mask'    : masks.numpy(),
        'pred_mask'  : pred_mask.numpy(),
        'dice_fg'    : dice_scores[0],
        'dice_bg'    : dice_scores[1],
        'dice_border': dice_scores[2],
        'mean_dice'  : mean_dice,
        'pix_acc'    : pix_acc,
    })

    print(f"Image {count+1}: {fname} | "
          f"Dice:{mean_dice:.4f} "
          f"PixAcc:{pix_acc:.4f}")
    count += 1

# ══════════════════════════════════════════════════════════════
# Log to WandB
# ══════════════════════════════════════════════════════════════
wandb.init(
    project='da6401-assignment2',
    name='2.6-segmentation-dice-vs-pixacc',
    config={
        'section'    : '2.6',
        'num_images' : 5,
        'num_classes': 3,
    }
)

# ── Plot: Original | GT Mask | Predicted Mask ──────────────────
for r in results:
    fname     = r['fname']
    gt_color  = mask_to_color(r['gt_mask'])
    pred_color = mask_to_color(r['pred_mask'])

    # Load original image
    img_path = os.path.join(DATA_DIR, 'images',
                             fname + '.jpg')
    orig_img = np.array(
        Image.open(img_path).convert('RGB').resize(
            (224, 224)))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(
        f'{fname}\n'
        f'Dice:{r["mean_dice"]:.4f}  '
        f'PixAcc:{r["pix_acc"]:.4f}',
        fontsize=10, fontweight='bold'
    )

    axes[0].imshow(orig_img)
    axes[0].set_title('Original Image', fontsize=9)
    axes[0].axis('off')

    axes[1].imshow(gt_color)
    axes[1].set_title(
        'Ground Truth Trimap\n'
        '🟢 FG  ⬛ BG  🔴 Border',
        fontsize=9)
    axes[1].axis('off')

    axes[2].imshow(pred_color)
    axes[2].set_title(
        'Predicted Trimap\n'
        '🟢 FG  ⬛ BG  🔴 Border',
        fontsize=9)
    axes[2].axis('off')

    plt.tight_layout()
    wandb.log({
        f'section_2_6/sample_{fname}': wandb.Image(
            fig,
            caption=f'{fname} | '
                    f'Dice:{r["mean_dice"]:.4f} | '
                    f'PixAcc:{r["pix_acc"]:.4f}'
        )
    })
    plt.close()

# ── Per image metrics table ────────────────────────────────────
table = wandb.Table(
    columns=['Image', 'Dice_FG', 'Dice_BG',
             'Dice_Border', 'Mean_Dice', 'Pixel_Accuracy']
)
for r in results:
    table.add_data(
        r['fname'],
        round(r['dice_fg'],     4),
        round(r['dice_bg'],     4),
        round(r['dice_border'], 4),
        round(r['mean_dice'],   4),
        round(r['pix_acc'],     4),
    )
wandb.log({'section_2_6/metrics_table': table})

# ── Log mean metrics ───────────────────────────────────────────
mean_dice_all = np.mean([r['mean_dice'] for r in results])
mean_pix_all  = np.mean([r['pix_acc']   for r in results])

wandb.log({
    'section_2_6/mean_dice_score'    : mean_dice_all,
    'section_2_6/mean_pixel_accuracy': mean_pix_all,
    'section_2_6/dice_vs_pixacc_gap' : mean_pix_all
                                       - mean_dice_all,
})

# ── Bar chart: Dice vs Pixel Accuracy ─────────────────────────
fig2, ax = plt.subplots(figsize=(10, 5))
x     = np.arange(len(results))
width = 0.35
names = [r['fname'].split('_')[0] + '_'
         + r['fname'].split('_')[-1]
         for r in results]

bars1 = ax.bar(x - width/2,
               [r['mean_dice'] for r in results],
               width, label='Dice Score',
               color='steelblue', alpha=0.8)
bars2 = ax.bar(x + width/2,
               [r['pix_acc'] for r in results],
               width, label='Pixel Accuracy',
               color='coral', alpha=0.8)

ax.set_title(
    'Section 2.6 — Dice Score vs Pixel Accuracy\n'
    'Pixel Accuracy is artificially high',
    fontsize=11, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=15, fontsize=8)
ax.set_ylabel('Score')
ax.set_ylim(0, 1)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(mean_dice_all, color='blue',
           linestyle='--', linewidth=1.5,
           label=f'Mean Dice={mean_dice_all:.3f}')
ax.axhline(mean_pix_all, color='red',
           linestyle='--', linewidth=1.5,
           label=f'Mean PixAcc={mean_pix_all:.3f}')
ax.legend()
plt.tight_layout()

wandb.log({
    'section_2_6/dice_vs_pixacc_chart': wandb.Image(
        fig2,
        caption='Dice Score vs Pixel Accuracy comparison'
    )
})
plt.close()

wandb.finish()
print("\n✅ Section 2.6 fully logged to WandB!")

# ── Print summary ──────────────────────────────────────────────
print(f"\n{'Image':<25} {'Dice':>8} {'PixAcc':>8}")
print("-" * 44)
for r in results:
    print(f"{r['fname']:<25} "
          f"{r['mean_dice']:>8.4f} "
          f"{r['pix_acc']:>8.4f}")
print(f"\nMean Dice Score   : {mean_dice_all:.4f}")
print(f"Mean Pixel Accuracy: {mean_pix_all:.4f}")
print(f"Gap (PixAcc-Dice)  : "
      f"{mean_pix_all - mean_dice_all:.4f}")