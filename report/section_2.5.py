import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import wandb
from torch.utils.data import DataLoader
from PIL import Image

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_DIR, 'data')
CKPT_DIR = os.path.join(REPO_DIR, 'checkpoints')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

from data.pets_dataset import PetsDataset
from models.multitask import MultiTaskPerceptionModel

# ══════════════════════════════════════════════════════════════
# Load MultiTask model
# ══════════════════════════════════════════════════════════════
CLS_PATH = os.path.join(CKPT_DIR, 'classifier.pth')
LOC_PATH = os.path.join(CKPT_DIR, 'localizer.pth')
SEG_PATH = os.path.join(CKPT_DIR, 'unet.pth')

model = MultiTaskPerceptionModel(
    num_breeds=37,
    seg_classes=3,
    classifier_path=CLS_PATH,
    localizer_path=LOC_PATH,
    unet_path=SEG_PATH,
).to(DEVICE)
model.eval()
print("✅ Model loaded")

# ══════════════════════════════════════════════════════════════
# Load 10 test images
# ══════════════════════════════════════════════════════════════
test_ds = PetsDataset(DATA_DIR, split='test')
test_dl = DataLoader(test_ds, batch_size=1,
                     shuffle=False, num_workers=0)

# ══════════════════════════════════════════════════════════════
# IoU helper
# ══════════════════════════════════════════════════════════════
def compute_iou(pred, target):
    """Both in [cx, cy, w, h] format."""
    px1 = pred[0] - pred[2] / 2
    py1 = pred[1] - pred[3] / 2
    px2 = pred[0] + pred[2] / 2
    py2 = pred[1] + pred[3] / 2

    tx1 = target[0] - target[2] / 2
    ty1 = target[1] - target[3] / 2
    tx2 = target[0] + target[2] / 2
    ty2 = target[1] + target[3] / 2

    ix1 = max(px1, tx1)
    iy1 = max(py1, ty1)
    ix2 = min(px2, tx2)
    iy2 = min(py2, ty2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    pred_area   = pred[2]   * pred[3]
    target_area = target[2] * target[3]
    union = pred_area + target_area - inter + 1e-6

    return inter / union

# ══════════════════════════════════════════════════════════════
# Run inference on 10 images
# ══════════════════════════════════════════════════════════════
results = []
count   = 0

for batch in test_dl:
    if count >= 10:
        break

    imgs   = batch['image'].to(DEVICE)
    bboxes = batch['bbox'][0].numpy()   # GT bbox
    fname  = batch['fname'][0]

    with torch.no_grad():
        outputs = model(imgs)

    # Classification confidence
    cls_logits = outputs['classification']
    probs      = torch.softmax(cls_logits, dim=1)
    confidence = probs.max(dim=1).values.item()
    pred_class = probs.argmax(dim=1).item()

    # Predicted bbox
    pred_bbox = outputs['localization'][0].cpu().numpy()

    # IoU
    iou = compute_iou(pred_bbox, bboxes)

    results.append({
        'fname'     : fname,
        'gt_bbox'   : bboxes,
        'pred_bbox' : pred_bbox,
        'confidence': confidence,
        'iou'       : iou,
        'pred_class': pred_class,
    })

    print(f"Image {count+1:2d}: {fname} | "
          f"Conf:{confidence:.3f} IoU:{iou:.3f}")
    count += 1

# ══════════════════════════════════════════════════════════════
# Log to WandB
# ══════════════════════════════════════════════════════════════
wandb.init(
    project='da6401-assignment2',
    name='2.5-object-detection-confidence-iou',
    config={'section': '2.5', 'num_images': 10}
)

# ── Create bbox overlay images ─────────────────────────────────
wandb_images = []

for r in results:
    fname     = r['fname']
    gt_bbox   = r['gt_bbox']
    pred_bbox = r['pred_bbox']
    iou       = r['iou']
    conf      = r['confidence']

    # Load original image
    img_path = os.path.join(DATA_DIR, 'images',
                             fname + '.jpg')
    img = np.array(
        Image.open(img_path).convert('RGB').resize(
            (224, 224)))

    fig, ax = plt.subplots(1, figsize=(5, 5))
    ax.imshow(img)

    # ── Green box = Ground Truth ───────────────────────────
    gx1 = gt_bbox[0] - gt_bbox[2] / 2
    gy1 = gt_bbox[1] - gt_bbox[3] / 2
    gt_rect = patches.Rectangle(
        (gx1, gy1), gt_bbox[2], gt_bbox[3],
        linewidth=2, edgecolor='green',
        facecolor='none', label='Ground Truth'
    )
    ax.add_patch(gt_rect)

    # ── Red box = Prediction ───────────────────────────────
    px1 = pred_bbox[0] - pred_bbox[2] / 2
    py1 = pred_bbox[1] - pred_bbox[3] / 2
    pred_rect = patches.Rectangle(
        (px1, py1), pred_bbox[2], pred_bbox[3],
        linewidth=2, edgecolor='red',
        facecolor='none', label='Prediction'
    )
    ax.add_patch(pred_rect)

    ax.set_title(
        f'{fname}\nConf:{conf:.3f}  IoU:{iou:.3f}',
        fontsize=8)
    ax.legend(fontsize=7, loc='upper right')
    ax.axis('off')
    plt.tight_layout()

    wandb_images.append(
        wandb.Image(
            fig,
            caption=f'{fname} | '
                    f'Conf:{conf:.3f} | '
                    f'IoU:{iou:.3f}'
        )
    )
    plt.close()

# Log all images
wandb.log({
    'section_2_5/bbox_predictions': wandb_images
})

# ── WandB Table ────────────────────────────────────────────────
table = wandb.Table(
    columns=['Image', 'Confidence',
             'IoU', 'Pass@0.5',
             'GT_cx', 'GT_cy', 'GT_w', 'GT_h',
             'Pred_cx', 'Pred_cy', 'Pred_w', 'Pred_h']
)

for r in results:
    gt   = r['gt_bbox']
    pred = r['pred_bbox']
    table.add_data(
        r['fname'],
        round(r['confidence'], 4),
        round(r['iou'],        4),
        '✅' if r['iou'] >= 0.5 else '❌',
        round(float(gt[0]),   2),
        round(float(gt[1]),   2),
        round(float(gt[2]),   2),
        round(float(gt[3]),   2),
        round(float(pred[0]), 2),
        round(float(pred[1]), 2),
        round(float(pred[2]), 2),
        round(float(pred[3]), 2),
    )

wandb.log({'section_2_5/predictions_table': table})

# ── Find failure case ──────────────────────────────────────────
# High confidence but low IoU
failure = max(results,
              key=lambda x: x['confidence'] - x['iou'])

print(f"\n🔍 Failure case identified:")
print(f"   Image     : {failure['fname']}")
print(f"   Confidence: {failure['confidence']:.4f}")
print(f"   IoU       : {failure['iou']:.4f}")

wandb.log({
    'section_2_5/failure_case_image'     : failure['fname'],
    'section_2_5/failure_case_confidence': failure['confidence'],
    'section_2_5/failure_case_iou'       : failure['iou'],
})

# ── Summary stats ──────────────────────────────────────────────
ious  = [r['iou']        for r in results]
confs = [r['confidence'] for r in results]
wandb.log({
    'section_2_5/mean_iou'       : np.mean(ious),
    'section_2_5/mean_confidence': np.mean(confs),
    'section_2_5/acc_at_iou_0.5' : sum(
        1 for i in ious if i >= 0.5) / len(ious),
})

wandb.finish()
print("\n✅ Section 2.5 fully logged to WandB!")

# ── Print summary ──────────────────────────────────────────────
print(f"\n{'Image':<30} {'Conf':>6} {'IoU':>6} {'Pass':>5}")
print("-" * 52)
for r in results:
    passed = '✅' if r['iou'] >= 0.5 else '❌'
    print(f"{r['fname']:<30} "
          f"{r['confidence']:>6.3f} "
          f"{r['iou']:>6.3f} "
          f"{passed:>5}")
print(f"\nMean IoU        : {np.mean(ious):.4f}")
print(f"Mean Confidence : {np.mean(confs):.4f}")
print(f"Acc @ IoU=0.5   : "
      f"{sum(1 for i in ious if i>=0.5)/len(ious):.2%}")