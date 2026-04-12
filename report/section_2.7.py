import sys, os
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(REPO_DIR, 'checkpoints')
REPORT_DIR = os.path.join(REPO_DIR, 'report')
WANDB_DIR = os.path.join(REPORT_DIR, 'wandb')
TMP_DIR = os.path.join(REPORT_DIR, '.tmp')
OUTPUT_DIR = os.path.join(REPORT_DIR, 'section_2_7_outputs')

os.makedirs(WANDB_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ.setdefault('WANDB_DIR', WANDB_DIR)
os.environ.setdefault('WANDB_CACHE_DIR', os.path.join(WANDB_DIR, 'cache'))
os.environ.setdefault('WANDB_CONFIG_DIR', os.path.join(WANDB_DIR, 'config'))
os.environ.setdefault('TMP', TMP_DIR)
os.environ.setdefault('TEMP', TMP_DIR)
tempfile.tempdir = TMP_DIR

import wandb
IMG_DIR_CANDIDATES = [
    os.path.join(REPO_DIR, 'report', 'wild_images'),
    os.path.join(REPO_DIR, 'reports', 'wild_images'),
    os.path.join(REPO_DIR, 'wild_images'),
    os.path.join(REPO_DIR, 'data', 'wild_images'),
    os.path.join(REPO_DIR, 'data', 'images'),
]

IMG_DIR = next((path for path in IMG_DIR_CANDIDATES
                if os.path.isdir(path)), IMG_DIR_CANDIDATES[0])

if not os.path.isdir(IMG_DIR):
    raise FileNotFoundError(
        "Could not find an image directory for section 2.7. "
        "Expected one of: "
        + ", ".join(IMG_DIR_CANDIDATES)
    )

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

from models.segmentation import VGG11UNet
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer

# ══════════════════════════════════════════════════════════════
# Load all 3 models separately
# ══════════════════════════════════════════════════════════════
# Classification
cls_model = VGG11Classifier(num_classes=37).to(DEVICE)
cls_ckpt  = torch.load(
    os.path.join(CKPT_DIR, 'classifier.pth'),
    map_location=DEVICE)
cls_model.load_state_dict(cls_ckpt, strict=False)
cls_model.eval()
print("[OK] Classifier loaded")

# Localization
loc_model = VGG11Localizer().to(DEVICE)
loc_ckpt  = torch.load(
    os.path.join(CKPT_DIR, 'localizer.pth'),
    map_location=DEVICE)
loc_model.load_state_dict(loc_ckpt, strict=False)
loc_model.eval()
print("[OK] Localizer loaded")

# Segmentation
seg_model = VGG11UNet(num_classes=3).to(DEVICE)
seg_ckpt  = torch.load(
    os.path.join(CKPT_DIR, 'unet.pth'),
    map_location=DEVICE)
seg_model.load_state_dict(seg_ckpt, strict=False)
seg_model.eval()
print("[OK] UNet loaded")

# ══════════════════════════════════════════════════════════════
# Preprocessing
# ══════════════════════════════════════════════════════════════
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])

def preprocess(img_path):
    """Load and preprocess image for inference."""
    orig = Image.open(img_path).convert('RGB')
    orig_resized = orig.resize((224, 224))
    img_np = np.array(orig_resized)
    tensor = transform(image=img_np)['image']
    return tensor.unsqueeze(0).to(DEVICE), \
           np.array(orig_resized)

# ══════════════════════════════════════════════════════════════
# Class names from Oxford Pets dataset
# ══════════════════════════════════════════════════════════════
CLASS_NAMES = [
    'Abyssinian', 'american_bulldog', 'american_pit_bull_terrier',
    'basset_hound', 'beagle', 'Bengal', 'Birman', 'Bombay',
    'boxer', 'British_Shorthair', 'chihuahua', 'Egyptian_Mau',
    'english_cocker_spaniel', 'english_setter', 'german_shorthaired',
    'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond',
    'leonberger', 'Maine_Coon', 'miniature_pinscher', 'newfoundland',
    'Persian', 'pomeranian', 'pug', 'Ragdoll', 'Russian_Blue',
    'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu',
    'Siamese', 'Sphynx', 'staffordshire_bull_terrier',
    'wheaten_terrier', 'yorkshire_terrier'
]

def mask_to_color(mask):
    """0=FG green, 1=BG black, 2=border red."""
    h, w  = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    color[mask == 0] = [0,   200, 0]
    color[mask == 1] = [0,   0,   0]
    color[mask == 2] = [200, 0,   0]
    return color

# ══════════════════════════════════════════════════════════════
# Run pipeline on each image
# ══════════════════════════════════════════════════════════════
image_files = [f for f in os.listdir(IMG_DIR)
               if f.endswith(('.jpg', '.jpeg', '.png'))]
image_files = sorted(image_files)[:3]  # take only 3

if not image_files:
    raise FileNotFoundError(
        f"No images found in {IMG_DIR}. "
        "Add .jpg/.jpeg/.png files to run section 2.7."
    )

print(f"\nUsing image directory: {IMG_DIR}")
print(f"\nFound images: {image_files}")

results = []
for img_file in image_files:
    img_path = os.path.join(IMG_DIR, img_file)
    print(f"\nProcessing: {img_file}")

    tensor, orig_img = preprocess(img_path)

    with torch.no_grad():
        # Classification
        cls_logits = cls_model(tensor)
        probs      = torch.softmax(cls_logits, dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs.max(dim=1).values.item()

        # Localization
        bbox = loc_model(tensor)[0].cpu().numpy()

        # Segmentation
        seg_logits = seg_model(tensor)
        pred_mask  = seg_logits.argmax(dim=1)[0].cpu().numpy()

    breed_name = CLASS_NAMES[pred_class] \
        if pred_class < len(CLASS_NAMES) \
        else f'class_{pred_class}'

    print(f"  Breed     : {breed_name} "
          f"(conf={confidence:.3f})")
    print(f"  BBox      : {bbox}")

    results.append({
        'fname'     : img_file,
        'orig_img'  : orig_img,
        'pred_mask' : pred_mask,
        'bbox'      : bbox,
        'breed'     : breed_name,
        'confidence': confidence,
        'pred_class': pred_class,
    })

# ══════════════════════════════════════════════════════════════
# Log to WandB
# ══════════════════════════════════════════════════════════════
def init_wandb_run():
    init_kwargs = {
        'project': 'da6401-assignment2',
        'name': '2.7-wild-images-pipeline',
        'dir': WANDB_DIR,
        'config': {
            'section': '2.7',
            'num_images': len(results),
        }
    }

    try:
        return wandb.init(**init_kwargs)
    except Exception as exc:
        print(f"[WARN] Online WandB init failed: {exc}")
        print("[INFO] Falling back to local file outputs.")
        return None


wandb_run = init_wandb_run()

for r in results:
    orig_img  = r['orig_img']
    pred_mask = r['pred_mask']
    bbox      = r['bbox']
    fname     = r['fname']

    # Color mask
    mask_color = mask_to_color(pred_mask)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Image: {fname}\n"
        f"Predicted Breed: {r['breed']} "
        f"(conf={r['confidence']:.3f})",
        fontsize=11, fontweight='bold'
    )

    # ── Panel 1: Original + BBox ───────────────────────────
    axes[0].imshow(orig_img)
    cx, cy, w, h = bbox
    x1 = cx - w / 2
    y1 = cy - h / 2
    rect = patches.Rectangle(
        (x1, y1), w, h,
        linewidth=2, edgecolor='red',
        facecolor='none'
    )
    axes[0].add_patch(rect)
    axes[0].set_title(
        f'BBox Prediction\n'
        f'[{cx:.0f},{cy:.0f},{w:.0f},{h:.0f}]',
        fontsize=9)
    axes[0].axis('off')

    # ── Panel 2: Segmentation mask ─────────────────────────
    axes[1].imshow(mask_color)
    axes[1].set_title(
        'Segmentation Mask\n'
        'FG=green  BG=black  Border=red',
        fontsize=9)
    axes[1].axis('off')

    # ── Panel 3: Overlay mask on image ─────────────────────
    overlay = orig_img.copy().astype(float)
    mask_float = mask_color.astype(float)
    blended = (0.6 * overlay + 0.4 * mask_float).astype(np.uint8)
    axes[2].imshow(blended)
    axes[2].set_title('Overlay', fontsize=9)
    axes[2].axis('off')

    plt.tight_layout()
    if wandb_run is not None:
        wandb.log({
            f'section_2_7/{fname}': wandb.Image(
                fig,
                caption=f'{fname} | '
                        f'Breed:{r["breed"]} | '
                        f'Conf:{r["confidence"]:.3f}'
            )
        })
    else:
        stem, _ = os.path.splitext(fname)
        fig.savefig(os.path.join(OUTPUT_DIR, f'{stem}_pipeline.png'))
    plt.close()

# ── Summary table ──────────────────────────────────────────────
if wandb_run is not None:
    table = wandb.Table(
        columns=['Image', 'Predicted Breed',
                 'Confidence', 'BBox_cx',
                 'BBox_cy', 'BBox_w', 'BBox_h']
    )
    for r in results:
        b = r['bbox']
        table.add_data(
            r['fname'],
            r['breed'],
            round(r['confidence'], 4),
            round(float(b[0]), 2),
            round(float(b[1]), 2),
            round(float(b[2]), 2),
            round(float(b[3]), 2),
        )
    wandb.log({'section_2_7/pipeline_results': table})
    wandb.finish()
    print("\n[OK] Section 2.7 fully logged to WandB!")
else:
    summary_path = os.path.join(OUTPUT_DIR, 'pipeline_results.csv')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('Image,Predicted Breed,Confidence,BBox_cx,BBox_cy,BBox_w,BBox_h\n')
        for r in results:
            b = r['bbox']
            f.write(
                f'{r["fname"]},{r["breed"]},{r["confidence"]:.4f},'
                f'{float(b[0]):.2f},{float(b[1]):.2f},'
                f'{float(b[2]):.2f},{float(b[3]):.2f}\n'
            )
    print("\n[OK] Section 2.7 completed with local outputs.")
    print(f"[INFO] Saved results to: {OUTPUT_DIR}")
