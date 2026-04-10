
"""Training entrypoint"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import numpy as np
import wandb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.pets_dataset import PetsDataset
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from models.multitask import MultiTaskPerceptionModel
from losses.iou_loss import IoULoss

# ── Paths ───────────────────────────────────────────────────
DRIVE_BASE = '/content/drive/MyDrive/da6401_assignment_2'
CKPT_DIR   = f'{DRIVE_BASE}/checkpoints'
DATA_DIR   = '/content/da6401_assignment_2/data/data'
os.makedirs(CKPT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# HELPER — Dice Score
# ══════════════════════════════════════════════════════════════
def dice_score(pred_mask, true_mask, num_classes=3, eps=1e-6):
    pred_mask = pred_mask.argmax(dim=1)
    dice = 0.0
    for c in range(num_classes):
        pred_c = (pred_mask == c).float()
        true_c = (true_mask == c).float()
        inter  = (pred_c * true_c).sum()
        dice  += (2 * inter + eps) / \
                 (pred_c.sum() + true_c.sum() + eps)
    return (dice / num_classes).item()


# ══════════════════════════════════════════════════════════════
# TASK 1 — Classification
# ══════════════════════════════════════════════════════════════
def train_classification(args):
    print("\n" + "="*50)
    print("TASK 1 — VGG11 Classification")
    print("="*50)

    DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
    CKPT_PATH = f'{CKPT_DIR}/classifier.pth'

    wandb.init(
        project='da6401-assignment2',
        name=f'task1-cls-dropout{args.dropout_p}',
        config={
            'task'      : 'classification',
            'epochs'    : args.epochs,
            'lr'        : args.lr,
            'batch_size': args.batch_size,
            'dropout_p' : args.dropout_p,
        }
    )

    train_ds = PetsDataset(DATA_DIR, split='train')
    val_ds   = PetsDataset(DATA_DIR, split='val')
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True,  num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size,
                          shuffle=False, num_workers=2, pin_memory=True)

    model     = VGG11Classifier(num_classes=37,
                                 dropout_p=args.dropout_p).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    best_f1   = 0.0

    for epoch in range(args.epochs):
        model.train()
        train_loss, preds, labels = 0.0, [], []
        for batch in train_dl:
            imgs  = batch['image'].to(DEVICE)
            lbls  = batch['label'].to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds.extend(logits.argmax(1).cpu().numpy())
            labels.extend(lbls.cpu().numpy())

        train_f1 = f1_score(labels, preds,
                             average='macro', zero_division=0)

        model.eval()
        val_loss, val_preds, val_labels = 0.0, [], []
        with torch.no_grad():
            for batch in val_dl:
                imgs  = batch['image'].to(DEVICE)
                lbls  = batch['label'].to(DEVICE)
                logits = model(imgs)
                val_loss += criterion(logits, lbls).item()
                val_preds.extend(logits.argmax(1).cpu().numpy())
                val_labels.extend(lbls.cpu().numpy())

        val_f1 = f1_score(val_labels, val_preds,
                           average='macro', zero_division=0)
        scheduler.step()

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_loss/len(train_dl):.4f} "
              f"F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss/len(val_dl):.4f} "
              f"F1: {val_f1:.4f}")

        wandb.log({
            'epoch'     : epoch + 1,
            'train/loss': train_loss / len(train_dl),
            'train/f1'  : train_f1,
            'val/loss'  : val_loss   / len(val_dl),
            'val/f1'    : val_f1,
            'lr'        : scheduler.get_last_lr()[0],
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), CKPT_PATH)
            print(f"  ✅ Saved best model (val F1={val_f1:.4f})")

    wandb.finish()
    print(f"\nBest val F1: {best_f1:.4f}")
    print(f"Checkpoint : {CKPT_PATH}")
    return CKPT_PATH


# ══════════════════════════════════════════════════════════════
# TASK 2 — Localization
# ══════════════════════════════════════════════════════════════
def train_localization(args):
    print("\n" + "="*50)
    print("TASK 2 — VGG11 Localization")
    print("="*50)

    DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
    CKPT_PATH = f'{CKPT_DIR}/localizer.pth'
    CLS_PATH  = f'{CKPT_DIR}/classifier.pth'

    wandb.init(
        project='da6401-assignment2',
        name='task2-localization',
        config={
            'task'           : 'localization',
            'epochs'         : args.epochs,
            'lr'             : args.lr,
            'batch_size'     : args.batch_size,
            'freeze_backbone': args.freeze_backbone,
        }
    )

    train_ds = PetsDataset(DATA_DIR, split='train',
                            task='localization')
    val_ds   = PetsDataset(DATA_DIR, split='val',
                            task='localization')
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True,  num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size,
                          shuffle=False, num_workers=2, pin_memory=True)

    model = VGG11Localizer(dropout_p=args.dropout_p).to(DEVICE)

    if os.path.exists(CLS_PATH):
        model.load_pretrained_backbone(CLS_PATH,
                                        freeze=args.freeze_backbone)
    else:
        print("⚠️  No classifier checkpoint — training from scratch")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=10, gamma=0.5)
    criterion = IoULoss(reduction='mean')
    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch in train_dl:
            imgs   = batch['image'].to(DEVICE)
            bboxes = batch['bbox'].to(DEVICE)
            optimizer.zero_grad()
            preds  = model(imgs)
            loss   = criterion(preds, bboxes)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                imgs   = batch['image'].to(DEVICE)
                bboxes = batch['bbox'].to(DEVICE)
                preds  = model(imgs)
                val_loss += criterion(preds, bboxes).item()

        scheduler.step()
        tl = train_loss / len(train_dl)
        vl = val_loss   / len(val_dl)

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train IoU Loss: {tl:.4f} | "
              f"Val IoU Loss  : {vl:.4f}")

        wandb.log({
            'epoch'         : epoch + 1,
            'train/iou_loss': tl,
            'val/iou_loss'  : vl,
            'lr'            : scheduler.get_last_lr()[0],
        })

        if vl < best_loss:
            best_loss = vl
            torch.save(model.state_dict(), CKPT_PATH)
            print(f"  ✅ Saved best model (val IoU loss={vl:.4f})")

    wandb.finish()
    print(f"\nBest val IoU loss: {best_loss:.4f}")
    print(f"Checkpoint       : {CKPT_PATH}")
    return CKPT_PATH


# ══════════════════════════════════════════════════════════════
# TASK 3 — Segmentation
# ══════════════════════════════════════════════════════════════
def train_segmentation(args):
    print("\n" + "="*50)
    print("TASK 3 — U-Net Segmentation")
    print("="*50)

    DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
    CKPT_PATH = f'{CKPT_DIR}/unet.pth'
    CLS_PATH  = f'{CKPT_DIR}/classifier.pth'

    wandb.init(
        project='da6401-assignment2',
        name=f'task3-seg-freeze_{args.freeze_seg}',
        config={
            'task'       : 'segmentation',
            'epochs'     : args.epochs,
            'lr'         : args.lr,
            'batch_size' : args.batch_size,
            'freeze_seg' : args.freeze_seg,
        }
    )

    train_ds = PetsDataset(DATA_DIR, split='train',
                            task='segmentation')
    val_ds   = PetsDataset(DATA_DIR, split='val',
                            task='segmentation')
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True,  num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size,
                          shuffle=False, num_workers=2, pin_memory=True)

    model = VGG11UNet(num_classes=3,
                       dropout_p=args.dropout_p).to(DEVICE)

    if os.path.exists(CLS_PATH):
        model.load_pretrained_backbone(CLS_PATH, freeze=args.freeze_seg)
    else:
        print("⚠️  No classifier checkpoint — training from scratch")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    best_dice = 0.0

    for epoch in range(args.epochs):
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

        model.eval()
        val_loss, val_dice = 0.0, 0.0
        with torch.no_grad():
            for batch in val_dl:
                imgs   = batch['image'].to(DEVICE)
                masks  = batch['mask'].to(DEVICE)
                logits = model(imgs)
                val_loss += criterion(logits, masks).item()
                val_dice += dice_score(logits.cpu(), masks.cpu())

        scheduler.step()
        tl = train_loss / len(train_dl)
        vl = val_loss   / len(val_dl)
        vd = val_dice   / len(val_dl)

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {tl:.4f} | "
              f"Val Loss: {vl:.4f} | "
              f"Val Dice: {vd:.4f}")

        wandb.log({
            'epoch'     : epoch + 1,
            'train/loss': tl,
            'val/loss'  : vl,
            'val/dice'  : vd,
            'lr'        : scheduler.get_last_lr()[0],
        })

        if vd > best_dice:
            best_dice = vd
            torch.save(model.state_dict(), CKPT_PATH)
            print(f"  ✅ Saved best model (val Dice={vd:.4f})")

    wandb.finish()
    print(f"\nBest val Dice: {best_dice:.4f}")
    print(f"Checkpoint   : {CKPT_PATH}")
    return CKPT_PATH


# ══════════════════════════════════════════════════════════════
# TASK 4 — Multitask
# ══════════════════════════════════════════════════════════════
def train_multitask(args):
    print("\n" + "="*50)
    print("TASK 4 — Unified Multitask Training")
    print("="*50)

    DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
    CKPT_PATH = f'{CKPT_DIR}/multitask.pth'
    CLS_PATH  = f'{CKPT_DIR}/classifier.pth'
    LOC_PATH  = f'{CKPT_DIR}/localizer.pth'
    SEG_PATH  = f'{CKPT_DIR}/unet.pth'

    wandb.init(
        project='da6401-assignment2',
        name='task4-multitask',
        config={
            'task'      : 'multitask',
            'epochs'    : args.epochs,
            'lr'        : args.lr,
            'batch_size': args.batch_size,
        }
    )

    # ── Data ──
    train_ds = PetsDataset(DATA_DIR, split='train')
    val_ds   = PetsDataset(DATA_DIR, split='val')
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True,  num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size,
                          shuffle=False, num_workers=2, pin_memory=True)

    # ── Model ──
    # Load pretrained weights from Tasks 1, 2, 3
    model = MultiTaskPerceptionModel(
        num_breeds=37,
        seg_classes=3,
        classifier_path=CLS_PATH,
        localizer_path=LOC_PATH,
        unet_path=SEG_PATH,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=10, gamma=0.5)

    # ── Loss functions for each task ──
    cls_criterion = nn.CrossEntropyLoss()
    loc_criterion = IoULoss(reduction='mean')
    seg_criterion = nn.CrossEntropyLoss()

    # Loss weights — how much each task contributes to total loss
    # Classification and segmentation are harder so weighted higher
    W_CLS = 1.0
    W_LOC = 1.0
    W_SEG = 1.0

    best_val_loss = float('inf')

    for epoch in range(args.epochs):

        # ── Train ──
        model.train()
        train_cls_loss = 0.0
        train_loc_loss = 0.0
        train_seg_loss = 0.0
        all_preds, all_labels = [], []

        for batch in train_dl:
            imgs   = batch['image'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            bboxes = batch['bbox'].to(DEVICE)
            masks  = batch['mask'].to(DEVICE)

            optimizer.zero_grad()

            # Single forward pass — all 3 outputs
            outputs = model(imgs)

            # Individual losses
            loss_cls = cls_criterion(outputs['classification'], labels)
            loss_loc = loc_criterion(outputs['localization'], bboxes)
            loss_seg = seg_criterion(outputs['segmentation'], masks)

            # Combined weighted loss
            total_loss = (W_CLS * loss_cls +
                          W_LOC * loss_loc +
                          W_SEG * loss_seg)

            total_loss.backward()
            optimizer.step()

            train_cls_loss += loss_cls.item()
            train_loc_loss += loss_loc.item()
            train_seg_loss += loss_seg.item()

            all_preds.extend(
                outputs['classification'].argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_f1 = f1_score(all_labels, all_preds,
                             average='macro', zero_division=0)

        # ── Validate ──
        model.eval()
        val_cls_loss = 0.0
        val_loc_loss = 0.0
        val_seg_loss = 0.0
        val_dice     = 0.0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for batch in val_dl:
                imgs   = batch['image'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                bboxes = batch['bbox'].to(DEVICE)
                masks  = batch['mask'].to(DEVICE)

                outputs = model(imgs)

                val_cls_loss += cls_criterion(
                    outputs['classification'], labels).item()
                val_loc_loss += loc_criterion(
                    outputs['localization'], bboxes).item()
                val_seg_loss += seg_criterion(
                    outputs['segmentation'], masks).item()
                val_dice     += dice_score(
                    outputs['segmentation'].cpu(), masks.cpu())

                val_preds.extend(
                    outputs['classification'].argmax(1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_f1 = f1_score(val_labels, val_preds,
                           average='macro', zero_division=0)
        scheduler.step()

        # Averages
        n_train = len(train_dl)
        n_val   = len(val_dl)
        tcl = train_cls_loss / n_train
        tll = train_loc_loss / n_train
        tsl = train_seg_loss / n_train
        vcl = val_cls_loss   / n_val
        vll = val_loc_loss   / n_val
        vsl = val_seg_loss   / n_val
        vd  = val_dice       / n_val
        total_val = vcl + vll + vsl

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train — cls:{tcl:.3f} loc:{tll:.3f} seg:{tsl:.3f} "
              f"F1:{train_f1:.3f} | "
              f"Val — cls:{vcl:.3f} loc:{vll:.3f} seg:{vsl:.3f} "
              f"Dice:{vd:.3f} F1:{val_f1:.3f}")

        wandb.log({
            'epoch'           : epoch + 1,
            'train/cls_loss'  : tcl,
            'train/loc_loss'  : tll,
            'train/seg_loss'  : tsl,
            'train/f1'        : train_f1,
            'val/cls_loss'    : vcl,
            'val/loc_loss'    : vll,
            'val/seg_loss'    : vsl,
            'val/dice'        : vd,
            'val/f1'          : val_f1,
            'lr'              : scheduler.get_last_lr()[0],
        })

        if total_val < best_val_loss:
            best_val_loss = total_val
            torch.save(model.state_dict(), CKPT_PATH)
            print(f"  ✅ Saved best model (val total loss={total_val:.4f})")

    wandb.finish()
    print(f"\nBest val total loss: {best_val_loss:.4f}")
    print(f"Checkpoint         : {CKPT_PATH}")
    return CKPT_PATH


# ══════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ══════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(
        description='DA6401 Assignment 2 Training')

    parser.add_argument('--task', type=str,
                        choices=['classification', 'localization',
                                 'segmentation', 'multitask'],
                        required=True,
                        help='Which task to train')
    parser.add_argument('--epochs',     type=int,   default=30)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--dropout_p',  type=float, default=0.5)
    parser.add_argument('--freeze_backbone', type=bool, default=False)
    parser.add_argument('--freeze_seg', type=str,   default='none',
                        choices=['all', 'partial', 'none'])

    return parser.parse_args()


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    args = parse_args()

    if args.task == 'classification':
        train_classification(args)
    elif args.task == 'localization':
        train_localization(args)
    elif args.task == 'segmentation':
        train_segmentation(args)
    elif args.task == 'multitask':
        train_multitask(args)
