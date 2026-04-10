
import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import numpy as np

# Fix import path when running as !python train.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.pets_dataset import PetsDataset
from models.classification import ClassificationModel
from models.localization import LocalizationModel
from models.segmentation import SegmentationModel
from losses.iou_loss import IoULoss


# ──────────────────────────────────────────
# Helper: Dice score for segmentation
# ──────────────────────────────────────────
def dice_score(pred_logits, target, num_classes=3, eps=1e-6):
    pred = pred_logits.argmax(dim=1)
    dice = 0.0
    for c in range(num_classes):
        p     = (pred == c).float()
        t     = (target == c).float()
        dice += (2 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps)
    return dice / num_classes


# ──────────────────────────────────────────
# Task 1 — Classification
# ──────────────────────────────────────────
def train_classifier(args):
    import wandb
    wandb.init(
        project  = 'da6401-assignment2',
        name     = 'task1-classification',
        config   = vars(args),
        settings = wandb.Settings(start_method='thread')
    )

    device    = args.device
    print(f"[CLS] Device: {device}", flush=True)
    print(f"[CLS] Loading dataset...", flush=True)

    train_ds  = PetsDataset(args.data_dir, split='train')
    val_ds    = PetsDataset(args.data_dir, split='val')

    # num_workers=0 prevents silent DataLoader hang on Colab + Drive
    train_dl  = DataLoader(train_ds, batch_size=args.batch_size,
                           shuffle=True,  num_workers=0, pin_memory=False)
    val_dl    = DataLoader(val_ds,   batch_size=args.batch_size,
                           shuffle=False, num_workers=0, pin_memory=False)

    print(f"[CLS] Train: {len(train_ds)} samples ({len(train_dl)} batches)", flush=True)
    print(f"[CLS] Val  : {len(val_ds)} samples ({len(val_dl)} batches)", flush=True)

    model     = ClassificationModel(num_classes=37, dropout_p=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    ckpt_path = os.path.join(args.ckpt_dir, 'classifier.pth')
    best_f1   = 0.0

    # Resume if checkpoint exists
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"[CLS] Resumed from {ckpt_path}", flush=True)

    print("[CLS] Starting training...", flush=True)
    print("-" * 70, flush=True)

    for epoch in range(args.epochs):

        # ── Train ──
        model.train()
        tr_loss, tr_preds, tr_labels = 0.0, [], []

        for batch_idx, batch in enumerate(train_dl):
            imgs   = batch['image'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            tr_preds.extend(logits.argmax(1).cpu().numpy())
            tr_labels.extend(labels.cpu().numpy())

            # Print batch progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"  [CLS] Epoch {epoch+1}/{args.epochs} "
                      f"Batch {batch_idx+1}/{len(train_dl)} "
                      f"Loss {loss.item():.4f}", flush=True)

        tr_f1 = f1_score(tr_labels, tr_preds, average='macro', zero_division=0)

        # ── Validate ──
        model.eval()
        val_loss, val_preds, val_labels = 0.0, [], []
        with torch.no_grad():
            for batch in val_dl:
                imgs   = batch['image'].to(device)
                labels = batch['label'].to(device)
                logits = model(imgs)
                val_loss += criterion(logits, labels).item()
                val_preds.extend(logits.argmax(1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        scheduler.step()

        print(f"[CLS] Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss {tr_loss/len(train_dl):.4f} F1 {tr_f1:.4f} | "
              f"Val Loss {val_loss/len(val_dl):.4f} F1 {val_f1:.4f}",
              flush=True)

        wandb.log({'epoch'         : epoch + 1,
                   'cls/train_loss': tr_loss  / len(train_dl),
                   'cls/train_f1'  : tr_f1,
                   'cls/val_loss'  : val_loss / len(val_dl),
                   'cls/val_f1'    : val_f1})

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ Saved best classifier (val F1={val_f1:.4f})", flush=True)

    wandb.finish()
    print(f"[CLS] Done. Best val F1: {best_f1:.4f}", flush=True)
    return ckpt_path


# ──────────────────────────────────────────
# Task 2 — Localization
# ──────────────────────────────────────────
def train_localizer(args, classifier_ckpt):
    import wandb
    wandb.init(
        project  = 'da6401-assignment2',
        name     = 'task2-localization',
        config   = vars(args),
        settings = wandb.Settings(start_method='thread')
    )

    device   = args.device
    print(f"[LOC] Device: {device}", flush=True)
    print(f"[LOC] Loading backbone from: {classifier_ckpt}", flush=True)

    train_ds = PetsDataset(args.data_dir, split='train')
    val_ds   = PetsDataset(args.data_dir, split='val')
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True,  num_workers=0, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size,
                          shuffle=False, num_workers=0, pin_memory=False)

    print(f"[LOC] Train: {len(train_ds)} samples | Val: {len(val_ds)} samples", flush=True)

    model     = LocalizationModel(pretrained_path=classifier_ckpt,
                                   freeze_backbone=False).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    mse_loss  = nn.MSELoss()
    iou_loss  = IoULoss(reduction='mean')
    ckpt_path = os.path.join(args.ckpt_dir, 'localizer.pth')
    best_iou  = 0.0

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"[LOC] Resumed from {ckpt_path}", flush=True)

    print("[LOC] Starting training...", flush=True)
    print("-" * 70, flush=True)

    for epoch in range(args.epochs):

        # ── Train ──
        model.train()
        tr_loss = 0.0
        for batch_idx, batch in enumerate(train_dl):
            imgs   = batch['image'].to(device)
            bboxes = batch['bbox'].to(device)
            optimizer.zero_grad()
            pred   = model(imgs)
            loss   = mse_loss(pred, bboxes) + iou_loss(pred, bboxes)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"  [LOC] Epoch {epoch+1}/{args.epochs} "
                      f"Batch {batch_idx+1}/{len(train_dl)} "
                      f"Loss {loss.item():.4f}", flush=True)

        # ── Validate ──
        model.eval()
        val_loss, val_iou = 0.0, 0.0
        with torch.no_grad():
            for batch in val_dl:
                imgs   = batch['image'].to(device)
                bboxes = batch['bbox'].to(device)
                pred   = model(imgs)
                val_loss += (mse_loss(pred, bboxes) + iou_loss(pred, bboxes)).item()
                val_iou  += (1 - iou_loss(pred, bboxes)).item()

        scheduler.step()
        avg_iou = val_iou / len(val_dl)

        print(f"[LOC] Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss {tr_loss/len(train_dl):.4f} | "
              f"Val Loss {val_loss/len(val_dl):.4f} | "
              f"Val IoU {avg_iou:.4f}",
              flush=True)

        wandb.log({'epoch'         : epoch + 1,
                   'loc/train_loss': tr_loss  / len(train_dl),
                   'loc/val_loss'  : val_loss / len(val_dl),
                   'loc/val_iou'   : avg_iou})

        if avg_iou > best_iou:
            best_iou = avg_iou
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ Saved best localizer (val IoU={avg_iou:.4f})", flush=True)

    wandb.finish()
    print(f"[LOC] Done. Best val IoU: {best_iou:.4f}", flush=True)
    return ckpt_path


# ──────────────────────────────────────────
# Task 3 — Segmentation
# ──────────────────────────────────────────
def train_segmentation(args, classifier_ckpt):
    import wandb
    wandb.init(
        project  = 'da6401-assignment2',
        name     = 'task3-segmentation',
        config   = vars(args),
        settings = wandb.Settings(start_method='thread')
    )

    device   = args.device
    print(f"[SEG] Device: {device}", flush=True)
    print(f"[SEG] Loading backbone from: {classifier_ckpt}", flush=True)

    train_ds = PetsDataset(args.data_dir, split='train')
    val_ds   = PetsDataset(args.data_dir, split='val')
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True,  num_workers=0, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size,
                          shuffle=False, num_workers=0, pin_memory=False)

    print(f"[SEG] Train: {len(train_ds)} samples | Val: {len(val_ds)} samples", flush=True)

    model     = SegmentationModel(num_classes=3,
                                   pretrained_path=classifier_ckpt,
                                   freeze_mode='partial').to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    ckpt_path = os.path.join(args.ckpt_dir, 'unet.pth')
    best_dice = 0.0

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"[SEG] Resumed from {ckpt_path}", flush=True)

    print("[SEG] Starting training...", flush=True)
    print("-" * 70, flush=True)

    for epoch in range(args.epochs):

        # ── Train ──
        model.train()
        tr_loss = 0.0
        for batch_idx, batch in enumerate(train_dl):
            imgs  = batch['image'].to(device)
            masks = batch['mask'].to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"  [SEG] Epoch {epoch+1}/{args.epochs} "
                      f"Batch {batch_idx+1}/{len(train_dl)} "
                      f"Loss {loss.item():.4f}", flush=True)

        # ── Validate ──
        model.eval()
        val_loss, val_dice, val_px = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in val_dl:
                imgs   = batch['image'].to(device)
                masks  = batch['mask'].to(device)
                logits = model(imgs)
                val_loss += criterion(logits, masks).item()
                val_dice += dice_score(logits, masks).item()
                preds    =  logits.argmax(dim=1)
                val_px  += (preds == masks).float().mean().item()

        scheduler.step()
        avg_dice = val_dice / len(val_dl)
        avg_px   = val_px   / len(val_dl)

        print(f"[SEG] Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss {tr_loss/len(train_dl):.4f} | "
              f"Val Loss {val_loss/len(val_dl):.4f} | "
              f"Dice {avg_dice:.4f} | PixAcc {avg_px:.4f}",
              flush=True)

        wandb.log({'epoch'           : epoch + 1,
                   'seg/train_loss'  : tr_loss  / len(train_dl),
                   'seg/val_loss'    : val_loss / len(val_dl),
                   'seg/val_dice'    : avg_dice,
                   'seg/val_pix_acc' : avg_px})

        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ Saved best segmentation model (Dice={avg_dice:.4f})", flush=True)

    wandb.finish()
    print(f"[SEG] Done. Best Dice: {best_dice:.4f}", flush=True)
    return ckpt_path


# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',       type=str,   default='all',
                        choices=['cls', 'loc', 'seg', 'all'])
    parser.add_argument('--data_dir',   type=str,   default='data')
    parser.add_argument('--ckpt_dir',   type=str,   default='checkpoints')
    parser.add_argument('--epochs',     type=int,   default=30)
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--device',     type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)

    print(f"Task     : {args.task}",       flush=True)
    print(f"Device   : {args.device}",     flush=True)
    print(f"Epochs   : {args.epochs}",     flush=True)
    print(f"Batch    : {args.batch_size}", flush=True)
    print(f"LR       : {args.lr}",         flush=True)
    print(f"Data dir : {args.data_dir}",   flush=True)
    print(f"Ckpt dir : {args.ckpt_dir}",   flush=True)
    print("=" * 70, flush=True)

    cls_ckpt = None

    if args.task in ('cls', 'all'):
        cls_ckpt = train_classifier(args)

    if args.task in ('loc', 'all'):
        if cls_ckpt is None:
            cls_ckpt = os.path.join(args.ckpt_dir, 'classifier.pth')
        train_localizer(args, cls_ckpt)

    if args.task in ('seg', 'all'):
        if cls_ckpt is None:
            cls_ckpt = os.path.join(args.ckpt_dir, 'classifier.pth')
        train_segmentation(args, cls_ckpt)

    print("=" * 70, flush=True)
    print("All tasks complete.", flush=True)
