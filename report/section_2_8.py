import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

import wandb
import matplotlib.pyplot as plt

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ══════════════════════════════════════════════════════════════
# Fetch all runs from WandB
# ══════════════════════════════════════════════════════════════
api     = wandb.Api()

#  Change this to your actual project path
PROJECT = "ma24m018-iit-ma/da6401-assignment2"
runs    = api.runs(PROJECT)

# Collect run data
run_data = {}
for run in runs:
    name = run.name
    print(f"Found run: {name}")
    history      = run.history(samples=200)
    run_data[name] = history

print(f"\nTotal runs found: {len(run_data)}")

# ══════════════════════════════════════════════════════════════
# Start WandB run for plots
# ══════════════════════════════════════════════════════════════
wandb.init(
    project='da6401-assignment2',
    name='2.8-meta-analysis-plots',
    config={'section': '2.8'}
)

colors = ['royalblue', 'darkgreen',
          'darkred', 'purple', 'orange']

# ══════════════════════════════════════════════════════════════
# Plot 1: Task 1 Classification
# ══════════════════════════════════════════════════════════════
cls_runs = {k: v for k, v in run_data.items()
            if 'task1' in k.lower()
            or 'classification' in k.lower()
            or 'dropout' in k.lower()}

if cls_runs:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        'Task 1 — Classification Training History',
        fontsize=13, fontweight='bold')

    for i, (name, hist) in enumerate(cls_runs.items()):
        c = colors[i % len(colors)]
        if 'train/loss' in hist.columns:
            axes[0].plot(
                hist['train/loss'].dropna(),
                color=c, linewidth=2,
                label=f'{name} Train')
            axes[0].plot(
                hist['val/loss'].dropna(),
                color=c, linewidth=2,
                linestyle='--',
                label=f'{name} Val')
        if 'train/f1' in hist.columns:
            axes[1].plot(
                hist['train/f1'].dropna(),
                color=c, linewidth=2,
                label=f'{name} Train')
            axes[1].plot(
                hist['val/f1'].dropna(),
                color=c, linewidth=2,
                linestyle='--',
                label=f'{name} Val')

    axes[0].set_title('Loss Curves')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title('F1 Score Curves')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Macro F1')
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    wandb.log({
        'section_2_8/task1_classification':
            wandb.Image(fig,
                caption='Task 1 Classification')
    })
    plt.close()
    print(" Task 1 plot done")
else:
    print("  No Task 1 runs found")

# ══════════════════════════════════════════════════════════════
# Plot 2: Task 2 Localization
# ══════════════════════════════════════════════════════════════
loc_runs = {k: v for k, v in run_data.items()
            if 'task2' in k.lower()
            or 'local' in k.lower()}

if loc_runs:
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    fig2.suptitle(
        'Task 2 — Localization Training History',
        fontsize=13, fontweight='bold')

    for i, (name, hist) in enumerate(loc_runs.items()):
        c = colors[i % len(colors)]
        if 'train/iou_loss' in hist.columns:
            ax2.plot(
                hist['train/iou_loss'].dropna(),
                color=c, linewidth=2,
                label=f'{name} Train')
            ax2.plot(
                hist['val/iou_loss'].dropna(),
                color=c, linewidth=2,
                linestyle='--',
                label=f'{name} Val')

    ax2.set_title('IoU Loss Curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('IoU Loss')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    wandb.log({
        'section_2_8/task2_localization':
            wandb.Image(fig2,
                caption='Task 2 Localization')
    })
    plt.close()
    print(" Task 2 plot done")
else:
    print("  No Task 2 runs found")

# ══════════════════════════════════════════════════════════════
# Plot 3: Task 3 Segmentation
# ══════════════════════════════════════════════════════════════
seg_runs = {k: v for k, v in run_data.items()
            if 'task3' in k.lower()
            or 'seg' in k.lower()
            or 'unet' in k.lower()}

if seg_runs:
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
    fig3.suptitle(
        'Task 3 — Segmentation Training History',
        fontsize=13, fontweight='bold')

    for i, (name, hist) in enumerate(seg_runs.items()):
        c = colors[i % len(colors)]
        if 'train/loss' in hist.columns:
            axes3[0].plot(
                hist['train/loss'].dropna(),
                color=c, linewidth=2,
                label=f'{name} Train')
            axes3[0].plot(
                hist['val/loss'].dropna(),
                color=c, linewidth=2,
                linestyle='--',
                label=f'{name} Val')
        if 'val/dice' in hist.columns:
            axes3[1].plot(
                hist['val/dice'].dropna(),
                color=c, linewidth=2,
                label=f'{name}')

    axes3[0].set_title('Loss Curves')
    axes3[0].set_xlabel('Epoch')
    axes3[0].set_ylabel('Loss')
    axes3[0].legend(fontsize=7)
    axes3[0].grid(True, alpha=0.3)

    axes3[1].set_title('Validation Dice Score')
    axes3[1].set_xlabel('Epoch')
    axes3[1].set_ylabel('Dice Score')
    axes3[1].legend(fontsize=7)
    axes3[1].grid(True, alpha=0.3)

    plt.tight_layout()
    wandb.log({
        'section_2_8/task3_segmentation':
            wandb.Image(fig3,
                caption='Task 3 Segmentation')
    })
    plt.close()
    print(" Task 3 plot done")
else:
    print("  No Task 3 runs found")

# ══════════════════════════════════════════════════════════════
# Plot 4: Task 4 Multitask
# ══════════════════════════════════════════════════════════════
multi_runs = {k: v for k, v in run_data.items()
              if 'task4' in k.lower()
              or 'multi' in k.lower()}

if multi_runs:
    fig4, axes4 = plt.subplots(1, 3, figsize=(18, 5))
    fig4.suptitle(
        'Task 4 — Multitask Training History',
        fontsize=13, fontweight='bold')

    for name, hist in multi_runs.items():
        if 'train/cls_loss' in hist.columns:
            axes4[0].plot(
                hist['train/cls_loss'].dropna(),
                color='royalblue', linewidth=2,
                label='Train')
            axes4[0].plot(
                hist['val/cls_loss'].dropna(),
                color='skyblue', linewidth=2,
                linestyle='--', label='Val')
            axes4[0].set_title('Classification Loss')
            axes4[0].set_xlabel('Epoch')
            axes4[0].legend(fontsize=8)
            axes4[0].grid(True, alpha=0.3)

        if 'train/loc_loss' in hist.columns:
            axes4[1].plot(
                hist['train/loc_loss'].dropna(),
                color='darkgreen', linewidth=2,
                label='Train')
            axes4[1].plot(
                hist['val/loc_loss'].dropna(),
                color='lightgreen', linewidth=2,
                linestyle='--', label='Val')
            axes4[1].set_title('Localization Loss')
            axes4[1].set_xlabel('Epoch')
            axes4[1].legend(fontsize=8)
            axes4[1].grid(True, alpha=0.3)

        if 'train/seg_loss' in hist.columns:
            axes4[2].plot(
                hist['train/seg_loss'].dropna(),
                color='darkred', linewidth=2,
                label='Train')
            axes4[2].plot(
                hist['val/seg_loss'].dropna(),
                color='salmon', linewidth=2,
                linestyle='--', label='Val')
            axes4[2].set_title('Segmentation Loss')
            axes4[2].set_xlabel('Epoch')
            axes4[2].legend(fontsize=8)
            axes4[2].grid(True, alpha=0.3)

    plt.tight_layout()
    wandb.log({
        'section_2_8/task4_multitask':
            wandb.Image(fig4,
                caption='Task 4 Multitask')
    })
    plt.close()
    print(" Task 4 plot done")
else:
    print("  No Task 4 runs found")

# ══════════════════════════════════════════════════════════════
# Plot 5: All tasks overview
# ══════════════════════════════════════════════════════════════
fig5, ax5 = plt.subplots(figsize=(12, 6))
fig5.suptitle(
    'Section 2.8 — All Tasks Validation Loss Overview',
    fontsize=13, fontweight='bold')

task_colors = {
    'task1': 'royalblue',
    'task2': 'darkgreen',
    'task3': 'darkred',
    'task4': 'purple',
}

plotted = False
for run_name, hist in run_data.items():
    for task, color in task_colors.items():
        if task in run_name.lower():
            if 'val/loss' in hist.columns:
                ax5.plot(
                    hist['val/loss'].dropna(),
                    color=color, linewidth=2,
                    label=f'{run_name}')
                plotted = True
            elif 'val/iou_loss' in hist.columns:
                ax5.plot(
                    hist['val/iou_loss'].dropna(),
                    color=color, linewidth=2,
                    label=f'{run_name}')
                plotted = True

if plotted:
    ax5.set_title('All Tasks — Validation Loss')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Loss')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    plt.tight_layout()
    wandb.log({
        'section_2_8/all_tasks_overview':
            wandb.Image(fig5,
                caption='All Tasks Overview')
    })
    print(" Overview plot done")
else:
    print("  No matching runs for overview")
plt.close()

wandb.finish()
print("\n Section 2.8 fully logged to WandB!")