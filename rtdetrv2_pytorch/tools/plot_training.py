"""Plot training/validation losses and learning rate of a finished (or running) training.

Reads `<output_dir>/log.txt` (one JSON line per epoch) and, if present, the TensorBoard
files in `<output_dir>/summary` for the per-iteration learning rate of every param group.

Usage:
    python tools/plot_training.py output/rtdetrv2_r18vd_lumen

Writes to `<output_dir>/plots/`:
    overview.png         train loss + val loss (top), learning rate (bottom), shared epoch axis
    train_loss.png       total train loss and its final-layer part
    val_loss.png         validation loss
    lr.png               learning rate of every param group, per iteration
    loss_components.png  loss_vfl / loss_bbox / loss_giou, train vs val

Note on comparability: the total training loss also sums auxiliary decoder-layer, encoder
and denoising losses, while the validation loss (eval mode) only has the final decoder layer.
"Train loss (final layer)" = train_loss_vfl + train_loss_bbox + train_loss_giou is the
quantity directly comparable to the validation loss.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


COMPONENTS = ['loss_vfl', 'loss_bbox', 'loss_giou']

TRAIN_COLOR = '#2a78d6'
VAL_COLOR = '#eb6834'
LR_COLOR = '#52514e'
GROUP_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
TEXT_COLOR = '#0b0b0b'
MUTED_COLOR = '#52514e'
GRID_COLOR = '#e4e3df'


def setup_style():
    plt.rcParams.update({
        'figure.dpi': 120,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
        'font.size': 10,
        'axes.edgecolor': GRID_COLOR,
        'axes.labelcolor': MUTED_COLOR,
        'axes.titlecolor': TEXT_COLOR,
        'axes.titlesize': 11,
        'axes.titleweight': 'bold',
        'axes.titlelocation': 'left',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.color': GRID_COLOR,
        'grid.linewidth': 0.8,
        'xtick.color': MUTED_COLOR,
        'ytick.color': MUTED_COLOR,
        'legend.frameon': False,
        'lines.linewidth': 2,
    })


def load_log(path: Path):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    # a resumed run appends to the same log; keep the latest record per epoch
    by_epoch = {r['epoch']: r for r in records}
    return [by_epoch[e] for e in sorted(by_epoch)]


def column(records, key):
    return [r.get(key) for r in records]


def final_layer_sum(records, prefix):
    keys = [f'{prefix}_{c}' for c in COMPONENTS]
    if not all(k in records[0] for k in keys):
        return None
    return [sum(r[k] for k in keys) for r in records]


def load_lr_from_tensorboard(summary_dir: Path):
    """Returns {group_index: (steps, values)} or None."""
    if not summary_dir.is_dir() or not any(summary_dir.glob('events.out.tfevents.*')):
        return None
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    acc = EventAccumulator(str(summary_dir), size_guidance={'scalars': 0})
    acc.Reload()
    groups = {}
    for tag in acc.Tags().get('scalars', []):
        if tag.startswith('Lr/pg_'):
            events = sorted({e.step: e.value for e in acc.Scalars(tag)}.items())
            groups[int(tag.split('_')[-1])] = ([s for s, _ in events], [v for _, v in events])
    return groups or None


def merge_identical_groups(groups):
    """Groups with the same LR curve are drawn once: [(label, steps, values), ...]."""
    merged = []
    for g in sorted(groups):
        steps, values = groups[g]
        for item in merged:
            if item[2] == values:
                item[0].append(g)
                break
        else:
            merged.append(([g], steps, values))
    return [(f"param group{'s' if len(ids) > 1 else ''} {', '.join(map(str, ids))} "
             f"(peak {max(values):.0e})", steps, values) for ids, steps, values in merged]


def find_lr_drops(epochs, lr):
    """Epochs where the LR drops by more than 2x from the previous epoch."""
    return [epochs[i] for i in range(1, len(lr)) if lr[i - 1] and lr[i] < lr[i - 1] / 2]


def mark_lr_drops(ax, drops, label=False):
    for e in drops:
        ax.axvline(e, color=MUTED_COLOR, linewidth=1, linestyle='--', zorder=0)
        if label:
            ax.annotate('LR drop', (e, 1), xycoords=('data', 'axes fraction'),
                        xytext=(-4, -4), textcoords='offset points', va='top', ha='right',
                        fontsize=9, color=MUTED_COLOR)


def save(fig, path: Path):
    fig.savefig(path)
    plt.close(fig)
    print(f'saved {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('output_dir', type=Path, help='training output_dir (contains log.txt)')
    parser.add_argument('--out', type=Path, help='where to save plots (default: <output_dir>/plots)')
    parser.add_argument('--iters-per-epoch', type=int,
                        help='to place per-iteration LR on the epoch axis (default: estimated from the logs)')
    args = parser.parse_args()

    setup_style()
    records = load_log(args.output_dir / 'log.txt')
    out_dir = args.out or args.output_dir / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = column(records, 'epoch')
    train_total = column(records, 'train_loss')
    train_main = final_layer_sum(records, 'train')
    has_val = 'test_loss' in records[0]
    val_loss = column(records, 'test_loss') if has_val else None
    if not has_val:
        print('no validation loss in log.txt (run trained before val loss was added); skipping it')

    # learning rate: per-iteration from TensorBoard if available, else epoch average of group 0
    lr_groups = load_lr_from_tensorboard(args.output_dir / 'summary')
    if lr_groups:
        n_steps = max(max(steps) for steps, _ in lr_groups.values()) + 1
        iters_per_epoch = args.iters_per_epoch or n_steps / len(epochs)
        lr_curves = merge_identical_groups(lr_groups)
        # overview shows the main (highest) LR
        _, main_steps, main_values = max(lr_curves, key=lambda c: max(c[2]))
        lr_x = [s / iters_per_epoch for s in main_steps]
        lr_y = main_values
        lr_label = 'Learning rate (main param group), per iteration'
    else:
        lr_curves = None
        lr_x, lr_y = epochs, column(records, 'train_lr')
        lr_label = 'Learning rate (param group 0), epoch average'
    lr_drops = find_lr_drops(epochs, column(records, 'train_lr'))

    # 1. overview: losses on top, LR below, same epoch axis
    fig, (ax_loss, ax_lr) = plt.subplots(2, 1, sharex=True, figsize=(9, 6.5),
                                         gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.25})
    ax_loss.plot(epochs, train_main or train_total, color=TRAIN_COLOR,
                 label='Train (final layer)' if train_main else 'Train (total)')
    if has_val:
        ax_loss.plot(epochs, val_loss, color=VAL_COLOR, label='Validation')
    ax_loss.set_title('Loss')
    ax_loss.set_ylabel('loss')
    ax_loss.legend(loc='upper center')
    mark_lr_drops(ax_loss, lr_drops, label=True)
    ax_lr.plot(lr_x, lr_y, color=LR_COLOR)
    ax_lr.set_yscale('log')
    ax_lr.set_title(lr_label)
    ax_lr.set_ylabel('lr')
    ax_lr.set_xlabel('epoch')
    mark_lr_drops(ax_lr, lr_drops)
    save(fig, out_dir / 'overview.png')

    # 2. train loss
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(epochs, train_total, color=LR_COLOR, label='Total (incl. aux, encoder, denoising)')
    if train_main:
        ax.plot(epochs, train_main, color=TRAIN_COLOR, label='Final layer (comparable to validation)')
    ax.set_title('Training loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.legend(loc='upper center')
    mark_lr_drops(ax, lr_drops, label=True)
    save(fig, out_dir / 'train_loss.png')

    # 3. validation loss
    if has_val:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(epochs, val_loss, color=VAL_COLOR)
        ax.set_title('Validation loss (final layer)')
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        mark_lr_drops(ax, lr_drops, label=True)
        save(fig, out_dir / 'val_loss.png')

    # 4. learning rate
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if lr_curves:
        for i, (label, steps, values) in enumerate(lr_curves):
            ax.plot(steps, values, color=GROUP_COLORS[i % len(GROUP_COLORS)], label=label)
        ax.set_xlabel('iteration')
        ax.legend(loc='lower right')
        ax.set_title('Learning rate per param group')
    else:
        ax.plot(lr_x, lr_y, color=LR_COLOR)
        ax.set_xlabel('epoch')
        ax.set_title(lr_label)
    ax.set_yscale('log')
    ax.set_ylabel('lr')
    save(fig, out_dir / 'lr.png')

    # 5. loss components, train vs val
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 9), gridspec_kw={'hspace': 0.3})
    for ax, comp in zip(axes, COMPONENTS):
        if f'train_{comp}' in records[0]:
            ax.plot(epochs, column(records, f'train_{comp}'), color=TRAIN_COLOR, label='Train (final layer)')
        if f'test_{comp}' in records[0]:
            ax.plot(epochs, column(records, f'test_{comp}'), color=VAL_COLOR, label='Validation')
        ax.set_title(comp)
        ax.set_ylabel('loss')
        mark_lr_drops(ax, lr_drops)
    axes[0].legend(loc='upper center')
    axes[-1].set_xlabel('epoch')
    save(fig, out_dir / 'loss_components.png')


if __name__ == '__main__':
    main()
