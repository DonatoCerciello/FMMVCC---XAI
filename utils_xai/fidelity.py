"""Fidelity (deletion/insertion AUC) and stability (cross-seed correlation)
evaluation of the explanation methods, and the summary table."""
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

from utils_xai.gradients import compute_attributions
from utils_xai.timeshap import (
    get_intrinsic_and_captum_cached,
    get_timeshap_event_profile_cached,
)

WITH_TIMESHAP = '--with-timeshap' in sys.argv


def compute_method_profiles_for_example(model, x_single, target_cluster, args, cache, idx):
    intrinsic_captum = get_intrinsic_and_captum_cached(cache['intrinsic'], model, x_single, target_cluster, idx, args.ig_steps)
    profiles = {
        'intrinsic w_{b,t,c}': intrinsic_captum['w_btc'],
        'Captum IntegratedGradients': intrinsic_captum['ig_time_profile'],
    }
    if WITH_TIMESHAP:
        profiles['TimeSHAP event-level'] = get_timeshap_event_profile_cached(
            cache['event'], model, x_single, target_cluster, idx, args.seed, args,
        )
    return profiles


def _batched_predict_membership(model, x_batch_np, target_cluster, chunk_size):
    outs = []
    with torch.no_grad():
        for start in range(0, x_batch_np.shape[0], chunk_size):
            chunk = torch.from_numpy(x_batch_np[start:start + chunk_size]).float().to(model.device)
            outs.append(model.predict_membership(chunk)[:, target_cluster].detach().cpu().numpy())
    return np.concatenate(outs)


def compute_deletion_insertion_curves(model, x_single, target_cluster, profile, chunk_size):
    T, D = x_single.shape[1], x_single.shape[2]
    order = np.argsort(-np.abs(profile))
    x_np = x_single.detach().cpu().numpy()

    deletion_batch = np.repeat(x_np, T + 1, axis=0)
    insertion_batch = np.zeros((T + 1, T, D), dtype=x_np.dtype)
    for k in range(1, T + 1):
        idx = order[:k]
        deletion_batch[k, idx, :] = 0.0
        insertion_batch[k, idx, :] = x_np[0, idx, :]

    del_scores = _batched_predict_membership(model, deletion_batch, target_cluster, chunk_size)
    ins_scores = _batched_predict_membership(model, insertion_batch, target_cluster, chunk_size)
    return del_scores, ins_scores


def _curve_auc(scores):
    T = len(scores) - 1
    trapezoid = getattr(np, 'trapezoid', None) or np.trapz
    return float(trapezoid(scores, dx=1.0 / T))


def compute_stability(model, x_single, target_cluster, args, seeds, cache, idx):
    if not WITH_TIMESHAP:
        return {}, []
    profiles = [
        np.nan_to_num(get_timeshap_event_profile_cached(cache, model, x_single, target_cluster, idx, seed, args), nan=0.0)
        for seed in seeds
    ]
    corrs = [
        np.corrcoef(profiles[i], profiles[j])[0, 1]
        for i in range(len(profiles)) for j in range(i + 1, len(profiles))
    ]
    stability = {'TimeSHAP event-level': float(np.mean(corrs))} if corrs else {}
    return stability, profiles


def plot_stability_profiles(profiles, seeds, save_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    for seed, profile in zip(seeds, profiles):
        ax.plot(profile, label=f'seed={seed}', linewidth=1.3, alpha=0.85)
    ax.set_xlabel('timestep')
    ax.set_ylabel('Shapley value')
    legend = ax.legend(fontsize=8)
    for text in legend.get_texts():
        text.set_fontweight('bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_fidelity_curves(method_curves, curve_type, save_path):
    pretty = {'intrinsic w_{b,t,c}': 'Intrinsic', 'Captum IntegratedGradients': 'Integrated Gradients',
              'TimeSHAP event-level': 'TimeSHAP'}
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    for method, curves in method_curves.items():
        arr = np.stack(curves, axis=0)  # [n_examples, T+1]
        T = arr.shape[1] - 1
        x = np.linspace(0, 1, T + 1)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        ax.plot(x, mean, label=pretty.get(method, method), linewidth=1.6)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15)
    verb = 'masked' if curve_type == 'deletion' else 'revealed'
    ax.set_xlabel(f'fraction {verb}', fontsize=11)
    ax.set_ylabel('cluster membership', fontsize=11)
    legend = ax.legend(fontsize=9)
    for text in legend.get_texts():
        text.set_fontweight('bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_fidelity_auc_distribution(method_del_aucs, method_ins_aucs, save_path):
    methods = list(method_del_aucs.keys())
    fig, axes = plt.subplots(1, 2, figsize=(max(6, 2.2 * len(methods)) * 2, 4.5))
    panels = [(method_del_aucs, 'deletion AUC (lower = better)'), (method_ins_aucs, 'insertion AUC (higher = better)')]
    for ax, (data, title) in zip(axes, panels):
        ax.boxplot([data[m] for m in methods], showmeans=True)
        ax.set_xticks(range(1, len(methods) + 1))
        ax.set_xticklabels(methods, rotation=20, ha='right', fontsize=8)
        ax.set_ylabel(f'AUC ({title})')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_fidelity_per_cluster(cluster_method_del_aucs, cluster_method_ins_aucs, save_path):
    clusters = sorted(cluster_method_del_aucs.keys())
    methods = sorted({m for c in clusters for m in cluster_method_del_aucs[c]})
    width = 0.8 / max(1, len(methods))
    x_pos = np.arange(len(clusters))
    fig, axes = plt.subplots(1, 2, figsize=(max(6, 2.4 * len(clusters)) * 2, 4.5))
    panels = [(cluster_method_del_aucs, 'deletion AUC (lower = better)'),
              (cluster_method_ins_aucs, 'insertion AUC (higher = better)')]
    for ax, (data, title) in zip(axes, panels):
        for i, method in enumerate(methods):
            means = [float(np.mean(data[c][method])) for c in clusters]
            ax.bar(x_pos + i * width, means, width=width, label=method)
        ax.set_xticks(x_pos + width * (len(methods) - 1) / 2)
        ax.set_xticklabels([f'cluster {c}' for c in clusters])
        ax.set_ylabel(f'AUC ({title})')
    legend = axes[0].legend(fontsize=7)
    for text in legend.get_texts():
        text.set_fontweight('bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_fidelity_stability_table(rows, save_path):
    col_labels = ['method', 'deletion AUC (lower = better)', 'insertion AUC (higher = better)',
                  'stability (Pearson r across seeds)']
    cell_text = [
        [method, f'{del_auc:.3f}', f'{ins_auc:.3f}', 'n/a (no --with-timeshap)' if stab is None else f'{stab:.3f}']
        for method, del_auc, ins_auc, stab in rows
    ]
    fig, ax = plt.subplots(figsize=(11, 0.9 + 0.6 * len(rows)))
    ax.axis('off')
    table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.0)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor('#dddddd')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


