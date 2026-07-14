"""Intrinsic explanations: membership heatmap, prototype paths, view-ablation
importance, nearest-medoid prototypes, cluster similarity and class mix."""

import math
import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_membership_heatmap(membership, output_dir):
    membership_np = membership.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(max(6, membership_np.shape[1] * 0.4), max(4, membership_np.shape[0] * 0.25)))
    im = ax.imshow(membership_np, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Sample (test batch)')
    cbar = fig.colorbar(im, ax=ax, label='Membership')
    cbar.ax.yaxis.label.set_weight('bold')
    fig.tight_layout()
    fig.savefig(output_dir / 'membership_heatmap.png', dpi=150)
    plt.close(fig)


def plot_prototype_paths(prototype_paths, output_dir, num_samples=4):
    num_views = len(prototype_paths)
    B, T = prototype_paths[0].shape
    num_samples = min(num_samples, B)
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 2.2 * num_samples), sharex=True, squeeze=False)
    for s in range(num_samples):
        ax = axes[s][0]
        for v in range(num_views):
            path = prototype_paths[v][s].detach().cpu().numpy()
            ax.step(range(T), path, where='post', label=f'view {v}', alpha=0.8)
        ax.set_ylabel(f'sample {s}\nprototype')
        if s == 0:
            legend = ax.legend(loc='upper right', fontsize=8, ncol=num_views)
            for text in legend.get_texts():
                text.set_fontweight('bold')
    axes[-1][0].set_xlabel('timestep')
    fig.tight_layout()
    fig.savefig(output_dir / 'prototype_path.png', dpi=150)
    plt.close(fig)


def plot_view_importance(importance, output_dir):
    importance_np = importance.detach().cpu().numpy()
    means = importance_np.mean(axis=1)
    stds = importance_np.std(axis=1)
    num_views = importance_np.shape[0]
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    ax.bar(range(num_views), means, yerr=stds, capsize=4, color='steelblue')
    ax.set_xticks(range(num_views))
    ax.set_xticklabels([f'view {v}' for v in range(num_views)], fontsize=10)
    ax.set_ylabel('importance', fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / 'view_ablation_importance.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_nearest_medoid_prototypes(prototypes, medoid_indices, output_dir, channel_names=None):
    prototypes_np = prototypes.detach().cpu().numpy()
    num_cluster, T, D = prototypes_np.shape
    channel_names = channel_names or [f'ch{d}' for d in range(D)]
    n_cols = math.ceil(math.sqrt(num_cluster))
    n_rows = math.ceil(num_cluster / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2 * n_rows), squeeze=False)
    for c in range(num_cluster):
        ax = axes[c // n_cols][c % n_cols]
        for d in range(D):
            ax.plot(prototypes_np[c, :, d], linewidth=0.8, label=channel_names[d])
        ax.set_title(f'cluster {c} (idx {int(medoid_indices[c])})', fontsize=8, fontweight='bold')
        ax.tick_params(labelsize=6)
        if c == 0:
            legend = ax.legend(fontsize=5, ncol=min(D, 3))
            for text in legend.get_texts():
                text.set_fontweight('bold')
    for c in range(num_cluster, n_rows * n_cols):
        axes[c // n_cols][c % n_cols].axis('off')
    fig.tight_layout()
    fig.savefig(output_dir / 'nearest_medoid_prototypes.png', dpi=150)
    plt.close(fig)


def plot_cluster_similarity(similarity, output_dir):
    sim_np = similarity.detach().cpu().numpy()
    num_cluster = sim_np.shape[0]
    fig, ax = plt.subplots(figsize=(max(8, num_cluster * 0.45), max(5, num_cluster * 0.35)))
    im = ax.imshow(sim_np, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Cluster')
    cbar = fig.colorbar(im, ax=ax, label='Cosine similarity')
    cbar.ax.yaxis.label.set_weight('bold')
    fig.tight_layout()
    fig.savefig(output_dir / 'cluster_center_similarity.png', dpi=150)
    plt.close(fig)


def find_similar_cluster_groups(similarity_np, threshold):
    n = similarity_np.shape[0]
    visited = [False] * n
    groups = []
    for pivot in range(n):
        if visited[pivot]:
            continue
        visited[pivot] = True
        neighbors = [
            j for j in range(n)
            if j != pivot and not visited[j] and similarity_np[pivot, j] >= threshold
        ]
        for j in neighbors:
            visited[j] = True
        group = [pivot] + neighbors
        if len(group) > 1:
            groups.append(group)
    return groups


def plot_nearest_medoid_prototypes_comparison(prototypes, medoid_indices, similarity, threshold, output_dir,
                                                channel_names=None):
    similarity_np = similarity.detach().cpu().numpy()
    prototypes_np = prototypes.detach().cpu().numpy()
    groups = find_similar_cluster_groups(similarity_np, threshold)
    if not groups:
        print(f"\nnearest_medoid_prototypes_comparison: no cluster group with similarity >= {threshold}, skipping plot.")
        return

    D = prototypes_np.shape[2]
    channel_names = channel_names or [f'ch{d}' for d in range(D)]
    n_rows = len(groups)
    n_cols = max(len(g) for g in groups)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.4 * n_rows), squeeze=False)
    for row, group in enumerate(groups):
        for col in range(n_cols):
            ax = axes[row][col]
            if col < len(group):
                c = group[col]
                for d in range(D):
                    ax.plot(prototypes_np[c, :, d], linewidth=0.8, label=channel_names[d])
                ax.set_title(f'cluster {c} (idx {int(medoid_indices[c])})', fontsize=9, fontweight='bold')
                ax.tick_params(labelsize=6)
                if row == 0 and col == 0:
                    legend = ax.legend(fontsize=5, ncol=min(D, 3))
                    for text in legend.get_texts():
                        text.set_fontweight('bold')
            else:
                ax.axis('off')
    fig.tight_layout()
    fig.savefig(output_dir / 'nearest_medoid_prototypes_comparison.png', dpi=150)
    plt.close(fig)


def cluster_prototype_similarity(model, prototype_series, member_series_list):
    batch = np.stack([prototype_series] + list(member_series_list), axis=0)
    batch_t = torch.from_numpy(batch).float().to(model.device)
    with torch.no_grad():
        emb = model.encode_with_pooling(batch_t)  # already L2-normalized
    proto_emb = emb[0:1]
    member_emb = emb[1:]
    sims = torch.matmul(member_emb, proto_emb.T).squeeze(1)
    return sims.mean().item()


def compute_cluster_purity(pred_labels_np, true_labels_np, num_cluster):
    purity = {}
    for c in range(num_cluster):
        member_indices = np.nonzero(pred_labels_np == c)[0]
        true_classes = sorted(set(true_labels_np[member_indices].tolist())) if member_indices.size > 0 else []
        purity[c] = (member_indices, true_classes)
    return purity


def plot_cluster_class_mix(model, prototypes, medoid_indices, raw_series_np, pred_labels_np, true_labels_np,
                            output_dir, max_examples=20, channel_names=None):
    prototypes_np = prototypes.detach().cpu().numpy()
    medoid_indices_np = medoid_indices.detach().cpu().numpy()
    num_cluster = prototypes_np.shape[0]
    D = prototypes_np.shape[2]
    channel_names = channel_names or [f'ch{d}' for d in range(D)]

    purity = compute_cluster_purity(pred_labels_np, true_labels_np, num_cluster)
    mixed_clusters = [(c, mi, tc) for c, (mi, tc) in purity.items() if len(tc) >= 2]

    if not mixed_clusters:
        print("\ncluster_class_mix: no cluster with mixed true classes, skipping plot.")
        return

    for c, member_indices, true_classes in mixed_clusters:
        proto_series = prototypes_np[c]

        # example + score for each true class, computed only once
        # (not per channel)
        columns = []
        for true_class in true_classes:
            class_member_indices = member_indices[true_labels_np[member_indices] == true_class]
            example_idx = int(class_member_indices[0])
            example_series = raw_series_np[example_idx]

            sample_indices = class_member_indices[:max_examples]
            member_series_list = [raw_series_np[i] for i in sample_indices]
            score = cluster_prototype_similarity(model, proto_series, member_series_list)

            title = (
                f'true class {true_class}, score={score:.3f}\n'
                f'(n={sample_indices.size}, example idx {example_idx})'
            )
            columns.append((example_series, title))

        n_cols = 1 + len(columns)
        fig, axes = plt.subplots(
            D, n_cols, figsize=(3.2 * n_cols, 1.6 * D), sharex=True, sharey='row', squeeze=False,
        )

        for d in range(D):
            axes[d][0].plot(proto_series[:, d], linewidth=0.9, color='black')
            axes[d][0].set_ylabel(channel_names[d], fontsize=9, fontweight='bold')
            if d == 0:
                axes[d][0].set_title(f'cluster {c}\nprototype (idx {int(medoid_indices_np[c])})', fontsize=10)

            for col, (example_series, title) in enumerate(columns, start=1):
                axes[d][col].plot(example_series[:, d], linewidth=0.9, color='tab:red')
                if d == 0:
                    axes[d][col].set_title(title, fontsize=9)

        for col in range(n_cols):
            axes[-1][col].set_xlabel('timestep')

        fig.tight_layout()
        fig.savefig(output_dir / f'idx{int(medoid_indices_np[c])}_cluster{c}.png', dpi=150)
        plt.close(fig)


