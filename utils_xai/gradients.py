"""Captum gradient attributions (Integrated Gradients, GradientSHAP, Occlusion)
and channel/time importance plots, including the IG time-channel heatmap."""
import numpy as np
import torch
import matplotlib.pyplot as plt
try:
    from captum.attr import GradientShap, IntegratedGradients, Occlusion
except ImportError:
    raise SystemExit(
        "captum is not installed in this Python interpreter. It is, however, "
        "already a declared dependency of the project in dockerimg/Dockerfile "
        "(the `pip3 install ... captum` line), so it should be installed there "
        "if it's genuinely missing, not silently by this script.\n"
        "Note: this project has no dedicated requirements.txt; if you want "
        "one for XAI, add 'captum' there (recommended version: the one "
        "already pinned in the Docker image)."
    )

def make_forward_func(model, target_cluster):
    def forward_func(x):
        return model.predict_membership(x)[:, target_cluster]
    return forward_func


def _gradient_shap_chunked(gs, x, baseline, n_samples, stdevs, chunk_size):
    chunks = []
    for start in range(0, x.shape[0], chunk_size):
        end = start + chunk_size
        chunks.append(
            gs.attribute(x[start:end], baselines=baseline[start:end], n_samples=n_samples, stdevs=stdevs)
        )
    return torch.cat(chunks, dim=0)


def compute_attributions(model, x, target_cluster, methods, ig_steps, gradientshap_samples,
                          occlusion_max_cells, internal_batch_size, verbose=True):
    forward_func = make_forward_func(model, target_cluster)
    x = x.clone().detach().requires_grad_(True)
    baseline = torch.zeros_like(x)

    results = {}

    ig = IntegratedGradients(forward_func)
    attr, delta = ig.attribute(
        x, baselines=baseline, n_steps=ig_steps, internal_batch_size=internal_batch_size,
        return_convergence_delta=True,
    )
    results['integrated_gradients'] = attr.detach()
    if verbose:
        print(f"  IntegratedGradients: convergence delta per sample = "
              f"{delta.detach().cpu().numpy().round(6).tolist()}")

    if 'gradientshap' in methods:
        gs = GradientShap(forward_func)
        gs_chunk_size = max(1, internal_batch_size // gradientshap_samples)
        attr_gs = _gradient_shap_chunked(gs, x, baseline, gradientshap_samples, 0.01, gs_chunk_size)
        results['gradient_shap'] = attr_gs.detach()
        if verbose:
            print("  GradientShap: done.")

    if 'occlusion' in methods:
        B, T, D = x.shape
        if T * D > occlusion_max_cells:
            if verbose:
                print(f"  Occlusion: skipped (T*D={T * D} > --occlusion-max-cells={occlusion_max_cells}, "
                      "one forward pass per cell would be too slow).")
        else:
            occ = Occlusion(forward_func)
            attr_occ = occ.attribute(x.detach(), sliding_window_shapes=(1, 1), baselines=0.0)
            results['occlusion'] = attr_occ.detach()
            if verbose:
                print("  Occlusion: done.")

    return results


def plot_attribution_heatmap(x_sample, attributions_by_method, save_path, channel_names=None):
    T, D = x_sample.shape
    channel_names = channel_names or [f'ch{d}' for d in range(D)]
    methods = list(attributions_by_method.keys())
    n_rows = 1 + len(methods)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 2.8 * n_rows), sharex=True)

    ax0 = axes[0]
    for d in range(D):
        ax0.plot(x_sample[:, d], linewidth=0.9, label=channel_names[d])
    ax0.set_ylabel('original series\nvalue')
    if D <= 10:
        legend = ax0.legend(fontsize=6, ncol=min(D, 5))
        for text in legend.get_texts():
            text.set_fontweight('bold')

    for row, method in enumerate(methods, start=1):
        ax = axes[row]
        attr = attributions_by_method[method]
        vmax = np.abs(attr).max()
        vmax = vmax if vmax > 0 else 1.0
        im = ax.imshow(attr.T, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_ylabel(f'{method}\nchannel')
        cbar = fig.colorbar(im, ax=ax, label='attribution')
        cbar.ax.yaxis.label.set_weight('bold')

    axes[-1].set_xlabel('timestep')
    fig.tight_layout(h_pad=3.0)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_time_importance(x_sample, attributions_by_method, save_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    for method, attr in attributions_by_method.items():
        profile = np.abs(attr).sum(axis=1)  # [T]
        ax.plot(profile, label=method, linewidth=1.5)
    ax.set_xlabel('timestep')
    ax.set_ylabel('total |attribution|')
    legend = ax.legend(fontsize=8)
    for text in legend.get_texts():
        text.set_fontweight('bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_channel_importance(x_sample, attributions_by_method, save_path, channel_names=None):
    D = x_sample.shape[1]
    channel_names = channel_names or [f'ch{d}' for d in range(D)]
    methods = list(attributions_by_method.keys())
    width = 0.8 / len(methods)
    x_pos = np.arange(D)
    fig, ax = plt.subplots(figsize=(max(6, D * 0.8), 4))
    for i, method in enumerate(methods):
        profile = np.abs(attributions_by_method[method]).sum(axis=0)  # [D]
        ax.bar(x_pos + i * width, profile, width=width, label=method)
    ax.set_xticks(x_pos + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(channel_names)
    ax.set_ylabel('total |attribution|')
    legend = ax.legend(fontsize=8)
    for text in legend.get_texts():
        text.set_fontweight('bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_channel_importance_aggregate(channel_importance_by_method, n_used, save_path, channel_names=None):
    methods = list(channel_importance_by_method.keys())
    D = len(next(iter(channel_importance_by_method.values())))
    channel_names = channel_names or [f'ch{d}' for d in range(D)]
    width = 0.8 / len(methods)
    x_pos = np.arange(D)
    fig, ax = plt.subplots(figsize=(max(7, D * 1.1), 4))
    for i, method in enumerate(methods):
        ax.bar(x_pos + i * width, channel_importance_by_method[method], width=width, label=method)
    ax.set_xticks(x_pos + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(channel_names, fontsize=10)
    ax.set_ylabel('mean |attribution|')
    legend = ax.legend(fontsize=9)
    for text in legend.get_texts():
        text.set_fontweight('bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_channel_importance_combined(channel_importance_by_method, timeshap_feature, save_path,
                                     channel_names=None):
    pretty = {'integrated_gradients': 'Integrated Gradients', 'gradient_shap': 'GradientSHAP'}
    combined = {}
    for method, vec in channel_importance_by_method.items():
        if method in pretty:
            combined[pretty[method]] = np.asarray(vec, dtype=float)
    combined['TimeSHAP'] = np.asarray(timeshap_feature, dtype=float)
    D = len(next(iter(combined.values())))
    channel_names = channel_names or [f'ch{d}' for d in range(D)]
    x_pos = np.arange(D)
    width = 0.8 / len(combined)
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    for i, (method, vec) in enumerate(combined.items()):
        total = vec.sum()
        rel = vec / total if total > 0 else vec
        ax.bar(x_pos + i * width, rel, width=width, label=method)
    ax.set_xticks(x_pos + width * (len(combined) - 1) / 2)
    ax.set_xticklabels(channel_names, fontsize=10)
    ax.set_ylabel('relative importance', fontsize=11)
    legend = ax.legend(fontsize=9)
    for text in legend.get_texts():
        text.set_fontweight('bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_time_importance_aggregate(time_importance_by_method, n_used, save_path):
    pretty = {'integrated_gradients': 'Integrated Gradients', 'gradient_shap': 'GradientSHAP'}
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for method, profile in time_importance_by_method.items():
        ax.plot(profile, label=pretty.get(method, method), linewidth=1.5)
    ax.set_xlabel('timestep')
    ax.set_ylabel('mean |attribution|')
    legend = ax.legend(fontsize=9)
    for text in legend.get_texts():
        text.set_fontweight('bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _accumulate_importance_profiles(channel_totals, time_totals, model, x_batch, target_cluster,
                                     methods, ig_steps, gradientshap_samples, occlusion_max_cells,
                                     internal_batch_size, chunk_size, heatmap_totals=None):
    for start in range(0, x_batch.shape[0], chunk_size):
        chunk = x_batch[start:start + chunk_size]
        attributions = compute_attributions(
            model, chunk, target_cluster, methods, ig_steps, gradientshap_samples,
            occlusion_max_cells, internal_batch_size, verbose=False,
        )
        for method, attr in attributions.items():
            attr_abs = attr.abs()
            channel_totals[method] = channel_totals.get(method, 0.0) + attr_abs.sum(dim=(0, 1)).cpu().numpy()
            time_totals[method] = time_totals.get(method, 0.0) + attr_abs.sum(dim=(0, 2)).cpu().numpy()
            if heatmap_totals is not None:
                heatmap_totals[method] = heatmap_totals.get(method, 0.0) + attr_abs.sum(dim=0).cpu().numpy()
    return x_batch.shape[0]


def compute_mean_importance_profiles(model, test_loader, target_cluster, methods, ig_steps,
                                      gradientshap_samples, occlusion_max_cells, internal_batch_size,
                                      chunk_size, max_samples, return_heatmap=False):
    target_cluster = min(target_cluster, model.num_cluster - 1)
    channel_totals, time_totals = {}, {}
    heatmap_totals = {} if return_heatmap else None
    n_seen = 0
    for x_batch, _, _ in test_loader:
        if 0 < max_samples <= n_seen:
            break
        x_batch = x_batch.to(model.device)
        if max_samples > 0:
            x_batch = x_batch[:max_samples - n_seen]
        n_seen += _accumulate_importance_profiles(
            channel_totals, time_totals, model, x_batch, target_cluster, methods, ig_steps,
            gradientshap_samples, occlusion_max_cells, internal_batch_size, chunk_size,
            heatmap_totals=heatmap_totals,
        )
    channel_means = {m: t / n_seen for m, t in channel_totals.items()}
    time_means = {m: t / n_seen for m, t in time_totals.items()}
    if return_heatmap:
        heatmap_means = {m: t / n_seen for m, t in heatmap_totals.items()}
        return channel_means, time_means, heatmap_means, n_seen
    return channel_means, time_means, n_seen


def compute_mean_importance_profiles_from_tensor(model, x_all, target_cluster, methods, ig_steps,
                                                   gradientshap_samples, occlusion_max_cells,
                                                   internal_batch_size, chunk_size):
    target_cluster = min(target_cluster, model.num_cluster - 1)
    channel_totals, time_totals = {}, {}
    n_seen = _accumulate_importance_profiles(
        channel_totals, time_totals, model, x_all.to(model.device), target_cluster, methods, ig_steps,
        gradientshap_samples, occlusion_max_cells, internal_batch_size, chunk_size,
    )
    channel_means = {m: t / n_seen for m, t in channel_totals.items()}
    time_means = {m: t / n_seen for m, t in time_totals.items()}
    return channel_means, time_means, n_seen


def plot_saliency_overlay(x_sample, attr_sample, method_name, save_path, channel_names=None):
    T, D = x_sample.shape
    channel_names = channel_names or [f'ch{d}' for d in range(D)]
    importance = np.abs(attr_sample).sum(axis=1)  # [T]
    importance = importance / (importance.max() + 1e-12)

    y_min, y_max = x_sample.min(), x_sample.max()
    pad = 0.1 * (y_max - y_min + 1e-8)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(
        importance[np.newaxis, :], aspect='auto', cmap='OrRd', alpha=0.55,
        extent=[0, T - 1, y_min - pad, y_max + pad], origin='lower', vmin=0, vmax=1,
    )
    for d in range(D):
        ax.plot(x_sample[:, d], linewidth=1.0, label=channel_names[d])
    ax.set_xlim(0, T - 1)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xlabel('timestep')
    ax.set_ylabel(f'{method_name}\nvalue')
    if D <= 10:
        legend = ax.legend(fontsize=6, ncol=min(D, 5))
        for text in legend.get_texts():
            text.set_fontweight('bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_aggregate_heatmap(attributions_by_method_full, save_path):
    methods = list(attributions_by_method_full.keys())
    fig, axes = plt.subplots(len(methods), 1, figsize=(11, 3.2 * len(methods)), sharex=True, squeeze=False)
    for row, method in enumerate(methods):
        mean_abs = np.abs(attributions_by_method_full[method]).mean(axis=0)  # [T,D]
        ax = axes[row][0]
        im = ax.imshow(mean_abs.T, aspect='auto', cmap='viridis')
        ax.set_ylabel(f'{method}\nchannel')
        cbar = fig.colorbar(im, ax=ax, label='mean |attribution|')
        cbar.ax.yaxis.label.set_weight('bold')
    axes[-1][0].set_xlabel('timestep')
    fig.tight_layout(h_pad=3.0)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_ig_heatmap(mean_abs_TD, channel_names, save_path):
    D = mean_abs_TD.shape[1]
    channel_names = channel_names or [f'ch{d}' for d in range(D)]
    scaled = mean_abs_TD * 1e3
    vmax = float(np.percentile(scaled[1:], 99)) if scaled.shape[0] > 1 else None
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    im = ax.imshow(scaled.T, aspect='auto', cmap='viridis', vmax=vmax)
    ax.set_yticks(range(D))
    ax.set_yticklabels(channel_names, fontsize=9)
    ax.set_xlabel('timestep', fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_title(r'$\times 10^{-3}$', fontsize=9, pad=6)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_method_agreement(attributions_by_method_full, save_path):
    methods = list(attributions_by_method_full.keys())
    if len(methods) < 2:
        return
    flat = {m: attributions_by_method_full[m].reshape(-1) for m in methods}
    n = len(methods)
    corr = np.eye(n)
    for i in range(n):
        for j in range(n):
            corr[i, j] = np.corrcoef(flat[methods[i]], flat[methods[j]])[0, 1]

    fig, ax = plt.subplots(figsize=(3.0 + 2.2 * n, 1.8 + n))
    im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(methods, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center', fontsize=9,
                     color='white' if abs(corr[i, j]) > 0.6 else 'black')
    cbar = fig.colorbar(im, ax=ax, label='correlation')
    cbar.ax.yaxis.label.set_weight('bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def run_gradient_explanations(model, x, target_cluster, methods, output_dir, ig_steps,
                               gradientshap_samples, occlusion_max_cells, internal_batch_size,
                               num_examples, channel_names=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    target_cluster = min(target_cluster, model.num_cluster - 1)
    channel_names = channel_names or [f'ch{d}' for d in range(x.shape[2])]

    with torch.no_grad():
        membership = model.predict_membership(x)
    print(f"\npredict_membership (Captum): shape {tuple(membership.shape)}, target cluster = {target_cluster}")
    print(f"  sample 0 membership: {membership[0].detach().cpu().numpy().round(3)}")

    attributions = compute_attributions(
        model, x, target_cluster, methods, ig_steps, gradientshap_samples,
        occlusion_max_cells, internal_batch_size,
    )
    attributions_np_full = {m: a.cpu().numpy() for m, a in attributions.items()}
    x_np = x.detach().cpu().numpy()

    n_examples = min(num_examples, x.shape[0])
    for i in range(n_examples):
        attr_by_method = {m: a[i] for m, a in attributions_np_full.items()}
        plot_attribution_heatmap(x_np[i], attr_by_method, output_dir / f'heatmap_example{i}.png', channel_names)
        plot_time_importance(x_np[i], attr_by_method, output_dir / f'time_importance_example{i}.png')
        plot_channel_importance(x_np[i], attr_by_method, output_dir / f'channel_importance_example{i}.png', channel_names)
        primary_method = 'integrated_gradients' if 'integrated_gradients' in attr_by_method else next(iter(attr_by_method))
        plot_saliency_overlay(
            x_np[i], attr_by_method[primary_method], primary_method,
            output_dir / f'saliency_overlay_example{i}.png', channel_names,
        )

    plot_aggregate_heatmap(attributions_np_full, output_dir / 'aggregate_heatmap.png')
    plot_method_agreement(attributions_np_full, output_dir / 'method_agreement.png')

    print(f"  Plots saved to: {output_dir}")
