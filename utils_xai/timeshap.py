"""TimeSHAP event/feature/cell-level explanations and SHAP-style plots
(force, waterfall, violin, beeswarm, scatter, feature/event impact)."""
import sys
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients

# TimeSHAP requires an isolated numpy/scipy/shap stack in ./timeshap_pkgs/
# (see main_xai.py); the path is normally prepended by the entry point before
# numpy is imported. Re-assert it here so this module is self-contained, and
# import the timeshap library only when --with-timeshap is active.
WITH_TIMESHAP = '--with-timeshap' in sys.argv
if WITH_TIMESHAP:
    _pkgs = Path(__file__).resolve().parent.parent / 'timeshap_pkgs'
    if _pkgs.is_dir() and str(_pkgs) not in sys.path:
        sys.path.insert(0, str(_pkgs))
    from timeshap.explainer import calc_local_report
    from timeshap.explainer.feature_level import local_feat
    from timeshap.explainer.event_level import local_event


def make_timeshap_forward(model, target_cluster):
    def f(x_3d: np.ndarray) -> np.ndarray:
        x_t = torch.from_numpy(np.asarray(x_3d, dtype=np.float32)).to(model.device)
        with torch.no_grad():
            out = model.predict_membership(x_t)[:, target_cluster]
        return out.detach().cpu().numpy()[:, None]
    return f


def safe_event_nsamples(nsamples, T):
    safe_min = 3 * T
    if nsamples < safe_min:
        print(f"    (event-level: nsamples={nsamples} < 3*T={safe_min} for T={T} risks "
              f"a numerical explosion of the KernelSHAP solve -- raised to {safe_min})")
        return safe_min
    return nsamples


def timeshap_event_label_to_timestep(label, T):
    m = re.match(r'^Event (-?\d+)$', str(label))
    if not m:
        return None
    k = -int(m.group(1))
    return T - k


def feature_df_to_array(feature_data, feature_names):
    vals = feature_data.set_index('Feature')['Shapley Value']
    return np.array([vals[name] for name in feature_names])


def compute_base_value(model, target_cluster, T, D):
    forward_func = make_timeshap_forward(model, target_cluster)
    return float(forward_func(np.zeros((1, T, D)))[0, 0])


def compute_intrinsic_and_captum_for_timeshap(model, x_single, target_cluster, ig_steps):
    with torch.no_grad():
        _, cluster_weights_views, _ = model.encode_with_pooling(x_single, return_cluster_weights=True)
    w_btc = torch.stack([w[0, :, target_cluster] for w in cluster_weights_views], dim=0).mean(dim=0)
    w_btc = w_btc.detach().cpu().numpy()

    def forward_func(x):
        return model.predict_membership(x)[:, target_cluster]

    x_grad = x_single.clone().detach().requires_grad_(True)
    ig = IntegratedGradients(forward_func)
    attr, delta = ig.attribute(
        x_grad, baselines=torch.zeros_like(x_grad), n_steps=ig_steps, return_convergence_delta=True,
    )
    attr_np = attr[0].detach().cpu().numpy()  # [T,D]
    return {
        'w_btc': w_btc,
        'ig_time_profile': np.abs(attr_np).sum(axis=1),
        'ig_convergence_delta': float(delta.detach().cpu().numpy()[0]),
    }


def get_intrinsic_and_captum_cached(cache, model, x_single, target_cluster, idx, ig_steps):
    key = (target_cluster, idx)
    if key not in cache:
        cache[key] = compute_intrinsic_and_captum_for_timeshap(model, x_single, target_cluster, ig_steps)
    return cache[key]


def get_timeshap_event_profile_cached(cache, model, x_single, target_cluster, idx, seed, args):
    key = (target_cluster, idx, seed)
    if key not in cache:
        T, D = x_single.shape[1], x_single.shape[2]
        x_np = x_single.detach().cpu().numpy().astype(np.float64)
        baseline = np.zeros((1, D))
        forward_func = make_timeshap_forward(model, target_cluster)
        event_dict = {'rs': seed, 'nsamples': safe_event_nsamples(args.event_nsamples, T)}
        event_data = local_event(forward_func, x_np, event_dict, None, None, baseline, pruned_idx=0)
        cache[key] = timeshap_event_profile_from_df(event_data, T)
    return cache[key]


def compute_mean_intrinsic_captum(model, x_all, target_cluster, ig_steps):
    n, T, D = x_all.shape
    w_btc_accum = np.zeros(T)
    ig_accum = np.zeros(T)
    for i in range(n):
        res = compute_intrinsic_and_captum_for_timeshap(model, x_all[i:i + 1], target_cluster, ig_steps)
        w_btc_accum += res['w_btc']
        ig_accum += res['ig_time_profile']
    return w_btc_accum / n, ig_accum / n


def plot_timeshap_cell(cell_data, T, save_path):
    df = cell_data[~cell_data['Event'].isin(['Pruned Events'])].copy()
    df['timestep'] = df['Event'].apply(lambda lbl: timeshap_event_label_to_timestep(lbl, T))
    df_events = df.dropna(subset=['timestep']).sort_values('timestep')
    if df_events.empty:
        return
    timesteps = sorted(df_events['timestep'].unique())
    feats = sorted(df_events['Feature'].unique())
    grid = np.full((len(feats), len(timesteps)), np.nan)
    for _, row in df_events.iterrows():
        i = feats.index(row['Feature'])
        j = timesteps.index(row['timestep'])
        grid[i, j] = row['Shapley Value']

    fig, ax = plt.subplots(figsize=(max(6, len(timesteps) * 0.8), max(3, len(feats) * 0.6)))
    vmax = np.nanmax(np.abs(grid)) if not np.all(np.isnan(grid)) else 1.0
    im = ax.imshow(grid, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(timesteps)))
    ax.set_xticklabels([int(t) for t in timesteps])
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats)
    ax.set_xlabel('timestep (top pruned/event-relevant only)')
    cbar = fig.colorbar(im, ax=ax, label='Shapley value')
    cbar.ax.yaxis.label.set_weight('bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def timeshap_event_profile_from_df(event_data, T):
    df = event_data.copy()
    df['timestep'] = df['Feature'].apply(lambda lbl: timeshap_event_label_to_timestep(lbl, T))
    df = df.dropna(subset=['timestep'])
    profile = np.full(T, np.nan)
    for _, row in df.iterrows():
        profile[int(row['timestep'])] = row['Shapley Value']
    return profile


def print_cross_method_correlations(w_btc, ig_profile, timeshap_profile):
    valid = ~np.isnan(timeshap_profile) & ~np.isnan(w_btc) & ~np.isnan(ig_profile)
    if valid.sum() >= 2:
        corr_ts_ig = np.corrcoef(timeshap_profile[valid], ig_profile[valid])[0, 1]
        corr_ts_wbtc = np.corrcoef(timeshap_profile[valid], w_btc[valid])[0, 1]
        print(f"    correlation (Pearson) TimeSHAP vs IG:        {corr_ts_ig:.3f}")
        print(f"    correlation (Pearson) TimeSHAP vs w_{{b,t,c}}: {corr_ts_wbtc:.3f}")


def _normalize_profile(v):
    v = np.asarray(v, dtype=float)
    denom = np.nanmax(np.abs(v))
    return v / denom if denom > 0 else v


def plot_cross_method_comparison_arrays(w_btc, ig_profile, timeshap_profile, save_path, subtitle=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(_normalize_profile(w_btc), label='intrinsic w_{b,t,c}', linewidth=1.5)
    ax.plot(_normalize_profile(ig_profile), label='Captum IntegratedGradients', linewidth=1.5)
    ax.plot(_normalize_profile(timeshap_profile), label='TimeSHAP event-level', linewidth=1.5, linestyle='--')
    ax.set_xlabel('timestep')
    ax.set_ylabel(f'normalized importance\n({subtitle})' if subtitle else 'normalized importance')
    legend = ax.legend(fontsize=8)
    for text in legend.get_texts():
        text.set_fontweight('bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_cross_method_comparison_all_classes(profiles_by_cluster, save_path):
    clusters = sorted(profiles_by_cluster.keys())
    fig, axes, n_rows, n_cols = _grid_axes(len(clusters), 5, 3.2)
    for i, c in enumerate(clusters):
        ax = axes[i // n_cols][i % n_cols]
        w_btc, ig_profile, timeshap_profile = profiles_by_cluster[c]
        ax.plot(_normalize_profile(w_btc), label='intrinsic', linewidth=1.2)
        ax.plot(_normalize_profile(ig_profile), label='Captum IG', linewidth=1.2)
        ax.plot(_normalize_profile(timeshap_profile), label='TimeSHAP', linewidth=1.2, linestyle='--')
        ax.set_title(f'cluster {c}', fontsize=9)
        if i == 0:
            legend = ax.legend(fontsize=6)
            for text in legend.get_texts():
                text.set_fontweight('bold')
    for i in range(len(clusters), n_rows * n_cols):
        axes[i // n_cols][i % n_cols].axis('off')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _draw_cascading_bars(ax, shap_row, order, feature_names, base_value):
    cum = base_value
    max_abs = np.abs(shap_row).max() if shap_row.size else 1.0
    for row, idx in enumerate(order):
        val = shap_row[idx]
        color = '#ff0051' if val >= 0 else '#008bfb'
        ax.barh(row, val, left=cum, color=color, edgecolor='black', linewidth=0.5)
        sign = '+' if val >= 0 else ''
        label = f'{sign}{val:.3f}'
        if max_abs > 0 and abs(val) > 0.15 * max_abs:
            ax.text(cum + val / 2, row, label, va='center', ha='center', fontsize=8, color='white')
        else:
            ha = 'left' if val >= 0 else 'right'
            ax.text(cum + val + (0.01 if val >= 0 else -0.01), row, label, va='center', ha=ha, fontsize=8)
        cum += val
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_names[i] for i in order])
    ax.margins(x=0.15)
    return cum


def plot_shap_force(shap_row, feature_names, base_value, output_value, save_path):
    order = list(range(len(shap_row)))
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    cum = _draw_cascading_bars(ax, shap_row, order, feature_names, base_value)
    ax.axvline(base_value, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('fuzzy membership', fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_shap_waterfall(shap_row, feature_names, base_value, output_value, save_path):
    order = np.argsort(np.abs(shap_row))
    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(order))))
    cum = _draw_cascading_bars(ax, shap_row, order, feature_names, base_value)
    ax.axvline(base_value, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel(f'model output (predict_membership) — E[f(x)]={base_value:.3f}, f(x)={cum:.3f}')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _grid_axes(n_panels, panel_w, panel_h):
    n_cols = min(3, n_panels)
    n_rows = math.ceil(n_panels / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_w * n_cols, panel_h * n_rows), squeeze=False)
    return fig, axes, n_rows, n_cols


def plot_shap_force_all_classes(shap_by_cluster, feature_names, base_value_by_cluster, save_path):
    clusters = sorted(shap_by_cluster.keys())
    D = len(feature_names)
    fig, axes, n_rows, n_cols = _grid_axes(len(clusters), 5, max(3, 0.5 * D))
    order = list(range(D))
    for i, c in enumerate(clusters):
        ax = axes[i // n_cols][i % n_cols]
        base_value = base_value_by_cluster[c]
        cum = _draw_cascading_bars(ax, shap_by_cluster[c], order, feature_names, base_value)
        ax.axvline(base_value, color='gray', linestyle='--', linewidth=1)
        ax.set_title(f'cluster {c}\nE[f(x)]={base_value:.3f} -> f(x)={cum:.3f}', fontsize=9)
    for i in range(len(clusters), n_rows * n_cols):
        axes[i // n_cols][i % n_cols].axis('off')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_shap_waterfall_all_classes(shap_by_cluster, feature_names, base_value_by_cluster, save_path):
    clusters = sorted(shap_by_cluster.keys())
    D = len(feature_names)
    fig, axes, n_rows, n_cols = _grid_axes(len(clusters), 5, max(3, 0.5 * D))
    for i, c in enumerate(clusters):
        ax = axes[i // n_cols][i % n_cols]
        base_value = base_value_by_cluster[c]
        order = np.argsort(np.abs(shap_by_cluster[c]))
        cum = _draw_cascading_bars(ax, shap_by_cluster[c], order, feature_names, base_value)
        ax.axvline(base_value, color='gray', linestyle='--', linewidth=1)
        ax.set_title(f'cluster {c}\nE[f(x)]={base_value:.3f} -> f(x)={cum:.3f}', fontsize=9)
    for i in range(len(clusters), n_rows * n_cols):
        axes[i // n_cols][i % n_cols].axis('off')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_shap_violin(shap_matrix, feature_names, save_path, subtitle):
    D = shap_matrix.shape[1]
    order = np.argsort(np.abs(shap_matrix).mean(axis=0))
    fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * D)))
    data = [shap_matrix[:, d] for d in order]
    ax.violinplot(data, vert=False, showmeans=True)
    ax.set_yticks(np.arange(1, D + 1))
    ax.set_yticklabels(np.array(feature_names)[order])
    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_xlabel(f'SHAP value ({subtitle})')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_shap_beeswarm(shap_matrix, feature_values, feature_names, save_path, subtitle):
    n, D = shap_matrix.shape
    order = np.argsort(np.abs(shap_matrix).mean(axis=0))
    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * D)))
    rng = np.random.default_rng(0)
    fv_range = feature_values.max(axis=0) - feature_values.min(axis=0)
    fv_norm = (feature_values - feature_values.min(axis=0)) / (fv_range + 1e-12)
    sc = None
    for row, d in enumerate(order):
        y_jitter = row + 1 + rng.uniform(-0.3, 0.3, size=n)
        sc = ax.scatter(shap_matrix[:, d], y_jitter, c=fv_norm[:, d], cmap='coolwarm', s=30, vmin=0, vmax=1)
    ax.set_yticks(np.arange(1, D + 1))
    ax.set_yticklabels(np.array(feature_names)[order])
    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label('feature value')
        cbar.ax.yaxis.label.set_weight('bold')
    ax.set_xlabel(f'SHAP value ({subtitle})')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_shap_scatter(shap_matrix, feature_values, feature_names, save_path):
    D = shap_matrix.shape[1]
    n_cols = min(3, D)
    n_rows = math.ceil(D / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False)
    for d in range(D):
        ax = axes[d // n_cols][d % n_cols]
        ax.scatter(feature_values[:, d], shap_matrix[:, d], color='#008bfb')
        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
        ax.set_xlabel(f'{feature_names[d]} value')
        ax.set_ylabel('SHAP value')
    for d in range(D, n_rows * n_cols):
        axes[d // n_cols][d % n_cols].axis('off')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_shap_feature_impact_single(mean_abs_feature, feature_names, n_used, save_path, subtitle):
    order = np.argsort(mean_abs_feature)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(feature_names))))
    ax.barh(np.array(feature_names)[order], mean_abs_feature[order], color='#008bfb')
    ax.set_xlabel('mean(|SHAP value|)')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_shap_feature_impact_all_classes(mean_abs_by_cluster, feature_names, save_path):
    clusters = sorted(mean_abs_by_cluster.keys())
    D = len(feature_names)
    x_pos = np.arange(D)
    width = 0.8 / max(1, len(clusters))
    fig, ax = plt.subplots(figsize=(max(7, D * 1.0), 5))
    cmap = plt.get_cmap('tab20')
    for i, c in enumerate(clusters):
        ax.bar(x_pos + i * width, mean_abs_by_cluster[c], width=width, label=f'cluster {c}', color=cmap(i % 20))
    ax.set_xticks(x_pos + width * (len(clusters) - 1) / 2)
    ax.set_xticklabels(feature_names)
    ax.set_ylabel('mean(|SHAP value|)')
    legend = ax.legend(fontsize=7, ncol=min(len(clusters), 6))
    for text in legend.get_texts():
        text.set_fontweight('bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_shap_event_impact(mean_abs_event, n_used, save_path, subtitle):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(mean_abs_event, color='#008bfb', linewidth=1.5)
    ax.set_xlabel('timestep')
    ax.set_ylabel('mean(|SHAP value|)')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def explain_one_timeshap_example(model, x_single, target_cluster, example_dir, args, label, cache, idx,
                                  channel_names=None):
    T, D = x_single.shape[1], x_single.shape[2]
    channel_names = channel_names or [f'ch{d}' for d in range(D)]
    example_dir.mkdir(parents=True, exist_ok=True)

    intrinsic_captum = get_intrinsic_and_captum_cached(cache['intrinsic'], model, x_single, target_cluster, idx, args.ig_steps)
    print(f"  [{label}] intrinsic/Captum done (IG convergence delta = "
          f"{intrinsic_captum['ig_convergence_delta']:.6f})")

    x_np = x_single.detach().cpu().numpy().astype(np.float64)
    baseline = np.zeros((1, D))
    forward_func = make_timeshap_forward(model, target_cluster)

    pruning_dict = {'tol': args.tol}
    event_dict = {'rs': args.seed, 'nsamples': safe_event_nsamples(args.event_nsamples, T)}
    feature_dict = {'rs': args.seed, 'nsamples': args.feature_nsamples,
                     'feature_names': channel_names}
    cell_dict = {'rs': args.seed, 'nsamples': args.cell_nsamples,
                 'top_x_events': args.cell_top_events, 'top_x_feats': args.cell_top_feats}

    _, event_data, feature_data, cell_data = calc_local_report(
        forward_func, x_np, pruning_dict, event_dict, feature_dict, cell_dict, baseline=baseline,
    )
    print(f"  [{label}] TimeSHAP done (pruning + event + feature + cell level)")
    plot_timeshap_cell(cell_data, T, example_dir / 'timeshap_cell.png')

    timeshap_profile = timeshap_event_profile_from_df(event_data, T)
    cache['event'][(target_cluster, idx, args.seed)] = timeshap_profile
    print_cross_method_correlations(intrinsic_captum['w_btc'], intrinsic_captum['ig_time_profile'], timeshap_profile)


def compute_feature_event_profiles(model, x_all, target_cluster, event_nsamples, feature_nsamples, seed,
                                    channel_names=None):
    n, T, D = x_all.shape
    channel_names = channel_names or [f'ch{d}' for d in range(D)]
    baseline = np.zeros((1, D))
    forward_func = make_timeshap_forward(model, target_cluster)
    feature_dict = {'rs': seed, 'nsamples': feature_nsamples, 'feature_names': channel_names}
    event_dict = {'rs': seed, 'nsamples': safe_event_nsamples(event_nsamples, T)}

    feat_accum = np.zeros(D)
    event_accum = np.zeros(T)
    shap_matrix = np.zeros((n, D))
    feature_values = np.zeros((n, D))
    for i in range(n):
        x_np = x_all[i:i + 1].detach().cpu().numpy().astype(np.float64)
        feat_data = local_feat(forward_func, x_np, feature_dict, None, None, baseline, pruned_idx=0)
        event_data = local_event(forward_func, x_np, event_dict, None, None, baseline, pruned_idx=0)
        signed = feature_df_to_array(feat_data, channel_names)
        shap_matrix[i] = signed
        feature_values[i] = x_np[0].mean(axis=0)
        feat_accum += np.abs(signed)
        for _, row in event_data.iterrows():
            t = timeshap_event_label_to_timestep(row['Feature'], T)
            if t is not None:
                event_accum[t] += abs(row['Shapley Value'])
    return feat_accum / n, event_accum / n, shap_matrix, feature_values, n

