"""Runs FMMVCC_Model's XAI functions on an already-trained model.
Example:
    python main_xai.py --dataset-name WriGest
    python main_xai.py --dataset-name WriGest --with-captum --with-timeshap --with-fidelity
"""
import sys
from pathlib import Path

PATH = Path(__file__).parent.absolute()

WITH_TIMESHAP = '--with-timeshap' in sys.argv
if WITH_TIMESHAP:
    TIMESHAP_PKGS = PATH / 'timeshap_pkgs'
    if not TIMESHAP_PKGS.is_dir():
        raise SystemExit(
            f"{TIMESHAP_PKGS} not found. Create it once with:\n"
            f"  pip install --target={TIMESHAP_PKGS} --no-deps numpy==1.26.4 scipy==1.11.4 shap==0.37.0\n"
            f"  pip install --target={TIMESHAP_PKGS} --no-deps timeshap pandas seaborn plotly altair "
            "feedzai-altair-theme\n"
            "(the system shap/numpy are incompatible with timeshap)."
        )
    sys.path.insert(0, str(TIMESHAP_PKGS))

import argparse  
import numpy as np  
import torch  
from main import WRIGEST_MACRO_CATEGORIES  

from utils_xai.common import *       
from utils_xai.prototypes import *   
from utils_xai.gradients import *    
from utils_xai.timeshap import *     
from utils_xai.fidelity import *     
from utils_xai.fidelity import _curve_auc  


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the XAI explanations (predict_membership, prototype '
                     'path, view-ablation importance, nearest-medoid prototypes) '
                     'on an already-trained FMMVCC model.'
    )
    parser.add_argument('--dataset-type', type=str, default='multivariate',
                         choices=['univariate', 'multivariate'])
    parser.add_argument('--dataset-position', type=int, default=None)
    parser.add_argument('--dataset-name', type=str, default=None)
    parser.add_argument(
        '--wrigest-macro-category',
        type=str,
        default=None,
        choices=sorted(WRIGEST_MACRO_CATEGORIES.keys()),
        help='WriGest only: explain a checkpoint trained on just this macro category '
              '(see main.py --wrigest-macro-category, WriGest/category.txt). Requires '
              '--dataset-name WriGest. Filters the loaded data the same way training did, '
              "and loads/saves under '<dataset-name>_<category>Gesture' (e.g. "
              'WriGest_NumberGesture) instead of plain WriGest, matching where main.py '
              'saved that checkpoint.',
    )
    parser.add_argument('--mode', type=str, default='unidirectional',
                         choices=['unidirectional', 'bidirectional'])
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--output-dims', type=int, default=64)
    parser.add_argument('--hidden-dims', type=int, default=64)
    parser.add_argument('--n-layers', type=int, default=4)
    # These three determine the name of the checkpoint to load (see
    # batch_run.py): they must match the training configuration.
    parser.add_argument('--num-views', type=int, default=4)
    parser.add_argument('--separation-weight', type=float, default=1.0)
    parser.add_argument('--balance-weight', type=float, default=0.5)
    parser.add_argument('--m', type=float, default=1.5)
    parser.add_argument('--pretraining-epoch', type=int, default=100,
                         help='Not used at inference time: only required by the FMMVCC_Model constructor.')
    parser.add_argument('--device', type=str, default=None,
                        help="Default: 'cuda' if available, otherwise 'cpu'.")
    parser.add_argument('--max-medoid-samples', type=int, default=None,
                        help='Limits the number of train samples used for '
                              'nearest-medoid (default: uses the whole train set).')
    parser.add_argument('--medoid-batch-size', type=int, default=None,
                        help='Chunk size for the encode_with_pooling/predict_membership forward '
                              'passes over the (possibly whole) train set used by '
                              'nearest_medoid_prototypes and cluster_purity: a single unbatched '
                              'pass over a large train set (e.g. WriGest) can OOM. '
                              'Default: --batch-size.')
    parser.add_argument('--similarity-threshold', type=float, default=0.95,
                        help='Cosine similarity threshold above which two '
                              'clusters are flagged as overlapping '
                              '(default: 0.95).')
    parser.add_argument('--max-similarity-examples', type=int, default=20,
                        help='Maximum number of cluster series used to '
                              'compute the prototype-cluster score in the '
                              'idx* plots (default: 20; uses however many are '
                              'available if the cluster has fewer).')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Directory where XAI results are saved (.png only). '
                              'Default: plot/<FMMVCC[_mode]>/<dataset_name>/'
                              'NViews{V}_Sep{S}_Bal{B}/XAI/.')
    parser.add_argument('--skip-plots', action='store_true',
                        help='Save only the .npy files, without generating the .png plots.')

    # --- Captum: gradient attributions on predict_membership(x)[:, c] ---
    parser.add_argument('--target-cluster', type=int, default=0,
                        help='Index of the cluster c explained by the Captum attributions '
                              '(predict_membership(x)[:, c]).')
    parser.add_argument('--gradient-methods', type=str, nargs='+',
                        default=['ig', 'gradientshap', 'occlusion'],
                        choices=['ig', 'gradientshap', 'occlusion'],
                        help="Captum methods to run (ig is mandatory and always included).")
    parser.add_argument('--ig-steps', type=int, default=64)
    parser.add_argument('--gradientshap-samples', type=int, default=20)
    parser.add_argument('--occlusion-max-cells', type=int, default=2000,
                         help='Occlusion evaluates one forward pass per (t,d): with large inputs '
                              'the method is skipped above this T*D threshold, to stay fast.')
    parser.add_argument('--gradient-max-samples', type=int, default=16,
                         help='Number of test-set samples used for the Captum attributions '
                              '(independent of --batch-size, to limit GPU time/memory).')
    parser.add_argument('--internal-batch-size', type=int, default=32,
                         help="IntegratedGradients internally evaluates n_steps copies of the batch "
                              "all at once: with heavy batches/models this can saturate GPU "
                              "memory. This parameter breaks that internal computation into chunks of "
                              "this size.")
    parser.add_argument('--channel-importance-max-samples', type=int, default=200,
                         help="Number of test-set samples over which "
                              "channel_importance_all_test.png/time_importance_all_test.png "
                              "are averaged (summary plots outside clusters/, more robust than the "
                              "--gradient-max-samples batch used by aggregate_heatmap/method_agreement). "
                              "<= 0 to use the entire test set.")
    parser.add_argument('--with-captum', action='store_true',
                         help="Also run the Captum attributions (IntegratedGradients/GradientShap/"
                              "Occlusion). Like --with-timeshap: opt-in, off by default.")

    # --- Per-cluster Captum: same analysis repeated on each cluster's members ---
    parser.add_argument('--cluster-gradient-examples', type=int, default=5,
                         help='Number of member samples of each cluster (predicted via argmax of '
                              'predict_membership on the train set) to analyze in '
                              '.../XAI/gradients/clusters/cluster_{c}/, explaining '
                              'predict_membership(x)[:, c] (the cluster those samples themselves belong to).')
    parser.add_argument('--skip-cluster-gradients', action='store_true',
                         help="Skip the per-cluster Captum analysis (can be slow with many "
                              "clusters: repeats IG/GradientShap/Occlusion once per active cluster).")

    # --- TimeSHAP (--with-timeshap): reuses --target-cluster, --ig-steps, --device, --output-dir ---
    parser.add_argument('--with-timeshap', action='store_true',
                         help="Also run the TimeSHAP attributions (requires the isolated "
                              './timeshap_pkgs/ environment, see the explicit error if missing). Note: the '
                              "actual check is on sys.argv at the top of the file (needed BEFORE "
                              "importing numpy), this flag is only for --help/validation.")
    parser.add_argument('--timeshap-cluster-examples', type=int, default=2,
                         help='Detail examples (cell-level only: it is the only TimeSHAP plot without '
                              'an *_all_cluster* aggregate equivalent) for each active cluster.')
    parser.add_argument('--tol', type=float, default=0.025, help='Tolerance for TimeSHAP pruning.')
    parser.add_argument('--event-nsamples', type=int, default=200,
                         help="KernelSHAP nsamples for the event-level (one 'player' per timestep, "
                              "pruned_idx=0): treated as a MINIMUM, automatically raised to 3*T "
                              "if T requires it (see safe_event_nsamples -- below an "
                              "empirical threshold around ~1.9*T the KernelSHAP solve is underdetermined and "
                              "explodes numerically, e.g. max|Shapley value|~4e13 observed with "
                              "nsamples=200 on T=640).")
    parser.add_argument('--feature-nsamples', type=int, default=200)
    parser.add_argument('--cell-nsamples', type=int, default=150)
    parser.add_argument('--cell-top-events', type=int, default=5)
    parser.add_argument('--cell-top-feats', type=int, default=3)
    parser.add_argument('--timeshap-cluster-avg-samples', type=int, default=0,
                         help='Members of each cluster over which |Shapley value| is averaged for '
                              'timeshap_*_all_cluster.png. <= 0 (default) = use ALL members.')
    parser.add_argument('--timeshap-global-avg-samples', type=int, default=0,
                         help="Generic test-set samples over which the 'random data' TimeSHAP plots "
                              "outside clusters/ are averaged. <= 0 (default) = use the WHOLE test set.")
    parser.add_argument('--avg-event-nsamples', type=int, default=80,
                         help="Like --event-nsamples but for the TimeSHAP aggregates (averaged over "
                              "several samples): same MINIMUM auto-scaled to 3*T (safe_event_nsamples), "
                              "necessary to avoid the per-sample numerical explosion -- an average "
                              "of exploded values is itself still exploded, it doesn't average away.")
    parser.add_argument('--avg-feature-nsamples', type=int, default=80)
    parser.add_argument('--skip-cluster-timeshap', action='store_true',
                         help="Skip the per-cluster TimeSHAP analysis (can be very slow with "
                              'many clusters: one full KernelSHAP pass per example per cluster).')
    parser.add_argument('--seed', type=int, default=42, help='Seed for KernelSHAP (TimeSHAP) and torch.')

    # --- Fidelity/stability (--with-fidelity): applied uniformly to all available methods
    # (intrinsic w_{b,t,c}, always; Captum IntegratedGradients, always, doesn't require --with-captum;
    # TimeSHAP event-level, only if --with-timeshap) ---
    parser.add_argument('--with-fidelity', action='store_true',
                         help='Computes fidelity (insertion/deletion AUC) and stability (correlation '
                              'across different seeds) for all available methods, and saves ONE final '
                              'comparison table (fidelity_stability_table.png). Opt-in, like '
                              '--with-captum/--with-timeshap.')
    parser.add_argument('--fidelity-examples', type=int, default=5,
                         help='Real examples for each active cluster on which deletion/insertion '
                              'AUC is computed (then averaged over all clusters into one number per method).')
    parser.add_argument('--fidelity-stability-seeds', type=int, default=3,
                         help='Number of different seeds (starting from --seed) over which the '
                              'TimeSHAP event-level profile is recomputed for stability. The deterministic '
                              'methods in this implementation (intrinsic w_{b,t,c}, Captum '
                              'IntegratedGradients with fixed baseline/steps) have no source of '
                              'randomness at all: their stability is trivially 1.0, not a fabricated value.')
    return parser.parse_args()




def main():
    args = parse_args()

    dataset_name, X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, n_clusters = load_dataset(args)
    print(
        f"Dataset: {dataset_name} | train={X_train_scaled.shape} "
        f"test={X_test_scaled.shape} | n_cluster={n_clusters}"
    )

    model, train_loader, test_loader = build_model(
        args, dataset_name, X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, n_clusters
    )

    finetune_path, centers_path = resolve_checkpoint_paths(
        dataset_name, args.mode, args.num_views, args.separation_weight, args.balance_weight
    )
    print(f"Loading checkpoint: {finetune_path}")
    print(f"Loading centers:    {centers_path}")
    model.load_for_xai(str(finetune_path), str(centers_path))
    print(f"u_mean loaded correctly: shape {tuple(model.u_mean.shape)}")

    run_label = 'FMMVCC' if args.mode == 'unidirectional' else f'FMMVCC_{args.mode}'
    output_dir = args.output_dir or (
        PATH / 'plot' / run_label / dataset_name /
        f"NViews{args.num_views}_Sep{args.separation_weight}_Bal{args.balance_weight}" /
        'XAI'
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- sample/view-level explanations on a test batch ---
    x_test, _, _ = next(iter(test_loader))
    x_test = x_test.to(model.device)
    # Single source of truth for channel labels across every plot in this run
    # (real sensor names for WriGest, generic ch0/ch1/... otherwise).
    channel_names = get_channel_names(dataset_name, x_test.shape[2])

    # inference only: no gradient needed here, avoids keeping the
    # autograd graph through the Mamba encoder (expensive in GPU memory)
    with torch.no_grad():
        membership = model.predict_membership(x_test)
    print(f"\npredict_membership: shape {tuple(membership.shape)}")
    print(f"  row 0 example: {membership[0].detach().cpu().numpy().round(3)}")

    with torch.no_grad():
        fused, cluster_weights_views, pooled_views = model.encode_with_pooling(
            x_test, return_cluster_weights=True
        )
    prototype_paths = model.prototype_path(cluster_weights_views)
    print(f"\nprototype_path per view: shapes {[tuple(p.shape) for p in prototype_paths]}")
    print(f"  sample 0 example, view 0: {prototype_paths[0][0].detach().cpu().numpy().tolist()}")

    importance = model.view_ablation_importance(x_test)
    mean_importance = importance.mean(dim=1).detach().cpu().numpy()
    std_importance = importance.std(dim=1).detach().cpu().numpy()
    print(f"\nview_ablation_importance (mean +- std over {x_test.shape[0]} samples, per view):")
    for i in range(len(mean_importance)):
        print(f"  view {i}: {mean_importance[i]:.4f} +- {std_importance[i]:.4f}")

    # --- nearest-medoid prototypes on (a subset of) the train set ---
    n_medoid = X_train_scaled.shape[0]
    if args.max_medoid_samples is not None:
        n_medoid = min(n_medoid, args.max_medoid_samples)
    medoid_batch_size = args.medoid_batch_size or args.batch_size
    raw_series = torch.from_numpy(X_train_scaled[:n_medoid]).float()
    prototypes, medoid_indices = model.nearest_medoid_prototypes(raw_series, chunk_size=medoid_batch_size)
    print(
        f"\nnearest_medoid_prototypes: shape {tuple(prototypes.shape)}, "
        f"indices in the train set (first {n_medoid} samples): "
        f"{medoid_indices.detach().cpu().numpy().tolist()}"
    )

    # hard cluster assignment (argmax of membership) for each
    # raw_series sample, used to pick real examples per cluster. Chunked
    # (medoid_batch_size) like nearest_medoid_prototypes above: predict_membership
    # -> encode_with_pooling has no batching of its own, and a single
    # unbatched pass over the whole train set can OOM on large datasets.
    with torch.no_grad():
        pred_labels_chunks = []
        for start in range(0, raw_series.shape[0], medoid_batch_size):
            chunk = raw_series[start:start + medoid_batch_size].to(model.device)
            pred_labels_chunks.append(model.predict_membership(chunk).argmax(dim=1).cpu())
        pred_labels_np = torch.cat(pred_labels_chunks).numpy()
    raw_series_np = raw_series.numpy()
    true_labels_np = y_train_encoded[:n_medoid]

    # --- cluster purity: how many true classes end up under the same cluster ---
    purity = compute_cluster_purity(pred_labels_np, true_labels_np, n_clusters)
    active_clusters = [c for c, (mi, tc) in purity.items() if mi.size > 0]
    pure_clusters = [c for c, (mi, tc) in purity.items() if len(tc) == 1]
    mixed_clusters_list = [c for c, (mi, tc) in purity.items() if len(tc) >= 2]
    print(
        f"\ncluster_purity (from argmax(predict_membership) on {n_medoid} train samples): "
        f"{len(active_clusters)}/{n_clusters} active clusters, "
        f"{len(pure_clusters)} pure (1 true class), "
        f"{len(mixed_clusters_list)} mixed (>= 2 true classes)"
    )
    for c in mixed_clusters_list:
        member_indices, true_classes = purity[c]
        print(f"  cluster {c}: {member_indices.size} samples, true classes {true_classes}")

    # --- similarity between cluster centers: which ones occupy the same space ---
    similarity = model.cluster_center_similarity()
    similarity_np = similarity.detach().cpu().numpy()
    n_cluster_sim = similarity_np.shape[0]
    overlapping_pairs = [
        (i, j, similarity_np[i, j])
        for i in range(n_cluster_sim)
        for j in range(i + 1, n_cluster_sim)
        if similarity_np[i, j] >= args.similarity_threshold
    ]
    overlapping_pairs.sort(key=lambda t: t[2], reverse=True)
    print(
        f"\ncluster_center_similarity: shape {tuple(similarity.shape)}, "
        f"pairs with cosine similarity >= {args.similarity_threshold}: {len(overlapping_pairs)}"
    )
    for i, j, sim in overlapping_pairs[:20]:
        print(f"  cluster {i} <-> cluster {j}: {sim:.3f}")
    if len(overlapping_pairs) > 20:
        print(f"  ... and {len(overlapping_pairs) - 20} more pairs")

    if not args.skip_plots:
        prototypes_dir = output_dir / 'prototypes'
        prototypes_dir.mkdir(parents=True, exist_ok=True)

        plot_membership_heatmap(membership, output_dir)
        plot_prototype_paths(prototype_paths, output_dir)
        plot_view_importance(importance, output_dir)
        plot_cluster_similarity(similarity, output_dir)
        plot_nearest_medoid_prototypes(prototypes, medoid_indices, prototypes_dir, channel_names)
        plot_nearest_medoid_prototypes_comparison(
            prototypes, medoid_indices, similarity, args.similarity_threshold, prototypes_dir, channel_names
        )
        plot_cluster_class_mix(
            model, prototypes, medoid_indices, raw_series_np, pred_labels_np, true_labels_np, prototypes_dir,
            max_examples=args.max_similarity_examples, channel_names=channel_names,
        )

    if args.with_captum:
        print("\n" + "=" * 78)
        print("Captum attributions (IntegratedGradients/GradientShap/Occlusion) on "
              "predict_membership(x)[:, c]")
        print("=" * 78)
        check_captum_dependency()
        gradients_dir = output_dir / 'gradients'

        # Outside clusters/, ONLY the summary plots are kept (averaged over
        # many samples): no per-single-example heatmap/time_importance/
        # saliency_overlay, which instead live inside clusters/cluster_{c}/
        # (num_examples=0 disables the per-example loop in
        # run_gradient_explanations, still leaving aggregate_heatmap.png
        # and method_agreement.png, computed over --gradient-max-samples samples).
        x_grad = x_test[:args.gradient_max_samples]
        run_gradient_explanations(
            model, x_grad, args.target_cluster, args.gradient_methods,
            gradients_dir,
            ig_steps=args.ig_steps, gradientshap_samples=args.gradientshap_samples,
            occlusion_max_cells=args.occlusion_max_cells, internal_batch_size=args.internal_batch_size,
            num_examples=0, channel_names=channel_names,
        )

        # channel_importance_all_test.png / time_importance_all_test.png:
        # unlike aggregate_heatmap.png (limited to --gradient-max-samples for
        # cost reasons), here the average is over (a larger subset or) the
        # entire test set, for a much more robust estimate of which
        # channels/instants really matter for the target cluster.
        print(f"\nchannel/time_importance_all_test: averaging over "
              f"{'the whole test set' if args.channel_importance_max_samples <= 0 else f'max {args.channel_importance_max_samples} samples'} "
              f"of the test set...")
        channel_means, time_means, heatmap_means, n_used = compute_mean_importance_profiles(
            model, test_loader, args.target_cluster, args.gradient_methods, args.ig_steps,
            args.gradientshap_samples, args.occlusion_max_cells, args.internal_batch_size,
            chunk_size=args.gradient_max_samples, max_samples=args.channel_importance_max_samples,
            return_heatmap=True,
        )
        plot_channel_importance_aggregate(channel_means, n_used, gradients_dir / 'channel_importance_all_test.png', channel_names)
        plot_time_importance_aggregate(time_means, n_used, gradients_dir / 'time_importance_all_test.png')
        if 'integrated_gradients' in heatmap_means:
            plot_ig_heatmap(heatmap_means['integrated_gradients'],
                            channel_names, gradients_dir / 'ig_heatmap_all_test.png')
        print(f"  channel_importance_all_test.png / time_importance_all_test.png computed over {n_used} test-set samples.")

        # --- same Captum analysis repeated on each cluster's members ---
        # unlike the block above (where target_cluster is fixed, e.g. 0,
        # for generic test-set samples), here for each cluster c
        # predict_membership(x)[:, c] is explained using ONLY train-set
        # samples assigned (via argmax) to that cluster c (the same
        # population already used above for cluster_purity/nearest_medoid_prototypes).
        if not args.skip_cluster_gradients:
            print("\n" + "=" * 78)
            print(f"Per-cluster Captum attributions ({len(active_clusters)} active clusters, "
                  f"up to {args.cluster_gradient_examples} samples each)")
            print("=" * 78)
            clusters_dir = gradients_dir / 'clusters'
            for c in active_clusters:
                member_indices, _ = purity[c]
                selected = member_indices[:args.cluster_gradient_examples]
                x_cluster = torch.from_numpy(raw_series_np[selected]).float().to(model.device)
                cluster_dir = clusters_dir / f'cluster_{c}'
                print(f"\ncluster {c}: {x_cluster.shape[0]} samples (of {member_indices.size} total members)")
                run_gradient_explanations(
                    model, x_cluster, c, args.gradient_methods,
                    cluster_dir,
                    ig_steps=args.ig_steps, gradientshap_samples=args.gradientshap_samples,
                    occlusion_max_cells=args.occlusion_max_cells, internal_batch_size=args.internal_batch_size,
                    num_examples=x_cluster.shape[0], channel_names=channel_names,
                )

                # channel_importance_all_cluster.png / time_importance_all_cluster.png:
                # averaged over ALL members of the cluster (not just the
                # --cluster-gradient-examples used above for the per-example plots).
                x_cluster_all = torch.from_numpy(raw_series_np[member_indices]).float()
                channel_means_c, time_means_c, n_used_c = compute_mean_importance_profiles_from_tensor(
                    model, x_cluster_all, c, args.gradient_methods, args.ig_steps,
                    args.gradientshap_samples, args.occlusion_max_cells, args.internal_batch_size,
                    chunk_size=args.gradient_max_samples,
                )
                plot_channel_importance_aggregate(channel_means_c, n_used_c, cluster_dir / 'channel_importance_all_cluster.png', channel_names)
                plot_time_importance_aggregate(time_means_c, n_used_c, cluster_dir / 'time_importance_all_cluster.png')
                print(f"  channel/time_importance_all_cluster.png computed over all {n_used_c} members of cluster {c}.")

    # Shared across the TimeSHAP and --with-fidelity sections below: avoids
    # recomputing intrinsic/Captum IG or a TimeSHAP KernelSHAP run for the
    # exact same (example, target_cluster[, seed]) more than once when the
    # two sections' example sets overlap (see get_intrinsic_and_captum_cached/
    # get_timeshap_event_profile_cached). Defined unconditionally so it's
    # available regardless of which of --with-timeshap/--with-fidelity ran.
    profile_cache = {'intrinsic': {}, 'event': {}}

    if WITH_TIMESHAP:
        torch.manual_seed(args.seed)
        print("\n" + "=" * 78)
        print("TimeSHAP (event/feature/cell-level) on predict_membership, with a cross-method "
              "comparison against w_{b,t,c} and Captum IntegratedGradients")
        print("=" * 78)
        timeshap_dir = output_dir / 'timeshap'
        timeshap_dir.mkdir(parents=True, exist_ok=True)
        feature_names = channel_names

        # GLOBAL (outside clusters/): ONLY summary (aggregate) plots, on
        # generic test-set samples, fixed target-cluster (--target-cluster,
        # the same one already used for the global Captum gradients).
        n_global = x_test.shape[0] if args.timeshap_global_avg_samples <= 0 \
            else min(args.timeshap_global_avg_samples, x_test.shape[0])
        x_global = x_test[:n_global]
        global_subtitle = f'random test data, target cluster {args.target_cluster}'
        print(f"\nGLOBAL: averaging over {n_global} generic test-set samples "
              f"{'(the whole test set)' if args.timeshap_global_avg_samples <= 0 else ''}, "
              f"target cluster = {args.target_cluster}")
        feat_avg_global, event_avg_global, shap_matrix_global, feat_values_global, n_used_global = \
            compute_feature_event_profiles(
                model, x_global, args.target_cluster, args.avg_event_nsamples, args.avg_feature_nsamples, args.seed,
                channel_names=feature_names,
            )
        plot_shap_feature_impact_single(
            feat_avg_global, feature_names, n_used_global,
            timeshap_dir / 'timeshap_feature_random_data.png', subtitle=global_subtitle,
        )
        plot_shap_event_impact(
            event_avg_global, n_used_global, timeshap_dir / 'timeshap_event_random_data.png', subtitle=global_subtitle,
        )
        plot_shap_violin(shap_matrix_global, feature_names, timeshap_dir / 'timeshap_violin_random_data.png',
                          subtitle=global_subtitle)
        plot_shap_beeswarm(shap_matrix_global, feat_values_global, feature_names,
                            timeshap_dir / 'timeshap_beeswarm_random_data.png', subtitle=global_subtitle)
        plot_shap_scatter(shap_matrix_global, feat_values_global, feature_names,
                           timeshap_dir / 'timeshap_scatter_random_data.png')
        mean_signed_global = shap_matrix_global.mean(axis=0)
        base_value_global = compute_base_value(model, args.target_cluster, x_global.shape[1], x_global.shape[2])
        plot_shap_force(mean_signed_global, feature_names, base_value_global,
                         base_value_global + mean_signed_global.sum(), timeshap_dir / 'force_plot_random_data.png')
        plot_shap_waterfall(mean_signed_global, feature_names, base_value_global,
                             base_value_global + mean_signed_global.sum(),
                             timeshap_dir / 'waterfall_plot_random_data.png')
        w_btc_avg_global, ig_avg_global = compute_mean_intrinsic_captum(
            model, x_global, args.target_cluster, args.ig_steps,
        )
        plot_cross_method_comparison_arrays(
            w_btc_avg_global, ig_avg_global, event_avg_global,
            timeshap_dir / 'cross_method_comparison_random_data.png',
            subtitle=f'mean over {n_used_global} samples, {global_subtitle}',
        )
        print(f"  'random data' plots saved (n={n_used_global})")

        # Combined per-channel importance across ALL channel-capable methods
        # (Captum gradient methods + TimeSHAP feature level) in one figure.
        # Requires the --with-captum section above to have computed
        # channel_means over the test set for the same target cluster.
        if args.with_captum:
            plot_channel_importance_combined(
                channel_means, feat_avg_global,
                timeshap_dir / 'channel_importance_combined.png', feature_names,
            )
            print("  channel_importance_combined.png saved (Captum + TimeSHAP, normalized)")

        # PER CLUSTER: reuses raw_series_np/pred_labels_np/purity/active_clusters
        # already computed above for cluster_purity/Captum gradients.
        if args.skip_cluster_timeshap:
            print("\n--skip-cluster-timeshap: skipping the per-cluster TimeSHAP analysis.")
        else:
            avg_desc = 'all members' if args.timeshap_cluster_avg_samples <= 0 \
                else f'up to {args.timeshap_cluster_avg_samples} members'
            print(f"\nPER-CLUSTER: {len(active_clusters)} active clusters, up to "
                  f"{args.timeshap_cluster_examples} cell-level examples + average over {avg_desc}, each.")
            clusters_dir = timeshap_dir / 'clusters'
            cluster_feat_avgs = {}
            cluster_signed_avgs = {}
            cluster_base_values = {}
            cluster_cross_method_profiles = {}
            for c in active_clusters:
                member_indices, _ = purity[c]
                cluster_dir = clusters_dir / f'cluster_{c}'
                print(f"\ncluster {c}: {member_indices.size} total members")

                n_examples = min(args.timeshap_cluster_examples, member_indices.size)
                for i in range(n_examples):
                    idx = int(member_indices[i])
                    x_single = torch.from_numpy(raw_series_np[idx:idx + 1]).float().to(model.device)
                    explain_one_timeshap_example(
                        model, x_single, c, cluster_dir, args, label=f'cluster {c} example {i}',
                        cache=profile_cache, idx=idx, channel_names=feature_names,
                    )
                    src = cluster_dir / 'timeshap_cell.png'
                    if src.exists():
                        src.rename(cluster_dir / f'timeshap_cell_example{i}.png')

                n_avg = member_indices.size if args.timeshap_cluster_avg_samples <= 0 \
                    else min(args.timeshap_cluster_avg_samples, member_indices.size)
                x_cluster_avg = torch.from_numpy(raw_series_np[member_indices[:n_avg]]).float().to(model.device)
                cluster_subtitle = f'cluster {c}'
                feat_avg_c, event_avg_c, shap_matrix_c, feat_values_c, n_used_c = compute_feature_event_profiles(
                    model, x_cluster_avg, c, args.avg_event_nsamples, args.avg_feature_nsamples, args.seed,
                    channel_names=feature_names,
                )
                cluster_feat_avgs[c] = feat_avg_c
                plot_shap_feature_impact_single(
                    feat_avg_c, feature_names, n_used_c, cluster_dir / 'timeshap_feature_all_cluster.png',
                    subtitle=cluster_subtitle,
                )
                plot_shap_event_impact(
                    event_avg_c, n_used_c, cluster_dir / 'timeshap_event_all_cluster.png', subtitle=cluster_subtitle,
                )
                plot_shap_violin(shap_matrix_c, feature_names, cluster_dir / 'timeshap_violin_all_cluster.png',
                                  subtitle=cluster_subtitle)
                plot_shap_beeswarm(shap_matrix_c, feat_values_c, feature_names,
                                    cluster_dir / 'timeshap_beeswarm_all_cluster.png', subtitle=cluster_subtitle)
                plot_shap_scatter(shap_matrix_c, feat_values_c, feature_names,
                                   cluster_dir / 'timeshap_scatter_all_cluster.png')

                mean_signed_c = shap_matrix_c.mean(axis=0)
                base_value_c = compute_base_value(model, c, x_cluster_avg.shape[1], x_cluster_avg.shape[2])
                cluster_signed_avgs[c] = mean_signed_c
                cluster_base_values[c] = base_value_c
                plot_shap_force(mean_signed_c, feature_names, base_value_c,
                                 base_value_c + mean_signed_c.sum(), cluster_dir / 'force_plot_all_cluster.png')
                plot_shap_waterfall(mean_signed_c, feature_names, base_value_c,
                                     base_value_c + mean_signed_c.sum(), cluster_dir / 'waterfall_plot_all_cluster.png')
                w_btc_avg_c, ig_avg_c = compute_mean_intrinsic_captum(model, x_cluster_avg, c, args.ig_steps)
                cluster_cross_method_profiles[c] = (w_btc_avg_c, ig_avg_c, event_avg_c)
                plot_cross_method_comparison_arrays(
                    w_btc_avg_c, ig_avg_c, event_avg_c, cluster_dir / 'cross_method_comparison_all_cluster.png',
                    subtitle=f'mean over {n_used_c} samples, {cluster_subtitle}',
                )
                print(f"  cluster {c}: saved to {cluster_dir}")

            if cluster_feat_avgs:
                plot_shap_feature_impact_all_classes(
                    cluster_feat_avgs, feature_names, timeshap_dir / 'timeshap_feature_all_classes.png',
                )
                plot_shap_force_all_classes(
                    cluster_signed_avgs, feature_names, cluster_base_values,
                    timeshap_dir / 'force_plot_all_classes.png',
                )
                plot_shap_waterfall_all_classes(
                    cluster_signed_avgs, feature_names, cluster_base_values,
                    timeshap_dir / 'waterfall_plot_all_classes.png',
                )
                plot_cross_method_comparison_all_classes(
                    cluster_cross_method_profiles, timeshap_dir / 'cross_method_comparison_all_classes.png',
                )
                print(f"\nfeature/force/waterfall/cross_method_all_classes.png saved to {timeshap_dir} "
                      f"({len(cluster_feat_avgs)} clusters)")

    if args.with_fidelity:
        print("\n" + "=" * 78)
        print("Fidelity (insertion/deletion AUC) and stability (correlation across seeds) for all "
              "available methods")
        print("=" * 78)
        if not WITH_TIMESHAP:
            print("  --with-timeshap not active: TimeSHAP excluded from the table, only intrinsic "
                  "w_{b,t,c} and Captum IntegratedGradients.")
        fidelity_dir = output_dir / 'fidelity'
        fidelity_dir.mkdir(parents=True, exist_ok=True)

        # deletion/insertion AUC: averaged over --fidelity-examples real examples for each
        # active cluster (each example explained on its OWN predicted cluster c, as in the rest
        # of the per-cluster analysis). Besides the flat per-method lists (for the final table),
        # the raw curves are also kept (for the actual deletion/insertion plot) and the
        # per-cluster breakdown (for the cross-cluster homogeneity plot).
        method_del_aucs, method_ins_aucs = {}, {}
        method_del_curves, method_ins_curves = {}, {}
        cluster_method_del_aucs, cluster_method_ins_aucs = {}, {}
        n_examples_used = 0
        for c in active_clusters:
            member_indices, _ = purity[c]
            n_ex = min(args.fidelity_examples, member_indices.size)
            for i in range(n_ex):
                idx = int(member_indices[i])
                x_single = torch.from_numpy(raw_series_np[idx:idx + 1]).float().to(model.device)
                profiles = compute_method_profiles_for_example(model, x_single, c, args, cache=profile_cache, idx=idx)
                for method, profile in profiles.items():
                    profile_filled = np.nan_to_num(profile, nan=0.0)
                    del_scores, ins_scores = compute_deletion_insertion_curves(
                        model, x_single, c, profile_filled, args.internal_batch_size,
                    )
                    del_auc, ins_auc = _curve_auc(del_scores), _curve_auc(ins_scores)
                    method_del_aucs.setdefault(method, []).append(del_auc)
                    method_ins_aucs.setdefault(method, []).append(ins_auc)
                    method_del_curves.setdefault(method, []).append(del_scores)
                    method_ins_curves.setdefault(method, []).append(ins_scores)
                    cluster_method_del_aucs.setdefault(c, {}).setdefault(method, []).append(del_auc)
                    cluster_method_ins_aucs.setdefault(c, {}).setdefault(method, []).append(ins_auc)
                n_examples_used += 1
        print(f"  fidelity computed over {n_examples_used} real examples ({len(active_clusters)} "
              f"active clusters, up to {args.fidelity_examples} each)")

        # stability: a single representative example (first member of the first active cluster),
        # recomputing the TimeSHAP event-level profile with --fidelity-stability-seeds different
        # seeds. stability_seeds[0] == args.seed, which is normally already cached (this same
        # example was just explained with --seed above, in this same fidelity loop and/or in the
        # per-cluster TimeSHAP detail) -- see get_timeshap_event_profile_cached.
        stability_by_method = {}
        if active_clusters and WITH_TIMESHAP:
            rep_cluster = active_clusters[0]
            rep_idx = int(purity[rep_cluster][0][0])
            x_rep = torch.from_numpy(raw_series_np[rep_idx:rep_idx + 1]).float().to(model.device)
            stability_seeds = [args.seed + i for i in range(args.fidelity_stability_seeds)]
            stability_by_method, stability_profiles = compute_stability(
                model, x_rep, rep_cluster, args, stability_seeds, cache=profile_cache['event'], idx=rep_idx,
            )
            print(f"  stability (TimeSHAP, {len(stability_seeds)} different seeds on the representative "
                  f"example of cluster {rep_cluster}): {stability_by_method}")
            if stability_profiles:
                plot_stability_profiles(stability_profiles, stability_seeds, fidelity_dir / 'stability_profiles.png')

        rows, rows_full = [], []
        for method in method_del_aucs:
            del_list, ins_list = method_del_aucs[method], method_ins_aucs[method]
            del_auc, ins_auc = float(np.mean(del_list)), float(np.mean(ins_list))
            if method == 'TimeSHAP event-level':
                stab = stability_by_method.get(method)
            else:
                # deterministic in this implementation (fixed baseline/n_steps, no
                # sampling): its stability is trivially 1.0, not a fabricated value.
                stab = 1.0
            rows.append((method, del_auc, ins_auc, stab))
            rows_full.append((method, del_list, ins_list, stab))
            print(f"    {method}: deletion AUC={del_auc:.3f}, insertion AUC={ins_auc:.3f}, "
                  f"stability={'n/a' if stab is None else f'{stab:.3f}'}")

        plot_fidelity_stability_table(rows, fidelity_dir / 'fidelity_stability_table.png')
        plot_fidelity_curves(method_del_curves, 'deletion', fidelity_dir / 'deletion_curve.png')
        plot_fidelity_curves(method_ins_curves, 'insertion', fidelity_dir / 'insertion_curve.png')
        plot_fidelity_auc_distribution(method_del_aucs, method_ins_aucs, fidelity_dir / 'fidelity_auc_distribution.png')
        plot_fidelity_per_cluster(cluster_method_del_aucs, cluster_method_ins_aucs, fidelity_dir / 'fidelity_per_cluster.png')
        print(f"  fidelity_stability_table.png/.tex, deletion/insertion_curve.png, "
              f"fidelity_auc_distribution.png, fidelity_per_cluster.png saved to {fidelity_dir}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()
