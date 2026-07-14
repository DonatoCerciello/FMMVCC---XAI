# %%
import argparse
from pathlib import Path
import pandas as pd
from scipy.io import arff
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from utils import (
    update_dataset_registry,
    estimate_seasonality_generic,
    plot_mean_series_with_period,
    encode_in_batches,
    plot_latent_space,
    hungarian_label_alignment,
    compute_latent_projections
)
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, rand_score, f1_score
from models.Metrics import acc as accuracy
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import contingency_matrix

PATH = Path(__file__).parent.absolute()

WRIGEST_MACRO_CATEGORIES = {
    'Directional': {'back', 'down', 'front', 'left', 'right', 'up'},
    'Number': {f'number_{d}' for d in range(10)},
    'Character': {
        'capital_letter_f', 'capital_letter_t', 'lower_letter_a', 'lower_letter_b',
        'lower_letter_c', 'lower_letter_d', 'lower_letter_e',
    },
    'Sign': {'check_mark', 'division_sign', 'minus_sign', 'multiplication_sign', 'plus_sign'},
    'Fine': {
        'gesture_ok', 'gesture_call', 'gesture_grip', 'gesture_release',
        'gesture_tapping', 'gesture_appreciation',
    },
    'Daily': {
        'daily_blow_nose', 'daily_comb_hair', 'daily_drink', 'daily_eating',
        'daily_smoking', 'daily_wash_hand',
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description='Run clustering pipeline for UCR/UEA datasets.')
    parser.add_argument(
        '--dataset-type',
        type=str,
        default='multivariate',
        choices=['univariate', 'multivariate'],
        help='Type of dataset to load.',
    )
    parser.add_argument(
        '--dataset-position',
        type=int,
        default=None,
        help='Dataset index in the sorted file list (used when --dataset-name is not provided).',
    )
    parser.add_argument(
        '--dataset-name',
        type=str,
        default=None,
        help='Dataset folder name. Overrides --dataset-position if provided.',
    )
    parser.add_argument(
        '--launch',
        type=str,
        default='FMMVCC',
        choices=['FMMVCC'],
        help='Training launch method (only FMMVCC is available in this copy: EMTC/FCACC were not ported to XAI/).',
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='unidirectional',
        choices=['unidirectional', 'bidirectional'],
        help='Encoder mode for FMMVCC.',
    )
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--output-dims', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--pretraining-epoch', type=int, default=100)
    parser.add_argument('--max-iter', type=int, default=100)
    parser.add_argument('--m', type=float, default=1.5)
    parser.add_argument('--skip-seasonality', action='store_true', help='Skip seasonality analysis and plotting.')
    parser.add_argument(
        '--wrigest-macro-category',
        type=str,
        default=None,
        choices=sorted(WRIGEST_MACRO_CATEGORIES.keys()),
        help='WriGest only: restrict train/test to just this macro category\'s fine-grained '
              'classes (e.g. Number = number_0..number_9), instead of all 40 classes across all '
              '6 categories. Requires --dataset-name WriGest. Checkpoints/plots/results are then '
              "saved under '<dataset-name>_<category>Gesture' (e.g. WriGest_NumberGesture) "
              'instead of plain WriGest, so a full-dataset run and a per-category run never '
              'overwrite each other.',
    )
    parser.add_argument(
        '--plot-root',
        type=Path,
        default=Path(PATH / 'plot'),
        help='Root directory where plots and summary metrics are saved.',
    )
    return parser.parse_args()

def select_dataset(file_list, dataset_name=None, dataset_position=None):
    if not file_list:
        raise ValueError('No datasets found in the configured data path.')

    if dataset_name:
        # Match dataset names in a case-insensitive way for convenience.
        dataset_map = {name.lower(): name for name in file_list}
        selected = dataset_map.get(dataset_name.lower())
        if selected is None:
            raise ValueError(
                f"Dataset '{dataset_name}' not found. Available dataset count: {len(file_list)}"
            )
        return selected, file_list.index(selected)

    # If no explicit position is provided, auto-select by available dataset names.
    if dataset_position is None:
        if len(file_list) == 1:
            return file_list[0], 0
        print(
            'No dataset position provided. Multiple datasets found; '
            f"using the first one alphabetically: '{file_list[0]}'."
        )
        dataset_position = 0

    if dataset_position < 0 or dataset_position >= len(file_list):
        print(
            f"dataset_position={dataset_position} is out of range [0, {len(file_list) - 1}]. "
            'Falling back to the first dataset.'
        )
        dataset_position = 0

    return file_list[dataset_position], dataset_position


def main():
    args = parse_args()

    data_path = Path(__file__).parent.absolute() / 'tsc_datasets' / 'extracted'

    # Support both univariate and multivariate datasets
    if hasattr(args, 'dataset_type') and args.dataset_type == 'multivariate':
        data_path = data_path / 'Multivariate2018_arff' / 'Multivariate_arff'
        is_multivariate = True
        print("Loading MULTIVARIATE datasets...")
    else:
        data_path = data_path / 'Univariate2018_arff' / 'Univariate_arff'
        is_multivariate = False
        print("Loading UNIVARIATE datasets...")

    file_list = sorted([
        p.name for p in data_path.iterdir() if p.is_dir()
    ])
    print(f"Found {len(file_list)} datasets")
    selected_file, file_position = select_dataset(
        file_list,
        dataset_name=args.dataset_name,
        dataset_position=args.dataset_position,
    )
    print("Selected file:", selected_file)

    full_path = data_path / selected_file
    print("Complete path of the selected file:", full_path)

    # Count dimensions by checking for files that match the expected naming pattern
    dimensions = 0
    for file in full_path.iterdir():
        # Files must end in arff and have "Dimension" in the name
        if file.suffix == '.arff' and 'Dimension' in file.name:
            dimensions += 1
    dimensions = dimensions // 2
    print(f"Detected dimensions in dataset: {dimensions}")

    # Load data from ARFF files
    train = {}
    test = {}
    for dimension in range(1, dimensions + 1):
        print(f"Checking for dimension {dimension} files...")

        train_file = full_path / f"{selected_file}Dimension{dimension}_TRAIN.arff"
        test_file = full_path / f"{selected_file}Dimension{dimension}_TEST.arff"

        if not train_file.exists() or not test_file.exists():
            print(f"Warning: Expected files for dimension {dimension} not found.")
            if not train_file.exists():
                print(f"Missing train file: {train_file}")
            if not test_file.exists():
                print(f"Missing test file: {test_file}")
            continue

        print(f"✓ Found files for dimension {dimension}")

        train_local, _ = arff.loadarff(train_file)
        test_local, _ = arff.loadarff(test_file)
        train[dimension] = pd.DataFrame(train_local).fillna(0)
        test[dimension] = pd.DataFrame(test_local).fillna(0)

    # Create the train and test datasets. Each dimension corresponds to a channel, rows are the number of time series, columns are the length of the time series.
    # Dimension --> (n_samples, series_length, n_channels)
    train_dataset = np.array([])
    test_dataset = np.array([])
    y_train = None
    y_test = None
    for dimension in range(1, dimensions + 1):
        if dimension not in train or dimension not in test:
            print(f"Skipping dimension {dimension} due to missing data.")
            continue

        # Rename columns to avoid conflicts
        train_dim = train[dimension].add_suffix(f"_dim{dimension}")
        test_dim = test[dimension].add_suffix(f"_dim{dimension}")
        if y_train is None:
            y_train = train_dim.iloc[:, -1]
        else:
            # Check if labels match across dimensions
            if not y_train.equals(train_dim.iloc[:, -1]):
                print(f"Warning: Label mismatch detected in dimension {dimension} for training data.")
                print(f"Unique labels in previous dimensions: {y_train.unique()}")
                print(f"Unique labels in current dimension: {train_dim.iloc[:, -1].unique()}")
                raise ValueError("Label mismatch across dimensions in training data.")
        if y_test is None:
            y_test = test_dim.iloc[:, -1]
        else:
            # Check if labels match across dimensions
            if not y_test.equals(test_dim.iloc[:, -1]):
                print(f"Warning: Label mismatch detected in dimension {dimension} for test data.")
                print(f"Unique labels in previous dimensions: {y_test.unique()}")
                print(f"Unique labels in current dimension: {test_dim.iloc[:, -1].unique()}")
                raise ValueError("Label mismatch across dimensions in test data.")

        # Merge the data to form (n_samples, series_length, n_channels)
        train_dataset = np.dstack((train_dataset, train_dim.iloc[:, :-1].values)) if train_dataset.size else train_dim.iloc[:, :-1].values
        test_dataset = np.dstack((test_dataset, test_dim.iloc[:, :-1].values)) if test_dataset.size else test_dim.iloc[:, :-1].values

    # For multivariate data: handle ARFF relational format
    if is_multivariate:
        print("\n[MULTIVARIATE] Processing features (relational ARFF format)...")
        # Check if features contain nested arrays/lists (relational format)
        if train_dataset.dtype == object:
            print("Converting relational nested arrays to numeric matrix...")
            try:
                # Each row is a nested array/list of values
                X_train = np.array([np.asarray(row, dtype=np.float64) for row in train_dataset], dtype=np.float64)
                X_test = np.array([np.asarray(row, dtype=np.float64) for row in test_dataset], dtype=np.float64)
            except (ValueError, TypeError) as e:
                print(f"Error during conversion: {e}")
                print(f"First sample type: {type(train_dataset[0])}, First sample: {train_dataset[0]}")
                raise
        else:
            X_train = train_dataset.astype(np.float64)
            X_test = test_dataset.astype(np.float64)
    else:
        X_train = train_dataset.astype(np.float64)
        X_test = test_dataset.astype(np.float64)

    # Encode labels
    y_train = np.array(y_train).astype(str)
    y_test = np.array(y_test).astype(str)

    # WriGest-only macro-category filter : restricts to just one semantically
    # homogeneous group of fine-grained classes instead of all 40 at once.
    save_name = selected_file
    if args.wrigest_macro_category is not None:
        if selected_file.lower() != 'wrigest':
            raise ValueError(
                f"--wrigest-macro-category is only meaningful for --dataset-name WriGest "
                f"(got dataset '{selected_file}')."
            )
        macro_classes = WRIGEST_MACRO_CATEGORIES[args.wrigest_macro_category]
        train_mask = np.isin(y_train, list(macro_classes))
        test_mask = np.isin(y_test, list(macro_classes))
        if not train_mask.any() or not test_mask.any():
            raise ValueError(
                f"No samples found for macro category '{args.wrigest_macro_category}' "
                f"(expected classes: {sorted(macro_classes)}). Check that the ARFF was "
                "regenerated with the fine-grained (post-fix) gesture labels."
            )
        X_train, X_test = X_train[train_mask], X_test[test_mask]
        y_train, y_test = y_train[train_mask], y_test[test_mask]
        save_name = f"{selected_file}_{args.wrigest_macro_category}Gesture"
        print(
            f"\nWriGest macro-category filter: '{args.wrigest_macro_category}' "
            f"({sorted(macro_classes)})\n"
            f"  train: {train_mask.sum()}/{len(train_mask)} samples kept, "
            f"test: {test_mask.sum()}/{len(test_mask)} samples kept\n"
            f"  saving under dataset name: {save_name}"
        )

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    # Count how many unique clusters are present
    unique_clusters = set(y_train_encoded)
    n_clusters = len(unique_clusters)
    print(f"Number of clusters: {n_clusters}")

    # Update dataset registry
    json_path = Path(__file__).parent.absolute() / "dataset_registry.json"
    update_dataset_registry(
        json_path=json_path,
        dataset_name=save_name,
        position=file_position,
        univariate=not is_multivariate,
        n_clusters=n_clusters,
        train_shape=X_train.shape[0],
        temporal_length=X_train.shape[1],
        number_of_channels=X_train.shape[2] if is_multivariate else 1,
    )

    # Normalize time series
    scaler = TimeSeriesScalerMeanVariance()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Setting metrics variables
    ari = None
    ri = None
    nmi = None
    f1 = None
    acc = None

    # Seasonality analysis
    print("\n" + "="*80)
    print("SEASONALITY ANALYSIS")
    print("="*80)
    if not args.skip_seasonality:
        print("Attempting to estimate seasonality...")
        seasonality_period = estimate_seasonality_generic(X_train_scaled)
        if seasonality_period is not None:
            print(f"✓ Estimated seasonality period: {seasonality_period}")
            plot_mean_series_with_period(X_train_scaled[:, :seasonality_period * 20], seasonality_period)
        else:
            print("✗ No significant seasonality detected.")
            if is_multivariate:
                print("  (This is expected for some multivariate datasets - using --skip-seasonality to proceed)")
            raise ValueError(
                'No significant seasonality detected. Use --skip-seasonality to bypass this step.'
            )
    else:
        print("Seasonality analysis skipped by user flag.")

    print("\n" + "="*80)
    print("DATA SUMMARY BEFORE MODEL")
    print("="*80)
    print(f"Training data shape: {X_train_scaled.shape}")
    print(f"Test data shape: {X_test_scaled.shape}")
    print(f"Training labels shape: {y_train_encoded.shape}")
    print(f"Test labels shape: {y_test_encoded.shape}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Is multivariate: {is_multivariate}\n")

    print("Using FMMVCC")
    from batch_run import run_FMMVCC
    sep_weight = 1.0
    bal_weight = 0.5
    N_view = 4

    run_label = 'FMMVCC' if args.mode == 'unidirectional' else f'FMMVCC_{args.mode}'
    config = {
        'batch_size': args.batch_size,
        'output_dims': args.output_dims,
        'lr': args.lr,
        'pretraining_epoch': args.pretraining_epoch,
        'MaxIter': args.max_iter,
        'm': args.m,
        'separation_weight': sep_weight,
        'balance_weight': bal_weight,
        'num_views': N_view
    }
    acc, nmi, ari, ri, fmi, f1, model = run_FMMVCC(
        X_train_scaled,
        X_test_scaled,
        y_train_encoded,
        y_test_encoded,
        save_name,
        config,
        args.mode,
    )

    print(f"Results: acc={acc}, nmi={nmi}, ari={ari}, ri={ri}, fmi={fmi}, f1={f1}")

    # Folder in which predictions are saved
    results_dir_name = 'results' if args.mode == 'unidirectional' else f'results_{args.mode}'
    results_folder = Path.cwd() / results_dir_name / save_name / 'label'

    # Verify files exist before reading
    y_true_path = results_folder / f"{save_name}_label_true.csv"
    y_pred_path = results_folder / f"{save_name}_label_pred.csv"
    if not y_true_path.exists() or not y_pred_path.exists():
        raise FileNotFoundError(
            f"Prediction files not found in {results_folder}."
        )

    y_pred_train = pd.read_csv(y_true_path)['label_true'].values
    y_pred_test = pd.read_csv(y_pred_path)['label_pred'].values
    y_pred_train = np.array(y_pred_train).astype(int)
    y_pred_test = np.array(y_pred_test).astype(int)
    y_pred_train = y_pred_train[:len(y_train_encoded)]
    y_pred_test = y_pred_test[:len(y_test_encoded)]

    label_pred_aligned = hungarian_label_alignment(y_test_encoded, y_pred_test, n_clusters)

    # Paths
    sep_weight = config.get('separation_weight', 0.5)
    bal_weight = config.get('balance_weight', 0.2)
    latent_plot_subdir = f"{run_label}/{save_name}/NViews{config.get('num_views', 4)}_Sep{sep_weight}_Bal{bal_weight}"
    launch_name = 'FMMVCC' if args.mode == 'unidirectional' else f'FMMVCC_{args.mode}'
    results_dirname = f"{save_name}/NViews{config.get('num_views', 4)}_Sep{sep_weight}_Bal{bal_weight}"

    u1 = encode_in_batches(model, X_train_scaled)
    u2 = encode_in_batches(model, X_test_scaled)

    # Plot latent space
    plot_latent_space(compute_latent_projections(u1), y_train_encoded, latent_plot_subdir, 'Training Data', args.plot_root)

    Dataset = f'Position {file_position} - Name {save_name} - Separation {sep_weight} - Balance {bal_weight}'

    # Test
    labels_test = [
        label_pred_aligned,
        y_pred_test,
        y_test_encoded
    ]
    titles_test = [
        'Test Data Pred Adj',
        'Test Data Pred',
        'Test Data'
    ]
    plot_latent_space(compute_latent_projections(u2), labels_test, latent_plot_subdir, titles_test, args.plot_root)

    # Metrics
    ari = adjusted_rand_score(y_test_encoded, y_pred_test) if ari is None else ari
    ri = rand_score(y_test_encoded, y_pred_test) if ri is None else ri
    nmi = normalized_mutual_info_score(y_test_encoded, y_pred_test) if nmi is None else nmi
    acc = accuracy(y_test_encoded, y_pred_test, n_clusters) if acc is None else acc
    f1 = f1_score(y_test_encoded, label_pred_aligned, average='macro') if f1 is None else f1

    results = {
        "ARI": ari,
        "RI": ri,
        "NMI": nmi,
        "F1": f1,
        "ACC": acc,
        "Data set": Dataset,
    }

    # Save results to a file in the plot folder
    results_path = args.plot_root / launch_name / results_dirname
    results_path.mkdir(parents=True, exist_ok=True)
    results_file = results_path / 'clustering_results.txt'
    with open(results_file, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")

    print("Finished clustering and plotting.")

# %%
if __name__ == '__main__':
    main()
