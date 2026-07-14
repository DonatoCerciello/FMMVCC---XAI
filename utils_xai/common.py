"""Dataset loading, model building, channel names, and shared matplotlib setup."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

import datautils
from fmmvcc import FMMVCC_Model
from main import select_dataset, WRIGEST_MACRO_CATEGORIES

plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'

# Real sensor channel names for WriGest (3-axis accelerometer + 3-axis
# gyroscope + heart rate) .
PATH = Path(__file__).resolve().parent.parent

WRIGEST_CHANNEL_NAMES = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ', 'HR']


def get_channel_names(dataset_name, D):
    if dataset_name.startswith('WriGest') and D == len(WRIGEST_CHANNEL_NAMES):
        return list(WRIGEST_CHANNEL_NAMES)
    return [f'ch{d}' for d in range(D)]


def check_captum_dependency():
    import captum
    print(f"captum: successfully imported from this interpreter (version {captum.__version__}).")
    dockerfile = PATH / 'dockerimg' / 'Dockerfile'
    if dockerfile.is_file() and 'captum' in dockerfile.read_text():
        print(f"  Found as a pip dependency in {dockerfile.relative_to(PATH)}.")
    else:
        print("  WARNING: captum is not listed in dockerimg/Dockerfile.")
    requirements_files = list(PATH.glob('**/requirements*.txt'))
    if requirements_files:
        print(f"  requirements* files found in the project: {[str(p.relative_to(PATH)) for p in requirements_files]}")
    else:
        print("  No dedicated requirements.txt found in this XAI project "
              "(only mamba/pyproject.toml, which lists mamba_ssm's own "
              "dependencies, not this project's). If you create one, add 'captum' there.")



def load_dataset(args):
    data_path = PATH / 'tsc_datasets' / 'extracted'
    is_multivariate = args.dataset_type == 'multivariate'
    data_path = data_path / (
        'Multivariate2018_arff/Multivariate_arff' if is_multivariate
        else 'Univariate2018_arff/Univariate_arff'
    )

    file_list = sorted([p.name for p in data_path.iterdir() if p.is_dir()])
    selected_file, _ = select_dataset(
        file_list, dataset_name=args.dataset_name, dataset_position=args.dataset_position
    )
    full_path = data_path / selected_file

    dimensions = sum(
        1 for f in full_path.iterdir() if f.suffix == '.arff' and 'Dimension' in f.name
    ) // 2

    train, test = {}, {}
    for dimension in range(1, dimensions + 1):
        train_file = full_path / f"{selected_file}Dimension{dimension}_TRAIN.arff"
        test_file = full_path / f"{selected_file}Dimension{dimension}_TEST.arff"
        if not train_file.exists() or not test_file.exists():
            continue
        train_local, _ = arff.loadarff(train_file)
        test_local, _ = arff.loadarff(test_file)
        train[dimension] = pd.DataFrame(train_local).fillna(0)
        test[dimension] = pd.DataFrame(test_local).fillna(0)

    train_dataset = np.array([])
    test_dataset = np.array([])
    y_train = None
    y_test = None
    for dimension in range(1, dimensions + 1):
        if dimension not in train or dimension not in test:
            continue
        train_dim = train[dimension].add_suffix(f"_dim{dimension}")
        test_dim = test[dimension].add_suffix(f"_dim{dimension}")
        if y_train is None:
            y_train = train_dim.iloc[:, -1]
        if y_test is None:
            y_test = test_dim.iloc[:, -1]
        train_dataset = (
            np.dstack((train_dataset, train_dim.iloc[:, :-1].values))
            if train_dataset.size else train_dim.iloc[:, :-1].values
        )
        test_dataset = (
            np.dstack((test_dataset, test_dim.iloc[:, :-1].values))
            if test_dataset.size else test_dim.iloc[:, :-1].values
        )

    if is_multivariate and train_dataset.dtype == object:
        X_train = np.array([np.asarray(row, dtype=np.float64) for row in train_dataset], dtype=np.float64)
        X_test = np.array([np.asarray(row, dtype=np.float64) for row in test_dataset], dtype=np.float64)
    else:
        X_train = train_dataset.astype(np.float64)
        X_test = test_dataset.astype(np.float64)

    y_train = np.array(y_train).astype(str)
    y_test = np.array(y_test).astype(str)

    # WriGest-only macro-category filter, mirroring main.py's
    save_name = selected_file
    if getattr(args, 'wrigest_macro_category', None) is not None:
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
                f"(expected classes: {sorted(macro_classes)})."
            )
        X_train, X_test = X_train[train_mask], X_test[test_mask]
        y_train, y_test = y_train[train_mask], y_test[test_mask]
        save_name = f"{selected_file}_{args.wrigest_macro_category}Gesture"
        print(
            f"\nWriGest macro-category filter: '{args.wrigest_macro_category}' "
            f"({sorted(macro_classes)})\n"
            f"  train: {train_mask.sum()}/{len(train_mask)} samples kept, "
            f"test: {test_mask.sum()}/{len(test_mask)} samples kept\n"
            f"  loading/saving under dataset name: {save_name}"
        )

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    n_clusters = len(set(y_train_encoded))

    scaler = TimeSeriesScalerMeanVariance()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return save_name, X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, n_clusters


def resolve_checkpoint_paths(dataset_name, mode, num_views, sep_weight, bal_weight):
    dataset_dir = (
        PATH / f'launches_{mode}' / dataset_name if mode != 'unidirectional'
        else PATH / 'launches' / dataset_name
    )
    suffix = f"NViews{num_views}_Sep{sep_weight}_Bal{bal_weight}.pt"
    finetune_path = dataset_dir / f"Finetuning_phase_{suffix}"
    centers_path = dataset_dir / f"Centers_{suffix}"
    return finetune_path, centers_path


def build_model(args, dataset_name, X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, n_clusters):
    train_index = np.arange(X_train_scaled.shape[0])
    test_index = np.arange(X_test_scaled.shape[0])
    train_loader, test_loader = datautils.create_data_loader(
        X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded,
        train_index, test_index, args.batch_size,
    )

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    model = FMMVCC_Model(
        data_loader=train_loader,
        dataset_size=X_train_scaled.shape[0],
        timesteps_len=X_train_scaled.shape[1],
        batch_size=args.batch_size,
        pretraining_epoch=args.pretraining_epoch,
        n_cluster=n_clusters,
        dataset_name=dataset_name,
        input_dims=X_train_scaled.shape[2],
        output_dims=args.output_dims,
        hidden_dims=args.hidden_dims,
        n_layers=args.n_layers,
        m=args.m,
        num_views=args.num_views,
        separation_weight=args.separation_weight,
        balance_weight=args.balance_weight,
        device=device,
        mode=args.mode,
    )
    return model, train_loader, test_loader


