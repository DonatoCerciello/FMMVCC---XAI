"""Builds the WriGest dataset directly into the per-dimension ARFF format
used by the FMMVCC XAI pipeline's existing loader (main_xai.py/main.py),
from the raw per-sensor CSV files in 2_ActionData.
"""
from pathlib import Path

import glob
import re

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = (
    BASE_DIR.parent / "tsc_datasets" / "extracted" / "Multivariate2018_arff" /
    "Multivariate_arff" / "WriGest"
)

TARGET_LEN = 500
SEED = 42
TEST_FRACTION = 0.1
FEATURE_NAMES = ["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ", "HR"]
N_CHANNELS_MAP = {"Accelerometer": 3, "Gyroscope": 3, "HeartRatePPG": 1}


def find_raw_dir(base_dir):
    """Locates the 2_ActionData raw folder under base_dir: robust to
    whether WriGest.zip was extracted flat (WriGest/2_ActionData) or with
    an extra nested folder matching the zip's own internal root
    (WriGest/WriGest/2_ActionData, the case as of this fresh extraction)."""
    candidates = sorted(base_dir.glob("**/2_ActionData"))
    if not candidates:
        raise FileNotFoundError(
            f"No 2_ActionData folder found under {base_dir}. "
            "Extract WriGest.zip first (see unzip.py)."
        )
    return candidates[0]


def parse_csv_path(csv_path):
    """Parses a CSV filename into its components. Returns None for
    filenames that don't match the expected 5+-token pattern."""
    parts = csv_path.stem.split("_")
    if len(parts) < 5:
        return None
    gesture = "_".join(parts[2:-2])
    gesture = re.sub(r"\(\d+\)$", "", gesture)
    return {
        "sensor": parts[0],
        "session_id": parts[1],
        "gesture": gesture,
        "rate": parts[-2],
        "action": parts[-1],
    }


def read_sensor_csv(csv_path):
    """Reads a sensor CSV file without a header row.
    Format: session_id, sensor_type, timestamp, x, y, z (IMU)
    or: session_id, sensor_type, timestamp, value, ...padding zeros (PPG)."""
    try:
        return pd.read_csv(csv_path, header=None)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return pd.DataFrame()


def resample_sensor_data(df, target_len, sensor_type):
    """Resamples one trial's sensor stream to target_len timesteps (linear
    interpolation over normalized time) and per-trial z-scores it. Only the
    expected number of value columns is used per sensor (HeartRatePPG has
    many trailing always-zero padding columns in the raw files past the
    first one; only that first, genuinely varying column is kept)."""
    n_channels = N_CHANNELS_MAP.get(sensor_type, 1)
    if df.empty:
        return np.zeros((target_len, n_channels), dtype=np.float64)

    t = df.iloc[:, 2].to_numpy(dtype=float)
    values = df.iloc[:, 3:3 + n_channels].to_numpy(dtype=np.float64)
    if len(t) < 2 or values.shape[1] == 0:
        return np.zeros((target_len, n_channels), dtype=np.float64)

    t_norm = (t - t[0]) / max(t[-1] - t[0], 1e-9)
    t_target = np.linspace(0.0, 1.0, target_len)

    rs = np.zeros((target_len, n_channels), dtype=np.float64)
    for c in range(min(n_channels, values.shape[1])):
        rs[:, c] = np.interp(t_target, t_norm, values[:, c])

    mu = rs.mean(axis=0, keepdims=True)
    sd = rs.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (rs - mu) / sd


def scan_dataset(raw_dir):
    """Groups the per-sensor CSVs into trials keyed by (session_id, gesture,
    action) -- safe from cross-gesture collisions since gesture is now the
    FULL, corrected string (see module docstring)."""
    all_files = sorted(glob.glob(str(raw_dir / "**" / "*.csv"), recursive=True))
    if not all_files:
        raise ValueError(f"No CSV files found in {raw_dir}")
    print(f"Found {len(all_files)} CSV files under {raw_dir}")

    trial_groups = {}
    for csv_file in all_files:
        csv_path = Path(csv_file)
        parsed = parse_csv_path(csv_path)
        if parsed is None:
            continue
        key = (parsed["session_id"], parsed["gesture"], parsed["action"])
        trial_groups.setdefault(key, []).append(csv_path)

    print(f"Found {len(trial_groups)} distinct trials (after fixing the gesture-parsing bug)")
    return trial_groups


def build_tensor(trial_groups, target_len):
    """Builds the resampled+standardized tensor X[N,T,7] and the (now
    fine-grained, correct) per-trial gesture label array."""
    n = len(trial_groups)
    X = np.zeros((n, target_len, 7), dtype=np.float64)
    gesture_labels = np.empty(n, dtype=object)

    for i, (key, csv_paths) in enumerate(sorted(trial_groups.items())):
        _, gesture, _ = key
        acc_data = gyr_data = hr_data = None
        for csv_path in csv_paths:
            parsed = parse_csv_path(csv_path)
            df = read_sensor_csv(csv_path)
            if df.empty:
                continue
            if parsed["sensor"] == "Accelerometer":
                acc_data = df
            elif parsed["sensor"] == "Gyroscope":
                gyr_data = df
            elif parsed["sensor"] == "HeartRatePPG":
                hr_data = df

        acc_rs = resample_sensor_data(acc_data, target_len, "Accelerometer") if acc_data is not None else np.zeros((target_len, 3))
        gyr_rs = resample_sensor_data(gyr_data, target_len, "Gyroscope") if gyr_data is not None else np.zeros((target_len, 3))
        hr_rs = resample_sensor_data(hr_data, target_len, "HeartRatePPG") if hr_data is not None else np.zeros((target_len, 1))

        X[i] = np.concatenate([acc_rs, gyr_rs, hr_rs], axis=1)
        gesture_labels[i] = gesture

        if (i + 1) % 500 == 0:
            print(f"  Resampled {i + 1}/{n} trials")

    return X, np.array([str(g) for g in gesture_labels])


def stratified_train_test_split(labels, test_fraction=TEST_FRACTION, seed=SEED):
    """90/10 (default) split, independent PER CLASS: for each class, its
    indices are shuffled and test_fraction is taken as test, the rest as
    train. 
    Returns (train_idx, test_idx) as sorted arrays."""
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for cls in np.unique(labels):
        cls_idx = np.nonzero(labels == cls)[0].copy()
        rng.shuffle(cls_idx)
        n_test = max(1, round(len(cls_idx) * test_fraction))
        test_idx.extend(cls_idx[:n_test].tolist())
        train_idx.extend(cls_idx[n_test:].tolist())
    return np.sort(train_idx), np.sort(test_idx)


def write_arff_dimension(X_dim, labels, dimension_idx, class_names, split_name, output_dir):
    """Writes a single "flattened" ARFF file for one dimension/channel:
    one numeric attribute per timestep + a final nominal target attribute,
    the same format as the other UEA datasets already used by main_xai.py
    (e.g. AtrialFibrillationDimension1_TRAIN.arff)."""
    n, T = X_dim.shape
    lines = [f"@relation WriGest_channel_{dimension_idx}", ""]
    lines += [f"@attribute channel_{dimension_idx}_{t} numeric" for t in range(T)]
    lines.append(f"@attribute target {{{','.join(class_names)}}}")
    lines.append("")
    lines.append("@data")
    for i in range(n):
        values = ",".join(f"{v:.6f}" for v in X_dim[i])
        lines.append(f"{values},{labels[i]}")

    file_path = output_dir / f"WriGestDimension{dimension_idx + 1}_{split_name}.arff"
    file_path.write_text("\n".join(lines) + "\n")
    return file_path


def prepare_wrigest_arff(output_dir=OUTPUT_DIR, target_len=TARGET_LEN,
                          test_fraction=TEST_FRACTION, seed=SEED):
    """Main entry point: scans the raw CSVs, builds the per-trial tensor
    with the corrected fine-grained gesture labels, splits 90/10 per class,
    and writes the ARFF files -- the single output of this script."""
    print("=" * 70)
    print("WriGest -> ARFF (tsc_datasets) pipeline")
    print("=" * 70)

    raw_dir = find_raw_dir(BASE_DIR)
    trial_groups = scan_dataset(raw_dir)

    print("\nBuilding per-trial tensor (resample to "
          f"T={target_len}, per-trial z-score)...")
    X, labels = build_tensor(trial_groups, target_len)
    class_names = sorted(np.unique(labels).tolist())
    print(f"\nClasses ({len(class_names)}): {class_names}")

    train_idx, test_idx = stratified_train_test_split(labels, test_fraction, seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for d in range(X.shape[2]):
        written.append(
            write_arff_dimension(X[train_idx, :, d], labels[train_idx], d, class_names, "TRAIN", output_dir)
        )
        written.append(
            write_arff_dimension(X[test_idx, :, d], labels[test_idx], d, class_names, "TEST", output_dir)
        )

    print(f"\nARFF written to: {output_dir}")
    print(f"  channels ({X.shape[2]}): {FEATURE_NAMES}")
    print(f"  split: train={len(train_idx)} trials, test={len(test_idx)} trials "
          f"(90/10 per class, seed={seed})")
    for cls in class_names:
        n_tr = int((labels[train_idx] == cls).sum())
        n_te = int((labels[test_idx] == cls).sum())
        print(f"    {cls}: train={n_tr}, test={n_te}")
    print(f"  files written: {len(written)} ({X.shape[2]} channels x 2 splits)")
    return output_dir


if __name__ == "__main__":
    prepare_wrigest_arff()
