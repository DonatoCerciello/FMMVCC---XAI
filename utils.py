import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
import json
import torch
from collections import Counter
from pathlib import Path
from scipy.optimize import linear_sum_assignment

PATH = Path(__file__).parent.absolute()
plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

def plot_latent_space(
        projections,
        labels,
        model_name,
        save_title="",
        plot_root=Path(PATH / "plot")
    ):
    # -----------------------------
    output_dir = os.path.join(plot_root, model_name)
    os.makedirs(output_dir, exist_ok=True)
    if not isinstance(labels, list):
        labels = [labels]
    if not isinstance(save_title, list):
        save_title = [save_title]

    for y, title in zip(labels, save_title):
        y = np.asarray(y)
        classes = np.unique(y)
        cmap = plt.get_cmap("viridis", len(classes))
        norm = matplotlib.colors.BoundaryNorm(
            np.arange(len(classes) + 1) - 0.5,
            len(classes)
        )
        for method_name, data_2d in projections.items():
            if method_name.lower() == "tsne":
                xlabel, ylabel = "t-SNE 1", "t-SNE 2"
            elif method_name.lower() == "umap" and title in ["Test Data", "Test Data Pred Adj"]:
                xlabel, ylabel = "UMAP 1", "UMAP 2"
            elif method_name.lower() == "pca" and title in ["Test Data", "Test Data Pred Adj"]:
                xlabel, ylabel = "Component 1", "Component 2"
            else:
                continue

            fig, ax = plt.subplots(figsize=(10, 8))

            scatter = ax.scatter(
                data_2d[:, 0],
                data_2d[:, 1],
                c=y,
                cmap=cmap,
                norm=norm,
                s=100,
                edgecolors="black",
                linewidths=1,
                alpha=0.9
            )

            ax.set_xlabel(xlabel, fontweight="bold", fontsize=18)
            ax.set_ylabel(ylabel, fontweight="bold", fontsize=18)

            ax.tick_params(width=2, length=6)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            for spine in ax.spines.values():
                spine.set_linewidth(1.8)

            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_fontweight("bold")
                tick.set_fontsize(16)

            cbar = fig.colorbar(
                scatter,
                ax=ax,
                ticks=classes,
                fraction=0.05,
                pad=0.03
            )

            cbar.set_label(
                "Class",
                fontsize=18,
                fontweight="bold"
            )

            cbar.ax.tick_params(
                labelsize=16,
                width=1.8,
                length=5
            )

            for tick in cbar.ax.get_yticklabels():
                tick.set_fontweight("bold")

            cbar.outline.set_linewidth(2)

            plt.tight_layout(pad=0.2)

            filename = f"{title}_Latent_Space_{method_name.upper()}.png"

            fig.savefig(
                os.path.join(output_dir, filename),
                dpi=1200,
                bbox_inches="tight"
            )

            plt.close(fig)

def compute_latent_projections(u):
    u_scaled = StandardScaler().fit_transform(u)

    pca = PCA(n_components=2)
    u1_pca = pca.fit_transform(u_scaled)

    perplexity = min(30, len(u_scaled) - 1)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=200,
        max_iter=1000,
        random_state=42
    )

    umap_model = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.0,
        metric="euclidean",
        random_state=42
    )

    return {
        "pca": u1_pca,
        "tsne": tsne.fit_transform(u_scaled),
        "umap": umap_model.fit_transform(u_scaled)
    }

def build_label_mapping(y_true, y_pred):
    mapping = {}
    for pred_label in np.unique(y_pred):
        true_vals = y_true[y_pred == pred_label]
        best_true = Counter(true_vals).most_common(1)[0][0]
        mapping[pred_label] = best_true
    return mapping

def apply_mapping(y_pred, mapping_train, mapping_test=None):
    aligned_pred = []
    unique_labels = set(mapping_train.values())

    for v in y_pred:
        if v in mapping_train:
            aligned_pred.append(mapping_train[v])
        elif mapping_test and v in mapping_test:
            aligned_pred.append(mapping_test[v])
        else:
            new_label = f"new_{len(unique_labels)}"
            aligned_pred.append(new_label)
            unique_labels.add(new_label)

    return np.array(aligned_pred)

def update_dataset_registry(
    json_path: Path,
    dataset_name: str,
    position: int,
    univariate: bool,
    n_clusters: int,
    train_shape: int,
    temporal_length: int,
    number_of_channels: int = 1
    ):
    """
    Update a JSON registry of processed datasets.

    Structure:
    {
        "univariate": {dataset_name: position, ...},
        "multivariate": {dataset_name: position, ...}
    }
    """

    key = "univariate" if univariate else "multivariate"

    # Initialize empty structure if file does not exist
    if json_path.exists():
        with open(json_path, "r") as f:
            registry = json.load(f)
    else:
        registry = {"univariate": {}, "multivariate": {}}

    # Safety check
    if key not in registry:
        registry[key] = {}

    # Add dataset only if not present
    if dataset_name not in registry[key]:
        registry[key][dataset_name] = {
            "position": position,
            "n_clusters": n_clusters,
            "train_shape": train_shape,
            "temporal_length": temporal_length,
            "number_of_channels": number_of_channels
        }

        # Sort datasets alphabetically
        registry[key] = dict(sorted(registry[key].items()))

        # Save back to file
        with open(json_path, "w") as f:
            json.dump(registry, f, indent=4)

        return True  # Added
    else:
        return False  # Already present

def estimate_seasonality_generic(X, max_period=None):
    """
    X: np.array shape (N, T, F)
    ritorna: periodo dominante globale o None
    """

    X = np.asarray(X)
    N, T, F = X.shape

    if max_period is None:
        max_period = T // 2

    all_ffts = []

    for f in range(F):
        ts = X[:, :, f].mean(axis=0)  # mean across series for feature f
        ts = ts - ts.mean()

        fft = np.abs(np.fft.rfft(ts))
        freqs = np.fft.rfftfreq(T, d=1)

        fft = fft[1:]
        freqs = freqs[1:]

        periods = 1 / freqs

        valid = (periods > 1) & (periods <= max_period)

        if np.any(valid):
            all_ffts.append((periods[valid], fft[valid]))

    if not all_ffts:
        return None

    # unisco contributi di tutte le feature
    all_periods = np.concatenate([p for p, _ in all_ffts])
    all_power = np.concatenate([f for _, f in all_ffts])

    best_period = int(round(all_periods[np.argmax(all_power)]))

    return best_period

def plot_mean_series_with_period(X, period):
    """
    X: np.array shape (N, T, 1)
    period: periodo stimato
    """
    X = np.asarray(X)
    mean_ts = X[:, :, 0].mean(axis=0)

    plt.figure(figsize=(15, 5))
    plt.plot(mean_ts, label='Mean series', color='blue')

    # linee verticali ad ogni periodo
    for i in range(period, len(mean_ts), period):
        plt.axvline(i, color='red', linestyle='--', alpha=0.5)

    plt.title(f'Mean Series with Estimated Period = {period}', fontsize=16)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

def encode_in_batches(model, X, batch_size=64):
    embeddings = []
    for i in range(0, len(X), batch_size):
        batch = torch.tensor(X[i:i+batch_size]).float().to(model.device)
        with torch.no_grad():
            z = model.encode_with_pooling(batch).detach().cpu()
        embeddings.append(z)
    return torch.cat(embeddings).numpy()


def encode_in_batches_single(model, X, batch_size=64):
    embeddings = []
    for i in range(0, len(X), batch_size):
        batch = torch.tensor(X[i:i+batch_size]).float().to(model.device)
        with torch.no_grad():
            z, _, _ = model.encoder(batch)
        embeddings.append(z.detach().cpu())
    return torch.cat(embeddings).numpy()


def hungarian_label_alignment(y_true, y_pred, n_clusters):
    # --- Hungarian label alignment ---
    w = np.zeros((n_clusters, n_clusters))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    # mapping cluster -> class
    mapping = dict(zip(row_ind, col_ind))

    # apply mapping
    label_pred_aligned = np.array([
        mapping.get(label, label) for label in y_pred
    ])
    return label_pred_aligned
