# analysis/visualize_clusters.py
# Prettier + more "compact" UMAP plot for HDBSCAN clusters
from pathlib import Path
import pickle

import numpy as np
import matplotlib.pyplot as plt

# Optional: UMAP (recommended). If not installed, fallback to t-SNE.
try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

from sklearn.manifold import TSNE


def load_cluster_data(cluster_dir: Path):
    pkl_path = cluster_dir / "cluster_data.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Not found: {pkl_path}. Run `python example.py` to generate it."
        )

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    labels = np.asarray(data["labels"])
    embeddings = np.asarray(data["embeddings"])
    return embeddings, labels


def reduce_to_2d(X: np.ndarray):
    """
    Make the plot less 'spread out' / less 'empty' by:
    - using smaller min_dist (tighter clusters)
    - moderate n_neighbors (keep local structure)
    """
    if HAS_UMAP:
        reducer = umap.UMAP(
            n_neighbors=10,     # smaller -> more local, often looks tighter
            min_dist=0.02,      # smaller -> clusters look more compact
            spread=0.9,         # slightly lower spread can reduce emptiness
            metric="cosine",    # embeddings often work better with cosine
            random_state=42
        )
        Z = reducer.fit_transform(X)
        method = "UMAP"
    else:
        # t-SNE typically looks compact by default
        reducer = TSNE(
            n_components=2,
            perplexity=min(30, max(5, (len(X) - 1) // 3)),
            random_state=42,
            init="pca",
            learning_rate="auto"
        )
        Z = reducer.fit_transform(X)
        method = "t-SNE"
    return Z, method


def main():
    root = Path(__file__).resolve().parents[1]     # qa_github/
    cluster_dir = root / "data" / "clusters"

    X, labels = load_cluster_data(cluster_dir)
    Z, method = reduce_to_2d(X)

    # ---- Pretty plotting ----
    plt.figure(figsize=(9, 6))

    noise_mask = labels == -1
    cluster_ids = [cid for cid in np.unique(labels) if cid != -1]
    k = len(cluster_ids)

    # 1) Draw noise in background (light, transparent)
    if noise_mask.any():
        plt.scatter(
            Z[noise_mask, 0], Z[noise_mask, 1],
            s=14, alpha=0.12, c="#9aa0a6", linewidths=0, label="noise"
        )

    # 2) Categorical colormap for clusters
    cmap = plt.get_cmap("tab20" if k <= 20 else "nipy_spectral")

    # 3) Draw clusters (no per-cluster legend; it destroys aesthetics)
    for i, cid in enumerate(cluster_ids):
        idx = labels == cid
        color = cmap(i % cmap.N)

        plt.scatter(
            Z[idx, 0], Z[idx, 1],
            s=28, alpha=0.82, c=[color], linewidths=0
        )

        # 4) Mark cluster "center" (mean of 2D projection) + label
        cx, cy = Z[idx, 0].mean(), Z[idx, 1].mean()
        plt.scatter(
            cx, cy,
            s=160, marker="X", c=[color],
            edgecolors="black", linewidths=0.7, zorder=10
        )
        plt.text(
            cx, cy, str(cid),
            fontsize=10, weight="bold",
            ha="center", va="center", zorder=11
        )

    # 5) Styling: remove axes for clean look
    plt.title(f"{method} visualization of HDBSCAN clusters", fontsize=14)
    plt.xticks([])
    plt.yticks([])
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    # Small legend only for noise (optional)
    if noise_mask.any():
        plt.legend(loc="upper left", frameon=False)

    plt.tight_layout()

    out_png = cluster_dir / f"cluster_{method.lower()}_pretty.png"
    plt.savefig(out_png, dpi=320, bbox_inches="tight")
    print(f"Saved plot to: {out_png}")


if __name__ == "__main__":
    main()
