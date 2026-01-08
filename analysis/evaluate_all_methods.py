# analysis/evaluate_all_methods.py
"""
Evaluate clustering quality for all supported methods (kmeans/hierarchical/hdbscan)
in ONE run, and write results into a single TXT file.

Assumptions:
- You already have embeddings saved from any previous clustering run:
  <cluster_dir>/cluster_data.pkl  (created by AutoClusterer.save)
  which contains: {'labels': ..., 'embeddings': ...}

Usage example:
  python analysis/evaluate_all_methods.py \
      --cluster_dir data/clusters/hdbscan \
      --out data/clusters/evaluation_all_methods.txt

Notes:
- For HDBSCAN, label -1 is treated as noise and excluded from metric computation.
- Metrics are set to "N/A" when not mathematically valid (e.g., <2 clusters).
"""

from __future__ import annotations

import argparse
import os
import pickle
from datetime import datetime
from typing import Dict, Any, Tuple

import numpy as np

try:
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
except ImportError as e:
    raise ImportError("scikit-learn is required to compute clustering metrics.") from e

# Import your project's clusterer
# This assumes clustering.py is importable (e.g., at project root or in PYTHONPATH)
#from clustering import AutoClusterer
from src.clustering import AutoClusterer



SUPPORTED_METHODS = ("kmeans", "hierarchical", "hdbscan")


def load_embeddings_from_cluster_dir(cluster_dir: str) -> np.ndarray:
    pkl_path = os.path.join(cluster_dir, "cluster_data.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(
            f"Cannot find {pkl_path}. "
            f"Please pass a cluster_dir that contains cluster_data.pkl (created by AutoClusterer.save)."
        )
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if "embeddings" not in data:
        raise KeyError(f"{pkl_path} does not contain 'embeddings' key.")
    emb = np.asarray(data["embeddings"])
    if emb.ndim != 2 or emb.shape[0] < 2:
        raise ValueError(f"Invalid embeddings shape: {emb.shape}. Expect (n_samples, n_dims).")
    return emb


def build_dummy_documents(n: int) -> list[Dict[str, Any]]:
    # AutoClusterer.fit needs documents, even if embeddings are provided.
    # Keep them minimal.
    return [{"id": i, "title": f"doc_{i}", "content": ""} for i in range(n)]


def safe_metrics(X: np.ndarray, labels: np.ndarray, noise_label: int = -1) -> Dict[str, Any]:
    labels = np.asarray(labels)

    # remove noise for metric computation (important for HDBSCAN)
    mask = labels != noise_label
    X2 = X[mask]
    y2 = labels[mask]

    # Basic counts
    unique = np.unique(y2) if y2.size > 0 else np.array([])
    n_samples = int(X2.shape[0])
    n_clusters = int(unique.size)

    # cluster sizes
    sizes = {}
    if y2.size > 0:
        for cid in unique:
            sizes[int(cid)] = int(np.sum(y2 == cid))

    # Decide if metrics are valid
    # Silhouette requires at least 2 clusters and all clusters should have >=2 samples ideally.
    def _valid_for_pairwise() -> bool:
        if n_clusters < 2 or n_samples < 3:
            return False
        # If any cluster has 1 sample, silhouette can error depending on sklearn version
        if any(sz < 2 for sz in sizes.values()):
            return False
        return True

    def _valid_for_db_ch() -> bool:
        # DB and CH both require >=2 clusters and n_samples > n_clusters
        if n_clusters < 2:
            return False
        if n_samples <= n_clusters:
            return False
        return True

    out: Dict[str, Any] = {
        "n_samples_used": n_samples,
        "n_clusters": n_clusters,
        "noise_points": int(np.sum(~mask)),
        "cluster_sizes": sizes,
        "silhouette": "N/A",
        "davies_bouldin": "N/A",
        "calinski_harabasz": "N/A",
    }

    # Compute metrics with guards
    if _valid_for_pairwise():
        try:
            out["silhouette"] = float(silhouette_score(X2, y2, metric="euclidean"))
        except Exception:
            out["silhouette"] = "N/A"

    if _valid_for_db_ch():
        try:
            out["davies_bouldin"] = float(davies_bouldin_score(X2, y2))
        except Exception:
            out["davies_bouldin"] = "N/A"
        try:
            out["calinski_harabasz"] = float(calinski_harabasz_score(X2, y2))
        except Exception:
            out["calinski_harabasz"] = "N/A"

    return out


def run_one_method(method: str, embeddings: np.ndarray, base_config: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    # Make a config dict that AutoClusterer understands (it uses config.get('clustering.xxx'))
    cfg = dict(base_config) if base_config else {}
    cfg["clustering.algorithm"] = method

    clusterer = AutoClusterer(cfg)
    docs = build_dummy_documents(embeddings.shape[0])

    result = clusterer.fit(documents=docs, embeddings=embeddings)
    labels = np.asarray(result.get("labels", clusterer.cluster_labels))

    metrics = safe_metrics(embeddings, labels, noise_label=-1)
    return labels, metrics


def format_value(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def write_report(out_path: str, metrics_by_method: Dict[str, Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("Clustering Evaluation Summary (All Methods)")
    lines.append(f"Generated at: {now}")
    lines.append("Methods: " + ", ".join(SUPPORTED_METHODS))
    lines.append("-" * 70)

    for method in SUPPORTED_METHODS:
        m = metrics_by_method[method]
        lines.append(f"\n[{method.upper()}]")
        lines.append(f"n_samples_used: {m['n_samples_used']}")
        lines.append(f"noise_points:  {m['noise_points']}  (only meaningful for HDBSCAN)")
        lines.append(f"n_clusters:    {m['n_clusters']}")
        lines.append("cluster_sizes: " + (str(m["cluster_sizes"]) if m["cluster_sizes"] else "{}"))
        lines.append("")
        lines.append("silhouette:        " + format_value(m["silhouette"]))
        lines.append("davies_bouldin:    " + format_value(m["davies_bouldin"]))
        lines.append("calinski_harabasz: " + format_value(m["calinski_harabasz"]))
        lines.append("-" * 70)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_dir", required=True, help="A directory containing cluster_data.pkl (with embeddings).")
    parser.add_argument("--out", default="evaluation_all_methods.txt", help="Output txt file path.")
    parser.add_argument("--min_cluster_size", type=int, default=3, help="Used by HDBSCAN effective min cluster size.")
    parser.add_argument("--max_clusters", type=int, default=20, help="Upper bound used by kmeans/hierarchical.")
    args = parser.parse_args()

    embeddings = load_embeddings_from_cluster_dir(args.cluster_dir)

    # Minimal base config; you can extend if your project expects more keys
    base_config = {
        "clustering.min_cluster_size": args.min_cluster_size,
        "clustering.max_clusters": args.max_clusters,
        # Naming/dedup not needed for metrics; keep defaults from AutoClusterer
    }

    metrics_by_method = {}
    for method in SUPPORTED_METHODS:
        _, metrics = run_one_method(method, embeddings, base_config)
        metrics_by_method[method] = metrics

    write_report(args.out, metrics_by_method)
    print(f"[OK] Wrote evaluation summary to: {args.out}")


if __name__ == "__main__":
    main()
