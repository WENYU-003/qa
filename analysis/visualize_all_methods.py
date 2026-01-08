from __future__ import annotations

import argparse
import json
import os
import pickle
from glob import glob
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

try:
    import umap
except ImportError as e:
    raise ImportError("Please install umap-learn: pip install umap-learn") from e

from src.clustering import AutoClusterer

SUPPORTED_METHODS = ("kmeans", "hierarchical", "hdbscan")


def load_cluster_pkl(cluster_dir: str) -> Dict[str, Any]:
    pkl_path = os.path.join(cluster_dir, "cluster_data.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Cannot find {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{pkl_path} does not contain a dict.")
    return data


def load_embeddings(cluster_dir: str) -> np.ndarray:
    data = load_cluster_pkl(cluster_dir)
    emb = np.asarray(data.get("embeddings"))
    if emb.ndim != 2:
        raise ValueError(f"Invalid embeddings shape: {emb.shape}")
    return emb


def load_documents_auto(cluster_dir: str) -> List[dict]:
    data = load_cluster_pkl(cluster_dir)
    docs = data.get("documents")
    if isinstance(docs, list) and (len(docs) == 0 or isinstance(docs[0], dict)):
        return docs

    doc_json = os.path.join("data", "documents", "documents.json")
    if os.path.exists(doc_json):
        with open(doc_json, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list) and (len(obj) == 0 or isinstance(obj[0], dict)):
            return obj

    for fp in glob(os.path.join("data", "documents", "*.json")):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict):
                return obj
        except Exception:
            pass

    return []


def build_dummy_documents(n: int) -> List[dict]:
    return [{"id": i, "title": f"doc_{i}", "content": ""} for i in range(n)]


def _compact_label(cid: int, cluster_obj: dict, max_keywords: int = 2) -> str:
    name = cluster_obj.get("name", "") or cluster_obj.get("title", "") or ""
    keywords = cluster_obj.get("keywords", None)

    kw_str = ""
    if isinstance(keywords, list) and len(keywords) > 0:
        kw_str = " • ".join([str(x) for x in keywords[:max_keywords]])

    if name and kw_str:
        return f"{cid}: {name} • {kw_str}"
    if name:
        return f"{cid}: {name}"
    if kw_str:
        return f"{cid}: {kw_str}"
    return f"{cid}: cluster {cid}"


def run_labels_and_names(
    method: str,
    embeddings: np.ndarray,
    documents: List[dict],
    base_config: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[int, str]]:
    cfg = dict(base_config)
    cfg["clustering.algorithm"] = method

    clusterer = AutoClusterer(cfg)

    if not documents or len(documents) != embeddings.shape[0]:
        documents = build_dummy_documents(embeddings.shape[0])

    result = clusterer.fit(documents=documents, embeddings=embeddings)

    labels = result.get("labels", None)
    if labels is None:
        labels = getattr(clusterer, "cluster_labels", None)
    if labels is None:
        raise RuntimeError(f"Failed to get labels for method={method}")
    labels = np.asarray(labels)

    label_text: Dict[int, str] = {}
    clusters = result.get("clusters", []) or []
    if isinstance(clusters, list):
        for c in clusters:
            if not isinstance(c, dict):
                continue
            cid = c.get("id", None)
            if cid is None:
                continue
            try:
                cid_int = int(cid)
            except Exception:
                continue
            label_text[cid_int] = _compact_label(cid_int, c, max_keywords=2)

    return labels, label_text


def _stack_labels_vertical(
    label_positions: List[Tuple[int, float, float]],
    min_y_gap: float,
) -> Dict[int, Tuple[float, float]]:
    """
    Resolve vertical overlaps by pushing labels downward when too close in y.
    label_positions: list of (cid, x, y_initial)
    Strategy:
      - sort by y descending (top to bottom)
      - if current y is within min_y_gap of previous label y, push it down
    Returns: cid -> (x, y_adjusted)
    """
    # sort by y (descending)
    sorted_pos = sorted(label_positions, key=lambda t: t[2], reverse=True)
    out: Dict[int, Tuple[float, float]] = {}

    prev_y = None
    for cid, x, y in sorted_pos:
        y_adj = y
        if prev_y is not None and (prev_y - y_adj) < min_y_gap:
            # push down so that gap is respected
            y_adj = prev_y - min_y_gap
        out[cid] = (x, y_adj)
        prev_y = y_adj

    return out


def plot_umap(
    coords: np.ndarray,
    labels: np.ndarray,
    title: str,
    out_path: str,
    label_text_map: Optional[Dict[int, str]] = None,
    noise_label: int = -1,
    point_size: int = 26,
    label_fontsize: int = 12,
    label_offset_frac: float = 0.025,   # default downward offset (fraction of y-range)
    min_label_gap_frac: float = 0.035,  # min vertical gap between labels (fraction of y-range)
) -> None:
    x = coords[:, 0]
    y = coords[:, 1]
    labels = np.asarray(labels)

    plt.figure(figsize=(12, 8))

    # draw points first
    mask_noise = labels == noise_label
    if np.any(mask_noise):
        plt.scatter(x[mask_noise], y[mask_noise], s=point_size, alpha=0.35, label="noise (-1)")

    mask_cluster = ~mask_noise
    unique = np.unique(labels[mask_cluster]) if np.any(mask_cluster) else np.array([])

    for cid in unique:
        m = labels == cid
        plt.scatter(x[m], y[m], s=point_size, alpha=0.85, label=f"cluster {int(cid)}")

    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")

    # y-range for offsets in data units
    y_min, y_max = float(np.min(y)), float(np.max(y))
    y_range = max(1e-9, (y_max - y_min))
    y_offset = y_range * float(label_offset_frac)
    min_y_gap = y_range * float(min_label_gap_frac)

    if label_text_map is None:
        label_text_map = {}

    # initial label positions: centroid then shift DOWN
    label_positions: List[Tuple[int, float, float]] = []
    for cid in unique:
        m = labels == cid
        cx = float(np.mean(x[m]))
        cy = float(np.mean(y[m])) - y_offset
        label_positions.append((int(cid), cx, cy))

    # resolve overlaps vertically (push down when necessary)
    adjusted = _stack_labels_vertical(label_positions, min_y_gap=min_y_gap)

    # draw labels
    for cid in unique:
        cid = int(cid)
        tx, ty = adjusted[cid]
        text = label_text_map.get(cid, f"{cid}: cluster {cid}")

        plt.text(
            tx, ty, text,
            fontsize=label_fontsize,
            fontweight="bold",
            ha="center",
            va="top",
        )

    # legend only if not too many
    n_clusters = int(len(np.unique(labels[labels != noise_label])))
    if n_clusters <= 12:
        plt.legend(markerscale=1.2, fontsize=9, loc="upper right")
    else:
        leg = plt.legend()
        if leg:
            leg.remove()

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_dir", required=True, help="Directory containing cluster_data.pkl")
    parser.add_argument("--out_dir", default="data/clusters", help="Directory to save png files")

    # UMAP config
    parser.add_argument("--umap_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--umap_metric", type=str, default="cosine")
    parser.add_argument("--random_state", type=int, default=42)

    # clustering knobs
    parser.add_argument("--min_cluster_size", type=int, default=3)
    parser.add_argument("--max_clusters", type=int, default=20)

    # plot knobs
    parser.add_argument("--label_offset_frac", type=float, default=0.025,
                        help="Label downward offset as a fraction of UMAP y-range (e.g., 0.025).")
    parser.add_argument("--min_label_gap_frac", type=float, default=0.035,
                        help="Minimum vertical gap between labels as a fraction of UMAP y-range (e.g., 0.035).")
    parser.add_argument("--label_fontsize", type=int, default=12)
    parser.add_argument("--point_size", type=int, default=26)

    args = parser.parse_args()

    embeddings = load_embeddings(args.cluster_dir)
    documents = load_documents_auto(args.cluster_dir)

    reducer = umap.UMAP(
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        random_state=args.random_state,
    )
    coords = reducer.fit_transform(embeddings)

    base_config = {
        "clustering.min_cluster_size": args.min_cluster_size,
        "clustering.max_clusters": args.max_clusters,
    }

    for method in SUPPORTED_METHODS:
        labels, label_text_map = run_labels_and_names(method, embeddings, documents, base_config)
        out_path = os.path.join(args.out_dir, f"umap_{method}.png")
        title = f"UMAP visualization of {method.upper()} clusters"
        plot_umap(
            coords,
            labels,
            title,
            out_path,
            label_text_map=label_text_map,
            point_size=args.point_size,
            label_fontsize=args.label_fontsize,
            label_offset_frac=args.label_offset_frac,
            min_label_gap_frac=args.min_label_gap_frac,
        )

    print(f"[OK] Saved UMAP plots to: {args.out_dir} (umap_<method>.png)")


if __name__ == "__main__":
    main()
