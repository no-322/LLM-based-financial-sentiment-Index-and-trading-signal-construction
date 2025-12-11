"""
Semantic deduplication pipeline.

Reads 'news_data.csv' (expects columns: date_time, ticker, headline, source, summary).
Produces 'news_deduped.csv' with one row per semantic cluster and aggregated fields:
- cluster_id
- canonical_headline
- repeat_count
- source_list (semicolon-separated)
- timestamp_first_seen
- timestamp_last_seen
- members: optionally the indices or original headlines (not required)

Algorithm:
1. Normalize headlines using src.cleaning.text_preprocess.normalize_headline
2. Generate embeddings via src.utils.embeddings.get_embeddings (all-MiniLM-L6-v2)
3. AgglomerativeClustering(metric='cosine', linkage='average', distance_threshold=0.25)
4. Optional merge of borderline cluster pairs (centroid cosine similarity between 0.85 and 0.90)
   using local LLM verification if available (controlled by environment variable VERIFY_MODEL).
   If local LLM is not configured, falls back to stricter thresholding.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from text_preprocess import normalize_dataframe_headlines, normalize_headline
from embeddings import get_embeddings

# Configurable parameters (tuneable)
DISTANCE_THRESHOLD = float(os.getenv("DEDUP_DISTANCE_THRESHOLD", 0.25))  # ~0.2-0.3 recommended
BORDERLINE_LOW = float(os.getenv("DEDUP_BORDERLINE_LOW", 0.85))   # cosine similarity lower bound for borderline
BORDERLINE_HIGH = float(os.getenv("DEDUP_BORDERLINE_HIGH", 0.90))  # upper bound
VERIFY_MODEL = os.getenv("VERIFY_MODEL", "")  # optional local HF model name for verification

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = Path(os.getenv("DEDUP_OUTPUT", BASE_DIR / "data" / "news_deduped.csv"))
INPUT_PATH = Path(os.getenv("DEDUP_INPUT", BASE_DIR / "data" / "alpha_news_all_tickers.csv"))  # path to news CSV


# -------------------------
# Optional local LLM verifier
# -------------------------
def verify_same_event_with_local_llm(text_a: str, text_b: str) -> bool:
    """
    Optional: use a local (free) HF model to verify if two headlines refer to the same event.
    This function will attempt to run a small local model if VERIFY_MODEL env var is set.
    If VERIFY_MODEL is not set or the model cannot be loaded, this falls back to a
    conservative heuristic using cosine similarity.

    Returns True if the verifier judges the two texts as the same event; False otherwise.
    """
    # Simple heuristic fallback: if normalized edit distance or token overlap is high, say True
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    except Exception:
        # transformers not available -> fallback heuristic
        return False

    model_name = VERIFY_MODEL.strip()
    if not model_name:
        return False

    try:
        # Use a text-classification or seq2seq zero-shot approach if model supports it
        # We'll use a simple prompt for small models. Implementation depends on model capability.
        # WARNING: This block will only work if a compatible model is present locally and transformers can load it.
        prompt = (
            "Question: Do the following two headlines describe the same real-world event? "
            "Answer 'yes' or 'no'.\n\n"
            f"Headline A: {text_a}\n\nHeadline B: {text_b}\n\nAnswer:"
        )

        # Try sequence-to-sequence models first (safer for local small models)
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_new_tokens=16)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
        if answer.startswith('yes'):
            return True
        if answer.startswith('no'):
            return False
        # fallback: interpret presence of 'yes' substring
        return 'yes' in answer
    except Exception:
        # If anything fails, return False to avoid accidental merges
        return False


# -------------------------
# Core deduplication pipeline
# -------------------------
def cluster_embeddings(emb_matrix: np.ndarray, distance_threshold: float = DISTANCE_THRESHOLD) -> np.ndarray:
    """
    Cluster the normalized embeddings using Agglomerative Clustering (cosine metric).
    Returns cluster labels (length = N).
    """
    if emb_matrix.shape[0] == 0:
        return np.array([], dtype=int)

    # AgglomerativeClustering with metric='cosine' (sklearn will compute 1 - cosine for distance)
    # distance_threshold: clusters merged until this threshold reached (0..2 for cosine distance)
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric='cosine',
        linkage='average'
    )
    labels = model.fit_predict(emb_matrix)
    return labels


def merge_borderline_clusters(df: pd.DataFrame, emb_matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Optionally merge clusters if their centroids are very similar (cosine similarity in borderline range).
    Uses optional LLM verification (verify_same_event_with_local_llm) if configured.
    Returns updated labels (possibly with cluster ids merged).
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1:
        return labels

    # compute centroid for each cluster (average of embeddings)
    centroids = []
    label_to_idx = {}
    for lbl in unique_labels:
        idxs = np.where(labels == lbl)[0]
        label_to_idx[lbl] = idxs
        centroid = emb_matrix[idxs].mean(axis=0)
        # normalize
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroids.append(centroid)
    centroids = np.vstack(centroids)  # (K, D)

    # pairwise cosine similarities between cluster centroids
    sim = cosine_similarity(centroids)  # (K, K)

    # union-find to merge cluster labels when verified
    parent = {int(l): int(l) for l in unique_labels}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # mapping from index in centroids array -> actual label
    idx_to_label = {i: int(lbl) for i, lbl in enumerate(unique_labels)}

    # iterate upper triangle
    K = len(unique_labels)
    for i in range(K):
        for j in range(i + 1, K):
            sim_ij = float(sim[i, j])
            if BORDERLINE_LOW <= sim_ij <= BORDERLINE_HIGH:
                # borderline case: attempt verification
                # pick representative headlines (shortest normalized headline in each cluster)
                idxs_i = label_to_idx[idx_to_label[i]]
                idxs_j = label_to_idx[idx_to_label[j]]
                # representative texts
                rep_i = df.iloc[idxs_i]['headline'].astype(str).tolist()
                rep_j = df.iloc[idxs_j]['headline'].astype(str).tolist()
                # choose short headlines for verifier
                cand_a = min(rep_i, key=lambda x: len(str(x)))
                cand_b = min(rep_j, key=lambda x: len(str(x)))

                verified = verify_same_event_with_local_llm(cand_a, cand_b)
                if verified:
                    union(idx_to_label[i], idx_to_label[j])
            elif sim_ij > BORDERLINE_HIGH:
                # very high similarity -> merge without verification
                union(idx_to_label[i], idx_to_label[j])
            # else: similarity too low -> do not merge

    # build mapping from old label -> new merged label
    merged_map = {}
    for lbl in unique_labels:
        merged_map[int(lbl)] = find(int(lbl))

    # apply merged_map to labels array
    new_labels = np.array([merged_map[int(l)] for l in labels], dtype=int)
    return new_labels


def build_cluster_rows(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    Build per-occurrence rows with cumulative counts and sources.

    Output columns:
    - cluster_id (int)
    - ticker (str)
    - clean_headline (str)
    - timestamps (datetime as string)
    - first_seen (datetime)
    - last_seen (datetime)
    - repeat_count (int) cumulative count up to this timestamp
    - source_list (list[str]) cumulative unique sources up to this timestamp
    """
    df2 = df.copy()
    df2["cluster_id"] = labels
    df2["date_time_dt"] = pd.to_datetime(df2["date_time"], errors="coerce")

    rows = []
    for cid, group in df2.groupby("cluster_id"):
        group = group.dropna(subset=["date_time_dt"])
        if group.empty:
            continue
        ticker_mode = group["ticker"].mode()
        ticker_val = ticker_mode.iloc[0] if not ticker_mode.empty else ""
        clean_headline = min(group["headline_norm"].astype(str).tolist(), key=len)
        first_seen = group["date_time_dt"].min()
        last_seen = group["date_time_dt"].max()

        # walk chronologically and accumulate unique sources seen so far
        running_sources = set()
        group_sorted = group.sort_values("date_time_dt")
        for ts, ts_group in group_sorted.groupby("date_time_dt"):
            running_sources.update(ts_group["source"].dropna().astype(str).tolist())
            cum_count = (group_sorted["date_time_dt"] <= ts).sum()
            rows.append(
                {
                    "cluster_id": int(cid),
                    "ticker": ticker_val,
                    "clean_headline": clean_headline,
                    "timestamps": pd.Timestamp(ts),
                    "first_seen": first_seen,
                    "last_seen": last_seen,
                    "repeat_count": cum_count,
                    "source_list": json.dumps(sorted(running_sources)),
                }
            )

    out = pd.DataFrame(rows)
    out = out.sort_values(["cluster_id", "timestamps"]).reset_index(drop=True)
    return out


def run_dedup_pipeline(
    input_path: Union[str, Path] = INPUT_PATH,
    output_path: Union[str, Path] = OUTPUT_PATH,
    distance_threshold: float = DISTANCE_THRESHOLD
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute the deduplication pipeline end-to-end.

    Returns:
      (df_raw, df_clusters) where df_raw is the original data with cluster labels,
      df_clusters is the aggregated cluster-level DataFrame written to disk.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    print(f"Loading input CSV from: {input_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found at {input_path}")
    df = pd.read_csv(input_path)
    # ensure required columns exist
    expected_cols = {'date_time', 'ticker', 'headline', 'source', 'summary'}
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(f"Input CSV must contain columns: {expected_cols}")

    if df.empty:
        print("Input CSV is empty; skipping deduplication.")
        return df, pd.DataFrame(
            columns=[
                "cluster_id",
                "canonical_headline",
                "repeat_count",
                "source_list",
                "timestamp_first_seen",
                "timestamp_last_seen",
                "members_count",
            ]
        )

    # normalize headlines
    print("Normalizing headlines...")
    df = normalize_dataframe_headlines(df, headline_col='headline', inplace=False)

    # use normalized headline for embedding (but keep original headline for canonical selection)
    texts = df['headline_norm'].fillna('').astype(str).tolist()

    # get embeddings
    print("Generating embeddings (this may take a moment)...")
    emb_matrix = get_embeddings(texts)

    # clustering
    print(f"Clustering embeddings with distance_threshold={distance_threshold} (cosine metric)...")
    labels = cluster_embeddings(emb_matrix, distance_threshold=distance_threshold)

    # optional merging of borderline clusters
    print("Merging borderline clusters (if any)...")
    labels = merge_borderline_clusters(df, emb_matrix, labels)

    # attach labels to raw df
    df['cluster_id'] = labels

    # aggregate cluster-level info
    print("Aggregating clusters with cumulative counts...")
    df_clusters = build_cluster_rows(df, labels)

    # save outputs
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing aggregated clusters to {output_path}")
    df_clusters.to_csv(output_path, index=False)

    # also write raw with cluster ids for debugging
    raw_out = output_path.with_suffix('').as_posix() + '_members.csv'
    print(f"Writing raw member mapping to {raw_out}")
    df.to_csv(raw_out, index=False)

    print("Deduplication pipeline complete.")
    return df, df_clusters


if __name__ == "__main__":
    # CLI entrypoint for quick runs
    df_raw, df_clusters = run_dedup_pipeline()
