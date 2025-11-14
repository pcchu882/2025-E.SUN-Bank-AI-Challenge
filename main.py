"""
2025 玉山人工智慧公開挑戰賽 – 警示帳戶預測 (TEAM_9686)

This script implements the end-to-end pipeline for the E.SUN AI CUP 2025
"Alert Account Prediction" task:

1. Load raw CSVs: acct_transaction / acct_alert / acct_predict
2. Build account-level features from transaction records
3. Construct an account graph (nodes = accounts, edges = transactions)
4. Train a GraphSAGE-like GNN (SimpleSAGEv2) with class-imbalance handling
5. Calibrate predictions via threshold and Top-K F1 search on a validation split
6. Generate result.csv in the required format (acct, label)

Author: 褚柏均
"""

import os
import math
import random
from types import SimpleNamespace
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold


def set_seed(seed: int = 42):
    """
    Set random seeds for Python, NumPy and PyTorch to ensure reproducibility.

    Args:
        seed (int): Random seed value used for all RNGs.

    Notes:
        - Also sets CUDA seeds when a GPU is available.
        - This function is called once at the beginning of the script.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


def find_col(df: pd.DataFrame, cand: List[str]) -> Optional[str]:
    """
    Find a suitable column name in `df` given a list of candidate names.

    The function tries the following strategies in order:
        1. Exact match (case-sensitive)
        2. Exact match (case-insensitive)
        3. Fuzzy match: candidate string is contained in a column name (case-insensitive)

    Args:
        df (pd.DataFrame): Input DataFrame whose columns are to be searched.
        cand (List[str]): Candidate column names or patterns.

    Returns:
        Optional[str]: The selected column name in `df` if found, otherwise None.

    用途:
        用來對應不同版本資料中欄位命名可能略有差異的情況
        (例如 from_acct / payer_acct_id / src_acct 等)。
    """
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}
    # exact / case-insensitive
    for name in cand:
        if name in cols:
            return name
        ln = name.lower()
        if ln in lower:
            return lower[ln]
    # prefix / contains
    for name in cand:
        ln = name.lower()
        for c in cols:
            if ln in c.lower():
                return c
    return None


def load_csvs(dir_path: str):
    """
    Load the three main CSV files required by the competition.

    Expected filenames (under `dir_path`):
        - acct_transaction.csv
        - acct_alert.csv
        - acct_predict.csv

    Args:
        dir_path (str): Directory path that contains the three CSV files.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            (tx, alert, test) where
            - tx    : transaction records
            - alert : labeled alert accounts
            - test  : accounts to be predicted

    Raises:
        FileNotFoundError: If any of the expected files is missing.

    用途:
        比賽官方提供的三個核心檔案的讀取。
    """
    tx = pd.read_csv(os.path.join(dir_path, 'acct_transaction.csv'))
    alert = pd.read_csv(os.path.join(dir_path, 'acct_alert.csv'))
    test = pd.read_csv(os.path.join(dir_path, 'acct_predict.csv'))
    print("(Finish) Load Dataset.")
    return tx, alert, test


def resolve_columns(tx: pd.DataFrame, alert: pd.DataFrame, test: pd.DataFrame):
    """
    Resolve and normalize column names across different CSV files.

    This function searches for semantically equivalent columns under different
    naming conventions, e.g.:
        - from_acct / payer_acct_id / src_acct / source_account
        - to_acct / receiver_acct_id / dst_acct / target_account
        - txn_amt / amount / tx_amount / trade_amount
        - acct / acct_id / account_id / account

    Args:
        tx (pd.DataFrame): Transaction records dataframe.
        alert (pd.DataFrame): Alert accounts dataframe.
        test (pd.DataFrame): Accounts to be predicted dataframe.

    Returns:
        Dict[str, str]: A mapping from logical names to actual column names:
            {
                "src": str,
                "dst": str,
                "amt": str,
                "from_type": Optional[str],
                "to_type": Optional[str],
                "alert_acct": str,
                "alert_date": Optional[str],
                "predict_acct": str,
            }

    Raises:
        ValueError: If mandatory columns such as source/destination account
            or account ID in alert/predict files cannot be found.

    用途:
        讓後續程式都只接觸「標準化後」的欄位名稱，避免因欄位命名不同而壞掉。
    """
    # transaction columns (with common synonyms)
    src = find_col(tx, ["from_acct", "payer_acct_id", "src", "src_acct", "from", "acct_from", "source_account"])
    dst = find_col(tx, ["to_acct", "receiver_acct_id", "dst", "dst_acct", "to", "acct_to", "target_account"])
    amt = find_col(tx, ["txn_amt", "amount", "tx_amount", "amt", "money", "trade_amount"])
    from_type = find_col(tx, ["from_acct_type", "src_acct_type", "is_esun_from", "from_type"])
    to_type = find_col(tx, ["to_acct_type", "dst_acct_type", "is_esun_to", "to_type"])

    if src is None or dst is None:
        raise ValueError(f"Cannot find src/dst columns in acct_transaction.csv. Got src={src}, dst={dst}")

    if amt is None:
        # fall back to 1.0 if amount missing
        tx["_AMT_"] = 1.0
        amt = "_AMT_"

    # alert/predict columns
    a_acct = find_col(alert, ["acct", "acct_id", "account_id", "account"])
    a_date = find_col(alert, ["event_date", "alert_date", "date", "datetime", "month"])  # not essential here
    p_acct = find_col(test, ["acct", "acct_id", "account_id", "account"])

    if a_acct is None or p_acct is None:
        raise ValueError(f"Cannot find acct columns in alert/predict. alert_acct={a_acct}, predict_acct={p_acct}")

    return {
        "src": src, "dst": dst, "amt": amt,
        "from_type": from_type, "to_type": to_type,
        "alert_acct": a_acct, "alert_date": a_date,
        "predict_acct": p_acct
    }


def build_account_features_v2(
    tx: pd.DataFrame,
    col: Dict[str, str],
    windows=(7, 30, 90),
    add_heavy=False,
    clip_quantile=0.999
) -> pd.DataFrame:
    """
    Build rich account-level features from raw transaction logs (v2 version).

    This version extends the basic feature set with:
        - Temporal windows (e.g. 7 / 30 / 90 days) transaction stats
        - High-amount behavior (90 / 99 percentile counts & sums)
        - Recency features (days since last in/out transaction)
        - Optional "heavy" features such as entropy and reciprocity

    Args:
        tx (pd.DataFrame): Transaction records.
        col (Dict[str, str]): Column mapping produced by `resolve_columns`.
        windows (Tuple[int, ...]): Time windows (in days) for additional features.
        add_heavy (bool): Whether to compute heavy statistics like entropy,
            which may be more computationally expensive.
        clip_quantile (float or None): If not None, clip numeric features at
            the specified upper quantile to reduce the influence of outliers.

    Returns:
        pd.DataFrame:
            A dataframe where each row corresponds to one account, containing:
                - "acct" : account ID
                - "is_esun" : indicator for E.SUN accounts
                - various aggregated numeric features
                - log1p-transformed versions for selected numeric fields

    用途:
        產生更完整的帳戶特徵（時間窗 / 高金額 / 熵等），可切換 add_heavy 來平衡精度與效能。
    """
    src, dst, amt = col["src"], col["dst"], col["amt"]
    tx = tx[[src, dst, amt] + [c for c in [col.get("from_type"), col.get("to_type")] if c in tx.columns]].copy()

    # detect datetime column if available
    date_col = None
    for cand in ["txn_date", "tx_date", "date", "datetime", "event_time", "event_date", "trans_dt", "time"]:
        if cand in tx.columns:
            date_col = cand
            break
    if date_col is not None:
        tx["_dt"] = pd.to_datetime(tx[date_col], errors="coerce")
        max_t = tx["_dt"].max()
        tx["_days_ago"] = (max_t - tx["_dt"]).dt.days
    else:
        tx["_dt"] = pd.NaT
        tx["_days_ago"] = np.nan

    # 基礎聚合：流入/流出金額與筆數
    send = tx.groupby(src)[amt].sum().rename('total_send_amt')
    recv = tx.groupby(dst)[amt].sum().rename('total_recv_amt')
    max_send = tx.groupby(src)[amt].max().rename('max_send_amt')
    min_send = tx.groupby(src)[amt].min().rename('min_send_amt')
    avg_send = tx.groupby(src)[amt].mean().rename('avg_send_amt')
    max_recv = tx.groupby(dst)[amt].max().rename('max_recv_amt')
    min_recv = tx.groupby(dst)[amt].min().rename('min_recv_amt')
    avg_recv = tx.groupby(dst)[amt].mean().rename('avg_recv_amt')
    out_deg = tx.groupby(src)[dst].nunique().rename('out_deg')
    in_deg = tx.groupby(dst)[src].nunique().rename('in_deg')
    tx_cnt_out = tx.groupby(src)[dst].count().rename('out_tx_count')
    tx_cnt_in = tx.groupby(dst)[src].count().rename('in_tx_count')

    df_out = pd.concat([max_send, min_send, avg_send, send, out_deg, tx_cnt_out], axis=1)
    df_in = pd.concat([max_recv, min_recv, avg_recv, recv, in_deg, tx_cnt_in], axis=1)

    idx = pd.Index(df_out.index).union(df_in.index)
    feat = pd.DataFrame(index=idx).join(df_out, how="left").join(df_in, how="left").fillna(0.0)
    feat = feat.reset_index().rename(columns={"index": "acct"})

    # is_esun feature
    from_type, to_type = col.get("from_type"), col.get("to_type")
    if from_type is not None and from_type in tx.columns:
        df_from = tx[[src, from_type]].dropna().drop_duplicates().rename(columns={src: "acct", from_type: "is_esun"})
    else:
        df_from = pd.DataFrame(columns=["acct", "is_esun"])
    if to_type is not None and to_type in tx.columns:
        df_to = tx[[dst, to_type]].dropna().drop_duplicates().rename(columns={dst: "acct", to_type: "is_esun"})
    else:
        df_to = pd.DataFrame(columns=["acct", "is_esun"])
    df_acc = pd.concat([df_from, df_to], ignore_index=True).drop_duplicates()
    if not df_acc.empty:
        df_acc = df_acc.groupby("acct")["is_esun"].max().reset_index()
        feat = feat.merge(df_acc, on="acct", how="left")
        feat["is_esun"] = feat["is_esun"].fillna(1)
    else:
        feat["is_esun"] = 1

    # derived features
    eps = 1e-9
    feat["net_flow_amt"] = feat["total_send_amt"] - feat["total_recv_amt"]
    feat["flow_ratio"] = feat["net_flow_amt"] / (feat["total_send_amt"] + feat["total_recv_amt"] + eps)
    feat["send_per_outdeg"] = feat["total_send_amt"] / (feat["out_deg"] + eps)
    feat["recv_per_indeg"] = feat["total_recv_amt"] / (feat["in_deg"] + eps)
    feat["tx_per_outdeg"] = feat["out_tx_count"] / (feat["out_deg"] + eps)
    feat["tx_per_indeg"] = feat["in_tx_count"] / (feat["in_deg"] + eps)
    feat["mean_amt_overall"] = (feat["total_send_amt"] + feat["total_recv_amt"]) / (
        feat["out_tx_count"] + feat["in_tx_count"] + eps
    )

    # high-amount behaviors (90 / 99 percentile)
    q90 = tx[amt].quantile(0.90)
    q99 = tx[amt].quantile(0.99)
    high90_out = tx[tx[amt] >= q90].groupby(src)[amt].agg(['count', 'sum']).add_prefix('out_high90_')
    high90_in = tx[tx[amt] >= q90].groupby(dst)[amt].agg(['count', 'sum']).add_prefix('in_high90_')
    high99_out = tx[tx[amt] >= q99].groupby(src)[amt].agg(['count', 'sum']).add_prefix('out_high99_')
    high99_in = tx[tx[amt] >= q99].groupby(dst)[amt].agg(['count', 'sum']).add_prefix('in_high99_')
    for df in [high90_out, high90_in, high99_out, high99_in]:
        feat = feat.merge(df, left_on="acct", right_index=True, how="left")
    for c in [
        "out_high90_count", "out_high90_sum", "in_high90_count", "in_high90_sum",
        "out_high99_count", "out_high99_sum", "in_high99_count", "in_high99_sum"
    ]:
        if c not in feat.columns:
            feat[c] = 0.0

    # temporal window-based features and recency
    if date_col is not None and tx["_dt"].notna().any():
        for w in windows:
            mask = (tx["_days_ago"] >= 0) & (tx["_days_ago"] < w)
            txw = tx[mask]
            f_out = txw.groupby(src)[dst].count().rename(f'out_tx_count_{w}d')
            m_out = txw.groupby(src)[amt].sum().rename(f'out_amt_sum_{w}d')
            f_in = txw.groupby(dst)[src].count().rename(f'in_tx_count_{w}d')
            m_in = txw.groupby(dst)[amt].sum().rename(f'in_amt_sum_{w}d')
            for df in [f_out, m_out, f_in, m_in]:
                feat = feat.merge(df, left_on="acct", right_index=True, how="left")
            for c in [f'out_tx_count_{w}d', f'out_amt_sum_{w}d', f'in_tx_count_{w}d', f'in_amt_sum_{w}d']:
                feat[c] = feat[c].fillna(0.0)
        last_out = tx.groupby(src)["_days_ago"].min().rename("recency_out_days")
        last_in = tx.groupby(dst)["_days_ago"].min().rename("recency_in_days")
        feat = feat.merge(last_out, left_on="acct", right_index=True, how="left")
        feat = feat.merge(last_in, left_on="acct", right_index=True, how="left")
        feat["recency_out_days"] = feat["recency_out_days"].fillna(1e6)
        feat["recency_in_days"] = feat["recency_in_days"].fillna(1e6)
    else:
        for w in windows:
            for c in [f'out_tx_count_{w}d', f'out_amt_sum_{w}d', f'in_tx_count_{w}d', f'in_amt_sum_{w}d']:
                feat[c] = 0.0
        feat["recency_out_days"] = 1e6
        feat["recency_in_days"] = 1e6

    # optional heavy features (entropy / reciprocity)
    if add_heavy:
        # outgoing destination entropy
        pair_out = tx.groupby([src, dst]).size().rename("cnt")
        total_out = pair_out.groupby(level=0).sum()
        p = pair_out / total_out.loc[pair_out.index.get_level_values(0)].values
        ent_out = (-p * np.log(p + 1e-12)).groupby(level=0).sum().rename("out_dst_entropy")
        feat = feat.merge(ent_out, left_on="acct", right_index=True, how="left")

        # incoming source entropy
        pair_in = tx.groupby([dst, src]).size().rename("cnt")
        total_in = pair_in.groupby(level=0).sum()
        p2 = pair_in / total_in.loc[pair_in.index.get_level_values(0)].values
        ent_in = (-p2 * np.log(p2 + 1e-12)).groupby(level=0).sum().rename("in_src_entropy")
        feat = feat.merge(ent_in, left_on="acct", right_index=True, how="left")

        # reciprocity: number of bidirectional neighbors
        g = tx[[src, dst]].dropna().drop_duplicates()
        rev = g.rename(columns={src: dst, dst: src})
        bi = g.merge(rev, on=[src, dst], how="inner")
        bi_out = bi.groupby(src)[dst].nunique().rename("bi_out_deg")
        feat = feat.merge(bi_out, left_on="acct", right_index=True, how="left")
        feat["bi_out_deg"] = feat["bi_out_deg"].fillna(0.0)
        feat["reciprocity_rate"] = feat["bi_out_deg"] / (feat["out_deg"] + 1e-6)
    else:
        feat["out_dst_entropy"] = 0.0
        feat["in_src_entropy"] = 0.0
        feat["reciprocity_rate"] = 0.0

    # log1p transforms for selected numeric features
    log_cols = [
        "total_send_amt", "total_recv_amt", "max_send_amt", "max_recv_amt", "min_send_amt", "min_recv_amt",
        "avg_send_amt", "avg_recv_amt", "out_tx_count", "in_tx_count",
        "send_per_outdeg", "recv_per_indeg", "tx_per_outdeg", "tx_per_indeg", "mean_amt_overall",
        "out_high90_count", "out_high90_sum", "in_high90_count", "in_high90_sum",
        "out_high99_count", "out_high99_sum", "in_high99_count", "in_high99_sum",
    ]
    for w in windows:
        log_cols += [f'out_tx_count_{w}d', f'out_amt_sum_{w}d', f'in_tx_count_{w}d', f'in_amt_sum_{w}d']
    for c in log_cols:
        if c in feat.columns:
            feat[f"log1p_{c}"] = np.log1p(feat[c])

    # clip outliers
    if clip_quantile is not None:
        num_cols = [c for c in feat.columns if c not in ["acct", "is_esun"]]
        hi = feat[num_cols].quantile(clip_quantile)
        for c in num_cols:
            hi_c = hi.get(c, np.nan)
            if np.isfinite(hi_c):
                feat[c] = np.minimum(feat[c], hi_c)

    # select final feature set
    keep = [
        "acct", "is_esun",
        "total_send_amt", "total_recv_amt", "max_send_amt", "min_send_amt", "avg_send_amt",
        "max_recv_amt", "min_recv_amt", "avg_recv_amt",
        "out_deg", "in_deg", "out_tx_count", "in_tx_count",
        "net_flow_amt", "flow_ratio",
        "send_per_outdeg", "recv_per_indeg", "tx_per_outdeg", "tx_per_indeg",
        "mean_amt_overall",
        "out_high90_count", "out_high90_sum", "in_high90_count", "in_high90_sum",
        "out_high99_count", "out_high99_sum", "in_high99_count", "in_high99_sum",
        "out_dst_entropy", "in_src_entropy", "reciprocity_rate",
        "recency_out_days", "recency_in_days",
    ]
    for w in windows:
        keep += [f'out_tx_count_{w}d', f'out_amt_sum_{w}d', f'in_tx_count_{w}d', f'in_amt_sum_{w}d']
    keep += [c for c in feat.columns if c.startswith("log1p_")]

    feat = feat[keep].fillna(0.0)
    return feat


def build_account_features(tx: pd.DataFrame, col: Dict[str, str]) -> pd.DataFrame:
    """
    Build basic account-level features from transaction logs.

    Compared to `build_account_features_v2`, this version focuses on a smaller
    feature set (no temporal windows or entropy-based features) and is lighter
    to compute.

    Args:
        tx (pd.DataFrame): Transaction records.
        col (Dict[str, str]): Column mapping produced by `resolve_columns`.

    Returns:
        pd.DataFrame:
            A dataframe where each row corresponds to one account, containing:
                - "acct" : account ID
                - "is_esun" : indicator for E.SUN accounts
                - basic aggregated statistics of transaction amount and degree
                - log1p-transformed versions of key numeric features

    用途:
        比賽最終提交使用的主特徵版本（較輕量、較穩定）。
    """
    src, dst, amt = col["src"], col["dst"], col["amt"]
    # total sent / recv
    send = tx.groupby(src)[amt].sum().rename('total_send_amt')
    recv = tx.groupby(dst)[amt].sum().rename('total_recv_amt')
    # max/min/avg sent
    max_send = tx.groupby(src)[amt].max().rename('max_send_amt')
    min_send = tx.groupby(src)[amt].min().rename('min_send_amt')
    avg_send = tx.groupby(src)[amt].mean().rename('avg_send_amt')
    # max/min/avg recv
    max_recv = tx.groupby(dst)[amt].max().rename('max_recv_amt')
    min_recv = tx.groupby(dst)[amt].min().rename('min_recv_amt')
    avg_recv = tx.groupby(dst)[amt].mean().rename('avg_recv_amt')
    # degree-like stats
    out_deg = tx.groupby(src)[dst].nunique().rename('out_deg')
    in_deg = tx.groupby(dst)[src].nunique().rename('in_deg')
    tx_cnt_out = tx.groupby(src)[dst].count().rename('out_tx_count')
    tx_cnt_in = tx.groupby(dst)[src].count().rename('in_tx_count')

    df_out = pd.concat([max_send, min_send, avg_send, send, out_deg, tx_cnt_out], axis=1)
    df_in = pd.concat([max_recv, min_recv, avg_recv, recv, in_deg, tx_cnt_in], axis=1)

    # unify index
    idx = pd.Index(df_out.index).union(df_in.index)
    feat = pd.DataFrame(index=idx).join(df_out, how="left").join(df_in, how="left").fillna(0.0)
    feat = feat.reset_index().rename(columns={"index": "acct"})

    # is_esun
    from_type, to_type = col["from_type"], col["to_type"]
    if from_type is not None and from_type in tx.columns:
        df_from = tx[[src, from_type]].dropna().drop_duplicates().rename(columns={src: "acct", from_type: "is_esun"})
    else:
        df_from = pd.DataFrame(columns=["acct", "is_esun"])
    if to_type is not None and to_type in tx.columns:
        df_to = tx[[dst, to_type]].dropna().drop_duplicates().rename(columns={dst: "acct", to_type: "is_esun"})
    else:
        df_to = pd.DataFrame(columns=["acct", "is_esun"])
    df_acc = pd.concat([df_from, df_to], ignore_index=True).drop_duplicates()
    if not df_acc.empty:
        df_acc = df_acc.groupby("acct")["is_esun"].max().reset_index()
        feat = feat.merge(df_acc, on="acct", how="left")
        feat["is_esun"] = feat["is_esun"].fillna(1)  # default to 1
    else:
        feat["is_esun"] = 1

    # log transform helps stability
    for c in [
        "total_send_amt", "total_recv_amt", "max_send_amt", "max_recv_amt", "min_send_amt", "min_recv_amt",
        "avg_send_amt", "avg_recv_amt", "out_tx_count", "in_tx_count"
    ]:
        feat[f"log1p_{c}"] = np.log1p(feat[c])

    keep = [
        "acct", "is_esun",
        "total_send_amt", "total_recv_amt", "max_send_amt", "min_send_amt", "avg_send_amt",
        "max_recv_amt", "min_recv_amt", "avg_recv_amt",
        "out_deg", "in_deg", "out_tx_count", "in_tx_count"
    ] + [
        f"log1p_{c}" for c in [
            "total_send_amt", "total_recv_amt", "max_send_amt", "max_recv_amt",
            "min_send_amt", "min_recv_amt", "avg_send_amt", "avg_recv_amt",
            "out_tx_count", "in_tx_count"
        ]
    ]
    feat = feat[keep].fillna(0.0)
    return feat


def build_graph(tx: pd.DataFrame, col: Dict[str, str], node_index: Dict[str, int], undirected=True):
    """
    Build a sparse adjacency matrix for the account-level transaction graph.

    Each node represents an account, and an edge (i -> j) is created whenever
    there exists at least one transaction from account i to account j.

    Args:
        tx (pd.DataFrame): Transaction records dataframe.
        col (Dict[str, str]): Column mapping produced by `resolve_columns`.
        node_index (Dict[str, int]): Mapping from account ID (string) to node index (int).
        undirected (bool): If True, convert each directed edge into two symmetric
            edges (i -> j and j -> i).

    Returns:
        torch.Tensor:
            A row-normalized sparse adjacency matrix in COO format with shape (N, N),
            where N = number of accounts.

    用途:
        建立 GNN 訓練用的圖結構，並做 row-normalization，方便後續 message passing。
    """
    src, dst = col["src"], col["dst"]
    # unique pairs to keep graph compact
    g = tx[[src, dst]].dropna().drop_duplicates()
    # map to indices (unknown accounts will be dropped)
    node_keys = set(node_index.keys())
    g = g[g[src].astype(str).isin(node_keys) & g[dst].astype(str).isin(node_keys)]
    u = g[src].astype(str).map(node_index).astype(np.int64).values
    v = g[dst].astype(str).map(node_index).astype(np.int64).values
    if undirected:
        uu = np.concatenate([u, v], axis=0)
        vv = np.concatenate([v, u], axis=0)
    else:
        uu, vv = u, v

    n = len(node_index)
    # row-normalized adjacency values: 1/deg_out(i) for each edge i->j
    deg = np.bincount(uu, minlength=n)
    deg[deg == 0] = 1
    vals = 1.0 / deg[uu]
    idx = np.vstack([uu, vv])
    i = torch.from_numpy(idx).long()
    v = torch.from_numpy(vals.astype(np.float32))
    A = torch.sparse_coo_tensor(i, v, (n, n))
    A = A.coalesce()
    return A


def dropedge(A: torch.Tensor, p: float) -> torch.Tensor:
    """
    Randomly drop edges from a sparse adjacency matrix (DropEdge).

    Edges are kept with probability (1 - p), and the remaining edge weights
    are scaled by 1/(1 - p) to preserve the expected total weight.

    Args:
        A (torch.Tensor): Input sparse adjacency matrix in COO format.
        p (float): Drop probability for each edge (0 <= p < 1).

    Returns:
        torch.Tensor: A new sparse adjacency matrix after DropEdge.

    用途:
        作為 GNN 的正則化手段，降低 overfitting 風險。
    """
    if (not A.is_sparse) or p <= 0.0 or not A._nnz():
        return A
    i = A.indices()
    v = A.values()
    m = v.numel()
    device = v.device
    keep = (torch.rand(m, device=device) > p)
    if keep.sum() == 0:
        return A  # avoid empty
    i2 = i[:, keep]
    v2 = v[keep] / (1.0 - p)
    return torch.sparse_coo_tensor(i2, v2, A.shape, device=device).coalesce()


class _SAGEBlock(nn.Module):
    """
    A single GraphSAGE-style message passing block.

    It performs:
        h_new = GELU( LayerNorm( W_self * x + A * (W_neigh * x) ) )
        and applies dropout afterwards.

    Args:
        in_dim (int): Input feature dimension per node.
        out_dim (int): Output feature dimension per node.
        dropout (float): Dropout probability.

    用途:
        作為 SimpleSAGE / SimpleSAGEv2 的基礎組件，每一層負責一次鄰居聚合。
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim, bias=True)
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.lin_self.weight)
        nn.init.xavier_uniform_(self.lin_neigh.weight)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SAGE block.

        Args:
            x (torch.Tensor): Node feature matrix of shape (N, in_dim).
            A (torch.Tensor): Sparse adjacency matrix (row-normalized).

        Returns:
            torch.Tensor: Updated node features of shape (N, out_dim).
        """
        h_self = self.lin_self(x)
        h_neig = torch.sparse.mm(A, self.lin_neigh(x))
        h = h_self + h_neig
        h = self.norm(h)
        h = F.gelu(h)
        h = self.drop(h)
        return h


class SimpleSAGEv2(nn.Module):
    """
    A deeper GraphSAGE-like GNN with Jumping Knowledge and DropEdge support.

    This model stacks multiple `_SAGEBlock`s and optionally concatenates all
    intermediate layer outputs (JK = 'cat') or simply uses the last layer
    output (JK = 'last'). A final linear layer projects the node embeddings
    to a single logit per node.

    Args:
        in_dim (int): Input feature dimension per node.
        hidden (int): Hidden dimension of each SAGE block.
        depth (int): Number of SAGE blocks (>= 2, recommended 3).
        dropout (float): Dropout probability for each block.
        jk (str): Jumping Knowledge mode, "cat" or "last".
        dropedge_p (float): DropEdge probability applied to adjacency during training.

    用途:
        本比賽的主要 GNN 模型，用於帳戶節點二元分類（是否為警示帳戶）。
    """

    def __init__(self, in_dim: int, hidden: int = 256, depth: int = 3,
                 dropout: float = 0.3, jk: str = "cat", dropedge_p: float = 0.1):
        super().__init__()
        assert depth >= 2
        self.depth = depth
        self.jk = jk
        self.dropedge_p = dropedge_p

        blocks = []
        d_in = in_dim
        for _ in range(depth):
            blocks.append(_SAGEBlock(d_in, hidden, dropout))
            d_in = hidden
        self.blocks = nn.ModuleList(blocks)

        out_dim = hidden * depth if jk == "cat" else hidden
        self.out = nn.Linear(out_dim, 1)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x: torch.Tensor, A: torch.Tensor):
        """
        Forward pass of the SimpleSAGEv2 model.

        Args:
            x (torch.Tensor): Node feature matrix, shape (N, in_dim).
            A (torch.Tensor): Sparse adjacency matrix (row-normalized).

        Returns:
            torch.Tensor: Logits of shape (N,), one logit per node.
        """
        A_use = dropedge(A, self.dropedge_p) if self.training and self.dropedge_p > 0 else A
        hs = []
        h = x
        for blk in self.blocks:
            h_new = blk(h, A_use)
            # residual if same dim
            if h_new.shape == h.shape:
                h = h + h_new
            else:
                h = h_new
            hs.append(h)
        h_final = torch.cat(hs, dim=-1) if self.jk == "cat" else hs[-1]
        logit = self.out(h_final).squeeze(-1)
        return logit


class SimpleSAGE(nn.Module):
    """
    A simpler 2-layer GraphSAGE-like GNN baseline.

    This model uses two fixed SAGE-style layers followed by a linear output
    layer. It is included as a reference / baseline implementation.

    Args:
        in_dim (int): Input feature dimension per node.
        hidden (int): Hidden dimension of internal layers.
        dropout (float): Dropout probability.

    用途:
        早期實驗用的簡化版 GNN，目前主結果使用 SimpleSAGEv2。
    """

    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.lin_self1 = nn.Linear(in_dim, hidden, bias=True)
        self.lin_neigh1 = nn.Linear(in_dim, hidden, bias=False)
        self.lin_self2 = nn.Linear(hidden, hidden, bias=True)
        self.lin_neigh2 = nn.Linear(hidden, hidden, bias=False)
        self.out = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.lin_self1.weight)
        nn.init.xavier_uniform_(self.lin_neigh1.weight)
        nn.init.xavier_uniform_(self.lin_self2.weight)
        nn.init.xavier_uniform_(self.lin_neigh2.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x: torch.Tensor, A: torch.Tensor):
        """
        Forward pass of the SimpleSAGE model.

        Args:
            x (torch.Tensor): Node feature matrix, shape (N, in_dim).
            A (torch.Tensor): Sparse adjacency matrix (row-normalized).

        Returns:
            torch.Tensor: Logits of shape (N,), one logit per node.
        """
        # layer 1
        h_self = self.lin_self1(x)
        h_neig = torch.sparse.mm(A, self.lin_neigh1(x))
        h = F.relu(h_self + h_neig)
        h = self.drop(h)
        # layer 2
        h2_self = self.lin_self2(h)
        h2_neig = torch.sparse.mm(A, self.lin_neigh2(h))
        h2 = F.relu(h2_self + h2_neig)
        h2 = self.drop(h2)
        logit = self.out(h2).squeeze(-1)  # (N,)
        return logit


def train_gnn(
    model,
    x,
    A,
    y,
    train_idx,
    val_idx,
    lr=1e-3,
    weight_decay=1e-4,
    epochs=60,
    pos_weight=None,
    device="cpu"
):
    """
    Train a GNN model for node-level binary classification with early stopping.

    The training uses BCEWithLogitsLoss with an optional positive class weight
    to handle class imbalance. A small subset of nodes is held out as a
    validation set to monitor F1 score and select the best model.

    Args:
        model (nn.Module): GNN model (SimpleSAGE or SimpleSAGEv2).
        x (torch.Tensor): Node feature matrix of shape (N, F).
        A (torch.Tensor): Sparse adjacency matrix in COO format.
        y (torch.Tensor): Label vector of shape (N,), with 0/1 labels for
            training nodes and 0 for unlabeled nodes.
        train_idx (torch.Tensor): 1D tensor of node indices used for training.
        val_idx (torch.Tensor): 1D tensor of node indices used for validation.
        lr (float): Learning rate for Adam optimizer.
        weight_decay (float): Weight decay (L2 regularization).
        epochs (int): Maximum number of training epochs.
        pos_weight (float or None): Positive class weight. If None, it is
            computed as N_neg / N_pos on the training subset.
        device (str): "cpu" or "cuda", or "auto" handled outside.

    Returns:
        nn.Module: The trained model, with weights restored to the best
            validation F1 checkpoint and moved back to CPU.

    用途:
        主要訓練迴圈，並在 val F1 最佳時儲存模型狀態，用於後續推論。
    """
    model = model.to(device)
    x = x.to(device)
    A = A.to(device)
    y = y.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if pos_weight is None:
        # handle imbalance: pos_weight = (Nneg / Npos)
        pos = y[train_idx].sum().item()
        neg = len(train_idx) - pos
        pos_weight = torch.tensor([neg / max(1.0, pos)], dtype=torch.float32, device=device)
    else:
        pos_weight = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val = -1.0
    best_state = None
    patience = 2000
    bad = 0

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(x, A)
        loss = criterion(logits[train_idx], y[train_idx])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()

        # validation (F1 approx with 0.5 threshold)
        model.eval()
        with torch.no_grad():
            val_logit = logits[val_idx]
            val_prob = torch.sigmoid(val_logit)
            pred = (val_prob >= 0.5).float()
            yv = y[val_idx]
            tp = (pred.eq(1) & yv.eq(1)).sum().item()
            fp = (pred.eq(1) & yv.eq(0)).sum().item()
            fn = (pred.eq(0) & yv.eq(1)).sum().item()
            prec = tp / (tp + fp + 1e-9)
            rec = tp / (tp + fn + 1e-9)
            f1 = 2 * prec * rec / (prec + rec + 1e-9)

        if f1 > best_val:
            best_val = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if ep % 5 == 0 or ep == 1:
            print(f"[Epoch {ep:03d}] loss={loss.item():.4f} valF1={f1:.4f} (best {best_val:.4f})")
        if bad >= patience:
            print(f"Early stop at epoch {ep}, best val F1 = {best_val:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to("cpu")
    return model


def best_topk_by_f1(y_true: np.ndarray, y_prob: np.ndarray, max_k: Optional[int] = None):
    """
    Search for the best K (Top-K) that maximizes F1 on a validation set.

    The procedure:
        1. Sort samples by predicted probability in descending order.
        2. For each K from 1 to max_k, treat the top-K samples as positive.
        3. Compute F1 with TP/FP/FN derived from the prefix.
        4. Return the K with the highest F1.

    Args:
        y_true (np.ndarray): Ground-truth binary labels of shape (N,).
        y_prob (np.ndarray): Predicted probabilities of shape (N,).
        max_k (int or None): Maximum K to scan. If None or invalid, use N.

    Returns:
        Tuple[int, float]:
            (best_k, best_f1) where
            - best_k is the argmax K for F1
            - best_f1 is the corresponding F1 score

    用途:
        在驗證集上找出「要挑出幾個帳戶當作陽性」最有利於 F1 的 K，之後可外推到測試集。
    """
    assert y_true.ndim == 1 and y_prob.ndim == 1 and len(y_true) == len(y_prob)
    n = len(y_true)
    if max_k is None or max_k <= 0 or max_k > n:
        max_k = n

    order = np.argsort(-y_prob)  # 機率由大到小
    y_sorted = y_true[order].astype(np.int32)
    ctp = np.cumsum(y_sorted)  # TP(K) = 前 K 個 1 的數量
    P = int(y_true.sum())      # 驗證集真陽性總數

    best_k, best_f1 = 0, 0.0
    for K in range(1, max_k + 1):
        tp = int(ctp[K - 1])
        fp = K - tp
        fn = P - tp
        denom = (2 * tp + fp + fn)
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_k = K
    return best_k, best_f1


def project_k_from_val_to_test(k_val: int, n_val: int, n_test: int, alpha: float = 1.0) -> int:
    """
    Project the best K found on validation set to an appropriate K on test set.

    The projected K is computed by:
        ratio  = k_val / n_val
        k_test = ceil( ratio * n_test * alpha )

    where `alpha` is a scaling factor to make predictions more conservative
    (alpha < 1.0) or more aggressive (alpha > 1.0).

    Args:
        k_val (int): Best K on the validation set.
        n_val (int): Number of samples in the validation set.
        n_test (int): Number of samples in the test set.
        alpha (float): Scaling factor for K (default 1.0).

    Returns:
        int: Projected K for the test set, clamped between 1 and n_test.

    用途:
        將「驗證集最佳 K」依比例放大/縮小到測試集，以控制預測陽性帳戶的數量。
    """
    if n_val <= 0:
        return max(1, int(alpha * n_test * 0.01))  # fallback: 取 1% 作保底
    ratio = (k_val / n_val) * max(0.0, alpha)
    k_test = int(np.ceil(ratio * n_test))
    return int(np.clip(k_test, 1, n_test))


# === Main pipeline logic ===

try:
    tx  # 檢查是否已存在
    alert
    test
    col
    print("tx/alert/test/col 已存在，無需重新載入。")
except NameError:
    from types import SimpleNamespace

    # 若 args 尚未宣告，給預設值（與筆記本的參數設定一致）
    if "args" not in globals():
        args = SimpleNamespace(
            data_dir="preliminary_data",
            hidden=128,
            epochs=500,
            lr=1e-3,
            weight_decay=1e-4,
            seed=42,
            device="auto",
            out_csv="result.csv",
            thr=0.5,
            topk_mode="fallback",  # ["off", "force", "fallback", "auto"]
            topk_alpha=1.0,
            min_pos_pred=1,
            # use_v2=True,
            # v2_windows=(7,30,90),
            # v2_add_heavy=False,
            # v2_clip_quantile=0.999,
        )

    # 若 set_seed / load_csvs / resolve_columns 還沒定義，提醒先跑對應 cells
    for fn in ["set_seed", "load_csvs", "resolve_columns"]:
        if fn not in globals():
            raise RuntimeError(f"請先執行定義 {fn} 的函式後再重試。")

    set_seed(args.seed)
    device = ("cuda" if args.device == "auto" and torch.cuda.is_available()
              else args.device if args.device in ["cpu", "cuda"] else "cpu")

    tx, alert, test = load_csvs(args.data_dir)
    col = resolve_columns(tx, alert, test)
    print("[Column mapping]", col)

# 預設使用 v1 特徵（若要切換 v2，可改用註解區塊）
feat = build_account_features(tx, col)   # columns: acct, is_esun, ...features...

# if getattr(args, "use_v2", False):
#     feat = build_account_features_v2(
#         tx, col,
#         windows=getattr(args, "v2_windows", (7,30,90)),
#         add_heavy=getattr(args, "v2_add_heavy", False),
#         clip_quantile=getattr(args, "v2_clip_quantile", 0.999),
#     )
# else:
#     feat = build_account_features(tx, col)

# Who is positive?
pos_set = set(alert[col["alert_acct"]].astype(str).tolist())
# Test accts (prediction target)
test_list = test[col["predict_acct"]].astype(str).tolist()
test_set = set(test_list)

# Train / Test split (align baseline: only esun accounts for train, and exclude test accts)
train_df = feat[(~feat["acct"].astype(str).isin(test_set)) & (feat["is_esun"] == 1)].copy()
y_train = train_df["acct"].astype(str).map(lambda a: 1 if a in pos_set else 0).astype(np.int64).values

# Build complete node index for graph inference
all_accts = pd.Index(feat["acct"].astype(str).unique())
acct2idx = {a: i for i, a in enumerate(all_accts)}
# map features to X
feat_cols = [c for c in feat.columns if c != "acct"]
X_all = feat.set_index("acct").loc[all_accts][feat_cols].astype(np.float32).values

# Train index / Test index
train_idx = np.array([acct2idx[a] for a in train_df["acct"].astype(str).tolist()], dtype=np.int64)
test_idx_all = np.array([acct2idx[a] for a in test_list if a in acct2idx], dtype=np.int64)

# Standardize using train nodes only
scaler = StandardScaler()
X_all_train_scaled = scaler.fit_transform(X_all[train_idx])
X_all[train_idx] = X_all_train_scaled
# transform others with same scaler
mean = scaler.mean_
scale = getattr(scaler, "scale_", np.sqrt(scaler.var_ + 1e-9))
mask_rest = np.ones(len(X_all), dtype=bool)
mask_rest[train_idx] = False
X_all[mask_rest] = np.clip((X_all[mask_rest] - mean) / (scale + 1e-12), -5, 5)

# Graph (sparse, undirected, row-normalized)
A = build_graph(tx, col, acct2idx, undirected=True)

# Labels vector for all nodes (only those in train_idx used for training)
y_all = np.zeros(len(all_accts), dtype=np.float32)
y_all[train_idx] = y_train

# Build small validation split from train
if y_train.sum() > 0:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    tr_split, va_split = next(skf.split(train_idx, y_train))
    train_nodes = train_idx[tr_split]
    val_nodes = train_idx[va_split]
else:
    rng = np.random.default_rng(args.seed)
    rng.shuffle(train_idx)
    k = max(1, int(len(train_idx) * 0.15))
    val_nodes = train_idx[:k]
    train_nodes = train_idx[k:]

print(f"[Split] train_nodes={len(train_nodes)}, val_nodes={len(val_nodes)}, "
      f"train_pos={int(y_all[train_nodes].sum())}, "
      f"train_neg={len(train_nodes)-int(y_all[train_nodes].sum())}")

x = torch.from_numpy(X_all)         # (N, F)
y = torch.from_numpy(y_all)         # (N,)
train_nodes_t = torch.from_numpy(train_nodes)
val_nodes_t = torch.from_numpy(val_nodes)

# model = SimpleSAGE(in_dim=x.shape[1], hidden=args.hidden, dropout=0.2)
model = SimpleSAGEv2(in_dim=x.shape[1], hidden=160, depth=3, dropout=0.3,
                     jk="cat", dropedge_p=0.1)
device = ("cuda" if args.device == "auto" and torch.cuda.is_available()
          else args.device if args.device in ["cpu", "cuda"] else "cpu")
model = train_gnn(model, x, A, y, train_nodes_t, val_nodes_t,
                  lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs, device=device)

# --- Inference & Top-K calibration ---

set_seed(args.seed)
device = ("cuda" if args.device == "auto" and torch.cuda.is_available()
          else args.device if args.device in ["cpu", "cuda"] else "cpu")

tx, alert, test = load_csvs(args.data_dir)
col = resolve_columns(tx, alert, test)
print("[Column mapping]", col)

model.eval()
with torch.no_grad():
    logits_all = model(x, A)  # (N,)
    prob_all = torch.sigmoid(logits_all).cpu().numpy()

# 驗證集機率與標籤
val_nodes_np = val_nodes_t.cpu().numpy()
yv = y.cpu().numpy()[val_nodes_np].astype(int)
pv = prob_all[val_nodes_np]

# 1) 閾值法
thr = args.thr
yp_thr_val = (pv >= thr).astype(int)
tp = (yp_thr_val & (yv == 1)).sum()
fp = (yp_thr_val & (yv == 0)).sum()
fn = ((yp_thr_val == 0) & (yv == 1)).sum()
prec_thr = tp / (tp + fp + 1e-9)
rec_thr = tp / (tp + fn + 1e-9)
f1_thr = 2 * prec_thr * rec_thr / (prec_thr + rec_thr + 1e-9)

# 2) Top-K 法：在驗證集找最佳 K，並外推到測試集大小
k_val_best, f1_topk_val = best_topk_by_f1(y_true=yv, y_prob=pv, max_k=None)
k_test = project_k_from_val_to_test(k_val=k_val_best, n_val=len(val_nodes_np),
                                    n_test=len(test_idx_all), alpha=args.topk_alpha)

print(f"[Valid] threshold={thr:.3f} -> F1={f1_thr:.4f} | "
      f"TopK (K_val*={k_val_best}) -> F1={f1_topk_val:.4f} ; "
      f"K_test={k_test}")

# 測試集機率（按 acct_predict.csv 順序對齊）
test_list_order = test[col["predict_acct"]].astype(str).tolist()
ptest = prob_all[test_idx_all]
ytest_thr = (ptest >= thr).astype(int)


def need_topk_fallback():
    """
    Decide whether to fall back from threshold-based prediction to Top-K.

    Returns:
        bool: True if the number of positives predicted by the threshold
            is smaller than `min_pos_pred`, False otherwise.

    用途:
        若閾值法在測試集預測的陽性數量過少，則改用 Top-K 方案避免 F1 掉得太低。
    """
    return (ytest_thr.sum() < max(1, args.min_pos_pred))


# 選擇模式
if args.topk_mode == "off":
    final_use_topk = False
elif args.topk_mode == "force":
    final_use_topk = True
elif args.topk_mode == "fallback":
    final_use_topk = need_topk_fallback()
elif args.topk_mode == "auto":
    final_use_topk = (f1_topk_val >= f1_thr - 1e-12)
else:
    final_use_topk = False

if final_use_topk:
    order = np.argsort(-ptest)
    ytest = np.zeros_like(ytest_thr)
    ytest[order[:max(1, int(k_test))]] = 1
    print(f"[Test] Using Top-K with K_test={k_test} (mode={args.topk_mode})")
else:
    ytest = ytest_thr
    print(f"[Test] Using threshold={thr:.3f} (mode={args.topk_mode})")

# 組輸出（維持原檔順序；沒有節點索引的帳戶預設 0）
acct_order = []
y_order = []
# 建立 test acct -> index 映射（只對存在於圖的帳戶）
idx_map = {a: i for i, a in enumerate([a for a in test_list_order if a in acct2idx])}
# 但 ytest 是 test_idx_all 對應的序列，需建立 acct -> y 的映射
acct_in_graph = [a for a in test_list_order if a in acct2idx]
y_map = {a: int(ytest[j]) for j, a in enumerate(acct_in_graph)}

for a in test_list_order:
    acct_order.append(a)
    y_order.append(y_map.get(a, 0))

out = pd.DataFrame({"acct": acct_order, "label": y_order})
out.to_csv(args.out_csv, index=False)
print(f"(Finish) Output saved to {args.out_csv}")
