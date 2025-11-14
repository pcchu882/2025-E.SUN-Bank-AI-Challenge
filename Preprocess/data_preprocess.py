"""
Preprocess module for 2025 E.SUN Bank AI Challenge (TEAM_9686).

This module handles:
1. Loading raw CSV files from the competition (acct_transaction, acct_alert, acct_predict)
2. Resolving column names with different naming conventions
3. Building account-level features from transaction records

Author: 褚柏均
"""

from typing import List, Optional, Dict, Tuple
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------
#  Utility: find column name under different naming rules
# ------------------------------------------------------
def find_col(df: pd.DataFrame, cand: List[str]) -> Optional[str]:
    """
    Find a suitable column name in `df` given a list of candidate names.

    The function tries, in this order:
        1. Exact match (case-sensitive)
        2. Exact match (case-insensitive)
        3. Fuzzy match (candidate string is contained in a column name)

    Args:
        df (pd.DataFrame): Input DataFrame whose columns will be searched.
        cand (List[str]): Candidate column names or patterns.

    Returns:
        Optional[str]: The selected column name in `df` if found, otherwise None.
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


# ------------------------------------------------------
# Load CSVs
# ------------------------------------------------------
def load_csvs(dir_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the three main CSV files required by the competition.

    Expected filenames (under `dir_path`):
        - acct_transaction.csv
        - acct_alert.csv
        - acct_predict.csv

    Args:
        dir_path (str): Directory that contains the three CSV files.

    Returns:
        (tx, alert, test) Three dataframes.
    """
    tx = pd.read_csv(os.path.join(dir_path, 'acct_transaction.csv'))
    alert = pd.read_csv(os.path.join(dir_path, 'acct_alert.csv'))
    test = pd.read_csv(os.path.join(dir_path, 'acct_predict.csv'))
    print("(Finish) Load Dataset.")
    return tx, alert, test


# ------------------------------------------------------
# Resolve column names under different naming conventions
# ------------------------------------------------------
def resolve_columns(tx: pd.DataFrame, alert: pd.DataFrame, test: pd.DataFrame) -> Dict[str, str]:
    """
    Resolve and normalize column names across different CSV files.

    This handles different naming conventions:
    - from_acct / payer_acct_id / src / source_account
    - to_acct / receiver_acct_id / dst / target_account
    - txn_amt / amount / tx_amount / trade_amount
    - acct / acct_id / account_id / account

    Returns:
        dict mapping logical column roles to actual dataframe columns.
    """
    src = find_col(tx, ["from_acct", "payer_acct_id", "src", "src_acct", "from", "acct_from", "source_account"])
    dst = find_col(tx, ["to_acct", "receiver_acct_id", "dst", "dst_acct", "to", "acct_to", "target_account"])
    amt = find_col(tx, ["txn_amt", "amount", "tx_amount", "amt", "money", "trade_amount"])

    from_type = find_col(tx, ["from_acct_type", "src_acct_type", "is_esun_from", "from_type"])
    to_type = find_col(tx, ["to_acct_type", "dst_acct_type", "is_esun_to", "to_type"])

    if src is None or dst is None:
        raise ValueError(f"Cannot find src/dst columns in acct_transaction.csv")

    if amt is None:
        tx["_AMT_"] = 1.0
        amt = "_AMT_"

    a_acct = find_col(alert, ["acct", "acct_id", "account_id", "account"])
    a_date = find_col(alert, ["event_date", "alert_date", "date", "datetime", "month"])
    p_acct = find_col(test, ["acct", "acct_id", "account_id", "account"])

    if a_acct is None or p_acct is None:
        raise ValueError("Cannot find acct columns in alert/predict files")

    return {
        "src": src, "dst": dst, "amt": amt,
        "from_type": from_type, "to_type": to_type,
        "alert_acct": a_acct, "alert_date": a_date,
        "predict_acct": p_acct
    }


# ------------------------------------------------------
# Build basic account-level features
# ------------------------------------------------------
def build_account_features(tx: pd.DataFrame, col: Dict[str, str]) -> pd.DataFrame:
    """
    Build basic account-level features from transaction logs.

    This matches the version used in the final submission.
    """
    src, dst, amt = col["src"], col["dst"], col["amt"]

    # total sent / recv
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

    # is_esun
    from_type, to_type = col["from_type"], col["to_type"]
    if from_type and from_type in tx.columns:
        df_from = tx[[src, from_type]].dropna().drop_duplicates().rename(columns={src: "acct", from_type: "is_esun"})
    else:
        df_from = pd.DataFrame(columns=["acct", "is_esun"])

    if to_type and to_type in tx.columns:
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

    # add log transforms
    for c in [
        "total_send_amt","total_recv_amt","max_send_amt","max_recv_amt",
        "min_send_amt","min_recv_amt","avg_send_amt","avg_recv_amt",
        "out_tx_count","in_tx_count"
    ]:
        feat[f"log1p_{c}"] = np.log1p(feat[c])

    keep = [
        "acct","is_esun",
        "total_send_amt","total_recv_amt",
        "max_send_amt","min_send_amt","avg_send_amt",
        "max_recv_amt","min_recv_amt","avg_recv_amt",
        "out_deg","in_deg","out_tx_count","in_tx_count"
    ] + [f"log1p_{c}" for c in [
        "total_send_amt","total_recv_amt","max_send_amt","max_recv_amt",
        "min_send_amt","min_recv_amt","avg_send_amt","avg_recv_amt",
        "out_tx_count","in_tx_count"
    ]]

    feat = feat[keep].fillna(0.0)
    return feat


# ------------------------------------------------------
# Standardize features
# ------------------------------------------------------
def standardize_features(
    X_all: np.ndarray,
    train_idx: np.ndarray
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Standardize node features using only the training nodes, then apply the
    same transformation to all nodes.

    Returns:
        X_all_std, scaler.
    """
    scaler = StandardScaler()
    X_all_train_scaled = scaler.fit_transform(X_all[train_idx])
    X_all[train_idx] = X_all_train_scaled

    mean = scaler.mean_
    scale = getattr(scaler, "scale_", np.sqrt(scaler.var_ + 1e-9))

    mask_rest = np.ones(len(X_all), dtype=bool)
    mask_rest[train_idx] = False

    X_all[mask_rest] = np.clip((X_all[mask_rest] - mean) / (scale + 1e-12), -5, 5)
    return X_all, scaler
