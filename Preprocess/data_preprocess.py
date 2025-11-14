"""
Preprocess module for 2025 E.SUN Bank AI Challenge (TEAM_9686).

This module handles:
1. Loading raw CSV files from the competition (acct_transaction, acct_alert, acct_predict)
2. Resolving column names with different naming conventions
3. Building account-level features from transaction records

"""

from typing import List, Optional, Dict, Tuple
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------
#  Utility: pick a column name from candidates
# ------------------------------------------------------
def pick_col_name(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Pick a suitable column name in `df` from a list of candidate names.

    Matching order:
        1. Exact match (case-sensitive)
        2. Exact match (case-insensitive)
        3. Substring match (candidate appears in column name, case-insensitive)
    """
    col_list = list(df.columns)
    lower_map = {c.lower(): c for c in col_list}

    # 1 & 2: exact match
    for cand in candidates:
        if cand in col_list:
            return cand
        cand_lower = cand.lower()
        if cand_lower in lower_map:
            return lower_map[cand_lower]

    # 3: substring match
    for cand in candidates:
        cand_lower = cand.lower()
        for col in col_list:
            if cand_lower in col.lower():
                return col

    return None


# ------------------------------------------------------
# Load CSVs
# ------------------------------------------------------
def read_competition_csvs(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read the three official CSV files from the given directory.

    Files:
        - acct_transaction.csv
        - acct_alert.csv
        - acct_predict.csv
    """
    tx_path = os.path.join(data_dir, "acct_transaction.csv")
    alert_path = os.path.join(data_dir, "acct_alert.csv")
    predict_path = os.path.join(data_dir, "acct_predict.csv")

    tx_df = pd.read_csv(tx_path)
    alert_df = pd.read_csv(alert_path)
    predict_df = pd.read_csv(predict_path)

    print("[Preprocess] Loaded transaction / alert / predict CSVs from:", data_dir)
    return tx_df, alert_df, predict_df


# ------------------------------------------------------
# Resolve column names under different naming conventions
# ------------------------------------------------------
def infer_column_mapping(
    tx_df: pd.DataFrame,
    alert_df: pd.DataFrame,
    predict_df: pd.DataFrame,
) -> Dict[str, str]:
    """
    Infer and normalize column names across different CSV files.

    This handles different naming conventions, e.g.:
        - from_acct / payer_acct_id / src / source_account
        - to_acct   / receiver_acct_id / dst / target_account
        - txn_amt   / amount / tx_amount / trade_amount
        - acct      / acct_id / account_id / account

    Returns:
        A dict mapping:
            - "src", "dst", "amt"
            - "from_type", "to_type"
            - "alert_acct", "alert_date"
            - "predict_acct"
    """
    src_col = pick_col_name(
        tx_df,
        ["from_acct", "payer_acct_id", "src", "src_acct", "from", "acct_from", "source_account"],
    )
    dst_col = pick_col_name(
        tx_df,
        ["to_acct", "receiver_acct_id", "dst", "dst_acct", "to", "acct_to", "target_account"],
    )
    amt_col = pick_col_name(
        tx_df,
        ["txn_amt", "amount", "tx_amount", "amt", "money", "trade_amount"],
    )

    from_type_col = pick_col_name(
        tx_df,
        ["from_acct_type", "src_acct_type", "is_esun_from", "from_type"],
    )
    to_type_col = pick_col_name(
        tx_df,
        ["to_acct_type", "dst_acct_type", "is_esun_to", "to_type"],
    )

    if src_col is None or dst_col is None:
        raise ValueError("Cannot find src/dst columns in acct_transaction.csv")

    # 若沒有金額欄位，就用常數 1.0，代表「交易次數」
    if amt_col is None:
        tx_df["_AMT_"] = 1.0
        amt_col = "_AMT_"

    alert_acct_col = pick_col_name(alert_df, ["acct", "acct_id", "account_id", "account"])
    alert_date_col = pick_col_name(alert_df, ["event_date", "alert_date", "date", "datetime", "month"])
    predict_acct_col = pick_col_name(predict_df, ["acct", "acct_id", "account_id", "account"])

    if alert_acct_col is None or predict_acct_col is None:
        raise ValueError("Cannot find acct columns in alert/predict files")

    col_map = {
        "src": src_col,
        "dst": dst_col,
        "amt": amt_col,
        "from_type": from_type_col,
        "to_type": to_type_col,
        "alert_acct": alert_acct_col,
        "alert_date": alert_date_col,
        "predict_acct": predict_acct_col,
    }
    print("[Preprocess] Column mapping:", col_map)
    return col_map


# ------------------------------------------------------
# Build basic account-level features
# ------------------------------------------------------
def make_account_features_basic(tx_df: pd.DataFrame, col_map: Dict[str, str]) -> pd.DataFrame:
    """
    Build basic account-level features from transaction logs.

    Feature list roughly matches the version used in the final submission,
    but the implementation is kept simple for readability.
    """
    src = col_map["src"]
    dst = col_map["dst"]
    amt = col_map["amt"]

    # --- 1. 針對「付款方」聚合金額與次數 ---
    src_agg_amt = tx_df.groupby(src)[amt].agg(
        total_send_amt="sum",
        max_send_amt="max",
        min_send_amt="min",
        avg_send_amt="mean",
    )
    src_agg_cnt = tx_df.groupby(src)[dst].agg(
        out_deg="nunique",
        out_tx_count="count",
    )
    send_side = pd.concat([src_agg_amt, src_agg_cnt], axis=1)

    # --- 2. 針對「收款方」聚合金額與次數 ---
    dst_agg_amt = tx_df.groupby(dst)[amt].agg(
        total_recv_amt="sum",
        max_recv_amt="max",
        min_recv_amt="min",
        avg_recv_amt="mean",
    )
    dst_agg_cnt = tx_df.groupby(dst)[src].agg(
        in_deg="nunique",
        in_tx_count="count",
    )
    recv_side = pd.concat([dst_agg_amt, dst_agg_cnt], axis=1)

    # --- 3. 合併成帳戶層級表格 ---
    all_acct_index = pd.Index(send_side.index).union(recv_side.index)
    feat_df = (
        pd.DataFrame(index=all_acct_index)
        .join(send_side, how="left")
        .join(recv_side, how="left")
        .fillna(0.0)
        .reset_index()
        .rename(columns={"index": "acct"})
    )

    # --- 4. is_esun 標記（若有帳戶型態欄位） ---
    from_type_col = col_map["from_type"]
    to_type_col = col_map["to_type"]
    acct_esun_list = []

    if from_type_col and from_type_col in tx_df.columns:
        tmp = (
            tx_df[[src, from_type_col]]
            .dropna()
            .drop_duplicates()
            .rename(columns={src: "acct", from_type_col: "is_esun"})
        )
        acct_esun_list.append(tmp)

    if to_type_col and to_type_col in tx_df.columns:
        tmp = (
            tx_df[[dst, to_type_col]]
            .dropna()
            .drop_duplicates()
            .rename(columns={dst: "acct", to_type_col: "is_esun"})
        )
        acct_esun_list.append(tmp)

    if len(acct_esun_list) > 0:
        acct_esun_df = pd.concat(acct_esun_list, ignore_index=True).drop_duplicates()
        acct_esun_df = acct_esun_df.groupby("acct")["is_esun"].max().reset_index()
        feat_df = feat_df.merge(acct_esun_df, on="acct", how="left")
        feat_df["is_esun"] = feat_df["is_esun"].fillna(1)
    else:
        # 若完全沒有帳戶型態欄位，預設視為玉山帳戶
        feat_df["is_esun"] = 1

    # --- 5. log1p 特徵 ---
    numeric_cols = [
        "total_send_amt",
        "total_recv_amt",
        "max_send_amt",
        "max_recv_amt",
        "min_send_amt",
        "min_recv_amt",
        "avg_send_amt",
        "avg_recv_amt",
        "out_tx_count",
        "in_tx_count",
    ]
    for col_name in numeric_cols:
        if col_name in feat_df.columns:
            feat_df[f"log1p_{col_name}"] = np.log1p(feat_df[col_name])
        else:
            feat_df[f"log1p_{col_name}"] = 0.0

    # 欄位順序：acct / is_esun 在最前面
    ordered_columns = (
        ["acct", "is_esun"]
        + [
            "total_send_amt",
            "total_recv_amt",
            "max_send_amt",
            "min_send_amt",
            "avg_send_amt",
            "max_recv_amt",
            "min_recv_amt",
            "avg_recv_amt",
            "out_deg",
            "in_deg",
            "out_tx_count",
            "in_tx_count",
        ]
        + [f"log1p_{c}" for c in numeric_cols]
    )

    feat_df = feat_df[ordered_columns].fillna(0.0)
    print("[Preprocess] Built account-level features, shape =", feat_df.shape)
    return feat_df


# ------------------------------------------------------
# Standardize features
# ------------------------------------------------------
def normalize_features_by_train(
    X_all: np.ndarray,
    train_index: np.ndarray,
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Standardize node features using only the training nodes, then apply the
    same transformation to all nodes. Non-training nodes are clipped to [-5, 5].
    """
    scaler = StandardScaler()

    # 只用訓練節點 fit scaler
    X_all[train_index] = scaler.fit_transform(X_all[train_index])

    mean_vec = scaler.mean_
    scale_vec = getattr(scaler, "scale_", np.sqrt(scaler.var_ + 1e-9))

    # 其餘節點使用相同 mean/scale 做標準化
    mask_rest = np.ones(len(X_all), dtype=bool)
    mask_rest[train_index] = False

    X_all[mask_rest] = (X_all[mask_rest] - mean_vec) / (scale_vec + 1e-12)
    X_all[mask_rest] = np.clip(X_all[mask_rest], -5, 5)

    return X_all, scaler
