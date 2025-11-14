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
    column_list = list(df.columns)
    # 建一個小寫對應表，方便之後做不分大小寫比對
    lower_to_original = {col.lower(): col for col in column_list}

    # 1 & 2: exact match
    for cand in candidates:
        # 先試完全一樣（大小寫敏感）
        if cand in column_list:
            return cand

        # 再試不分大小寫
        cand_lower = cand.lower()
        if cand_lower in lower_to_original:
            return lower_to_original[cand_lower]

    # 3: substring match（候選字串被包含在欄位名中）
    for cand in candidates:
        cand_lower = cand.lower()
        for col in column_list:
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
    if amt_col is None:
        tx_df["_CONST_AMT_"] = 1.0
        amt_col = "_CONST_AMT_"

    # 警示帳戶檔與預測名單檔中的帳戶欄位
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
    but with simpler implementation and different column names.
    """
    # 取出實際欄位名稱
    src_col = col_map["src"]   # 付款帳戶
    dst_col = col_map["dst"]   # 收款帳戶
    amt_col = col_map["amt"]   # 交易金額（或常數 1）

    # ---------- 1. 針對「付款方」聚合金額與次數 ----------
    send_group = tx_df.groupby(src_col)[amt_col]

    send_amt_sum = send_group.sum().rename("out_amt_sum")
    send_amt_max = send_group.max().rename("out_amt_max")
    send_amt_min = send_group.min().rename("out_amt_min")
    send_amt_mean = send_group.mean().rename("out_amt_mean")

    # 連結相關：不同收款對手數量、交易筆數
    send_partner_count = (
        tx_df.groupby(src_col)[dst_col]
        .nunique()
        .rename("out_partner_cnt")
    )
    send_tx_count = (
        tx_df.groupby(src_col)[dst_col]
        .count()
        .rename("out_tx_cnt")  
    )

    send_features = pd.concat(
        [
            send_amt_sum,
            send_amt_max,
            send_amt_min,
            send_amt_mean,
            send_partner_count,
            send_tx_count,
        ],
        axis=1,
    )

    # ---------- 2. 針對「收款方」聚合金額與次數 ----------
    recv_group = tx_df.groupby(dst_col)[amt_col]

    recv_amt_sum = recv_group.sum().rename("in_amt_sum") 
    recv_amt_max = recv_group.max().rename("in_amt_max")
    recv_amt_min = recv_group.min().rename("in_amt_min") 
    recv_amt_mean = recv_group.mean().rename("in_amt_mean") 

    recv_partner_count = (
        tx_df.groupby(dst_col)[src_col]
        .nunique()
        .rename("in_partner_cnt")
    )
    recv_tx_count = (
        tx_df.groupby(dst_col)[src_col]
        .count()
        .rename("in_tx_cnt") 
    )

    recv_features = pd.concat(
        [
            recv_amt_sum,
            recv_amt_max,
            recv_amt_min,
            recv_amt_mean,
            recv_partner_count,
            recv_tx_count,
        ],
        axis=1,
    )

    # ---------- 3. 合併成帳戶層級表格 ----------
    all_accounts_index = pd.Index(send_features.index).union(recv_features.index)
    account_features = pd.DataFrame(index=all_accounts_index)
    account_features = account_features.join(send_features, how="left")
    account_features = account_features.join(recv_features, how="left")
    account_features = account_features.fillna(0.0)
    account_features = account_features.reset_index().rename(columns={"index": "acct"})

    # ---------- 4. is_esun 標記（若有帳戶型態欄位） ----------
    from_type_col = col_map["from_type"]
    to_type_col = col_map["to_type"]
    esun_rows = []

    # 從付款方角度取 is_esun
    if from_type_col is not None and from_type_col in tx_df.columns:
        tmp_from = tx_df[[src_col, from_type_col]].copy()
        tmp_from = tmp_from.dropna()
        tmp_from = tmp_from.drop_duplicates()
        tmp_from = tmp_from.rename(columns={src_col: "acct", from_type_col: "is_esun"})
        esun_rows.append(tmp_from)

    # 從收款方角度取 is_esun
    if to_type_col is not None and to_type_col in tx_df.columns:
        tmp_to = tx_df[[dst_col, to_type_col]].copy()
        tmp_to = tmp_to.dropna()
        tmp_to = tmp_to.drop_duplicates()
        tmp_to = tmp_to.rename(columns={dst_col: "acct", to_type_col: "is_esun"})
        esun_rows.append(tmp_to)

    if len(esun_rows) > 0:
        esun_df = pd.concat(esun_rows, ignore_index=True)
        esun_df = esun_df.drop_duplicates()
        esun_df = esun_df.groupby("acct")["is_esun"].max().reset_index()
        account_features = account_features.merge(esun_df, on="acct", how="left")
        account_features["is_esun"] = account_features["is_esun"].fillna(1)
    else:
        # 若完全沒有帳戶型態欄位，預設全部視為玉山帳戶
        account_features["is_esun"] = 1

    # ---------- 5. log1p 特徵 ----------
    numeric_cols = [
        "out_amt_sum",
        "in_amt_sum",
        "out_amt_max",
        "in_amt_max",
        "out_amt_min",
        "in_amt_min",
        "out_amt_mean",
        "in_amt_mean",
        "out_tx_cnt",
        "in_tx_cnt",
    ]

    for col_name in numeric_cols:
        if col_name in account_features.columns:
            new_col_name = f"log1p_{col_name}"
            account_features[new_col_name] = np.log1p(account_features[col_name])
        else:
            new_col_name = f"log1p_{col_name}"
            account_features[new_col_name] = 0.0

    base_cols = [
        "acct",
        "is_esun",
        "out_amt_sum",
        "in_amt_sum",
        "out_amt_max",
        "out_amt_min",
        "out_amt_mean",
        "in_amt_max",
        "in_amt_min",
        "in_amt_mean",
        "out_partner_cnt",
        "in_partner_cnt",
        "out_tx_cnt",
        "in_tx_cnt",
    ]

    log_cols = [f"log1p_{c}" for c in numeric_cols]

    ordered_columns = base_cols + log_cols

    account_features = account_features[ordered_columns].fillna(0.0)

    print("[Preprocess] Built account-level features, shape =", account_features.shape)
    return account_features


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

    # 只用訓練節點去 fit scaler
    train_features = X_all[train_index]
    train_features_scaled = scaler.fit_transform(train_features)
    X_all[train_index] = train_features_scaled

    # 取出 mean / scale，之後手動套到其他節點
    mean_vec = scaler.mean_
    scale_vec = getattr(scaler, "scale_", np.sqrt(scaler.var_ + 1e-9))

    # 建一個 boolean mask，把「不是訓練節點」的位置標出來
    mask_rest = np.ones(len(X_all), dtype=bool)
    mask_rest[train_index] = False

    # 對非訓練節點套用同一組 mean / scale，並做 clip
    X_rest = X_all[mask_rest]
    X_rest = (X_rest - mean_vec) / (scale_vec + 1e-12)
    X_rest = np.clip(X_rest, -5, 5)
    X_all[mask_rest] = X_rest

    return X_all, scaler
