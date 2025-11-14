"""
2025 玉山人工智慧公開挑戰賽 – 警示帳戶預測 (TEAM_9686)

Pipeline:
1. 讀取官方 CSV
2. 特徵工程 → 帳戶層級特徵
3. 建立帳戶圖 (sparse adjacency)
4. 訓練 GNN 模型
5. 閾值 + Top-K 校準
6. 輸出 result.csv

"""

import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from Preprocess.data_preprocess import (
    read_competition_csvs,
    infer_column_mapping,
    make_account_features_basic,
    normalize_features_by_train,
)

from Model.gnn_model import (
    make_account_graph_matrix,
    AlertAccountGNN,
    train_gnn_model,
    find_best_topk_f1,
    transfer_k_from_val_to_test,
)


# -----------------------------------------------------
# Seed
# -----------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------
# Default args
# -----------------------------------------------------
args = SimpleNamespace(
    data_dir="preliminary_data",
    hidden=160,
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
)


# -----------------------------------------------------
# Pipeline Start
# -----------------------------------------------------
def main():
    set_seed(args.seed)

    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device
        if args.device in ["cpu", "cuda"]
        else "cpu"
    )

    # 1. 讀取 CSV
    tx_df, alert_df, predict_df = read_competition_csvs(args.data_dir)

    # 2. 欄位對應
    col_map = infer_column_mapping(tx_df, alert_df, predict_df)

    # 3. 帳戶層級特徵
    feat_df = make_account_features_basic(tx_df, col_map)

    # 4. 建立標籤與訓練 / 測試帳戶集合
    pos_accts = set(alert_df[col_map["alert_acct"]].astype(str).tolist())
    predict_acct_list = predict_df[col_map["predict_acct"]].astype(str).tolist()
    predict_acct_set = set(predict_acct_list)

    train_df = feat_df[
        (~feat_df["acct"].astype(str).isin(predict_acct_set)) & (feat_df["is_esun"] == 1)
    ].copy()
    y_train = (
        train_df["acct"]
        .astype(str)
        .map(lambda a: 1 if a in pos_accts else 0)
        .astype(np.int64)
        .values
    )

    # 5. 建立全帳戶索引
    all_accts = pd.Index(feat_df["acct"].astype(str).unique())
    acct_to_idx = {acct: idx for idx, acct in enumerate(all_accts)}

    feature_cols = [c for c in feat_df.columns if c != "acct"]
    X_all = (
        feat_df.set_index("acct")
        .loc[all_accts][feature_cols]
        .astype(np.float32)
        .values
    )

    train_idx = np.array(
        [acct_to_idx[a] for a in train_df["acct"].astype(str).tolist()],
        dtype=np.int64,
    )
    test_idx_all = np.array(
        [acct_to_idx[a] for a in predict_acct_list if a in acct_to_idx],
        dtype=np.int64,
    )

    # 6. 特徵標準化
    X_all, scaler = normalize_features_by_train(X_all, train_idx)

    # 7. 建立圖結構
    adj = make_account_graph_matrix(tx_df, col_map, acct_to_idx, undirected=True)

    # 8. 準備標籤向量（所有節點）
    y_all = np.zeros(len(all_accts), dtype=np.float32)
    y_all[train_idx] = y_train

    # 9. 切 train / val
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

    print(f"[Split] train_nodes={len(train_nodes)}, val_nodes={len(val_nodes)}")

    x_tensor = torch.from_numpy(X_all)
    y_tensor = torch.from_numpy(y_all)
    train_nodes_t = torch.from_numpy(train_nodes)
    val_nodes_t = torch.from_numpy(val_nodes)

    # 10. 建立模型
    model = AlertAccountGNN(
        in_dim=x_tensor.shape[1],
        hidden_dim=args.hidden,
        depth=3,
        dropout=0.3,
        use_jk=True,
        edge_drop=0.1,
    )

    # 11. 訓練
    model = train_gnn_model(
        model,
        x_tensor,
        adj,
        y_tensor,
        train_nodes_t,
        val_nodes_t,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        device=device,
    )

    # 12. 推論
    model.eval()
    with torch.no_grad():
        logits_all = model(x_tensor.to(device), adj.to(device)).cpu().numpy()
    prob_all = 1.0 / (1.0 + np.exp(-logits_all))

    # 13. 驗證集評估（threshold vs Top-K）
    val_idx_np = val_nodes_t.numpy()
    y_val = y_tensor.numpy()[val_idx_np].astype(int)
    p_val = prob_all[val_idx_np]

    thr = args.thr
    y_hat_thr = (p_val >= thr).astype(int)

    tp = int(((y_hat_thr == 1) & (y_val == 1)).sum())
    fp = int(((y_hat_thr == 1) & (y_val == 0)).sum())
    fn = int(((y_hat_thr == 0) & (y_val == 1)).sum())

    if tp == 0 or tp + fp == 0 or tp + fn == 0:
        f1_thr = 0.0
    else:
        prec_thr = tp / (tp + fp)
        rec_thr = tp / (tp + fn)
        f1_thr = 2 * prec_thr * rec_thr / (prec_thr + rec_thr)

    best_k_val, f1_topk_val = find_best_topk_f1(y_true=y_val, y_prob=p_val)
    k_test = transfer_k_from_val_to_test(
        best_k_val, len(val_idx_np), len(test_idx_all), args.topk_alpha
    )

    print(
        f"[Valid] threshold F1={f1_thr:.4f} | TopK F1={f1_topk_val:.4f} | "
        f"K_val={best_k_val}, K_test={k_test}"
    )

    # 14. 測試集預測
    p_test = prob_all[test_idx_all]
    y_test_thr = (p_test >= thr).astype(int)

    def need_topk_fallback() -> bool:
        return y_test_thr.sum() < max(1, args.min_pos_pred)

    if args.topk_mode == "off":
        use_topk = False
    elif args.topk_mode == "force":
        use_topk = True
    elif args.topk_mode == "fallback":
        use_topk = need_topk_fallback()
    elif args.topk_mode == "auto":
        use_topk = f1_topk_val >= f1_thr - 1e-12
    else:
        use_topk = False

    if use_topk:
        order = np.argsort(-p_test)
        y_test = np.zeros_like(y_test_thr)
        y_test[order[:max(1, int(k_test))]] = 1
    else:
        y_test = y_test_thr

    # 15. 輸出 result.csv
    acct_to_pred = {
        acct: int(y_test[i])
        for i, acct in enumerate([a for a in predict_acct_list if a in acct_to_idx])
    }

    acct_out = []
    label_out = []
    for acct in predict_acct_list:
        acct_out.append(acct)
        label_out.append(acct_to_pred.get(acct, 0))

    output_df = pd.DataFrame({"acct": acct_out, "label": label_out})
    output_df.to_csv(args.out_csv, index=False)
    print(f"(Finish) Output saved to {args.out_csv}")


if __name__ == "__main__":
    main()
