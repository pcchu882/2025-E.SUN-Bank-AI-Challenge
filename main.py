"""
2025 玉山人工智慧公開挑戰賽 – 警示帳戶預測 (TEAM_9686)

Main pipeline:
1. Load raw CSV via Preprocess module
2. Build features
3. Build graph
4. Train GNN
5. Threshold + Top-K calibration
6. Output result.csv

Author: 褚柏均
"""


import os
import random
import numpy as np
import pandas as pd
import torch
from types import SimpleNamespace
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Import from Preprocess and Model modules
from Preprocess.data_preprocess import (
    load_csvs,
    resolve_columns,
    build_account_features,
    standardize_features,
)

from Model.gnn_model import (
    build_graph,
    SimpleSAGEv2,
    train_gnn,
    best_topk_by_f1,
    project_k_from_val_to_test,
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
set_seed(args.seed)

device = ("cuda" if args.device == "auto" and torch.cuda.is_available()
          else args.device if args.device in ["cpu", "cuda"] else "cpu")

# Load CSVs
tx, alert, test = load_csvs(args.data_dir)
col = resolve_columns(tx, alert, test)
print("[Column mapping]", col)

# Build features
feat = build_account_features(tx, col)

# Label info
pos_set = set(alert[col["alert_acct"]].astype(str).tolist())
test_list = test[col["predict_acct"]].astype(str).tolist()
test_set = set(test_list)

# Train df (exclude test + only esun)
train_df = feat[(~feat["acct"].astype(str).isin(test_set)) & (feat["is_esun"] == 1)].copy()
y_train = train_df["acct"].astype(str).map(lambda a: 1 if a in pos_set else 0).astype(np.int64).values

# Build node index
all_accts = pd.Index(feat["acct"].astype(str).unique())
acct2idx = {a: i for i, a in enumerate(all_accts)}

feat_cols = [c for c in feat.columns if c != "acct"]
X_all = feat.set_index("acct").loc[all_accts][feat_cols].astype(np.float32).values

train_idx = np.array([acct2idx[a] for a in train_df["acct"].astype(str).tolist()], dtype=np.int64)
test_idx_all = np.array([acct2idx[a] for a in test_list if a in acct2idx], dtype=np.int64)

# Standardize
X_all, scaler = standardize_features(X_all, train_idx)

# Build graph
A = build_graph(tx, col, acct2idx, undirected=True)

# Labels for all nodes
y_all = np.zeros(len(all_accts), dtype=np.float32)
y_all[train_idx] = y_train

# Train/Val split
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

x = torch.from_numpy(X_all)
y = torch.from_numpy(y_all)
train_nodes_t = torch.from_numpy(train_nodes)
val_nodes_t = torch.from_numpy(val_nodes)

# Create model
model = SimpleSAGEv2(in_dim=x.shape[1], hidden=160, depth=3,
                     dropout=0.3, jk="cat", dropedge_p=0.1)

# Train
model = train_gnn(model, x, A, y, train_nodes_t, val_nodes_t,
                  lr=args.lr, weight_decay=args.weight_decay,
                  epochs=args.epochs, device=device)

# Inference
model.eval()
with torch.no_grad():
    logits_all = model(x, A)
    prob_all = torch.sigmoid(logits_all).cpu().numpy()

# Validation
val_nodes_np = val_nodes_t.numpy()
yv = y.numpy()[val_nodes_np].astype(int)
pv = prob_all[val_nodes_np]

# Threshold method
thr = args.thr
yp_thr = (pv >= thr).astype(int)

tp = (yp_thr & (yv == 1)).sum()
fp = (yp_thr & (yv == 0)).sum()
fn = ((yp_thr == 0) & (yv == 1)).sum()
prec_thr = tp / (tp + fp + 1e-9)
rec_thr = tp / (tp + fn + 1e-9)
f1_thr = 2 * prec_thr * rec_thr / (prec_thr + rec_thr + 1e-9)

# Top-K method
k_val_best, f1_topk_val = best_topk_by_f1(y_true=yv, y_prob=pv)
k_test = project_k_from_val_to_test(k_val_best, len(val_nodes_np), len(test_idx_all), args.topk_alpha)

print(f"[Valid] threshold F1={f1_thr:.4f} | TopK F1={f1_topk_val:.4f} | K_val={k_val_best}, K_test={k_test}")

# Test set predict
ptest = prob_all[test_idx_all]
ytest_thr = (ptest >= thr).astype(int)

def need_topk_fallback():
    return (ytest_thr.sum() < max(1, args.min_pos_pred))

# Select mode
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
else:
    ytest = ytest_thr

# Output
acct_order = []
y_order = []
acct_map = {a: int(ytest[j]) for j, a in enumerate([a for a in test_list if a in acct2idx])}

for a in test_list:
    acct_order.append(a)
    y_order.append(acct_map.get(a, 0))

out = pd.DataFrame({"acct": acct_order, "label": y_order})
out.to_csv(args.out_csv, index=False)
print(f"(Finish) Output saved to {args.out_csv}")
