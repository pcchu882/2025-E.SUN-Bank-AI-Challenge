"""
GNN model module for 2025 E.SUN Bank AI Challenge (TEAM_9686).

This module handles:
1. Building a sparse account graph from transaction records
2. Defining a GraphSAGE-like GNN (AlertAccountGNN)
3. Training the GNN with class imbalance handling and simple early stopping
4. Top-K F1 search and K scaling from validation to test

Author: 褚柏均
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F


# ------------------------------------------------------
#  Graph construction
# ------------------------------------------------------
def make_account_graph_matrix(
    tx_df: pd.DataFrame,
    col_map: Dict[str, str],
    acct_to_idx: Dict[str, int],
    undirected: bool = True,
) -> torch.Tensor:
    """
    Build a row-normalized adjacency matrix in sparse COO format.

    Args:
        tx_df: Transaction dataframe.
        col_map: Column name mapping dict (must contain "src", "dst").
        acct_to_idx: Mapping from account string to node index.
        undirected: If True, add reverse edges.

    Returns:
        torch.sparse_coo_tensor: [N, N] row-normalized adjacency.
    """
    src_col = col_map["src"]
    dst_col = col_map["dst"]

    # 只保留 src / dst 兩欄，移除缺失值
    edge_df = tx_df[[src_col, dst_col]].dropna()
    edge_df[src_col] = edge_df[src_col].astype(str)
    edge_df[dst_col] = edge_df[dst_col].astype(str)

    # 只保留出現在 acct_to_idx 中的帳戶
    mask = edge_df[src_col].isin(acct_to_idx) & edge_df[dst_col].isin(acct_to_idx)
    edge_df = edge_df[mask]

    if edge_df.empty:
        # 沒有任何邊時，回傳零矩陣（避免程式壞掉）
        n_nodes = len(acct_to_idx)
        indices = torch.zeros((2, 0), dtype=torch.long)
        values = torch.zeros((0,), dtype=torch.float32)
        return torch.sparse_coo_tensor(indices, values, (n_nodes, n_nodes))

    u = edge_df[src_col].map(acct_to_idx).to_numpy(dtype=np.int64)
    v = edge_df[dst_col].map(acct_to_idx).to_numpy(dtype=np.int64)

    if undirected:
        row_idx = np.concatenate([u, v], axis=0)
        col_idx = np.concatenate([v, u], axis=0)
    else:
        row_idx = u
        col_idx = v

    num_nodes = len(acct_to_idx)

    # row-normalized adjacency: 每個節點的 outgoing 邊權重總和為 1
    ones = np.ones(len(row_idx), dtype=np.float32)
    out_degree = np.bincount(row_idx, minlength=num_nodes).astype(np.float32)
    out_degree[out_degree == 0] = 1.0
    norm_values = ones / out_degree[row_idx]

    indices = torch.from_numpy(
        np.vstack([row_idx, col_idx])
    ).long()  # shape: [2, num_edges]
    values = torch.from_numpy(norm_values)  # shape: [num_edges]

    adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))
    adj = adj.coalesce()
    return adj


# ------------------------------------------------------
#  Edge dropout (DropEdge)
# ------------------------------------------------------
def apply_edge_dropout(adj: torch.Tensor, drop_prob: float) -> torch.Tensor:
    """
    Randomly drop edges in the adjacency matrix (DropEdge).

    Args:
        adj: Sparse adjacency matrix in COO format.
        drop_prob: Probability of dropping each edge.

    Returns:
        New sparse adjacency after dropping some edges.
    """
    if drop_prob <= 0.0 or adj._nnz() == 0:
        return adj

    adj = adj.coalesce()
    idx = adj.indices()  # [2, E]
    val = adj.values()   # [E]

    num_edges = val.size(0)
    keep_prob = 1.0 - drop_prob

    mask = torch.rand(num_edges, device=val.device) < keep_prob
    if mask.sum() == 0:
        # 避免剛好全部被丟棄，至少保留一條
        mask[0] = True

    idx_new = idx[:, mask]
    val_new = val[mask] / keep_prob  # rescale

    out = torch.sparse_coo_tensor(idx_new, val_new, adj.shape, device=adj.device)
    return out.coalesce()


# ------------------------------------------------------
#  GNN building blocks
# ------------------------------------------------------
class SageLayer(nn.Module):
    """
    A single GraphSAGE-like layer with:
        - self linear
        - neighbor aggregation via A @ x
        - LayerNorm + GELU + Dropout
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim, bias=True)
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.lin_self.weight)
        nn.init.xavier_uniform_(self.lin_neigh.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [N, F_in]
            adj: Sparse adjacency [N, N]
        """
        neigh = torch.sparse.mm(adj, x)  # [N, F_in]
        h = self.lin_self(x) + self.lin_neigh(neigh)
        h = self.norm(h)
        h = F.gelu(h)
        h = self.drop(h)
        return h


class AlertAccountGNN(nn.Module):
    """
    A simple multi-layer GraphSAGE-style GNN for alert-account prediction.

    - Multiple SageLayer stacked
    - Optional residual connection when dimensions match
    - Jumping Knowledge (concat all layer outputs)
    - Final linear layer outputs a logit per node
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 160,
        depth: int = 3,
        dropout: float = 0.3,
        use_jk: bool = True,
        edge_drop: float = 0.1,
    ):
        super().__init__()
        self.depth = depth
        self.use_jk = use_jk
        self.edge_drop = edge_drop

        layers = []
        input_dim = in_dim
        for _ in range(depth):
            layer = SageLayer(input_dim, hidden_dim, dropout)
            layers.append(layer)
            input_dim = hidden_dim
        self.layers = nn.ModuleList(layers)

        if use_jk:
            final_in_dim = hidden_dim * depth
        else:
            final_in_dim = hidden_dim

        self.out_linear = nn.Linear(final_in_dim, 1)
        nn.init.xavier_uniform_(self.out_linear.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [N, F]
            adj: Sparse adjacency [N, N]
        Returns:
            logits: [N] (squeezed)
        """
        if self.training and self.edge_drop > 0.0:
            adj_used = apply_edge_dropout(adj, self.edge_drop)
        else:
            adj_used = adj

        h = x
        layer_outputs = []

        for layer in self.layers:
            new_h = layer(h, adj_used)
            if new_h.shape == h.shape:
                h = h + new_h
            else:
                h = new_h
            layer_outputs.append(h)

        if self.use_jk:
            h_cat = torch.cat(layer_outputs, dim=-1)
        else:
            h_cat = layer_outputs[-1]

        logits = self.out_linear(h_cat).squeeze(-1)
        return logits


# ------------------------------------------------------
#  Training loop
# ------------------------------------------------------
def train_gnn_model(
    model: nn.Module,
    node_feat: torch.Tensor,
    adj: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 500,
    device: str = "cpu",
    print_every: int = 20,
) -> nn.Module:
    """
    Train the GNN with BCEWithLogitsLoss and simple early stopping on F1.

    Args:
        model: GNN model.
        node_feat: Node features [N, F].
        adj: Sparse adjacency [N, N].
        labels: Node labels [N] in {0,1}, float or long.
        train_idx: Indices of training nodes.
        val_idx: Indices of validation nodes.
        lr, weight_decay, epochs, device: Training hyper-parameters.

    Returns:
        The model with the best validation F1.
    """
    device = torch.device(device)
    model = model.to(device)
    node_feat = node_feat.to(device)
    adj = adj.to(device)
    labels = labels.to(device)

    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)

    y_train = labels[train_idx]
    num_pos = (y_train == 1).sum().item()
    num_neg = (y_train == 0).sum().item()

    if num_pos > 0:
        pos_weight = torch.tensor([num_neg / max(num_pos, 1)], device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_f1 = -1.0
    best_state = None
    patience = 0
    max_patience = 80  # simple early stopping

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(node_feat, adj)
        train_logits = logits[train_idx]
        train_labels = labels[train_idx]

        loss = loss_fn(train_logits, train_labels.float())
        loss.backward()
        optimizer.step()

        # 評估驗證集 F1
        if epoch % print_every == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                val_logits = model(node_feat, adj)[val_idx]
                val_probs = torch.sigmoid(val_logits)
                val_pred = (val_probs >= 0.5).long()
                val_true = labels[val_idx].long()

                tp = (val_pred.eq(1) & val_true.eq(1)).sum().item()
                fp = (val_pred.eq(1) & val_true.eq(0)).sum().item()
                fn = (val_pred.eq(0) & val_true.eq(1)).sum().item()

                if tp + fp == 0 or tp + fn == 0:
                    f1 = 0.0
                else:
                    precision = tp / (tp + fp)
                    recall = tp / (tp + fn)
                    if precision + recall == 0:
                        f1 = 0.0
                    else:
                        f1 = 2 * precision * recall / (precision + recall)

            if f1 > best_f1 + 1e-6:
                best_f1 = f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1

            print(
                f"Epoch {epoch:03d} loss={loss.item():.4f} "
            )

            if patience >= max_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# ------------------------------------------------------
#  Top-K search & K projection
# ------------------------------------------------------
def find_best_topk_f1(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[int, float]:
    """
    Search the best K (Top-K) on validation set to maximize F1.

    Args:
        y_true: Ground truth labels (0/1), shape [N].
        y_prob: Predicted probabilities, shape [N].

    Returns:
        (best_k, best_f1).
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    n = len(y_true)
    if n == 0:
        return 1, 0.0

    order = np.argsort(-y_prob)
    y_sorted = y_true[order]

    total_pos = y_true.sum()
    if total_pos == 0:
        # 沒有任何正樣本，無論怎麼選都是 F1=0
        return 1, 0.0

    cum_pos = np.cumsum(y_sorted)
    best_k = 1
    best_f1 = 0.0

    for k in range(1, n + 1):
        tp = cum_pos[k - 1]
        fp = k - tp
        fn = total_pos - tp

        if tp == 0:
            f1 = 0.0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)

        if f1 > best_f1:
            best_f1 = f1
            best_k = k

    return best_k, float(best_f1)


def transfer_k_from_val_to_test(
    k_val: int,
    n_val: int,
    n_test: int,
    alpha: float = 1.0,
) -> int:
    """
    Scale K from validation to test set.

    K_test = ceil( (K_val / N_val) * N_test * alpha )

    Args:
        k_val: Best K on validation.
        n_val: Number of validation samples.
        n_test: Number of test samples.
        alpha: Scaling factor for K.

    Returns:
        Projected K for the test set (clamped to [1, n_test]).
    """
    if n_test <= 0:
        return 0

    if n_val <= 0 or k_val <= 0:
        # fallback: roughly 1% as a starting guess
        fallback = int(np.ceil(0.01 * n_test))
        return max(1, min(n_test, fallback))

    ratio = k_val / float(n_val)
    ratio = max(ratio * max(alpha, 0.0), 0.0)
    k_test = int(np.ceil(ratio * n_test))
    k_test = max(1, min(n_test, k_test))
    return k_test
