"""
GNN model module for 2025 E.SUN Bank AI Challenge (TEAM_9686).

This module handles:
1. Building the account graph from transaction records
2. Defining GraphSAGE-like GNN models (SimpleSAGE / SimpleSAGEv2)
3. Training the GNN with class-imbalance handling and early stopping
4. Top-K F1 search and K projection from validation to test

Author: 褚柏均
"""

from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold


def build_graph(tx: pd.DataFrame, col: Dict[str, str], node_index: Dict[str, int], undirected: bool = True) -> torch.Tensor:
    """
    Build a sparse adjacency matrix for the account-level transaction graph.

    Each node represents an account, and an edge (i -> j) is created whenever
    there exists at least one transaction from account i to account j.

    Args:
        tx (pd.DataFrame): Transaction records dataframe.
        col (Dict[str, str]): Column mapping produced by `resolve_columns`.
        node_index (Dict[str, int]): Mapping from account ID to node index.
        undirected (bool): If True, convert each directed edge into two
            symmetric edges (i -> j and j -> i).

    Returns:
        torch.Tensor: Row-normalized sparse adjacency matrix in COO format.
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
        torch.Tensor: New sparse adjacency matrix after DropEdge.
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

    Args:
        in_dim (int): Input feature dimension per node.
        hidden (int): Hidden dimension of each SAGE block.
        depth (int): Number of SAGE blocks.
        dropout (float): Dropout probability.
        jk (str): Jumping Knowledge mode, "cat" or "last".
        dropedge_p (float): DropEdge probability applied during training.
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


def train_gnn(
    model: nn.Module,
    x: torch.Tensor,
    A: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 500,
    pos_weight: Optional[float] = None,
    device: str = "cpu"
) -> nn.Module:
    """
    Train a GNN model for node-level binary classification with early stopping.

    Args:
        model (nn.Module): GNN model (SimpleSAGEv2 in our final solution).
        x (torch.Tensor): Node feature matrix (N, F).
        A (torch.Tensor): Sparse adjacency matrix in COO format.
        y (torch.Tensor): Label vector (N,) with 0/1 labels.
        train_idx (torch.Tensor): Indices of nodes used for training.
        val_idx (torch.Tensor): Indices of nodes held out for validation.
        lr (float): Learning rate.
        weight_decay (float): Weight decay (L2 regularization).
        epochs (int): Maximum number of epochs.
        pos_weight (float or None): Positive class weight for BCEWithLogitsLoss.
        device (str): "cpu" or "cuda".

    Returns:
        nn.Module: Trained model restored to the best validation F1 checkpoint.
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


def best_topk_by_f1(y_true: np.ndarray, y_prob: np.ndarray, max_k: Optional[int] = None) -> Tuple[int, float]:
    """
    Search for the best K (Top-K) that maximizes F1 on a validation set.

    Args:
        y_true (np.ndarray): Ground-truth binary labels (N,).
        y_prob (np.ndarray): Predicted probabilities (N,).
        max_k (int or None): Maximum K to scan. If None, use N.

    Returns:
        Tuple[int, float]: (best_k, best_f1).
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

    Args:
        k_val (int): Best K on validation.
        n_val (int): Number of validation samples.
        n_test (int): Number of test samples.
        alpha (float): Scaling factor for K.

    Returns:
        int: Projected K for the test set.
    """
    if n_val <= 0:
        return max(1, int(alpha * n_test * 0.01))  # fallback: 取 1% 作保底
    ratio = (k_val / n_val) * max(0.0, alpha)
    k_test = int(np.ceil(ratio * n_test))
    return int(np.clip(k_test, 1, n_test))
