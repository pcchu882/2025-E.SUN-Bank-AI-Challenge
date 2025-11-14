# Model 模組 – GNN 模型架構與訓練流程

本資料夾包含本專案所有與 **圖神經網路（GNN）模型** 相關的程式碼。模型負責根據帳戶的交易特徵與帳戶之間的關聯圖，預測哪些帳戶可能為「警示帳戶」。本模組的所有功能皆由 `main.py` 自動呼叫，不需另外執行。

---

## 1. 帳戶關聯圖建立（Graph Construction）

由 `make_account_graph_matrix()` 生成：

* **節點（Node）＝帳戶**
* **邊（Edge）＝帳戶間曾發生交易**
* 依據交易紀錄建立 **無向圖（Undirected Graph）**
* 對每個節點的 outgoing 邊做 **row-normalized** 權重，使得 GNN 訊息傳遞更穩定

此步驟可有效表達帳戶之間的金流關係，並讓 GNN 能捕捉鄰居模式（neighbor behavior）。

---

## 2. DropEdge 正則化

由 `apply_edge_dropout()` 於訓練階段隨機丟棄部分邊：

* 降低 overfitting
* 增加模型泛化能力
* 特別適用於金融交易圖（邊密度高、噪音多）

丟棄後會進行 rescale，保持期望值一致。

---

## 3. 核心模型 – AlertAccountGNN（SimpleSAGEv2）

本次比賽最佳表現模型為 **SimpleSAGEv2**，由 `AlertAccountGNN` 定義。

### 🔧 模型特色：

* **GraphSAGE-style** 聚合器（self + neighbor）
* **3 層** SageLayer（depth 可調）
* 每層包含：LayerNorm、GELU、Dropout
* **Residual Connection**：維度相同時保留殘差
* **Jumping Knowledge (JK-cat)**：將每層輸出串接，融合多階層資訊
* **DropEdge** 於 forward 自動作用
* 最後使用線性層輸出 **每個帳戶的 logit**

此架構對低比例異常偵測（例如警示帳戶）具良好穩定性，也能有效利用圖結構資訊。

---

## 4. 訓練流程（train_gnn_model）

`train_gnn_model()` 內含完整 GNN 訓練邏輯：

* 使用 **BCEWithLogitsLoss**
* 自動計算 **pos_weight** 處理類別不平衡
* 以 validation F1 作為 early stopping 依據
* 每 N epoch 進行驗證評估
* 儲存 F1 最佳權重
* 最終返回最佳模型

在本比賽資料上，此方法能穩定達到良好 F1 表現（私榜最佳 0.3849）。

---

## 5. Top-K F1 校準策略

因為警示帳戶比例極低，僅使用 threshold 會導致：

* 通常預測不到足夠陽性
* F1 顯著下降

因此採用 **Top-K 搜尋**：

### (1) `find_best_topk_f1()`

在 validation set 依照排序後逐一嘗試 K＝1..N，找出 F1 最高的 K。

### (2) `transfer_k_from_val_to_test()`

將驗證集最佳 K 外推到測試集：

```
K_test = ceil( (K_val / N_val) * N_test * alpha )
```

此方式能比單純 threshold 更穩定地找到最佳預測比例，特別適合本次競賽的極度不平衡標籤。

---

## 檔案說明

| 檔案                 | 用途                                    |
| ------------------ | ------------------------------------- |
| **`gnn_model.py`** | GNN 模型定義、DropEdge、訓練流程、Top-K 搜尋與 K 對應 |

---

## 與 main.py 的整合位置

在 `main.py` 中使用：

```python
from Model.gnn_model import (
    make_account_graph_matrix,
    AlertAccountGNN,
    train_gnn_model,
    find_best_topk_f1,
    transfer_k_from_val_to_test,
)
```

並於主流程中：

1. 建立 adjacency graph
2. 建立 GNN 模型
3. 訓練 GNN
4. 推論所有帳戶分數
5. Top-K / threshold 校準
6. 產生 result.csv

---

如果你想加入：模型架構圖、流程圖、F1 曲線圖等，我也可以幫你補上！
