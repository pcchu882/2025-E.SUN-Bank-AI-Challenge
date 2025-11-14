
# Model 模組 – GNN 模型架構與訓練流程  


本資料夾包含本專案所有與 **圖神經網路（GNN）模型** 相關的程式碼。  
模型負責根據帳戶的交易特徵與帳戶之間的關聯圖，預測哪些帳戶可能為「警示帳戶」。

所有功能皆由 `main.py` 自動呼叫。

---

## 1. 建立帳戶關聯圖（Graph Construction）

由 `build_graph()` 建立：

- 節點（Node）代表帳戶
- 邊（Edge）代表兩帳戶之間有交易紀錄
- 建構無向圖（undirected graph）
- 對邊做 row-normalization，適合後續 GNN 訊息傳遞

---

## 2. DropEdge 正則化

`dropedge()` 在訓練階段隨機丟棄部分邊，並對權重做縮放：

- 提升泛化能力
- 降低 overfitting
- 特別適合金融類圖資料（邊密度高但噪音多）

---

## 3. GNN 模型架構 – SimpleSAGEv2

本次提交的最終版本使用：

### **SimpleSAGEv2（競賽最佳模型）**

特點：

- 3 層 GraphSAGE block（預設 depth=3）
- LayerNorm + GELU 加速收斂、提升穩定性
- 殘差連接（Residual Connection）
- Jumping Knowledge（JK-cat）：融合不同層的資訊
- DropEdge
- 最後以 Linear 層輸出每個節點的 logit（預測分數）

該模型在本次比賽資料上表現最佳（Private LB）。

---

## 4. 訓練流程（train_gnn）

`train_gnn()` 內包含：

- BCEWithLogitsLoss  
- 自動計算 imbalance 用的 pos_weight  
- Stratified KFold 產生 train/val split  
- Validation 以 F1 作為 early stopping 判定  
- Gradient clipping  
- 儲存最佳模型狀態  

此流程可穩定在私榜上產生接近最優的結果。

---

## 5. Top-K F1 校準策略

競賽的警示帳戶比例極低，僅用 threshold 會導致：

- 太少的陽性
- F1 下降

因此使用：

### 1. `best_topk_by_f1()`  
在驗證集上找出使 F1 最大的 K。

### 2. `project_k_from_val_to_test()`  
將驗證集的最佳 K 外推到測試集：

