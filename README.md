# 2025 玉山人工智慧公開挑戰賽 – 警示帳戶預測

> **隊伍編號：TEAM_9686**
> **組別：學生組**
> **組長：褚柏均**
> **Private Leaderboard F1：0.3849494**

本專案為 2025 玉山人工智慧公開挑戰賽「警示帳戶預測」初賽資格審查所需之完整程式碼，包含：

* **資料前處理**（欄位解析、特徵工程、標準化）
* **帳戶關聯圖建構（Account Graph）**
* **圖神經網路（GNN, SimpleSAGEv2）訓練與推論**
* **閾值與 Top-K F1 校準策略**
* 一鍵執行產生 `result.csv`

所有流程由 `main.py` 串接，評審者可直接執行並重現結果。

---

# 1. 專案結構

```
.
├── Preprocess/
│   ├── data_preprocess.py     # 資料解析、特徵工程、標準化
│   └── README.md              # Preprocess 模組說明
│
├── Model/
│   ├── gnn_model.py           # 建圖、GNN 模型、訓練、Top-K
│   └── README.md              # Model 模組說明
│
├── preliminary_data/          # 官方提供資料 (未上傳)
│   ├── acct_transaction.csv
│   ├── acct_alert.csv
│   └── acct_predict.csv
│
├── main.py                    # 主流程：前處理 → 建圖 → 訓練 → 推論
├── requirements.txt           # 套件需求
├── result.csv                 # 執行結果範例
└── README.md                  # 主說明文件（本檔）
```

---

# 2. 執行環境

* Python **3.10.19**

安裝必要套件：

```bash
pip install -r requirements.txt
```

請將官方資料放入：

```
preliminary_data/
```

並確認三個 CSV 皆存在。

---

# 3. 一鍵執行

```bash
python main.py
```

流程包含：

1. 讀取 `acct_transaction.csv / acct_alert.csv / acct_predict.csv`
2. 特徵工程：將交易紀錄聚合成帳戶特徵
3. 建立帳戶關聯圖（Sparse Graph）
4. 訓練 GraphSAGEv2
5. 在驗證集比較：threshold vs Top-K
6. 在測試集套用最佳校準策略
7. 輸出 `result.csv`（格式：acct, label）

---

# 4. 方法概述

## 4.1 帳戶特徵工程

來自交易資料（amount / degree / counts）的帳戶層級特徵：

* 流出金額：sum / max / min / mean
* 流入金額：sum / max / min / mean
* 流入、流出對手數量（in/out degree）
* 流入、流出交易筆數
* 是否為玉山帳戶（若無欄位則預設為 1）
* 所有金額/筆數加入 `log1p()` 特徵（如 `log1p_out_amt_sum`）

並以 **訓練節點資料 fit StandardScaler**，避免資料洩漏，再將所有節點標準化到合理範圍（含 clip [-5,5]）。

---

## 4.2 帳戶關聯圖（Account Graph）

* 每個**帳戶為一個節點**
* 若兩帳戶曾有交易 → 形成**無向邊**
* 進行 row-normalization，權重為：

  ```
  1 / out_degree
  ```

並於訓練期間支援 **DropEdge** 正則化，降低 overfitting。

---

## 4.3 模型：SimpleSAGEv2（最佳競賽模型）

* 多層 GraphSAGE-style Layer
* LayerNorm + GELU + Dropout
* Residual 殘差連接
* Jumping Knowledge (JK-cat)
* DropEdge during training
* 最終輸出節點 logit → sigmoid → 機率

此模型在本競賽資料上表現最佳。

---

## 4.4 訓練策略

* Loss：`BCEWithLogitsLoss`
* pos_weight 自動平衡類別不均
* 5-fold Stratified Split（取 1 fold 作為驗證）
* 以驗證 F1 作 early stopping 依據
* 儲存最佳權重

---

## 4.5 預測與 Top-K 校準

驗證階段計算：

### ① Threshold（thr=0.5）

直接以 0.5 二值化，但通常陽性太少。

### ② Top-K 搜尋

逐一測 K＝1…N，找出：

```
K_val = argmax_F1(K)
```

並外推到 test：

```
K_test = ceil((K_val / N_val) * N_test * alpha)
```

若 threshold 預測陽性過低 → fallback 改用 Top-K（本次提交使用）。

---
