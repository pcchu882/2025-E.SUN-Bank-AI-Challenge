# Preprocess 模組 – 資料前處理與特徵工程

本資料夾包含本專案所有與 **資料前處理（Preprocessing）** 相關的程式碼。所有流程皆會由 `main.py` 自動呼叫，使用者不需要手動執行任何前處理腳本。

本模組負責以下工作：

1. 讀取官方 CSV 資料
2. 自動解析不同欄位命名（欄位對應）
3. 帳戶層級特徵工程（Feature Engineering）
4. 特徵標準化（避免資料洩漏）

---

## 1. 載入官方資料集

競賽資料包含三個主要 CSV：

* `acct_transaction.csv`：完整交易紀錄
* `acct_alert.csv`：被標記為警示帳戶的資料
* `acct_predict.csv`：需預測之帳戶名單

透過以下函式自動載入：

```python
tx_df, alert_df, predict_df = read_competition_csvs(data_dir)
```

---

## 2. 自動對應欄位名稱（避免不同命名造成錯誤）

官方資料在不同版本可能使用不同欄位名稱，例如：

| 欄位語意 | 可能名稱                                              |
| ---- | ------------------------------------------------- |
| 付款帳戶 | from_acct / payer_acct_id / src / source_account  |
| 收款帳戶 | to_acct / receiver_acct_id / dst / target_account |
| 交易金額 | txn_amt / amount / money / trade_amount           |
| 帳戶欄位 | acct / acct_id / account_id / account             |

本模組使用：

```python
col_map = infer_column_mapping(tx_df, alert_df, predict_df)
```

自動完成：

* 完全比對
* 不分大小寫比對
* 字串包含比對（substring）

最終會輸出統一的欄位名稱 `col_map`，確保整個 Pipeline 不因欄位命名差異而出錯。

---

## 3. 帳戶層級特徵工程（Account-Level Feature Engineering）

使用以下函式進行：

```python
feat_df = make_account_features_basic(tx_df, col_map)
```

將所有交易資料由「交易層級」轉換成「帳戶層級」，每個帳戶一筆特徵。包含：

### (A) 流出特徵（付款方）

* out_amt_sum：總流出金額
* out_amt_max / min / mean
* out_partner_cnt：不同收款對手數量
* out_tx_cnt：流出交易筆數

### (B) 流入特徵（收款方）

* in_amt_sum：總流入金額
* in_amt_max / min / mean
* in_partner_cnt：不同付款對手數量
* in_tx_cnt：流入交易筆數

### (C) 帳戶屬性

* is_esun：是否為玉山帳戶（若資料無此欄位 → 預設全部為玉山帳戶）

### (D) log1p 平滑特徵（穩定模型、降低極端值影響）

會自動生成：

* log1p_out_amt_sum
* log1p_in_amt_sum
* log1p_out_tx_cnt
* …等所有金額與筆數的 log1p 版本

---

## 4. 特徵標準化（Standardization + 不洩漏資料）

模型訓練前，需對所有帳戶特徵進行標準化，但**不能洩漏驗證或測試資訊**。

使用：

```python
X_all, scaler = normalize_features_by_train(X_all, train_index)
```

流程：

1. **僅使用訓練節點** fit StandardScaler（避免 Data Leakage）
2. 套用到所有帳戶特徵
3. 對非訓練節點 clip 到 `[-5, 5]`（避免極端值影響 GNN）

---

## 檔案說明

| 檔案                   | 說明                     |
| -------------------- | ---------------------- |
| `data_preprocess.py` | 主要前處理模組（欄位解析、特徵工程、標準化） |

---

## 與 main.py 的整合流程

以下程式會在 `main.py` 自動執行：

```python
from Preprocess.data_preprocess import (
    read_competition_csvs,
    infer_column_mapping,
    make_account_features_basic,
    normalize_features_by_train,
)
```

整個資料前處理 Pipeline（讀檔 → 欄位解析 → 特徵工程 → 標準化）皆會由 `main.py` 自動串接完成。
