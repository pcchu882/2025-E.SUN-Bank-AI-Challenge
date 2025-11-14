# Preprocess 模組 – 資料前處理與特徵工程  


本資料夾包含本專案所有與「資料前處理」相關的程式碼。  
所有功能皆由 `main.py` 自動呼叫，不需要額外手動操作。

本模組主要負責以下內容：

---

## 1. 載入官方提供的資料集

包含以下三個 CSV：

- `acct_transaction.csv`（完整交易記錄）
- `acct_alert.csv`（有被標記為警示帳戶的資料）
- `acct_predict.csv`（需預測的帳戶名單）

由 `load_csvs()` 自動讀取。

---

## 2. 統一欄位名稱（自動對應不同命名）

官方資料可能使用不同欄位命名，例如：

- `from_acct` / `payer_acct_id` / `source_account`
- `txn_amt` / `amount` / `trade_amount`
- `acct_id` / `account_id` / `acct`

`resolve_columns()` 會自動偵測並標準化成固定欄位名稱，確保整個 pipeline 都能正常運作。

---

## 3. 建立帳戶層級特徵（Feature Engineering）

從交易紀錄（transaction-level）聚合成帳戶層級（account-level），包含：

### 金額相關
- 總流出金額
- 總流入金額
- 流入/流出金額的最大值、最小值、平均值

### 結構 / 連結特徵
- 不同收款對手數量（out_deg）
- 不同付款對手數量（in_deg）
- 流出/流入交易筆數

### 衍生特徵
- log1p() 變換（穩定分布）
- E.SUN 帳戶 vs 非 E.SUN 帳戶指標（is_esun）

最終特徵 DataFrame 由 `build_account_features()` 輸出。

---

## 4. 標準化（Standardization）

`standardize_features()` 使用 **僅訓練資料(train_idx)** 來 fit StandardScaler，  
並將同一組 scaler 套用至所有節點（避免資料洩漏 Data Leakage）。

同時會把輸出 clip 到 `[-5, 5]` 防止極端值影響模型訓練。

---

## 檔案說明

| 檔案 | 用途 |
|------|------|
| **`data_preprocess.py`** | 提供前處理流程（載入資料、特徵工程、欄位解析、標準化） |

---

## 與 main.py 的整合流程

`main.py` 中會執行：

```python
from Preprocess.data_preprocess import (
    load_csvs,
    resolve_columns,
    build_account_features,
    standardize_features
)
