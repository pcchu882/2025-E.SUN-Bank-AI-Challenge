# 2025 玉山人工智慧公開挑戰賽 – 警示帳戶預測

> 隊伍編號：TEAM_9686   
> 組別：學生組  
> 組長(組員)：褚柏均  
> 私有榜成績：F1 = 0.3849494  

本專案實作 2025 玉山人工智慧公開挑戰賽之「警示帳戶預測」任務，目標是根據帳戶交易紀錄，預測哪些帳戶會被標記為警示帳戶。  
整體方法為：

- 先將**交易紀錄聚合成帳戶層級特徵**（帳戶的流入/流出金額、筆數、對手數量、金額分布等）
- 再利用交易圖建構**帳戶關聯圖**（節點為帳戶、邊為交易關係）
- 以改良版 **GraphSAGE (SimpleSAGEv2)** 進行圖上的節點分類
- 最後以 **閾值 + Top-K F1 校準策略**，在驗證集上自動選擇較好的預測邏輯，投影到測試集生成最終 `result.csv` 提交檔

---
**結構說明**  
.  
├── preliminary_data/  
│   ├── acct_alert.csv    
│   ├── acct_predict.csv  
│   └── acct_transaction.csv        
├── main.py   
├── result.csv   
├── requirements.txt  
└── README.md     

preliminary_data/         # 存放Dataset位置  
result.csv                # 輸出預測結果  
main.py                   # 執行完整流程：前處理 → 訓練 → 產生 result.csv  
requirements.txt          # 套件需求  
README.md                 # 說明專案概述、環境、使用方式與實驗結果  



## 1. 環境需求

- Python：3.10.19

安裝所需套件
```bash
pip install -r requirements.txt
```

設定資料集 `preliminary_data/` ，確認裡面資料存在


## 2. 執行

```bash
python main.py
```
1. 讀取 ```acct_transaction.csv / acct_alert.csv / acct_predict.csv```
2. 產生帳戶層級特徵
3. 建立交易圖與節點索引
4. 訓練 GraphSAGEv2 模型，並在訓練資料中切出一部份做驗證
5. 在驗證集上搜尋最佳 Top-K 以最大化 F1，再外推至測試集
6. 在專案根目錄或指定位置輸出 ```result.csv```（欄位為 ```acct, label```）






# 方法概述
**帳戶特徵工程:**

對每一個帳戶，我們從交易紀錄中抽取下列特徵（主要實作在 build_account_features / build_account_features_v2）：  
金額相關：
- 總流出金額 ```total_send_amt```  
- 總流入金額 ```total_recv_amt```  
- 流出 / 流入金額的最大值、最小值、平均值

結構 / 連結相關：
- 不同收款對手數量 ```out_deg```  
- 不同付款對手數量 ```in_deg```  
- 流出交易筆數 ```out_tx_count```  
- 流入交易筆數 ```in_tx_count```
  
衍生指標：
- 淨流量 ```net_flow_amt = total_send_amt - total_recv_amt```  
- 淨流量比例 ```flow_ratio = net_flow_amt / (total_send_amt + total_recv_amt + eps)```  
- 每個對手的平均金額、每個對手的平均交易次數  
- 整體平均金額 ```mean_amt_overall```

高金額行為：
- 金額高於 90 / 99 百分位時的筆數與總額（分別對應流出/流入）  
時間相關（若資料有日期欄位）：  
- 不同時間窗（例如 7 / 30 / 90 天）的交易筆數與金額累積  
- 最近一次流出 / 流入距今天的天數（recency）

其它：
- 帳戶是否為玉山帳戶 ```is_esun```  
- 針對金額 / 筆數等欄位進行 ```log1p``` 變換，以穩定數值分布  
- 對連續特徵進行分位數裁切（如 ```clip_quantile=0.999```）以降低極端值影響  
  
最後，我們會對訓練節點的特徵做標準化（StandardScaler），並用同一組 scaler 轉換其餘節點，數值裁切到合理範圍（例如 [-5, 5]）。  

---
**圖結構建模**  

節點：帳戶（以 ```acct``` 為 key 建立全體帳戶的 index）  
邊：帳戶間發生過至少一次交易，即建立一個無向邊  
權重 / 正規化：  
- 對每一個節點，將其外出邊權重正規化為 ```1 / out_degree```  
- 最終得到 row-normalized 的稀疏 adjacency matrix A（PyTorch sparse tensor）  

---
**模型架構 – SimpleSAGEv2**

核心模型為 SimpleSAGEv2，是一個多層的 GraphSAGE 變形，主要設計如下：
多層 _SAGEBlock 堆疊（預設 depth=3），每層包含：  
- 自身線性轉換 ```lin_self```  
- 鄰居訊息聚合 ```lin_neigh``` + ```A @ x```   
- LayerNorm + GELU + Dropout  

殘差連接：
- 若前後維度相同，加入殘差 ```h = h + h_new```，有助於穩定深層訓練

Jumping Knowledge：
- ```jk="cat"``` 時，會將各層輸出串接後再送入最終線性層，有助於融合不同層次的圖訊息

DropEdge：
- 訓練階段對 adjacency 以機率 ```dropedge_p``` 隨機丟邊，再依比例縮放，有助於正則化與避免 overfitting

最終輸出：
- 線性層將節點表示映射到 scalar logit，代表該帳戶為警示帳戶的傾向

---
**訓練策略**

訓練流程主要在 ```train_gnn``` 中完成，重點如下：

損失函數：
- 使用 ```BCEWithLogitsLoss```，並針對正負樣本不平衡設定 ```pos_weight = N_neg / N_pos```

資料切分：
- 僅使用「非預測清單」且「is_esun==1」的帳戶作為訓練資料
- 透過 ```StratifiedKFold``` 將訓練節點切成 train / val（例如 5 折中取一折為驗證）

超參數（實際使用設定）：
- hidden 維度：160
- depth：3
- dropout：0.3
- dropedge_p：0.1
- learning rate：1e-3
- weight decay：1e-4
- epochs：500

早停：
- 以驗證集 F1 作為監控指標，若連續若干 epoch 未提升則 early stop

評估：
- 每幾個 epoch 計算一次驗證集 F1，並保存最佳權重

---
**預測與 Top-K 校準**

在得到所有節點的預測機率之後，我們針對驗證集進行兩種策略比較：

固定閾值法：
- 使用預設閾值（例如 ```thr=0.5```）將機率轉為 0/1 標籤
- 計算 F1 分數

Top-K 搜索法：
- 依機率遞減排序，從 K=1 ～ N 逐一測試
- 對每個 K 計算相對應的 TP / FP / FN，進而算出 F1
- 選出使得 F1 最佳的 ```K_val*```，同時記錄此時的 F1

接著，我們將驗證集最佳 K 投影到測試集大小：
```text
K_test = ceil( (K_val* / N_val) * N_test * alpha )
```

其中 ```alpha``` 為調整係數（預設 1.0，可用於保守或激進預測），最後在測試集上擇一策略輸出：
- 若模式為 ```topk_mode="fallback"```：
- 先用閾值法，如果預測的陽性數量過少（低於 ```min_pos_pred```），則改用 Top-K
- 其它模式則可強制使用 Top-K 或閾值（程式中已支援）

