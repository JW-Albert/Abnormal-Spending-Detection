# 消費異常偵測系統

## 系統簡介
本系統可用於消費資料的異常偵測，支援單筆/多筆消費資料輸入，並可自動分析是否有異常消費行為。

## 系統架構與運作流程

### 1. 資料處理流程
1. **資料載入與預處理**
   - 從 `data` 目錄載入所有消費資料
   - 將類別型資料（Location、Item、Category）轉換為數值編碼
   - 日期資料轉換為 datetime 格式

2. **特徵工程**
   - **時域特徵**：計算每個數值欄位的統計量
     - RMS（均方根值）
     - 平均值
     - 標準差
     - 峰峰值（Peak-to-Peak）
     - 峰度（Kurtosis）
     - 偏度（Skewness）
   
   - **頻域特徵**：使用 FFT 進行頻譜分析
     - 計算主要頻率成分的振幅
     - 分析不同頻率區間的訊號強度

3. **資料標準化與降維**
   - 使用 StandardScaler 進行特徵標準化
   - 應用 PCA 進行維度降低
   - 使用 LDA 進行監督式降維

### 2. 異常偵測模型

1. **Hotelling's T² 統計量**
   - 原理：計算樣本點到主成分空間的馬氏距離
   - 用途：檢測樣本是否偏離正常分布
   - 閾值：使用 95% 分位數作為異常判定標準

2. **SPE（Squared Prediction Error）**
   - 原理：計算原始資料與重建資料的誤差平方和
   - 用途：檢測模型無法解釋的變異
   - 閾值：同樣使用 95% 分位數

3. **One-Class SVM**
   - 原理：在特徵空間中建立一個超平面，將正常樣本與原點分開
   - 特點：適合處理高維資料，對異常樣本敏感

4. **Isolation Forest**
   - 原理：隨機選擇特徵和切分點，建立決策樹
   - 特點：計算效率高，適合處理大規模資料

5. **邏輯回歸（Logistic Regression）**
   - 用途：作為基準模型，評估特徵的有效性
   - 特點：可解釋性強，訓練速度快

### 3. 模型評估與視覺化

1. **評估指標**
   - 準確率（Accuracy）
   - 精確率（Precision）
   - 召回率（Recall）
   - F1 分數
   - ROC 曲線與 AUC 值

2. **視覺化輸出**
   - 每日消費比較圖
   - PCA 投影圖
   - LDA 視覺化
   - 時域特徵分布圖
   - 頻域特徵分布圖
   - 模型比較圖表

### 4. 系統輸出
- 所有圖表保存在 `img` 目錄
- 處理後的資料保存在 `data/merged.csv`
- 模型評估報告包含詳細的分類指標

---

## 模型成效分析與改善建議

### 1. 分類報表（classification_report）

```
              precision    recall  f1-score   support

           0     1.0000    0.8571    0.9231         7
           1     0.9630    1.0000    0.9811        26

    accuracy                         0.9697        33
   macro avg     0.9815    0.9286    0.9521        33
weighted avg     0.9708    0.9697    0.9688        33
```

- 監督式分類（Logistic Regression）在測試集上表現良好，但異常樣本（0）僅 7 筆，資料仍有不平衡現象。
- 高分不代表模型真的有辨識異常能力，需注意過擬合與資料分布。

### 2. 各異常偵測模型比較

```
Anomaly Detection Models Comparison:
                  Accuracy  Precision    Recall        F1
Hotelling T²      0.181818   0.222222  0.015385  0.028777
SPE               0.230303   0.666667  0.046154  0.086331
One-Class SVM     0.157576   0.000000  0.000000  0.000000
Isolation Forest  0.193939   0.333333  0.023077  0.043165
```
- 無監督異常偵測模型在本資料集上表現普遍不佳，recall 與 F1 score 極低。
- SPE 的 precision 較高，代表預測為異常時較準，但 recall 低，漏掉大部分異常。
- OCSVM 幾乎無法偵測異常。

### 3. 圖片說明
- `img/model_comparison.png`：各模型指標條狀圖，直觀比較模型表現。
- `img/roc_auc.png`：監督式模型 ROC 曲線，AUC 越高代表分類能力越好。
- `img/pca_projection.png`、`img/lda_visualization.png`：降維後資料分布，若異常/正常混雜，代表特徵區分力有限。
- `img/ocsvm_scores.png`、`img/iso_forest_scores.png`、`img/t_squared.png`、`img/spe.png`：各異常分數分布與閾值線。
- `img/fisher_score_top10.png`：Fisher Score 前 10 名特徵，顯示哪些特徵對分類最有貢獻。
- `img/time_features.png`、`img/freq_features.png`：時域/頻域特徵分布。
- `img/daily_spending.png`：每日消費金額正常/異常對比。

### 4. 綜合結論與建議
- 目前資料特徵對異常與正常的區分力有限，尤其是無監督模型效果不佳。
- 監督式模型在測試集上表現佳，但異常樣本數量仍偏少，指標易受影響。
- 建議持續增加異常樣本、優化特徵設計，並可嘗試更多異常偵測模型（如 LOF、AutoEncoder 等）。
- 針對異常樣本進行特徵工程加強，並可考慮半監督學習、集成方法等。

### 5. 可擴充的分析模型建議
- **Local Outlier Factor (LOF)**：基於鄰近密度的異常偵測。
- **Elliptic Envelope (Robust Covariance)**：適合多變量常態分布資料。
- **AutoEncoder/Deep Learning**：用於高維資料的非線性特徵學習。
- **Ensemble Methods**：結合多種異常偵測模型提升穩健性。
- **Prophet/ARIMA**：若有時間序列特性，可用於異常點偵測。

---

如需針對每張圖做更細緻的說明，或想討論資料/特徵如何優化，請參考 img 資料夾內的圖檔，或聯絡開發者。 