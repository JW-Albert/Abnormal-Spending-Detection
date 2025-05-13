# 消費異常偵測系統

## 系統簡介
本系統用於消費資料的異常偵測，支援每日消費資料分析，並自動產生多種特徵與異常分數，協助辨識異常消費行為。

## 系統架構與運作流程

### 1. 資料處理流程
- 從 `data` 目錄載入所有消費資料，依日期分組（每日一 block）。
- 將類別型資料（Location、Item、Category）轉換為數值編碼，並自動產生 mapping 檔（詳見 API/mapping 說明）。
- 日期資料轉換為 datetime 格式。

### 2. 特徵工程
- **時域特徵**：RMS、平均、標準差、峰峰值、峰度、偏度。
- **頻域特徵**：FFT 頻譜能量。
- **細緻行為特徵**：
  - 每日不同商品數、類別數、地點數
  - 高價商品比例、最大/最小/平均單品金額
  - 消費集中度（同一商品最大佔比）
  - 星期幾（週期性特徵）
- **補值策略**：特徵計算遇到全 0、全 NaN 或資料全相同時，該統計量自動補 0，避免 nan 影響模型。

### 3. 資料標準化與降維
- 使用 StandardScaler 進行特徵標準化。
- 應用 PCA 進行維度降低。
- 使用 LDA 進行監督式降維。

### 4. 異常偵測模型
- **Hotelling's T²**
- **SPE（Squared Prediction Error）**
- **One-Class SVM**
- **Isolation Forest**
- **Local Outlier Factor (LOF)**
- **EllipticEnvelope (Robust Covariance)**
- **Logistic Regression**（監督式基準模型）
- **AutoEncoder**：深度學習自編碼器，僅用正常樣本訓練，重建誤差高視為異常。
- **Ensemble（集成方法）**：將多模型異常分數標準化後加權平均，提升穩健性。

### 5. 模型評估與視覺化
- 評估指標：Accuracy、Precision、Recall、F1、ROC/AUC。
- 圖片輸出：每日消費比較、PCA/LDA 投影、各模型分數分布、模型比較條狀圖、Fisher Score、時域/頻域特徵分布等。
- 所有圖表保存在 `img` 目錄。
- 處理後的資料保存在 `data/merged.csv`。
- 模型比較表與圖已納入 AutoEncoder、Ensemble 等新方法。

### 6. API/mapping/ 設計
- `API/mapping/item.json`：商品名稱對應代碼。
- `API/mapping/location.json`：商店名稱對應代碼。
- `API/mapping/category.json`：商品類別對應代碼。
- `API/mapping/location_weight.json`：**以商店代碼為 key，對應該店的權重**。
- 所有 mapping 由 `get_include.py` 自動產生，前後端共用，確保一致性。

---

## 模型成效分析與改善建議

### 1. 分類報表（Logistic Regression）

```
              precision    recall  f1-score   support

           0     1.0000    1.0000    1.0000        14
           1     1.0000    1.0000    1.0000        59

    accuracy                         1.0000        73
   macro avg     1.0000    1.0000    1.0000        73
weighted avg     1.0000    1.0000    1.0000        73
```
- 監督式分類在測試集上表現極佳，但需注意資料分割與樣本分布，避免過擬合。

### 2. 各異常偵測模型比較

```
Anomaly Detection Models Comparison:
                  Accuracy  Precision    Recall        F1
Hotelling T²      0.17      0.37        0.02      0.04
SPE               0.23      0.89        0.06      0.11
One-Class SVM     0.13      0.00        0.00      0.00
Isolation Forest  0.18      0.47        0.03      0.06
LOF               ...       ...         ...       ...
EllipticEnvelope  ...       ...         ...       ...
AutoEncoder       ...       ...         ...       ...
Ensemble          ...       ...         ...       ...
```
- 新增 AutoEncoder、Ensemble 等方法，提升異常偵測多樣性與穩健性。
- 無監督模型 recall 普遍偏低，建議持續優化特徵與資料。

### 3. 圖片說明
- `img/model_comparison.png`：各模型指標條狀圖。
- `img/roc_auc.png`：監督式模型 ROC 曲線。
- `img/pca_projection.png`、`img/lda_visualization.png`：降維後資料分布。
- `img/ocsvm_scores.png`、`img/iso_forest_scores.png`、`img/lof_scores.png`、`img/elliptic_scores.png`、`img/t_squared.png`、`img/spe.png`：各異常分數分布。
- `img/fisher_score_top10.png`：Fisher Score 前 10 名特徵。
- `img/time_features.png`、`img/freq_features.png`：時域/頻域特徵分布。
- `img/daily_spending.png`：每日消費金額正常/異常對比。

### 4. 綜合結論與建議
- 特徵計算已修正 nan 問題，資料利用率提升。
- 建議持續增加異常樣本、優化特徵設計。
- 可考慮深度學習（AutoEncoder）、集成方法（Ensemble）、半監督學習等進階模型。
- 前後端請統一使用 API/mapping/ 下的 mapping 進行資料查詢與顯示。

---

如需針對每張圖做更細緻的說明，或想討論資料/特徵如何優化，請參考 img 資料夾內的圖檔，或聯絡開發者。 