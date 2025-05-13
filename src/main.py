import warnings
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from get_include import get_all_data

from scipy.stats import kurtosis, skew
from scipy.fft import fft
from scipy.signal import hilbert
import pywt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.exceptions import ConvergenceWarning
import joblib

# 靜音收斂警告
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# 目錄設置
DATA_DIR = 'data'
IMG_DIR = 'img'
os.makedirs(IMG_DIR, exist_ok=True)

# --------------------------------------------------
# 1. 載入資料並加上 Label
# --------------------------------------------------
df_all = get_all_data('data')
print(f"Loaded total rows={len(df_all)}")

# --------------------------------------------------
# 2. 資料簡化與標籤編碼
# --------------------------------------------------
def simplify_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ['Location','Item','Category']:
        df[col + '_Code'] = pd.Categorical(df[col]).codes
    return df

df_all = simplify_columns(df_all)

# --------------------------------------------------
# 3. 每日消費比較圖
# --------------------------------------------------
if 'Date' in df_all.columns:
    df_all['Date'] = pd.to_datetime(df_all['Date'])
    daily = df_all.groupby(['Date','Label'])['Total Daily Spending'].sum().unstack(fill_value=0)
    plt.figure(figsize=(12,5))
    plt.plot(daily.index, daily[1], label='Normal',   color='blue')
    plt.plot(daily.index, daily[0], label='Abnormal', color='red')
    plt.title('Daily Spending Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR,'daily_spending.png'))
    plt.close()

# --------------------------------------------------
# 4. 時域特徵函式
# --------------------------------------------------
def time_domain(df: pd.DataFrame, cols: list, unit: int) -> pd.DataFrame:
    ops = {
        '_rms':       lambda x: np.sqrt((x**2).mean()),
        '_mean':      np.mean,
        '_std':       np.std,
        '_ptp':       lambda x: np.ptp(x),
        '_kurtosis':  kurtosis,
        '_skewness':  skew
    }
    feature_names = [f"{col}{suf}" for col in cols for suf in ops]
    records = []
    for i in range(0, len(df), unit):
        block = df.iloc[i:i+unit]
        vals  = []
        for fn in ops.values():
            for col in cols:
                vals.append(fn(block[col].values))
        records.append(vals)
    return pd.DataFrame(records, columns=feature_names)

# --------------------------------------------------
# 5. 頻域特徵函式
# --------------------------------------------------
def frequency_domain(df: pd.DataFrame, cols: list, fs: int, base: int, n: int, unit: int) -> pd.DataFrame:
    records = []
    for i in range(0, len(df), unit):
        block = df.iloc[i:i+unit]
        L     = len(block)
        if L == 0: break
        dfreq = fs / L
        half  = L // 2
        freqs = np.arange(half) * dfreq
        fftv  = np.abs(fft(block[cols].values, axis=0))[:half]
        row   = []
        for j,col in enumerate(cols):
            mag = fftv[:,j]
            for k in range(1, n+1):
                tgt  = base * k
                mask = (freqs>=tgt-8)&(freqs<=tgt+8)
                row.append(float(mag[mask].max()) if mask.any() else 0.0)
        records.append(row)
    feature_names = [f"{col}_freq_{k}" for col in cols for k in range(1,n+1)]
    return pd.DataFrame(records, columns=feature_names)

# --------------------------------------------------
# 6. Fisher Score 函式
# --------------------------------------------------
def fisher_score(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    scores = []
    for i in range(X.shape[1]):
        m0, m1 = X[y==0,i].mean(), X[y==1,i].mean()
        v0, v1 = X[y==0,i].var(),  X[y==1,i].var()
        scores.append((m0-m1)**2/(v0+v1+1e-6))
    return np.array(scores)

# --------------------------------------------------
# 7. 計算時頻域特徵並繪圖
# --------------------------------------------------
num_cols = ['Price','Quantity','Location Weight','Abnormal Score','Total Daily Spending']
time_df  = time_domain(df_all, num_cols, unit=10)
freq_df  = frequency_domain(df_all, num_cols, fs=1, base=1, n=3, unit=10)

# 新增：合併地點權重
location_weights = []
for i in range(0, len(df_all), 10):
    blk = df_all.iloc[i:i+10]
    if 'Weight' in blk.columns:
        mode_weight = blk['Weight'].mode()
        if not mode_weight.empty:
            location_weights.append(mode_weight.iloc[0])
        else:
            location_weights.append(blk['Weight'].iloc[0])
    else:
        location_weights.append(0)

features = pd.concat([time_df, freq_df], axis=1)
features['Location_Weight'] = location_weights

# 時域首筆示意圖
plt.figure(figsize=(14,4))
sns.barplot(x=time_df.columns, y=time_df.iloc[0].values)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR,'time_features.png'))
plt.close()

# 頻域首筆示意圖
plt.figure(figsize=(14,4))
sns.barplot(x=freq_df.columns, y=freq_df.iloc[0].values)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR,'freq_features.png'))
plt.close()

# --------------------------------------------------
# 8. 合併 block‐level 標籤
# --------------------------------------------------
labels = []
for i in range(0, len(df_all), 10):
    blk = df_all.iloc[i:i+10]
    labels.append(int(blk['Label'].mode()[0]))
labels = np.array(labels)

# 這裡自動將 NaN 補 0
if features.isna().any().any():
    print("\n=== 以下 block-level 特徵有 NaN，將自動補 0 ===")
    for idx, row in features[features.isna().any(axis=1)].iterrows():
        nan_cols = row.index[row.isna()].tolist()
        print(f"Block {idx} 缺失欄位: {nan_cols}")
        block = df_all.iloc[idx*10:idx*10+10]
        for i, orig_row in block.iterrows():
            missing = orig_row.isna()
            if missing.any():
                miss_cols = orig_row.index[missing].tolist()
                print(f"  原始資料 index {i} 缺失欄位: {miss_cols}")
    features = features.fillna(0)

# 產生 merged.csv 用 dropna 後的 features/labels
merged = features.copy()
merged['Label'] = labels
merged.to_csv(os.path.join(DATA_DIR,'merged.csv'), index=False)

# --------------------------------------------------
# 9. 標準化＋PCA
# --------------------------------------------------
X     = StandardScaler().fit_transform(features)
y     = labels
pca   = PCA(n_components=2).fit(X)
X_pca = pca.transform(X)

plt.figure(figsize=(8,5))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='coolwarm', alpha=0.7)
plt.title('PCA Projection')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR,'pca_projection.png'))
plt.close()

# --------------------------------------------------
# 10. Hotelling's T² & SPE
# --------------------------------------------------
t2    = np.sum((X_pca/np.std(X_pca,0))**2, axis=1)
t2th  = np.percentile(t2,95)
spe   = np.sum((X - pca.inverse_transform(X_pca))**2, axis=1)
speth = np.percentile(spe,95)

plt.figure(figsize=(10,4))
plt.plot(t2, label="T²")
plt.axhline(t2th, ls='--', c='red', label='95%')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR,'t_squared.png'))
plt.close()

plt.figure(figsize=(10,4))
plt.plot(spe, label="SPE")
plt.axhline(speth, ls='--', c='red', label='95%')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR,'spe.png'))
plt.close()

# --------------------------------------------------
# 11. LDA 可視化
# --------------------------------------------------
classes = np.unique(y)
if len(classes) > 1:
    lda_n  = min(len(classes)-1, X.shape[1])
    lda    = LDA(n_components=lda_n).fit(X, y)
    X_lda  = lda.transform(X)
    plt.figure(figsize=(8,5))
    if lda_n == 1:
        plt.hist(X_lda[y==1], bins=30, alpha=0.6, label='Normal',   color='blue')
        plt.hist(X_lda[y==0], bins=30, alpha=0.6, label='Abnormal', color='red')
        plt.xlabel('LDA1')
    else:
        plt.scatter(X_lda[:,0], X_lda[:,1], c=y, cmap='coolwarm', alpha=0.7)
        plt.xlabel('LDA1'); plt.ylabel('LDA2')
    plt.title('LDA Projection')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR,'lda_visualization.png'))
    plt.close()
else:
    print("LDA: 只有一類，跳過")

# --------------------------------------------------
# 12. Logistic Regression 邊界 + 評估 + ROC/AUC + classification_report
# --------------------------------------------------
if len(classes) > 1:
    # 決策邊界 (PCA 空間)
    lr = LogisticRegression(max_iter=5000, class_weight='balanced').fit(X_pca, y)
    x0_min,x0_max = X_pca[:,0].min()-1, X_pca[:,0].max()+1
    x1_min,x1_max = X_pca[:,1].min()-1, X_pca[:,1].max()+1
    xx,yy = np.meshgrid(np.linspace(x0_min,x0_max,200),
                        np.linspace(x1_min,x1_max,200))
    Z     = lr.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.figure(figsize=(8,6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='coolwarm', edgecolors='k')
    plt.title('Logistic Boundary')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR,'logistic_boundary.png'))
    plt.close()

    # cross‐val 評估＋ROC
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          stratify=y, random_state=42)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    clf2  = LogisticRegression(max_iter=5000, class_weight='balanced').fit(Xtr_s, ytr)
    ypred = clf2.predict(Xte_s)
    yprob = clf2.predict_proba(Xte_s)[:,1]

    # classification report
    print(classification_report(yte, ypred, digits=4))

    # ROC/AUC
    fpr, tpr, _ = roc_curve(yte, yprob)
    roc_auc     = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'AUC={roc_auc:.3f}')
    plt.plot([0,1],[0,1],'--', c='gray')
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR,'roc_auc.png'))
    plt.close()

# --------------------------------------------------
# 13. Fisher Score Top10
# --------------------------------------------------
fs     = fisher_score(X, y)
idx    = np.argsort(fs)[::-1][:10]
fnames = features.columns[idx]
fvals  = fs[idx]

plt.figure(figsize=(10,5))
plt.bar(fnames, fvals, color=plt.cm.viridis(np.linspace(0,1,10)))
plt.xticks(rotation=45)
plt.title('Top10 Fisher Scores')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR,'fisher_score_top10.png'))
plt.close()

# --------------------------------------------------
# 14. One-Class SVM
# --------------------------------------------------
# 只使用正常樣本訓練
X_normal = X[y == 1]
ocsvm = OneClassSVM(kernel='rbf', nu=0.1).fit(X_normal)
ocsvm_scores = -ocsvm.score_samples(X)  # 負分數表示異常程度

plt.figure(figsize=(10,4))
plt.plot(ocsvm_scores, label="One-Class SVM Score")
plt.axhline(np.percentile(ocsvm_scores, 95), ls='--', c='red', label='95%')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR,'ocsvm_scores.png'))
plt.close()

# --------------------------------------------------
# 15. Isolation Forest
# --------------------------------------------------
iso_forest = IsolationForest(contamination=0.1, random_state=42).fit(X)
iso_scores = -iso_forest.score_samples(X)  # 負分數表示異常程度

plt.figure(figsize=(10,4))
plt.plot(iso_scores, label="Isolation Forest Score")
plt.axhline(np.percentile(iso_scores, 95), ls='--', c='red', label='95%')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR,'iso_forest_scores.png'))
plt.close()

# --------------------------------------------------
# 16. 模型比較
# --------------------------------------------------
# 計算各個模型的異常分數
scores = {
    'Hotelling T²': t2,
    'SPE': spe,
    'One-Class SVM': ocsvm_scores,
    'Isolation Forest': iso_scores
}

# 計算各個模型的異常檢測結果
thresholds = {name: np.percentile(score, 95) for name, score in scores.items()}
predictions = {name: score > thresholds[name] for name, score in scores.items()}

# 計算各個模型的評估指標
results = {}
for name, pred in predictions.items():
    results[name] = {
        'Accuracy': accuracy_score(y, pred),
        'Precision': precision_score(y, pred),
        'Recall': recall_score(y, pred),
        'F1': f1_score(y, pred)
    }

# 將結果轉換為DataFrame並顯示
results_df = pd.DataFrame(results).T
print("\nAnomaly Detection Models Comparison:")
print(results_df)

# 繪製模型比較圖
plt.figure(figsize=(12,6))
results_df.plot(kind='bar', ax=plt.gca())
plt.title('Model Comparison')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR,'model_comparison.png'))
plt.close()

# --------------------------------------------------
# 17. 保存模型
# --------------------------------------------------
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# 保存模型
joblib.dump(ocsvm, os.path.join(MODEL_DIR, 'ocsvm_model.pkl'))
joblib.dump(iso_forest, os.path.join(MODEL_DIR, 'iso_forest_model.pkl'))
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))

print("\nModels saved successfully!")
