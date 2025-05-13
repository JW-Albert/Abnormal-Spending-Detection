import warnings
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lib.get_include import get_all_data

from scipy.stats import kurtosis, skew
from scipy.fft import fft
from scipy.signal import hilbert
import pywt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import OneClassSVM, SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.exceptions import ConvergenceWarning
import joblib
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.covariance import EllipticEnvelope
from lib.autoencoder_model import get_autoencoder_scores
from lib.ensemble import get_ensemble_score
import json
from sklearn.tree import DecisionTreeClassifier

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
def safe_stat(fn, arr):
    try:
        if np.all(np.isnan(arr)) or np.all(arr == arr[0]):
            return 0.0
        val = fn(arr)
        if np.isnan(val) or np.isinf(val):
            return 0.0
        return val
    except Exception:
        return 0.0

def time_domain(df: pd.DataFrame, cols: list, unit: int) -> pd.DataFrame:
    ops = {
        '_rms':       lambda x: safe_stat(lambda y: np.sqrt((y**2).mean()), x),
        '_mean':      lambda x: safe_stat(np.mean, x),
        '_std':       lambda x: safe_stat(np.std, x),
        '_ptp':       lambda x: safe_stat(np.ptp, x),
        '_kurtosis':  lambda x: safe_stat(kurtosis, x),
        '_skewness':  lambda x: safe_stat(skew, x)
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
        try:
            fftv  = np.abs(fft(block[cols].values, axis=0))[:half]
        except Exception:
            fftv = np.zeros((half, len(cols)))
        row   = []
        for j,col in enumerate(cols):
            mag = fftv[:,j] if L else np.zeros(half)
            for k in range(1, n+1):
                tgt  = base * k
                mask = (freqs>=tgt-8)&(freqs<=tgt+8)
                try:
                    row.append(float(mag[mask].max()) if mask.any() else 0.0)
                except Exception:
                    row.append(0.0)
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
# 7. 計算每日特徵並繪圖
# --------------------------------------------------
if 'Date' in df_all.columns:
    df_all['Date'] = pd.to_datetime(df_all['Date'])
    df_all = df_all.sort_values('Date')
    grouped = list(df_all.groupby('Date'))
else:
    grouped = [(None, df_all)]

# 商品、地點、類別編碼
for col in ['Location','Item','Category']:
    df_all[col + '_Code'] = pd.Categorical(df_all[col]).codes

# 計算每日特徵並繪圖
num_cols = ['Price','Quantity','Location Weight','Abnormal Score','Total Daily Spending','Item_Code']
time_records, freq_records, location_weights, labels = [], [], [], []
extra_features = []
for date, blk in grouped:
    # 時域特徵
    time_row = []
    for fn in [lambda x: np.sqrt((x**2).mean()), np.mean, np.std, np.ptp, kurtosis, skew]:
        for col in num_cols:
            time_row.append(fn(blk[col].values))
    time_records.append(time_row)
    # 頻域特徵
    L = len(blk)
    dfreq = 1 / L if L else 1
    half = L // 2
    freqs = np.arange(half) * dfreq
    fftv = np.abs(fft(blk[num_cols].values, axis=0))[:half] if L else np.zeros((half, len(num_cols)))
    freq_row = []
    for j, col in enumerate(num_cols):
        mag = fftv[:,j] if L else np.zeros(half)
        for k in range(1, 4):
            tgt = 1 * k
            mask = (freqs>=tgt-8)&(freqs<=tgt+8)
            freq_row.append(float(mag[mask].max()) if mask.any() else 0.0)
    freq_records.append(freq_row)
    # 地點權重
    if 'Weight' in blk.columns:
        mode_weight = blk['Weight'].mode()
        if not mode_weight.empty:
            location_weights.append(mode_weight.iloc[0])
        else:
            location_weights.append(blk['Weight'].iloc[0])
    else:
        location_weights.append(0)
    # 標籤：只要有異常就標異常
    labels.append(0 if (blk['Label'] == 0).any() else 1)
    # 新增細緻特徵
    n_items = blk['Item'].nunique()
    n_categories = blk['Category'].nunique()
    n_locations = blk['Location'].nunique()
    high_price_ratio = (blk['Price'] > 500).mean() if len(blk) > 0 else 0
    max_price = blk['Price'].max() if len(blk) > 0 else 0
    min_price = blk['Price'].min() if len(blk) > 0 else 0
    mean_price = blk['Price'].mean() if len(blk) > 0 else 0
    item_concentration = blk['Item'].value_counts(normalize=True).max() if len(blk) > 0 else 0
    weekday = blk['Date'].iloc[0].weekday() if 'Date' in blk.columns else -1
    extra_features.append([
        n_items, n_categories, n_locations, high_price_ratio,
        max_price, min_price, mean_price, item_concentration, weekday
    ])

time_df = pd.DataFrame(time_records, columns=[f"{col}{suf}" for suf in ['_rms','_mean','_std','_ptp','_kurtosis','_skewness'] for col in num_cols])
freq_df = pd.DataFrame(freq_records, columns=[f"{col}_freq_{k}" for col in num_cols for k in range(1,4)])
extra_df = pd.DataFrame(extra_features, columns=[
    'n_items','n_categories','n_locations','high_price_ratio',
    'max_price','min_price','mean_price','item_concentration','weekday'])
features = pd.concat([time_df, freq_df, extra_df], axis=1)
features['Location_Weight'] = location_weights
labels = np.array(labels)

# 強制補 0
features = features.fillna(0)

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

# 這裡不用再重新產生 labels
# 直接產生 merged
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

    # 新增多個監督式模型
    rf = RandomForestClassifier(class_weight='balanced', random_state=42).fit(Xtr_s, ytr)
    gb = GradientBoostingClassifier(random_state=42).fit(Xtr_s, ytr)
    dt = DecisionTreeClassifier(class_weight='balanced', random_state=42).fit(Xtr_s, ytr)
    knn = KNeighborsClassifier().fit(Xtr_s, ytr)
    svc = SVC(probability=True, class_weight='balanced', random_state=42).fit(Xtr_s, ytr)

    # 評估各監督式模型
    sup_models = {
        'LR': clf2,
        'RF': rf,
        'GB': gb,
        'DT': dt,
        'KNN': knn,
        'SVC': svc
    }
    sup_results = {}
    for name, model in sup_models.items():
        y_pred = model.predict(Xte_s)
        sup_results[name] = {
            'Accuracy': accuracy_score(yte, y_pred),
            'Precision': precision_score(yte, y_pred),
            'Recall': recall_score(yte, y_pred),
            'F1': f1_score(yte, y_pred)
        }
    sup_results_df = pd.DataFrame(sup_results).T
    print("\nSupervised Models Comparison:")
    print(sup_results_df)

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

# 15.5 Local Outlier Factor (LOF)
lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
lof.fit(X[y == 1])
lof_scores = -lof.decision_function(X)
plt.figure(figsize=(10,4))
plt.plot(lof_scores, label="LOF Score")
plt.axhline(np.percentile(lof_scores, 95), ls='--', c='red', label='95%')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR,'lof_scores.png'))
plt.close()

# 15.6 Robust Covariance (EllipticEnvelope)
try:
    ee = EllipticEnvelope(contamination=0.1, random_state=42).fit(X[y == 1])
    ee_scores = -ee.decision_function(X)
except Exception:
    ee_scores = np.zeros(len(X))
plt.figure(figsize=(10,4))
plt.plot(ee_scores, label="EllipticEnvelope Score")
plt.axhline(np.percentile(ee_scores, 95), ls='--', c='red', label='95%')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR,'elliptic_scores.png'))
plt.close()

# --------------------------------------------------
# 16. 模型比較
# --------------------------------------------------
# AutoEncoder 異常分數
try:
    ae_scores = get_autoencoder_scores(X, y)
except Exception as e:
    print(f"AutoEncoder error: {e}")
    ae_scores = np.zeros(len(X))
scores = {
    'Hotelling T²': t2,
    'SPE': spe,
    'One-Class SVM': ocsvm_scores,
    'Isolation Forest': iso_scores,
    'LOF': lof_scores,
    'EllipticEnvelope': ee_scores,
    'AutoEncoder': ae_scores
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

# 16. 模型比較
ensemble_score = get_ensemble_score(scores)
scores['Ensemble'] = ensemble_score

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

# 保存特徵順序
with open(os.path.join(MODEL_DIR, 'feature_order.json'), 'w', encoding='utf-8') as f:
    json.dump(list(features.columns), f)

# 保存監督式模型
if len(classes) > 1:
    joblib.dump(clf2, os.path.join(MODEL_DIR, 'lr_model.pkl'))
    joblib.dump(rf, os.path.join(MODEL_DIR, 'rf_model.pkl'))
    joblib.dump(gb, os.path.join(MODEL_DIR, 'gb_model.pkl'))
    joblib.dump(dt, os.path.join(MODEL_DIR, 'dt_model.pkl'))
    joblib.dump(knn, os.path.join(MODEL_DIR, 'knn_model.pkl'))
    joblib.dump(svc, os.path.join(MODEL_DIR, 'svc_model.pkl'))
    print("LR, RF, GB, DT, KNN, SVC models saved successfully!")

print("\nModels saved successfully!")
