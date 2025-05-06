import warnings
import os
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# 靜音收斂警告
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# 目錄設置
DATA_DIR = 'data'
IMG_DIR = 'img'
os.makedirs(IMG_DIR, exist_ok=True)

# 資料簡化與標籤編碼
def simplify_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ['Location', 'Item', 'Category']:
        df[col + '_Code'] = pd.Categorical(df[col]).codes
    return df

# 時域特徵
def time_domain(data: pd.DataFrame, cols: list, unit: int) -> pd.DataFrame:
    ops = {
        '_rms': lambda x: np.sqrt((x**2).mean()),
        '_mean': np.mean,
        '_std': np.std,
        '_ptp': lambda x: np.ptp(x),
        '_kurtosis': kurtosis,
        '_skewness': skew
    }
    names = [col + suf for col in cols for suf in ops]
    res = []
    for i in range(0, len(data), unit):
        block = data.iloc[i:i+unit]
        vals = []
        for col in cols:
            for fn in ops.values():
                vals.append(fn(block[col].values))
        res.append(vals)
    return pd.DataFrame(res, columns=names)

# 頻域特徵（含空遮罩檢查）
def frequency_domain(data: pd.DataFrame,
                     columns_to_frequency: list,
                     fs: int,
                     base_freq: int,
                     n: int,
                     unit_size: int) -> pd.DataFrame:
    all_feats = []
    for start in range(0, len(data), unit_size):
        block = data.iloc[start:start+unit_size]
        dfreq = fs / len(block)
        half = len(block) // 2
        freqs = np.arange(half) * dfreq
        fft_vals = np.abs(fft(block[columns_to_frequency].values, axis=0))[:half]
        feats = []
        for col_idx, _ in enumerate(columns_to_frequency):
            magn = fft_vals[:, col_idx]
            for i in range(1, n+1):
                target = base_freq * i
                mask = (freqs >= target-8) & (freqs <= target+8)
                feats.append(float(magn[mask].max()) if mask.any() else 0.0)
        all_feats.append(feats)
    names = [f"{col}_freq_{i}" 
             for col in columns_to_frequency 
             for i in range(1, n+1)]
    return pd.DataFrame(all_feats, columns=names)

# Fisher 分數
def fisher_score(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    scores = []
    for i in range(X.shape[1]):
        m0, m1 = X[y==0,i].mean(), X[y==1,i].mean()
        v0, v1 = X[y==0,i].var(),  X[y==1,i].var()
        scores.append((m0-m1)**2/(v0+v1+1e-6))
    return np.array(scores)

# 1. 讀取資料
normal_files   = [os.path.join(DATA_DIR,'normal',f)   for f in os.listdir(os.path.join(DATA_DIR,'normal'))   if f.endswith('.csv')]
abnormal_files = [os.path.join(DATA_DIR,'abnormal',f) for f in os.listdir(os.path.join(DATA_DIR,'abnormal')) if f.endswith('.csv')]

dfs = []
for f in normal_files:
    df = pd.read_csv(f); df['Label'] = 1; dfs.append(df)
for f in abnormal_files:
    df = pd.read_csv(f); df['Label'] = 0; dfs.append(df)
df_all = pd.concat(dfs, ignore_index=True)
print(f"Loaded normal={len(normal_files)} files, abnormal={len(abnormal_files)} files, total rows={len(df_all)}")

# 2. 特徵工程
df_all = simplify_columns(df_all)

# 畫每日消費曲線
if 'Date' in df_all.columns:
    df_all['Date'] = pd.to_datetime(df_all['Date'])
    daily = df_all.groupby(['Date','Label'])['Total Daily Spending'].sum().unstack(fill_value=0)
    plt.figure(figsize=(12,5))
    plt.plot(daily.index, daily[1], label='Normal',   color='blue')
    plt.plot(daily.index, daily[0], label='Abnormal', color='red')
    plt.title('Daily Spending Comparison')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR,'daily_spending.png'))
    plt.close()

# 時域＆頻域特徵
num_cols = ['Price','Quantity','Total Daily Spending']
time_df = time_domain(df_all, num_cols, unit=10)
freq_df = frequency_domain(
    data=df_all,
    columns_to_frequency=num_cols,
    fs=1,
    base_freq=1,
    n=3,
    unit_size=10
)

# 儲存時頻特徵圖
plt.figure(figsize=(14,4))
sns.barplot(x=time_df.columns, y=time_df.iloc[0].values)
plt.xticks(rotation=90); plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR,'time_features.png'))
plt.close()

plt.figure(figsize=(14,4))
sns.barplot(x=freq_df.columns, y=freq_df.iloc[0].values)
plt.xticks(rotation=90); plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR,'freq_features.png'))
plt.close()

# 合併block-level標籤\labels = []
labels = []
for start in range(0, len(df_all), 10):
    block = df_all.iloc[start:start+10]
    # 取這一塊中出現頻率最高的 Label
    labels.append(int(block['Label'].mode()[0]))
labels = np.array(labels)

# 合併特徵與標籤
features = pd.concat([time_df, freq_df], axis=1)
merged   = features.copy(); merged['Label'] = labels
merged.to_csv(os.path.join(DATA_DIR,'merged.csv'), index=False)

# 3. 標準化 & PCA
X = StandardScaler().fit_transform(features)
pca = PCA(n_components=2); X_pca = pca.fit_transform(X)
plt.figure(figsize=(8,5))
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='coolwarm', alpha=0.7)
plt.title('PCA Projection'); plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR,'pca_projection.png')); plt.close()

# 4. Hotelling's T² & SPE
t2    = np.sum((X_pca/np.std(X_pca,0))**2,axis=1)
t2th  = np.percentile(t2,95)
spe   = np.sum((X - pca.inverse_transform(X_pca))**2,axis=1)
speth = np.percentile(spe,95)

plt.figure(figsize=(10,4))
plt.plot(t2, label="T²"); plt.axhline(t2th, ls='--', c='red', label='95%'); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(IMG_DIR,'t_squared.png')); plt.close()

plt.figure(figsize=(10,4))
plt.plot(spe, label="SPE"); plt.axhline(speth, ls='--', c='red', label='95%'); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(IMG_DIR,'spe.png')); plt.close()

# 5. LDA 可視化
unique = np.unique(labels)
if len(unique)>1:
    lda_n = min(len(unique)-1, X.shape[1])
    lda = LDA(n_components=lda_n)
    X_lda = lda.fit_transform(X, labels)
    plt.figure(figsize=(8,5))
    if lda_n==1:
        plt.hist(X_lda[labels==1], bins=30, alpha=0.6, label='Normal', color='blue')
        plt.hist(X_lda[labels==0], bins=30, alpha=0.6, label='Abnormal', color='red')
        plt.xlabel('LDA1')
    else:
        plt.scatter(X_lda[:,0], X_lda[:,1], c=labels, cmap='coolwarm', alpha=0.7)
        plt.xlabel('LDA1'); plt.ylabel('LDA2')
    plt.title('LDA Projection'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR,'lda_visualization.png')); plt.close()
else:
    print("LDA: 只有一類，跳過")

# 6. Logistic Regression 邊界 & 評估
if len(unique)>1:
    clf = LogisticRegression(max_iter=5000, class_weight='balanced')
    clf.fit(X_pca, labels)
    x0_min,x0_max=X_pca[:,0].min()-1,X_pca[:,0].max()+1
    x1_min,x1_max=X_pca[:,1].min()-1,X_pca[:,1].max()+1
    xx,yy = np.meshgrid(np.linspace(x0_min,x0_max,200), np.linspace(x1_min,x1_max,200))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.figure(figsize=(8,6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='coolwarm', edgecolors='k')
    plt.title('Logistic Boundary'); plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR,'logistic_boundary.png')); plt.close()

    # LR 評估
    X_tr,X_te,y_tr,y_te = train_test_split(X, labels, test_size=0.2, stratify=labels, random_state=42)
    clf2 = LogisticRegression(max_iter=5000, class_weight='balanced')
    clf2.fit(X_tr,y_tr)
    y_pred = clf2.predict(X_te)
    print('Accuracy:',accuracy_score(y_te,y_pred))
    print('Precision:',precision_score(y_te,y_pred))
    print('Recall:',   recall_score(y_te,y_pred))
    print('F1 Score:', f1_score(y_te,y_pred))
    print('CMatrix:\n',confusion_matrix(y_te,y_pred))

# 7. Fisher Score Top10
fs = fisher_score(X, labels)
idx = np.argsort(fs)[::-1][:10]
fn = features.columns[idx]; fv = fs[idx]
plt.figure(figsize=(10,5))
colors = plt.cm.viridis(np.linspace(0,1,len(fn)))
plt.bar(fn, fv, color=colors)
plt.xticks(rotation=45); plt.title('Top10 Fisher Scores')
plt.tight_layout(); plt.savefig(os.path.join(IMG_DIR,'fisher_score_top10.png')); plt.close()
