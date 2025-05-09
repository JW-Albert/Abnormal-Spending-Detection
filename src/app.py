from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
from scipy.stats import kurtosis, skew
from scipy.fft import fft

app = Flask(__name__)

# 載入模型和 scaler
MODEL_DIR = '../models'
os.makedirs(MODEL_DIR, exist_ok=True)

# 初始化模型和 scaler
models = {
    'ocsvm': None,
    'iso_forest': None,
    'scaler': None
}

def load_models():
    try:
        models['ocsvm'] = joblib.load(os.path.join(MODEL_DIR, 'ocsvm_model.pkl'))
        models['iso_forest'] = joblib.load(os.path.join(MODEL_DIR, 'iso_forest_model.pkl'))
        models['scaler'] = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    except:
        print("Models not found. Please train the models first.")

def time_domain_features(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    ops = {
        '_rms':       lambda x: np.sqrt((x**2).mean()),
        '_mean':      np.mean,
        '_std':       np.std,
        '_ptp':       lambda x: np.ptp(x),
        '_kurtosis':  kurtosis,
        '_skewness':  skew
    }
    feature_names = [f"{col}{suf}" for col in cols for suf in ops]
    vals = []
    for fn in ops.values():
        for col in cols:
            vals.append(fn(df[col].values))
    return pd.DataFrame([vals], columns=feature_names)

def frequency_domain_features(df: pd.DataFrame, cols: list, fs: int = 1, base: int = 1, n: int = 3) -> pd.DataFrame:
    L = len(df)
    dfreq = fs / L
    half = L // 2
    freqs = np.arange(half) * dfreq
    fftv = np.abs(fft(df[cols].values, axis=0))[:half]
    row = []
    for j, col in enumerate(cols):
        mag = fftv[:,j]
        for k in range(1, n+1):
            tgt = base * k
            mask = (freqs>=tgt-8)&(freqs<=tgt+8)
            row.append(float(mag[mask].max()) if mask.any() else 0.0)
    feature_names = [f"{col}_freq_{k}" for col in cols for k in range(1,n+1)]
    return pd.DataFrame([row], columns=feature_names)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # list of dicts
        df = pd.DataFrame(data)
        # 編碼
        for col in ['Location', 'Item', 'Category']:
            df[col + '_Code'] = pd.Categorical(df[col]).codes
        # 產生時域/頻域特徵
        num_cols = ['Price', 'Quantity', 'Total Daily Spending']
        time_features = time_domain_features(df, num_cols)
        freq_features = frequency_domain_features(df, num_cols)
        # 分類特徵用 mode
        block_cats = df[['Location_Code', 'Item_Code', 'Category_Code']].mode()
        if block_cats.empty:
            block_cats = df[['Location_Code', 'Item_Code', 'Category_Code']].iloc[[0]]
        else:
            block_cats = block_cats.iloc[[0]]
        # 合併
        all_features = pd.concat([time_features, freq_features, block_cats], axis=1)
        expected_features = [
            'Price_rms', 'Price_mean', 'Price_std', 'Price_ptp', 'Price_kurtosis', 'Price_skewness',
            'Quantity_rms', 'Quantity_mean', 'Quantity_std', 'Quantity_ptp', 'Quantity_kurtosis', 'Quantity_skewness',
            'Total Daily Spending_rms', 'Total Daily Spending_mean', 'Total Daily Spending_std',
            'Total Daily Spending_ptp', 'Total Daily Spending_kurtosis', 'Total Daily Spending_skewness',
            'Price_freq_1', 'Price_freq_2', 'Price_freq_3',
            'Quantity_freq_1', 'Quantity_freq_2', 'Quantity_freq_3',
            'Total Daily Spending_freq_1', 'Total Daily Spending_freq_2', 'Total Daily Spending_freq_3',
            'Location_Code', 'Item_Code', 'Category_Code'
        ]
        all_features = all_features[expected_features]
        # 標準化
        X = models['scaler'].transform(all_features)
        # 預測
        predictions = {
            'ocsvm': float(models['ocsvm'].score_samples(X)[0]),
            'iso_forest': float(models['iso_forest'].score_samples(X)[0])
        }
        combined_score = (predictions['ocsvm'] + predictions['iso_forest']) / 2
        is_anomaly = combined_score < -0.5
        return jsonify({
            'success': True,
            'predictions': predictions,
            'combined_score': combined_score,
            'is_anomaly': bool(is_anomaly)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    load_models()
    app.run(debug=True) 