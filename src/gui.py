from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
from scipy.stats import kurtosis, skew
from scipy.fft import fft
from get_include import get_unique_options

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

# 儲存當日消費資料
daily_spending = []

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

@app.route('/get_options')
def get_options():
    options = get_unique_options('data')
    return jsonify(options)

@app.route('/add_spending', methods=['POST'])
def add_spending():
    try:
        data = request.get_json()
        # 將新的消費資料加入當日消費列表，正確存入權重
        daily_spending.append({
            'Location': data['location'],
            'Weight': data.get('weight', ''),
            'Item': data['item'],
            'Price': float(data['price']),
            'Quantity': 1,  # 預設數量為1
            'Total Daily Spending': float(data['price'])
        })
        return jsonify({
            'success': True,
            'message': '消費資料已新增',
            'current_spending': daily_spending
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if not daily_spending:
            return jsonify({
                'success': False,
                'error': '沒有消費資料可供分析'
            })

        df = pd.DataFrame(daily_spending)
        for col in ['Location', 'Item']:
            df[col + '_Code'] = pd.Categorical(df[col]).codes

        num_cols = ['Price', 'Quantity', 'Total Daily Spending']
        time_features = time_domain_features(df, num_cols)
        freq_features = frequency_domain_features(df, num_cols)

        block_cats = df[['Location_Code', 'Item_Code']].mode()
        if block_cats.empty:
            block_cats = df[['Location_Code', 'Item_Code']].iloc[[0]]
        else:
            block_cats = block_cats.iloc[[0]]

        # 新增：地點權重特徵
        if 'Weight' in df.columns:
            location_weight = df['Weight'].mode()
            if not location_weight.empty:
                location_weight = location_weight.iloc[0]
            else:
                location_weight = df['Weight'].iloc[0]
        else:
            location_weight = 0
        location_weight_df = pd.DataFrame({'Location_Weight': [location_weight]})

        # 合併所有特徵
        all_features = pd.concat([time_features, freq_features, block_cats, location_weight_df], axis=1)

        # 特徵順序要和訓練時一致
        expected_features = [
            'Price_rms', 'Price_mean', 'Price_std', 'Price_ptp', 'Price_kurtosis', 'Price_skewness',
            'Quantity_rms', 'Quantity_mean', 'Quantity_std', 'Quantity_ptp', 'Quantity_kurtosis', 'Quantity_skewness',
            'Total Daily Spending_rms', 'Total Daily Spending_mean', 'Total Daily Spending_std',
            'Total Daily Spending_ptp', 'Total Daily Spending_kurtosis', 'Total Daily Spending_skewness',
            'Price_freq_1', 'Price_freq_2', 'Price_freq_3',
            'Quantity_freq_1', 'Quantity_freq_2', 'Quantity_freq_3',
            'Total Daily Spending_freq_1', 'Total Daily Spending_freq_2', 'Total Daily Spending_freq_3',
            'Location_Code', 'Item_Code', 'Location_Weight'
        ]
        all_features = all_features[expected_features]

        X = models['scaler'].transform(all_features)
        predictions = {
            'ocsvm': float(models['ocsvm'].score_samples(X)[0]),
            'iso_forest': float(models['iso_forest'].score_samples(X)[0])
        }
        combined_score = (predictions['ocsvm'] + predictions['iso_forest']) / 2
        is_anomaly = combined_score < -0.5
        daily_spending.clear()
        return jsonify({
            'success': True,
            'predictions': predictions,
            'combined_score': combined_score,
            'is_anomaly': bool(is_anomaly),
            'message': '分析完成'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    load_models()
    app.run(debug=True) 