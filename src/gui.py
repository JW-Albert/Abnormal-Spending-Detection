from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
from scipy.stats import kurtosis, skew
from scipy.fft import fft
from get_include import get_unique_options
import json

app = Flask(__name__)

# 載入模型和 scaler
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# 初始化模型和 scaler
models = {
    'ocsvm': None,
    'iso_forest': None,
    'scaler': None,
    'lof': None,
    'elliptic': None,
    'lr': None,
    'rf': None,
    'gb': None,
    'dt': None,
    'knn': None,
    'svc': None
}

# 儲存當日消費資料
daily_spending = []

# 載入 mapping
API_DIR = 'API/mapping'
def load_mapping(name):
    with open(os.path.join(API_DIR, f'{name}.json'), encoding='utf-8') as f:
        return json.load(f)
item_map = load_mapping('item')
location_map = load_mapping('location')
category_map = load_mapping('category')
location_weight_map = load_mapping('location_weight')

# 載入特徵順序
with open(os.path.join(MODEL_DIR, 'feature_order.json'), encoding='utf-8') as f:
    feature_order = json.load(f)

# 載入 autoencoder, ensemble
try:
    from autoencoder_model import get_autoencoder_scores
    from ensemble import get_ensemble_score
    autoencoder_loaded = True
except ImportError:
    autoencoder_loaded = False

def load_models():
    try:
        models['ocsvm'] = joblib.load(os.path.join(MODEL_DIR, 'ocsvm_model.pkl'))
        models['iso_forest'] = joblib.load(os.path.join(MODEL_DIR, 'iso_forest_model.pkl'))
        models['scaler'] = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        # 新增 LOF 與 EllipticEnvelope 載入
        lof_path = os.path.join(MODEL_DIR, 'lof_model.pkl')
        if os.path.exists(lof_path):
            models['lof'] = joblib.load(lof_path)
        elliptic_path = os.path.join(MODEL_DIR, 'elliptic_model.pkl')
        if os.path.exists(elliptic_path):
            models['elliptic'] = joblib.load(elliptic_path)
        # 新增 LR 及多監督式模型載入
        lr_path = os.path.join(MODEL_DIR, 'lr_model.pkl')
        if os.path.exists(lr_path):
            models['lr'] = joblib.load(lr_path)
        rf_path = os.path.join(MODEL_DIR, 'rf_model.pkl')
        if os.path.exists(rf_path):
            models['rf'] = joblib.load(rf_path)
        gb_path = os.path.join(MODEL_DIR, 'gb_model.pkl')
        if os.path.exists(gb_path):
            models['gb'] = joblib.load(gb_path)
        dt_path = os.path.join(MODEL_DIR, 'dt_model.pkl')
        if os.path.exists(dt_path):
            models['dt'] = joblib.load(dt_path)
        knn_path = os.path.join(MODEL_DIR, 'knn_model.pkl')
        if os.path.exists(knn_path):
            models['knn'] = joblib.load(knn_path)
        svc_path = os.path.join(MODEL_DIR, 'svc_model.pkl')
        if os.path.exists(svc_path):
            models['svc'] = joblib.load(svc_path)
    except Exception as e:
        print(f"Models not found or error: {e}. Please train the models first.")

load_models()  # 確保不論如何都會載入模型

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

def extra_features_block(blk):
    n_items = blk['Item'].nunique()
    n_categories = blk['Category'].nunique()
    n_locations = blk['Location'].nunique()
    high_price_ratio = (blk['Price'] > 500).mean() if len(blk) > 0 else 0
    max_price = blk['Price'].max() if len(blk) > 0 else 0
    min_price = blk['Price'].min() if len(blk) > 0 else 0
    mean_price = blk['Price'].mean() if len(blk) > 0 else 0
    item_concentration = blk['Item'].value_counts(normalize=True).max() if len(blk) > 0 else 0
    weekday = blk['Date'].iloc[0].weekday() if 'Date' in blk.columns else -1
    return [n_items, n_categories, n_locations, high_price_ratio, max_price, min_price, mean_price, item_concentration, weekday]

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
            'Category': data.get('category', ''),
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

@app.route('/delete_spending', methods=['POST'])
def delete_spending():
    try:
        data = request.get_json()
        idx = data.get('index')
        if idx is not None and 0 <= idx < len(daily_spending):
            daily_spending.pop(idx)
            return jsonify({'success': True, 'current_spending': daily_spending})
        else:
            return jsonify({'success': False, 'error': '索引錯誤'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/clear_spending', methods=['POST'])
def clear_spending():
    try:
        daily_spending.clear()
        return jsonify({'success': True, 'current_spending': daily_spending})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if not daily_spending:
            return jsonify({'success': False, 'error': '沒有消費資料可供分析'})
        if models['scaler'] is None:
            return jsonify({'success': False, 'error': 'scaler 未載入，請先執行模型訓練'})
        df = pd.DataFrame(daily_spending)
        # robust 處理 Category 欄位
        if 'Category' not in df.columns:
            df['Category'] = ''
        df['Item_Code'] = df['Item'].map(item_map).fillna(-1).astype(int)
        df['Location_Code'] = df['Location'].map(location_map).fillna(-1).astype(int)
        df['Category_Code'] = df['Category'].map(category_map).fillna(-1).astype(int)
        df['Date'] = pd.Timestamp.today()
        # 權重
        def get_location_weight(x):
            loc_name = str(x).strip()
            code = location_map.get(loc_name, -1)
            # 支援 int/str key
            return location_weight_map.get(str(code), location_weight_map.get(int(code), 0))
        df['Location_Weight'] = df['Location'].map(get_location_weight)
        weight_warning = None
        if (df['Location_Weight'] == 0).any():
            weight_warning = '地點權重為0，請檢查 mapping 檔案或地點名稱是否正確'
        num_cols = ['Price', 'Quantity', 'Location_Weight', 'Total Daily Spending', 'Item_Code']
        time_features = time_domain_features(df, num_cols)
        freq_features = frequency_domain_features(df, num_cols)
        extra = extra_features_block(df)
        extra_df = pd.DataFrame([extra], columns=['n_items','n_categories','n_locations','high_price_ratio','max_price','min_price','mean_price','item_concentration','weekday'])
        all_features = pd.concat([time_features, freq_features, extra_df], axis=1)
        all_features['Location_Weight'] = df['Location_Weight'].mode()[0] if not df['Location_Weight'].mode().empty else 0
        # 對齊特徵順序，補齊缺的欄位
        for col in feature_order:
            if col not in all_features.columns:
                all_features[col] = 0
        all_features = all_features[feature_order]
        nan_columns = list(all_features.columns[all_features.isna().any()])
        all_features = all_features.fillna(0)
        X = models['scaler'].transform(all_features)
        # 多監督式模型推論
        results = {}
        votes = []
        for name in ['lr', 'rf', 'gb', 'dt', 'knn', 'svc']:
            model = models.get(name)
            if model is not None:
                prob = float(model.predict_proba(X)[0, 1])
                results[name] = prob
                # 以 0.5 為閾值，1=正常, 0=異常
                votes.append(prob >= 0.5)
        # 多數決集成（超過三分之二同意）
        ensemble_majority = None
        is_anomaly = None
        if votes:
            agree = sum(votes)
            if agree > (2/3)*len(votes):
                ensemble_majority = '正常'
                is_anomaly = False
            elif (len(votes)-agree) > (2/3)*len(votes):
                ensemble_majority = '異常'
                is_anomaly = True
            else:
                ensemble_majority = '無明顯共識'
                is_anomaly = None
        # 依照指定公式計算 abnormal_score
        n_locations = extra[2]
        max_price = extra[4]
        total_spending = df['Total Daily Spending'].sum()
        n_items = extra[0]
        avg_location_weight = df['Location_Weight'].mean() if 'Location_Weight' in df.columns else 0
        abnormal_score = (
            -36.80
            + 16.78 * n_locations
            + 0.03 * max_price
            + 0.01 * total_spending
            + 2.72 * n_items
            + 5.5 * avg_location_weight
        )
        daily_spending.clear()
        return jsonify({
            'success': True,
            'results': results,
            'is_anomaly': is_anomaly,
            'ensemble_majority': ensemble_majority,
            'abnormal_score': abnormal_score,
            'weight_warning': weight_warning,
            'nan_columns': nan_columns,
            'message': '分析完成'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 