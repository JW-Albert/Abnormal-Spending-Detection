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

def read_csv(file_path) -> pd.DataFrame:
    """
    Reads a CSV file and returns a pd DataFrame.
    """
    return pd.read_csv(file_path)

def get_item(df: pd.DataFrame) -> list:
    """
    Returns a list of unique items from the 'Item' column in the DataFrame.
    """
    if 'Item' not in df.columns:
        raise ValueError("DataFrame does not contain 'Item' column.")
    
    item_list = sorted(df['Item'].dropna().unique().tolist())
    return item_list

def get_quantity(df: pd.DataFrame) -> list:
    """
    Returns a list of unique items from the 'Quantitytem' column in the DataFrame.
    """
    if 'Quantity' not in df.columns:
        raise ValueError("DataFrame does not contain 'Quantity' column.")
    
    item_list = sorted(df['Quantity'].dropna().unique().tolist())
    return item_list

def get_Category(df: pd.DataFrame) -> list:
    """
    Returns a list of unique items from the 'Category' column in the DataFrame.
    """
    if 'Category' not in df.columns:
        raise ValueError("DataFrame does not contain 'Category' column.")
    
    item_list = sorted(df['Category'].dropna().unique().tolist())
    return item_list

def get_location(df: pd.DataFrame) -> list:
    """
    Returns a list of unique items from the 'Location' column in the DataFrame.
    """
    if 'Location' not in df.columns:
        raise ValueError("DataFrame does not contain 'Location' column.")
    
    item_list = sorted(df['Location'].dropna().unique().tolist())
    return item_list

def get_LocationWeight(df: pd.DataFrame) -> list:
    """
    Returns a list of unique items from the 'Location Weight' column in the DataFrame.
    """
    if 'Location Weight' not in df.columns:
        raise ValueError("DataFrame does not contain 'Location Weight' column.")
    
    item_list = sorted(df['Location Weight'].dropna().unique().tolist())
    return item_list

def get_location_weight_dict(df: pd.DataFrame) -> dict:
    """
    Returns a dictionary where keys are Locations and values are their corresponding Location Weights.
    """
    if 'Location' not in df.columns or 'Location Weight' not in df.columns:
        raise ValueError("DataFrame must contain 'Location' and 'Location Weight' columns.")

    # 移除重複組合，確保一對一映射
    location_weight_df = df[['Location', 'Location Weight']].drop_duplicates()
    
    # 轉為 dict
    location_weight_dict = sorted(dict(zip(location_weight_df['Location'], location_weight_df['Location Weight'])))
    return location_weight_dict

def get_item_category_dict(df: pd.DataFrame) -> dict:
    """
    Returns a dictionary where keys are Items and values are their corresponding Categories.
    """
    if 'Item' not in df.columns or 'Category' not in df.columns:
        raise ValueError("DataFrame must contain 'Item' and 'Category' columns.")
    
    item_category_df = df[['Item', 'Category']].drop_duplicates()
    item_category_dict = sorted(dict(zip(item_category_df['Item'], item_category_df['Category'])))
    return item_category_dict

def get_category_items_dict(df: pd.DataFrame) -> dict:
    """
    Returns a dictionary where keys are Categories (sorted)
    and values are sorted lists of corresponding Items.
    """
    if 'Item' not in df.columns or 'Category' not in df.columns:
        raise ValueError("DataFrame must contain 'Item' and 'Category' columns.")

    grouped = df[['Item', 'Category']].drop_duplicates()
    grouped = grouped.groupby('Category')['Item'].apply(lambda x: sorted(x.tolist()))
    return dict(sorted(grouped.to_dict().items()))

def simplify_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    將 'Location', 'Location Weight', 'Item', 'Category' 簡化為標籤代碼。
    同時保留對應的對照字典。
    """
    # 建立標籤字典
    location_dict = {name: idx for idx, name in enumerate(sorted(df['Location'].dropna().unique()))}
    item_dict = {name: idx for idx, name in enumerate(sorted(df['Item'].dropna().unique()))}
    category_dict = {name: idx for idx, name in enumerate(sorted(df['Category'].dropna().unique()))}

    # 將欄位轉為代碼
    df['Location_Code'] = df['Location'].map(location_dict)
    df['Item_Code'] = df['Item'].map(item_dict)
    df['Category_Code'] = df['Category'].map(category_dict)

    # 若 Location Weight 是與 Location 綁定，直接保留原本欄位或用 dict 整合轉換
    # location_weight_dict = get_location_weight_dict(df) ← 亦可視情況使用

    return df, location_dict, item_dict, category_dict

def save_daily_spending_plot(daily_normal, daily_abnormal, save_dir="img", filename="daily_spending_comparison.png"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(14, 6))
    plt.plot(daily_normal.index, daily_normal.values, label='Normal', color='blue')
    plt.plot(daily_abnormal.index, daily_abnormal.values, label='Abnormal', color='red')
    plt.title("Daily Spending Comparison (Normal vs Abnormal)")
    plt.xlabel("Date")
    plt.ylabel("Total Daily Spending")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

def time_domain(data: pd.DataFrame, columns_to_evaluate: list, unit_size: int) -> pd.DataFrame:
    # Define a dictionary of feature operations
    time_domain_features = {
        "_rms": lambda x: np.sqrt(np.mean(x**2)),
        "_mean": np.mean,
        "_std": np.std,
        "_peak_to_peak": lambda x: np.max(x) - np.min(x),
        "_kurtosis": kurtosis,
        "_skewness": skew,
        "_crest_indicator": lambda x: abs(np.max(x)) / np.sqrt(np.mean(x**2)),
        "_clearance_indicator": lambda x: abs(np.max(x)) / np.mean(np.sqrt(abs(x))) ** 2,
        "_shape_indicator": lambda x: np.sqrt(np.mean(x**2)) / np.mean(abs(x)),
        "_impulse_indicator": lambda x: abs(np.max(x)) / np.mean(abs(x)),
    }

    # Initialize a list to hold the results for all units
    results = []

    # Loop through the data in chunks of unit_size
    for start in range(0, data.shape[0], unit_size):
        unit_data = data.iloc[start:start + unit_size]
        # Loop through each column in columns_to_evaluate and apply each operation
        unit_results = [
            feature_function(unit_data[column])
            for column in columns_to_evaluate
            for feature_function in time_domain_features.values()
        ]
        results.append(unit_results)

    # Create a list of feature names
    feature_names = [
        f"{column}{feature_name}"
        for column in columns_to_evaluate
        for feature_name in time_domain_features
    ]

    # Return the results as a DataFrame
    return pd.DataFrame(results, columns=feature_names)

def plot_time_features(time_df, save_dir="img", filename="time_features.png"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(14, 5))
    sns.barplot(x=time_df.columns, y=time_df.iloc[0].values)
    plt.xticks(rotation=90)
    plt.title("Time Domain Features (First Sample)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

def frequency_domain(data: pd.DataFrame, columns_to_frequency:list, fs: int, base_freq: int, n: int, unit_size: int) -> pd.DataFrame:
    # Initialize list to hold all frequency domain features for all units
    all_units_features = []

    # Loop through the data in chunks of unit_size
    for start in range(0, data.shape[0], unit_size):
        unit_data = data.iloc[start:start + unit_size]
        frequency_domain_features = []
        df = fs / len(unit_data)  # Frequency resolution
        freq = np.linspace(0, len(unit_data) // 2 - 1, len(unit_data) // 2) * df  # Frequency values

        # Loop through the columns specified in columns_to_frequency
        for column in columns_to_frequency:
            # Perform FFT on the selected column for the current unit
            fft_data = abs(fft(unit_data[column].values)) * 2 / unit_data.shape[0]
            fft_data = pd.DataFrame(fft_data[:len(unit_data) // 2], index=freq)

            # Extract features from harmonics
            for i in range(1, n + 1):
                target_freq = base_freq * i
                feature_value = float(fft_data.loc[target_freq - 8:target_freq + 8].max().iloc[0])
                frequency_domain_features.append(feature_value)

        # Append features for this unit to the list
        all_units_features.append(frequency_domain_features)

    # Create a list of feature names
    feature_names = [f"{column}_freq_{i}" for column in columns_to_frequency for i in range(1, n + 1)]

    '''plt.figure(figsize=(12, 6))
    plt.plot(freq, all_units_features[:len(freq)])
    plt.xlim(0, 200)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('FFT Data (0-200 Hz)')'''

    # Return the results as a DataFrame
    return pd.DataFrame(all_units_features, columns=feature_names)

def plot_freq_features(freq_df, save_dir="img", filename="freq_features.png"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(14, 5))
    sns.barplot(x=freq_df.columns, y=freq_df.iloc[0].values)
    plt.xticks(rotation=90)
    plt.title("Frequency Domain Features (First Sample)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()


num_normal = 0
num_abnormal = 0

normal_data = []
abnormal_data = []

folder_path_normal = os.path.join('data', 'normal')
folder_path_abnormal = os.path.join('data', 'abnormal')


# 自動讀取 normal 資料夾所有 CSV
for file in os.listdir(folder_path_normal):
    if file.endswith(".csv"):
        path = os.path.join(folder_path_normal, file)
        df = pd.read_csv(path)
        df["Label"] = 1  # 正常資料標籤為 1
        normal_data.append(df)
        num_normal += len(df)

# 自動讀取 abnormal 資料夾所有 CSV
for file in os.listdir(folder_path_abnormal):
    if file.endswith(".csv"):
        path = os.path.join(folder_path_abnormal, file)
        df = pd.read_csv(path)
        df["Label"] = 0  # 異常資料標籤為 0
        abnormal_data.append(df)
        num_abnormal += len(df)

# 合併所有資料為一個 DataFrame
df_all = pd.concat(normal_data + abnormal_data, ignore_index=True)

print(f"已讀取正常資料筆數：{num_normal}")
print(f"已讀取異常資料筆數：{num_abnormal}")
print(f"總資料筆數：{len(df_all)}")

df_all, loc_map, item_map, cat_map = simplify_columns(df_all)
ml_df = df_all[['Location_Code', 'Item_Code', 'Category_Code', 'Price', 'Quantity', 'Total Daily Spending', 'Label']]

# 確保 Date 欄位為 datetime 格式
df_all['Date'] = pd.to_datetime(df_all['Date'])

# 分類資料
df_normal = df_all[df_all['Label'] == 1]
df_abnormal = df_all[df_all['Label'] == 0]

# 群組後求每日總支出
daily_normal = df_normal.groupby('Date')['Total Daily Spending'].sum()
daily_abnormal = df_abnormal.groupby('Date')['Total Daily Spending'].sum()

# 畫圖
save_daily_spending_plot(daily_normal, daily_abnormal)

unit_size = 10  # 每10筆視為一個樣本
numerical_columns = ['Price', 'Quantity', 'Total Daily Spending']

# 特徵提取
time_features = time_domain(ml_df, numerical_columns, unit_size)
freq_features = frequency_domain(ml_df, numerical_columns, fs=1, base_freq=1, n=3, unit_size=unit_size)

# 合併特徵
all_features = pd.concat([time_features, freq_features], axis=1)

# 建立標籤（每10筆一個）
labels = df_all['Label'].values[::unit_size][:len(all_features)]

plot_time_features(time_features)
plot_freq_features(freq_features)

# 標準化
scaler = StandardScaler()
scaled_features = scaler.fit_transform(all_features)

# PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)

plt.figure(figsize=(8, 5))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
plt.title("PCA Projection")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.savefig(os.path.join("img", "PCA.png"))
plt.tight_layout()
plt.close()

# 加入週期性特徵
df_all['DayOfWeek'] = df_all['Date'].dt.dayofweek  # 0=Monday
df_all['Day'] = df_all['Date'].dt.day
df_all['Month'] = df_all['Date'].dt.month
df_all['IsWeekend'] = df_all['DayOfWeek'].isin([5, 6]).astype(int)

# 加入移動平均
df_all['rolling_mean_7'] = df_all['Total Daily Spending'].rolling(window=7, min_periods=1).mean()
df_all['rolling_std_7'] = df_all['Total Daily Spending'].rolling(window=7, min_periods=1).std().fillna(0)

# 加入 Z-score
df_all['z_score_spending'] = (df_all['Total Daily Spending'] - df_all['Total Daily Spending'].mean()) / df_all['Total Daily Spending'].std()

t_squared = np.sum((pca_result / np.std(pca_result, axis=0))**2, axis=1)
threshold_t2 = np.percentile(t_squared, 95)

X_proj = pca.inverse_transform(pca_result)
spe = np.sum((scaled_features - X_proj)**2, axis=1)
threshold_spe = np.percentile(spe, 95)

X_train, X_test, y_train, y_test = train_test_split(all_features, labels, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=1000, class_weight='balanced')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

def fisher_score(X, y):
    score = []
    for i in range(X.shape[1]):
        mean0 = X[y == 0][:, i].mean()
        mean1 = X[y == 1][:, i].mean()
        var0 = X[y == 0][:, i].var()
        var1 = X[y == 1][:, i].var()
        score.append((mean0 - mean1)**2 / (var0 + var1 + 1e-6))
    return np.array(score)

f_scores = fisher_score(scaled_features, labels)
top_indices = np.argsort(f_scores)[::-1][:10]
top_features = [all_features.columns[i] for i in top_indices]
print("Top features by Fisher Score:", top_features)

# 儲存特徵與標籤供未來使用
merged_df = all_features.copy()
merged_df["Label"] = labels
os.makedirs("data", exist_ok=True)
merged_df.to_csv(os.path.join("data", "merged.csv"), index=False)