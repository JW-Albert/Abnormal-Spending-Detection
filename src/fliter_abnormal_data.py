import pandas as pd
import os

def sort_abnormal_data(input_csv):
    """
    根據 Is Abnormal 欄位將資料分類到 normal.csv 和 abnormal.csv
    input_csv: 輸入的 CSV 檔案路徑
    """
    # 讀取 CSV 檔案
    df = pd.read_csv(input_csv)
    
    # 確保 Is Abnormal 欄位存在
    if 'Is Abnormal' not in df.columns:
        raise ValueError("CSV 檔案中沒有 'Is Abnormal' 欄位")
    
    # 將資料分類
    normal_data = df[df['Is Abnormal'] == 0]
    abnormal_data = df[df['Is Abnormal'] == 1]
    
    # 設定輸出目錄
    base_dir = 'data'
    normal_dir = os.path.join(base_dir, 'normal')
    abnormal_dir = os.path.join(base_dir, 'abnormal')
    
    # 確保輸出目錄存在
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(abnormal_dir, exist_ok=True)
    
    # 設定輸出檔案路徑
    normal_path = os.path.join(normal_dir, 'normal.csv')
    abnormal_path = os.path.join(abnormal_dir, 'abnormal.csv')
    
    # 儲存分類後的資料
    normal_data.to_csv(normal_path, index=False, encoding='utf-8-sig')
    abnormal_data.to_csv(abnormal_path, index=False, encoding='utf-8-sig')
    
    print(f"已將正常資料儲存至: {normal_path}")
    print(f"已將異常資料儲存至: {abnormal_path}")
    print(f"正常資料筆數: {len(normal_data)}")
    print(f"異常資料筆數: {len(abnormal_data)}")

if __name__ == '__main__':
    input_file = input('請輸入要分類的 CSV 檔案路徑：')
    sort_abnormal_data(input_file) 