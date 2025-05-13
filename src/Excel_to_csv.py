import os
import pandas as pd

def xlsx_to_csv(xlsx_path, output_dir=None):
    """
    將指定 xlsx 檔案的所有工作表轉換為 csv 檔案。
    每個工作表轉成一個 csv，命名為 檔名_工作表名.csv。
    output_dir: 輸出 csv 的資料夾，預設為 xlsx 檔案所在資料夾。
    """
    if output_dir is None:
        output_dir = os.path.dirname(xlsx_path)
    base_name = os.path.splitext(os.path.basename(xlsx_path))[0]
    excel = pd.ExcelFile(xlsx_path)
    for sheet in excel.sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=sheet)
        csv_name = f"{base_name}.csv"
        csv_path = os.path.join(output_dir, csv_name)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"已匯出: {csv_path}")

if __name__ == '__main__':
    xlsx_file = input('請輸入要轉換的 xlsx 檔案路徑（例如 data/202505.xlsx）：')
    xlsx_to_csv(xlsx_file) 