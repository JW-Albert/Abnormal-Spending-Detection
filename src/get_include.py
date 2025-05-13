import os
import pandas as pd
import json

def save_mapping(df, col, path):
    mapping = {k: v for v, k in enumerate(df[col].dropna().unique())}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    return mapping

def save_location_weight(df, mapping, path):
    # 權重欄位名稱可能為 'Location Weight' 或 'Weight'
    weight_col = 'Location Weight' if 'Location Weight' in df.columns else 'Weight'
    # 取每個店名的權重眾數
    weights = df.groupby('Location')[weight_col].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
    # 用商店代碼當 key
    code_weights = {str(mapping[loc]): weights[loc] for loc in weights.index if loc in mapping}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(code_weights, f, ensure_ascii=False, indent=2)
    return code_weights

def get_all_data(data_dir='data'):
    dfs = []
    for subfolder in ['normal', 'abnormal']:
        folder = os.path.join(data_dir, subfolder)
        if not os.path.exists(folder):
            continue
        label = 1 if subfolder == 'normal' else 0
        for file in os.listdir(folder):
            if file.endswith('.csv'):
                fpath = os.path.join(folder, file)
                df = pd.read_csv(fpath)
                df['Label'] = label
                dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    df_all = pd.concat(dfs, ignore_index=True)

    # 產生 mapping 檔案
    api_dir = os.path.join('API', 'mapping')
    item_map = save_mapping(df_all, 'Item', os.path.join(api_dir, 'item.json'))
    location_map = save_mapping(df_all, 'Location', os.path.join(api_dir, 'location.json'))
    save_mapping(df_all, 'Category', os.path.join(api_dir, 'category.json'))
    save_location_weight(df_all, location_map, os.path.join(api_dir, 'location_weight.json'))

    # 以 Date 分組，每天一個 block
    if 'Date' in df_all.columns:
        df_all['Date'] = pd.to_datetime(df_all['Date'])
        df_all = df_all.sort_values('Date')
    return df_all

def get_unique_options(data_dir='data'):
    df = get_all_data(data_dir)
    # Location 與 Location Weight 綁定
    loc_df = df[['Location', 'Location Weight']].drop_duplicates().dropna()
    locations = [
        {"name": row['Location'], "weight": row['Location Weight']}
        for _, row in loc_df.iterrows()
    ]
    # Category 與 Item 綁定
    cat_item_df = df[['Category', 'Item']].drop_duplicates().dropna()
    cat_dict = {}
    for _, row in cat_item_df.iterrows():
        cat = row['Category']
        item = row['Item']
        if cat not in cat_dict:
            cat_dict[cat] = set()
        cat_dict[cat].add(item)
    categories = [
        {"name": cat, "items": sorted(list(items))}
        for cat, items in cat_dict.items()
    ]
    return {
        "locations": locations,
        "categories": categories
    } 