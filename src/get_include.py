import os
import pandas as pd

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
    return pd.concat(dfs, ignore_index=True)

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