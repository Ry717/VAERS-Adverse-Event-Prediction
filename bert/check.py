import pandas as pd

# 讀取你的原始資料 (如果是 csv 的話)
df_raw = pd.read_csv("VAERS_2014_2025_FLU4.csv", encoding="utf-8-sig")

# 印出所有的欄位名稱，找找看有沒有 TEXT 相關的欄位
print("欄位名稱：", df_raw.columns.tolist())

# 印出前兩筆資料的文字內容來確認
print("\n前兩筆資料預覽：\n", df_raw.head(2))