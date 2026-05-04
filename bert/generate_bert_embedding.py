import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

# 讀取原始 VAERS 資料
print("讀取原始資料中...")
df_raw = pd.read_csv("VAERS_2014_2025_FLU4.csv", encoding="utf-8-sig")

# 篩選出關鍵欄位
target_columns = ['DIED', 'L_THREAT', 'ER_VISIT', 'HOSPITAL', 'X_STAY', 'ER_ED_VISIT']
keep_columns = ['VAERS_ID', 'SYMPTOM_TEXT', 'AGE_YRS', 'SEX'] + target_columns
df_clean = df_raw[keep_columns].copy()

# 移除沒有填寫症狀文字的無效資料
df_clean = df_clean.dropna(subset=['SYMPTOM_TEXT']).reset_index(drop=True)
print(f"清洗後剩餘有效資料筆數：{len(df_clean)}")

print("轉換目標標籤格式中 ('Y' -> 1, 空值 -> 0)...")
for col in target_columns:
    df_clean[col] = df_clean[col].apply(lambda x: 1 if str(x).strip().upper() == 'Y' else 0)

# 檢查一下轉換後的重症分佈
Y_any = (df_clean[target_columns] == 1).any(axis=1)
print(f"其中包含重症/住院等目標的筆數：{Y_any.sum()} 筆")

# 載入 BERT 模型
print("\n正在載入 Sentence-BERT 神經網路模型...")
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

# 開始轉換！將文字變成高階嵌入向量
print(f"開始轉換文字為 Embedding 向量...")
embeddings = model.encode(df_clean['SYMPTOM_TEXT'].tolist(), show_progress_bar=True)

# 把向量結果存進 DataFrame 
df_clean['article_vector'] = embeddings.tolist()

output_name = "merged_data_bert.csv"
df_clean.to_csv(output_name, index=False, encoding="utf-8-sig")
print(f"\nBERT 向量資料已儲存為：{output_name}")