import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neural_network import MLPClassifier

print("讀取資料中...")
df = pd.read_csv("merged_data_bert.csv", encoding="utf-8-sig")

def parse_vector(vec):
    try:
        return ast.literal_eval(vec)
    except Exception as e:
        return None

df['article_vector'] = df['article_vector'].apply(parse_vector)
df = df[df['article_vector'].notnull()].reset_index(drop=True)

# 設定目標 Y (只要任一欄為1即為正類)
target_columns = ['DIED', 'L_THREAT', 'ER_VISIT', 'HOSPITAL', 'X_STAY', 'ER_ED_VISIT']
df = df.dropna(subset=target_columns).reset_index(drop=True)
Y = (df[target_columns] == 1).any(axis=1).astype(int).values

# 處理年齡與性別
X_age_sex = df[['AGE_YRS', 'SEX']].copy()
X_age_sex['AGE_YRS'] = X_age_sex['AGE_YRS'].fillna(X_age_sex['AGE_YRS'].median())
X_age_sex['SEX'] = X_age_sex['SEX'].fillna('U')

preprocessor = ColumnTransformer(
    transformers=[
        ('age', StandardScaler(), ['AGE_YRS']),
        ('sex', OneHotEncoder(drop='first', sparse_output=False), ['SEX']),
    ])
X_meta_processed = preprocessor.fit_transform(X_age_sex)

X_vector = np.array(df['article_vector'].tolist())
X_vector = X_vector.reshape(X_vector.shape[0], -1) 

X_final = np.hstack([X_vector, X_meta_processed])

print("進行資料切分與欠採樣...")
X_train, X_test, Y_train, Y_test = train_test_split(
    X_final, Y, test_size=0.2, random_state=42, stratify=Y
)

sampler = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
X_train_res, Y_train_res = sampler.fit_resample(X_train, Y_train)

print("開始訓練 MLP 類神經網路模型...")
clf = MLPClassifier(
    hidden_layer_sizes=(128, 64), 
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42,
    early_stopping=True,
    n_iter_no_change=20
)

clf.fit(X_train_res, Y_train_res)
print("MLP 訓練完成！\n")

Y_pred = clf.predict(X_test)
Y_pred_proba = clf.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(Y_test, Y_pred_proba)
aucpr = average_precision_score(Y_test, Y_pred_proba)

print("MLP 類神經網路：")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"AUCPR: {aucpr:.4f}\n")

print("詳細分類報告：")
print(classification_report(Y_test, Y_pred, digits=2))