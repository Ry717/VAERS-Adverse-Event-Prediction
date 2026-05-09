import pandas as pd 
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler
from sklearn.svm import SVC

print("讀取資料中...")
merged_df = pd.read_csv("merged_data_bert.csv")

def parse_vector(vec):
    try:
        return ast.literal_eval(vec)
    except Exception as e:
        return None

merged_df['article_vector'] = merged_df['article_vector'].apply(parse_vector)
merged_df = merged_df[merged_df['article_vector'].notnull()].reset_index(drop=True)

merged_df['AGE_YRS'] = merged_df['AGE_YRS'].fillna(merged_df['AGE_YRS'].median())
merged_df['SEX'] = merged_df['SEX'].fillna('U')

X_vector = np.array(merged_df['article_vector'].tolist())
X_vector = X_vector.reshape(X_vector.shape[0], -1)

X_age_sex = merged_df[['AGE_YRS', 'SEX']]

target_columns = ['DIED', 'L_THREAT', 'ER_VISIT', 'HOSPITAL', 'X_STAY', 'ER_ED_VISIT']
merged_df = merged_df.dropna(subset=target_columns).reset_index(drop=True)
Y = (merged_df[target_columns] == 1).any(axis=1).astype(int)

X_vector = X_vector[merged_df.index]
X_age_sex = X_age_sex.loc[merged_df.index]

preprocessor = ColumnTransformer(
    transformers=[
        ('age', StandardScaler(), ['AGE_YRS']),
        ('sex', OneHotEncoder(drop='first', sparse_output=False), ['SEX']),
    ])
X_age_sex_processed = preprocessor.fit_transform(X_age_sex)

X = np.hstack([X_vector, X_age_sex_processed])

print("進行資料切分與欠採樣...")
# 80% 訓練、20% 測試
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

rus = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

print("開始訓練 SVM 模型...")

svm_model = SVC(
    kernel='linear',          # rbf徑向基函數 linear線性核函數
    probability=True,      # 必須打開這個，才會吐出「機率值」讓我們算 AUCPR
    random_state=42        
)

svm_model.fit(X_train_resampled, y_train_resampled) 
print("SVM 訓練完成！")

y_pred_svm = svm_model.predict(X_test)
y_pred_proba_svm = svm_model.predict_proba(X_test)[:, 1]

auc_svm = roc_auc_score(y_test, y_pred_proba_svm)
aucpr_svm = average_precision_score(y_test, y_pred_proba_svm)

print("\nSVM ：")
print(f"ROC-AUC: {auc_svm:.4f}")
print(f"AUCPR: {aucpr_svm:.4f}")
print("\n詳細分類報告：")
print(classification_report(y_test, y_pred_svm))