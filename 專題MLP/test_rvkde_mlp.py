import pandas as pd 
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import RVKDE  

print("\n讀取資料中...")
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

print("\nPCA 降維...")
pca = PCA(n_components=128, random_state=42)
X_vector_reduced = pca.fit_transform(X_vector)
explained_variance = sum(pca.explained_variance_ratio_)
print(f"成功將 768 維壓縮至 128 維 (保留資訊量: {explained_variance*100:.2f}%)")

print("\n切分資料集 (80% 訓練, 20% 測試)...")
X_train_text, X_test_text, age_sex_train, age_sex_test, y_train, y_test = train_test_split(
    X_vector_reduced, X_age_sex_processed, Y, test_size=0.2, random_state=42, stratify=Y
)

print("\nRVKDE，將 128 維壓縮至 1 維 PDF...")
dim_pca = 128
K2 = 10
backend_choice = "faiss" # 或 "cuml"，請確認你的 GPU 環境

nn_model = RVKDE.build_nn_kernels(X_train_text, K2=K2+1, backend=backend_choice)
sigmas, _ = RVKDE.rvkde_sigmas(X_train_text, beta=1.0, dim=dim_pca, backend=backend_choice)

pdf_train = RVKDE.cross_group_density_pairwise(
    X_query=X_train_text, X_kernels=X_train_text, sigmas_k=sigmas,
    nn=nn_model, K2=K2, dim=dim_pca, same_dataset=True
).reshape(-1, 1)

pdf_test = RVKDE.cross_group_density_pairwise(
    X_query=X_test_text, X_kernels=X_train_text, sigmas_k=sigmas,
    nn=nn_model, K2=K2, dim=dim_pca, same_dataset=False
).reshape(-1, 1)

print("RVKDE 轉換完成")

print("\n組合最終特徵並進行欠採樣...")
X_train_final = np.hstack([pdf_train, age_sex_train])
X_test_final = np.hstack([pdf_test, age_sex_test])

print(f"最終進入模型的特徵僅剩：{X_train_final.shape[1]} 維 (PDF + 年齡 + 性別)")

rus = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
X_train_resampled, y_train_resampled = rus.fit_resample(X_train_final, y_train)

print("\n開始訓練 MLP 類神經網路...")
mlp_model = MLPClassifier(
    hidden_layer_sizes=(64,), # 由於特徵已經極度濃縮，改用單層 64 顆神經元測試
    activation='relu',            
    solver='adam',                
    max_iter=1000,                
    early_stopping=True,          
    n_iter_no_change=20,          
    random_state=42               
)

mlp_model.fit(X_train_resampled, y_train_resampled) 
print("MLP 訓練完成！\n")

y_pred_mlp = mlp_model.predict(X_test_final)
y_pred_proba_mlp = mlp_model.predict_proba(X_test_final)[:, 1]

auc_mlp = roc_auc_score(y_test, y_pred_proba_mlp)
aucpr_mlp = average_precision_score(y_test, y_pred_proba_mlp)

print("="*60)
print("RVKDE + MLP 測試結果")
print("="*60)
print(f"ROC-AUC: {auc_mlp:.4f}")
print(f"AUCPR  : {aucpr_mlp:.4f}")
print("\n詳細分類報告：")
print(classification_report(y_test, y_pred_mlp))