import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import RVKDE 

print("讀取 BERT 嵌入向量與資料...")
df = pd.read_csv("merged_data_bert.csv", encoding="utf-8-sig")

def parse_vector(vec):
    try:
        return ast.literal_eval(vec)
    except Exception as e:
        return None

df['article_vector'] = df['article_vector'].apply(parse_vector)
df = df[df['article_vector'].notnull()].reset_index(drop=True)

target_columns = ['DIED', 'L_THREAT', 'ER_VISIT', 'HOSPITAL', 'X_STAY', 'ER_ED_VISIT']
df = df.dropna(subset=target_columns).reset_index(drop=True)
Y = (df[target_columns] == 1).any(axis=1).astype(int).values

X_age_sex = df[['AGE_YRS', 'SEX']].copy()
X_age_sex['AGE_YRS'] = X_age_sex['AGE_YRS'].fillna(X_age_sex['AGE_YRS'].median())
X_age_sex['SEX'] = X_age_sex['SEX'].fillna('U')

preprocessor = ColumnTransformer(
    transformers=[
        ('age', StandardScaler(), ['AGE_YRS']),
        ('sex', OneHotEncoder(drop='first', sparse_output=False), ['SEX']),
    ])
X_meta_processed = preprocessor.fit_transform(X_age_sex)

# 處理文字向量與 PCA 降維
X_vector = np.array(df['article_vector'].tolist())
X_vector = X_vector.reshape(X_vector.shape[0], -1) 

print("\n執行 PCA 降維 (768 -> 128)...")
pca = PCA(n_components=128, random_state=42)
X_text_reduced = pca.fit_transform(X_vector)

print("\n切分訓練集與測試集...")
X_text_tr, X_text_te, X_meta_tr, X_meta_te, Y_tr, Y_te = train_test_split(
    X_text_reduced, X_meta_processed, Y, test_size=0.2, random_state=42, stratify=Y
)

print("\nRVKDE 特徵轉換 (single)...")
dim_pca = 128
K2 = 10
backend_choice = "faiss"

nn_model = RVKDE.build_nn_kernels(X_text_tr, K2=K2+1, backend=backend_choice)
sigmas, _ = RVKDE.rvkde_sigmas(X_text_tr, beta=1.0, dim=dim_pca, backend=backend_choice)

pdf_tr = RVKDE.cross_group_density_pairwise(
    X_query=X_text_tr, X_kernels=X_text_tr, sigmas_k=sigmas,
    nn=nn_model, K2=K2, dim=dim_pca, same_dataset=True
).reshape(-1, 1)

pdf_te = RVKDE.cross_group_density_pairwise(
    X_query=X_text_te, X_kernels=X_text_tr, sigmas_k=sigmas,
    nn=nn_model, K2=K2, dim=dim_pca, same_dataset=False
).reshape(-1, 1)

X_tr_final = np.hstack([pdf_tr, X_meta_tr])
X_te_final = np.hstack([pdf_te, X_meta_te])

print("\n套用 Optuna 算出的最佳參數進行訓練...")
sampler = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
X_train_res, Y_train_res = sampler.fit_resample(X_tr_final, Y_tr)

trials_df = pd.read_csv('svm_rvkde_single_aucpr_optuna.csv') 
best_trial = trials_df.sort_values(by='value', ascending=False).iloc[0]

best_kernel = str(best_trial.get('params_kernel', 'rbf'))

raw_gamma = best_trial.get('params_gamma', 'scale')
if str(raw_gamma) in ['scale', 'auto']:
    final_gamma = str(raw_gamma)
else:
    if pd.isna(raw_gamma):
        final_gamma = 'scale'
    else:
        final_gamma = float(raw_gamma)

raw_cw = best_trial.get('params_class_weight', None)
if pd.isna(raw_cw) or str(raw_cw) == 'None' or str(raw_cw) == 'nan':
    final_cw = None
else:
    final_cw = str(raw_cw)

best_params = {
    'kernel': best_kernel,
    'C': float(best_trial['params_C']), 
    'gamma': final_gamma,
    'class_weight': final_cw
}

print(f"抓取最佳參數成功：{best_params}")

clf = SVC(
    **best_params,
    probability=True, 
    random_state=42
)
clf.fit(X_train_res, Y_train_res)

print("\n繪製並儲存 ROC 與 PR 曲線圖...")
Y_pred_proba = clf.predict_proba(X_te_final)[:, 1]

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

fpr, tpr, _ = roc_curve(Y_te, Y_pred_proba)
roc_auc = auc(fpr, tpr)
ax1.plot(fpr, tpr, color='crimson', lw=2.5, label=f'RVKDE(Single)+SVM (AUC = {roc_auc:.4f})')
ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel('偽陽性率 (False Positive Rate)', fontsize=12)
ax1.set_ylabel('真陽性率 (True Positive Rate)', fontsize=12)
ax1.set_title('ROC 曲線：模型整體鑑別力', fontsize=15, fontweight='bold')
ax1.legend(loc="lower right", fontsize=12)
ax1.grid(alpha=0.3)

precision, recall, _ = precision_recall_curve(Y_te, Y_pred_proba)
pr_auc = average_precision_score(Y_te, Y_pred_proba)
ax2.plot(recall, precision, color='forestgreen', lw=2.5, label=f'RVKDE(Single)+SVM (AUC-PR = {pr_auc:.4f})')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('召回率 (Recall)', fontsize=12)
ax2.set_ylabel('精確度 (Precision)', fontsize=12)
ax2.set_title('PR 曲線：不平衡資料下的極限表現', fontsize=15, fontweight='bold')
ax2.legend(loc="lower left", fontsize=12)
ax2.grid(alpha=0.3)

plt.tight_layout()
output_filename = 'final_svm_rvkde_single.png' 
plt.savefig(output_filename, dpi=300)

print(f"\n圖表已成功儲存為：【{output_filename}】")