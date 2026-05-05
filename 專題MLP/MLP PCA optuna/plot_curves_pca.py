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
from sklearn.neural_network import MLPClassifier

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

print("\nPCA 降維 (768 -> 128)...")
pca = PCA(n_components=128, random_state=42)
X_text_reduced = pca.fit_transform(X_vector)

print("\n切分訓練集與測試集...")
X_text_tr, X_text_te, X_meta_tr, X_meta_te, Y_tr, Y_te = train_test_split(
    X_text_reduced, X_meta_processed, Y, test_size=0.2, random_state=42, stratify=Y
)

print("\n組合特徵 (PCA 128維 + 年齡/性別)...")
X_tr_final = np.hstack([X_text_tr, X_meta_tr])
X_te_final = np.hstack([X_text_te, X_meta_te])

print("\n套用 Optuna 算出的最佳參數進行訓練...")
sampler = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
X_train_res, Y_train_res = sampler.fit_resample(X_tr_final, Y_tr)

# 讀取Optuna紀錄檔 (請確保檔名正確)
trials_df = pd.read_csv('mlp_aucpr_optuna.csv')
best_trial = trials_df.sort_values(by='value', ascending=False).iloc[0]

# 動態組裝 hidden_layer_sizes
n_layers = int(best_trial['n_layers'])
layers = []
for i in range(n_layers):
    layers.append(int(best_trial[f'n_units_layer_{i}']))
hidden_layer_sizes = tuple(layers)

best_params = {
    'hidden_layer_sizes': hidden_layer_sizes,
    'alpha': best_trial['alpha'],
    'learning_rate_init': best_trial['learning_rate_init'],
    'activation': best_trial['activation']
}

print(f"抓取最佳參數成功：{best_params}")

clf = MLPClassifier(
    **best_params,
    solver='adam',
    max_iter=500,
    random_state=42,
    early_stopping=True 
)
clf.fit(X_train_res, Y_train_res)

print("\n繪製並儲存 ROC 與 PR 曲線圖...")
Y_pred_proba = clf.predict_proba(X_te_final)[:, 1]

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

fpr, tpr, _ = roc_curve(Y_te, Y_pred_proba)
roc_auc = auc(fpr, tpr)

ax1.plot(fpr, tpr, color='royalblue', lw=2.5, label=f'PCA+MLP (Baseline) (AUC = {roc_auc:.4f})')
ax1.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel('偽陽性率 (False Positive Rate)', fontsize=12)
ax1.set_ylabel('真陽性率 (True Positive Rate)', fontsize=12)
ax1.set_title('ROC 曲線：模型整體鑑別力', fontsize=15, fontweight='bold')
ax1.legend(loc="lower right", fontsize=12)
ax1.grid(alpha=0.3)

precision, recall, _ = precision_recall_curve(Y_te, Y_pred_proba)
pr_auc = average_precision_score(Y_te, Y_pred_proba)

ax2.plot(recall, precision, color='darkorange', lw=2.5, label=f'PCA+MLP (Baseline) (AUC-PR = {pr_auc:.4f})')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('召回率 (Recall)', fontsize=12)
ax2.set_ylabel('精確度 (Precision)', fontsize=12)
ax2.set_title('PR 曲線：不平衡資料下的極限表現', fontsize=15, fontweight='bold')
ax2.legend(loc="lower left", fontsize=12)
ax2.grid(alpha=0.3)

plt.tight_layout()
output_filename = 'final_mlp_pca_baseline.png' 
plt.savefig(output_filename, dpi=300)

print(f"\n圖表已成功儲存為：【{output_filename}】")