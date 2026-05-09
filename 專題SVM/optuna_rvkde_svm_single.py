import pandas as pd 
import numpy as np
import ast
import sys
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, roc_auc_score, average_precision_score, 
                             precision_score, recall_score, f1_score, log_loss, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import RVKDE  
import locale

def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

class Tee:
    def __init__(self, file):
        self.file = file
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

output_file = open('svm_rvkde_single_optuna_output.txt', 'w', encoding='utf-8-sig')
sys.stdout = Tee(output_file)

print("讀取資料中...")
merged_df = pd.read_csv("merged_data_bert.csv")
original_count = len(merged_df)

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

print(f"原始資料筆數 : {original_count}")
print(f"移除目標欄位缺失值後的資料筆數 : {len(merged_df)}\n")

X_vector = X_vector[merged_df.index]
X_age_sex = X_age_sex.loc[merged_df.index]

preprocessor = ColumnTransformer(
    transformers=[
        ('age', StandardScaler(), ['AGE_YRS']),
        ('sex', OneHotEncoder(drop='first', sparse_output=False), ['SEX']),
    ])
X_age_sex_processed = preprocessor.fit_transform(X_age_sex)

print("PCA 降維...")
pca = PCA(n_components=128, random_state=42)
X_vector_reduced = pca.fit_transform(X_vector)
explained_variance = sum(pca.explained_variance_ratio_)
print(f"降至 128 維！(保留了原始文字向量 {explained_variance*100:.2f}% 的資訊量)\n")

print("切分資料集 (80% 訓練, 20% 測試)...")
X_train_text, X_test_text, age_sex_train, age_sex_test, y_train, y_test = train_test_split(
    X_vector_reduced, X_age_sex_processed, Y, test_size=0.2, random_state=42, stratify=Y
)

print("\n將 128 維壓縮至 1 維 PDF...")
dim_pca = 128
K2 = 10
backend_choice = "faiss" 

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

print("組合最終特徵並進行欠採樣...")
X_train_final = np.hstack([pdf_train, age_sex_train])
X_test_final = np.hstack([pdf_test, age_sex_test])
print(f"SVM 模型的特徵剩：{X_train_final.shape[1]} 維 (PDF + 年齡 + 性別)\n")

rus = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
X_train_resampled, y_train_resampled = rus.fit_resample(X_train_final, y_train)

print("進行 SVM + RVKDE(single) 超參數優化...\n")

def objective(trial):
    svm_kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
    svm_c = trial.suggest_float("C", 1e-3, 1e2, log=True)
    
    if svm_kernel == "rbf":
        svm_gamma = trial.suggest_categorical("gamma", ["scale", "auto", 0.1, 0.01, 0.001])
    else:
        svm_gamma = "scale"
        
    svm_class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])

    model = SVC(
        kernel=svm_kernel,
        C=svm_c,
        gamma=svm_gamma,
        class_weight=svm_class_weight,
        probability=True,
        random_state=42
    )
    
    model.fit(X_train_resampled, y_train_resampled)
    y_pred_proba = model.predict_proba(X_test_final)[:, 1]
    
    aucpr = average_precision_score(y_test, y_pred_proba)
    return aucpr

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=True)

# 提取最佳參數重訓最終模型
best_params = study.best_params
final_gamma = best_params.get("gamma", "scale") 

final_svm_model = SVC(
    kernel=best_params["kernel"],
    C=best_params["C"],
    gamma=final_gamma,
    class_weight=best_params["class_weight"],
    probability=True,
    random_state=42
)
final_svm_model.fit(X_train_resampled, y_train_resampled) 

y_pred_best = final_svm_model.predict(X_test_final)
y_pred_proba_best = final_svm_model.predict_proba(X_test_final)[:, 1]

# 計算所有詳細指標
aucpr_best = average_precision_score(y_test, y_pred_proba_best)
auc_best = roc_auc_score(y_test, y_pred_proba_best)
lloss = log_loss(y_test, y_pred_proba_best)

precision_pos = precision_score(y_test, y_pred_best, pos_label=1)
recall_pos = recall_score(y_test, y_pred_best, pos_label=1)
f1_pos = f1_score(y_test, y_pred_best, pos_label=1)

precision_neg = precision_score(y_test, y_pred_best, pos_label=0)
recall_neg = recall_score(y_test, y_pred_best, pos_label=0)
f1_neg = f1_score(y_test, y_pred_best, pos_label=0)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_best).ravel()

print(f"\n最佳試驗編號: {study.best_trial.number}")
print(f"最佳 AUCPR : {aucpr_best:.4f}")
print("最佳架構與參數:")
for key, value in best_params.items():
    print(f"    {key}: {value}")

print("\n最佳試驗詳細指標:")
print(f"    正類 Precision (precision_pos): {precision_pos:.4f}")
print(f"    正類 Recall    (recall_pos)   : {recall_pos:.4f}")
print(f"    正類 F1-score  (f1_pos)       : {f1_pos:.4f}")
print(f"    負類 Precision (precision_neg): {precision_neg:.4f}")
print(f"    負類 Recall    (recall_neg)   : {recall_neg:.4f}")
print(f"    負類 F1-score  (f1_neg)       : {f1_neg:.4f}")
print(f"    ROC-AUC        (auc)          : {auc_best:.4f}")
print(f"    真陽性         (TP)           : {tp} 人")
print(f"    真陰性         (TN)           : {tn} 人")
print(f"    Log Loss       (logloss)      : {lloss:.4f}")

csv_filename = "svm_rvkde_single_aucpr_optuna.csv"
study.trials_dataframe().to_csv(csv_filename, index=False)
print(f"\n所有 trial 已成功儲存至 {csv_filename}")

sys.stdout = sys.stdout.stdout
output_file.close()