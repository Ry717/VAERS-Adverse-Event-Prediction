import pandas as pd 
import numpy as np
import ast
import sys
import optuna
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import locale
import RVKDE 

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

output_file = open('mlp_rvkde_aucpr_optuna.txt', 'w', encoding='utf-8-sig')
sys.stdout = Tee(output_file)

merged_df = pd.read_csv("merged_data_bert.csv")
print(f"原始資料筆數：{len(merged_df)}")

def parse_vector(vec):
    try:
        return ast.literal_eval(vec)
    except Exception as e:
        return None

merged_df['article_vector'] = merged_df['article_vector'].apply(parse_vector)
merged_df = merged_df[merged_df['article_vector'].notnull()].reset_index(drop=True)

merged_df['AGE_YRS'] = merged_df['AGE_YRS'].fillna(merged_df['AGE_YRS'].median())
merged_df['SEX'] = merged_df['SEX'].fillna('U')

target_columns = ['DIED', 'L_THREAT', 'ER_VISIT', 'HOSPITAL', 'X_STAY', 'ER_ED_VISIT']
merged_df = merged_df.dropna(subset=target_columns).reset_index(drop=True)
print(f"移除目標欄位缺失值後的資料筆數：{len(merged_df)}")
Y = (merged_df[target_columns] == 1).any(axis=1).astype(int).values # 轉成 numpy array

X_vector = np.array(merged_df['article_vector'].tolist())
X_vector = X_vector.reshape(X_vector.shape[0], -1) 

print("\nPCA 降維")
pca = PCA(n_components=128, random_state=42)
X_text_reduced = pca.fit_transform(X_vector)
explained_variance = sum(pca.explained_variance_ratio_)
print(f"成功降至 128 維！(保留了原始文字向量 {explained_variance*100:.2f}% 的資訊量)\n")

X_age_sex = merged_df[['AGE_YRS', 'SEX']]
preprocessor = ColumnTransformer(
    transformers=[
        ('age', StandardScaler(), ['AGE_YRS']),
        ('sex', OneHotEncoder(drop='first', sparse_output=False), ['SEX']),
    ])
X_meta_processed = preprocessor.fit_transform(X_age_sex)

def objective(trial):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    layers = []
    for i in range(n_layers):
        layers.append(trial.suggest_int(f'n_units_layer_{i}', 32, 128))
    hidden_layer_sizes = tuple(layers)
    
    alpha = trial.suggest_float('alpha', 1e-5, 1e-2, log=True)
    learning_rate_init = trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True)
    activation = trial.suggest_categorical('activation', ['relu', 'tanh'])
    sampler_name = trial.suggest_categorical("sampler", ["undersampling"]) 
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    precision_pos_list, recall_pos_list, f1_pos_list = [], [], []
    precision_neg_list, recall_neg_list, f1_neg_list = [], [], []
    tn_list, tp_list = [], []
    auc_list, aucpr_list = [], []
    logloss_list = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(Y)), Y)):
        
        X_text_tr, X_text_va = X_text_reduced[train_idx], X_text_reduced[val_idx]
        X_meta_tr, X_meta_va = X_meta_processed[train_idx], X_meta_processed[val_idx]
        Y_tr, Y_va = Y[train_idx], Y[val_idx]

        dim_pca = 128
        K2 = 10
        backend_choice = "faiss"

        # 將訓練集依照 Y_tr (0與1) 拆開
        X_tr_pos = X_text_tr[Y_tr == 1]
        X_tr_neg = X_text_tr[Y_tr == 0]

        # 分別建立「正類」與「負類」的近鄰搜尋樹與 Sigmas
        nn_pos = RVKDE.build_nn_kernels(X_tr_pos, K2=K2+1, backend=backend_choice)
        sigmas_pos, _ = RVKDE.rvkde_sigmas(X_tr_pos, beta=1.0, dim=dim_pca, backend=backend_choice)

        nn_neg = RVKDE.build_nn_kernels(X_tr_neg, K2=K2+1, backend=backend_choice)
        sigmas_neg, _ = RVKDE.rvkde_sigmas(X_tr_neg, beta=1.0, dim=dim_pca, backend=backend_choice)

        # 計算 Query 點(訓練集本身) 對正類與負類的 PDF
        pdf_tr_pos = RVKDE.cross_group_density_pairwise(X_query=X_text_tr, X_kernels=X_tr_pos, sigmas_k=sigmas_pos, nn=nn_pos, K2=K2, dim=dim_pca, same_dataset=False).reshape(-1, 1)
        pdf_tr_neg = RVKDE.cross_group_density_pairwise(X_query=X_text_tr, X_kernels=X_tr_neg, sigmas_k=sigmas_neg, nn=nn_neg, K2=K2, dim=dim_pca, same_dataset=False).reshape(-1, 1)

        # 計算 Query 點(驗證集) 對正類與負類的 PDF
        pdf_va_pos = RVKDE.cross_group_density_pairwise(X_query=X_text_va, X_kernels=X_tr_pos, sigmas_k=sigmas_pos, nn=nn_pos, K2=K2, dim=dim_pca, same_dataset=False).reshape(-1, 1)
        pdf_va_neg = RVKDE.cross_group_density_pairwise(X_query=X_text_va, X_kernels=X_tr_neg, sigmas_k=sigmas_neg, nn=nn_neg, K2=K2, dim=dim_pca, same_dataset=False).reshape(-1, 1)

        # 組合最終的 5 維特徵 (正類PDF + 負類PDF + PCA原本特徵 + 年齡 + 性別)
        X_tr_final = np.hstack([pdf_tr_pos, pdf_tr_neg, X_meta_tr])
        X_va_final = np.hstack([pdf_va_pos, pdf_va_neg, X_meta_va])
        # -------------------------------------------------------------

        # 採樣處理
        if sampler_name == "undersampling":
            rus = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
            X_res, Y_res = rus.fit_resample(X_tr_final, Y_tr)
        else:
            X_res, Y_res = X_tr_final, Y_tr

        clf = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            solver='adam',
            max_iter=300,
            random_state=42,
            early_stopping=True 
        )
        clf.fit(X_res, Y_res)

        # 預測驗證集
        preds_va = clf.predict(X_va_final)
        proba_va = clf.predict_proba(X_va_final)[:, 1]
        
        # 結算分數
        precision_pos_list.append(precision_score(Y_va, preds_va, pos_label=1, zero_division=0))
        recall_pos_list.append(recall_score(Y_va, preds_va, pos_label=1, zero_division=0))
        f1_pos_list.append(f1_score(Y_va, preds_va, pos_label=1, zero_division=0))
        precision_neg_list.append(precision_score(Y_va, preds_va, pos_label=0, zero_division=0))
        recall_neg_list.append(recall_score(Y_va, preds_va, pos_label=0, zero_division=0))
        f1_neg_list.append(f1_score(Y_va, preds_va, pos_label=0, zero_division=0))
        
        cm = confusion_matrix(Y_va, preds_va)
        tn_list.append(cm.ravel()[0])
        tp_list.append(cm.ravel()[3])
        auc_list.append(roc_auc_score(Y_va, proba_va))
        aucpr_list.append(average_precision_score(Y_va, proba_va))
        logloss_list.append(log_loss(Y_va, proba_va))

    trial.set_user_attr("precision_pos", np.mean(precision_pos_list))
    trial.set_user_attr("recall_pos",    np.mean(recall_pos_list))
    trial.set_user_attr("f1_pos",        np.mean(f1_pos_list))
    trial.set_user_attr("precision_neg", np.mean(precision_neg_list))
    trial.set_user_attr("recall_neg",    np.mean(recall_neg_list))
    trial.set_user_attr("f1_neg",        np.mean(f1_neg_list))
    trial.set_user_attr("tn_avg",        np.mean(tn_list))
    trial.set_user_attr("tp_avg",        np.mean(tp_list))
    trial.set_user_attr("auc",           np.mean(auc_list))
    trial.set_user_attr("aucpr",         np.mean(aucpr_list))
    trial.set_user_attr("logloss",       np.mean(logloss_list))

    return np.mean(aucpr_list)

print("進行 RVKDE + MLP 超參數優化...")
study = optuna.create_study(direction="maximize", study_name="RVKDE_MLP_Optuna")
study.optimize(objective, n_trials=50) 

print(f"\n最佳試驗編號: {study.best_trial.number}")
print(f"最佳 AUCPR : {study.best_value:.4f}")
print("最佳架構與參數:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

records = []
for trial in study.trials:
    rec = {
        "trial_number": trial.number,
        "value": trial.value,
    }
    rec.update(trial.params)
    rec.update(trial.user_attrs)
    records.append(rec)

df_all = pd.DataFrame(records)
df_all.to_csv("mlp_rvkde_dual_aucpr_optuna.csv", index=False, encoding="utf-8-sig")
print("\n所有 trial 已成功儲存至 mlp_rvkde_dual_aucpr_optuna.csv")

sys.stdout = sys.stdout.stdout
output_file.close()