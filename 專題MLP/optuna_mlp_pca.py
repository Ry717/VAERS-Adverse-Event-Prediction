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
def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

# 自定義類別：同時將 stdout 寫入檔案和螢幕
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

# 把所有 print 的內容同步到文字檔
output_file = open('mlp_optuna_output.txt', 'w', encoding='utf-8-sig')
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
Y = (merged_df[target_columns] == 1).any(axis=1).astype(int)

# 提取特徵
X_vector = np.array(merged_df['article_vector'].tolist())
X_vector = X_vector.reshape(X_vector.shape[0], -1) 

print("\nPCA 降維")
pca = PCA(n_components=128, random_state=42)
X_vector_reduced = pca.fit_transform(X_vector)
explained_variance = sum(pca.explained_variance_ratio_)
print(f"成功降至 128 維！(保留了原始文字向量 {explained_variance*100:.2f}% 的資訊量)\n")

# 特徵預處理：年齡標準化 + 性別 One-Hot
X_age_sex = merged_df[['AGE_YRS', 'SEX']]
preprocessor = ColumnTransformer(
    transformers=[
        ('age', StandardScaler(), ['AGE_YRS']),
        ('sex', OneHotEncoder(drop='first', sparse_output=False), ['SEX']),
    ])
X_age_sex_processed = preprocessor.fit_transform(X_age_sex)

# 結合降維後的文字向量與傳統變數
X = np.hstack([X_vector_reduced, X_age_sex_processed])

def objective(trial):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    layers = []
    for i in range(n_layers):
        layers.append(trial.suggest_int(f'n_units_layer_{i}', 32, 128))
    hidden_layer_sizes = tuple(layers)
    
    alpha = trial.suggest_float('alpha', 1e-5, 1e-2, log=True)
    learning_rate_init = trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True)
    activation = trial.suggest_categorical('activation', ['relu', 'tanh'])
    sampler_name = trial.suggest_categorical("sampler", ["undersampling"]) # 處理不平衡資料
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 指標儲存區
    precision_pos_list, recall_pos_list, f1_pos_list = [], [], []
    precision_neg_list, recall_neg_list, f1_neg_list = [], [], []
    tn_list, tp_list = [], []
    auc_list, aucpr_list = [], []
    logloss_list = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, Y)):
        X_tr, Y_tr = X[train_idx], Y[train_idx]
        X_va, Y_va = X[val_idx], Y[val_idx]

        # 採樣
        if sampler_name == "undersampling":
            rus = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
            X_res, Y_res = rus.fit_resample(X_tr, Y_tr)
        else:
            X_res, Y_res = X_tr, Y_tr

        clf = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            solver='adam',
            max_iter=300,
            random_state=42,
            early_stopping=True # 啟動內部早停機制，加速實驗
        )
        clf.fit(X_res, Y_res)

        preds_va = clf.predict(X_va)
        proba_va = clf.predict_proba(X_va)[:, 1]
        
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

    # 寫入 User Attributes
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

print("開始進行 MLP 超參數優化...")
study = optuna.create_study(direction="maximize", study_name="MLP_PCA_Optuna")
study.optimize(objective, n_trials=50) # MLP較花時間，建議先測 50 或 100 次

print(f"\n最佳試驗編號: {study.best_trial.number}")
print(f"最佳 AUCPR : {study.best_value:.4f}")
print("最佳大腦架構與參數:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

records = []
for trial in study.trials:
    rec = {
        "trial_number": trial.number,
        "value": trial.value,
    }
    # 直接把 trial.params 裡面的所有超參數倒進去字典裡
    # 這樣無論 Optuna 這次配了 1 層還是 3 層，程式都不會漏抓
    rec.update(trial.params)
    rec.update(trial.user_attrs)
    records.append(rec)

df_all = pd.DataFrame(records)
df_all.to_csv("mlp_aucpr_optuna.csv", index=False, encoding="utf-8-sig")
print("\n所有 trial 已成功儲存至 mlp_aucpr_optuna.csv")

sys.stdout = sys.stdout.stdout
output_file.close()