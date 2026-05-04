import pandas as pd 
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import sys
from io import StringIO
import optuna
from sklearn.model_selection import StratifiedKFold

# 設置 UTF-8 編碼以避免中文亂碼
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
output_file = open('predict_output_spw_alpha.txt', 'w', encoding='utf-8-sig')
sys.stdout = Tee(output_file)

# 讀取資料
merged_df = pd.read_csv("merged_data_bert.csv")
print(f"原始資料筆數：{len(merged_df)}")

# 轉換 article_vector 欄位為 list
def parse_vector(vec):
    try:
        return ast.literal_eval(vec)
    except Exception as e:
        print("轉換失敗:", vec, e)
        return None

merged_df['article_vector'] = merged_df['article_vector'].apply(parse_vector)
merged_df = merged_df[merged_df['article_vector'].notnull()].reset_index(drop=True)
print(f"移除無效 article_vector 後的資料筆數：{len(merged_df)}")

# 處理年齡和性別缺失值
merged_df['AGE_YRS'] = merged_df['AGE_YRS'].fillna(merged_df['AGE_YRS'].median())
merged_df['SEX'] = merged_df['SEX'].fillna('U')

# 提取 article_vector、年齡、性別特徵
X_vector = np.array(merged_df['article_vector'].tolist())
X_vector = X_vector.reshape(X_vector.shape[0], -1)  # 攤平成一維

X_age_sex = merged_df[['AGE_YRS', 'SEX']]

# 目標欄位：只要任一欄為1即為正類
target_columns = ['DIED', 'L_THREAT', 'ER_VISIT', 'HOSPITAL', 'X_STAY', 'ER_ED_VISIT']
merged_df = merged_df.dropna(subset=target_columns).reset_index(drop=True)
print(f"移除目標欄位缺失值後的資料筆數：{len(merged_df)}")
Y = (merged_df[target_columns] == 1).any(axis=1).astype(int)

# 同步 X_vector & X_age_sex
X_vector = X_vector[merged_df.index]
X_age_sex = X_age_sex.loc[merged_df.index]

# 特徵預處理：年齡標準化 + 性別 One-Hot
preprocessor = ColumnTransformer(
    transformers=[
        ('age', StandardScaler(), ['AGE_YRS']),
        ('sex', OneHotEncoder(drop='first', sparse_output=False), ['SEX']),
    ])
X_age_sex_processed = preprocessor.fit_transform(X_age_sex)
X = np.hstack([X_vector, X_age_sex_processed])

# 參數空間設定
alpha_list = list(np.arange(1.0, 4.0 + 1e-8, 0.2))  # 1.0, 1.2, ..., 4.0
mcw_list   = list(np.arange(0.2, 1.7 + 1e-8, 0.05)) # 0.2, 0.25, ..., 1.7
n_estimators_list = [100, 200, 250, 275, 300, 325, 350, 400, 500, 800, 1000]
learning_rate_list = [0.3, 0.2, 0.1, 0.07, 0.06, 0.05, 0.04, 0.03, 0.01]
spw_list   = list(np.arange(1, 6.0 + 1e-8, 0.1))    # 0.5, 0.6, ..., 5.0

subsample = 0.8
random_state = 42

# 三種採樣策略：None、undersampling、oversampling
samplers = {
    "none": None,
    "undersampling": RandomUnderSampler(sampling_strategy=1.0, random_state=42),
    "oversampling": SMOTE(sampling_strategy=1.0, random_state=42)
}

def objective(trial):
    # 把超參數從 trial 拿出來
    alpha = trial.suggest_float("alpha", 0.1, 4.0, step=0.1)
    mcw   = trial.suggest_float("min_child_weight", 0.1, 3.0, step=0.05)
    n_est = trial.suggest_categorical("n_estimators",
                                      [100, 200, 250, 275, 300, 325, 350, 400, 500, 800, 1000])
    lr    = trial.suggest_categorical("learning_rate",
                                      [0.3, 0.2, 0.1, 0.07, 0.06, 0.05, 0.04, 0.03, 0.01])
    spw   = trial.suggest_float("scale_pos_weight", 0.1, 4.0, step=0.1)
    sampler_name = trial.suggest_categorical("sampler",
                                             ["undersampling"])
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 用於儲存每折的各種指標
    precision_pos_list, recall_pos_list, f1_pos_list = [], [], []
    precision_neg_list, recall_neg_list, f1_neg_list = [], [], []
    tn_list, tp_list = [], []
    auc_list, aucpr_list = [], []
    logloss_list = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, Y)):
        X_tr, Y_tr = X[train_idx], Y[train_idx]
        X_va, Y_va = X[val_idx], Y[val_idx]

        # 對這一折的訓練子集做採樣
        if sampler_name == "none":
            X_res, Y_res = X_tr, Y_tr
        elif sampler_name == "undersampling":
            rus = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
            X_res, Y_res = rus.fit_resample(X_tr, Y_tr)
        else:  # oversampling
            sm = SMOTE(sampling_strategy=1.0, random_state=42)
            X_res, Y_res = sm.fit_resample(X_tr, Y_tr)

        # 用 resample 後的資料訓練模型
        clf = XGBClassifier(
            alpha=alpha,
            min_child_weight=mcw,
            n_estimators=n_est,
            learning_rate=lr,
            scale_pos_weight=spw,
            subsample=0.8,
            random_state=42,
            eval_metric="logloss",
            tree_method='hist',     
            device='cuda:0',  
            early_stopping_rounds=50,        
        )
        clf.fit(
            X_res, Y_res,
            eval_set=[(X_va, Y_va)],  
            verbose=False             # 關閉每棵樹的文字輸出，保持終端機畫面乾淨
        )

        # 在這一折的驗證子集上預測
        preds_va = clf.predict(X_va)
        proba_va = clf.predict_proba(X_va)[:, 1]  # 取得正類機率
        
        # 計算正類指標
        prec_pos = precision_score(Y_va, preds_va, pos_label=1, zero_division=0)
        rec_pos  = recall_score(Y_va, preds_va, pos_label=1, zero_division=0)
        f1_pos   = f1_score(Y_va, preds_va, pos_label=1, zero_division=0)
        
        # 計算負類指標
        prec_neg = precision_score(Y_va, preds_va, pos_label=0, zero_division=0)
        rec_neg  = recall_score(Y_va, preds_va, pos_label=0, zero_division=0)
        f1_neg   = f1_score(Y_va, preds_va, pos_label=0, zero_division=0)
        
        # 計算混淆矩陣
        cm = confusion_matrix(Y_va, preds_va)
        tn, fp, fn, tp = cm.ravel()
        
        # 計算 AUC、AUC-PR、Logloss
        auc = roc_auc_score(Y_va, proba_va)
        aucpr = average_precision_score(Y_va, proba_va)
        ll = log_loss(Y_va, proba_va)
        
        # 儲存這一折的結果
        precision_pos_list.append(prec_pos)
        recall_pos_list.append(rec_pos)
        f1_pos_list.append(f1_pos)
        precision_neg_list.append(prec_neg)
        recall_neg_list.append(rec_neg)
        f1_neg_list.append(f1_neg)
        tn_list.append(tn)
        tp_list.append(tp)
        auc_list.append(auc)
        aucpr_list.append(aucpr)
        logloss_list.append(ll)

    # 計算5折的平均值並設為 user attributes
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

    # 回傳要優化的目標值
    return np.mean(aucpr_list)

# 執行 Optuna 優化
print("開始進行超參數優化...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=300)

print(f"\n最佳試驗編號: {study.best_trial.number}")
print(f"最佳 : {study.best_value:.4f}")
print("最佳參數:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

print("\n最佳試驗的詳細指標:")
best_attrs = study.best_trial.user_attrs
for key, value in best_attrs.items():
    print(f"  {key}: {value:.4f}")

# 建立所有試驗的結果 DataFrame
records = []

# 逐一讀取每個 trial
for trial in study.trials:
    rec = {
        "trial_number": trial.number,
        "value": trial.value,
        # 把所有參數都讀出來
        "alpha": trial.params.get("alpha"),
        "min_child_weight": trial.params.get("min_child_weight"),
        "n_estimators": trial.params.get("n_estimators"),
        "learning_rate": trial.params.get("learning_rate"),
        "scale_pos_weight": trial.params.get("scale_pos_weight"),
        "sampler": trial.params.get("sampler"),
        # 再把 user_attrs 也都讀出來
        "precision_pos": trial.user_attrs.get("precision_pos"),
        "recall_pos":    trial.user_attrs.get("recall_pos"),
        "f1_pos":        trial.user_attrs.get("f1_pos"),
        "precision_neg": trial.user_attrs.get("precision_neg"),
        "recall_neg":    trial.user_attrs.get("recall_neg"),
        "f1_neg":        trial.user_attrs.get("f1_neg"),
        "tn_avg":        trial.user_attrs.get("tn_avg"),
        "tp_avg":        trial.user_attrs.get("tp_avg"),
        "auc":           trial.user_attrs.get("auc"),
        "aucpr":         trial.user_attrs.get("aucpr"),
        "logloss":       trial.user_attrs.get("logloss")
    }
    records.append(rec)

# 把 list-of-dicts 轉成 DataFrame
df_all = pd.DataFrame(records)

# 檢查一下
print(f"\n前5筆試驗結果:")
print(df_all.head())
print(f"\n總共 {len(df_all)} 個 trial 記錄被匯出。")

# 最後寫成 CSV
df_all.to_csv("aucpr_300.csv", index=False, encoding="utf-8-sig")
print("所有 trial 已儲存至 aucpr_300.csv")

# 關閉檔案並恢復標準輸出
sys.stdout = sys.stdout.stdout
output_file.close()