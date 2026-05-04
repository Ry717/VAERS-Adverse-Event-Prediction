import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler

print("讀取 BERT 嵌入向量與資料...")
df = pd.read_csv("merged_data_bert.csv", encoding="utf-8-sig")

# 將字串格式的陣列還原回Python列表
df['article_vector'] = df['article_vector'].apply(ast.literal_eval)
X_vector = np.array(df['article_vector'].tolist())

# 設定目標Y(已經清洗好的1和0)
target_columns = ['DIED', 'L_THREAT', 'ER_VISIT', 'HOSPITAL', 'X_STAY', 'ER_ED_VISIT']
Y = (df[target_columns] == 1).any(axis=1).astype(int)

# 處理年齡與性別
X_age_sex = df[['AGE_YRS', 'SEX']].copy()
X_age_sex['AGE_YRS'] = X_age_sex['AGE_YRS'].fillna(X_age_sex['AGE_YRS'].median())
X_age_sex['SEX'] = X_age_sex['SEX'].fillna('U')

preprocessor = ColumnTransformer(
    transformers=[
        ('age', StandardScaler(), ['AGE_YRS']),
        ('sex', OneHotEncoder(drop='first', sparse_output=False), ['SEX']),
    ])
X_age_sex_processed = preprocessor.fit_transform(X_age_sex)

# 結合BERT向量與基本特徵
X = np.hstack([X_vector, X_age_sex_processed])

print("切分訓練集與測試集...")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

print("套用 Optuna 算出的參數進行極限訓練...")
# 採樣策略：undersampling
sampler = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
X_train_res, Y_train_res = sampler.fit_resample(X_train, Y_train)

# 讀取Optuna匯出的300次試驗紀錄CSV
trials_df = pd.read_csv('aucpr_300.csv')

# 自動找出分數最高的那一筆試驗
best_trial = trials_df.sort_values(by='value', ascending=False).iloc[0]

# 把最佳參數提取成一個字典
best_params = {
    'alpha': best_trial['alpha'],
    'min_child_weight': best_trial['min_child_weight'],
    'n_estimators': int(best_trial['n_estimators']),
    'learning_rate': best_trial['learning_rate'],
    'scale_pos_weight': best_trial['scale_pos_weight']
}

print(f"抓取最佳參數成功：{best_params}")

# 使用 **best_params 直接把字典解包塞進XGBoost
clf = XGBClassifier(
    **best_params, 
    random_state=42,
    eval_metric='logloss',
    early_stopping_rounds=50
)
clf.fit(
    X_train_res, Y_train_res,
    eval_set=[(X_test, Y_test)],  
    verbose=True  # 打開文字輸出，看在哪一棵樹煞車
)

print("繪製並儲存 ROC 與 PR 曲線圖...")
Y_pred_proba = clf.predict_proba(X_test)[:, 1]

# 設定圖表字體，避免中文亂碼
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 建立畫布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 第一張圖：ROC 曲線
fpr, tpr, _ = roc_curve(Y_test, Y_pred_proba)
roc_auc = auc(fpr, tpr)
ax1.plot(fpr, tpr, color='crimson', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})')
ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel('偽陽性率 (False Positive Rate)', fontsize=12)
ax1.set_ylabel('真陽性率 (True Positive Rate)', fontsize=12)
ax1.set_title('ROC 曲線：模型整體鑑別力', fontsize=15, fontweight='bold')
ax1.legend(loc="lower right", fontsize=12)
ax1.grid(alpha=0.3)

# 第二張圖：PR 曲線
precision, recall, _ = precision_recall_curve(Y_test, Y_pred_proba)
pr_auc = average_precision_score(Y_test, Y_pred_proba)
ax2.plot(recall, precision, color='forestgreen', lw=2.5, label=f'PR Curve (AUC-PR = {pr_auc:.4f})')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('召回率 (Recall)', fontsize=12)
ax2.set_ylabel('精確度 (Precision)', fontsize=12)
ax2.set_title('PR 曲線：不平衡資料下的極限表現', fontsize=15, fontweight='bold')
ax2.legend(loc="lower left", fontsize=12)
ax2.grid(alpha=0.3)

# 儲存
plt.tight_layout()
output_filename = 'final.png'
plt.savefig(output_filename, dpi=300)

print(f"\n圖表已成功儲存為：【{output_filename}】")