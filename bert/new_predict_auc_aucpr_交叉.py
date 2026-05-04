"""
XGBoost 交叉驗證分析程式 - 包含採樣策略與正類權重優化
=====================================

此程式用於執行 XGBoost 分類器的 5 折交叉驗證，測試不同的：
1. 採樣策略：無採樣、欠採樣、過採樣 (SMOTE)
2. 正類權重 (scale_pos_weight) 設定
3. 評估多種分類指標：精確度、召回率、F1、AUC、AUC-PR、LogLoss

主要用途：醫療不良事件預測模型的超參數調優與性能評估
"""

import pandas as pd
import numpy as np
import ast
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

# =========================
# 編碼設定與輸出重導
# =========================
# 設置 UTF-8 編碼以避免中文亂碼
import locale
def getpreferredencoding(do_setlocale=True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

# 自定義類別：同時將輸出寫入檔案和螢幕
class Tee:
    """同時輸出到檔案和控制台的類別"""
    def __init__(self, file):
        self.file = file
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

# 設置輸出檔案，使用 UTF-8-SIG 編碼（Windows 相容）
output_file = open('predict_output.txt', 'w', encoding='utf-8-sig')
sys.stdout = Tee(output_file)

# =========================
# 資料讀取與前處理
# =========================
# 讀取合併後的資料集
merged_df = pd.read_csv("merged_data_dst.csv")
print(f"原始資料筆數：{len(merged_df)}")

# 轉換 article_vector 欄位為數值陣列
def parse_vector(vec):
    """
    將字串格式的向量轉換為 Python list
    
    Args:
        vec: 字串格式的向量 (如 "[1,2,3,...]")
    
    Returns:
        list 或 None (轉換失敗時)
    """
    try:
        return ast.literal_eval(vec)
    except Exception as e:
        print("轉換失敗:", vec, e)
        return None

# 應用向量轉換並移除無效資料
merged_df['article_vector'] = merged_df['article_vector'].apply(parse_vector)
merged_df = merged_df[merged_df['article_vector'].notnull()].reset_index(drop=True)
print(f"移除無效 article_vector 後的資料筆數：{len(merged_df)}")

# 處理缺失值
merged_df['AGE_YRS'] = merged_df['AGE_YRS'].fillna(merged_df['AGE_YRS'].median())  # 年齡用中位數填補
merged_df['SEX'] = merged_df['SEX'].fillna('U')  # 性別用 'U' (Unknown) 填補

# =========================
# 特徵工程
# =========================
# 提取文章向量特徵 (主要特徵)
X_vector = np.array(merged_df['article_vector'].tolist())
X_vector = X_vector.reshape(X_vector.shape[0], -1)  # 攤平成一維向量

# 提取人口統計學特徵 (年齡與性別)
X_age_sex = merged_df[['AGE_YRS', 'SEX']]

# =========================
# 目標變數處理
# =========================
# 定義目標欄位：各種不良醫療事件
target_columns = ['DIED', 'L_THREAT', 'ER_VISIT', 'HOSPITAL', 'X_STAY', 'ER_ED_VISIT']

# 移除目標欄位有缺失的資料
merged_df = merged_df.dropna(subset=target_columns).reset_index(drop=True)
print(f"移除目標欄位缺失值後的資料筆數：{len(merged_df)}")

# 建立複合目標：任一不良事件發生即標記為正類 (1)，否則為負類 (0)
Y = (merged_df[target_columns] == 1).any(axis=1).astype(int)

# 確保特徵矩陣與目標向量對齊
X_vector = X_vector[merged_df.index]
X_age_sex = X_age_sex.loc[merged_df.index]

# =========================
# 特徵預處理管道
# =========================
# 定義預處理轉換器
preprocessor = ColumnTransformer(
    transformers=[
        ('age', StandardScaler(), ['AGE_YRS']),  # 年齡標準化
        ('sex', OneHotEncoder(drop='first', sparse_output=False), ['SEX']),  # 性別 One-Hot 編碼
    ])

# 應用預處理並合併所有特徵
X_age_sex_processed = preprocessor.fit_transform(X_age_sex)
X = np.hstack([X_vector, X_age_sex_processed])  # 合併文章向量與人口統計學特徵

# =========================
# 實驗設計：採樣策略
# =========================
# 定義三種採樣策略
samplers = {
    'none': None,  # 原始資料分布
    'undersampling': RandomUnderSampler(sampling_strategy=1.0, random_state=42),  # 隨機欠採樣至 1:1
    'oversampling': SMOTE(sampling_strategy=1.0, random_state=42)  # SMOTE 過採樣至 1:1
}

# =========================
# 實驗設計：正類權重範圍
# =========================
# 定義要測試的正類權重範圍
# range1 = np.arange(0.7, 1.0, 0.05)  # 小於 1 的權重：0.7, 0.75, ..., 0.95
weight_range = np.arange(6.0, 6.05, 0.1)  # 當前測試範圍：6.0 附近
# range2 = np.arange(1.1, 5.5, 0.1)  # 大於 1 的權重：1.1, 1.2, ..., 5.4
# weight_range = np.concatenate([range1, range2])

# =========================
# 交叉驗證設定
# =========================
# 定義 5 折分層交叉驗證（保持正負類比例）
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# =========================
# 結果儲存結構初始化
# =========================
# 初始化結果字典，包含所有評估指標
results = {
    'scale_pos_weight': [f"{w:.2f}" if w >= 1.0 else f"{w:.2f}".lstrip("0") for w in weight_range],
    
    # 真陽性數量 (True Positives)
    'tp_none': [None] * len(weight_range),
    'tp_undersampling': [None] * len(weight_range),
    'tp_oversampling': [None] * len(weight_range),
    
    # 正類指標 (針對不良事件預測)
    'precision_none': [None] * len(weight_range),          # 精確度：預測為正類中實際為正類的比例
    'precision_undersampling': [None] * len(weight_range),
    'precision_oversampling': [None] * len(weight_range),
    
    'recall_none': [None] * len(weight_range),             # 召回率：實際正類中被正確預測的比例
    'recall_undersampling': [None] * len(weight_range),
    'recall_oversampling': [None] * len(weight_range),
    
    'f1_none': [None] * len(weight_range),                 # F1分數：精確度與召回率的調和平均
    'f1_undersampling': [None] * len(weight_range),
    'f1_oversampling': [None] * len(weight_range),
    
    # 真陰性數量 (True Negatives)
    'tn_none': [None] * len(weight_range),
    'tn_undersampling': [None] * len(weight_range),
    'tn_oversampling': [None] * len(weight_range),
    
    # 負類指標 (針對無不良事件預測)
    'precision_neg_none': [None] * len(weight_range),      # 負類精確度
    'precision_neg_undersampling': [None] * len(weight_range),
    'precision_neg_oversampling': [None] * len(weight_range),
    
    'recall_neg_none': [None] * len(weight_range),         # 負類召回率
    'recall_neg_undersampling': [None] * len(weight_range),
    'recall_neg_oversampling': [None] * len(weight_range),
    
    'f1_neg_none': [None] * len(weight_range),             # 負類 F1 分數
    'f1_neg_undersampling': [None] * len(weight_range),
    'f1_neg_oversampling': [None] * len(weight_range),
    
    # 進階評估指標
    'auc_none': [None] * len(weight_range),                # ROC-AUC：接收者操作特徵曲線下面積
    'auc_undersampling': [None] * len(weight_range),
    'auc_oversampling': [None] * len(weight_range),
    
    'aucpr_none': [None] * len(weight_range),              # PR-AUC：精確度-召回率曲線下面積
    'aucpr_undersampling': [None] * len(weight_range),
    'aucpr_oversampling': [None] * len(weight_range),
    
    'logloss_none': [None] * len(weight_range),            # 對數損失：概率預測品質指標
    'logloss_undersampling': [None] * len(weight_range),
    'logloss_oversampling': [None] * len(weight_range)
}

# =========================
# 主要實驗迴圈
# =========================
# 對每種採樣策略進行完整測試
for sampler_name in samplers:
    print(f"\n=== {sampler_name.upper()} 結果 ===")

    # 測試每個正類權重設定
    for idx, weight in enumerate(weight_range):
        print(f"\n=== 測試正類權重: {weight:.2f} ===")

        # 初始化每折指標儲存列表
        precision_list, recall_list, f1_list = [], [], []
        precision_neg_list, recall_neg_list, f1_neg_list = [], [], []
        tn_list, tp_list = [], []
        auc_list, aucpr_list = [], []
        logloss_list = []

        # =========================
        # 5 折交叉驗證
        # =========================
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, Y)):
            # 分割訓練集與測試集
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]

            # 根據採樣策略處理訓練資料
            if sampler_name == 'none':
                X_train_resampled, Y_train_resampled = X_train, Y_train
            else:
                X_train_resampled, Y_train_resampled = samplers[sampler_name].fit_resample(X_train, Y_train)

            # =========================
            # 模型訓練
            # =========================
            # 初始化 XGBoost 分類器（使用調優後的超參數）
            clf = XGBClassifier(
                alpha=1.3,                    # L1 正則化參數
                subsample=0.8,                # 子採樣比例
                min_child_weight=1.45,        # 葉節點最小權重和
                n_estimators=800,             # 樹的數量
                learning_rate=0.03,           # 學習率
                scale_pos_weight=weight,      # 正類權重（當前測試參數）
                random_state=42,              # 隨機種子
                eval_metric='logloss'         # 評估指標
            )
            
            # 訓練模型
            clf.fit(X_train_resampled, Y_train_resampled)

            # =========================
            # 模型預測與評估
            # =========================
            # 進行預測
            Y_pred = clf.predict(X_test)                    # 類別預測
            Y_pred_proba = clf.predict_proba(X_test)[:, 1]  # 正類概率預測

            # 計算正類（不良事件）相關指標
            precision = precision_score(Y_test, Y_pred, pos_label=1, zero_division=0)
            recall = recall_score(Y_test, Y_pred, pos_label=1, zero_division=0)
            f1 = f1_score(Y_test, Y_pred, pos_label=1, zero_division=0)

            # 計算負類（無不良事件）相關指標
            precision_neg = precision_score(Y_test, Y_pred, pos_label=0, zero_division=0)
            recall_neg = recall_score(Y_test, Y_pred, pos_label=0, zero_division=0)
            f1_neg = f1_score(Y_test, Y_pred, pos_label=0, zero_division=0)

            # 計算混淆矩陣
            cm = confusion_matrix(Y_test, Y_pred)
            tn, fp, fn, tp = cm.ravel()

            # 計算進階指標
            auc = roc_auc_score(Y_test, Y_pred_proba)           # ROC-AUC
            aucpr = average_precision_score(Y_test, Y_pred_proba)  # PR-AUC
            ll = log_loss(Y_test, Y_pred_proba)                 # 對數損失

            # 儲存本折結果
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            precision_neg_list.append(precision_neg)
            recall_neg_list.append(recall_neg)
            f1_neg_list.append(f1_neg)
            tn_list.append(tn)
            tp_list.append(tp)
            auc_list.append(auc)
            aucpr_list.append(aucpr)
            logloss_list.append(ll)

        # =========================
        # 計算交叉驗證平均結果
        # =========================
        # 計算 5 折平均值並儲存到結果字典
        results[f'tp_{sampler_name}'][idx] = np.mean(tp_list)
        results[f'precision_{sampler_name}'][idx] = np.mean(precision_list)
        results[f'recall_{sampler_name}'][idx] = np.mean(recall_list)
        results[f'f1_{sampler_name}'][idx] = np.mean(f1_list)
        results[f'tn_{sampler_name}'][idx] = np.mean(tn_list)
        results[f'precision_neg_{sampler_name}'][idx] = np.mean(precision_neg_list)
        results[f'recall_neg_{sampler_name}'][idx] = np.mean(recall_neg_list)
        results[f'f1_neg_{sampler_name}'][idx] = np.mean(f1_neg_list)
        results[f'auc_{sampler_name}'][idx] = np.mean(auc_list)
        results[f'aucpr_{sampler_name}'][idx] = np.mean(aucpr_list)
        results[f'logloss_{sampler_name}'][idx] = np.mean(logloss_list)

        # 輸出本輪測試結果
        print(f"平均正類 Precision: {np.mean(precision_list):.4f}")
        print(f"平均正類 Recall: {np.mean(recall_list):.4f}")
        print(f"平均正類 F1-score: {np.mean(f1_list):.4f}")
        print(f"平均負類 Precision: {np.mean(precision_neg_list):.4f}")
        print(f"平均負類 Recall: {np.mean(recall_neg_list):.4f}")
        print(f"平均負類 F1-score: {np.mean(f1_neg_list):.4f}")
        print(f"平均真陰性 (TN): {np.mean(tn_list):.2f}")
        print(f"平均真陽性 (TP): {np.mean(tp_list):.2f}")
        print(f"平均 AUC: {np.mean(auc_list):.4f}")
        print(f"平均 AUC-PR: {np.mean(aucpr_list):.4f}")
        print(f"平均 Logloss: {np.mean(logloss_list):.4f}")

# =========================
# 結果輸出與儲存
# =========================
# 轉換結果為 DataFrame
results_df = pd.DataFrame(results)

# 定義輸出欄位順序
columns = [
    'scale_pos_weight',
    'tp_none', 'tp_undersampling', 'tp_oversampling',
    'precision_none', 'precision_undersampling', 'precision_oversampling',
    'recall_none', 'recall_undersampling', 'recall_oversampling',
    'f1_none', 'f1_undersampling', 'f1_oversampling',
    'tn_none', 'tn_undersampling', 'tn_oversampling',
    'precision_neg_none', 'precision_neg_undersampling', 'precision_neg_oversampling',
    'recall_neg_none', 'recall_neg_undersampling', 'recall_neg_oversampling',
    'f1_neg_none', 'f1_neg_undersampling', 'f1_neg_oversampling',
    'auc_none', 'auc_undersampling', 'auc_oversampling',
    'aucpr_none', 'aucpr_undersampling', 'aucpr_oversampling',
    'logloss_none', 'logloss_undersampling', 'logloss_oversampling'  
]

# 重新排列 DataFrame 欄位
results_df = results_df[columns]

# 儲存結果到 CSV 檔案
results_df.to_csv('xgboost_cv_sampling_weight_analysis.csv', index=False, encoding='utf-8-sig')
print("\n交叉驗證結果（包含正類和負類指標及 AUC/AUC-PR/Logloss）已儲存至 xgboost_cv_sampling_weight_analysis.csv")

# =========================
# 清理與結束
# =========================
# 恢復標準輸出並關閉檔案
sys.stdout = sys.stdout.stdout
output_file.close()

print("\n程式執行完成！")
print("輸出檔案：")
print("- predict_output.txt: 詳細執行日誌")
print("- logloss.csv: 交叉驗證結果摘要")