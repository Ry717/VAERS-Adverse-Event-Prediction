import pandas as pd
import numpy as np
import ast
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from xgboost import XGBRegressor

# 讀取資料
merged_df = pd.read_csv("merged_data.csv")

# 將 article_vector 欄位轉換為 list 並加入年齡與性別資訊
def process_vector(row):
    try:
        raw_vector = ast.literal_eval(row['article_vector'])  # 這應該是 40x8 的 list
        flat_vector = np.array(raw_vector).flatten().tolist()  # 攤平成一維
    except:
        return None

    age = float(row['AGE_YRS']) if pd.notna(row['AGE_YRS']) else 0.0
    # 在 append 前，把年齡做標準化（例如用 min-max 到 0~1）
    normalized_age = age / 100  # 假設最大年齡 100 歲


    sex = str(row['SEX']).strip().upper()
    sex_vec = {
        'F': [1, 0, 0],
        'M': [0, 1, 0],
        'U': [0, 0, 1]
    }.get(sex, [0, 0, 0])

    return flat_vector + [normalized_age] + sex_vec


# 應用轉換
merged_df['article_vector'] = merged_df.apply(process_vector, axis=1)
merged_df = merged_df[merged_df['article_vector'].notnull()]

# 移除目標欄位缺失值
target_cols = ['DIED', 'L_THREAT', 'ER_VISIT', 'HOSPITAL', 'X_STAY', 'ER_ED_VISIT']
merged_df = merged_df.dropna(subset=target_cols)

# 準備特徵與標籤
X = np.array(merged_df['article_vector'].tolist())
Y = (merged_df[target_cols] == 1).any(axis=1).astype(int)

# 分割訓練與測試集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 計算權重：負樣本數 / 正樣本數
neg, pos = np.bincount(Y_train)
scale = neg / pos
best_score = 0
best_weight = None

for weight in np.arange(4.8, 4.87, 0.1):
# 建立並訓練 XGBoost 模型
    eval_set = [(X_train, Y_train), (X_test, Y_test)]
    model = XGBClassifier(alpha=3 ,subsample=0.8, min_child_weight=1.4, n_estimators=300, learning_rate=0.05 , scale_pos_weight=4.4, eval_metric='auc',random_state = 42)
    model.fit(X_train, Y_train, eval_set=eval_set, verbose=False)
    

    print( (round(weight, 2)) ,"倍")
# 輸出準確率
    print("訓練集:", model.score(X_train, Y_train))
    print("測試集:", model.score(X_test, Y_test))

    # 預測並產生分類報告
    y_pred = model.predict(X_test)

    print("測試集分類報告：")
    print(classification_report(Y_test, y_pred, target_names=["Class 0", "Class 1"]))
# 顯示每一輪訓練與測試集上的 AUC 評估結果
    eval_result = model.evals_result()

    print("最後一輪 AUC 評估結果：")
    print(f"Train AUC: {eval_result['validation_0']['auc'][-1]:.4f}")
    print(f"Test  AUC: {eval_result['validation_1']['auc'][-1]:.4f}")
    # 放在 fit 後、eval_result 拿到之後
    plt.plot(eval_result['validation_0']['auc'], label='Train AUC')
    plt.plot(eval_result['validation_1']['auc'], label='Test AUC')
    plt.title(f'AUC over Boosting Rounds (weight={weight:.2f})')
    plt.xlabel('Boosting Round')
    plt.ylabel('AUC')
    plt.legend()
    plt.show()
    
# 定義你要評估的指標
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score),
        'auc': make_scorer(roc_auc_score)
    }

# 執行交叉驗證，得到多個指標
    cv_results = cross_validate(model, X, Y, cv=5, scoring=scoring)

# 輸出平均指標
    print("交叉驗證平均指標：")
    print(f"Accuracy : {np.mean(cv_results['test_accuracy']):.4f}")
    print(f"Precision: {np.mean(cv_results['test_precision']):.4f}")
    print(f"Recall   : {np.mean(cv_results['test_recall']):.4f}")
    print(f"F1-score : {np.mean(cv_results['test_f1']):.4f}")
    print(f"AUC      : {np.mean(cv_results['test_auc']):.4f}")

    f1 = np.mean(cv_results['test_f1'])
    

    print(f"scale_pos_weight={weight:.1f}，F1-score={f1:.4f}\n")
    
    if f1 > best_score:
        best_score = f1
        best_weight = round(weight, 2)

print(f"最佳權重為：{best_weight}，對應 F1-score={best_score:.4f}")