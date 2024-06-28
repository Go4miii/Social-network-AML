import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler



txs_classes = pd.read_csv("elliptic_txs_classes.csv")
txs_features = pd.read_csv("elliptic_txs_features2.csv")
data = pd.merge(txs_classes, txs_features, on='txId', how='inner')
data['class'].replace('unknown', np.nan, inplace=True)
data.dropna(subset=['class'], inplace=True)
data['class'] = data['class'].astype(int)
data['class'] = data['class'].map({1: 0, 2: 1})
print("Before dropping:", data.shape)

models = {
    'Logistic Regression': LogisticRegression(class_weight={0: 2, 1: 1}),
    'Decision Tree': DecisionTreeClassifier(class_weight={0: 2, 1: 1}),
    'MLP': MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, activation='relu', solver='adam', random_state=42),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'Stacking': StackingClassifier(estimators=[
        ('lr', LogisticRegression()),
        ('dt', DecisionTreeClassifier()),
        ('rf', RandomForestClassifier()),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ], final_estimator=LogisticRegression())
}

precisions = {name: [] for name in models}
recalls = {name: [] for name in models}

# 模型训练和评估
for i in range(35, 49):
    train_data = data[data['feature0'] <= i]
    test_data = data[data['feature0'] == i + 1]

    X_train = train_data.drop(['txId', 'class'], axis=1)
    y_train = train_data['class']
    X_test = test_data.drop(['txId', 'class'], axis=1)
    y_test = test_data['class']

    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        precision = precision_score(y_test, y_pred, pos_label=0)
        recall = recall_score(y_test, y_pred, pos_label=0)

        precisions[name].append(precision)
        recalls[name].append(recall)

print(precisions)
print(recalls)
# 创建并保存CSV
time_steps = range(36, 50)
results_df = pd.DataFrame(index=time_steps)
for name in models:
    results_df[f'{name} Precision'] = precisions[name]
    results_df[f'{name} Recall'] = recalls[name]

results_df.to_csv('model_precision_recall_by_time_step.csv')


# 绘制折线图
fig, ax = plt.subplots(figsize=(12, 7))
for name in models:
    ax.plot(range(35, 49), precisions[name], marker='o', linestyle='-', label=f'{name} Precision')
    ax.plot(range(35, 49), recalls[name], marker='x', linestyle='--', label=f'{name} Recall')

ax.set_xlabel('Time Step')
ax.set_ylabel('Scores')
ax.set_title('Precision and Recall by Time Step Threshold for Different Models')
ax.set_xticks(range(35, 49))
ax.legend()

# 保存图表
plt.savefig('model_performance_chart1.png', dpi=300)  # 保存为PNG格式，分辨率300dpi
plt.show()