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
from imblearn.over_sampling import SMOTE


data=pd.read_csv('wallets_features_classes_combined.csv').drop_duplicates(subset='address', keep='first')
data['class'].replace(3, 'unknown', inplace=True)  # 将3替换为 'unknown'
data['class'].replace('unknown', np.nan, inplace=True)  # 将'unknown'替换为 np.nan
data.dropna(subset=['class'], inplace=True)
data['class'] = data['class'].astype(int)
data['class'] = data['class'].map({1: 0, 2: 1})
print("Before dropping:", data.shape)
# 初始化SMOTE实例
smote = SMOTE(random_state=42)

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
    train_data = data[data['Timestep'] <= i]
    test_data = data[data['Timestep'] == i + 1]

    X_train = train_data.drop(['address', 'class'], axis=1)
    y_train = train_data['class']
    X_test = test_data.drop(['address', 'class'], axis=1)
    y_test = test_data['class']

    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    # 使用SMOTE进行上采样
    X_train, y_train = smote.fit_resample(X_train, y_train)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        precision = precision_score(y_test, y_pred, pos_label=0)
        recall = recall_score(y_test, y_pred, pos_label=0)

        precisions[name].append(precision)
        recalls[name].append(recall)
time_steps = range(36, 50)  # 包含49
results_df = pd.DataFrame(index=time_steps)
for name in models:
    results_df[f'{name} Precision'] = precisions[name]
    results_df[f'{name} Recall'] = recalls[name]

results_df.to_csv('model_precision_recall_by_time_step_actor2.csv')



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
plt.savefig('model_performance_chart2.png', dpi=300)  # 保存为PNG格式，分辨率300dpi
plt.show()