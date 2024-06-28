import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE


txs_classes = pd.read_csv("elliptic_txs_classes.csv")
txs_features = pd.read_csv("elliptic_txs_features2.csv")
data = pd.merge(txs_classes, txs_features, on='txId', how='inner')
data['class'].replace('unknown', np.nan, inplace=True)
data['class'] = data['class'].map({'1': 0, '2': 1})
# 分离已知和未知数据
known_data = data[data['class'].notnull()].copy()
unknown_data = data[data['class'].isnull()].copy()

# 把类别列转换成整数类型
known_data['class'] = known_data['class'].astype(int)

# 初始化缺失值填充器
imputer = SimpleImputer(strategy='mean')
# 例如，假设给标签为1的类别比标签为0的类别多2倍的权重
weights = {1: 2, 2: 1}

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
    train_data = known_data[known_data['feature0'] <= i]
    test_data = known_data[known_data['feature0'] == i + 1]
    # print("Before dropping:", train_data.shape)

    X_train = train_data.drop(['txId', 'class'], axis=1)
    y_train = train_data['class']
    X_test = test_data.drop(['txId', 'class'], axis=1)
    y_test = test_data['class']

    X_train, y_train = smote.fit_resample(X_train, y_train)
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    X_train = pd.DataFrame(X_train, columns=[col for col in train_data.drop(['txId', 'class'], axis=1).columns])
    X_test = pd.DataFrame(X_test, columns=[col for col in test_data.drop(['txId', 'class'], axis=1).columns])

    for name, model in models.items():
        # 结合之前的逻辑，定义一个列表来跟踪加入训练集的unknown数据索引
        added_unknown_indexes = []
        for round in range(3):  # 假设迭代3次

            if round > 0:
                # 通过索引移除上一次迭代中加入的unknown数据
                X_train = X_train.drop(added_unknown_indexes)
                y_train = y_train.drop(added_unknown_indexes)
                # 清空列表，为这一轮迭代记录新添加的unknown数据做准备
                added_unknown_indexes.clear()

            # 训练模型
            model.fit(X_train, y_train)
            # 评估模型
            y_pred = model.predict(X_test)

            precision = precision_score(y_test, y_pred, pos_label=0)
            recall = recall_score(y_test, y_pred, pos_label=0)
            if round ==2:

                precisions[name].append(precision)
                recalls[name].append(recall)


            unknown_data = data[(data['class'].isnull()) & (data['feature0'] <= i)]
            unknown_data_X = imputer.transform(unknown_data.drop(['txId', 'class'], axis=1))
            unknown_data['class'] = model.predict(unknown_data_X)

            # 更新训练数据集前，保留当前unknown数据的索引
            added_unknown_indexes = unknown_data.index.tolist()
            print(len(added_unknown_indexes))

            # 更新训练集
            X_train = pd.concat([X_train, pd.DataFrame(unknown_data_X,
                                                       index=added_unknown_indexes,
                                                       columns=[col for col in
                                                                known_data.drop(['txId', 'class'],
                                                                                axis=1).columns])]).reset_index(
                drop=True)

            y_train = pd.concat([y_train, pd.Series(unknown_data['class'], index=added_unknown_indexes)]).reset_index(
                drop=True)

            # 迭代结束，恢复未知数据的class为 np.nan，从训练集中移除这部分数据
            unknown_data['class'] = np.nan

            # 选择性地输出一些评估结果或调试信息
            print(f"Round {round + 1} complete.")


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
plt.savefig('model_performance_chart3.png', dpi=300)  # 保存为PNG格式，分辨率300dpi
plt.show()