import pandas as pd
import networkx as nx
from sklearn.cluster import SpectralClustering
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt

# 加载数据集
data = pd.read_csv('addr_combined.csv')

# 获取所有唯一的时间步长
time_steps = data['Time step'].unique()

# 存储每个时间步长的召回率
recall_scores = []

for time_step in time_steps:
    print(f"Processing Time step: {time_step}")

    # 筛选当前时间步长的数据
    step_data = data[data['Time step'] == time_step]

    # 构建图
    G = nx.Graph()
    for index, row in step_data.iterrows():
        G.add_node(row['input address'], class_=row['class_input_address'])
        G.add_node(row['output address'], class_=row['class_output_address'])
        G.add_edge(row['input address'], row['output address'])

    # 计算相似性矩阵（使用邻接矩阵）
    adjacency_matrix = nx.adjacency_matrix(G)

    # 谱聚类（使用k=2，可以调整）
    spectral_clustering = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=42)
    if len(G) > 0:  # 确保图中有节点
        labels = spectral_clustering.fit_predict(adjacency_matrix.toarray())
        # 将聚类标签添加到图节点中
        label_dict = {node: label for node, label in zip(G.nodes(), labels)}

        # 计算真实标签
        true_labels = {}
        for _, row in step_data.iterrows():
            true_labels[row['input address']] = row['class_input_address']
            true_labels[row['output address']] = row['class_output_address']

        # 将真实标签转换为二元类别（0 和 1）
        true_values = [1 if true_labels[node] == 1 else 0 for node in G.nodes()]
        # 将预测标签转换为二元类别（0 和 1）
        predicted_values = [1 if label_dict[node] == 1 else 0 for node in G.nodes()]

        # 计算召回率
        recall = recall_score(true_values, predicted_values)
        recall_scores.append(recall)

# 绘制召回率随时间步长变化的折线图
plt.figure(figsize=(10, 6))
plt.plot(time_steps, recall_scores, marker='o', linestyle='-', color='b')
plt.title('Recall Score over Time Steps')
plt.xlabel('Time Step')
plt.ylabel('Recall')
plt.xticks(time_steps)
plt.grid(True)
plt.tight_layout()
plt.show()
