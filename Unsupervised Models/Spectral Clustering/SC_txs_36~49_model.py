import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
import networkx as nx
import matplotlib.pyplot as plt

# 1. 加载数据集
data = pd.read_csv('txs_combined.csv')

# 获取所有唯一的时间步长
time_steps = data['Time step'].unique()

# 设置绘图大小和布局
plt.figure(figsize=(20, len(time_steps) * 5))

# 循环遍历每个时间步长
for i, time_step in enumerate(time_steps, start=1):
    # 筛选当前时间步长的数据
    step_data = data[data['Time step'] == time_step]

    # 构建图
    G = nx.Graph()
    for index, row in step_data.iterrows():
        G.add_node(row['txId1'])
        G.add_node(row['txId2'])
        G.add_edge(row['txId1'], row['txId2'])

    # 计算相似性矩阵（使用邻接矩阵）
    adjacency_matrix = nx.adjacency_matrix(G)

    # 谱聚类（使用k=2，可以调整）
    spectral_clustering = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=42)
    if len(G) > 0:  # 确保图中有节点
        labels = spectral_clustering.fit_predict(adjacency_matrix.toarray())
        # 将聚类标签添加到图节点中
        for node, label in zip(G.nodes(), labels):
            G.nodes[node]['label'] = label

    # 绘制图
    plt.subplot(len(time_steps), 1, i)
    pos = nx.spring_layout(G)
    node_colors = [G.nodes[node].get('label', 0) for node in G.nodes()]
    nx.draw_networkx(G, pos, node_color=node_colors, with_labels=False, node_size=50, cmap=plt.cm.rainbow)
    plt.title(f"Spectral Clustering of Bitcoin Transactions (Time step {time_step})")
    plt.show()
# 显示绘图
# plt.tight_layout()
# plt.show()
