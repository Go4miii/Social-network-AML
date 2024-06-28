# import pandas as pd
# import hdbscan
# from sklearn.preprocessing import StandardScaler
# import networkx as nx
# import matplotlib.pyplot as plt
#
# # 1. 读取数据
# df = pd.read_csv('txs_combined.csv')
#
# # 2. 按时间步分割数据并处理每个时间步
# time_steps = df['Time step'].unique()
# all_clusters = {}
#
# for time_step in time_steps:
#     # 提取特定时间步的子数据集
#     sub_df = df[df['Time step'] == time_step]
#
#     # 构建有向图
#     G = nx.from_pandas_edgelist(sub_df, 'txId1', 'txId2', create_using=nx.DiGraph())
#
#     # 提取节点特征
#     node_features = []
#     nodes = list(G.nodes)
#
#     for node in nodes:
#         degree = G.degree(node)
#         out_degree = G.out_degree(node)
#         in_degree = G.in_degree(node)
#         clustering_coefficient = nx.clustering(G.to_undirected(), node)
#         total_degree = out_degree + in_degree
#
#         # 构建节点特征向量
#         node_features.append([degree, out_degree, in_degree, total_degree, clustering_coefficient])
#
#     # 3. 特征标准化
#     scaler = StandardScaler()
#     node_features_scaled = scaler.fit_transform(node_features)
#
#     # 4. HDBSCAN聚类
#     clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
#     clusters = clusterer.fit_predict(node_features_scaled)
#
#     # 保存聚类结果
#     all_clusters[time_step] = clusters
#
#     # 输出每个时间步的聚类结果
#     print(f"Time Step {time_step}:")
#     for i, node in enumerate(nodes):
#         print(f"Node {node}: Cluster {clusters[i]}")
#
#     # 统计 cluster=-1 的个数
#     outlier_count = list(clusters).count(-1)
#     print(f"Outlier Count at Time Step {time_step}: {outlier_count}")
#
#     # 可视化网络图
#     pos = nx.spring_layout(G)
#     plt.figure(figsize=(12, 8))
#
#     # 绘制节点
#     node_colors = ['red' if c == -1 else 'blue' for c in clusters]
#     nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=50)
#
#     # 绘制边
#     nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5)
#
#     plt.title(f'Network Graph with HDBSCAN Clustering (Time Step {time_step})')
#     plt.axis('off')
#     plt.show()

import pandas as pd
import hdbscan
from sklearn.preprocessing import StandardScaler
import networkx as nx
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('txs_combined.csv')

# 按时间步分割数据并处理每个时间步
time_steps = df['Time step'].unique()
recall_rates = []

for time_step in time_steps:
    # 提取特定时间步的子数据集
    sub_df = df[df['Time step'] == time_step]

    # 构建有向图
    G = nx.from_pandas_edgelist(sub_df, 'txId1', 'txId2', create_using=nx.DiGraph())

    # 提取节点特征
    node_features = []
    nodes = list(G.nodes)

    for node in nodes:
        degree = G.degree(node)
        out_degree = G.out_degree(node)
        in_degree = G.in_degree(node)
        clustering_coefficient = nx.clustering(G.to_undirected(), node)
        total_degree = out_degree + in_degree

        # 构建节点特征向量
        node_features.append([degree, out_degree, in_degree, total_degree, clustering_coefficient])

    # 特征标准化
    scaler = StandardScaler()
    node_features_scaled = scaler.fit_transform(node_features)

    # HDBSCAN聚类
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
    clusters = clusterer.fit_predict(node_features_scaled)

    # 提取离群点
    outliers = [nodes[i] for i in range(len(nodes)) if clusters[i] == -1]

    # 提取原始数据中的非法点
    illicit_nodes_txId1 = sub_df[sub_df['class_txId1'] == 1]['txId1']
    illicit_nodes_txId2 = sub_df[sub_df['class_txId2'] == 1]['txId2']
    illicit_nodes = set(illicit_nodes_txId1).union(set(illicit_nodes_txId2))

    # 计算召回率
    true_positives = len(set(outliers).intersection(illicit_nodes))
    actual_illicit = len(illicit_nodes)
    recall = true_positives / actual_illicit if actual_illicit > 0 else 0
    recall_rates.append(recall)

    # 输出每个时间步的召回率
    print(f"Time Step {time_step}: Recall = {recall:.4f}")

# 绘制召回率的折线图
plt.figure(figsize=(10, 6))
plt.plot(time_steps, recall_rates, marker='o', linestyle='-', color='b')
plt.xlabel('Time Step')
plt.ylabel('Recall Rate')
plt.title('Recall Rate Over Time Steps')
plt.grid(True)
plt.show()
