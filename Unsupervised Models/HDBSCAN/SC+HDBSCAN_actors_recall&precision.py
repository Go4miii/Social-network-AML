# import pandas as pd
# import networkx as nx
# from sklearn.cluster import SpectralClustering
# from sklearn.metrics import recall_score, precision_score
# import hdbscan
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
#
# # 读取数据集
# data = pd.read_csv('addr_combined.csv')
#
# # 获取所有唯一的时间步长
# time_steps = data['Time step'].unique()
#
# # 存储每个时间步长的召回率和精确率
# results = []
#
# for time_step in time_steps:
#     print(f"Processing Time step: {time_step}")
#
#     # 筛选当前时间步长的数据
#     step_data = data[data['Time step'] == time_step]
#
#     # Spectral Clustering
#     G = nx.Graph()
#     for index, row in step_data.iterrows():
#         G.add_node(row['input address'], class_=row['class_input_address'])
#         G.add_node(row['output address'], class_=row['class_output_address'])
#         G.add_edge(row['input address'], row['output address'])
#
#     adjacency_matrix = nx.adjacency_matrix(G)
#     spectral_clustering = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=42)
#
#     if len(G) > 0:
#         labels = spectral_clustering.fit_predict(adjacency_matrix.toarray())
#         label_dict = {node: label for node, label in zip(G.nodes(), labels)}
#         true_labels = {}
#
#         for _, row in step_data.iterrows():
#             true_labels[row['input address']] = row['class_input_address']
#             true_labels[row['output address']] = row['class_output_address']
#
#         true_values = [1 if true_labels[node] == 1 else 0 for node in G.nodes()]  # 假设2代表正类
#         predicted_values = [1 if label_dict[node] == 1 else 0 for node in G.nodes()]  # 假设1代表正类
#         recall = recall_score(true_values, predicted_values)
#         precision = precision_score(true_values, predicted_values)
#         results.append({
#             'Time Step': time_step,
#             'Method': 'Spectral Clustering',
#             'Recall': recall,
#             'Precision': precision
#         })
#
#     # HDBSCAN
#     G = nx.from_pandas_edgelist(step_data, 'input address', 'output address', create_using=nx.DiGraph())
#     node_features = []
#     nodes = list(G.nodes)
#
#     for node in nodes:
#         degree = G.degree(node)
#         out_degree = G.out_degree(node)
#         in_degree = G.in_degree(node)
#         clustering_coefficient = nx.clustering(G.to_undirected(), node)
#         total_degree = out_degree + in_degree
#         node_features.append([degree, out_degree, in_degree, total_degree, clustering_coefficient])
#
#     scaler = StandardScaler()
#     node_features_scaled = scaler.fit_transform(node_features)
#     clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
#     clusters = clusterer.fit_predict(node_features_scaled)
#     outliers = [nodes[i] for i in range(len(nodes)) if clusters[i] == -1]
#     illicit_nodes_input = step_data[step_data['class_input_address'] == 1]['input address']
#     illicit_nodes_output = step_data[step_data['class_output_address'] == 1]['output address']
#     illicit_nodes = set(illicit_nodes_input).union(set(illicit_nodes_output))
#     true_positives = len(set(outliers).intersection(illicit_nodes))
#     actual_illicit = len(illicit_nodes)
#     recall = true_positives / actual_illicit if actual_illicit > 0 else 0
#     precision = true_positives / len(outliers) if len(outliers) > 0 else 0
#     results.append({
#         'Time Step': time_step,
#         'Method': 'HDBSCAN',
#         'Recall': recall,
#         'Precision': precision
#     })
#
# # 将结果保存到CSV文件
# results_df = pd.DataFrame(results)
# results_df.to_csv('addr_combined_spectral_hdbscan_recall_precision.csv', index=False)
#
# # 绘制折线图
# plt.figure(figsize=(10, 6))
#
# # 绘制 Spectral Clustering 的 Recall 和 Precision
# spectral_recall = results_df[results_df['Method'] == 'Spectral Clustering']['Recall']
# spectral_precision = results_df[results_df['Method'] == 'Spectral Clustering']['Precision']
# plt.plot(time_steps, spectral_recall, marker='o', linestyle='-', color='b', label='Spectral Clustering Recall')
# plt.plot(time_steps, spectral_precision, marker='s', linestyle='--', color='b', label='Spectral Clustering Precision')
#
# # 绘制 HDBSCAN 的 Recall 和 Precision
# hdbscan_recall = results_df[results_df['Method'] == 'HDBSCAN']['Recall']
# hdbscan_precision = results_df[results_df['Method'] == 'HDBSCAN']['Precision']
# plt.plot(time_steps, hdbscan_recall, marker='x', linestyle='-', color='g', label='HDBSCAN Recall')
# plt.plot(time_steps, hdbscan_precision, marker='^', linestyle='--', color='g', label='HDBSCAN Precision')
#
# plt.title('Recall and Precision over Time Steps')
# plt.xlabel('Time Step')
# plt.ylabel('Score')
# plt.legend()
# plt.xticks(time_steps)
# plt.grid(True)
# plt.tight_layout()
# plt.show()

import pandas as pd
import networkx as nx
from sklearn.cluster import SpectralClustering
from sklearn.metrics import recall_score, precision_score
import hdbscan
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import numpy as np

# 读取数据集
data = pd.read_csv('addr_combined.csv')

# 获取所有唯一的时间步长
time_steps = data['Time step'].unique()

# 存储每个时间步长的召回率和精确率
results = []

for time_step in time_steps:
    print(f"Processing Time step: {time_step}")

    # 筛选当前时间步长的数据
    step_data = data[data['Time step'] == time_step]

    # Spectral Clustering
    G = nx.Graph()
    for index, row in step_data.iterrows():
        G.add_node(row['input address'], class_=row['class_input_address'])
        G.add_node(row['output address'], class_=row['class_output_address'])
        G.add_edge(row['input address'], row['output address'])

    # 使用稀疏矩阵来表示邻接矩阵
    adjacency_matrix = nx.adjacency_matrix(G)
    spectral_clustering = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=42)

    if len(G) > 0:
        labels = spectral_clustering.fit_predict(adjacency_matrix)
        label_dict = {node: label for node, label in zip(G.nodes(), labels)}
        true_labels = {}

        for _, row in step_data.iterrows():
            true_labels[row['input address']] = row['class_input_address']
            true_labels[row['output address']] = row['class_output_address']

        true_values = [1 if true_labels[node] == 1 else 0 for node in G.nodes()]
        predicted_values = [1 if label_dict[node] == 1 else 0 for node in G.nodes()]
        recall = recall_score(true_values, predicted_values)
        precision = precision_score(true_values, predicted_values)
        results.append({
            'Time Step': time_step,
            'Method': 'Spectral Clustering',
            'Recall': recall,
            'Precision': precision
        })

    # HDBSCAN
    G = nx.from_pandas_edgelist(step_data, 'input address', 'output address', create_using=nx.DiGraph())
    node_features = []
    nodes = list(G.nodes)

    for node in nodes:
        degree = G.degree(node)
        out_degree = G.out_degree(node)
        in_degree = G.in_degree(node)
        clustering_coefficient = nx.clustering(G.to_undirected(), node)
        total_degree = out_degree + in_degree
        node_features.append([degree, out_degree, in_degree, total_degree, clustering_coefficient])

    scaler = StandardScaler()
    node_features_scaled = scaler.fit_transform(node_features)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
    clusters = clusterer.fit_predict(node_features_scaled)
    outliers = [nodes[i] for i in range(len(nodes)) if clusters[i] == -1]
    illicit_nodes_input = step_data[step_data['class_input_address'] == 1]['input address']
    illicit_nodes_output = step_data[step_data['class_output_address'] == 1]['output address']
    illicit_nodes = set(illicit_nodes_input).union(set(illicit_nodes_output))
    true_positives = len(set(outliers).intersection(illicit_nodes))
    actual_illicit = len(illicit_nodes)
    recall = true_positives / actual_illicit if actual_illicit > 0 else 0
    precision = true_positives / len(outliers) if len(outliers) > 0 else 0
    results.append({
        'Time Step': time_step,
        'Method': 'HDBSCAN',
        'Recall': recall,
        'Precision': precision
    })

# 将结果保存到CSV文件
results_df = pd.DataFrame(results)
results_df.to_csv('addr_combined_spectral_hdbscan_recall_precision.csv', index=False)

# 绘制折线图
plt.figure(figsize=(10, 6))

# 绘制 Spectral Clustering 的 Recall 和 Precision
spectral_recall = results_df[results_df['Method'] == 'Spectral Clustering']['Recall']
spectral_precision = results_df[results_df['Method'] == 'Spectral Clustering']['Precision']
plt.plot(time_steps, spectral_recall, marker='o', linestyle='-', color='b', label='Spectral Clustering Recall')
plt.plot(time_steps, spectral_precision, marker='s', linestyle='--', color='b', label='Spectral Clustering Precision')

# 绘制 HDBSCAN 的 Recall 和 Precision
hdbscan_recall = results_df[results_df['Method'] == 'HDBSCAN']['Recall']
hdbscan_precision = results_df[results_df['Method'] == 'HDBSCAN']['Precision']
plt.plot(time_steps, hdbscan_recall, marker='x', linestyle='-', color='g', label='HDBSCAN Recall')
plt.plot(time_steps, hdbscan_precision, marker='^', linestyle='--', color='g', label='HDBSCAN Precision')

plt.title('Recall and Precision over Time Steps')
plt.xlabel('Time Step')
plt.ylabel('Score')
plt.legend()
plt.xticks(time_steps)
plt.grid(True)
plt.tight_layout()
plt.show()
