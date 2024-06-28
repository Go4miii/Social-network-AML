import pandas as pd
import networkx as nx
from sklearn.cluster import SpectralClustering
import plotly.graph_objects as go

# # 1. 加载数据集
# data = pd.read_csv('txs_combined.csv')
#
# # 获取所有唯一的时间步长
# time_steps = data['Time step'].unique()
#
# def draw_network(G, labels=None, title=""):
#     pos = nx.spring_layout(G)
#     edge_x = []
#     edge_y = []
#     for edge in G.edges():
#         x0, y0 = pos[edge[0]]
#         x1, y1 = pos[edge[1]]
#         edge_x.append(x0)
#         edge_x.append(x1)
#         edge_x.append(None)
#         edge_y.append(y0)
#         edge_y.append(y1)
#         edge_y.append(None)
#
#     edge_trace = go.Scatter(
#         x=edge_x, y=edge_y,
#         line=dict(width=0.5, color='blue'),
#         hoverinfo='none',
#         mode='lines')
#
#     node_x = []
#     node_y = []
#     node_color = []
#     node_text = []
#     for node in G.nodes():
#         x, y = pos[node]
#         node_x.append(x)
#         node_y.append(y)
#         node_text.append(node)
#         if labels is not None:
#             node_color.append('blue' if labels[node] == 0 else 'red')
#         else:
#             node_color.append('red' if G.nodes[node]['class_'] == 1 else 'blue')
#
#     node_trace = go.Scatter(
#         x=node_x, y=node_y,
#         mode='markers',
#         hoverinfo='text',
#         marker=dict(
#             showscale=False,
#             color=node_color,
#             size=10,
#             line_width=2),
#         text=node_text
#     )
#
#     # 添加图例
#     legend_items = [
#         go.Scatter(
#             x=[None], y=[None],
#             mode='markers',
#             marker=dict(size=10, color='red'),
#             legendgroup='Illicit',
#             showlegend=True,
#             name='Illicit'
#         ),
#         go.Scatter(
#             x=[None], y=[None],
#             mode='markers',
#             marker=dict(size=10, color='blue'),
#             legendgroup='Licit',
#             showlegend=True,
#             name='Licit'
#         )
#     ]
#
#     fig = go.Figure(data=[edge_trace, node_trace] + legend_items,
#                     layout=go.Layout(
#                         title=title,
#                         titlefont_size=16,
#                         showlegend=True,
#                         hovermode='closest',
#                         margin=dict(b=20, l=5, r=5, t=40),
#                         annotations=[dict(
#                             showarrow=True,
#                             xref="paper", yref="paper",
#                             x=0.005, y=-0.002)],
#                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
#                     )
#     return fig
#
# # 示例时间步长
# time_step = 41
#
# # 筛选当前时间步长的数据
# step_data = data[data['Time step'] == time_step]
#
# # 构建图
# G = nx.DiGraph()
# for index, row in step_data.iterrows():
#     G.add_node(row['txId1'], class_=row['class_txId1'])
#     G.add_node(row['txId2'], class_=row['class_txId2'])
#     G.add_edge(row['txId1'], row['txId2'])
#
# # 绘制聚类之前的网络图
# fig_before = draw_network(G, title=f"Network Before Clustering (Time step {time_step})")
# fig_before.show()
#
# # 计算相似性矩阵（使用邻接矩阵）
# adjacency_matrix = nx.adjacency_matrix(G)
#
# # 谱聚类（使用k=2，可以调整）
# spectral_clustering = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=42)
# if len(G) > 0:  # 确保图中有节点
#     labels = spectral_clustering.fit_predict(adjacency_matrix.toarray())
#     # 将聚类标签添加到图节点中
#     label_dict = {node: label for node, label in zip(G.nodes(), labels)}
#
#     # 绘制聚类之后的网络图
#     fig_after = draw_network(G, labels=label_dict, title=f"Network After Clustering (Time step {time_step})")
#     fig_after.show()

import pandas as pd
import networkx as nx
from sklearn.cluster import SpectralClustering
import plotly.graph_objects as go

# 1. 加载数据集
data = pd.read_csv('txs_combined.csv')

# 获取所有唯一的时间步长
time_steps = data['Time step'].unique()

def draw_network(G, labels=None, title=""):
    pos = nx.spring_layout(G, seed=42)  # 使用固定的随机种子以确保布局一致性

    # 构建边的绘制数据
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # 绘制边的轨迹
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    # 构建节点的绘制数据
    node_x = []
    node_y = []
    node_color = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        if labels is not None:
            # 如果节点被正确聚类为illicit，标记为红色，否则为蓝色
            node_color.append('red' if (G.nodes[node]['class_'] == 1 and labels[node] == 1) else 'blue')
        else:
            node_color.append('red' if G.nodes[node]['class_'] == 1 else 'blue')

    # 绘制节点的轨迹
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='none',
        marker=dict(
            showscale=False,
            color=node_color,
            size=10,
            line_width=2)
    )

    # 绘制图
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=dict(
                            text=title,
                            font=dict(size=24, color='black', family='Arial', weight='bold'),
                            x=0.5,
                            xanchor='center'
                        ),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='rgba(0,0,0,0)'  # Set plot background color to transparent
                    ))
    return fig

# 示例时间步长
time_step = 41

# 筛选当前时间步长的数据
step_data = data[data['Time step'] == time_step]

# 构建图
G = nx.DiGraph()
for index, row in step_data.iterrows():
    G.add_node(row['txId1'], class_=row['class_txId1'])
    G.add_node(row['txId2'], class_=row['class_txId2'])
    G.add_edge(row['txId1'], row['txId2'])

# 计算相似性矩阵（使用邻接矩阵）
adjacency_matrix = nx.adjacency_matrix(G)

# 谱聚类（使用k=2，可以调整）
spectral_clustering = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=42)
if len(G) > 0:  # 确保图中有节点
    labels = spectral_clustering.fit_predict(adjacency_matrix.toarray())
    # 将聚类标签添加到图节点中
    label_dict = {node: label for node, label in zip(G.nodes(), labels)}

    # 绘制聚类后的网络图，并标注正确聚类的illicit点和标注错的illicit点
    fig_after = draw_network(G, labels=label_dict, title=f"Results of Spectral Clustering (Transactions,Time step ={time_step})")

    # 标注谱聚类后聚类正确的illicit点
    correct_illicit_nodes = []
    incorrect_illicit_nodes = []
    for node in G.nodes():
        # 检查该节点的真实类别和聚类标签是否匹配
        if G.nodes[node]['class_'] == 1 and label_dict[node] == 1:
            correct_illicit_nodes.append(node)
        elif G.nodes[node]['class_'] == 1 and label_dict[node] == 0:
            incorrect_illicit_nodes.append(node)

    # 添加标注
    for node in correct_illicit_nodes:
        x, y = nx.spring_layout(G, seed=42)[node]  # 使用相同的布局获取坐标
        fig_after.add_annotation(
            x=x, y=y,
            text="Correct Illicit",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-30,
            bgcolor="white",
            bordercolor="red",
            borderwidth=2,
            font=dict(color="red", size=14, weight='bold')
        )

    for node in incorrect_illicit_nodes:
        x, y = nx.spring_layout(G, seed=42)[node]  # 使用相同的布局获取坐标
        fig_after.add_annotation(
            x=x, y=y,
            text="Incorrect Illicit",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-30,
            bgcolor="white",
            bordercolor="blue",
            borderwidth=2,
            font=dict(color="blue", size=14, weight='bold')
        )

    fig_after.show()

