import pandas as pd

# 读取数
# 读取交易节点的边列表
txs_edgelist = pd.read_csv('D:/Users/dc1213/PycharmProjects/pythonProject1/txs_edgelist.csv')
# 读取交易的类型
txs_class = pd.read_csv('D:/Users/dc1213/PycharmProjects/pythonProject1/txs_classes.csv')
# 读取交易的特征
txs_features = pd.read_csv('D:/Users/dc1213/PycharmProjects/pythonProject1/txs_features.csv')
txs_features = txs_features.merge(txs_class,left_on='txId',right_on='txId',how='left')
# remove 'unknown' transactions
known_data = txs_features.loc[(txs_features['class']!=3), 'txId']
known_txs_features = txs_features.loc[txs_features['txId'].isin(known_data)]
known_txs_edgelist = txs_edgelist.loc[txs_edgelist['txId1'].isin(known_data) &
                                      txs_edgelist['txId2'].isin(known_data) ]

# 筛选出时间步长在36到49之间的交易
filtered_txs_features = known_txs_features[(known_txs_features['Time step'] >= 36) & (known_txs_features['Time step'] <= 49)]
filtered_txs_features.to_csv('filtered_txs_features.csv')
# 获取筛选后的交易ID
filtered_tx_ids = filtered_txs_features['txId'].tolist()

# 筛选交易流中涉及到这些交易ID的记录
filtered_txs_edgelist = known_txs_edgelist[(known_txs_edgelist['txId1'].isin(filtered_tx_ids)) | (known_txs_edgelist['txId2'].isin(filtered_tx_ids))]

# 筛选交易class中涉及到这些交易ID的记录
filtered_txs_classes = txs_class[txs_class['txId'].isin(filtered_tx_ids)]

# 合并交易流和时间步长
txs_combined = pd.merge(filtered_txs_edgelist, filtered_txs_features[['txId', 'Time step']], left_on='txId1', right_on='txId', how='left')
txs_combined = pd.merge(txs_combined, filtered_txs_features[['txId', 'Time step']], left_on='txId2', right_on='txId', how='left', suffixes=('_txId1', '_txId2'))

# 合并交易流和class
txs_combined = pd.merge(txs_combined, filtered_txs_classes, left_on='txId1', right_on='txId', how='left', suffixes=('', '_class_txId1'))
txs_combined = pd.merge(txs_combined, filtered_txs_classes, left_on='txId2', right_on='txId', how='left', suffixes=('', '_class_txId2'))

# 选择所需的列
txs_combined = txs_combined[['txId1', 'txId2', 'Time step_txId1', 'class', 'class_class_txId2']]

# 重命名列
txs_combined.columns = ['txId1', 'txId2', 'Time step', 'class_txId1', 'class_txId2']

# 结果展示
txs_combined.to_csv('txs_combined.csv')




