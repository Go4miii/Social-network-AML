import pandas as pd

# 读取数
wallets_classes_df = pd.read_csv('D:/Users/dc1213/PycharmProjects/pythonProject1/wallets_classes.csv')
df_wallets_features_classes_combined = pd.read_csv('C:/Users/dc1213/Downloads/wallets_features_classes_combined.csv').drop_duplicates(subset='address', keep='first')
# 读取交易节点的边列表
txs_edgelist = pd.read_csv('D:/Users/dc1213/PycharmProjects/pythonProject1/txs_edgelist.csv')
# 读取交易的类型
txs_class = pd.read_csv('D:/Users/dc1213/PycharmProjects/pythonProject1/txs_classes.csv')
# 读取用户地址图
addr_addr_edgelist = pd.read_csv('D:/Users/dc1213/PycharmProjects/pythonProject1/AddrAddr_edgelist.csv')
# 读取用户地址到交易的边列表
addr_tx_edgelist = pd.read_csv('AddrTx_edgelist.csv')
tx_addr_edgelist = pd.read_csv('C:/Users/dc1213/Downloads/TxAddr_edgelist.csv')

# 筛选 AddrTx_edgelist，使其中的 txId 在 txs_edgelist 中出现
# known_txs_class = txs_class.loc[(txs_class['class'] != 3), 'txId']
known_addr_data = wallets_classes_df.loc[(wallets_classes_df['class'] != 3), 'address']
known_addr_features = df_wallets_features_classes_combined.loc[df_wallets_features_classes_combined['address'].isin(known_addr_data)]
filtered_addr_features = known_addr_features[(known_addr_features['Time step'] >= 36) & (known_addr_features['Time step'] <= 49)]
# 获取筛选后的交易ID
filtered_address = filtered_addr_features['address'].tolist()
filtered_addr_classes = wallets_classes_df[wallets_classes_df["address"].isin(filtered_address)]
filtered_addr_tx_edgelist = addr_tx_edgelist[(addr_tx_edgelist['txId'].isin(txs_class['txId'])) & (addr_tx_edgelist['input_address'].isin(filtered_addr_classes['address']))]
filtered_tx_addr_edgelist = tx_addr_edgelist[(tx_addr_edgelist['txId'].isin(txs_class['txId'])) & (tx_addr_edgelist['output_address'].isin(filtered_addr_features['address']))]
# 基于筛选后的 AddrTx_edgelist 筛选 AddrAddr_edgelist，只保留其中与筛选后的 input_address 有关的记录
filtered_input_addresses = filtered_addr_tx_edgelist['input_address'].unique()
filtered_output_addresses = filtered_tx_addr_edgelist['output_address'].unique()

filtered_addr_addr_edgelist = addr_addr_edgelist[
    addr_addr_edgelist['input_address'].isin(filtered_input_addresses) &
    addr_addr_edgelist['output_address'].isin(filtered_output_addresses)
]

# 合并交易流和时间步长
addr_combined = pd.merge(filtered_addr_addr_edgelist, filtered_addr_features[['address', 'Time step']], left_on='input_address', right_on='address', how='left')
addr_combined = pd.merge(addr_combined, filtered_addr_features[['address', 'Time step']], left_on='output_address', right_on='address', how='left', suffixes=('_input address', '_output address'))

# 合并交易流和class
addr_combined  = pd.merge(addr_combined , filtered_addr_classes, left_on='input_address', right_on='address', how='left', suffixes=('', '_class_input_address'))
addr_combined = pd.merge(addr_combined, filtered_addr_classes, left_on='output_address', right_on='address', how='left', suffixes=('', '_class_output_address'))

# print(addr_combined.columns)
# 选择所需的列
addr_combined = addr_combined[['input_address', 'output_address', 'Time step_input address', 'class', 'class_class_output_address']]

# 重命名列
addr_combined.columns = ['input address', 'output address', 'Time step', 'class_input_address', 'class_output_address']

# 结果展示
addr_combined.to_csv('addr_combined.csv')

