#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2024/6/17 02:01 
# @Author : Iker Zhe 
# @Version：V 0.1
# @File : main.py
# @desc :
import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from dgl.nn import GraphConv
from tqdm import tqdm

from model import HeteroGNN, FocalLoss
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler



def create_graph(txs_df, wallets_df, txs2txs_edge, txs2wallet_edge, wallet2txs_edge, wallet2wallet_edge, device="cpu"):
    txs_id = txs_df.txId.to_list()
    wallets_id = wallets_df.address.to_list()
    scaler = StandardScaler()
    # 使用映射转换边
    txs2txs_edge = txs2txs_edge[txs2txs_edge['txId1'].isin(txs_id) & txs2txs_edge['txId2'].isin(txs_id)]
    txs2wallet_edge = txs2wallet_edge[
        txs2wallet_edge['txId'].isin(txs_id) & txs2wallet_edge['output_address'].isin(wallets_id)]
    wallet2txs_edge = wallet2txs_edge[
        wallet2txs_edge['input_address'].isin(wallets_id) & wallet2txs_edge['txId'].isin(txs_id)]
    wallet2wallet_edge = wallet2wallet_edge[
        wallet2wallet_edge['input_address'].isin(wallets_id) & wallet2wallet_edge['output_address'].isin(wallets_id)]

    txs_id_del = list(
        set(txs2txs_edge.txId1.to_list() + txs2txs_edge.txId2.to_list() +
            txs2wallet_edge.txId.to_list() + wallet2txs_edge.txId.to_list()))
    wallets_id_del = list(
        set(wallet2wallet_edge.input_address.to_list() +
            wallet2wallet_edge.output_address.to_list() +
            txs2wallet_edge.output_address.to_list() +
            wallet2txs_edge.input_address.to_list())
    )

    txs_df = txs_df[txs_df['txId'].isin(txs_id_del)]
    wallets_df = wallets_df[
        wallets_df['address'].isin(wallets_id_del)]

    txId_map = {txId: idx for idx, txId in enumerate(txs_df.txId.values)}
    wallet_map = {address: idx for idx, address in
                  enumerate(wallets_df.address.values)}

    # 使用映射转换边的索引
    txs2txs_edge['txId1'] = txs2txs_edge['txId1'].map(txId_map)
    txs2txs_edge['txId2'] = txs2txs_edge['txId2'].map(txId_map)
    txs2wallet_edge['txId'] = txs2wallet_edge['txId'].map(txId_map)
    txs2wallet_edge['output_address'] = txs2wallet_edge['output_address'].map(wallet_map)
    wallet2txs_edge['input_address'] = wallet2txs_edge['input_address'].map(wallet_map)
    wallet2txs_edge['txId'] = wallet2txs_edge['txId'].map(txId_map)
    wallet2wallet_edge['input_address'] = wallet2wallet_edge['input_address'].map(wallet_map)
    wallet2wallet_edge['output_address'] = wallet2wallet_edge['output_address'].map(wallet_map)

    # 删除任何包含 NaN 值的边
    txs2txs_edge.dropna(inplace=True)
    txs2wallet_edge.dropna(inplace=True)
    wallet2txs_edge.dropna(inplace=True)
    wallet2wallet_edge.dropna(inplace=True)

    data_dict = {
        ('transaction', 'relates', 'transaction'): (
            torch.tensor(txs2txs_edge['txId1'].dropna().values, dtype=torch.int64),
            torch.tensor(txs2txs_edge['txId2'].dropna().values, dtype=torch.int64)),
        ('transaction', 'linked_to', 'wallet'): (
            torch.tensor(txs2wallet_edge['txId'].dropna().values, dtype=torch.int64),
            torch.tensor(txs2wallet_edge['output_address'].dropna().values, dtype=torch.int64)),
        ('wallet', 'linked_to', 'transaction'): (
            torch.tensor(wallet2txs_edge['input_address'].dropna().values, dtype=torch.int64),
            torch.tensor(wallet2txs_edge['txId'].dropna().values, dtype=torch.int64)),
        ('wallet', 'relates', 'wallet'): (
            torch.tensor(wallet2wallet_edge['input_address'].dropna().values, dtype=torch.int64),
            torch.tensor(wallet2wallet_edge['output_address'].dropna().values, dtype=torch.int64))
    }

    g = dgl.heterograph(data_dict)

    # 对齐特征和图中的节点
    valid_tx_ids = txs_df[txs_df['txId'].isin(txId_map.keys())].index
    valid_wallet_ids = wallets_df[wallets_df['address'].isin(wallet_map.keys())].index

    # 过滤有效的特征数据
    tx_features = torch.tensor(scaler.fit_transform(txs_df.loc[valid_tx_ids, txs_df.columns[2:-1]].values),
                               dtype=torch.float32).to(device)
    tx_labels = torch.tensor(txs_df.loc[valid_tx_ids, 'class'].values, dtype=torch.long).to(device)  # 调整类标签为从0开始
    wallet_features = torch.tensor(
        scaler.fit_transform(wallets_df.loc[valid_wallet_ids, wallets_df.columns[3:]].values),
        dtype=torch.float32).to(device)
    wallet_labels = torch.tensor(wallets_df.loc[valid_wallet_ids, 'class'].values, dtype=torch.long).to(
        device)  # 调整类标签为从0开始

    # 添加节点特征和标签
    g.nodes['transaction'].data['features'] = tx_features
    g.nodes['transaction'].data['labels'] = tx_labels
    g.nodes['wallet'].data['features'] = wallet_features
    g.nodes['wallet'].data['labels'] = wallet_labels
    return g, txs_df, wallets_df


# 加载数据集
txs_df = pd.read_csv("./data/Elliptic++ Dataset/txs_features_classes_combined.csv")
txs_df = txs_df.apply(lambda x: x.fillna(x.mean()), axis=0)
wallets_df = pd.read_csv("./data/Elliptic++ Dataset/wallets_features_classes_combined_drop_duplicates.csv")
txs2txs_edge = pd.read_csv("./data/Elliptic++ Dataset/txs_edgelist.csv")
txs2wallet_edge = pd.read_csv("./data/Elliptic++ Dataset/TxAddr_edgelist.csv")
wallet2txs_edge = pd.read_csv("./data/Elliptic++ Dataset/AddrTx_edgelist.csv")
wallet2wallet_edge = pd.read_csv("./data/Elliptic++ Dataset/AddrAddr_edgelist.csv")

txs_id = txs_df.txId.to_list()
wallets_id = wallets_df.address.to_list()

txs2txs_edge = txs2txs_edge[txs2txs_edge['txId1'].isin(txs_id) & txs2txs_edge['txId2'].isin(txs_id)]
txs2wallet_edge = txs2wallet_edge[
    txs2wallet_edge['txId'].isin(txs_id) & txs2wallet_edge['output_address'].isin(wallets_id)]
wallet2txs_edge = wallet2txs_edge[
    wallet2txs_edge['input_address'].isin(wallets_id) & wallet2txs_edge['txId'].isin(txs_id)]
wallet2wallet_edge = wallet2wallet_edge[
    wallet2wallet_edge['input_address'].isin(wallets_id) & wallet2wallet_edge['output_address'].isin(wallets_id)]

txs_df['class'] = txs_df['class'] - 1
wallets_df['class'] = wallets_df['class'] - 1
txs_df['class'] = txs_df['class'].replace(2, -1)
wallets_df['class'] = wallets_df['class'].replace(2, -1)

recall_no_unknown_tx = []
recall_no_unknown_wallet = []
precision_no_unknown_tx = []
precision_no_unknown_wallet = []

recall_use_unknown_tx = []
recall_use_unknown_wallet = []
precision_use_unknown_tx = []
precision_use_unknown_wallet = []

for time_step in tqdm(range(35, 49)):
    train_txs = txs_df[txs_df['Time step'] <= time_step]
    test_txs = txs_df[txs_df['Time step'] == time_step + 1]
    train_wallets = wallets_df[wallets_df['Time step'] <= time_step]
    test_wallets = wallets_df[wallets_df['Time step'] == time_step + 1]

    train_txs_with_label = train_txs[train_txs['class'] != -1]
    train_wallets_with_label = train_wallets[train_wallets['class'] != -1]
    train_txs_without_label = train_txs[train_txs['class'] == -1]
    train_wallets_without_label = train_wallets[train_wallets['class'] == -1]

    test_txs_with_label = test_txs[test_txs['class'] != -1]
    test_wallets_with_label = test_wallets[test_wallets['class'] != -1]

    Gtest_with_label, test_txs_with_label_del, test_wallets_with_label_del = create_graph(
        txs_df=test_txs_with_label,
        wallets_df=test_wallets_with_label,
        txs2txs_edge=txs2txs_edge,
        txs2wallet_edge=txs2wallet_edge,
        wallet2txs_edge=wallet2txs_edge,
        wallet2wallet_edge=wallet2wallet_edge,
        device="cpu")

    Gtrain_with_label, train_txs_with_label_del, train_wallets_with_label_del = create_graph(
        txs_df=train_txs_with_label,
        wallets_df=train_wallets_with_label,
        txs2txs_edge=txs2txs_edge,
        txs2wallet_edge=txs2wallet_edge,
        wallet2txs_edge=wallet2txs_edge,
        wallet2wallet_edge=wallet2wallet_edge,
        device="cpu")
    Gtrain_without_label, train_txs_without_label_del, train_wallets_without_label_del = create_graph(
        txs_df=train_txs_without_label,
        wallets_df=train_wallets_without_label,
        txs2txs_edge=txs2txs_edge,
        txs2wallet_edge=txs2wallet_edge,
        wallet2txs_edge=wallet2txs_edge,
        wallet2wallet_edge=wallet2wallet_edge,
        device="cpu")

    train_test_txs_with_label_del = pd.concat([train_txs_with_label_del, test_txs_with_label_del], axis=0,
                                              ignore_index=True)
    train_test_wallets_with_label_del = pd.concat([train_wallets_with_label_del, test_wallets_with_label_del],
                                                  axis=0,
                                                  ignore_index=True)

    Gtrain_test_with_label, train_test_txs_with_label_del, train_test_wallets_with_label_del = create_graph(
        txs_df=train_test_txs_with_label_del,
        wallets_df=train_test_wallets_with_label_del,
        txs2txs_edge=txs2txs_edge,
        txs2wallet_edge=txs2wallet_edge,
        wallet2txs_edge=wallet2txs_edge,
        wallet2wallet_edge=wallet2wallet_edge,
        device="cpu")


    inputs_with_label = {
        'transaction': Gtrain_with_label.nodes['transaction'].data['features'],
        'wallet': Gtrain_with_label.nodes['wallet'].data['features']
    }

    inputs_without_label = {
        'transaction': Gtrain_without_label.nodes['transaction'].data['features'],
        'wallet': Gtrain_without_label.nodes['wallet'].data['features']
    }

    train_test_inputs_with_label = {
        'transaction': Gtrain_test_with_label.nodes['transaction'].data['features'],
        'wallet': Gtrain_test_with_label.nodes['wallet'].data['features']
    }



    # 计算类别权重
    device = "cpu"
    # 计算交易类别权重
    tx_class_sample_counts = np.bincount(Gtrain_with_label.nodes['transaction'].data['labels'].cpu().numpy())
    # tx_weight = 1. / tx_class_sample_counts
    # tx_weight = tx_class_sample_counts / len(Gtrain_with_label.nodes['transaction'].data['labels'].cpu().numpy())
    tx_weight = [tx_class_sample_counts[1] / tx_class_sample_counts[0], 1.8]
    tx_class_weights = torch.tensor(tx_weight, dtype=torch.float32).to(device)

    # 计算钱包类别权重
    wallet_class_sample_counts = np.bincount(Gtrain_with_label.nodes['wallet'].data['labels'].cpu().numpy())
    # wallet_weight = 1. / wallet_class_sample_counts
    wallet_weight = [wallet_class_sample_counts[1] / wallet_class_sample_counts[0], 1.2]
    # wallet_weight = wallet_class_sample_counts / len(Gtrain_with_label.nodes['wallet'].data['labels'].cpu().numpy())
    wallet_class_weights = torch.tensor(wallet_weight, dtype=torch.float32).to(device)

    # 定义损失函数并添加权重
    # tx_loss_fn = FocalLoss(alpha=1, gamma=2, weight=tx_class_weights)
    # wallet_loss_fn = FocalLoss(alpha=1, gamma=2, weight=wallet_class_weights)
    tx_loss_fn = nn.CrossEntropyLoss(weight=tx_class_weights)
    wallet_loss_fn = nn.CrossEntropyLoss(weight=wallet_class_weights)

    if time_step == 35:
        # 加载模型
        model = HeteroGNN(in_tx_feats=Gtrain_with_label.nodes['transaction'].data['features'].shape[1],
                          in_wallet_feats=Gtrain_with_label.nodes['wallet'].data['features'].shape[1],
                          hidden_size=64,
                          out_tx_feats=2,
                          out_wallet_feats=2).to(device)

        # 定义损失函数和优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)

        # 获取标签
        tx_labels = Gtrain_with_label.nodes['transaction'].data['labels']
        wallet_labels = Gtrain_with_label.nodes['wallet'].data['labels']

        model.train()
        for epoch in range(50):
            model.train()
            optimizer.zero_grad()
            outputs = model(Gtrain_with_label, inputs_with_label)
            tx_loss = tx_loss_fn(outputs['transaction'], tx_labels)
            wallet_loss = wallet_loss_fn(outputs['wallet'], wallet_labels)
            loss = tx_loss + wallet_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            print(f'Epoch {epoch}, Loss: {loss.item()}')
    else:
        # T >= 36时就只需要做微调
        tx_labels = Gtrain_with_label.nodes['transaction'].data['labels']
        wallet_labels = Gtrain_with_label.nodes['wallet'].data['labels']
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

        model.train()
        for epoch in range(10):
            model.train()
            optimizer.zero_grad()
            outputs = model(Gtrain_with_label, inputs_with_label)
            tx_loss = tx_loss_fn(outputs['transaction'], tx_labels)
            wallet_loss = wallet_loss_fn(outputs['wallet'], wallet_labels)
            loss = tx_loss + wallet_loss
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    tx_preds = outputs['transaction'].argmax(dim=1)
    wallet_preds = outputs['wallet'].argmax(dim=1)

    # 评估模型
    model.eval()
    with torch.no_grad():
        # outputs = model(Gtrain_without_label, inputs_without_label)
        outputs = model(Gtrain_test_with_label, train_test_inputs_with_label)
        tx_preds = outputs['transaction'].argmax(dim=1)
        wallet_preds = outputs['wallet'].argmax(dim=1)
        transaction_train_size = train_txs_with_label_del.shape[0]
        tx_real = Gtrain_test_with_label.nodes['transaction'].data['labels'][transaction_train_size:]
        cm = confusion_matrix(tx_real, tx_preds.cpu()[transaction_train_size:])
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix for Transactions')
        plt.savefig('./res1/confusion_matrix_tx_no_unknown_T={}.pdf'.format(time_step), format='pdf', dpi=300)
        plt.show()

        wallet_train_size = train_wallets_with_label_del.shape[0]
        wallet_real = Gtrain_test_with_label.nodes['wallet'].data['labels'][wallet_train_size:]
        cm = confusion_matrix(wallet_real, wallet_preds.cpu()[wallet_train_size:])
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix for Actors')
        plt.savefig('./res1/confusion_matrix_wallet_no_unknown_T={}.pdf'.format(time_step), format='pdf', dpi=300)
        plt.show()

        recall_no_unknown_tx.append(recall_score(tx_real, tx_preds.cpu()[transaction_train_size:], pos_label=0))
        recall_no_unknown_wallet.append(recall_score(wallet_real, wallet_preds.cpu()[wallet_train_size:], pos_label=0))
        precision_no_unknown_tx.append(precision_score(tx_real, tx_preds.cpu()[transaction_train_size:], pos_label=0))
        precision_no_unknown_wallet.append(precision_score(wallet_real, wallet_preds.cpu()[wallet_train_size:], pos_label=0))


    # tagging the unlabeled data
    model.eval()
    with torch.no_grad():
        outputs = model(Gtrain_without_label, inputs_without_label)
        tx_preds4_without = outputs['transaction'].argmax(dim=1)
        wallets_preds4_without = outputs['wallet'].argmax(dim=1)

    train_txs_without_label_del["class"] = tx_preds4_without
    train_wallets_without_label_del["class"] = wallets_preds4_without
    train_all_txs_with_label_del = pd.concat([train_txs_with_label_del, train_txs_without_label_del], axis=0,
                                             ignore_index=True)
    train_all_wallets_with_label_del = pd.concat([train_wallets_with_label_del, train_wallets_without_label_del],
                                                 axis=0,
                                                 ignore_index=True)

    Gtrain_all, train_all_txs_with_label_del, train_all_wallets_with_label_del = create_graph(
        txs_df=train_all_txs_with_label_del,
        wallets_df=train_all_wallets_with_label_del,
        txs2txs_edge=txs2txs_edge,
        txs2wallet_edge=txs2wallet_edge,
        wallet2txs_edge=wallet2txs_edge,
        wallet2wallet_edge=wallet2wallet_edge,
        device="cpu")

    train_all_inputs = {
        'transaction': Gtrain_all.nodes['transaction'].data['features'],
        'wallet': Gtrain_all.nodes['wallet'].data['features']
    }

    tx_labels_all = Gtrain_all.nodes['transaction'].data['labels']
    wallet_labels_all = Gtrain_all.nodes['wallet'].data['labels']

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    model.train()
    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        outputs = model(Gtrain_all, train_all_inputs)
        tx_loss = tx_loss_fn(outputs['transaction'], tx_labels_all)
        wallet_loss = wallet_loss_fn(outputs['wallet'], wallet_labels_all)
        loss = tx_loss + wallet_loss
        loss.backward()
        optimizer.step()
        # scheduler.step()
        print(f'Epoch {epoch}, Loss: {loss.item()}')

    model.eval()
    with torch.no_grad():
        # outputs = model(Gtrain_without_label, inputs_without_label)
        outputs = model(Gtrain_test_with_label, train_test_inputs_with_label)
        tx_preds = outputs['transaction'].argmax(dim=1)
        wallet_preds = outputs['wallet'].argmax(dim=1)
        transaction_train_size = train_txs_with_label_del.shape[0]
        tx_real = Gtrain_test_with_label.nodes['transaction'].data['labels'][transaction_train_size:]
        cm = confusion_matrix(tx_real, tx_preds.cpu()[transaction_train_size:])
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix for Transactions')
        plt.savefig('./res1/confusion_matrix_tx_T={}.pdf'.format(time_step), format='pdf', dpi=300)
        plt.show()

        wallet_train_size = train_wallets_with_label_del.shape[0]
        wallet_real = Gtrain_test_with_label.nodes['wallet'].data['labels'][wallet_train_size:]
        cm = confusion_matrix(wallet_real, wallet_preds.cpu()[wallet_train_size:])
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix for Actors')
        plt.savefig('./res1/confusion_matrix_wallet_T={}.pdf'.format(time_step), format='pdf', dpi=300)
        plt.show()

        recall_use_unknown_tx.append(recall_score(tx_real, tx_preds.cpu()[transaction_train_size:], pos_label=0))
        recall_use_unknown_wallet.append(recall_score(wallet_real, wallet_preds.cpu()[wallet_train_size:], pos_label=0))
        precision_use_unknown_tx.append(precision_score(tx_real, tx_preds.cpu()[transaction_train_size:], pos_label=0))
        precision_use_unknown_wallet.append(precision_score(wallet_real, wallet_preds.cpu()[wallet_train_size:], pos_label=0))
        # print(recall_score(tx_real, tx_preds.cpu()[transaction_train_size:], pos_label=0))
        # print(
        #     precision_score(tx_real, tx_preds.cpu()[transaction_train_size:], pos_label=0))
        #
        # print(recall_score(wallet_real, wallet_preds.cpu()[wallet_train_size:], pos_label=0))
        # print(precision_score(wallet_real, wallet_preds.cpu()[wallet_train_size:], pos_label=0))

        #
        # tx_accuracy = (tx_preds == tx_labels).float().mean()
        # wallet_accuracy = (wallet_preds == wallet_labels).float().mean()
res_dict = {
    'time_step': [x for x in range(36, 50)],
    'recall_no_unknown_tx': recall_no_unknown_tx,
    'recall_no_unknown_wallet': recall_no_unknown_wallet,
    'precision_no_unknown_tx': precision_no_unknown_tx,
    'precision_no_unknown_wallet': precision_no_unknown_wallet,
    'recall_use_unknown_tx': recall_use_unknown_tx,
    'recall_use_unknown_wallet': recall_use_unknown_wallet,
    'precision_use_unknown_tx': precision_use_unknown_tx,
    'precision_use_unknown_wallet': precision_use_unknown_wallet
}
res_df = pd.DataFrame(res_dict)
res_df.to_csv("./res1/rp_res.csv", index=False)


