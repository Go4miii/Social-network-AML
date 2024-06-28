# Social-network-AML

This repository contains the code and resources for a project focused on fraud detection using Graph Neural Networks (GNNs), along with several traditional and unsupervised models. The project is organized into different directories, each containing specific components of the analysis and models used.

## Directory Structure

```
├── Descriptive Analysis
│   ├── Descriptive Analysis.ipynb
│   └── Figures
│       ├── Actors Count over Classes.png
│       ├── Illicit Transactions Network (T=32).png
│       ├── Licit Transactions (T=32).png
│       └── Transactions count over classes.png
├── HeteroGNN
│   ├── main.py
│   └── model.py
├── README.md
├── Unsupervised Models
│   ├── HDBSCAN
│   │   ├── HDBSCAN2_actors.py
│   │   ├── HDBSCAN_txs.py
│   │   ├── SC+HDBSCAN_actors_recall&precision.py
│   │   └── SC+HDBSCAN_txs_recall&precision.py
│   └── Spectral Clustering
│       ├── Actors_data_preprocessing.py
│       ├── SC_actors_36~49.py
│       ├── SC_txs_36~49_model.py
│       ├── SC_txs_41_results_plot.py
│       ├── SC_txs_data_pre.py
│       ├── addr_combined.csv
│       └── txs_combined.csv
└── supervised models
    ├── transaction
    │   ├── transaction_nosmote_norw.py
    │   ├── transaction_smote_norw.py
    │   └── transaction_smote_rw.py
    └── wallet
        ├── wallet_nosmote_norw.py
        ├── wallet_smote_norw.py
        └── wallet_smote_rw.py
```

## Descriptive Analysis

The `Descriptive Analysis` directory contains a Jupyter notebook that provides an exploratory data analysis of the dataset. It also includes various figures that visualize different aspects of the data.

- `Descriptive Analysis.ipynb`: Jupyter notebook with exploratory data analysis.
- `Figures/`: Directory containing figures generated during the analysis.
  - `Actors Count over Classes.png`: Distribution of actor counts over different classes.
  - `Illicit Transactions Network (T=32).png`: Network visualization of illicit transactions at T=32.
  - `Licit Transactions (T=32).png`: Network visualization of licit transactions at T=32.
  - `Transactions count over classes.png`: Distribution of transaction counts over different classes.

## HeteroGNN

The `HeteroGNN` directory contains the implementation of the Heterogeneous Graph Neural Network model.

- `main.py`: Main script to train and evaluate the HeteroGNN model.
- `model.py`: Definition of the HeteroGNN model architecture.

## Unsupervised Models

The `Unsupervised Models` directory contains implementations of unsupervised learning methods, including HDBSCAN and Spectral Clustering.

### HDBSCAN

- `HDBSCAN2_actors.py`: HDBSCAN clustering on actor data.
- `HDBSCAN_txs.py`: HDBSCAN clustering on transaction data.
- `SC+HDBSCAN_actors_recall&precision.py`: Script to evaluate recall and precision of HDBSCAN on actor data.
- `SC+HDBSCAN_txs_recall&precision.py`: Script to evaluate recall and precision of HDBSCAN on transaction data.

### Spectral Clustering

- `Actors_data_preprocessing.py`: Preprocessing of actor data for Spectral Clustering.
- `SC_actors_36~49.py`: Spectral Clustering on actor data from T=36 to T=49.
- `SC_txs_36~49_model.py`: Spectral Clustering model for transaction data from T=36 to T=49.
- `SC_txs_41_results_plot.py`: Script to plot results of Spectral Clustering on transaction data at T=41.
- `SC_txs_data_pre.py`: Preprocessing of transaction data for Spectral Clustering.
- `addr_combined.csv`: Combined address data.
- `txs_combined.csv`: Combined transaction data.

## Supervised Models

The `supervised models` directory contains scripts for supervised learning methods applied to transaction and wallet data.

### Transaction

- `transaction_nosmote_norw.py`: Supervised model on transaction data without SMOTE and reweighting.
- `transaction_smote_norw.py`: Supervised model on transaction data with SMOTE but without reweighting.
- `transaction_smote_rw.py`: Supervised model on transaction data with SMOTE and reweighting.

### Wallet

- `wallet_nosmote_norw.py`: Supervised model on wallet data without SMOTE and reweighting.
- `wallet_smote_norw.py`: Supervised model on wallet data with SMOTE but without reweighting.
- `wallet_smote_rw.py`: Supervised model on wallet data with SMOTE and reweighting.

## How to Run

1. **Clone the repository**:
   ```sh
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

2. **Set up the environment**:
   - Install the required packages using `requirements.txt` (if available) or manually install the necessary packages.

3. **Run Descriptive Analysis**:
   - Open and execute `Descriptive Analysis/Descriptive Analysis.ipynb` in Jupyter Notebook.

4. **Train and Evaluate Models**:
   - For HeteroGNN:
     ```sh
     python HeteroGNN/main.py
     ```
   - For Unsupervised Models, navigate to the respective directories and run the scripts.
   - For Supervised Models, navigate to the `transaction` or `wallet` directories and run the scripts.



请根据具体情况调整 `README.md` 文件中的信息，例如 GitHub 仓库的 URL、使用的依赖项和环境设置步骤。
