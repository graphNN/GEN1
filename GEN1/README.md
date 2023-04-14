# Graph Ensemble Neural Network

## Requirements

python == 3.7.6

torch == 1.8.1

dgl == 0.5.2

## Results:
Note that the hyperparameters need to be adjusted manually.

# deep graph library
We run 'DGL_train_homophily' to get the results of we reproduced models GCN, GAT, SAGE, SGC, GIN, and APPNP on homophily datasets.
We run 'DGL_train_heterophily' to get the results of we reproduced models on heterophily datasets.

# graph ensemble neural network (ours)
We run 'GEN_train_homophily' to get the results of GEN on homophily datasets
We run 'GEN_train_heterophily' to get the results of GEN on heterophily datasets.

# analysis of hyperparameter, time and datasets
We run 'SALib_analysis' to get hyperparameter analysis.
We run 'dataset_analysis' to get the edge homophily ratio 'h' and time efficiency analysis.

# GCNII, BernNet, ChebNetII, H2GCN, and TWIRLS
We provide code links in the article.

## Citation

If using this code, please cite this paper.

