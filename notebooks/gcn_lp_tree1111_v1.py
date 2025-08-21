"""
Link Prediction with PyTorch Geometric
This script demonstrates how to perform link prediction using a Graph Convolutional Network (GCN) with PyTorch Geometric.
It includes data loading, model definition, training, and evaluation.

References:
https://github.com/pyg-team/pytorch_geometric/blob/master/examples/link_pred.py
"""


import os.path as osp
import sys
import torch
from sklearn.metrics import roc_auc_score

import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling


# Add project root to path
project_root = osp.abspath(osp.join(osp.dirname(__file__), ".."))
sys.path.append(project_root)

from utils.seed_everything import seed_everything
from models.link_prediction.gcn_lp import GCN_LP
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                      add_negative_train_samples=False),
])

# ------------------------
# Load features
# ------------------------
features = np.load('datasets/tree1111/g00_lp/g00_lp.feats.npz')

if 'data' in features and 'indices' in features and 'indptr' in features:
    # Looks like a scipy sparse CSR/CSC format
    from scipy.sparse import csr_matrix
    feats_sparse = csr_matrix((features['data'], features['indices'], features['indptr']),
                              shape=features['shape'])
    x = torch.tensor(feats_sparse.toarray(), dtype=torch.float)
elif 'data' in features:  
    # Already dense np array
    x = torch.tensor(features['data'].reshape(1111, 1000), dtype=torch.float)
else:
    raise ValueError("Unsupported .npz format for features!")


# ------------------------
# Load edges
# ------------------------
df = pd.read_csv('datasets/tree1111/g00_lp/g00_lp.edges.csv')  # columns: ['parent','child']
edge_index = torch.tensor(np.array([np.array(df['parent'].values), np.array(df['child'].values)]), dtype=torch.long)


# ------------------------
# Build PyG Data object
# ------------------------
data = Data(x=x, edge_index=edge_index)

seed_everything()  # Set random seed for reproducibility

# ------------------------
# Train/val/test split
# ------------------------
transform = T.Compose([
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=False,
                      add_negative_train_samples=False),
])
train_data, val_data, test_data = transform(data)




model = GCN_LP(data.num_features, 128, 64).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)

    # We perform a new round of negative sampling for every training epoch:
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

    out = model.decode(z, edge_label_index).view(-1)
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())


best_val_auc = final_test_auc = 0
for epoch in range(1, 201):
    loss = train()
    val_auc = test(val_data)
    test_auc = test(test_data)
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        final_test_auc = test_auc
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
          f'Test: {test_auc:.4f}')

print(f'Final Test: {final_test_auc:.4f}')
