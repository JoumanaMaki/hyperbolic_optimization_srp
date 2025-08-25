import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import torch_geometric.transforms as T
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data

# -------------------------
# Device selection
# -------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# -------------------------
# Custom GraphConvolution
# -------------------------
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, dropout, act, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.dropout = dropout
        self.act = act

    def forward(self, x, edge_index, num_nodes):
        # Build dense adjacency (MPS-safe)
        adj = torch.zeros((num_nodes, num_nodes), device=edge_index.device)
        adj[edge_index[0], edge_index[1]] = 1.0

        # Linear projection
        hidden = self.linear(x)

        # Dropout
        hidden = F.dropout(hidden, self.dropout, training=self.training)

        # Message passing: adj * hidden
        support = torch.mm(adj, hidden)

        # Activation
        return self.act(support)


# -------------------------
# GCN model using GraphConvolution
# -------------------------
class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GraphConvolution(in_channels, hidden_channels, dropout, act=F.relu)
        self.conv2 = GraphConvolution(
            hidden_channels, out_channels, dropout, act=lambda x: x
        )

    def encode(self, x, edge_index, num_nodes):
        x = self.conv1(x, edge_index, num_nodes)
        return self.conv2(x, edge_index, num_nodes)

    def decode(self, z, edge_label_index):
        # Dot product between embeddings of node pairs
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


# -------------------------
# Load your dataset (tree1111)
# -------------------------
features = np.load("datasets/tree1111/g00_lp/g00_lp.feats.npz")["data"].reshape(
    1111, 1000
)
x = torch.tensor(features, dtype=torch.float)

df = pd.read_csv("datasets/tree1111/g00_lp/g00_lp.edges.csv")  # columns: parent, child
edge_index = torch.tensor([df["parent"].values, df["child"].values], dtype=torch.long)

data = Data(x=x, edge_index=edge_index)

transform = T.Compose(
    [
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(
            num_val=0.05,
            num_test=0.1,
            is_undirected=False,  # <-- your dataset is a tree (directed)
            add_negative_train_samples=False,
        ),
    ]
)
train_data, val_data, test_data = transform(data)

# -------------------------
# Training setup
# -------------------------
model = Net(
    in_channels=x.size(1), hidden_channels=128, out_channels=64, dropout=0.5
).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index, train_data.num_nodes)

    # Negative sampling each epoch
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index,
        num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1),
        method="sparse",
    )

    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat(
        [
            train_data.edge_label,
            train_data.edge_label.new_zeros(neg_edge_index.size(1)),
        ],
        dim=0,
    )

    out = model.decode(z, edge_label_index).view(-1)
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index, data.num_nodes)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())


# -------------------------
# Training loop with early stopping
# -------------------------
patience = 10
best_val_auc = 0
final_test_auc = 0
best_epoch = 0
patience_counter = 0

for epoch in range(1, 201):
    loss = train()
    val_auc = test(val_data)
    test_auc = test(test_data)

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        final_test_auc = test_auc
        best_epoch = epoch
        patience_counter = 0
    else:
        patience_counter += 1

    print(
        f"Epoch: {epoch:03d}, Loss: {loss:.4f}, "
        f"Val: {val_auc:.4f}, Test: {test_auc:.4f}"
    )

    if patience_counter >= patience:
        print(
            f"Early stopping at epoch {epoch}. "
            f"Best epoch was {best_epoch} with Val AUC {best_val_auc:.4f}, "
            f"Test AUC {final_test_auc:.4f}"
        )
        break

print(f"Final Test AUC (best val): {final_test_auc:.4f}")
