import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import geoopt


class HyperbolicGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, manifold_type="lorentz"):
        super().__init__()
        if manifold_type == "lorentz":
            self.manifold = geoopt.Lorentz()
        elif manifold_type == "poincare":
            self.manifold = geoopt.PoincareBall()
        else:
            raise ValueError("Invalid manifold type")
        
        self.fc1 = GCNConv(input_dim, hidden_dim)
        self.fc2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.fc1(x, edge_index))
        x = self.fc2(x, edge_index)
        return F.log_softmax(x, dim=1)
