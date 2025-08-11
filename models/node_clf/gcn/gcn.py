"""
Simple GCN Model Architecture, matches the paper below
https://arxiv.org/pdf/1609.02907
"""
import torch
from torch.nn import Linear, ReLU
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, num_node_features: int, hidden_channels: int, num_classes: int):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.relu = ReLU()
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.out = Linear(hidden_channels, num_classes)

    def forward(self, x: torch.tensor, edge_index: torch.tensor) -> torch.tensor:
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.out(x)
        
        return x
