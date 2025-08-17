"""
Simple GCN Model Architecture, matches the paper below
Kipf & Welling ICLR'17-style GCN: Semi-Supervised Classification with Graph Convolutional Networks
https://arxiv.org/pdf/1609.02907

"""

import torch
from torch.nn import ReLU, Dropout
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    """
    - 2 x GCNConv
    - ReLU after first conv
    - Dropout (default 0.5) on input features and between layers
    - Second conv outputs logits directly (no extra Linear)
    """

    def __init__(
        self,
        num_node_features: int,
        hidden_channels: int,
        num_classes: int,
        dropout_p: float = 0.5,
    ):
        super().__init__()
        # GCNConv already handles self-loops + normalized adjacency when normalize=True
        self.conv1 = GCNConv(
            num_node_features, hidden_channels, add_self_loops=True, normalize=True
        )
        self.relu = ReLU()
        self.dropout = Dropout(p=dropout_p)
        self.conv2 = GCNConv(
            hidden_channels, num_classes, add_self_loops=True, normalize=True
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)  # Dropout on input features
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)  # Dropout after first GCN layer
        x = self.conv2(x, edge_index)  # Final logits (use CrossEntropyLoss)
        return x
