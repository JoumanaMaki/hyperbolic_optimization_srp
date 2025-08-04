import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from models.gnn_hyperbolic import HyperbolicGCN
from optimizers.riemannian import apply_riemannian_optimizer
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
dataset = Planetoid(root='./data/Cora', name='Cora', transform=T.NormalizeFeatures())
data = dataset[0].to(device)

# Instantiate model
model = HyperbolicGCN(
    input_dim=dataset.num_node_features,
    hidden_dim=16,
    output_dim=dataset.num_classes,
    manifold_type="lorentz"
).to(device)

# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


optimizer = apply_riemannian_optimizer(model)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test():
    model.eval()
    logits = model(data)
    pred = logits.argmax(dim=1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = pred[mask].eq(data.y[mask]).sum().item()
        acc = correct / mask.sum().item()
        accs.append(acc)
    return accs

# Train the model
for epoch in range(1, 201):
    loss = train()
    train_acc, val_acc, test_acc = test()
    if epoch % 20 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
