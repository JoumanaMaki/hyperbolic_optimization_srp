import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
import geoopt

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
dataset = Planetoid(root='data/Cora', name='Cora', transform=T.NormalizeFeatures())
data = dataset[0].to(device)

# Define a simple GCN with Geoopt manifold integration
class HyperbolicGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(HyperbolicGCN, self).__init__()
        self.manifold = geoopt.Lorentz(c=1.0)  # another option is PoincareBall
        self.fc1 = GCNConv(input_dim, hidden_dim)
        self.fc2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.fc1(x, edge_index))
        x = self.fc2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Instantiate model, optimizer
model = HyperbolicGCN(input_dim=dataset.num_node_features, hidden_dim=16, output_dim=dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Training loop
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Testing function
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
