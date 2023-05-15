from torch_geometric.datasets import Planetoid
from torch_geometric.nn.inits import glorot, zeros
import torch_geometric as pyg
from torch_geometric.nn import GCNConv, GAE
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_networkx, to_dense_adj
import torch_sparse
import evaluate
import argparse

from input_data import load_data

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=128, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=64, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--head', type=int, default=16, help='Head.')
parser.add_argument('--dataset_str', type=str, default='citeseer', help='type of dataset.')

args = parser.parse_args()


# dataset = Planetoid(root='./data', name='Cora')
# data = dataset[0]


print("Using {} dataset".format(args.dataset_str))

data = load_data(args.dataset_str)



adj = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)

print(adj.shape)

#adj = torch_sparse.SparseTensor(row=data.edge_index[0], col=data.edge_index[1])


class GATConv(pyg.nn.MessagePassing):
    def __init__(self, in_features, out_features, k_heads=1, concat=True):
        super().__init__(aggr="add", node_dim=0)
        self.in_features = in_features
        self.out_features = out_features
        self.k_heads = k_heads
        self.concat = concat
        
        self.weight = torch.nn.Parameter(torch.Tensor(in_features, out_features * k_heads))
        if concat:
            self.bias = torch.nn.Parameter(torch.Tensor(k_heads * out_features))
        else:
            self.bias = torch.nn.Parameter(torch.Tensor(1, out_features))
            
        self.att_l = torch.nn.Parameter(torch.Tensor(1, k_heads, out_features))
        self.att_r = torch.nn.Parameter(torch.Tensor(1, k_heads, out_features))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att_l)
        glorot(self.att_r)
        zeros(self.bias)
        
    def forward(self, x, edge_index):
        x_lin = (x @ self.weight).view(-1, self.k_heads, self.out_features)
        
        alpha_l = torch.sum(x_lin * self.att_l, dim=2)
        alpha_r = torch.sum(x_lin * self.att_r, dim=2)
        
        out = self.propagate(edge_index, x=x_lin, alpha=(alpha_l, alpha_r))
        
        if self.concat:
            out = out.view(-1, self.k_heads * self.out_features)
        else:
            out = out.mean(dim=1)
        
        out += self.bias
        
        return out
        
    def message(self, x_j, alpha_i, alpha_j, index):
        alpha = alpha_i + alpha_j
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = pyg.utils.softmax(alpha, index)
        return alpha.unsqueeze(-1) * x_j
        
        
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, k_heads=8):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, k_heads=k_heads)
        self.conv2 = GATConv(hidden_dim * k_heads, out_dim, k_heads=k_heads, concat=False)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))       
        return self.conv2(x, edge_index)
    
        
model = GAE(Encoder(data.num_features, hidden_dim=args.hidden1, out_dim=args.hidden2, k_heads=args.head)).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

def get_acc(adj_rec, adj_label):
    labels_all = adj_label.to_dense().view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy
    
    
for epoch in range(args.epochs):
    model.train()
    optimizer.zero_grad()
    
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    adj = adj.to(device)
    
    # z holds the node embeddings
    z = model.encode(x, edge_index)
    train_acc = get_acc(torch.sigmoid(torch.matmul(z, z.t())), adj)
    loss = model.recon_loss(z, edge_index)
    training_loss = loss.item()
    loss.backward()
    optimizer.step()
    print(f'Epoch: {epoch}, Loss: {loss.item():.4f}, ACC: {train_acc}')

with torch.no_grad():
    z = model.encode(data.x.to(device), edge_index)
    
embedding = z.cpu().detach().numpy()

y = data.y



print(".......................KMEANS..............................")

    
Knmi = evaluate.kmeans(embedding, y)


print("NMI : ", Knmi*100)

print("..................AGG CLUSTERING.......................")

Snmi =  evaluate.AGC(embedding, y)

print("NMI : ", Snmi*100)

print("..................GAUSSIAN MIXTURE MODEL.......................")

Gnmi =  evaluate.GMM(embedding, y)

print("NMI : ", Gnmi*100)

print("..................DBSCAN.......................")

dnmi =  evaluate.dbscan(embedding, y)

print("NMI : ", dnmi*100)

print(".......................FUZZY-C-MEANS..............................")


fnmi =  evaluate.fuzzy(embedding, y)

print("NMI : ", fnmi*100)

