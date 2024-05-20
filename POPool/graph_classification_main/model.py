import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCN, GAT, global_mean_pool, global_max_pool, GraphSAGE

from core.mypool import Pooling


class MyModel(nn.Module):
    def __init__(self, num_features, hidden_size, num_classes, coarse_layers, num_layers, ego_range, activate,
                 negative_slope, dropout, head, GNN):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.coarse_layers = coarse_layers
        self.ego_range = ego_range
        self.num_layers = num_layers
        self.head = head
        # Set the activation function based on the provided parameter
        if activate == "leaky_relu":
            self.activate_function = lambda x: F.leaky_relu(x, negative_slope)
        elif activate == "relu":
            self.activate_function = F.relu
        elif activate == "sigmoid":
            self.activate_function = torch.sigmoid
        elif activate == "tanh":
            self.activate_function = torch.tanh
        self.gnns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.transform = None
        # Initialize the GNN based on the provided parameter
        if GNN == "GCN":
            self.transform = GCN(num_features, hidden_size, self.num_layers, hidden_size)
        elif GNN == "GAT":
            self.transform = GAT(num_features, hidden_size, self.num_layers, hidden_size, heads=head)
        elif GNN == "GraphSAGE":
            self.transform = GraphSAGE(num_features, hidden_size, self.num_layers, hidden_size)
        # Pooling layers
        for i in range(coarse_layers):
            if GNN == "GCN":
                self.gnns.append(GCN(hidden_size, hidden_size, self.num_layers, hidden_size))
            elif GNN == "GAT":
                self.gnns.append(GAT(hidden_size, hidden_size, self.num_layers, hidden_size, heads=head))
            elif GNN == "GraphSAGE":
                self.gnns.append(GraphSAGE(hidden_size, hidden_size, self.num_layers, hidden_size))
            if i != self.coarse_layers - 1:
                self.pools.append(Pooling(hidden_size, self.ego_range, self.activate_function))
        # Classifier
        self.classifier1 = nn.Linear(hidden_size * 2, hidden_size)
        self.classifier2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x, edge_index, batch):
        # Transform the input features using the initial GNN
        x = self.transform(x, edge_index)
        x = self.activate_function(x)
        # Start coarse-graining
        readout_x = torch.zeros(batch.max() + 1, 2 * self.hidden_size, device=x.device)
        for i in range(self.coarse_layers):
            x = self.gnns[i](x, edge_index)
            x = self.activate_function(x)
            if i != self.coarse_layers - 1:
                x, edge_index, _, batch = self.pools[i](x, edge_index, batch)
                readout_x_mean = F.relu(global_mean_pool(x, batch))
                readout_x_max = F.relu(global_max_pool(x, batch))
                readout_x += torch.cat([readout_x_mean, readout_x_max], dim=-1)
        # Classification
        out = self.classifier1(readout_x)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.activate_function(out)
        out = self.classifier2(out)
        return out
