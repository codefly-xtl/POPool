import torch
import torch.nn.functional as F
from torch import nn

from coarsen_model import Coarsen
from refine_model import Refine

class MyModel(nn.Module):
    def __init__(self, num_features, hidden_size, num_classes, coarse_layers, num_layers, ego_range, activate,
                 negative_slope, dropout, head, GNN):
        super().__init__()
        # Set the activation function based on the provided parameter
        if activate == "leaky_relu":
            self.activate_function = lambda x: F.leaky_relu(x, negative_slope)
        elif activate == "relu":
            self.activate_function = F.relu
        elif activate == "sigmoid":
            self.activate_function = torch.sigmoid
        elif activate == "tanh":
            self.activate_function = torch.tanh
        self.hidden_size = hidden_size
        # Initialize the Coarsen and Refine models
        self.coarsen = Coarsen(num_features, hidden_size, coarse_layers, num_layers, ego_range, self.activate_function,
                               dropout, head, GNN)
        self.refine = Refine(hidden_size, num_classes, coarse_layers, self.activate_function, num_layers, dropout, GNN,
                             head)

    def forward(self, x, edge_index):
        # Coarsening phase
        x, x_list, edge_index_list, S_list = self.coarsen(x, edge_index)
        # Refinement phase
        out = self.refine(x, x_list, edge_index_list, S_list)
        return out
