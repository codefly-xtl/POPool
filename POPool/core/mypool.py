import torch
import torch.nn as nn
from torch_geometric.utils import remove_self_loops
from torch_sparse import spspmm

from core import myScore


class Pooling(nn.Module):
    def __init__(self, hidden_size, ego_range, activate_function, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.ego_range = ego_range
        self.dropout = dropout
        self.activate_function = activate_function
        # Calculate node scores
        self.getfitness = myScore.MyScore(hidden_size)
        self.mlp = nn.Linear(hidden_size, hidden_size)
        self.att = nn.Parameter(nn.Parameter(torch.randn(1, 2 * hidden_size)))

    def get_hop_network(self, num_nodes, edge_index, hop):
        # Undirected graph
        res_edge_index = [edge_index]
        temp_edge_index = edge_index
        for _ in range(hop - 1):
            temp_edge_index, _ = spspmm(temp_edge_index, None, edge_index, None,
                                        num_nodes, num_nodes, num_nodes,
                                        coalesced=True)
            res_edge_index.append(temp_edge_index)
        # Output the new edge_index
        res_edge_index = torch.cat(res_edge_index, dim=1)
        res_edge_index, _ = remove_self_loops(res_edge_index, None)
        return res_edge_index

    def select_nodes(self, edge_index, fitness):
        edge_index, _ = remove_self_loops(edge_index)
        all_nodes = torch.unique(edge_index.flatten())
        del_nodes = edge_index[1][(fitness[edge_index[1]] - fitness[edge_index[0]]) < 0].unique()
        return all_nodes[~torch.isin(all_nodes, del_nodes)]

    def get_next(self, ego_edge_index, selected_nodes, fitness, edge_index, num_nodes, x, batch):
        # Mask to mark edges connected to selected_nodes
        mask = torch.isin(ego_edge_index[1], selected_nodes)
        edege_index_del = ego_edge_index[:, mask]
        # Get the remaining nodes, i.e., the nodes not in the ego_network of selected_nodes
        all_nodes = torch.unique(edge_index.flatten())
        del_nodes = torch.unique(edege_index_del.flatten())
        surplus_nodes = all_nodes[~torch.isin(all_nodes, del_nodes)]
        # Create the allocation matrix S
        s_indices = torch.cat([edege_index_del, torch.stack([selected_nodes, selected_nodes], dim=0),
                               torch.stack([surplus_nodes, surplus_nodes], dim=0)], dim=-1)
        # Renumber
        next_num_nodes = selected_nodes.size(0) + surplus_nodes.size(0)
        next_nodes = torch.cat([selected_nodes, surplus_nodes])
        map = torch.zeros(num_nodes, dtype=torch.long, device=x.device)
        map[next_nodes] = torch.arange(next_num_nodes, dtype=torch.long, device=x.device)
        s_indices[1] = map[s_indices[1]]
        # Get the adjacency matrix A of the next layer
        S = torch.sparse_coo_tensor(indices=s_indices,
                                    values=torch.zeros_like(s_indices[1], dtype=torch.float32),
                                    size=[num_nodes, next_num_nodes])
        A = torch.sparse_coo_tensor(indices=edge_index,
                                    values=torch.zeros_like(edge_index[1], dtype=torch.float32),
                                    size=[num_nodes, num_nodes])
        next_A = S.t() @ A @ S
        next_edge_index = next_A._indices()
        next_x = x[next_nodes] * fitness[next_nodes].unsqueeze(-1)
        next_batch = batch[next_nodes]
        return next_x, next_edge_index, S, next_batch

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        num_nodes = x.shape[0]
        # Generate ego network
        ego_edge_index = self.get_hop_network(num_nodes, edge_index, self.ego_range)
        # Get node scores and edge scores
        fitness = self.getfitness(x, ego_edge_index)
        # Select nodes with "score greater than neighbor score" based on node scores
        selected_nodes = self.select_nodes(edge_index, fitness)
        # Generate coarsened graph
        next_x, next_edge_index, S, next_batch = self.get_next(ego_edge_index,
                                                               selected_nodes,
                                                               fitness,
                                                               edge_index, num_nodes,
                                                               x, batch)
        return next_x, next_edge_index, S, next_batch
