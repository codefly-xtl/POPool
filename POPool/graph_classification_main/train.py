import numpy as np
import torch
import torch.nn as nn
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from core.utils import split_dataset_indices
from model import MyModel

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        # Load the dataset based on the provided argument
        if args.dataset in ['PROTEINS', 'DD', 'NCI1', 'NCI109', 'MUTAG', 'Mutagenicity']:
            self.dataset = TUDataset(root='../data', name=args.dataset)
        self.feature_size = self.dataset.num_features
        self.num_classes = self.dataset.num_classes
        self.loss = nn.CrossEntropyLoss()
        self.dropout = args.dropout
        self.lr = args.lr
        self.ego_range = args.ego_range
        self.weight_decay = args.weight_decay
        self.hidden_size = args.hidden_size
        self.GNN = args.GNN
        self.head = args.head
        self.coarse_layers = args.coarse_layers
        self.num_layers = args.num_layers
        self.activate = args.activate
        self.negative_slope = args.negative_slope
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.val_ratio = args.val_ratio
        self.train_ratio = args.train_ratio
        self.model = None
        self.optimizer = None
        # Split the dataset into training, validation, and test sets
        train_idx, val_idx, test_idx = split_dataset_indices(len(self.dataset), self.train_ratio, self.val_ratio,
                                                             1 - self.train_ratio - self.val_ratio)
        train_idx = np.array(train_idx, dtype=np.int64)
        val_idx = np.array(val_idx, dtype=np.int64)
        test_idx = np.array(test_idx, dtype=np.int64)

        self.train_loader = DataLoader(self.dataset[train_idx], batch_size=self.args.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.dataset[val_idx], batch_size=self.args.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.dataset[test_idx], batch_size=self.args.batch_size, shuffle=True)

    def set_model(self):
        # Initialize the model
        self.model = MyModel(self.feature_size, self.hidden_size, self.num_classes, self.coarse_layers, self.num_layers,
                             self.ego_range, self.activate, self.negative_slope, self.dropout, self.head, self.GNN).to(self.device)

    def set_optimizer(self):
        # Initialize the optimizer
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def train_one_epoch(self, data_loader):
        self.model.train()
        total_loss = 0
        total_correct = 0
        for data in data_loader:
            self.optimizer.zero_grad()
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index, data.batch)
            loss = self.loss(out, data.y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            # Calculate training accuracy
            pred = out.max(dim=1)[1]
            correct = pred.eq(data.y).sum().item()
            total_correct += correct
        return total_loss / len(data_loader.dataset), total_correct / len(data_loader.dataset)

    def test(self, data_loader):
        self.model.eval()
        total_correct = 0
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                out = self.model(data.x, data.edge_index, data.batch)
                pred = out.max(dim=1)[1]
                correct = pred.eq(data.y).sum().item()
                total_correct += correct
        acc = total_correct / len(data_loader.dataset)
        return acc

    def train(self):
        self.set_model()
        self.set_optimizer()
        best_val_acc = 0
        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self.train_one_epoch(self.train_loader)
            val_acc = self.test(self.val_loader)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.args.dataset + self.GNN + 'best_model.pth')
            print(f"epoch: {epoch}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}")
        # Test the model
        self.set_model()
        self.model.load_state_dict(torch.load(self.args.dataset + self.GNN + 'best_model.pth'))
        test_acc = self.test(self.test_loader)
        print(f"Test accuracy: {test_acc:.2f}%")
        return test_acc
