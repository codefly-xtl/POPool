import os
import random

import numpy as np
import torch
from texttable import Texttable

def tab_printer(args):
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def split_dataset_indices(data_len, train_ratio, val_ratio, test_ratio):
    # Calculate the number of samples in each dataset
    train_size = int(train_ratio * data_len)
    val_size = int(val_ratio * data_len)

    # Create dataset indices
    indices = np.arange(data_len)

    # Shuffle the data randomly
    np.random.shuffle(indices)

    # Split the dataset indices
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    return train_idx, val_idx, test_idx

def random_split(data, num_train_per_class, num_val_per_class):
    num_nodes = data.num_nodes
    num_classes = torch.unique(data.y).size(0)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    for cls in range(num_classes):
        # Find the indices of nodes belonging to the current class
        idx = (data.y == cls).nonzero().view(-1)
        # Shuffle the indices randomly
        idx = idx[torch.randperm(idx.size(0))]
        # Set the training set mask
        train_mask[idx[:num_train_per_class]] = True
        # Set the validation set mask
        val_mask[idx[num_train_per_class:num_train_per_class + num_val_per_class]] = True
        # Set the test set mask
        test_mask[idx[num_train_per_class + num_val_per_class:]] = True
    return train_mask, val_mask, test_mask
