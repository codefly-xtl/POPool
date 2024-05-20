import argparse

def parameter_parser():
    parser = argparse.ArgumentParser(description="Graph classification model training script")
    parser.add_argument("--dataset", type=str, default='PROTEINS',
                        help="Choose dataset. Options include: 'PROTEINS', 'DD', 'NCI1', 'NCI109','MUTAG', 'Mutagenicity'.")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden unit size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--coarse_layers", type=int, default=3, help="Number of coarse layers")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GNN layers")
    parser.add_argument("--GNN", type=str, default="GCN", help="Type of GNN")
    parser.add_argument("--head", type=int, default=8, help="Number of heads in GAT")
    parser.add_argument("--ego_range", type=int, default=1, help="Range of the ego network")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--activate", type=str, default="leaky_relu", help="Activation function")
    parser.add_argument("--negative_slope", type=float, default=0.2, help="Negative slope in relu")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio")
    return parser.parse_args()
