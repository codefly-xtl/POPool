import numpy as np

from core.utils import tab_printer
from train import Trainer
from myparser import parameter_parser

if __name__ == "__main__":
    # Parse the command line arguments
    args = parameter_parser()
    # Print the parsed arguments in a tabular format
    tab_printer(args)
    results = []
    # Run the training process 10 times to evaluate stability and performance
    for i in range(10):
        trainer = Trainer(args)
        accuracy = trainer.train()
        results.append(accuracy)
    # Calculate the average accuracy and standard deviation from the results
    avg_accuracy = np.mean(results)
    std_deviation = np.std(results)
    # Print the average accuracy and standard deviation
    print(f"avg_accuracy: {avg_accuracy}")
    print(f"std_deviation: {std_deviation}")
