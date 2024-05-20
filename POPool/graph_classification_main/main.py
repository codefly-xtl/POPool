from core.utils import tab_printer
from train import Trainer
from myparser import parameter_parser
import numpy as np

if __name__ == "__main__":
    # Parse command line arguments
    args = parameter_parser()
    # Print the parsed arguments in a tabular format
    tab_printer(args)
    results = []
    # Train the model 10 times to evaluate stability and performance
    for i in range(10):
        trainer = Trainer(args)
        accuracy = trainer.train()
        results.append((accuracy, i))
    # Calculate the average accuracy and standard deviation
    avg_accuracy = np.mean([result[0] for result in results])
    std_deviation = np.std([result[0] for result in results])

    # Print the average accuracy and standard deviation
    print(f"avg_accuracy: {avg_accuracy}")
    print(f"std_deviation: {std_deviation}")
