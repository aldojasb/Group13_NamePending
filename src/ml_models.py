# Author: Nikita Shymberg
# Date: Nov 23 2021

"""
TODO: docs
TODO: docopt arguments
TODO: tests
"""
import pandas as pd


def read_data(path: str) -> pd.DataFrame:
    """
    TODO: docs
    TODO: tests
    """
    ...


def fit_model(model):
    """
    TODO: docs
    TODO: tests
    """
    ...


def evaluate_model(model):
    """
    TODO: docs
    TODO: tests
    """
    ...


def save_results(results, path):
    """
    TODO: docs
    TODO: tests
    """
    ...


def main():
    ...  # Read in processed data
    ...  # Create a dict of model names and objects
    ...  # Create a dict of dicts for each model have param_dict
    ...  # Hyperparameter optimization for each model
    ...  # Test each model on test set
    ...  # Save results somewhere


if __name__ == "__main__":
    main()
