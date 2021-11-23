# Author: Nikita Shymberg
# Date: Nov 23 2021

"""
TODO: docs
TODO: docopt arguments
TODO: tests
"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR


def read_data(path: str) -> pd.DataFrame:
    """
    TODO: docs
    TODO: tests
    TODO: does this even need to be in a function?
    """
    return pd.read_csv(path)


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
    Return the trained model and results
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
    # Create a dict of model names and objects
    models = {
        "Dummy": DummyRegressor(),
        "Ridge": Ridge(),
        "Random Forest": RandomForestRegressor(),
        "KNN": KNeighborsRegressor(),
        "Bayes": BayesianRidge(),
        "SVM": SVR(),
    }
    ...  # Create a dict of dicts for each model have param_dict
    ...  # Hyperparameter optimization for each model
    ...  # Test each model on test set
    ...  # Save results somewhere


if __name__ == "__main__":
    main()
