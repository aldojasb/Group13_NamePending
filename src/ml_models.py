# Author: Nikita Shymberg
# Date: Nov 23 2021

"""
TODO: docs
TODO: docopt arguments
TODO: tests
"""
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR

from scipy.stats import loguniform


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
        "Ridge": Ridge(random_state=522),
        "Random Forest": RandomForestRegressor(random_state=522, n_jobs=-1),
        "KNN": KNeighborsRegressor(random_state=522),
        "Bayes": BayesianRidge(random_state=522),
        "SVM": SVR(random_state=522),
    }
    param_grid = {
        "Dummy": None,
        "Ridge": {
            "ridge__alpha": np.logspace(-3, 2, 6)
        },
        "Random Forest": {
            "randomforest__n_estimators": np.arange(10, 500, 10),
            "randomforest__criterion": ["gini", "entropy"],
            "randomforest__max_depth": np.arange(3, 25),
            "randomforest__bootstrap": [True, False],
            "randomforest__class_weight": [
                None, "balanced", "balanced_subsample"
                ],
        },
        "KNN": {
            "kneighborsregressor__n_neighbors": np.arange(2, 50),
            "kneighborsregressor__weights": ["uniform", "distance"],
        },
        "Bayes": None,  # TODO: read the docs again to try and understand these
        "SVM": {
            "svr__kernel": ["linear", "poly", "rbf", "sigmoid"],
            "svr__degree": np.arange(2, 5),
            "svr__gamma": loguniform(1e-3, 1e3),
            "svr__C": loguniform(1e-3, 1e3),
        }
    }
    ...  # Create a dict of dicts for each model have param_dict
    ...  # Hyperparameter optimization for each model
    ...  # Test each model on test set
    ...  # Save results somewhere


if __name__ == "__main__":
    main()
