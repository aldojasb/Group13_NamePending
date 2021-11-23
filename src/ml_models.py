# Author: Nikita Shymberg
# Date: Nov 23 2021

"""
TODO: docs
TODO: docopt arguments
TODO: tests
"""
import pandas as pd
import numpy as np
from pandas.io.parsers import read_csv

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR
from sklearn.metrics import make_scorer

from scipy.stats import loguniform


def read_data(path: str) -> pd.DataFrame:
    """
    TODO: docs
    TODO: tests
    TODO: does this even need to be in a function?
    """
    return pd.read_csv(path)


def fit_model(model, X_train, params, metrics=None):
    """
    TODO: docs
    TODO: tests
    Returns the fit model
    """
    ...


def evaluate_model(model, X_test, metrics=None):
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
    # Read in processed data
    # TODO: change path
    X_train = read_data("data/raw/winequality/winequality-red.csv")

    # Create models and hyperparameters
    models = {
        "Dummy": DummyRegressor(),
        "Ridge": Ridge(random_state=522),
        "Random Forest": RandomForestRegressor(random_state=522, n_jobs=-1),
        "KNN": KNeighborsRegressor(),
        "Bayes": BayesianRidge(),
        "SVM": SVR(),
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
    mape_scorer = make_scorer(
        lambda true, pred: 100 * np.mean(np.abs(pred - true / true)),
        greater_is_better=False
        )  # TODO: attribute this to lab 1
    metrics = {
        "negative MSE": "neg_mean_squared_error",
        "negarive RMSE": "neg_root_mean_squared_error",
        "megative MAE": "neg_mean_absolute_error",
        "r-squared": "r2",
        "MAPE": mape_scorer,
    }  # TODO: do we need any more?

    # Hyperparameter optimization for each model
    for model_name in models:
        models[model_name] = fit_model(models[model_name], X_train,
                                       param_grid[model_name], metrics=metrics)

    # Evaluate the models on the test set
    # TODO: change path
    X_test = read_data("data/raw/winequality/winequality-red.csv")
    results = {}
    for model_name in models:
        results[model_name] = evaluate_model(
            models[model_name], X_test, metrics)

    # Save results
    save_results(results, "results/filename.something")  # TODO: fix path


if __name__ == "__main__":
    main()
