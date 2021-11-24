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
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scipy.stats import loguniform


def read_data(path):
    """
    TODO: docs
    TODO: tests
    TODO: does this even need to be in a function?
    """
    return pd.read_csv(path, sep=";")  # TODO: remove sep


def fit_model(model, X_train, y_train, params=None, metrics=None, n_iter=50):
    """
    TODO: docs
    TODO: tests
    Returns the fit model
    """
    pipe = make_pipeline(
        StandardScaler(),  # TODO: do we need something better?
        model
    )
    if params is not None:
        searcher = RandomizedSearchCV(
            pipe,
            param_distributions=params,
            n_jobs=-1,
            n_iter=n_iter,
            cv=5,
            random_state=522,
            return_train_score=True,
            scoring=metrics,
            refit=True  # TODO: how to use a metric?
        )
        searcher.fit(X_train, y_train)
        return searcher
    else:  # TODO: might need some additional handling here
        model.fit(X_train, y_train)
        return model


def save_results(results, model, path):
    """
    TODO: docs
    TODO: tests
    Saves the raw cross validation results
    """
    ...


def main():
    # Read in processed data
    # TODO: change path
    X_train = read_data("data/raw/winequality/winequality-red.csv")
    y_train = X_train["quality"]
    X_train = X_train.drop(columns=["quality"])

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
            "randomforestregressor__n_estimators": np.arange(10, 500, 10),
            "randomforestregressor__criterion": [
                "squared_error",
                "absolute_error",
                "poisson"],
            "randomforestregressor__max_depth": np.arange(3, 25),
            "randomforestregressor__bootstrap": [True, False],
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
    metrics = "r2"

    # Hyperparameter optimization for each model
    for model_name in models:
        models[model_name] = fit_model(models[model_name], X_train, y_train,
                                       param_grid[model_name], metrics=metrics,
                                       n_iter=1)

    # Save results
    # TODO: save the model as well
    # save_results(results, "results/filename.something")  # TODO: fix path
    print(models)
    print(models)


if __name__ == "__main__":
    main()
