# Author: Nikita Shymberg
# Date: Nov 23 2021

"""
This script trains several ML models on the pre-processed train data.
For each model, it does hyperparameter optimization and saves the model
that had the best cross-validation score. It also saves the
cross-validation report.
Usage: ml_models.py <train_path> <save_path>

Options:
<train_path>            The path to the training csv file
<save_path>             The folder to save the model results to

Example:
python ml_models.py data/train.csv, results/raw_results
"""
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

from scipy.stats import loguniform
from pathlib import Path
from docopt import docopt


def read_data(path):
    """
    TODO: docs
    TODO: tests
    TODO: does this even need to be in a function?
    """
    return pd.read_csv(path, sep=";")  # TODO: remove sep


def fit_model(model, X_train, y_train, params, metrics=None, n_iter=50):
    """
    TODO: docs
    TODO: tests
    Returns the fit model
    """
    pipe = make_pipeline(
        StandardScaler(),
        model
    )
    searcher = RandomizedSearchCV(
        pipe,
        param_distributions=params,
        n_jobs=-1,
        n_iter=n_iter,
        cv=5,
        random_state=522,
        return_train_score=True,
        scoring=metrics,
        refit=True  # FIXME: how to use a metric?
    )
    searcher.fit(X_train, y_train)
    return searcher


def save_results(results, model, path):
    """
    TODO: docs
    TODO: tests
    Saves the raw cross validation results
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(results)
    results.to_csv(f"{path}/{path.name}.csv")
    dump(model, f"{path}/{path.name}.joblib")


def load_model(path):
    """
    TODO: docs
    TODO: tests
    Returns the loaded model
    """
    return load(path)


def main():
    # parse arguments
    args = docopt(__doc__)
    train_path = args["<train_path>"]
    save_path = args["<save_path>"]

    # Read in pre-processed data
    X_train = read_data(train_path)
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
        "Dummy": {
            "dummyregressor__strategy": ["mean", "median"]
        },
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
        "Bayes": {
            "bayesianridge__alpha_1": loguniform(1e-7, 1e5),
            "bayesianridge__alpha_2": loguniform(1e-7, 1e5),
            "bayesianridge__lambda_1": loguniform(1e-7, 1e5),
            "bayesianridge__lambda_2": loguniform(1e-7, 1e5)
        },
        "SVM": {
            "svr__kernel": ["linear", "poly", "rbf", "sigmoid"],
            "svr__degree": np.arange(2, 5),
            "svr__gamma": loguniform(1e-3, 1e3),
            "svr__C": loguniform(1e-3, 1e3),
        }
    }
    metrics = {
        "negative MSE": "neg_mean_squared_error",
        "negarive RMSE": "neg_root_mean_squared_error",
        "megative MAE": "neg_mean_absolute_error",
        "r-squared": "r2",
    }
    metrics = "r2"

    # Hyperparameter optimization for each model
    for model_name in models:
        models[model_name] = fit_model(models[model_name], X_train, y_train,
                                       param_grid[model_name], metrics=metrics,
                                       n_iter=1)
        results = models[model_name].cv_results_
        results = pd.DataFrame(results)
        results = results[
            ["mean_test_score",
             "std_test_score",
             "mean_train_score",
             "std_train_score"]
        ]
        results = results.to_dict() | models[model_name].best_params_
        model = models[model_name].best_estimator_

        save_results(results, model, f"{save_path}/{model_name}")


if __name__ == "__main__":
    main()
