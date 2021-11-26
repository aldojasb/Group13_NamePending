"""
This script analyzes the results of the models,
choose the best model to run against the test data
and saves the scores to a csv file.
Usage: analysis.py --r_path=<arg1>

Options:
--r_path=<arg1>      Path to write the results to

Example:
python analysis.py --r_path=results
"""


from docopt import docopt
from pandas.io.parsers import read_csv
from pathlib import Path
import pandas as pd
import numpy as np
from joblib import dump, load
from sklearn.metrics import  mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score

def main():
    args = docopt(__doc__)
    result_path = args['--r_path']

    data_processed_path = 'data/processed/' # hardcoded for now since I don't want too many arguments
    X_test = read_csv(data_processed_path
                      + 'X_test.csv')
    y_test = read_csv(data_processed_path
                      + 'y_test.csv')

    X_test = X_test.drop('Unnamed: 0', axis=1)
    y_test = y_test.drop('Unnamed: 0', axis=1)

    
    models = {
        "Dummy",
        "Ridge",
        "Random Forest",
        "KNN",
        "Bayes",
        "SVM",
    }
    best_test_score = 0
    best_model_name = None
    for model in models:
        r_data = read_csv(f'results/raw_results/{model}/{model}.csv')
        if r_data['mean_test_score'].values[0] > best_test_score:
            best_test_score = r_data['mean_test_score'].values[0]
            best_model_name = model
    
    best_model = load(
        f'results/raw_results/{best_model_name}/{best_model_name}.joblib')

    # get the scoring from the best model
    predictions = best_model.predict(X_test)
    r_2_score = r2_score(y_test, predictions)
    mse_score = mean_squared_error(y_test, predictions)
    rmae_score = np.sqrt(mse_score)
    mae_score = mean_absolute_error(y_test, predictions)
    mse_log_score = mean_squared_log_error(y_test, predictions)
    mae_log_score = median_absolute_error(y_test, predictions)
 
    results = pd.DataFrame(
        {
            'model': [best_model_name],
            'r2_score': [r_2_score],
            'mse_score': [mse_score],
            'rmae_score': [rmae_score],
            'mae_score': [mae_score],
            'mse_log_score': [mse_log_score],
            'mae_log_score': [mae_log_score],
        }
    )

    results.to_csv(f"{result_path}/best_model.csv", index=False)
    print('best_model.csv created at location /{}/'.format(result_path))


if __name__ == "__main__":
    main()
