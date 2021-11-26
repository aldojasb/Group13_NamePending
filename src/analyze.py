"""
This script test the model
Usage: test_model.py --m_path=<arg1> --r_path=<arg2>

Options:
--m_path=<arg1>       Path (not including filename) of where to locate the model file
--r_path=<arg2>      Path (not including filename) of where to write the file

Example:
python analysis.py --m_path=results/raw_results/ --r_path=result/
"""


from docopt import docopt
from pandas.io.parsers import read_csv
from pathlib import Path
import pandas as pd
from joblib import dump, load
from sklearn.metrics import mean_squared_error

def main():

    # parse arguments
    args = docopt(__doc__)

    # assign args to variables
    model_path = args['--m_path']
    result_path = args['--r_path']

    # load model
    # model = ml.load_model(model_path)
    path = 'data/processed/'
    X_test = read_csv(path + 'X_test.csv')
    y_test = read_csv(path + 'y_test.csv')
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
    # find the score of the best model
    best_score = best_model.score(X_test, y_test)

    # ml performance
    print(f"Best model: {best_model_name}")
    print(f"Best score: {best_score}")

   

    # print(X_test)
    # test model
    # results = model.score(X_test, y_test)
    # print(results)


    # path = Path(path)
    # path.mkdir(parents=True, exist_ok=True)

    # results = pd.DataFrame(results)
    # results.to_csv(f"{path}/{path.name}.csv")


if __name__ == "__main__":
    main()
