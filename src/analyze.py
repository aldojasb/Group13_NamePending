"""
This script test the model
Usage: test_model.py --model_path=<arg1> --out_path_data=<arg2>

Options:
--model_path=<arg1>         Path (not including filename) of where to locate the model file
--out_path_data=<arg2>      Path (not including filename) of where to write the file

Example:
python analysis.py --model_path=data/raw/ --out_path=data/raw/
-model_path=results/raw_results/Bayes/Bayes.joblib --out_path=result/
"""


from docopt import docopt
from pandas.io.parsers import read_csv
from pathlib import Path
import ml_models as ml
import pandas as pd


def main():

    # parse arguments
    args = docopt(__doc__)

    # assign args to variables
    model_path = args['--model_path']
    out_path = args['--out_path_data']

    # load model
    model = ml.load_model(model_path)
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
    models_with_test_score = {}
    for model in models:
        m_result = read_csv(f'results/raw_results/{model}/{model}.csv')
        models_with_test_score[model] = m_result['mean_test_score'].values[0]
   

    

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
