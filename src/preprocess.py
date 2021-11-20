from sklearn.model_selection import train_test_split
import pandas as pd


def split_test_data(path: str):
    """
    Reads in the data from a csv file and splits into a train and test set.

    Parameters
    ----------
    path : string
        The path to the csv file

    Returns
    -------
    train_df, test_df :
        The train and test datasets
    """
    data = pd.read_csv(path, sep=";")
    train_df, test_df = train_test_split(data, random_state=522, test_size=0.2)
    return train_df, test_df
