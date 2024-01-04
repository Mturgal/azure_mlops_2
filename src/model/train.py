# Import libraries

import argparse
import glob
import os
import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", dest="training_data", type=str)
    parser.add_argument("--reg_rate", dest="reg_rate", default=0.01,
                        type=float)

    args = parser.parse_args()
    return args

def main(args):
    # TO DO: enable autologging
    mlflow.autolog()
    
    # read data
    df = get_csvs_df(args.training_data)

    # split data
    X_train, X_test, y_train, y_test = split_data(df)

    # train model
    lg = train_model(args.reg_rate, X_train, X_test, y_train, y_test)


def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)


def split_data(df: pd.DataFrame) -> list:
    """Splits the dataframe to dataframe to train x and y and test x and y 

    Args:
        df: Dataframe to be split to train and test sets

    Returns:
        A list of split outputs
    """

    X, y = df[['Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure',
               'TricepsThickness', 'SerumInsulin', 'BMI', 'DiabetesPedigree',
               'Age']].values, df['Diabetic'].values
    return train_test_split(X, y, test_size=0.3, random_state=0)


def get_model_metrics(class_model: LogisticRegression, X_test: pd.DataFrame, 
                      y_test: pd.DataFrame) -> dict:
    """ Calcuates f1 score and accuracy score of the trained model on the 
        x_test dataframe

        Args:
            class_model: Trained sklearn classificaton model
            X_test: Dataframe with feature value tuples
            y_test: classes for the X 

        Returns:
            Dict of model accuracy and f1 score
    """
    preds = class_model.predict(X_test)
    accuracy = accuracy_score(preds, y_test)
    f1 = f1_score(preds, y_test)
    metrics = {"accuracy": accuracy, "f1": f1}
    return metrics





def train_model(reg_rate, X_train, X_test, y_train, y_test):
    # train model
    LogisticRegression(C=1/reg_rate, solver="liblinear").fit(X_train, y_train)


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str)
    parser.add_argument("--reg_rate", dest='reg_rate',
                        type=float, default=0.01)

    # parse args
    args = parser.parse_args()

    # return args
    return args

# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
