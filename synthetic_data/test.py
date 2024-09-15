"""Model testing with PyTorch"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


def load_xy(df):
    """
    Load dataset

    args:
    - df: The dataframe
    """
    # replace nan with -99
    df = df.fillna(-99)
    target = "Loan Status"

    x_df = df.drop(target, axis=1)
    y_df = df[target]

    y_df = LabelEncoder().fit_transform(y_df)

    return x_df, y_df


def run_test(x_df, y_df):
    """
    Run test given features and labels

    args:
    - x_df: The features
    - y_df: The labels
    """
    mlp = MLPClassifier(max_iter=100)
    ## Feel free to play with these parameters if you want
    parameter_space = {
        "hidden_layer_sizes": [(5, 10), (12), (2, 5, 10, 15)],
        "activation": ["tanh", "relu", "logistic"],
        "solver": ["sgd", "adam"],
        "alpha": [0.0001, 0.05, 0.01],
        "learning_rate": ["constant", "adaptive"],
    }

    clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
    clf.fit(x_df, y_df)

    print("Best parameters found:\n", clf.best_params_)

    # Compare actuals with predicted values
    y_true, y_pred = y_df, clf.predict(x_df)

    print("Results on the test set:")
    print(classification_report(y_true, y_pred))


def test_model(datapath):
    """
    Test the trained model

    args:
    - datapath: The path to the data
    """
    df_base = pd.read_csv(datapath, sep=",")
    x_df, y_df = load_xy(df_base)
    run_test(x_df, y_df)
