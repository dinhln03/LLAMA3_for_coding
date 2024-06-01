import pytest
import pandas as pd
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.33
RANDOM_STATE = 42


@pytest.fixture(scope="module")
def binary_dataset():
    df = pd.read_csv("./resources/heart.csv")
    features = df.iloc[0:, :-1]
    labels = df.iloc[0:, -1].values.ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    return X_train, X_test, y_train


@pytest.fixture(scope="module")
def multiclass_dataset():
    df = pd.read_csv("./resources/glass.csv")
    features = df.iloc[0:, :-1]
    labels = df.iloc[0:, -1].values.ravel()

    X_train, X_test, y_train, _ = train_test_split(
        features, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    return X_train, X_test, y_train
