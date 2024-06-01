# build_features.py
# This module holds utility classes and functions that creates and manipulates input features
# This module also holds the various input transformers

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

def correlation_columns(dataset: pd.DataFrame, target_column: str, k: float=0.5):
    """
    Columns that are correlated to the target point

    Parameters
    ----------

    dataset: pd.DataFrame
        The pandas dataframe
    
    target_column: str
        The target column to calculate correlation against

    k: float
        The correlation cuttoff point; defaults to -0.5 and 0.5.
        The values passed in represents the negative and positive cutofff

    Returns
    -------

    columns: list
        A list of columns that are correlated to the target column based on the cutoff point
    """

    corr = np.abs(dataset.corr()[target_column])
    corr_sorted = corr.sort_values(ascending=False)
    columns = [col for col, value in zip(corr_sorted.index, corr_sorted.values) if value >= k and col != target_column]

    return columns

class ColumnExtractor(BaseEstimator, TransformerMixin):
    """Columns Extractor based on correlation to the output label"""
    
    def __init__(self, columns):
        print(columns)
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.columns]

if __name__ == '__main__':
    correlation_columns(pd.read_csv('././data/raw/creditcard.csv'), 'Class', k=0.3)