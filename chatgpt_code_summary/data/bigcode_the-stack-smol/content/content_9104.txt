from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression
from IMLearn.metrics import *

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)
    df = df.dropna()
    df = df[df['price'] > 0]
    df = df[df['yr_built'] > 0]
    df = df[df['bedrooms'] < 20]
    df['date'] = df['date'].apply(lambda x: int(str(x)[:4]))
    df = df[df['sqft_living'] <= df['sqft_lot']]
    labels_to_drop = ['zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']
    df.drop(columns=labels_to_drop, inplace=True)
    series = df.pop('price')
    return (df, series)


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    correlations = np.array(y.size)
    features = list(X)
    for feature in features:
        cov = np.cov(y, X[feature])
        std = np.std(X[feature]) * np.std(y)
        pearson_correlation = cov[0][1] / std
        np.append(correlations, pearson_correlation)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X[feature], y=y, mode="markers"))
        fig.update_layout(title=feature + " - Pearson Correlation  = "
                                + str(pearson_correlation),
                          xaxis_title=feature + " Feature values",
                          yaxis_title="House's Price")
        fig.write_image(f"{output_path}\\{feature}.png", format="png")


if __name__ == '__main__':

    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    df, series = load_data('../datasets/house_prices.csv')

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(df,series,'C:/Users/shahaf/Documents')

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(df, series, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall
    # training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10
    # times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon
    # of size (mean-2*std, mean+2*std)
    training_size = []
    average_loss = []
    var = []

    for p in range(10, 101):
        training_size.append(p / 100)
        mse_list = []
        for i in range(0, 10):
            train_sample = train_X.sample(frac=p / 100)
            sample_X, sample_y = train_sample, train_y.loc[
                train_sample.index]
            # model
            model = LinearRegression()
            model.fit(sample_X, sample_y)
            mse_list.append(model.loss(test_X, test_y))
        mse_arr = np.array(mse_list)
        average_loss.append(mse_arr.mean())
        var.append(mse_arr.std())
    var = np.array(var)
    average_loss = np.array(average_loss)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=training_size, y=average_loss,
                             mode="markers+lines",
                             marker=dict(color="LightSeaGreen"),name="Mean "
                                                                     "MSE"))
    fig.add_trace(go.Scatter(
        x=training_size, y=average_loss - 2 * var, mode="lines", line=dict(
            color="Aquamarine"),name="-2*STD"))
    fig.add_trace(
        go.Scatter(x=training_size,
                   y=average_loss + 2 * var, mode="lines", fill='tonexty',
                   line=dict(
                       color="Aquamarine"),name="+2*STD"))
    fig.update_layout(title = "Mean MSE vs Precentage of Samples for "
                              "Fitting")

    fig.show()
