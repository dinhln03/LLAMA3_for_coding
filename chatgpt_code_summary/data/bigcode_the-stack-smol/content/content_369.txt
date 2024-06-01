import os.path

import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


from  IMLearn.metrics.loss_functions import mean_square_error


CITY_TEMPERATURE_DATA_PATH = os.path.join(os.path.curdir, "..", "datasets", "City_Temperature.csv")


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    data = pd.read_csv(filename, parse_dates=["Date"]).drop_duplicates()
    data = data.drop(data[data["Temp"] < -70].index)  # invalid Temp
    data["DayOfYear"] = data['Date'].dt.dayofyear

    return data


def question_2(data):
    """ Exploring data specifically in Israel """
    data = data.copy()
    data = data[data["Country"] == "Israel"]
    data["Year"] = data["Year"].astype(str)

    fig = px.scatter(data, x="DayOfYear", y="Temp", color="Year", width=1500, height=700,
                     labels={"DayOfYear": "Day of Year", "Temp": "Temperature"},
                     title="Q2(1) The relation between the day in the year and the temperature in Israel")
    fig.update_xaxes(range=[0, 365], tick0=0, dtick=20)
    fig.show()

    std_by_month = data.groupby("Month").std().reset_index()
    fig = px.bar(std_by_month, x="Month", y="Temp", width=1500, height=700,
                 labels={"Temp": "Std of the daily temperatures"},
                 title="Q2(2) The Standard Deviation of the Daily Temperatures Per Month in Israel")
    fig.data[-1].text = np.round(std_by_month["Temp"], 3)
    fig.update_xaxes(tick0=1, dtick=1)
    fig.update_traces(textposition='outside')
    fig.show()


def question_3(data):
    """ Exploring differences between countries"""
    agg_data_mean = data.groupby(["Country", "Month"]).mean().reset_index()
    agg_data_std = data.groupby(["Country", "Month"]).std().reset_index()

    fig = px.line(agg_data_mean, x="Month", y="Temp", color="Country", error_y=agg_data_std["Temp"],
                  width=1500, height=700, labels={"Temp": "Averaged Temperature"},
                  title="Q3 The Average Monthly Temperatures in Different Countries")
    fig.update_xaxes(tick0=1, dtick=1)
    fig.show()


def question_4(data):
    """ Fitting model for different values of `k` """
    data = data[data["Country"] == "Israel"]
    train_X, train_y, test_X, test_y = split_train_test(data["DayOfYear"], data["Temp"])

    losses = np.array([])
    for k in range(1, 11):
        poly_fit = PolynomialFitting(k)
        poly_fit.fit(train_X.to_numpy(), train_y.to_numpy())
        loss = poly_fit.loss(test_X.to_numpy(), test_y.to_numpy())
        losses = np.append(losses, round(loss, 2))
        print(k, loss)

    fig = px.bar(x=range(1, 11), y=losses, width=1500, height=700,
                 labels={"x": "Polynomials Degrees (k)", "y": "Test Error (MSE)"},
                 title="Q4 Test Errors for Different Polynomials Degrees (k)")
    fig.data[-1].text = losses
    fig.update_xaxes(tick0=1, dtick=1)
    fig.update_traces(textposition="outside")
    fig.show()


def question_5(data):
    """ Evaluating fitted model on different countries """
    data_israel = data[data["Country"] == "Israel"]

    poly_fit = PolynomialFitting(k=5)
    poly_fit.fit(data_israel["DayOfYear"], data_israel["Temp"])

    other_countries = ["Jordan", "South Africa", "The Netherlands"]
    losses = np.array([])

    for country in other_countries:
        country_data = data[data["Country"] == country]
        loss = poly_fit.loss(country_data["DayOfYear"], country_data["Temp"])
        losses = np.append(losses, loss)

    fig = px.bar(x=np.array(other_countries), y=losses, width=700, height=700,
                 labels={"x": "Country", "y": "Losses (MSE)"}, title="Q5 Losses (MSE) per Country With k=5")
    fig.data[-1].text = np.round(losses, 3)
    fig.update_traces(textposition="outside")
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data(CITY_TEMPERATURE_DATA_PATH)

    # Question 2 - Exploring data for specific country
    question_2(data)

    # Question 3 - Exploring differences between countries
    question_3(data)

    # Question 4 - Fitting model for different values of `k`
    question_4(data)

    # Question 5 - Evaluating fitted model on different countries
    question_5(data)

