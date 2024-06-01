import datetime as dt
import matplotlib.pyplot as plt
import lifetimes
import numpy as np
import os
import pandas as pd
import seaborn as sns

def numcard(x):
    return x.nunique(), len(x)
def todateclean(x):
    return pd.to_datetime(x, errors='coerce').dt.date.astype('datetime64')

"""
- info, shape, dtypes
- df.isnull().sum()  #Check for null counts/ value_counts()
- Check for supposed imputed values (are there suspicious values of 0, like for Age. )
- change zeros to nans where appropriate
- Imputation of missing values
- handle stringified json
- df.dtypes # in case obj to (df.colname = df.colname.astype("category"))
- df['colname'] = pd.to_datetime(df['colname']).dt.date
- df.drop("colname", axis=1) # drop columns
- How balanced are the outcomes?  
X = df.drop("diagnosis", axis=1) # just saying which axis again
Y = df["diagnosis"] # this is just a series now

col = X.columns # if we do type(col), it's an Index
X.isnull().sum() # this covers every column in the df.

def rangenorm(x):
    return (x - x.mean())/(x.max() - x.min())
le = LabelEncoder()
le.fit(Y_norm)
"""

df = pd.read_csv("./ignoreland/onlineretail.csv")
df.info()
df.apply(lambda x: numcard(x))

datecols = ['InvoiceDate']
df.loc[:, datecols] = df.loc[:,datecols].apply(lambda x: todateclean(x))

dfnew = df[(df.Quantity>0) & (df.CustomerID.isnull()==False)]
dfnew['amt'] = dfnew['Quantity'] * dfnew['UnitPrice']
dfnew.describe()

from lifetimes.plotting import *
from lifetimes.utils import *
observation_period_end = '2011-12-09'
monetary_value_col = 'amt'
modeldata = summary_data_from_transaction_data(dfnew,
                                               'CustomerID',
                                               'InvoiceDate',
                                               monetary_value_col=monetary_value_col,
                                               observation_period_end=observation_period_end)

modeldata.head()
modeldata.info()  # 4 floats.
# Eyeball distribution of frequency (calculated)
modeldata['frequency'].plot(kind='hist', bins=50)
print(modeldata['frequency'].describe())
print(modeldata['recency'].describe())
print(sum(modeldata['frequency'] == 0)/float(len(modeldata)))

##### Lec21
from lifetimes import BetaGeoFitter
# similar to lifelines
bgf = BetaGeoFitter(penalizer_coef=0.0)  # no regularization param.

bgf.fit(modeldata['frequency'], modeldata['recency'], modeldata['T'])
print(bgf)
# See https://www.youtube.com/watch?v=guj2gVEEx4s and
# https://www.youtube.com/watch?v=gx6oHqpRgpY
## residual lifetime value is more useful construct

from lifetimes.plotting import plot_frequency_recency_matrix
plot_frequency_recency_matrix(bgf)
from lifetimes.plotting import plot_probability_alive_matrix
plot_probability_alive_matrix(bgf)

# lec 24:
# set an outer time boundary and predict cumulative purchases by that time
t = 10 # from now until now+t periods
modeldata['predicted_purchases'] = \
    bgf.conditional_expected_number_of_purchases_up_to_time(t,
                                                            modeldata['frequency'],
                                                            modeldata['recency'],
                                                            modeldata['T'])
modeldata.sort_values(by='predicted_purchases').tail(5)
modeldata.sort_values(by='predicted_purchases').head(5)
# lec 25: validation of model
from lifetimes.plotting import plot_period_transactions
plot_period_transactions(bgf) # this plot shows very clearly the model performance
# in terms of transaction volume fit

# Lec 26: splitting into train and test (by time period)
summary_cal_holdout = calibration_and_holdout_data(df,
                                                   'CustomerID',
                                                   'InvoiceDate',
                                                   calibration_period_end='2011-06-08',
                                                   observation_period_end='2011-12-09')

summary_cal_holdout.head()

bgf.fit(summary_cal_holdout['frequency_cal'],
        summary_cal_holdout['recency_cal'],
        summary_cal_holdout['T_cal'])

from lifetimes.plotting import plot_calibration_purchases_vs_holdout_purchases

plot_calibration_purchases_vs_holdout_purchases(bgf, summary_cal_holdout)

from lifetimes.plotting import plot_history_alive


days_since_birth = 365
fig = plt.figure(figsize=(12,8))
id = 14621  # choose a customer id
sp_trans = df.loc[df['CustomerID'] == id]  # specific customer's covariates
plot_history_alive(bgf, days_since_birth, sp_trans, 'InvoiceDate')

# Lec28: Subsetting to customers who repurchase.
returning_customers_summary = modeldata[modeldata['frequency']>0]
returning_customers_summary.head()
returning_customers_summary.shape
# Lec 29: gamma-gamma model for LTV
# Note: good practice to confirm small/no apparent corr for frequency and mean trxn value
# Rev per trxn: predict total monetary value.
# The Beta param for the gamma model of total spend is itself assumed gamma distributed
# that is where the name comes from.
# teh expectation of total spend for person i is calculated in empirical-bayes fashion, as a weighted
# mean of population average and the sample mean for person i.
# eq 5 in http://www.brucehardie.com/notes/025/gamma_gamma.pdf shows the arithmetic
# https://antonsruberts.github.io/lifetimes-CLV/ also great additional code.
# derivation here: http://www.brucehardie.com/notes/025/gamma_gamma.pdf
# Output of ggf fitter:
# p = the 'alpha' param in the gamma dist: E(Z|p, v) = p/v.  Alpha adds upon convolution.
# q = the alpha param in the gamma dist of v -- v is gamma(q, gam) in the pop
# v = the 'beta' param in gamma dist. constant upon convolution.
#     -- Note that v varies among customers (ie, is gamma distributed)
from lifetimes import GammaGammaFitter
ggf = GammaGammaFitter(penalizer_coef=0.0)

ggf.fit(returning_customers_summary['frequency'],
        returning_customers_summary['monetary_value'])
ggf.summary
ggf.conditional_expected_average_profit(modeldata['frequency'],
                                        modeldata['monetary_value'])
# cond_exp_avg_profit => gives prediction of mean trxn value.
a0 = returning_customers_summary['monetary_value'].shape[0] # 2790 customers
# Total spend:
a1 = returning_customers_summary['monetary_value'].sum()
# Total time units (here, days) with purchase:
a2 = returning_customers_summary['frequency'].sum()
# Mean monetary value (over all purchase days), roughly equal to estimated v
returning_customers_summary['monetary_value'].mean()
ggf.summary
p_here = ggf.summary.iloc[0,0]
q_here = ggf.summary.iloc[1,0]
v_here = ggf.summary.iloc[2,0]  # model says 486; empirical average is 477.

money_per_customer = a1/a0

###############
# review, per documentation:
bgf.summary
# r, alpha = shape, scale for gamma dist that represents sum (convolution) of purchase rates
# a = alpha param for beta dist of churn
# b = beta param for beta dist of churn
x  = np.random.gamma(.784, 49.28,10000) # r, alpha, n
bgf.summary.loc["a",:][0]/ (bgf.summary.loc["b",:][0] + bgf.summary.loc["a",:][0])

###################################
# lec31: other models
dfnew.dtypes
dfnew_train = dfnew[dfnew.InvoiceDate < '2011-11-09']
dfnew_test = dfnew[dfnew.InvoiceDate >= '2011-11-09']
dfnew_test.shape
dfnew_train.shape
maxdate = dfnew_train.InvoiceDate.max()
mindate = dfnew_train.InvoiceDate.min()

dfnew_train['duration'] = (maxdate - dfnew_train.InvoiceDate)/np.timedelta64(1,'D')
dfsum1 = dfnew_train.groupby(['CustomerID'])['duration'].min().reset_index()
dfsum1.rename(columns = {'duration':'lasttime'}, inplace=True)  # time from lasttime to now

dfsum2 = dfnew_train.groupby(['CustomerID'])['duration'].max().reset_index()
dfsum2.rename(columns = {'duration':'firsttime'}, inplace=True)  # time from firsttime to now

dfnew_train['freq'] = 1
dfsum3 = dfnew_train.groupby(['CustomerID'])['freq'].sum().reset_index()  # count of transactions by customer

dfnew_train['freq3m'] = 1
dfsum4 = dfnew_train[dfnew_train['duration'] < 91].groupby(['CustomerID'])['freq3m'].sum().reset_index()

# now let's merge the 3 customer-level datasets together.
# pd.concat uses indexes as the join keys,
from functools import reduce
dfs = [dfsum1, dfsum2, dfsum3, dfsum4]
dfsum = reduce(lambda left, right: pd.merge(left, right, on=['CustomerID'], how='outer'), dfs)
dfsum.shape
[_ for _ in map(lambda x: x.shape, dfs)]
dfsum.head()

###################
other_data = pd.read_csv("./ignoreland/oth.csv")
other_data.head()
dfsum = pd.merge(dfsum, other_data, on=['CustomerID'], how='left')

dfnew_test['target'] = 1
dfsum_target = dfnew_test.groupby(['CustomerID'])['target'].sum().reset_index()
dfsum = pd.merge(dfsum, dfsum_target, on=['CustomerID'], how='left')
dfsum = dfsum.fillna(0).sort_values(['target'], ascending=False)

list(dfsum.columns)
# Lec 35 Xgboost
"""
reduce(Create tree, use tree to predict residuals, add.)
lightgbm is a faster implementation
"""
# lec36:
# Use xgboost to model the count of transactions per customer
import xgboost
from sklearn.model_selection import train_test_split
xgb_model = xgboost.XGBRegressor(n_estimators=2000, objective='reg:squarederror', max_depth=5)
predictors = ['lasttime', 'firsttime', 'freq', 'freq3m', 'score', 'discount']
X = dfsum[predictors]
y = dfsum['target']
# Split x, x, y, y | train, test; give test frac and random state
x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.32, random_state=867)
xgb_model.fit(x_train, y_train)

pred = xgb_model.predict(x_valid)  # vector of predicted
err = (pred - y_valid)**2  # squared errors
mse = err.sum()/len(err)
rmse = np.sqrt(mse)

from xgboost import plot_importance
x = list(zip(predictors, xgb_model.feature_importances_))
x.sort(key=lambda x: -x[1])
x
plot_importance(xgb_model)
# https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27
### Some global measures of xgboost feature importance:
# weight: number of times feature is used to split data (over all trees)
# cover: weight, weighted by data points being touched by those splits
# gain: mean training loss reduction (reduction in test-train) when the feature is used.
# argsort here returns the indices of the (reverse-sorted) feature importance values.
# Useful for grabbing index values and then working with arbitrarily zipped other lists (as I did above)
sorted_idx = np.argsort(xgb_model.feature_importances_)[::-1]
for _ in sorted_idx:
    print([x_train.columns[_], xgb_model.feature_importances_[_]])

[_ for _ in map(lambda x: xgb_model.get_booster().get_score(importance_type=x),
                ['gain','weight','cover','total_gain','total_cover'])]


def importances(model, lst):
    output = {}
    for x in lst:
        output[x] = model.get_booster().get_score(importance_type=x).values()
    return pd.concat([pd.Series(model.get_booster().feature_names), pd.DataFrame(output, columns=lst)],
                     axis=1)

aa = importances(xgb_model,['gain','weight','cover','total_gain','total_cover'])
aa

pd.concat([pd.Series(xgb_model.get_booster().feature_names), aa], axis=1)



##################
# using lightgbm:
import lightgbm as lgb
lgbparams = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mse',
    'max_depth': 6,
    'learning_rate': 0.02,
}
X1, X2, y1, y2 = train_test_split(X, y, test_size=0.32, random_state=867)

x_train, x_valid, y_train, y_valid = train_test_split(X1, y1, test_size=0.1, random_state=867)

x_train = x_train[predictors]
x_valid = x_valid[predictors]

d_train = lgb.Dataset(x_train, label=y_train)
d_valid = lgb.Dataset(x_valid, label=y_valid)

watchlist = [d_valid]

n_estimators = 2000

lightmodel = lgb.train(lgbparams, d_train, n_estimators, watchlist, verbose_eval=1)

importancelist = ['gain','split']

lightmodel.feature_importance(importance_type=importancelist[0])

importancdf = pd.DataFrame(pd.Series(predictors), columns=['feature'])

importancedf = reduce(lambda left, right: pd.concat([left, right], axis=1),
       [pd.Series(lightmodel.feature_importance(_)) for _ in importancelist])

importancedf.corr()

"""
frequency = number of periods in which a non-first purchase was made
T = age in same units of each customer
recency = period[last purchase] - period[first purchase]
monetary_value = sum(money)/(frequency+1)

# use utility functions to aggregate into useable format.
# https://lifetimes.readthedocs.io/en/latest/More%20examples%20and%20recipes.html
# sql examples for aggregating into RFM and doing holdout split.
"""


"""
Also, per brucehardie,
The integrated (function of 2 functions) nature of these problems yields to 
The gaussian hypergeometric function trick for evaluating the double integral.
"""
