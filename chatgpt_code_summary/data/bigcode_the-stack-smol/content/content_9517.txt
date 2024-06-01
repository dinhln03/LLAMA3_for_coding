# -*- coding: utf-8 -*-
import QUANTAXIS as QA
from QUANTAXIS.QAFetch import QATusharePro as pro
import pandas as pd
import numpy as np
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark import SparkContext,SparkConf
from pyspark.sql.session import SparkSession
from QUANTAXIS.ML import RegUtil
from pyspark.sql.types import StructType,DoubleType,StructField,StringType
#from pyspark.sql.functions import
import copy
import talib

spark = SparkSession.builder.appName("my app").getOrCreate()
#spark.sparkContext.setLogLevel("INFO")
spark.conf.set("spark.sql.execution.arrow.enabled", "true")
start_3years_bf = '20150101'
industry_daily = pro.QA_fetch_get_industry_daily(start=start_3years_bf, end='20181231').sort_values(['industry','trade_date'], ascending = True)
industry_daily = spark.createDataFrame(industry_daily)
new_struct = ['q_dtprofit_ttm_poly', 'q_gr_poly', 'q_profit_poly', 'q_dtprofit_poly', 'q_opincome_poly', 'industry_roe', 'industry_pe', 'roe_ttm', 'industry_pe_ttm']
p1 = StructType()
p1.add(StructField('trade_date', StringType()))
p1.add(StructField('industry', StringType()))
list(map(lambda x: p1.add(StructField(x, DoubleType())), new_struct))
start = '20180101'
end = '20181231'
@pandas_udf(p1, PandasUDFType.GROUPED_MAP)
def _trend(key,data):
    dates = [str(int(start[0:4]) - 3) + '0831',str(int(start[0:4]) - 3) + '1031',
             str(int(start[0:4]) - 2) + '0431', str(int(start[0:4]) - 2) + '0831',
             str(int(start[0:4]) - 2) + '1031', str(int(start[0:4]) - 1) + '0431',
             str(int(start[0:4]) - 1) + '0831', str(int(start[0:4]) - 1) + '1031']
    _lam_f = lambda x, y: y[y.trade_date <= x].iloc[-1] if y[y.trade_date <= x].shape[0]>0 else None
    resampledf = pd.DataFrame(list(filter(lambda x:x is not None,map(_lam_f, dates,[data]*8))))
    col = ['trade_date', 'industry']
    col = col+new_struct
    indicator = pd.DataFrame(columns=col)
    df = data[data.trade_date >= start]
    df.reset_index(drop=True)
    for index,item in df.iterrows():
        if item.trade_date[4:8] <= "0831" and item.trade_date[4:8] > "0431" and item.trade_date[0:4] + '0431' not in dates:
            dates.append([item.trade_date[0:4] + '0431'])
            t = list(filter(lambda x:x is not None,map(_lam_f, [item.trade_date[0:4] + '0431'],[data])))
            if t is not None:
                resampledf = resampledf.append(t)
        if item.trade_date[4:8] <= "1031" and item.trade_date[4:8] > "0831" and item.trade_date[0:4] + '0831' not in dates:
            dates.append([item.trade_date[0:4] + '0831'])
            t = list(filter(lambda x: x is not None, map(_lam_f, [item.trade_date[0:4] + '0831'], [data])))
            if t is not None:
                resampledf = resampledf.append(t)
        if item.trade_date[4:8] > "1031" and item.trade_date[0:4] + '1031' not in dates:
            dates.append([item.trade_date[0:4] + '1031'])
            t = list(filter(lambda x: x is not None, map(_lam_f, [item.trade_date[0:4] + '1031'], [data])))
            if t is not None:
                resampledf = resampledf.append(t)
        resample = resampledf.append(list(map(_lam_f, [item.trade_date], [data])))
        resample = resample.dropna(how='all')
        ind = -8 if resample.shape[0]>8 else -resample.shape[0]
        fit, p3 = RegUtil.regress_y_polynomial(resample[ind:].q_dtprofit_ttm, poly=3, show=False)
        # fit, p4 = RegUtil.regress_y_polynomial(resample[-8:].q_opincome_ttm, poly=3, show=False)
        fit, p5 = RegUtil.regress_y_polynomial(resample[ind:].q_gr, poly=3, show=False)
        fit, p6 = RegUtil.regress_y_polynomial(resample[ind:].q_profit, poly=3, show=False)
        fit, p7 = RegUtil.regress_y_polynomial(resample[ind:].q_dtprofit, poly=3, show=False)
        fit, p8 = RegUtil.regress_y_polynomial(resample[ind:].q_opincome, poly=3, show=False)
        roe = item.q_dtprofit / item.total_hldr_eqy_exc_min_int
        pe = item.ind_total_mv*10000/item.q_dtprofit
        roe_ttm = item.q_dtprofit_ttm / item.total_hldr_eqy_exc_min_int
        pe_ttm = item.ind_total_mv*10000/item.q_dtprofit_ttm
        indicator.loc[index] = [item.trade_date,key[0],p3(8),p5(8),p6(8),p7(8),p8(8),roe,pe,roe_ttm,pe_ttm]
        #print(indicator.loc[index])
    return indicator
industry_daily = industry_daily.groupby("industry").apply(_trend).cache()
stock = pro.QA_SU_stock_info()
stock_spark = spark.createDataFrame(stock)
basic = pd.read_csv('/usr/local/spark/basic-2018.csv')
basic = spark.createDataFrame(basic)
#df = basic.join(stock_spark, basic.ts_code==stock_spark.ts_code, "inner")
df = basic.join(stock_spark,['ts_code'],"inner")
#industry_daily.count()
df = df.join(industry_daily,['industry', 'trade_date'],"inner")
new2_struct = [ 'cnt', 'mean', 'std', 'min', 'per25', 'per50', 'per75', 'per85', 'per95', 'max']
p2 = StructType()
p2.add(StructField('category', StringType()))
p2.add(StructField('industry', StringType()))
list(map(lambda x: p2.add(StructField(x, DoubleType())), new2_struct))
@pandas_udf(p2, PandasUDFType.GROUPED_MAP)
def _dailystat(key,df):
    d = df.loc[:, ['q_dtprofit_ttm_poly','q_gr_poly','q_profit_poly','q_dtprofit_poly','q_opincome_poly','industry_roe','industry_pe','roe_ttm','industry_pe_ttm']]
    st = d.describe([.25, .5, .75, .85, .95]).T.reset_index(level=0)
    col = ['category']
    col = col+new2_struct
    st.columns = col
    st.loc[:,'industry'] = key[0]
    median = d.median()
    mad = abs(d - median).median()
    d[d - (median - mad * 3 * 1.4826) < 0] = np.array((median - mad * 3 * 1.4826).tolist()*d.shape[0]).reshape((d.shape[0],d.columns.size))
    d[d - (median + mad * 3 * 1.4826) > 0] = np.array((median + mad * 3 * 1.4826).tolist()*d.shape[0]).reshape((d.shape[0],d.columns.size))

    st2 = d.describe([.25, .5, .85, .90, .95]).T.reset_index(level=0)
    st2.columns = col
    st2.loc[:,'industry'] = key[0]
    st2.category = st2.category+'_mad'
    return pd.concat([st, st2])

dailymarket = industry_daily.groupby('trade_date').apply(_dailystat).toPandas()
#
add3_struct = ['industry_roe_buy', 'industry_pe_buy', 'q_dtprofit_poly_buy', 'industry_roe_ttm_buy', 'industry_pe_ttm_buy', 'q_dtprofit_ttm_poly_buy', 'industry_roe_buy_mad', 'industry_pe_buy_mad', 'q_dtprofit_poly_buy_mad', 'industry_roe_ttm_buy_mad', 'industry_pe_ttm_buy_mad', 'q_dtprofit_ttm_poly_buy_mad']
p3 = copy.deepcopy(df.schema)
list(map(lambda x: p3.add(StructField(x, DoubleType())), add3_struct))
p3.add(StructField('key_flag', StringType()))
k = 0
d = []
ud = []
#print(p3)
#print(df.columns)
@pandas_udf(p3, PandasUDFType.GROUPED_MAP)
def _top10(key,df2):
    global dailymarket
    global k
    df = pd.concat([df2, pd.DataFrame(columns=add3_struct, dtype='float')])
    market = dailymarket[dailymarket.industry == key[0]]
    ud.append(key[0])
    #print(market)

    df.loc[:,'key_flag'] = key[0]
    if market.shape[0]:
        df.loc[:, 'industry_roe_buy'] = df.industry_roe - market[market.category == 'industry_roe'].per90[0]
        df.loc[:, 'industry_pe_buy'] = df.industry_pe - market[market.category == 'industry_pe'].per85[0]
        df.loc[:, 'q_dtprofit_poly_buy'] = df.q_dtprofit_poly - market[market.category == 'q_dtprofit_poly'].per85[0]
        df.loc[:, 'industry_roe_ttm_buy'] = df.roe_ttm - market[market.category == 'roe_ttm'].per90[0]
        df.loc[:, 'industry_pe_ttm_buy'] = df.industry_pe_ttm - market[market.category == 'industry_pe_ttm'].per85[0]
        df.loc[:, 'q_dtprofit_ttm_poly_buy'] = df.q_dtprofit_ttm_poly - market[market.category == 'q_dtprofit_ttm_poly'].per85[0]
        df.loc[:, 'industry_roe_buy_mad'] = df.industry_roe - market[market.category == 'industry_roe_mad'].per90[0]
        df.loc[:, 'industry_pe_buy_mad'] = df.industry_pe - market[market.category == 'industry_pe_mad'].per85[0]
        df.loc[:, 'q_dtprofit_poly_buy_mad'] = df.q_dtprofit_poly - market[market.category == 'q_dtprofit_poly_mad'].per85[0]
        df.loc[:, 'industry_roe_ttm_buy_mad'] = df.roe_ttm - market[market.category == 'roe_ttm_mad'].per90[0]
        df.loc[:, 'industry_pe_ttm_buy_mad'] = df.industry_pe_ttm - market[market.category == 'industry_pe_ttm_mad'].per85[0]
        df.loc[:, 'q_dtprofit_ttm_poly_buy_mad'] = df.q_dtprofit_ttm_poly - market[market.category == 'q_dtprofit_ttm_poly_mad'].per85[0]
    else:
        k = k+1
        d.append(key[0])
    return df
rs = df.groupby('trade_date').apply(_top10).toPandas().set_index(['trade_date', 'ts_code'], drop=False)
print('############rs key flag ############')
print(rs.key_flag.unique())
print('############rs total count ############')
print(len(rs))
print('############ mised key ############')
print(k)
print('############ first 5 key ############')
print(ud[0:5])
#print(rs.head)
#
# if __name__ == '__main__':
#     print('wtf')
#     # finacial = pd.read_csv('/usr/local/spark/finace-2018.csv')
#     # basic = pd.read_csv('/usr/local/spark/basic-2018.csv')
#
#     #df = spark.createDataFrame(basic.loc[:,['ts_code','trade_date']])
#     sv = simpleValued('20180101','20181231')
#     df = sv.non_finacal_top5_valued()
#     df1 = sv.industry_trend_top10(df)
#     df1.toPandas().set_index(['trade_date', 'ts_code'], drop=False)