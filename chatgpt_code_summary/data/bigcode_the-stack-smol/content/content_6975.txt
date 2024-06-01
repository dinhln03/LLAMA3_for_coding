from Transform.Transform import *

qhost = '10.0.0.10'
qport = 5100
bucket_name = 's3a://insighttmpbucket1/'
index_name = bucket_name + 'index.txt'

tickers = get_stock_list(index_name)
q_con, flint_con, spark_con = connect(qhost, qport)

#push_raw_table(q_con, spark_con, flint_con, bucket_name, tickers)
push_returns(q_con, spark_con, flint_con, bucket_name, tickers)
