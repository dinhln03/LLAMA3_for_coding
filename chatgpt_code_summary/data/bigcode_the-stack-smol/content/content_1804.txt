# coding: utf-8

from pytdx.hq import TdxHq_API
from pytdx.params import TDXParams
import pandas as pd
import numpy as np
import re
import csv
import io
import time
import traceback


if __name__ == '__main__':
    with io.open(r'..\all_other_data\symbol.txt', 'r', encoding='utf-8') as f:
        symbol = [s.strip() for s in f.readlines()]

    TDXHQ = TdxHq_API(raise_exception=True, auto_retry=True)
    if not TDXHQ.connect('121.14.110.200', 443):
        raise Exception("Can't connect.")

    #symbol = symbol[0:5]
    first_df = True

    for code in symbol:
        if code[0:2] == 'SH':
            market = 1
        else:
            market = 0
        code = code [2:]
        #quote_info = TDXHQ.get_security_quotes([(market, code)])
        quote_info = TDXHQ.get_security_bars(9, market, code, 0, 1)
        try:
            if first_df:
                columns =  ['code', 'price']
                quote_df = pd.DataFrame(columns=columns)
                first_df = False
            values = [code, quote_info[0]['close']]
            quote_df.loc[quote_df.shape[0]] = values
        except Exception as e:
            print "code {}, process bars error, skipped.".format(code)
            print e.message
            print quote_info

        quote_df = quote_df.rename(columns={
        'code':'代码',
        'price':'价格',
    })

    # string_columns = ['代码']
    # quote_df[string_columns] = quote_df[string_columns].applymap(
    #     lambda x: '=""' if type(x) is float else '="' + str(x) + '"')
    quote_df.to_csv(r"..\all_other_data\all_last_price.csv", encoding="gbk", quoting=csv.QUOTE_NONE, index=False)

    TDXHQ.disconnect()


