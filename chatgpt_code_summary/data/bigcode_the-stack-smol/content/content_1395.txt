
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A python project for algorithmic trading in FXCM                                           -- #
# -- --------------------------------------------------------------------------------------------------- -- #
# -- script: requirements.txt : text file with the required libraries for the project                    -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: MIT License                                                                                -- #
# -- --------------------------------------------------------------------------------------------------- -- #
# -- Template repository: https://github.com/IFFranciscoME/trading-project                               -- #
# -- --------------------------------------------------------------------------------------------------- -- #

# -- Packages for the script
import fxcmpy
import pandas as pd

# -- --------------------------------------------------------------------------------------------------- -- #
# -- --------------------------------------------------------------------------------------------------- -- #

api_token = "ba432..." # This token is obtained in the fxcm trading station platform
con = fxcmpy.fxcmpy(access_token=api_token, server='demo', log_level='error', log_file='fxcm_logs.txt')

# -- --------------------------------------------------------------------------------------------------- -- #
# -- --------------------------------------------------------------------------------------------------- -- #

def fxcm_ohlc(p_instrument, p_period, p_ini, p_end):
    """
    to download OHLC prices from FXCM broker

    Parameters
    ----------
    
    p_instrument: str
        The name of the instrument according to fxcmpy

    p_freq: str
        The frequency or granularity of prices, according to fxcmpy

    p_ini: str
        Initial timestamp, in format "yyyy-mm-dd hh:mm:ss"

    p_end: str
        final timestamp, in format "yyyy-mm-dd hh:mm:ss"

    Returns
    -------

    data_ohlc: DataFrame
        with columns Open, High, Low, Close and Timestamp as index

    """

    data_ohlc = con.get_candles(instrument=p_instrument, period=p_period,
                                start=p_ini, end=p_end)

    data_ohlc['open'] = (data_ohlc['bidopen'] + data_ohlc['askopen'])*0.5
    data_ohlc['high'] = (data_ohlc['bidhigh'] + data_ohlc['askhigh'])*0.5
    data_ohlc['low'] = (data_ohlc['bidlow'] + data_ohlc['asklow'])*0.5
    data_ohlc['close'] = (data_ohlc['bidclose'] + data_ohlc['askclose'])*0.5
    data_ohlc = data_ohlc[['open', 'high', 'low', 'close']]
    data_ohlc.index.name = 'timestamp'

    return data_ohlc
