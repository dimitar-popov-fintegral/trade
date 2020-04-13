import os
import sys
import pandas
import datetime
import time 
import yfinance as yf

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(THIS_DIR, '..'))

import dev.data as data
import dev.alpha_vantage as ap
import dev.data_util as dutil 


################################################################################
def fetch_av_data(tickers, date, redis):
    for ticker in tickers:
        print("AlphaVantage get: [%s]" %ticker)
        key = "%s_av_%s" %(date, ticker)
        print("key [%s]" %key)
        result =  ap.daily_adjusted(ticker)
        print("Saving to RedisDb as [%s]" %key)
        dutil.r_write(key, dutil.compress(result), redis)
        time.sleep(12.1)


################################################################################
def fetch_yf_data(tickers, date, redis):
    for ticker in tickers:
        print("Yahoo! get [%s]" %ticker)
        key = "%s_yf_%s" %(date, ticker)
        instrument = yf.Ticker(ticker)
        print("key [%s]" %key)
        df = instrument.history(auto_adjust=False, period="max")
        meta = instrument.info
        result = {"price": df, "meta": meta}
        dutil.r_write(key, dutil.compress(result), redis)
        time.sleep(1)


################################################################################
if __name__ == '__main__':

    import logging
    logging.basicConfig(format='%(asctime)s | %(levelname)-10s | %(processName)s | %(name)s | %(message)s')
    logger = logging.getLogger()
    logger.setLevel('INFO')
    logger.info('Controller active')
    today = datetime.datetime.now().strftime('%Y-%m-%d')

    with dutil.RedisDb() as redis:
        tickers = ["QQQ", "VEA", "QTEC", "BND", "SMH", "VWO", "EEM"]
        #fetch_av_data(tickers, today, redis)
        tickers = ["^IRX","QQQ", "VEA", "QTEC", "BND", "SMH", "VWO", "EEM"]
        fetch_yf_data(tickers, today, redis)
            
