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


##################################################
def fetch_av_data(tickers, date, redis):
    for ticker in tickers:
        print("AlphaVantage get: [%s]" %ticker)
        key = "%s_av_%s" %(date, ticker)
        print("key [%s]" %key)

        exists = redis.keys(key)
        if exists:
            print("[%s] already up-to-date" %ticker)
            continue

        try:
            result =  ap.daily_adjusted(ticker)
        except Exception:
            print("bad ticker [%s]" %ticker)
            continue

        print("Saving to RedisDb as [%s]" %key)
        dutil.r_write(key, dutil.compress(result), redis)
        time.sleep(12.1)


##################################################
def fetch_yf_data(tickers, date, redis):
    for ticker in tickers:
        print("Yahoo! get [%s]" %ticker)
        key = "%s_yf_%s" %(date, ticker)
        instrument = yf.Ticker(ticker)
        print("key [%s]" %key)

        exists = redis.keys(key)
        if exists:
            print("[%s] already up-to-date" %ticker)
            continue

        try:
            df = instrument.history(auto_adjust=False, period="max")
            meta = instrument.info
            result = {"price": df, "meta": meta}
        except Exception:
            print("bad ticker [%s]" %ticker)
            continue

        dutil.r_write(key, dutil.compress(result), redis)
        time.sleep(1)


##################################################
if __name__ == '__main__':

    today = datetime.datetime.now().strftime('%Y-%m-%d')

    clean = False
    if len(sys.argv) > 1:
        clean = bool(sys.argv[1])

    tickers = list()
    with open(os.path.join(data.data_dir(), "etf_tickers")) as file:
        for ticker in file:
            tickers.append(str(ticker).strip())

    with dutil.RedisDb() as redis:

        if not redis.ping:
            raise IOError("redis is not running - aborting")

        if clean:
            fetch = redis.keys("%s*" %today)
            clean_keys = list(map(bytes.decode, fetch))
            if clean_keys:
                redis.delete(*clean_keys)
 
        fetch_av_data(tickers, today, redis)
        fetch_yf_data(tickers, today, redis)
            
