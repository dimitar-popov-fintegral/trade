import os
import sys
import pandas
import numpy
import json 
import logging
import datetime
import requests
import multiprocessing
import logging
import time 

from typing import Callable
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.foreignexchange import ForeignExchange
from enum import Enum


################################################################################
BASE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')


################################################################################
def data_dir() -> str:
    return os.path.join(BASE_DIR, 'data')


################################################################################
def store_dir() -> str:
    return os.path.join(BASE_DIR, 'store')


################################################################################
class API_KEYS(Enum):
    alpha_vantage_key = read_alpha_vantage_api_key()


################################################################################
def ap_queue_requests(worker_function: Callable[[str], requests.models.Response], symbols: list) -> dict:
    '''alpha vantage limits to access API: 
    5-calls per minute
    500 calls per day
    '''
    logger = logging.getLogger(__name__)
    queue = list()
    results = dict()
    sleep_time = 12.5
    for symbol in symbols:
        logger.info('requesting [{}]'.format(symbol))
        try:
            payload = worker_function(symbol)
            if 'Error Message' in payload.json().keys():
                queue.append(symbol)
                time.sleep(sleep_time)
            elif 'Note' in payload.json().keys():
                queue.append(symbol)
                time.sleep(sleep_time)
            else:
                logging.info('success for [{}]'.format(symbol))
                results.update({symbol: payload})
                time.sleep(sleep_time)                
        except AssertionError:
            logger.debug('request for [{}] did not return status code 200'.format(symbol))
            queue.append(symbol)

    return results, queue


################################################################################
def ap_parallel_requests(symbols: list):
    '''process multiple requests via parallel requests'''
    store = list()
    with multiprocessing.Pool(8) as p:
        store.append(p.map(ap_weekly_adjusted, symbols))

    return store
        

################################################################################
def ap_weekly_adjusted(symbol):
    '''direct access to alpha vantage via requests library'''
    params = {
        'function': 'TIME_SERIES_WEEKLY_ADJUSTED',
        'apikey': API_KEYS.alpha_vantage_key.value,
        'symbol': symbol,
    }
    resp = requests.get(url=r'https://www.alphavantage.co/query', params=params)
    assert resp.status_code == 200

    return resp
    

################################################################################
def read_alpha_vantage_api_key() -> str:
    """Reads the AlphaVantage API key from non-versioned file"""
    api_key_file = os.path.join(store_dir(), 'secrets', 'alpha_vantage_api_key')
    assert os.path.isfile(api_key_file),\
        'missing AlphaVantage API key file, check ./store/secrets/ folder'
    with open(api_key_file) as keyfile:
        return keyfile.read()


################################################################################
def init_alpha_vantage_ts_class(key: str = API_KEYS.alpha_vantage_key.value) -> TimeSeries:
    """simple function to get a parametrized ALPHA VANTAGE price source class"""
    return TimeSeries(key=key)      


################################################################################
def init_alpha_vantage_fx_class(key: str = API_KEYS.alpha_vantage_key.value) -> ForeignExchange:
    """simple function to get a parametrized ALPHA VANTAGE FX source class"""
    return ForeignExchange(key=key)


################################################################################
def get_weekly_adjusted_data(symbol: str, object: TimeSeries = init_alpha_vantage_ts_class()):
    return object.get_weekly_adjusted(symbol=symbol)


################################################################################
def get_daily_adjusted_data(symbol: str, object: TimeSeries = init_alpha_vantage_ts_class()):
    return object.get_daily_adjusted(symbol=symbol)


################################################################################
def pull_data(tickers: dict, ts_caller: TimeSeries, fx_caller: ForeignExchange, size: str='full', **kwargs) -> dict:
    """Basic interface to create a data-set from online source"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    data_dict = dict()

    """Stocks"""
    out = {}
    element = 'stocks'
    logger.info('Downloading {}'.format(element))
    assert tickers.get(element, None) is not None,\
        'missing tickers entry <stocks>, aborting'
    for ticker in tickers[element]:
        logger.info('Ticker {}'.format(ticker))
        raw_data, raw_info = ts_caller(symbol=ticker)
        df = pandas.DataFrame(raw_data).T
        info = pandas.Series(raw_info)
        info['class'] = element
        out[ticker] = {}
        out[ticker]['data'] = df
        out[ticker]['meta'] = info
    data_dict[element] = out
    del element, out

    """Index"""
    out = {}
    element = 'index'
    logger.info('Downloading {}'.format(element))
    assert tickers.get(element, None) is not None,\
        'missing tickers entry <index>, aborting'
    for ticker in tickers[element]:
        logger.info('Ticker {}'.format(ticker))
        raw_data, raw_info = ts_caller(symbol=ticker)
        df = pandas.DataFrame(raw_data).T
        info = pandas.Series(raw_info)
        info['class'] = element
        out[ticker] = {}
        out[ticker]['data'] = df
        out[ticker]['meta'] = info
    data_dict[element] = out
    del element, out

    """Bonds"""    
    out = {}
    element = 'bonds'
    logger.info('Downloading {}'.format(element))
    assert tickers.get(element, None) is not None,\
        'missing tickers entry <bonds>, aborting'
    for ticker in tickers[element]:
        logger.info('Ticker {}'.format(ticker))
        raw_data, raw_info = ts_caller(symbol=ticker)
        df = pandas.DataFrame(raw_data).T
        info = pandas.Series(raw_info)
        info['class'] = element
        out[ticker] = {}
        out[ticker]['data'] = df
        out[ticker]['meta'] = info
    data_dict[element] = out
    del element, out

    """FX"""    
    out = {}
    element = 'fx'
    logger.info('Downloading {}'.format(element))
    assert tickers.get(element, None) is not None,\
        'missing tickers entry <fx>, aborting'
    for base, counter in tickers[element]:
        """applies the following concept for dealing with FX
        direct quotation: price of 1-unit foreign currency in terms of x-units
                          of domestic currency
                          i.e. 1 EUR = x USD
        nomenclature: base = foreign currency
                      counter = domestic currency
        """
        logger.info('Pair {}'.format(base, counter))
        raw_data, raw_info = fx_caller(from_symbol=counter, to_symbol=base)
        df = pandas.DataFrame(raw_data).T
        info = pandas.Series(raw_info)
        info['class'] = 'fx'
        pair = '{counter}/{base}'.format(counter=counter, base=base)
        out[pair] = {}
        out[pair]['data'] = df
        out[pair]['meta'] = info
    data_dict[element] = out
    del element, out

    return data_dict


################################################################################        
def store_data(to_store: dict, hdf_store: pandas.HDFStore) -> str:
    """Stores data-set by appending to .hdf file, returns the output file-path"""
    logger = logging.getLogger(__name__)
    
    ##    
    time_stamp = datetime.datetime.now().strftime("%a-%d-%B-%Y_%H-%M-%S")
    for asset_class in to_store.keys():
        for ticker in to_store[asset_class].keys():
            data_entry = '/av/{time_stamp}/time_series/{ticker}'.format(time_stamp=time_stamp, ticker=ticker)
            hdf_store[data_entry] = to_store[asset_class][ticker]['data'].astype(numpy.float32)

            meta_entry = '/av/{time_stamp}/meta/{ticker}'.format(time_stamp=time_stamp, ticker=ticker)
            hdf_store[meta_entry] = to_store[asset_class][ticker]['meta']


################################################################################        
def read_currency_list() -> pandas.DataFrame:
    """Read currency list"""
    return pandas.read_csv(os.path.join(data_dir, 'physical_currency_list.csv'), index_col=0, header=0)


################################################################################        
def read_six_etf_list() -> pandas.DataFrame:
    """Read available ETFs traded on the SIX stock exchange"""
    return pandas.read_csv(os.path.join(data_dir(), 'six_etf.csv'), header=0, index_col=0, sep='|')
