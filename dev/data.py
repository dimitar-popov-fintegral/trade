import os
import pandas
import numpy
import datetime
import logging
from typing import Callable

import dev.alpha_vantage as ap

################################################################################
BASE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')


################################################################################
def data_dir() -> str:
    return os.path.join(BASE_DIR, 'data')


################################################################################
def store_dir() -> str:
    return os.path.join(BASE_DIR, 'store')


################################################################################
def read_currency_list() -> pandas.DataFrame:
    """Read currency list"""
    return pandas.read_csv(os.path.join(data_dir(), 'physical_currency_list.csv'), index_col=0, header=0)


################################################################################
def read_six_etf_list() -> pandas.DataFrame:
    """Read available ETFs traded on the SIX stock exchange"""
    return pandas.read_csv(os.path.join(data_dir(), 'six_etf.csv'), header=0, index_col=0, sep='|')


################################################################################
def pull_data(tickers: dict, ts_caller: Callable[[str], dict], fx_caller: Callable[[str], dict], size: str = 'full', **kwargs) -> dict:
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
    
    time_stamp = datetime.datetime.now().strftime("%a-%d-%B-%Y_%H-%M-%S")
    for asset_class in to_store.keys():
        for ticker in to_store[asset_class].keys():
            data_entry = '/av/{time_stamp}/time_series/{ticker}'.format(time_stamp=time_stamp, ticker=ticker)
            hdf_store[data_entry] = to_store[asset_class][ticker]['data'].astype(numpy.float32)

            meta_entry = '/av/{time_stamp}/meta/{ticker}'.format(time_stamp=time_stamp, ticker=ticker)
            hdf_store[meta_entry] = to_store[asset_class][ticker]['meta']


