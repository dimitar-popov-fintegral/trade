import os
import pandas
import numpy
import datetime
import logging
from typing import Callable, List


################################################################################
BASE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')


################################################################################
def output_dir() -> str:
    path = os.path.join(BASE_DIR, 'output')
    if not os.path.exists(path):
        os.makedirs(path)
    return path


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
def store_data(to_store: dict, hdf_store: pandas.HDFStore) -> str:
    """Stores data-set by appending to .hdf file, returns the output file-path"""
    logger = logging.getLogger(__name__)
    
    time_stamp = datetime.datetime.now().strftime("%a-%d-%B-%Y_%H-%M-%S")
    for asset_class in to_store.keys():
        if len(to_store[asset_class]) == 0:
            continue
        for ticker in to_store[asset_class]['Result'].keys():
            data_entry = '/av/{time_stamp}/time_series/{ticker}'.format(time_stamp=time_stamp, ticker=ticker)
            hdf_store[data_entry] = to_store[asset_class]['Result'][ticker]['data'].astype(numpy.float32)

            meta_entry = '/av/{time_stamp}/meta/{ticker}'.format(time_stamp=time_stamp, ticker=ticker)
            hdf_store[meta_entry] = to_store[asset_class]['Result'][ticker]['meta']

    return 'in progress'
