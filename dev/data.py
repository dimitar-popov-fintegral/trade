import os
import pandas
import numpy 


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
    return pandas.read_csv(os.path.join(data_dir(), 'six_etf.csv'), header=0, index_col=0, sep=',', encoding='utf-8')


################################################################################
def log_returns(x):
    return numpy.log(x[1:]) - numpy.log(x[:-1])
    
    
