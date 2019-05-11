import requests
import os
import logging
import time
import multiprocessing
import pandas

from enum import Enum
from typing import Callable, Tuple, List

import dev.data as dt


################################################################################
class ReturnKeys(Enum):
    MetaData = "Meta Data"
    Daily = "Time Series (Daily)"
    DailyAdjusted = "Time Series (Daily)"
    Weekly = "Weekly Time Series"
    WeeklyAdjusted = "Weekly Adjusted Time Series"
    FX = "Time Series FX (Weekly)"


################################################################################
def read_alpha_vantage_api_key() -> str:
    """Reads the AlphaVantage API key from non-versioned file"""
    api_key_file = os.path.join(dt.store_dir(), 'secrets', 'alpha_vantage_api_key')
    assert os.path.isfile(api_key_file), \
        'missing AlphaVantage API key file, check ./store/secrets/ folder'
    with open(api_key_file) as keyfile:
        return keyfile.read()


################################################################################
class API_KEYS(Enum):
    alpha_vantage_key = read_alpha_vantage_api_key()


################################################################################
class RequestType(Enum):
    stocks = 'stocks'
    bonds = 'bonds'
    index = 'index'
    fx = 'fx'


################################################################################
def queue_requests(request_type: str,
                   worker_function: Callable,
                   symbols: list) \
        -> Tuple:
    """
    Alpha vantage limits to access API:
    - 5 calls per minute
    - 500 calls per day
    """
    logger = logging.getLogger(__name__)
    queue = list()
    results = dict()
    sleep_time = 12.1
    for symbol in symbols:
        logger.info('requesting [{}]'.format(symbol))
        try:
            meta, data = worker_function(symbol)
            logging.info('success for [{}]'.format(symbol))

            df = pandas.DataFrame(data).T
            info = pandas.Series(meta)
            info['class'] = request_type
            results[symbol] = {
                'data': df,
                'meta': info
            }
            time.sleep(sleep_time)

        except AssertionError as err:
            queue.append(symbol)
            logger.error('request for [{}] failed, got error -> [{}]'.format(symbol, err))

    return results, queue


################################################################################
def parallel_requests(symbols: list) -> list:
    """process multiple requests via parallel requests"""
    store = list()
    with multiprocessing.Pool(8) as p:
        store.append(p.map(weekly_adjusted, symbols))

    return store


################################################################################
def daily_adjusted(symbol: str):
    """direct access to alpha vantage via requests library"""
    params = {
        'function': 'TIME_SERIES_DAILY_ADJUSTED',
        'outputsize': 'full',
        'apikey': API_KEYS.alpha_vantage_key.value,
        'symbol': symbol,
    }
    resp = requests.get(url=r'https://www.alphavantage.co/query', params=params)
    assert resp.status_code == 200

    data = resp.json()
    return data[ReturnKeys.MetaData.value], \
           data[ReturnKeys.DailyAdjusted.value]


################################################################################
def daily(symbol):
    """direct access to alpha vantage via requests library"""
    params = {
        'function': 'TIME_SERIES_DAILY',
        'outputsize': 'full',
        'apikey': API_KEYS.alpha_vantage_key.value,
        'symbol': symbol,
    }
    resp = requests.get(url=r'https://www.alphavantage.co/query', params=params)
    assert resp.status_code == 200

    data = resp.json()
    return data[ReturnKeys.MetaData.value], \
           data[ReturnKeys.Daily.value]


################################################################################
def weekly_adjusted(symbol: str) -> List[dict]:
    """Direct access to alpha vantage via requests library"""
    params = {
        'symbol': '{}'.format(symbol),
        'function': 'TIME_SERIES_WEEKLY_ADJUSTED',
        'apikey': API_KEYS.alpha_vantage_key.value,
    }

    return_keys = [
        ReturnKeys.MetaData.value,
        ReturnKeys.WeeklyAdjusted.value
    ]

    resp = requests.get(url=r'https://www.alphavantage.co/query', params=params)
    assert resp.status_code == 200,\
        'response did not return 200, got [{}] instead'.format(resp.status_code)

    data = resp.json()
    assert all(name in data.keys() for name in return_keys),\
        'response did not contain required return_keys [{}]'.format(return_keys)

    return [data[key] for key in return_keys]


################################################################################
def weekly(symbol: str) -> Tuple:
    """direct access to alpha vantage via requests library"""
    params = {
        'function': 'TIME_SERIES_WEEKLY',
        'apikey': API_KEYS.alpha_vantage_key.value,
        'symbol': symbol,
    }
    resp = requests.get(url=r'https://www.alphavantage.co/query', params=params)
    assert resp.status_code == 200

    data = resp.json()
    return data[ReturnKeys.MetaData.value], \
           data[ReturnKeys.Weekly.value]


################################################################################
def weekly_fx(symbol: Tuple[str, str]) -> List[dict]:
    """direct access to FX time-series via requests library"""
    from_symbol, to_symbol = symbol
    params = {
        'function': 'FX_WEEKLY',
        'apikey': API_KEYS.alpha_vantage_key.value,
        'from_symbol': from_symbol,
        'to_symbol': to_symbol,
    }

    return_keys = [
        ReturnKeys.MetaData.value,
        ReturnKeys.FX.value
    ]

    resp = requests.get(url=r'https://www.alphavantage.co/query', params=params)
    assert resp.status_code == 200,\
        'response did not return 200, got [{}] instead'.format(resp.status_code)

    data = resp.json()
    assert all(name in data.keys() for name in return_keys),\
        'response did not contain required return_keys [{}]'.format(return_keys)

    return [data[key] for key in return_keys]
