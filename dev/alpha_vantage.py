import requests
import os
import logging
import time
import multiprocessing
from enum import Enum
from typing import Callable, Tuple

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
def queue_requests(worker_function: Callable[[str], requests.models.Response], symbols: list) -> Tuple:
    """alpha vantage limits to access API:
    5-calls per minute
    500 calls per day
    """
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
def weekly_adjusted(symbol):
    """direct access to alpha vantage via requests library"""
    params = {
        'function': 'TIME_SERIES_WEEKLY_ADJUSTED',
        'apikey': API_KEYS.alpha_vantage_key.value,
        'symbol': symbol,
    }
    resp = requests.get(url=r'https://www.alphavantage.co/query', params=params)
    assert resp.status_code == 200

    data = resp.json()
    return data[ReturnKeys.MetaData.value], \
           data[ReturnKeys.WeeklyAdjusted.value]


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
def weekly_fx(from_symbol, to_symbol) -> Tuple[dict, dict]:
    """direct access to FX time-series via requests library"""
    params = {
        'function': 'FX_WEEKLY',
        'apikey': API_KEYS.alpha_vantage_key.value,
        'from_symbol': from_symbol,
        'to_symbol': to_symbol,
    }
    resp = requests.get(url=r'https://www.alphavantage.co/query', params=params)
    assert resp.status_code == 200

    data = resp.json()
    return data[ReturnKeys.MetaData.value], \
           data[ReturnKeys.FX.value]
