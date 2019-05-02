import os
import sys
import logging 
import pandas

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(THIS_DIR, '..'))

import dev.data as data

################################################################################
def main():
    logger = logging.getLogger(__name__)

    logger.info('instruments defined')
    etf_meta = data.read_six_etf_list()
    etf_tickers = ['{}.SW'.format(x) for x in etf_meta.Symbol][:10]
    instruments = {
        'stocks': {*etf_tickers},
        'bonds':{},
        'index':{},
        'fx':{
            ('EUR', 'USD'),
            ('CHF', 'USD'),
        }
    }
    result = data.ap_queue_requests(worker_function=data.ap_weekly_adjusted, symbols=instruments['stocks'])
    '''
    ##
    base_ts_object = data.init_alpha_vantage_ts_class()
    base_fx_object = data.init_alpha_vantage_fx_class()

    def ts(symbol):
        def inner(symbol, size='full'):
            return base_ts_object.get_daily_adjusted(symbol, outputsize=size)
        return inner(symbol=symbol)

    def fx(from_symbol, to_symbol):
        def inner(from_symbol=from_symbol, to_symbol=to_symbol, size='compact'):
            return base_fx_object.get_currency_exchange_daily(from_symbol=from_symbol, to_symbol=to_symbol, outputsize=size)
        return inner(from_symbol=from_symbol, to_symbol=to_symbol)

    with pandas.HDFStore(os.path.join(data.data_dir(), 'time_series.hdf')) as store:
        to_store = data.pull_data(tickers=instruments, ts_caller=ts, fx_caller=fx, size='compact')
        path = data.store_data(to_store=to_store, hdf_store=store)
    '''

    return result

################################################################################
if __name__ == '__main__':

    import logging

    # Create a custom logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create handlers
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)

    logger.info('Controller active')
    result, queue = main()
    print(result)
