import os
import sys
import pandas

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(THIS_DIR, '..'))

import dev.data as data
import dev.alpha_vantage as ap


################################################################################
def etf_data():
    logger = logging.getLogger()

    ##
    logger.info('instruments defined')
    etf_meta = data.read_six_etf_list()
    etf_tickers = ['{}.SW'.format(x) for x in etf_meta.Symbol][400:800]

    instruments = {
        'stocks': list(etf_tickers),
        'bonds':{},
        'index':{},
        'fx': {
            ('EUR', 'USD'),
            ('CHF', 'USD'),
        }
    }

    ##
    logger.info('stocks, bonds, indices & fx')
    for instrument_type, instrument_list in instruments.items():
        if len(instrument_list) == 0:
            continue

        caller = ap.weekly_adjusted
        if instrument_type == 'fx':
            caller = ap.weekly_fx

        result, queue = ap.queue_requests(
            request_type=instrument_type,
            worker_function=caller,
            symbols=instrument_list
        )

        instruments.update({
            instrument_type: {
                'Result': result,
                'Queue': queue
            }})

        del result, queue

    store = pandas.HDFStore(os.path.join(data.output_dir(), 'time_series.h5'), 'w')
    ap.store_data(to_store=instruments, hdf_store=store)

    return instruments


################################################################################
if __name__ == '__main__':

    import logging
    logging.basicConfig(format='%(asctime)s | %(levelname)-10s | %(processName)s | %(name)s | %(message)s')
    logger = logging.getLogger()
    logger.setLevel('INFO')
    logger.info('Controller active')
    result = etf_data()
    print(result)
