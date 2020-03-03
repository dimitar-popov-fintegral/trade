import os
import sys
import pandas
import datetime

from typing import Tuple

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(THIS_DIR, '..'))

import dev.data as dt
import dev.alpha_vantage as ap


##################################################
def index_data() -> Tuple:
    """get data for ukranian futures"""
    ts, meta = ap.daily("UX Index")
    return ts, meta
    


##################################################
if __name__ == "__main__":
    import logging
    logging.basicConfig(format='%(asctime)s | %(levelname)-10s | %(processName)s | %(name)s | %(message)s')
    logger = logging.getLogger()
    logger.setLevel('INFO')
    logger.info('Controller active')
    print(index_data())
    
