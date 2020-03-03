import os
import sys

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(THIS_DIR, '..'))

import requests
import pandas
import datetime
import shutil 

from typing import Tuple
from enum import Enum

import dev.data as dt
import tempfile

##################################################
class DateFormat(Enum):
    YearMonthDay = "%Y-%m-%d"


##################################################
class URLS(Enum):
    WIG20 = "https://stooq.com/q/d/l/?s=wig20&i=d"
    FW20 = "https://stooq.com/q/d/l/?s=fw20&i=d"


##################################################
class Universe(Enum):
    WIG20 = "WIG20"
    FW20 = "FW20"


##################################################
def check_hash(response_content: str, current_content: str) -> bool:
    return hash(response_content) == hash(current_content)


##################################################
def pull_data(symbol: Universe, tmp: str) -> None:
    url = URLS[symbol.value]
    resp = requests.get(url.value)
    assert resp.status_code == 200, \
        "non 200 response, abort"
    resp_content = resp.content

    tmp_content_path = os.path.join(tmp, symbol.value)
    with open(tmp_content_path, 'w') as file:
        file.write(resp_content.decode("utf-8"))

    current_path = get_current_folder()
    current_content_path = os.path.join(current_path, symbol.value)
    current_symbols = os.listdir(current_path)
    if symbol.value in current_symbols:
        with open(current_content_path, 'r') as file:
            current_content = file.read()
        if check_hash(current_content, resp_content.decode("utf-8")):
            print("duplicate [%s]" %symbol.value)
    else:
        print("new entry [%s]" %symbol.value)
        shutil.copy(tmp_content_path, current_content_path)

    return None

##################################################
def get_current_folder() -> str:
    current_path = os.path.join(dt.store_dir(), "polish", "current")
    if not os.path.exists(current_path):
        os.makedirs(current_path)
    return current_path


##################################################
def store_data_folder(date: str, date_format: DateFormat) -> str:
    assert datetime.datetime.strptime(date, date_format.value),\
        "Unexpected date_format value passed"
    return os.path.join(dt.store_dir(), "polish", date)


##################################################
def main() -> int:

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        print('Created temporary directory:', tmp_dir_name)
        for symbol in Universe:
            pull_data(symbol, tmp_dir_name)

    return 0


##################################################
if __name__ == "__main__":
    assert main() == 0, "main() returned non-zero exit code"
