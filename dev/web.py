import os
import requests
import time
import numpy
import json
from selenium import webdriver
from selenium.common import exceptions
from bs4 import BeautifulSoup

import dev.data as dt


################################################################################
def six_url():
    return r'https://www.six-group.com/exchanges/funds/explorer/etf/closings_en.html'


################################################################################
def six_header():

    return {
        'Host': 'www.six-group.com',
        'Connection': 'keep-alive',
        'X-Requested-With': 'XMLHttpRequest',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': '*/*',
        'Upgrade-Insecure-Requests': '1',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en',
    }


################################################################################
def six_params():
    return {
        'Segment': 'funds',
        'ProductLine': 'ET%7CPE%7CAE',
        'dojo.preventCach': '1557077066938'
    }

################################################################################
def six_request(url, headers, parameters):
    '''makes a request to a SIX STOCK EXCHANGE data serving website'''
    resp = requests.get(url=url, headers=headers, params=parameters)
    return resp


################################################################################
def path_to_chrome_driver():
    return os.path.join(dt.BASE_DIR, '..', 'develop', 'web', 'chromedriver')


################################################################################
def headless_table_request(url: str):

    gather = dict()

    driver = webdriver.Chrome(executable_path=path_to_chrome_driver())
    driver.get(url=url)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);var lenOfPage=document.body.scrollHeight;return lenOfPage;")
    time.sleep(3)

    number_of_results_xpath = '''
    //*[@id="closingsfunds"]/table[2]/tbody/tr/td[18]    
    '''
    number_of_results = driver.find_element_by_xpath(number_of_results_xpath)
    elements = number_of_results.text.split(' ')
    assert int(elements[0]) == 1,\
        'ChromeDriver appears not to have started with the first result'
    per_page = int(elements[2])
    total = int(elements[4])
    button_clicks = total // per_page

    for i in range(button_clicks):

        try:
            nav_buttons_xpath = '''
            //*[@id="closingsfunds"]/table[2]/tbody/tr/td/button    
            '''
            nav_buttons = driver.find_elements_by_xpath(nav_buttons_xpath)
            for button in nav_buttons:
                if button.get_property('name') == 'chunking-next':
                    button.click()
                    time.sleep(1+numpy.random.rand())
                    break
        except exceptions.NoSuchElementException as err:
            print('found no such element on the page [{}]'.format(err))

        proposed_xpath = '''
        //*[@id="closingsfunds"]/tbody/tr 
        '''
        results = driver.find_elements_by_xpath(proposed_xpath)
        print('Number of results', len(results))
        gather.update({i: [x.text for x in results]})
        del results

    return gather
