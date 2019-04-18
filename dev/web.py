from bs4 import BeautifulSoup
import requests

################################################################################
def six_url():
    return r'https://www.six-group.com/exchanges/funds/explorer/etf/closings_en.html?Segment=funds&ProductLine=ET|PE|AE'


################################################################################
def six_header():

    return {
        'Content-Type': 'application/x-www-form-urlencoded',
        'X-Requested-With': 'XMLHttpRequest',
        'Host': 'www.six-group.com',
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:66.0) Gecko/20100101 Firefox/66.0',
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
        # 'Cookie': 'Navajo=A+/pZp8buQkJ8J9lDHTokZUkYfuktdFkfvLhsbQbcywFDQ4qmVuanveVytmba7En6mXJKbS4e4A-; tableConfig_closingsfunds=; _ga=GA1.2.1207636625.1554557063; _gid=GA1.2.693968857.1554557063',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0',
    }

    '''
    return {
        'Host': 'www.six-group.com',
        'Connection': 'keep-alive',
        'Pragma': 'no-cache',
        'Cache-Control': 'max-age=0',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/73.0.3683.75 Chrome/73.0.3683.75 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        # 'Referer': 'https://www.six-group.com/exchanges/funds/explorer/etf/closings_en.html?Segment=funds&ProductLine=ET|PE|AE&TradingState=T&sortBy=ClosingPerformance&sortOrder=0',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en',
        'Cookie': 'tableConfig_closingsfunds=; ProxySession=0c89f00a1aa2K0AAGwoaexQ7UriiaIV76EUVfeA0GSGNn52hPNqcWLAAjl; Navajo=1pAGNrx7UDTXYMvFMnq+UHuVqLJ71mhpYJ8u5ARNCHyic/8gSEyMl8rxtSgwWuulKWwyeQ6aUyQ-',
    }'''


################################################################################
def six_request(url, headers):
    '''makes a request to a SIX STOCK EXCHANGE data serving website'''
    resp = requests.get(url=url, headers=headers)
    return resp


################################################################################
def mock_table_request():
    '''example request to SIX STOCK EXCHANGE data service'''
    url = six_url()
    headers = header()
    resp = six_request(url=url, headers=headers)
    assert resp.status_code == 200
    
    soup = BeautifulSoup(resp.text, features='html.parser')
    for link in soup.find_all('td'):
        if 'class_closingfunds_column4' in str(link):
            print(link.get_text())

    return soup
