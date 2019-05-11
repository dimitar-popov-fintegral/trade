################################################################################
def test_mock_table_request():
    from bs4 import BeautifulSoup
    from dev.web import six_url, six_params, six_header, six_request
    '''example request to SIX STOCK EXCHANGE data service'''
    url = six_url()
    headers = six_header()
    parameters = six_params()
    resp = six_request(url=url, headers=headers, parameters=parameters)
    assert resp.status_code == 200

    soup = BeautifulSoup(resp.text, features='html.parser')
    for link in soup.find_all('td'):
        if 'class_closingsfunds_column2' in str(link):
            print(link.get_text())

    print(soup)


################################################################################
def test_headless_table_request():
    from dev.web import six_url, headless_table_request
    import json

    res = headless_table_request(url=six_url())

    with open('six_etf_list_2.txt', 'w') as file:
        file.write(json.dumps(res))

    [print(x) for iteration, x in res.items()]
