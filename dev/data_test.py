################################################################################
def test_ap_weekly_download():
    import dev.data as dt

    response = dt.ap_weekly_adjusted('MSFT')
    assert response.status_code == 200,\
        'response code not equal to 200, got [{}] instead'.format(response.status_code)


