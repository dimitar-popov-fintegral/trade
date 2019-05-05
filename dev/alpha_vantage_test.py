################################################################################
def test_ap_weekly_download():
    import dev.alpha_vantage as ap
    response = ap.weekly_adjusted('MSFT')
    assert response.status_code == 200,\
        'response code not equal to 200, got [{}] instead'.format(response.status_code)

