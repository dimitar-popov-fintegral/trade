################################################################################
def test_ap_weekly_download():
    import dev.alpha_vantage as ap
    response_meta, response_data = ap.weekly_adjusted('ZGLD.SW')
