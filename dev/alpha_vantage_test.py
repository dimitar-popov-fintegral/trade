################################################################################
def test_ap_weekly_download():
    import dev.alpha_vantage as av
    response_meta, response_data = av.weekly_adjusted('ZGLD.SW')


################################################################################
def test_scratch():
    import dev.alpha_vantage as av
    import dev.data as dt
    import os
    import pandas
    path = os.path.join(dt.output_dir(), 'time_series.h5.tmp')
    store = pandas.HDFStore(path, 'r')
    ts = 'Sat-11-May-2019_15-50-40'
    time_series, meta = av.read_raw_data(store, time_stamp=ts, time_series_element='4. close')
    a=1