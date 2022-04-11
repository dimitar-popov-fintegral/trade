import os
import sys

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(THIS_DIR, '..'))

import time
import datetime
import matplotlib.pyplot as plt 
import numpy
import calendar
import copy 
import pandas

from scipy.optimize import minimize
from enum import Enum
from itertools import compress 
import matplotlib.dates as mdates

import dev.data_util as du
import dev.data as dt
import dev.alpha_vantage as ap
    

##################################################
class DateTimeConvention(Enum):
    YearMonthDay = "%Y-%m-%d"


##################################################
class UniversalKeys(Enum):
    OPEN = 'OPEN'
    HIGH = 'HIGH'
    LOW = 'LOW'
    CLOSE = 'CLOSE'
    ADJ_CLOSE = 'ADJCLOSE'
    VOLUME = 'VOLUME'
    DIVIDENDS = 'DIVIDENDS'
    STOCK_SPLITS = 'STOCKSPLIT'


##################################################
class YFKeys(Enum):
    OPEN = 'Open'
    HIGH = 'High'
    LOW = 'Low'
    CLOSE = 'Close'
    ADJ_CLOSE = 'Adj Close'
    VOLUME = 'Volume'
    DIVIDENDS = 'Dividends'
    STOCK_SPLITS = 'Stock Splits'


##################################################
class AVKeys(Enum):
    OPEN = '1. open'
    HIGH = '2. high'
    LOW = '3. low'
    CLOSE = '4. close'
    ADJ_CLOSE = '5. adjusted close',
    VOLUME = '6. volume'
    DIVIDENDS = '7. dividend amount'
    STOCK_SPLITS = '8. split coefficient'



##################################################
def switch_keys(old_keys, new_keys):
    return {old_keys[k.name].value: new_keys[k.name].value for k in old_keys}


##################################################
def standardize_frame(frame, provider_keys):

    # COLUMNS
    columns = [k.value for k in provider_keys]
    check = [k not in frame.columns for k in columns]
    missing = list(compress(columns, check))
    assert len(missing) == 0, "missing columns [%s]" %" ".join(missing)
    frame = frame.rename(columns=switch_keys(provider_keys, UniversalKeys))

    # ROWS
    frame.index = [numpy.datetime64(i) for i in frame.index]

    return frame.sort_index()


##################################################
def access_yf_time_series(date: datetime.datetime, tickers: list):
    date_str = date.strftime(DateTimeConvention.YearMonthDay.value)
    keys = ["%s_yf_%s" %(date_str, ticker) for ticker in tickers]
    frames = list()
    with du.RedisDb() as redis:
        for key in keys:
            start = time.time()
            data_dict = ap.du.decompress(du.r_read(key, redis))            
            df = standardize_frame(data_dict["price"], YFKeys)
            meta = data_dict["meta"]
            ticker = meta.get("symbol")
            if ticker is not None:
                assert ticker in key, "mis-match key != ticker"
            frames.append(df)
            print("appending: ", ticker)        
            print("that took: [%s]s to load [%s] rows" \
                  %(time.time() - start, len(df)))
            
    return frames


##################################################
def access_av_time_series(date: datetime.datetime, tickers: list):

    keys = ["%s_av_%s" %(date_str, ticker) for ticker in tickers]
    frames = list()
    with du.RedisDb() as redis:
        for key in keys:
            start = time.time()
            response = ap.du.decompress(du.r_read(key, redis))
            meta, df = ap.parse_response(response)
            df = standardize_frame(df.astype(float), AVKeys)
            ticker = meta["2. Symbol"]
            assert ticker in key, "mis-match key != ticker"
            frames.append(df)
            print("appending: ", ticker)        
            print("that took: [%s]s to load [%s] rows" \
                  %(time.time() - start, len(df)))

    return frames

    
##################################################
def time_series(date: str, tickers: list):

    search_for = ["%s_%s" %(date, ticker) for ticker in tickers]
    frames = list()
    with du.RedisDb() as redis:
        all_keys = redis.keys()
        instrument_keys = list()
        for key in all_keys:
            key = key.decode("utf-8")
            if key in search_for:

                response = ap.du.decompress(du.r_read(key, redis))
                meta, df = ap.parse_response(response)
                ticker = meta["2. Symbol"]

                tickers.append(ticker)
                frames.append(df.astype(float).sort_index())
            else:
                continue

    return tickers, frames


##################################################
def load_and_plot():
    as_of_date = datetime.datetime(2020, 4, 9)
    tickers = ["QQQ", "BND"]
    tickers, frames = time_series(as_of_date.strftime(DateTimeConvention.YearMonthDay.value), tickers)

    for ticker, df in zip(tickers, frames):
        fig, ax = plt.subplots(3,1)
        
        axis = ax[0]
        df["1. open"].plot(ax=axis, title=ticker, legend=True)
        df["2. high"].plot(ax=axis, legend=True)
        df["3. low"].plot(ax=axis, legend=True)
        df["4. close"].plot(ax=axis, legend=True)
        df["5. adjusted close"].plot(ax=axis, legend=True)

        axis = ax[1]
        df["6. volume"].plot(ax=axis, legend=True)

        axis = ax[2]
        df["7. dividend amount"].plot(ax=axis, legend=True)
        df["8. split coefficient"].plot(ax=axis, legend=True)

        plt.show()


##################################################
def equally_weighted_risk_contributions(covariance: numpy.array):
    """computes the solution to ERC problem"""
    n = len(covariance)
    diag = numpy.diag(covariance)
    init_w = diag / numpy.sum(diag)

    def obj(w, *args):
        return numpy.sqrt(numpy.einsum("i,ij,j", w, covariance, w))

    def portfolio_diversification_constraint(w):
        return numpy.einsum("i,i", numpy.ones(n), numpy.log(w))

    bounds = [(0, None) for i in range(n)]
    constraints = [{"type": "ineq", "fun": portfolio_diversification_constraint}]

    res = minimize(obj, init_w, args=covariance, method="SLSQP",
                                  constraints=constraints, bounds=bounds)
    norm = numpy.einsum("i,ij,j", res.x, covariance, res.x)
    budget = numpy.einsum("i,ij,j->i", res.x, covariance, res.x) / norm
    #numpy.testing.assert_allclose(budget, numpy.ones(n) * (1 / n), rtol=1e-2)

    return res.x / numpy.sum(res.x)
   

##################################################
def test_erc_example():

    vol = numpy.array([.1, .2, .3, .4])
    corr = numpy.array([[1.00, 0.80, 0.00, 0.00],
                        [0.80, 1.00, 0.00, 0.00],
                        [0.00, 0.00, 1.00, -0.5],
                        [0.00, 0.00, -0.5, 1.00]])
    cov = corr * numpy.einsum("i,j", vol, vol)
    w = equally_weighted_risk_contributions(cov)
    rhs = numpy.array([0.384, 0.192, 0.243, 0.182])


##################################################
def regularize_cov(cov, ridge=0.1):
    n = len(cov)
    avg_vars = numpy.average(numpy.diag(cov))
    return cov / avg_vars + ridge * numpy.identity(n)


##################################################
def cov_to_corr(cov):
    return numpy.einsum("i,ij,i", numpy.diag(cov), cov, numpy.diag(cov))


##################################################
class Container(object):

    def __init__(self, tickers, prices, dates):
        """Create an object to store strategy data and parameters"""
        self.time_dim, self.asset_dim = prices.shape
        assert self.asset_dim == len(tickers)
        assert self.time_dim == len(dates)

        self.tickers = tickers
        self.prices = prices
        self.returns = numpy.vstack([numpy.array(self.asset_dim * [numpy.nan]), self.log_returns(prices)])
        self.dates = dates

        self.nav = numpy.zeros(self.time_dim)
        self.nav[0] = 1.0
        self.strat_returns = numpy.zeros(self.time_dim)


    @staticmethod
    def log_returns(prices):
        return numpy.log(prices[1:]) - numpy.log(prices[:-1])        


##################################################
def __common_dates(price_frames):
    dates = list()
    for df in price_frames:
        dates.append(set(df.index.sort_values().values))

    return set.intersection(*dates)

##################################################
def __closing_prices(price_frames, common_dates, fillna=False):
    instrument_closing = list()
    for df in price_frames:
        x = df.loc[common_dates, UniversalKeys.ADJ_CLOSE.value].sort_index()
        if fillna:
            x = x.fillna(method="ffill").fillna(method="backfill")
        instrument_closing.append(x.values)

    return numpy.vstack(instrument_closing).T


##################################################
def prepare_container(tickers: list, as_of_date: datetime.datetime, data_access_function, fillna=False):
    """Prepares a Container object for strategy testing"""
    price_frames = data_access_function(as_of_date, tickers)
    dates =  numpy.sort(numpy.array(list(__common_dates(price_frames))))
    prices = __closing_prices(price_frames, dates, fillna)

    return Container(tickers, prices, dates)


##################################################
def long_only_equal_risk_contributions_signal(burn_in: int, rebalance: int, container: Container):
    assert isinstance(container, Container)
    NAV = 1.0
    for t in range(container.time_dim):

        if t < burn_in:
            continue

        if t % rebalance == 0:
            signal = ema_cross_signal(container.prices[:t, 0], 10, 110)            
            cov = numpy.cov(container.returns[1:t].T)
            rcov = regularize_cov(cov)
            corr = cov_to_corr(rcov)
            budget = equally_weighted_risk_contributions(rcov)
            if signal > 0:
                budget = (1 / budget) / numpy.sum(1 / budget)
            display = ", ".join([str((i,j)) for i,j in zip(tickers, budget)])
            print("total budget: [%s]" %budget.sum())
            print("resulting budgets: [%s]" %display)
            print("weights: [%s]" %budget)
        container.strat_returns[t] = numpy.einsum("i,i", budget, numpy.exp(container.returns[t]))
        NAV *= container.strat_returns[t]
        container.nav[t] = NAV
        #signals[t] = signal
        print("[%s] NAV = [%3.5f]" %(container.dates[t], NAV))

    return container 


##################################################
def long_only_equal_risk_contributions(burn_in: int, rebalance: int, container: Container):
    assert isinstance(container, Container)
    NAV = 1.0
    for t in range(container.time_dim):

        if t < burn_in:
            continue

        if t % rebalance == 0:
            cov = numpy.cov(container.returns[1:t].T)
            rcov = regularize_cov(cov)
            corr = cov_to_corr(rcov)
            budget = equally_weighted_risk_contributions(rcov)
            display = ", ".join([str((i,j)) for i,j in zip(tickers, budget)])
            print("total budget: [%s]" %budget.sum())
            print("resulting budgets: [%s]" %display)
            print("weights: [%s]" %budget)
        container.strat_returns[t] = numpy.einsum("i,i", budget, numpy.exp(container.returns[t]))
        NAV *= container.strat_returns[t]
        container.nav[t] = NAV
        print("[%s] NAV = [%3.5f]" %(container.dates[t], NAV))

    return container 


##################################################
def back_test(tickers, benchmarks):
    BURN_IN = 120
    REBALANCE = 28
    
    as_of_date = datetime.datetime(2020, 4, 9)
    frames = access_av_time_series(as_of_date.strftime(DateTimeConvention.YearMonthDay.value), tickers)

    dates = list()
    for df in frames:
        dates.append(set(df.index.sort_values().values))
    common_dates = set.intersection(*dates)
    del dates
    
    instrument_closing = list()
    for df in frames:
        x = df.loc[common_dates, "5. adjusted close"].sort_index().values
        instrument_closing.append(x)

    dates = df.loc[common_dates,:].sort_index().index.values[1:]
    dates = numpy.array([numpy.datetime64(i) for i in dates])
    prices = numpy.vstack(instrument_closing).T
    returns = dt.log_returns(prices)
    n = len(returns)
    t = len(tickers)
    budget = numpy.zeros(t)
    ptf_performance = numpy.zeros(n)
    ptf_returns = numpy.zeros(n)
    NAV = 1.0
    signals = numpy.zeros(n)

    """Loop over time periods"""
    for i in range(n):
        print("iter %d" %i)
        if i < BURN_IN:
            continue

        if i % REBALANCE == 0:
            signal = ema_cross_signal(prices[:i, 0], 10, 110)            
            cov = numpy.cov(returns[:i].T)
            rcov = regularize_cov(cov)
            corr = cov_to_corr(cov)
            budget = equally_weighted_risk_contributions(rcov)
            if signal > 0:
                budget = (1 / budget) / numpy.sum(1 / budget)
            display = ", ".join([str((i,j)) for i,j in zip(tickers, budget)])
            print("total budget: [%s]" %budget.sum())
            print("resulting budgets: [%s]" %display)
            print("weights: [%s]" %budget)
        ptf_returns[i] = numpy.einsum("i,i", budget, numpy.exp(returns[i]))
        NAV *= ptf_returns[i]
        ptf_performance[i] = NAV
        signals[i] = signal
        print("NAV = [%f]" %NAV)

    """Plot results"""
    fig, ax = plot_series(ptf_performance[BURN_IN:], dates[BURN_IN:], "Ptf", subplots=2)
    asset = 0
    ax[0].plot(dates[BURN_IN:],
            numpy.cumprod(numpy.exp(returns[:,asset]))[BURN_IN:],
            label=tickers[asset])
    asset = 1
    ax[0].plot(dates[BURN_IN:],
            numpy.cumprod(numpy.exp(returns[:,asset]))[BURN_IN:],
            label=tickers[asset])
    ax[0].legend()

    ax[1].plot(dates[BURN_IN:], signals, label="signal")

    plt.show()


##################################################
def ema_cross_signal(x, lag0, lag1):
    """generates +1 for buy, -1 for sell"""
    assert lag0 < lag1, "lag0 GREATER THAN lag1 - signal accepts lag0 < lag1 only"
    signal = 1
    d0, d1 = (lag_to_decay(l) for l in [lag0, lag1])
    slow = compute_ema(x, lag0)
    fast = compute_ema(x, lag1)

    if slow[-1] < fast[-1]:
        signal = -1

    return signal


##################################################
def lag_to_decay(lag):
    return 1 / (lag + 1)


##################################################
def decay_to_lag(decay):
    return (1 - decay) / decay


##################################################
def __ema_helper(p, x_prev, decay):
    return (1 - decay) * x_prev + decay * p 


##################################################
def compute_ema(p_series, lag):
    """computes the exponential moving average of :arg p_series:, given :arg lag:"""
    assert len(p_series) > lag, "not enough data to compute EMA at lag [%i]" %lag
    assert isinstance(lag, int), "lag is not of type [%s]" %type(int)
    decay = lag_to_decay(lag)
    n = len(p_series)
    res = numpy.zeros(n)
    res[lag] = p_series[lag]
    for i in numpy.arange(lag + 1, n, 1):
        res[i] = __ema_helper(p_series[i], res[i - 1], decay)

    return res


##################################################
def test_compute_ema():
    import numpy
    import matplotlib.pyplot as plt
    RState = numpy.random.RandomState(12357)
    x = RState.standard_normal(1000)
    y = 100 + numpy.cumsum(x)
    ema_10 = compute_ema(y, 10)
    ema_100 = compute_ema(y, 100)

    """test shape and correct insert of EMA(0)"""
    assert (len(y) == len(ema_10)) and (len(ema_10) == len(ema_100))
    numpy.testing.assert_almost_equal(ema_10[10], y[10])
    numpy.testing.assert_almost_equal(ema_100[100], y[100])    

    """test calculation 10"""
    l = 10
    d = 1 / (l + 1)
    test = (1 - d) * y[l] + d * y[l + 1]
    numpy.testing.assert_almost_equal(ema_10[l + 1], test)
    del l, d, test

    """test_calculation_100"""
    l = 100
    d = 1 / (l + 1)
    test = (1 - d) * y[l] + d * y[l + 1]
    numpy.testing.assert_almost_equal(ema_100[l + 1], test)
    del l, d, test
    
    """for those which seeing seeing is believing
    fig, ax = plt.subplots(1)
    ax.plot(y)
    ax.plot(ema_10)
    ax.plot(ema_100)
    plt.show()
    """


##################################################
def plot_series(x, dates, label, subplots=1):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.cbook as cbook

    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    years_fmt = mdates.DateFormatter('%Y')

    fig, ax = plt.subplots(subplots, sharex=True)
    if subplots > 1:
        axis = ax[0]
    else:
        axis = ax

    axis.plot(dates, x, label=label)
    
    # format the ticks
    axis.xaxis.set_major_locator(years)
    axis.xaxis.set_major_formatter(years_fmt)
    axis.xaxis.set_minor_locator(months)

    # round to nearest years.
    datemin = numpy.datetime64(dates[0], 'Y')
    datemax = numpy.datetime64(dates[-1], 'Y') + numpy.timedelta64(1, 'Y')
    axis.set_xlim(datemin, datemax)

    # format the coords message box
    axis.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    axis.format_ydata = lambda x: '$%1.2f' % x  # format the price.
    axis.grid(True)

    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    fig.autofmt_xdate()

    return fig, ax


##################################################
def compute_sharpe(nav, risk_free):
    """compute the Sharpe ratio of excess returns div. volatility"""
    days_per_year = 250
    excess_return = numpy.diff(numpy.log(numpy.maximum(1e-6, nav))) - (risk_free[:-1] / days_per_year)
    return numpy.average(excess_return) / (1e-6 + numpy.std(excess_return, ddof=1)) * numpy.sqrt(days_per_year)


##################################################
if __name__ == "__main__":
    #test_erc_example()
    #test_compute_ema()
    #back_test(["QQQ", "BND"], [])

    ##
    as_of_date = datetime.datetime(2020, 5, 13)
    burn_in = 120
    rebalance = 20
    tickers = ["QQQ", "BND"]
    px_cont = prepare_container(tickers, as_of_date, access_yf_time_series)    
    erc_strat = long_only_equal_risk_contributions(burn_in=burn_in, rebalance=rebalance, container=copy.deepcopy(px_cont))
    erc_strat_sig = long_only_equal_risk_contributions_signal(burn_in=burn_in, rebalance=rebalance, container=copy.deepcopy(px_cont))
    
    ##
    ir_cont = prepare_container(["^IRX"], as_of_date, access_yf_time_series, fillna=True)
    ir = pandas.Series(data=ir_cont.prices.flatten(), index=ir_cont.dates, name="IRX") / 100
    nav_erc = pandas.Series(data=erc_strat.nav.flatten(), index=erc_strat.dates, name="ERC_NAV")
    nav_erc_sig = pandas.Series(data=erc_strat_sig.nav.flatten(), index=erc_strat.dates, name="ERC_SIG_NAV")
    common_dates = ir.index.intersection(nav_erc.index).intersection(nav_erc_sig.index)[burn_in:]

    fig, ax = plt.subplots(1)
    ax.plot(erc_strat.dates[burn_in:], erc_strat.nav[burn_in:],
            label="ERC-Portfolio sharpe=%1.3f" %compute_sharpe(nav_erc[common_dates], ir[common_dates]))
    ax.plot(erc_strat.dates[burn_in:], erc_strat_sig.nav[burn_in:],
            label="ERC-SIG-Portfolio sharpe=%1.3f" %compute_sharpe(nav_erc_sig[common_dates], ir[common_dates]))
    ax.legend()
    plt.show()
    
