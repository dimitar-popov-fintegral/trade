import os
import sys

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(THIS_DIR, '..'))

import time
import datetime
import matplotlib.pyplot as plt 
import numpy
import calendar

from scipy.optimize import minimize
from enum import Enum

import dev.data_util as du
import dev.data as dt
import dev.alpha_vantage as ap

##################################################
class AVKeys(Enum):
    SYMBOL = "2. Symbol"


##################################################
def access_time_series(date: str, tickers: list):
    keys = ["%s_%s" %(date, ticker) for ticker in tickers]
    frames = list()
    with du.RedisDb() as redis:
        for key in keys:
            start = time.time()
            response = ap.du.decompress(du.r_read(key, redis))
            meta, df = ap.parse_response(response)
            df = df.astype(float)
            df.index = [numpy.datetime64(i) for i in df.index]
            ticker = meta[AVKeys.SYMBOL.value]
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
    tickers, frames = time_series(as_of_date.strftime("%Y-%m-%d"), tickers)

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
def back_test(tickers, benchmarks):
    BURN_IN = 120
    REBALANCE = 30
    
    as_of_date = datetime.datetime(2020, 4, 9)
    frames = access_time_series(as_of_date.strftime("%Y-%m-%d"), tickers)

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
    prices = numpy.vstack(instrument_closing)
    returns = dt.log_returns(prices.T)
    n = len(returns)
    t = len(tickers)
    budget = numpy.zeros(t)
    ptf_performance = numpy.zeros(n)
    ptf_returns = numpy.zeros(n)
    NAV = 1.0

    """Loop over time periods"""
    for i in range(n):
        print("iter %d" %i)
        if i < BURN_IN:
            continue
        if i % REBALANCE == 0:
            cov = numpy.cov(returns[:i].T)
            rcov = regularize_cov(cov)
            corr = cov_to_corr(cov)
            budget = equally_weighted_risk_contributions(rcov)
            display = ", ".join([str((i,j)) for i,j in zip(tickers, budget)])
            print("total budget: [%s]" %budget.sum())
            print("resulting budgets: [%s]" %display)
            print("weights: [%s]" %budget)
        ptf_returns[i] = numpy.einsum("i,i", budget, numpy.exp(returns[i]))
        NAV *= ptf_returns[i]
        ptf_performance[i] = NAV
        #print("performance [%f] at [%d]" %(performance[i], i))
        print("NAV = [%f]" %NAV)

    """Plot results"""
    fig, ax = plot_series(ptf_performance[BURN_IN:], dates[BURN_IN:], "Ptf")
    asset = 0
    ax.plot(dates[BURN_IN:],
            numpy.cumprod(numpy.exp(returns[:,asset]))[BURN_IN:],
            label=tickers[asset])
    asset = 1
    ax.plot(dates[BURN_IN:],
            numpy.cumprod(numpy.exp(returns[:,asset]))[BURN_IN:],
            label=tickers[asset])
    ax.legend()

    plt.show()


##################################################
def plot_series(x, dates, label):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.cbook as cbook

    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    years_fmt = mdates.DateFormatter('%Y')

    fig, ax = plt.subplots()
    ax.plot(dates, x, label=label)
    
    # format the ticks
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(months)

    # round to nearest years.
    datemin = numpy.datetime64(dates[0], 'Y')
    datemax = numpy.datetime64(dates[-1], 'Y') + numpy.timedelta64(1, 'Y')
    ax.set_xlim(datemin, datemax)

    # format the coords message box
    ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    ax.format_ydata = lambda x: '$%1.2f' % x  # format the price.
    ax.grid(True)

    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    fig.autofmt_xdate()

    return fig, ax


##################################################
if __name__ == "__main__":
    test_erc_example()
    back_test(["QQQ", "EEM", "VWO", "SMH", "QTEC", "VEA"], [])

