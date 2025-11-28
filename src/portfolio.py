import pandas as pd
import yfinance as yf
import skfolio as skf
import skfolio.preprocessing as skp
import skfolio.optimization as sko


def portfolio(tickers,start_date = '2020-01-01'):
    """
    This function calculates the weights of a portfolio of stocks using the MeanRisk model.
    Parameters:
    tickers: list of stock tickers
    start_date: start date of the data
    Returns:
    weights_dict: dictionary of weights for each stock
    """
    prices = pd.DataFrame(yf.download(tickers, start=start_date,auto_adjust=True))
    prices = prices['Close'].ffill()
    rets = skp.prices_to_returns(prices)
    model = sko.MeanRisk()
    model.fit(rets)
    weights_dict = dict(zip(tickers, model.weights_))
    return weights_dict

tickers = ['AAPL', 'AMD']
weights = portfolio(tickers)
print(weights)