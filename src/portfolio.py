import pandas as pd
import yfinance as yf
import skfolio as skf
import skfolio.preprocessing as skp
import skfolio.optimization as sko



def portfolio(tickers, start_date='2020-01-01', **mean_risk_params):
    """
    This function calculates the weights of a portfolio of stocks using the MeanRisk model.
    Parameters:
    tickers: list of stock tickers
    start_date: start date of the data
    mean_risk_params: dictionary of parameters to pass to sko.MeanRisk()
                      (e.g., risk_measure, solver, max_volatility, etc.)
    Returns:
    weights_dict: dictionary of weights for each stock
    """
    prices = pd.DataFrame(yf.download(tickers, start=start_date, auto_adjust=True))
    prices = prices['Close'].ffill()
    rets = skp.prices_to_returns(prices)
    model = sko.MeanRisk(**mean_risk_params)  # Unpack the dictionary
    model.fit(rets)
    weights_dict = dict(zip(tickers, model.weights_))
    return weights_dict

tickers = ['AAPL', 'MSFT','NVDA']
weights = portfolio(tickers, risk_measure=skf.RiskMeasure.CVAR)
print(weights)