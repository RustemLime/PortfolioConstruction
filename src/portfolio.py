import pandas as pd
import yfinance as yf
import skfolio
import skfolio.preprocessing as skp
import skfolio.optimization as sko




def portfolio(tickers, start_date='2020-01-01', **mean_risk_params):
    """
    This function calculates the weights of a portfolio of stocks using the MeanRisk model.
    Parameters:
    tickers: list of stock tickers
    start_date: start date of the data
    
    objective_function : ObjectiveFunction, default=ObjectiveFunction.MINIMIZE_RISK
        :class:`~skfolio.optimization.ObjectiveFunction` of the optimization.
        Can be any of:

            * MINIMIZE_RISK
            * MAXIMIZE_RETURN
            * MAXIMIZE_UTILITY
            * MAXIMIZE_RATIO

        The default is `ObjectiveFunction.MINIMIZE_RISK`.

    risk_measure : RiskMeasure, default=RiskMeasure.VARIANCE
        :class:`~skfolio.meta.RiskMeasure` of the optimization.
        Can be any of:

            * VARIANCE
            * SEMI_VARIANCE
            * STANDARD_DEVIATION
            * SEMI_DEVIATION
            * MEAN_ABSOLUTE_DEVIATION
            * FIRST_LOWER_PARTIAL_MOMENT
            * CVAR
            * EVAR
            * WORST_REALIZATION
            * CDAR
            * MAX_DRAWDOWN
            * AVERAGE_DRAWDOWN
            * EDAR
            * ULCER_INDEX
            * GINI_MEAN_DIFFERENCE_RATIO

        The default is `RiskMeasure.VARIANCE`.
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
weights = portfolio(tickers, risk_measure=skfolio.RiskMeasure.CVAR)
print(weights)