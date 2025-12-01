import pandas as pd
import yfinance as yf
import skfolio
import skfolio.preprocessing as skp
import skfolio.optimization as sko


from yfinance.const import k




def portfolio(tickers, start_date='2020-01-01', meanrisk_kwargs={'risk_measure': "VARIANCE"}):
    if meanrisk_kwargs is not None:
        for key, value in meanrisk_kwargs.items():
            if key == 'objective_function':
                meanrisk_kwargs[key] = getattr(sko.ObjectiveFunction, value)
            elif key == 'risk_measure':
                meanrisk_kwargs[key] = getattr(skfolio.RiskMeasure, value)

    """
    This function calculates the weights of a portfolio of stocks using the MeanRisk model.
    Parameters:
    tickers: list of stock tickers
    start_date: start date of the data

    meanrisk_kwargs: dictionary of parameters to pass to skfolio function MeanRisk()
    Example:
    meanrisk_kwargs = {'risk_measure': "VARIANCE"}
    meanrisk_kwargs = {'objective_function': "MINIMIZE_RISK"}
    meanrisk_kwargs = {'objective_function': "MINIMIZE_RISK", 'risk_measure': "VARIANCE"}
    meanrisk_kwargs = {'objective_function': "MINIMIZE_RISK", 'risk_measure': "CVAR"}
    meanrisk_kwargs = {'objective_function': "MINIMIZE_RISK", 'risk_measure': "STANDARD_DEVIATION"}
    meanrisk_kwargs = {'objective_function': "MINIMIZE_RISK", 'risk_measure': "SEMI_DEVIATION"}
    meanrisk_kwargs = {'objective_function': "MINIMIZE_RISK", 'risk_measure': "MEAN_ABSOLUTE_DEVIATION"}
    meanrisk_kwargs = {'objective_function': "MINIMIZE_RISK", 'risk_measure': "FIRST_LOWER_PARTIAL_MOMENT"}

    ----------
    objective_function : 
        Can be any of:

            * MINIMIZE_RISK
            * MAXIMIZE_RETURN
            * MAXIMIZE_UTILITY
            * MAXIMIZE_RATIO

        The default is `ObjectiveFunction.MINIMIZE_RISK`.

    risk_measure : 
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

    # print(kwargs)
    # if kwargs is not None:
    #     for key, value in kwargs.items():
    #         print(key, value)
    #         if key == 'objective_function':
    #             kwargs[key] = getattr(sko.ObjectiveFunction, value)
    #         elif key == 'risk_measure':
    #             kwargs[key] = getattr(skfolio.RiskMeasure, value)

    model = sko.MeanRisk(**meanrisk_kwargs)  # Unpack the dictionary
    model.fit(rets)
    weights_dict = dict(zip(tickers, model.weights_))
    return weights_dict

tickers = ['AAPL', 'MSFT','NVDA']
weights = portfolio(tickers, meanrisk_kwargs={'risk_measure': "VARIANCE"})
print(weights)