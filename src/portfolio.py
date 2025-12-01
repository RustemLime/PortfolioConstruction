import pandas as pd
import yfinance as yf
import skfolio
import skfolio.preprocessing as skp
import skfolio.optimization as sko
from skfolio import Portfolio
import json
from data_store import save_data, get_data
import uuid


def prepare_returns(tickers, start_date='2020-01-01'):
    """
    This function prepares the returns for the portfolio.
    Parameters:
    tickers: list of stock tickers
    start_date: start date of the data
    Returns:
    rets: returns for the portfolio
    """
    prices = pd.DataFrame(yf.download(tickers, start=start_date, auto_adjust=True))
    prices = prices['Close'].ffill()
    rets = skp.prices_to_returns(prices)

    return rets


def portfolio(tickers, start_date='2020-01-01', meanrisk_kwargs={'risk_measure': "VARIANCE"}):
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
    response_payload: dictionary of response payload
    data_id: id of the data
    Example:
    response_payload = {
        "render_type": "table",
        "table_data": weights_dict,
        "data_id": data_id
    }
    """

    if meanrisk_kwargs is not None:
        for key, value in meanrisk_kwargs.items():
            if key == 'objective_function':
                meanrisk_kwargs[key] = getattr(sko.ObjectiveFunction, value)
            elif key == 'risk_measure':
                meanrisk_kwargs[key] = getattr(skfolio.RiskMeasure, value)



    rets = prepare_returns(tickers, start_date)

    model = sko.MeanRisk(**meanrisk_kwargs)  # Unpack the dictionary
    model.fit(rets)
    weights_dict = dict(zip(tickers, model.weights_))

    data_id = str(uuid.uuid4())
    save_data(data_id, weights_dict)

    response_payload = {
        "render_type": "table",
        "table_data": weights_dict,
        "data_id": data_id
        }  
    return json.dumps(response_payload)

def backtest(data_id):
    """
    This function backtests the portfolio.
    Parameters:
    data_id: id of the data
    Returns:
    portfolio.summary(): summary of the portfolio
    """
    weights_dict = get_data(data_id)

    print(list(weights_dict.keys()))


    rets = prepare_returns(list(weights_dict.keys()), '2020-01-01')

    weights = list(weights_dict.values())

    portfolio = Portfolio(X=rets, weights=weights)

    return portfolio.summary()






def simple_plot():
    """Возвращает данные о продажах для построения графика."""
    
    
    data = [10, 20, 15, 30]
    indexes = ["Jan", "Feb", "Mar", "Apr"]
    

    response_payload = {
        "render_type": "plot", 
        "title": f"Simple Plot",
        "plot_data": {
            "data": [{
                "x": indexes,
                "y": data,
                "type": "bar",
                "marker": {"color": "indianred"}
            }],
            "layout": {"title": f"Simple Plot"}
        }
    }
    
    # MCP ожидает строку, поэтому сериализуем
    return response_payload



# tickers = ['AAPL', 'TSLA','NVDA','QQQ']
# weights = portfolio(tickers, meanrisk_kwargs={'risk_measure': "VARIANCE", 'objective_function': "MINIMIZE_RISK", 'min_weights': 0.1, 'max_weights': 0.5})
# print(weights)

# json_weights = json.loads(weights)

# print(backtest(json_weights['data_id']))

# print(simple_plot())