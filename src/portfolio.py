import pandas as pd
import yfinance as yf
import skfolio
import skfolio.preprocessing as skp
import skfolio.optimization as sko
from skfolio import Portfolio
import json
from data_store import save_data, get_data
import uuid
from datetime import datetime
import requests
import io


def get_sp500_tickers():
    """
    This function returns the tickers of the S&P 500 index.
    Returns:
    tickers: list of stock tickers
    Example:
    tickers = get_sp500_tickers()
    print(tickers)
    """

    url = 'https://www.slickcharts.com/sp500'
    user_agent = 'Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/111.0'  # Default user-agent fails.
    response = requests.get(url, headers={'User-Agent': user_agent})
    df = pd.read_html(io.StringIO(response.text), match='Symbol')[0]
    df = df.set_index('Symbol')
    tickers = df.index.tolist()
    return tickers

def prepare_returns(tickers = None, universe = 'sp500', start_date='2020-01-01', end_date=datetime.now().strftime('%Y-%m-%d')):
    """
    This function prepares the returns for the portfolio.
    Parameters:
    tickers: list of stock tickers
    universe: universe of the portfolio
    start_date: start date of the data
    end_date: end date of the data
    Returns:
    response_payload: dictionary of response payload
    data_id: id of the data
    Example:
    response_payload = {
        "data_id": data_id
    }

    Example:
    rets = prepare_returns(universe='sp500')
    rets = prepare_returns(tickers=['AAPL', 'MSFT', 'GOOGL'])
    rets = prepare_returns(tickers=['AAPL', 'MSFT', 'GOOGL'], universe='custom')
    rets = prepare_returns(tickers=['AAPL', 'MSFT', 'GOOGL'], universe='custom', start_date='2021-01-01', end_date='2021-12-31')
    rets = prepare_returns(tickers=['AAPL', 'MSFT', 'GOOGL'], universe='custom', start_date='2021-01-01', end_date='2021-12-31')
    """


    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')


    if tickers is None and universe == 'sp500':
        tickers = get_sp500_tickers()
    if tickers is None:
        raise ValueError(f"Invalid universe or tickers: {tickers} {universe}")

    prices = pd.DataFrame(yf.download(tickers, start=start_date, end=end_date, auto_adjust=True))
    prices = prices['Close'].ffill()
    rets = skp.prices_to_returns(prices)

    data_id = str(uuid.uuid4())
    
    # Convert DataFrame to JSON-serializable format (preserves index as a column)
    rets_json = rets.reset_index().to_dict('records')
    save_data(data_id, rets_json)

    response_payload = {
        "data_id": data_id
    }
    return json.dumps(response_payload)


def portfolio(data_id, start_date='2020-01-01', end_date=datetime.now().strftime('%Y-%m-%d'), meanrisk_kwargs={'risk_measure': "VARIANCE"}):
    """
    This function calculates the weights of a portfolio of stocks using the MeanRisk model.
    Parameters:
    data_id: id of the data
    start_date: start date of the data
    end_date: end date of the data
    meanrisk_kwargs: dictionary of parameters to pass to skfolio function MeanRisk()
    Returns:
    response_payload: dictionary of response payload
    data_id: id of the data
    Example:
    response_payload = {
        "render_type": "table",
        "table_data": weights_dict,
        "data_id": data_id
    }
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

    Example:
    response_payload = portfolio(data_id, start_date='2020-01-01', end_date='2020-12-31', meanrisk_kwargs={'risk_measure': "VARIANCE"})
    response_payload = portfolio(data_id, start_date='2020-01-01', end_date='2020-12-31', meanrisk_kwargs={'risk_measure': "VARIANCE", 'objective_function': "MINIMIZE_RISK"})
    response_payload = portfolio(data_id, start_date='2020-01-01', end_date='2020-12-31', meanrisk_kwargs={'risk_measure': "VARIANCE", 'objective_function': "MINIMIZE_RISK", 'min_weights': 0.1, 'max_weights': 0.5})
    """
    

    if meanrisk_kwargs is not None:
        for key, value in meanrisk_kwargs.items():
            if key == 'objective_function':
                meanrisk_kwargs[key] = getattr(sko.ObjectiveFunction, value)
            elif key == 'risk_measure':
                meanrisk_kwargs[key] = getattr(skfolio.RiskMeasure, value)



    rets_data = get_data(data_id)
    rets = pd.DataFrame(rets_data)
    # Set the first column (date/index) as the index if it exists
    if len(rets.columns) > 0:
        rets = rets.set_index(rets.columns[0])
    rets = rets.loc[start_date:end_date]

    model = sko.MeanRisk(**meanrisk_kwargs)  # Unpack the dictionary
    model.fit(rets)
    weights_dict = dict(zip(rets.columns, model.weights_))

    data_id = str(uuid.uuid4())
    save_data(data_id, weights_dict)

    response_payload = {
        "render_type": "table",
        "table_data": weights_dict,
        "data_id": data_id
        }  
    return json.dumps(response_payload)

def backtest(portfolio_weights_data_id,rets_data_id):
    """
    This function backtests the portfolio.
    Parameters:
    portfolio_weights_data_id: id of the portfolio weights data
    rets_data_id: id of the returns data
    Returns:
    portfolio.summary(): summary of the portfolio
    """
    weights_dict = get_data(portfolio_weights_data_id)
    weights = list(weights_dict.values())




    rets = get_data(rets_data_id)
    rets = pd.DataFrame(rets)
    # Set the first column (date/index) as the index if it exists
    if len(rets.columns) > 0:
        rets = rets.set_index(rets.columns[0])


    portfolio = Portfolio(X=rets, weights=weights)

    portfolio_result = portfolio.summary()

    # data_id = str(uuid.uuid4())
    # save_data(data_id, portfolio_result)    

    # response_payload = {
    #     "render_type": "table",
    #     "table_data": portfolio_result,
    #     "data_id": data_id
    # }
    return portfolio_result






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





# print(get_sp500_tickers())
rets = prepare_returns(tickers=['AAPL', 'MSFT', 'GOOGL'])
print(rets)

weights = portfolio(json.loads(rets)['data_id'])
print(weights)
 
backtest_result = backtest(json.loads(weights)['data_id'], json.loads(rets)['data_id'])
print(backtest_result)