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
import limexhub
import os


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

    # Load API token from config file
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    limexhub_api_token = config['limexhub_api_token']
    client = limexhub.RestAPI(token=limexhub_api_token)
    prices = client.candles(tickers,
                            start=start_date, 
                            end=end_date, 
                            interval = "1d")
    
    prices = prices['close'].ffill()
    # Reshape: convert symbol from index level to columns
    if isinstance(prices.index, pd.MultiIndex):
        prices = prices.unstack()
        # Flatten MultiIndex columns if needed
        if isinstance(prices.columns, pd.MultiIndex):
            prices.columns = prices.columns.droplevel(0)

    # print(prices)


    # prices = pd.DataFrame(yf.download(tickers, start=start_date, end=end_date, auto_adjust=True))
    # prices = prices['Close'].ffill()

    # print(prices)
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
    response_payload = portfolio(data_id, start_date='2020-01-01', end_date='2020-12-31', meanrisk_kwargs={'risk_measure': "CVAR", 'objective_function': "MINIMIZE_RISK"})
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

    portfolio_cumulative_returns = portfolio.cumulative_returns_df


    data_id = str(uuid.uuid4())
    save_data(data_id, portfolio_cumulative_returns)


    portfolio_result = portfolio.summary()
    portfolio_result['portfolio_cumulative_returns_data_id'] = data_id
    return portfolio_result

    

    # data_id = str(uuid.uuid4())
    # save_data(data_id, portfolio_result)    

    # response_payload = {
    #     "render_type": "table",
    #     "table_data": portfolio_result,
    #     "data_id": data_id
    # }
    # return portfolio_result






def equity_curve(portfolio_cumulative_returns_data_id):
    """
    This function plots the equity curve of the portfolio.
    Parameters:
    portfolio_cumulative_returns_data_id: id of the portfolio cumulative returns data
    Returns:
    response_payload: dictionary of response payload
    """
    
    portfolio_returns = get_data(portfolio_cumulative_returns_data_id)
    
    # Convert to DataFrame if it's not already
    if not isinstance(portfolio_returns, pd.DataFrame):
        portfolio_returns = pd.DataFrame(portfolio_returns)
    
    # Ensure we have a datetime index or convert the first column to index
    if not isinstance(portfolio_returns.index, pd.DatetimeIndex):
        if len(portfolio_returns.columns) > 0:
            portfolio_returns = portfolio_returns.set_index(portfolio_returns.columns[0])
        portfolio_returns.index = pd.to_datetime(portfolio_returns.index)
    
    # Get the first column's values (or the Series if it's a Series)
    if len(portfolio_returns.columns) > 0:
        y_values = portfolio_returns.iloc[:, 0].tolist()
    else:
        y_values = portfolio_returns.values.flatten().tolist()
    
    # Convert datetime index to strings for JSON serialization
    # Ensure index is DatetimeIndex and format as strings
    datetime_index = pd.to_datetime(portfolio_returns.index)
    x_values = datetime_index.strftime('%Y-%m-%d').tolist()

    response_payload = {
        "render_type": "plot", 
        "title": f"Equity Curve",
        "plot_data": {
            "data": [{
                "x": x_values,
                "y": y_values,
                "type": "scatter",
                "mode": "lines"
            }],
            "layout": {"title": f"Equity Curve"}
        }
    }
    
    # MCP ожидает строку, поэтому сериализуем
    return response_payload





# print(get_sp500_tickers())
rets = prepare_returns(tickers=['AAPL', 'MSFT', 'NVDA'])
print(rets)



# weights = portfolio(json.loads(rets)['data_id'])
# print(weights)
 
# backtest_result = backtest(json.loads(weights)['data_id'], json.loads(rets)['data_id'])
# print(backtest_result)

# simple_plot_result = equity_curve(backtest_result['portfolio_cumulative_returns_data_id'])
# print(simple_plot_result)