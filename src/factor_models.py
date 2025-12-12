import getfactormodels
import statsmodels.api as sm
from data_store import get_data
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from portfolio import prepare_returns, portfolio, backtest
import json




def factor_model(portfolio_cumulative_returns_data_id = None):
    """
    This function fits a Fama-French 3-factor model to the portfolio returns.
    Parameters:
    portfolio_cumulative_returns_data_id: id of the portfolio cumulative returns data
    Returns:
    model: summary of the regression
    """

    if portfolio_cumulative_returns_data_id is None:


        # Generate random daily returns for the last 3 years (~756 trading days)
        end_date = datetime.today()
        start_date = end_date - timedelta(days=3*365)
        dates = pd.bdate_range(start=start_date, end=end_date)

        np.random.seed(42)  # for reproducibility
        daily_returns = np.random.normal(loc=0.0003, scale=0.015, size=len(dates))
        cumulative_returns = (1 + pd.Series(daily_returns, index=dates)).cumprod() - 1
        portfolio_returns = pd.DataFrame({"Return": cumulative_returns})
    else:
        portfolio_returns = get_data(portfolio_cumulative_returns_data_id)
        # If portfolio_returns is a Series, convert it to a DataFrame with column "Return"
        if isinstance(portfolio_returns, pd.Series):
            portfolio_returns = portfolio_returns.to_frame(name="Return")
        elif isinstance(portfolio_returns, dict):
            # If it's a dict, try to convert to DataFrame, then extract the column if only one
            portfolio_returns = pd.DataFrame(portfolio_returns)
            if portfolio_returns.shape[1] == 1:
                portfolio_returns.columns = ["Return"]
        # If it's already a DataFrame but columns aren't named "Return", try to fix
        elif isinstance(portfolio_returns, pd.DataFrame):
            if "Return" not in portfolio_returns.columns:
                # If only one column, rename
                if portfolio_returns.shape[1] == 1:
                    portfolio_returns.columns = ["Return"]
        # portfolio_returns = pd.DataFrame(portfolio_returns, columns=['Return'])

    factors = getfactormodels.get_factors(model='3', frequency='d')

    print(factors.index)
    print(portfolio_returns.index)

    # # Filter factor dates to match the asset
    # factors_subset = factors[
    #     factors.index.isin(portfolio_returns.index)
    # ].copy()
    factors_subset = factors

    # Step 3: Calculate excess returns for the asset
    factors_subset["Excess_Return"] = portfolio_returns["Return"] - factors_subset["RF"]

    factors_subset.dropna(inplace=True)
    
    print(factors_subset[["Mkt-RF", "SMB", "HML"]])
    print(factors_subset["Excess_Return"])

    # Prepare the independent variables (add a constant to the model)
    X = np.array(sm.add_constant(factors_subset[["Mkt-RF", "SMB", "HML"]]))
    # The dependent variable
    y = np.array(factors_subset["Excess_Return"])
    # Run the regression
    model = sm.OLS(y, X).fit()
    # Display the summary of the regression
    print(model.summary())



rets = prepare_returns(tickers=['AAPL', 'MSFT'])
print(rets)



weights = portfolio(json.loads(rets)['data_id'])
print(weights)
 
backtest_result = backtest(json.loads(weights)['data_id'], json.loads(rets)['data_id'])
print(backtest_result['portfolio_cumulative_returns_data_id'])
factor_model(backtest_result['portfolio_cumulative_returns_data_id'])




