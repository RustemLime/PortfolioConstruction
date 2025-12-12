import getfactormodels
import statsmodels.api as sm
from data_store import get_data
import pandas as pd
import numpy as np
from datetime import datetime, timedelta




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
        portfolio_returns = pd.DataFrame(portfolio_returns, columns=['Return'])

    factors = getfactormodels.get_factors(model='3', frequency='d')


    # Filter factor dates to match the asset
    factors_subset = factors[
        factors.index.isin(portfolio_returns.index)
    ].copy()

    # Step 3: Calculate excess returns for the asset
    factors_subset["Excess_Return"] = portfolio_returns["Return"] - factors_subset["RF"]
    


    # Prepare the independent variables (add a constant to the model)
    X = sm.add_constant(factors_subset[["Mkt-RF", "SMB", "HML"]])
    # The dependent variable
    y = factors_subset["Excess_Return"]
    # Run the regression
    model = sm.OLS(y, X).fit()
    # Display the summary of the regression
    print(model.summary())

# factor_model()




