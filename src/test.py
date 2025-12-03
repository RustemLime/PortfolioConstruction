# import pandas as pd
# import numpy as np
# import yfinance as yf
# import skfolio
# import skfolio.preprocessing as skp
# import skfolio.optimization as sko
# from skfolio import Portfolio
# import json
# from data_store import save_data, get_data
# import uuid
# from datetime import datetime
# import requests
# import io

# from skfolio.moments import ImpliedCovariance, ShrunkMu
# from skfolio.optimization import MeanRisk
# from skfolio.prior import EmpiricalPrior
# from skfolio.preprocessing import prices_to_returns
# from skfolio.datasets import load_sp500_dataset, load_sp500_implied_vol_dataset
# import skfolio.moments.expected_returns._base as base

# import numpy as np
# from plotly.io import show

# from skfolio import MultiPeriodPortfolio, Population, Portfolio
# from skfolio.datasets import load_sp500_dataset
# from skfolio.model_selection import WalkForward, cross_val_predict
# from skfolio.optimization import MeanRisk, ObjectiveFunction
# from skfolio.preprocessing import prices_to_returns


# from sklearn import set_config

# set_config(enable_metadata_routing=True)

# tickers = ['AAPL','MSFT']
# start_date = '2020-01-01'
# end_date = '2021-01-01'

# prices = pd.DataFrame(yf.download(tickers, start=start_date, end=end_date, auto_adjust=True))
# prices = prices['Close'].ffill()
# rets = skp.prices_to_returns(prices)

# signals = rets.copy()#pd.DataFrame(1, index=rets.index, columns=rets.columns)


# class CustomMu(base.BaseMu):
#     def __init__(self):
#         super().__init__()
#         self.mu_ = np.array([0.01])

#     def fit(self, X, y=None, *, signals=None):
#         if signals is None:
#             raise ValueError("signals parameter is required")
#         mu = signals.iloc[-1,:].values
#         mu = np.maximum(mu, 0.0001)
#         self.mu_ = mu
    
#         print(self.mu_)
#         return self

# model = sko.MeanRisk(objective_function=ObjectiveFunction.MAXIMIZE_RATIO, prior_estimator=EmpiricalPrior(mu_estimator=CustomMu().set_fit_request(signals=True)))  # Enable signals routing
# model.fit(rets, signals=signals)
# weights_dict = dict(zip(rets.columns, model.weights_))



# print(weights_dict)


# holding_period = 60
# fitting_period = 60
# cv = WalkForward(train_size=fitting_period, test_size=holding_period)
# pred1 = cross_val_predict(model, rets, cv=cv, n_jobs=-1, params={'signals': signals})

# holding_period = 20
# fitting_period = 100
# cv = WalkForward(train_size=fitting_period, test_size=holding_period)
# pred2 = cross_val_predict(model, rets, cv=cv, n_jobs=-1, params={'signals': signals})

# population = Population([pred1, pred2])
# fig = population.plot_cumulative_returns()
# show(fig)

