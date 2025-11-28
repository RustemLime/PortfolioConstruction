# Portfolio Construction

A Python tool for constructing optimal portfolios using MeanRisk optimization. This project uses historical stock data to calculate optimal portfolio weights based on mean-risk optimization.

## Features

- Download historical stock price data using Yahoo Finance
- Calculate optimal portfolio weights using MeanRisk optimization model
- Support for multiple stock tickers
- Customizable start date for historical data

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd PortfolioConstruction
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The main function `portfolio()` calculates optimal portfolio weights for a given list of stock tickers.

### Basic Example

```python
from src.portfolio import portfolio

# Define your stock tickers
tickers = ['AAPL', 'AMD']

# Calculate optimal weights
weights = portfolio(tickers)
print(weights)
```

### Custom Start Date

You can specify a custom start date for the historical data:

```python
tickers = ['AAPL', 'MSFT', 'GOOGL']
weights = portfolio(tickers, start_date='2021-01-01')
print(weights)
```

## How It Works

1. **Data Collection**: Downloads historical stock price data from Yahoo Finance for the specified tickers and date range
2. **Data Processing**: Converts price data to returns using forward-fill for missing values
3. **Optimization**: Uses the MeanRisk model from skfolio to calculate optimal portfolio weights
4. **Output**: Returns a dictionary mapping each ticker to its optimal weight in the portfolio

## Dependencies

- `pandas`: Data manipulation and analysis
- `yfinance`: Yahoo Finance data downloader
- `skfolio`: Portfolio optimization library
- `numpy`: Numerical computing
- `scikit-learn`: Machine learning utilities

See `requirements.txt` for the complete list of dependencies with versions.

## Project Structure

```
PortfolioConstruction/
├── README.md
├── requirements.txt
└── src/
    └── portfolio.py
```

## Notes

- The default start date is `'2020-01-01'`
- The function uses forward-fill (`ffill()`) to handle missing data
- Portfolio weights are calculated using the MeanRisk optimization model, which balances expected return and risk

## License

MIT

