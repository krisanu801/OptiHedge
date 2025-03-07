import pandas as pd
import numpy as np
import logging
from typing import Optional

# Configure logging
logger = logging.getLogger(__name__)


class Backtester:
    """
    Handles backtesting of trading strategies.

    Simulates trades based on generated signals and evaluates the performance of the strategy.
    """

    def __init__(self, data: pd.DataFrame, signals: pd.DataFrame, initial_capital: float = 100000, transaction_cost: float = 0.001) -> None:
        """
        Initializes the Backtester.

        Args:
            data (pd.DataFrame): A Pandas DataFrame containing historical stock data.
                                  The DataFrame should have a MultiIndex with tickers as the first level
                                  and columns like 'Close' as the second level.
            signals (pd.DataFrame): A Pandas DataFrame containing the trading signals.
                                     1 indicates a long position, -1 indicates a short position, and 0 indicates no position.
            initial_capital (float): The initial capital for the backtest.
            transaction_cost (float): The transaction cost as a percentage of the trade value.
        """
        self.data = data
        self.signals = signals
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.portfolio: pd.DataFrame = pd.DataFrame()

    def run_backtest(self) -> pd.DataFrame:
        """
        Runs the backtest simulation.

        Returns:
            pd.DataFrame: A Pandas DataFrame containing the portfolio value over time.
        """
        try:
            if self.signals.empty:
                logger.warning("No signals provided. Cannot run backtest.")
                return pd.DataFrame()

            if self.data.empty:
                logger.warning("No data provided. Cannot run backtest.")
                return pd.DataFrame()

            # Align signals with data
            aligned_signals = self.signals.reindex(self.data.index).fillna(0)

            # Get close prices
            close_prices = self.data.loc[:, (slice(None), 'Close')]

            # Initialize portfolio DataFrame
            portfolio = pd.DataFrame(index=self.data.index)
            portfolio['holdings'] = 0.0
            portfolio['cash'] = self.initial_capital
            portfolio['total'] = self.initial_capital
            portfolio['returns'] = 0.0

            # Ensure that the signals DataFrame has the same columns as the close_prices DataFrame
            # This is important for cases where the strategy only generates signals for a subset of the available assets.
            for col in close_prices.columns:
                if col not in aligned_signals.columns:
                    aligned_signals[col] = 0  # No signal for this asset

            # Iterate through the data and simulate trades
            for i in range(1, len(self.data)):
                today = self.data.index[i]
                yesterday = self.data.index[i - 1]

                # Get the signals for today
                today_signals = aligned_signals.loc[today]

                # Get the close prices for yesterday and today
                yesterday_prices = close_prices.loc[yesterday]
                today_prices = close_prices.loc[today]

                # Calculate the number of shares to buy or sell for each asset
                shares = (today_signals * portfolio['total'][yesterday]) / yesterday_prices

                # Calculate transaction costs
                transaction_costs = np.sum(self.transaction_cost * np.abs(shares * today_prices))

                # Calculate the value of the holdings today
                holdings_value = np.sum(shares * today_prices)

                # Update the portfolio
                portfolio.loc[today, 'holdings'] = holdings_value
                portfolio.loc[today, 'cash'] = portfolio['cash'][yesterday] - transaction_costs
                portfolio.loc[today, 'cash'] = portfolio.loc[today, 'cash'] + (portfolio['holdings'][yesterday] if i > 1 else 0) - holdings_value
                portfolio.loc[today, 'total'] = portfolio['cash'][today] + portfolio['holdings'][today]

                # Calculate returns
                portfolio.loc[today, 'returns'] = (portfolio['total'][today] - portfolio['total'][yesterday]) / portfolio['total'][yesterday]

            self.portfolio = portfolio
            logger.info("Backtest completed successfully.")
            return portfolio

        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    # Example Usage
    import yfinance as yf

    # Download sample data
    tickers = ["AAPL", "MSFT"]
    start_date = "2023-01-01"
    end_date = "2023-01-31"
    data = yf.download(tickers, start=start_date, end=end_date)

    if data is not None and not data.empty:
        # Create dummy signals (replace with your strategy's signals)
        signals = pd.DataFrame(0, index=data.index, columns=data['Close'].columns)
        signals.iloc[::5, 0] = 1  # Long AAPL every 5 days
        signals.iloc[::7, 1] = -1  # Short MSFT every 7 days

        # Stack the data to create a MultiIndex
        data = data.stack().unstack(level=1)

        # Initialize the backtester
        backtester = Backtester(data, signals)

        # Run the backtest
        portfolio = backtester.run_backtest()

        # Print the first few rows of the portfolio
        print("Portfolio:")
        print(portfolio.head())
    else:
        print("Failed to download data.")