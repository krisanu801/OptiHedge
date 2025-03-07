import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Optional

# Configure logging
logger = logging.getLogger(__name__)


class PairsTradingStrategy:
    """
    Implements a Pairs Trading strategy.

    This strategy identifies correlated stock pairs and generates trading signals based on the
    statistical arbitrage opportunity when the spread between the two stocks deviates from its mean.
    """

    def __init__(self, data: pd.DataFrame, lookback_period: int = 20, entry_threshold: float = 2.0, exit_threshold: float = 0.5) -> None:
        """
        Initializes the PairsTradingStrategy.

        Args:
            data (pd.DataFrame): A Pandas DataFrame containing historical stock data.
                                  The DataFrame should have a MultiIndex with tickers as the first level
                                  and columns like 'Close' as the second level.
            lookback_period (int): The period for calculating the rolling mean and standard deviation of the spread.
            entry_threshold (float): The number of standard deviations away from the mean to enter a trade.
            exit_threshold (float): The number of standard deviations away from the mean to exit a trade.
        """
        self.data = data
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.signals: pd.DataFrame = pd.DataFrame()  # Initialize signals DataFrame
        self.pair: Optional[Tuple[str, str]] = None

    def find_correlated_pair(self, tickers: List[str]) -> Optional[Tuple[str, str]]:
        """
        Finds the most correlated pair of stocks from a list of tickers.

        Args:
            tickers (List[str]): A list of stock tickers.

        Returns:
            Optional[Tuple[str, str]]: A tuple containing the tickers of the most correlated pair,
                                      or None if no suitable pair is found.
        """
        try:
            close_prices = self.data.loc[:, (slice(None), 'Close')]
            if close_prices.empty:
                logger.warning("No 'Close' prices found in the data. Cannot find correlated pair.")
                return None

            # Calculate correlation matrix
            correlation_matrix = close_prices.corr()

            # Exclude self-correlations and duplicates
            correlation_matrix = correlation_matrix.mask(np.tril(np.ones(correlation_matrix.shape)).astype(bool))

            # Find the pair with the highest correlation
            max_correlation = correlation_matrix.max().max()
            if np.isnan(max_correlation):
                logger.warning("No correlated pair found.")
                return None

            stock1, stock2 = np.unravel_index(correlation_matrix.to_numpy().argmax(), correlation_matrix.shape)
            pair = (correlation_matrix.columns[stock1], correlation_matrix.columns[stock2])

            logger.info(f"Found correlated pair: {pair} with correlation {max_correlation}")
            return pair

        except Exception as e:
            logger.error(f"Error finding correlated pair: {e}")
            return None

    def generate_signals(self) -> pd.DataFrame:
        """
        Generates trading signals based on the Pairs Trading strategy.

        Returns:
            pd.DataFrame: A Pandas DataFrame containing the trading signals.
                           Each column represents a ticker.
                           1 indicates a long position, -1 indicates a short position, and 0 indicates no position.
        """
        try:
            close_prices = self.data.loc[:, (slice(None), 'Close')]
            if close_prices.empty:
                logger.warning("No 'Close' prices found in the data. Returning empty signals.")
                return pd.DataFrame()

            # Find a correlated pair if not already found
            if self.pair is None:
                tickers = close_prices.columns.get_level_values(0).unique().tolist()
                self.pair = self.find_correlated_pair(tickers)

            if self.pair is None:
                logger.warning("No correlated pair found. Cannot generate signals.")
                return pd.DataFrame()

            stock1, stock2 = self.pair
            price1 = close_prices[stock1]['Close']
            price2 = close_prices[stock2]['Close']

            # Calculate the spread
            spread = price1 - price2

            # Calculate rolling mean and standard deviation of the spread
            rolling_mean = spread.rolling(window=self.lookback_period).mean()
            rolling_std = spread.rolling(window=self.lookback_period).std()

            # Generate signals based on the spread's deviation from the mean
            signals = pd.DataFrame(index=close_prices.index, columns=close_prices.columns)
            signals[:] = 0  # Initialize all signals to 0

            # Long stock1, short stock2 when spread is above entry_threshold
            signals.loc[spread > rolling_mean + self.entry_threshold * rolling_std, (stock1, 'Close')] = 1
            signals.loc[spread > rolling_mean + self.entry_threshold * rolling_std, (stock2, 'Close')] = -1

            # Short stock1, long stock2 when spread is below -entry_threshold
            signals.loc[spread < rolling_mean - self.entry_threshold * rolling_std, (stock1, 'Close')] = -1
            signals.loc[spread < rolling_mean - self.entry_threshold * rolling_std, (stock2, 'Close')] = 1

            # Exit positions when spread is within exit_threshold
            signals.loc[spread < rolling_mean + self.exit_threshold * rolling_std, (stock1, 'Close')] = 0
            signals.loc[spread < rolling_mean + self.exit_threshold * rolling_std, (stock2, 'Close')] = 0

            signals.loc[spread > rolling_mean - self.exit_threshold * rolling_std, (stock1, 'Close')] = 0
            signals.loc[spread > rolling_mean - self.exit_threshold * rolling_std, (stock2, 'Close')] = 0

            self.signals = signals  # Store the generated signals
            logger.info(f"Pairs trading signals generated successfully for pair: {self.pair}")
            return signals

        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return pd.DataFrame()  # Return an empty DataFrame in case of error


if __name__ == "__main__":
    # Example Usage
    import yfinance as yf

    # Download sample data
    tickers = ["AAPL", "MSFT", "GOOG"]
    start_date = "2023-01-01"
    end_date = "2023-01-31"
    data = yf.download(tickers, start=start_date, end=end_date)

    if data is not None and not data.empty:
        # Initialize the strategy
        strategy = PairsTradingStrategy(data)

        # Generate signals
        signals = strategy.generate_signals()

        # Print the first few rows of the signals
        print("Trading Signals:")
        print(signals.head())
    else:
        print("Failed to download data.")