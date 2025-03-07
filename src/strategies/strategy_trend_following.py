import pandas as pd
import numpy as np
import logging
from typing import List

# Configure logging
logger = logging.getLogger(__name__)


class TrendFollowingStrategy:
    """
    Implements a Trend Following strategy.

    This strategy identifies trends using a moving average crossover and generates
    long or short signals based on the crossover direction.
    """

    def __init__(self, data: pd.DataFrame, short_window: int = 20, long_window: int = 50) -> None:
        """
        Initializes the TrendFollowingStrategy.

        Args:
            data (pd.DataFrame): A Pandas DataFrame containing historical stock data.
                                  The DataFrame should have a MultiIndex with tickers as the first level
                                  and columns like 'Close' as the second level.
            short_window (int): The period for the short-term moving average.
            long_window (int): The period for the long-term moving average.
        """
        self.data = data
        self.short_window = short_window
        self.long_window = long_window
        self.signals: pd.DataFrame = pd.DataFrame()  # Initialize signals DataFrame

    def generate_signals(self) -> pd.DataFrame:
        """
        Generates trading signals based on the Trend Following strategy.

        Returns:
            pd.DataFrame: A Pandas DataFrame containing the trading signals.
                           1 indicates a long position, -1 indicates a short position, and 0 indicates no position.
        """
        try:
            close_prices = self.data.loc[:, (slice(None), 'Close')]
            if close_prices.empty:
                logger.warning("No 'Close' prices found in the data. Returning empty signals.")
                return pd.DataFrame()

            # Calculate short-term and long-term moving averages
            short_mavg = close_prices.rolling(window=self.short_window).mean()
            long_mavg = close_prices.rolling(window=self.long_window).mean()

            # Generate signals:
            # Long signal when short-term MA crosses above long-term MA
            # Short signal when short-term MA crosses below long-term MA
            signals = pd.DataFrame(0, index=close_prices.index, columns=close_prices.columns)

            # Initialize a DataFrame to store the crossover events
            crossover = pd.DataFrame(0, index=close_prices.index, columns=close_prices.columns)
            crossover[short_mavg > long_mavg] = 1
            crossover[short_mavg <= long_mavg] = -1

            # Detect the crossover points (where the sign changes)
            positions = crossover.diff()
            signals[positions == 2] = 1  # Short MA crosses above Long MA (Buy)
            signals[positions == -2] = -1  # Short MA crosses below Long MA (Sell)

            # Carry over the previous signal until a new signal is generated
            signals = signals.replace(0, np.nan).ffill().fillna(0)

            self.signals = signals  # Store the generated signals
            logger.info("Trading signals generated successfully.")
            return signals

        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return pd.DataFrame()  # Return an empty DataFrame in case of error


if __name__ == "__main__":
    # Example Usage
    import yfinance as yf

    # Download sample data
    tickers = ["AAPL", "MSFT"]
    start_date = "2023-01-01"
    end_date = "2023-01-31"
    data = yf.download(tickers, start=start_date, end=end_date)

    if data is not None and not data.empty:
        # Initialize the strategy
        strategy = TrendFollowingStrategy(data)

        # Generate signals
        signals = strategy.generate_signals()

        # Print the first few rows of the signals
        print("Trading Signals:")
        print(signals.head())
    else:
        print("Failed to download data.")