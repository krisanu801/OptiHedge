import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Union

# Configure logging
logger = logging.getLogger(__name__)


class LongShortEquityStrategy:
    """
    Implements a Long/Short Equity trading strategy.

    This strategy identifies overbought and oversold stocks based on a simple moving average (SMA)
    and generates long signals for oversold stocks and short signals for overbought stocks.
    """

    def __init__(self, data: pd.DataFrame, sma_period: int = 20) -> None:
        """
        Initializes the LongShortEquityStrategy.

        Args:
            data (pd.DataFrame): A Pandas DataFrame containing historical stock data.
                                  The DataFrame should have a MultiIndex with tickers as the first level
                                  and columns like 'Close' as the second level.
            sma_period (int): The period for calculating the Simple Moving Average (SMA).
        """
        self.data = data
        self.sma_period = sma_period
        self.signals: pd.DataFrame = pd.DataFrame()  # Initialize signals DataFrame

    def generate_signals(self) -> pd.DataFrame:
        """
        Generates trading signals based on the Long/Short Equity strategy.

        Returns:
            pd.DataFrame: A Pandas DataFrame containing the trading signals.
                           1 indicates a long position, -1 indicates a short position, and 0 indicates no position.
        """
        try:
            close_prices = self.data.loc[:, (slice(None), 'Close')]  # Select all tickers and 'Close' prices
            if close_prices.empty:
                logger.warning("No 'Close' prices found in the data. Returning empty signals.")
                return pd.DataFrame()

            # Calculate Simple Moving Average (SMA)
            sma = close_prices.rolling(window=self.sma_period).mean()

            # Calculate the deviation from the SMA
            deviation = close_prices - sma

            # Generate signals:
            # Long signal if the price is below the SMA (oversold)
            # Short signal if the price is above the SMA (overbought)
            signals = pd.DataFrame(np.where(deviation < 0, 1, np.where(deviation > 0, -1, 0)),
                                   index=close_prices.index, columns=close_prices.columns)

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
        strategy = LongShortEquityStrategy(data)

        # Generate signals
        signals = strategy.generate_signals()

        # Print the first few rows of the signals
        print("Trading Signals:")
        print(signals.head())
    else:
        print("Failed to download data.")