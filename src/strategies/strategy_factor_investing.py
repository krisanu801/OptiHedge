import pandas as pd
import numpy as np
import logging
from typing import List, Dict

# Configure logging
logger = logging.getLogger(__name__)


class FactorInvestingStrategy:
    """
    Implements a Factor Investing strategy.

    This strategy ranks stocks based on a combination of factors (e.g., value, momentum, quality)
    and generates long positions in the top-ranked stocks.
    """

    def __init__(self, data: pd.DataFrame, factors: Dict[str, str], top_n: int = 10) -> None:
        """
        Initializes the FactorInvestingStrategy.

        Args:
            data (pd.DataFrame): A Pandas DataFrame containing historical stock data.
                                  The DataFrame should have a MultiIndex with tickers as the first level
                                  and columns like 'Close', 'Volume', etc., as the second level.
            factors (Dict[str, str]): A dictionary specifying the factors to use and their weights.
                                       Example: {'value': 'P/E', 'momentum': 'SMA_12'}
            top_n (int): The number of top-ranked stocks to include in the portfolio.
        """
        self.data = data
        self.factors = factors
        self.top_n = top_n
        self.signals: pd.DataFrame = pd.DataFrame()  # Initialize signals DataFrame

    def calculate_factor_rankings(self) -> pd.DataFrame:
        """
        Calculates the rankings of stocks based on the specified factors.

        Returns:
            pd.DataFrame: A Pandas DataFrame containing the factor rankings for each stock.
        """
        try:
            factor_rankings = pd.DataFrame(index=self.data.index)

            for factor_name, factor_data_column in self.factors.items():
                # Assuming factor data is already present in self.data
                if (slice(None), factor_data_column) not in self.data.columns:
                    logger.warning(f"Factor data '{factor_data_column}' not found in the data. Skipping factor.")
                    continue

                factor_data = self.data.loc[:, (slice(None), factor_data_column)]

                # Handle missing data by filling with the mean
                factor_data = factor_data.fillna(factor_data.mean())

                # Rank the stocks based on the factor
                factor_rankings[factor_name] = factor_data.rank(axis=1, ascending=False).mean(axis=0)

            return factor_rankings

        except Exception as e:
            logger.error(f"Error calculating factor rankings: {e}")
            return pd.DataFrame()

    def generate_signals(self) -> pd.DataFrame:
        """
        Generates trading signals based on the Factor Investing strategy.

        Returns:
            pd.DataFrame: A Pandas DataFrame containing the trading signals.
                           1 indicates a long position, 0 indicates no position.
        """
        try:
            factor_rankings = self.calculate_factor_rankings()

            if factor_rankings.empty:
                logger.warning("No factor rankings calculated. Returning empty signals.")
                return pd.DataFrame()

            # Combine factor rankings into a single score
            combined_rankings = factor_rankings.mean(axis=1)

            # Select the top N stocks based on the combined rankings
            top_stocks = combined_rankings.nlargest(self.top_n).index.tolist()

            # Generate signals: 1 for top stocks, 0 for others
            signals = pd.DataFrame(0, index=self.data.index, columns=self.data.columns)
            for stock in top_stocks:
                signals.loc[:, (stock, 'Close')] = 1

            self.signals = signals  # Store the generated signals
            logger.info("Trading signals generated successfully.")
            return signals

        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    # Example Usage
    import yfinance as yf

    # Download sample data
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
    start_date = "2023-01-01"
    end_date = "2023-01-31"
    data = yf.download(tickers, start=start_date, end=end_date)

    if data is not None and not data.empty:
        # Example factors (replace with actual factor data)
        # For demonstration, we'll use closing prices as a proxy for a factor
        data = data.stack().unstack(level=1)
        data[('AAPL', 'SMA_12')] = data[('AAPL', 'Close')].rolling(window=12).mean()
        data[('MSFT', 'SMA_12')] = data[('MSFT', 'Close')].rolling(window=12).mean()
        data[('GOOG', 'SMA_12')] = data[('GOOG', 'Close')].rolling(window=12).mean()
        data[('AMZN', 'SMA_12')] = data[('AMZN', 'Close')].rolling(window=12).mean()
        data[('TSLA', 'SMA_12')] = data[('TSLA', 'Close')].rolling(window=12).mean()

        factors = {'momentum': 'SMA_12'}

        # Initialize the strategy
        strategy = FactorInvestingStrategy(data, factors, top_n=2)

        # Generate signals
        signals = strategy.generate_signals()

        # Print the first few rows of the signals
        print("Trading Signals:")
        print(signals.head())
    else:
        print("Failed to download data.")