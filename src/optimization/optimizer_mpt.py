import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Optional
from scipy.optimize import minimize

# Configure logging
logger = logging.getLogger(__name__)


class MPTOptimizer:
    """
    Implements Modern Portfolio Theory (MPT) optimization to find the optimal portfolio weights
    that maximize the Sharpe ratio.
    """

    def __init__(self, data: pd.DataFrame, risk_free_rate: float = 0.005) -> None:
        """
        Initializes the MPTOptimizer.

        Args:
            data (pd.DataFrame): A Pandas DataFrame containing historical stock data (e.g., daily returns).
                                  The DataFrame should have a MultiIndex with tickers as the first level
                                  and columns like 'Close' as the second level.
            risk_free_rate (float): The risk-free rate of return (annualized).
        """
        self.data = data
        self.risk_free_rate = risk_free_rate
        self.returns: pd.DataFrame = pd.DataFrame()

    def calculate_returns(self) -> pd.DataFrame:
        """
        Calculates the daily returns for each stock.

        Returns:
            pd.DataFrame: A Pandas DataFrame containing the daily returns.
        """
        try:
            close_prices = self.data.loc[:, (slice(None), 'Close')]
            if close_prices.empty:
                logger.warning("No 'Close' prices found in the data. Cannot calculate returns.")
                return pd.DataFrame()

            # Calculate daily returns
            returns = close_prices.pct_change().dropna()
            self.returns = returns
            logger.info("Daily returns calculated successfully.")
            return returns

        except Exception as e:
            logger.error(f"Error calculating returns: {e}")
            return pd.DataFrame()

    def portfolio_performance(self, weights: np.ndarray) -> Tuple[float, float]:
        """
        Calculates the portfolio return and volatility for a given set of weights.

        Args:
            weights (np.ndarray): A numpy array containing the portfolio weights.

        Returns:
            Tuple[float, float]: A tuple containing the portfolio return and volatility.
        """
        try:
            if self.returns.empty:
                logger.warning("Returns data is empty. Please calculate returns first.")
                return 0.0, 0.0

            # Calculate portfolio return
            portfolio_return = np.sum(self.returns.mean() * weights) * 252  # Annualize

            # Calculate portfolio volatility
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.returns.cov() * 252, weights)))  # Annualize

            return portfolio_return, portfolio_std

        except Exception as e:
            logger.error(f"Error calculating portfolio performance: {e}")
            return 0.0, 0.0

    def negative_sharpe_ratio(self, weights: np.ndarray) -> float:
        """
        Calculates the negative Sharpe ratio for a given set of weights.

        Args:
            weights (np.ndarray): A numpy array containing the portfolio weights.

        Returns:
            float: The negative Sharpe ratio.
        """
        try:
            portfolio_return, portfolio_std = self.portfolio_performance(weights)
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std
            return -sharpe_ratio  # Minimize the negative Sharpe ratio to maximize the Sharpe ratio

        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0

    def optimize_portfolio(self) -> Optional[np.ndarray]:
        """
        Optimizes the portfolio weights to maximize the Sharpe ratio.

        Returns:
            Optional[np.ndarray]: A numpy array containing the optimal portfolio weights, or None if optimization fails.
        """
        try:
            returns = self.calculate_returns()
            if returns.empty:
                logger.warning("No returns data available. Cannot optimize portfolio.")
                return None

            num_assets = len(returns.columns)
            # Initial guess: equal weights
            initial_weights = np.array([1/num_assets] * num_assets)

            # Constraints: weights must sum to 1
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

            # Bounds: weights must be between 0 and 1
            bounds = tuple((0, 1) for asset in range(num_assets))

            # Optimization
            result = minimize(self.negative_sharpe_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

            if result.success:
                optimal_weights = result.x
                logger.info("Portfolio optimization completed successfully.")
                return optimal_weights
            else:
                logger.warning(f"Portfolio optimization failed: {result.message}")
                return None

        except Exception as e:
            logger.error(f"Error optimizing portfolio: {e}")
            return None


if __name__ == "__main__":
    # Example Usage
    import yfinance as yf

    # Download sample data
    tickers = ["AAPL", "MSFT", "GOOG"]
    start_date = "2023-01-01"
    end_date = "2023-01-31"
    data = yf.download(tickers, start=start_date, end=end_date)

    if data is not None and not data.empty:
        # Initialize the optimizer
        optimizer = MPTOptimizer(data)

        # Optimize the portfolio
        optimal_weights = optimizer.optimize_portfolio()

        if optimal_weights is not None:
            # Print the optimal weights
            print("Optimal Portfolio Weights:")
            for i, ticker in enumerate(tickers):
                print(f"{ticker}: {optimal_weights[i]:.4f}")
        else:
            print("Portfolio optimization failed.")
    else:
        print("Failed to download data.")