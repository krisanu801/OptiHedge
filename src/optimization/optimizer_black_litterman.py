import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional
from scipy.linalg import inv

# Configure logging
logger = logging.getLogger(__name__)


class BlackLittermanOptimizer:
    """
    Implements the Black-Litterman model to incorporate investor views into portfolio optimization.
    """

    def __init__(self, data: pd.DataFrame, risk_free_rate: float = 0.005, tau: float = 0.025, risk_aversion: float = 2.5) -> None:
        """
        Initializes the BlackLittermanOptimizer.

        Args:
            data (pd.DataFrame): A Pandas DataFrame containing historical stock data (e.g., daily returns).
                                  The DataFrame should have a MultiIndex with tickers as the first level
                                  and columns like 'Close' as the second level.
            risk_free_rate (float): The risk-free rate of return (annualized).
            tau (float): A scalar representing the uncertainty in the prior equilibrium returns.
            risk_aversion (float): The investor's risk aversion coefficient.
        """
        self.data = data
        self.risk_free_rate = risk_free_rate
        self.tau = tau
        self.risk_aversion = risk_aversion
        self.returns: pd.DataFrame = pd.DataFrame()
        self.covariance_matrix: pd.DataFrame = pd.DataFrame()
        self.equilibrium_returns: pd.Series = pd.Series()

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

    def calculate_covariance_matrix(self) -> pd.DataFrame:
        """
        Calculates the covariance matrix of the asset returns.

        Returns:
            pd.DataFrame: A Pandas DataFrame representing the covariance matrix.
        """
        try:
            if self.returns.empty:
                logger.warning("Returns data is empty. Please calculate returns first.")
                return pd.DataFrame()

            covariance_matrix = self.returns.cov() * 252  # Annualize
            self.covariance_matrix = covariance_matrix
            logger.info("Covariance matrix calculated successfully.")
            return covariance_matrix

        except Exception as e:
            logger.error(f"Error calculating covariance matrix: {e}")
            return pd.DataFrame()

    def calculate_equilibrium_returns(self) -> pd.Series:
        """
        Calculates the equilibrium (implied) returns using the reverse optimization approach.

        Returns:
            pd.Series: A Pandas Series representing the equilibrium returns for each asset.
        """
        try:
            if self.covariance_matrix.empty:
                logger.warning("Covariance matrix is empty. Please calculate covariance matrix first.")
                return pd.Series()

            # Assume equal weights for the market portfolio
            weights = pd.Series([1 / len(self.covariance_matrix.columns)] * len(self.covariance_matrix.columns), index=self.covariance_matrix.columns)

            # Calculate equilibrium returns
            equilibrium_returns = self.risk_aversion * self.covariance_matrix.dot(weights)
            self.equilibrium_returns = equilibrium_returns
            logger.info("Equilibrium returns calculated successfully.")
            return equilibrium_returns

        except Exception as e:
            logger.error(f"Error calculating equilibrium returns: {e}")
            return pd.Series()

    def incorporate_views(self, views: Dict[str, float], view_confidences: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Incorporates investor views into the equilibrium returns using the Black-Litterman formula.

        Args:
            views (Dict[str, float]): A dictionary representing the investor's views on asset returns.
                                       Keys are asset tickers, and values are the expected returns.
            view_confidences (Dict[str, float]): A dictionary representing the confidence in each view.
                                                 Keys are asset tickers, and values are the confidence levels (0 to 1).

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the updated (posterior) expected returns and covariance matrix.
        """
        try:
            if self.equilibrium_returns.empty:
                logger.warning("Equilibrium returns are empty. Please calculate equilibrium returns first.")
                return np.array([]), np.array([])

            # Number of assets
            n = len(self.equilibrium_returns)

            # Number of views
            k = len(views)

            # Create the view matrix (Q) and the view constraint matrix (P)
            Q = np.array(list(views.values()))
            P = np.zeros((k, n))
            asset_names = list(views.keys())
            for i, asset in enumerate(asset_names):
                asset_index = self.equilibrium_returns.index.get_loc(asset)
                P[i, asset_index] = 1

            # Create the view uncertainty matrix (Omega)
            Omega = np.diag(list(view_confidences.values()))

            # Calculate the Black-Litterman posterior estimate of the returns
            Sigma = self.covariance_matrix.to_numpy()
            tau = self.tau
            risk_aversion = self.risk_aversion

            # Black-Litterman formula
            BL_returns = np.linalg.inv(np.linalg.inv(tau * Sigma) + P.T @ np.linalg.inv(Omega) @ P) @ (np.linalg.inv(tau * Sigma) @ self.equilibrium_returns.to_numpy() + P.T @ np.linalg.inv(Omega) @ Q)
            BL_covariance = Sigma + tau * Sigma - tau * Sigma @ P.T @ np.linalg.inv(P @ tau * Sigma @ P.T + Omega) @ P @ tau * Sigma

            logger.info("Investor views incorporated successfully.")
            return BL_returns, BL_covariance

        except Exception as e:
            logger.error(f"Error incorporating views: {e}")
            return np.array([]), np.array([])

    def optimize_portfolio(self, views: Dict[str, float], view_confidences: Dict[str, float]) -> Optional[np.ndarray]:
        """
        Optimizes the portfolio weights using the Black-Litterman model.

        Args:
            views (Dict[str, float]): A dictionary representing the investor's views on asset returns.
                                       Keys are asset tickers, and values are the expected returns.
            view_confidences (Dict[str, float]): A dictionary representing the confidence in each view.
                                                 Keys are asset tickers, and values are the confidence levels (0 to 1).

        Returns:
            Optional[np.ndarray]: A numpy array containing the optimal portfolio weights, or None if optimization fails.
        """
        try:
            # Calculate returns, covariance matrix, and equilibrium returns
            self.calculate_returns()
            self.calculate_covariance_matrix()
            self.calculate_equilibrium_returns()

            # Incorporate investor views
            BL_returns, BL_covariance = self.incorporate_views(views, view_confidences)

            if BL_returns.size == 0 or BL_covariance.size == 0:
                logger.warning("Failed to incorporate views. Cannot optimize portfolio.")
                return None

            # Calculate optimal weights using the Black-Litterman returns and covariance
            weights = np.linalg.inv(self.risk_aversion * BL_covariance) @ BL_returns

            # Normalize weights to sum to 1
            weights = weights / np.sum(weights)

            logger.info("Portfolio optimization completed successfully using Black-Litterman model.")
            return weights

        except Exception as e:
            logger.error(f"Error optimizing portfolio using Black-Litterman model: {e}")
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
        optimizer = BlackLittermanOptimizer(data)

        # Define investor views and confidences
        views = {"AAPL": 0.001, "MSFT": 0.0015}  # Example: Investor expects AAPL to return 0.1% and MSFT to return 0.15% daily
        view_confidences = {"AAPL": 0.5, "MSFT": 0.7}  # Example: Investor is 50% confident in AAPL view and 70% confident in MSFT view

        # Optimize the portfolio
        optimal_weights = optimizer.optimize_portfolio(views, view_confidences)

        if optimal_weights is not None:
            # Print the optimal weights
            print("Optimal Portfolio Weights (Black-Litterman):")
            for i, ticker in enumerate(tickers):
                print(f"{ticker}: {optimal_weights[i]:.4f}")
        else:
            print("Portfolio optimization failed.")
    else:
        print("Failed to download data.")