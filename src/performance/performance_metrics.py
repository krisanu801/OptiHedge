import pandas as pd
import numpy as np
import logging
from typing import Dict

# Configure logging
logger = logging.getLogger(__name__)


def calculate_performance_metrics(portfolio: pd.DataFrame, risk_free_rate: float = 0.005) -> Dict[str, float]:
    """
    Calculates performance metrics for a given portfolio.

    Args:
        portfolio (pd.DataFrame): A Pandas DataFrame containing the portfolio value over time.
                                  The DataFrame should have a 'returns' column representing the daily returns.
        risk_free_rate (float): The risk-free rate of return (annualized).

    Returns:
        Dict[str, float]: A dictionary containing the calculated performance metrics.
    """
    try:
        if portfolio.empty:
            logger.warning("Portfolio data is empty. Cannot calculate performance metrics.")
            return {}

        if 'returns' not in portfolio.columns:
            logger.error("The 'returns' column is missing from the portfolio DataFrame.")
            return {}

        # Calculate total returns
        total_return = (portfolio['total'].iloc[-1] / portfolio['total'].iloc[0]) - 1

        # Calculate annualized return (CAGR)
        years = len(portfolio) / 252  # Assuming 252 trading days per year
        cagr = (1 + total_return)**(1/years) - 1

        # Calculate daily returns
        daily_returns = portfolio['returns']

        # Calculate volatility (annualized)
        volatility = daily_returns.std() * np.sqrt(252)

        # Calculate Sharpe ratio
        sharpe_ratio = (cagr - risk_free_rate) / volatility if volatility > 0 else 0

        # Calculate downside deviation
        downside_returns = daily_returns[daily_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)

        # Calculate Sortino ratio
        sortino_ratio = (cagr - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0

        # Calculate maximum drawdown
        peak = portfolio['total'].cummax()
        drawdown = (portfolio['total'] - peak) / peak
        max_drawdown = drawdown.min()

        # Calculate alpha and beta (requires a benchmark, not implemented here)
        # This would involve comparing the portfolio's returns to a benchmark index (e.g., S&P 500)

        metrics = {
            'Total Return': total_return,
            'CAGR': cagr,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Max Drawdown': max_drawdown,
            # 'Alpha': None,  # Requires benchmark data
            # 'Beta': None   # Requires benchmark data
        }

        logger.info("Performance metrics calculated successfully.")
        return metrics

    except Exception as e:
        logger.error(f"Error calculating performance metrics: {e}")
        return {}


if __name__ == "__main__":
    # Example Usage
    # Create a sample portfolio DataFrame
    data = {
        'total': [100000, 101000, 102000, 101500, 103000],
        'returns': [0.0, 0.01, 0.0099, -0.0049, 0.0148]
    }
    portfolio = pd.DataFrame(data)

    # Calculate performance metrics
    metrics = calculate_performance_metrics(portfolio)

    # Print the metrics
    print("Performance Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")