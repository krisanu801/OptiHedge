import pytest
import pandas as pd
import numpy as np
import logging

# Configure logging (if not already configured)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the performance_metrics module (adjust path if needed)
try:
    from src.performance.performance_metrics import calculate_performance_metrics
except ImportError as e:
    logger.error(f"Error importing performance_metrics: {e}.  Make sure your project structure is correct and the module is importable.")
    raise


def create_sample_portfolio(total_values, returns_values):
    """Helper function to create a sample portfolio DataFrame for testing."""
    data = {'total': total_values, 'returns': returns_values}
    return pd.DataFrame(data)


def test_calculate_performance_metrics_success():
    """Test successful calculation of performance metrics."""
    portfolio = create_sample_portfolio(
        total_values=[100000, 101000, 102000, 101500, 103000],
        returns_values=[0.0, 0.01, 0.0099, -0.0049, 0.0148]
    )
    metrics = calculate_performance_metrics(portfolio, risk_free_rate=0.005)

    assert isinstance(metrics, dict)
    assert 'Total Return' in metrics
    assert 'CAGR' in metrics
    assert 'Volatility' in metrics
    assert 'Sharpe Ratio' in metrics
    assert 'Sortino Ratio' in metrics
    assert 'Max Drawdown' in metrics

    assert np.isclose(metrics['Total Return'], 0.03)
    assert np.isclose(metrics['CAGR'], 0.1520200507422633)
    assert np.isclose(metrics['Volatility'], 0.1555122447285766)
    assert np.isclose(metrics['Sharpe Ratio'], 0.945529111244274)
    assert np.isclose(metrics['Sortino Ratio'], 1.741678428958547)
    assert np.isclose(metrics['Max Drawdown'], -0.004901960784313771)


def test_calculate_performance_metrics_empty_portfolio():
    """Test calculation of performance metrics with an empty portfolio."""
    portfolio = pd.DataFrame()
    metrics = calculate_performance_metrics(portfolio, risk_free_rate=0.005)

    assert isinstance(metrics, dict)
    assert not metrics


def test_calculate_performance_metrics_missing_returns():
    """Test calculation of performance metrics with a portfolio missing the 'returns' column."""
    portfolio = pd.DataFrame({'total': [100000, 101000]})
    metrics = calculate_performance_metrics(portfolio, risk_free_rate=0.005)

    assert isinstance(metrics, dict)
    assert not metrics


def test_calculate_performance_metrics_zero_volatility():
    """Test calculation of performance metrics with zero volatility."""
    portfolio = create_sample_portfolio(
        total_values=[100000, 100000, 100000],
        returns_values=[0.0, 0.0, 0.0]
    )
    metrics = calculate_performance_metrics(portfolio, risk_free_rate=0.005)

    assert isinstance(metrics, dict)
    assert metrics['Sharpe Ratio'] == 0
    assert metrics['Sortino Ratio'] == 0