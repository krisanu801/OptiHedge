import pytest
import pandas as pd
import numpy as np
import logging

# Configure logging (if not already configured)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the backtester module (adjust path if needed)
try:
    from src.backtesting.backtester import Backtester
except ImportError as e:
    logger.error(f"Error importing backtester: {e}.  Make sure your project structure is correct and the module is importable.")
    raise


def create_sample_data(dates, prices):
    """Helper function to create sample data for testing."""
    data = pd.DataFrame({'Close': prices}, index=pd.to_datetime(dates))
    data = data.stack().unstack(level=1) # Create multi-index
    return data

def create_sample_signals(dates, signals):
    """Helper function to create sample signals for testing."""
    return pd.DataFrame(signals, index=pd.to_datetime(dates))


def test_backtester_initialization():
    """Test the initialization of the Backtester class."""
    data = create_sample_data(['2023-01-01', '2023-01-02'], [100, 101])
    signals = create_sample_signals(['2023-01-01', '2023-01-02'], [1, -1])
    backtester = Backtester(data, signals, initial_capital=100000, transaction_cost=0.001)

    assert backtester.data is data
    assert backtester.signals is signals
    assert backtester.initial_capital == 100000
    assert backtester.transaction_cost == 0.001
    assert isinstance(backtester.portfolio, pd.DataFrame)
    assert backtester.portfolio.empty


def test_backtester_run_backtest_success():
    """Test a successful backtest run."""
    dates = ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04']
    prices = [100, 101, 102, 103]
    signals_values = [1, -1, 1, 0]

    data = create_sample_data(dates, prices)
    signals = create_sample_signals(dates, signals_values)

    backtester = Backtester(data, signals, initial_capital=100000, transaction_cost=0.001)
    portfolio = backtester.run_backtest()

    assert isinstance(portfolio, pd.DataFrame)
    assert not portfolio.empty
    assert 'holdings' in portfolio.columns
    assert 'cash' in portfolio.columns
    assert 'total' in portfolio.columns
    assert 'returns' in portfolio.columns
    assert portfolio['total'].iloc[0] == 100000  # Initial capital


def test_backtester_run_backtest_empty_signals():
    """Test backtest run with empty signals."""
    data = create_sample_data(['2023-01-01', '2023-01-02'], [100, 101])
    signals = pd.DataFrame()  # Empty signals
    backtester = Backtester(data, signals, initial_capital=100000, transaction_cost=0.001)
    portfolio = backtester.run_backtest()

    assert isinstance(portfolio, pd.DataFrame)
    assert portfolio.empty


def test_backtester_run_backtest_empty_data():
    """Test backtest run with empty data."""
    data = pd.DataFrame()  # Empty data
    signals = create_sample_signals(['2023-01-01', '2023-01-02'], [1, -1])
    backtester = Backtester(data, signals, initial_capital=100000, transaction_cost=0.001)
    portfolio = backtester.run_backtest()

    assert isinstance(portfolio, pd.DataFrame)
    assert portfolio.empty


def test_backtester_run_backtest_uneven_data_signals():
    """Test backtest run with data and signals having different lengths."""
    data = create_sample_data(['2023-01-01', '2023-01-02', '2023-01-03'], [100, 101, 102])
    signals = create_sample_signals(['2023-01-01', '2023-01-02'], [1, -1])
    backtester = Backtester(data, signals, initial_capital=100000, transaction_cost=0.001)
    portfolio = backtester.run_backtest()

    assert isinstance(portfolio, pd.DataFrame)
    assert not portfolio.empty
    assert len(portfolio) == len(data) # Portfolio should have the same length as the data