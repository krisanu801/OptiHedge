import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
import logging

# Configure logging (if not already configured)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock yfinance to avoid actual API calls during testing
@pytest.fixture
def mock_yfinance_download():
    with patch("yfinance.download") as mock:
        yield mock

# Import the data_handler module (adjust path if needed)
try:
    from src.data.data_handler import fetch_data, preprocess_data
except ImportError as e:
    logger.error(f"Error importing data_handler: {e}.  Make sure your project structure is correct and the module is importable.")
    raise

def test_fetch_data_success(mock_yfinance_download):
    """Test successful data fetching from yfinance."""
    # Mock the return value of yfinance.download
    mock_yfinance_download.return_value = pd.DataFrame({'Close': [100, 101, 102]}, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']))
    tickers = ["AAPL"]
    start_date = "2023-01-01"
    end_date = "2023-01-03"

    data = fetch_data(tickers, start_date, end_date)

    assert data is not None
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert 'Close' in data.columns.get_level_values(1)
    assert tickers[0] in data.columns.get_level_values(0)

    mock_yfinance_download.assert_called_once_with(tickers, start=start_date, end=end_date)


def test_fetch_data_no_data(mock_yfinance_download):
    """Test data fetching when yfinance returns an empty DataFrame."""
    mock_yfinance_download.return_value = pd.DataFrame()
    tickers = ["INVALID_TICKER"]
    start_date = "2023-01-01"
    end_date = "2023-01-03"

    data = fetch_data(tickers, start_date, end_date)

    assert data is None
    mock_yfinance_download.assert_called_once_with(tickers, start=start_date, end=end_date)


def test_fetch_data_error(mock_yfinance_download):
    """Test data fetching when yfinance raises an exception."""
    mock_yfinance_download.side_effect = Exception("Yfinance error")
    tickers = ["AAPL"]
    start_date = "2023-01-01"
    end_date = "2023-01-03"

    data = fetch_data(tickers, start_date, end_date)

    assert data is None
    mock_yfinance_download.assert_called_once_with(tickers, start=start_date, end=end_date)


def test_preprocess_data_success():
    """Test successful data preprocessing."""
    data = pd.DataFrame({'Close': [100, np.nan, 102, np.nan]}, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04']))
    expected_data = pd.DataFrame({'Close': [100.0, 100.0, 102.0, 102.0]}, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04']))

    processed_data = preprocess_data(data)

    assert isinstance(processed_data, pd.DataFrame)
    assert not processed_data.empty
    pd.testing.assert_frame_equal(processed_data, expected_data)


def test_preprocess_data_empty():
    """Test data preprocessing with an empty DataFrame."""
    data = pd.DataFrame()

    processed_data = preprocess_data(data)

    assert isinstance(processed_data, pd.DataFrame)
    assert processed_data.empty


def test_preprocess_data_no_nan():
    """Test data preprocessing with no NaN values."""
    data = pd.DataFrame({'Close': [100, 101, 102]}, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']))

    processed_data = preprocess_data(data)

    assert isinstance(processed_data, pd.DataFrame)
    assert not processed_data.empty
    pd.testing.assert_frame_equal(processed_data, data)