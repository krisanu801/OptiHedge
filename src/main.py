import sys
import os
import logging
from typing import List

# Dynamically adjust sys.path to allow imports from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Local imports
try:
    from src.data.data_handler import fetch_data  # type: ignore
    from src.strategies.strategy_long_short import LongShortEquityStrategy  # type: ignore
    from src.backtesting.backtester import Backtester  # type: ignore
    from src.performance.performance_metrics import calculate_performance_metrics  # type: ignore
    from configs.config import Config  # type: ignore
    from configs.logging_config import setup_logging  # type: ignore
except ImportError as e:
    print(f"ImportError: {e}")
    print("Please ensure your project structure is correct and dependencies are installed.")
    sys.exit(1)


# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)


def main() -> None:
    """
    Main function to execute the algorithmic trading simulation.
    Fetches data, applies a trading strategy, backtests, and evaluates performance.
    """
    try:
        # Load configuration
        config = Config()
        tickers: List[str] = config.tickers
        start_date: str = config.start_date
        end_date: str = config.end_date
        initial_capital: float = config.initial_capital

        # Fetch data
        data = fetch_data(tickers, start_date, end_date)
        if data is None or data.empty:
            logger.error("Failed to fetch data. Exiting.")
            return

        # Initialize strategy
        strategy = LongShortEquityStrategy(data)

        # Generate signals
        signals = strategy.generate_signals()

        # Initialize backtester
        backtester = Backtester(data, signals, initial_capital)

        # Run backtest
        portfolio = backtester.run_backtest()

        # Evaluate performance
        metrics = calculate_performance_metrics(portfolio)

        # Log results
        logger.info(f"Backtesting completed. Performance metrics: {metrics}")

        # Example usage: Print the portfolio
        print(portfolio.head())

    except Exception as e:
        logger.exception(f"An error occurred during execution: {e}")


if __name__ == "__main__":
    main()

    # Example Usage:
    # 1. Run the script: python src/main.py
    # 2. Check the logs in the logs/ directory for detailed information.
    # 3. Modify the tickers, start_date, and end_date in configs/config.py to test different scenarios.