import os
import logging
from dotenv import load_dotenv
from typing import List

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


class Config:
    """
    Configuration class for the project.

    Loads configuration parameters from environment variables or sets default values.
    """

    def __init__(self) -> None:
        """
        Initializes the Config class.
        """
        try:
            # Define configuration parameters
            self.tickers: List[str] = os.getenv("TICKERS", "AAPL,MSFT").split(",")
            self.start_date: str = os.getenv("START_DATE", "2023-01-01")
            self.end_date: str = os.getenv("END_DATE", "2023-01-31")
            self.initial_capital: float = float(os.getenv("INITIAL_CAPITAL", "100000"))
            self.transaction_cost: float = float(os.getenv("TRANSACTION_COST", "0.001"))
            self.risk_free_rate: float = float(os.getenv("RISK_FREE_RATE", "0.005"))
            self.data_dir: str = os.getenv("DATA_DIR", "data")
            self.log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()

            # Validate configuration parameters
            self._validate_config()

            logger.info("Configuration loaded successfully.")

        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    def _validate_config(self) -> None:
        """
        Validates the configuration parameters.
        """
        try:
            # Validate tickers
            if not isinstance(self.tickers, list) or not all(isinstance(ticker, str) for ticker in self.tickers):
                raise ValueError("TICKERS must be a comma-separated string of ticker symbols.")

            # Validate start and end dates
            # Add more robust date validation if needed
            if not isinstance(self.start_date, str) or not isinstance(self.end_date, str):
                raise ValueError("START_DATE and END_DATE must be strings in YYYY-MM-DD format.")

            # Validate initial capital
            if not isinstance(self.initial_capital, float) or self.initial_capital <= 0:
                raise ValueError("INITIAL_CAPITAL must be a positive float.")

            # Validate transaction cost
            if not isinstance(self.transaction_cost, float) or not 0 <= self.transaction_cost <= 1:
                raise ValueError("TRANSACTION_COST must be a float between 0 and 1.")

            # Validate risk-free rate
            if not isinstance(self.risk_free_rate, float):
                raise ValueError("RISK_FREE_RATE must be a float.")

            # Validate data directory
            if not isinstance(self.data_dir, str):
                raise ValueError("DATA_DIR must be a string.")

            # Validate log level
            log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if self.log_level not in log_levels:
                raise ValueError(f"LOG_LEVEL must be one of: {', '.join(log_levels)}")

        except ValueError as e:
            logger.error(f"Invalid configuration parameter: {e}")
            raise


if __name__ == "__main__":
    # Example Usage
    # Create an instance of the Config class
    config = Config()

    # Access configuration parameters
    print("Tickers:", config.tickers)
    print("Start Date:", config.start_date)
    print("End Date:", config.end_date)
    print("Initial Capital:", config.initial_capital)
    print("Transaction Cost:", config.transaction_cost)
    print("Risk-Free Rate:", config.risk_free_rate)
    print("Data Directory:", config.data_dir)
    print("Log Level:", config.log_level)

    # To use this configuration:
    # 1. Create a .env file in the project root directory.
    # 2. Define the configuration parameters in the .env file, e.g.:
    #    TICKERS=AAPL,MSFT,GOOG
    #    START_DATE=2023-01-01
    # 3. The Config class will automatically load the parameters from the .env file.