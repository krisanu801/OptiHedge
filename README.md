# OptiHedge: AI-Driven Hedge Fund Simulator

## Project Description

OptiHedge is an AI-driven hedge fund simulator built using Python, focusing on quantitative finance and algorithmic trading strategies. The project leverages Yahoo Finance (yfinance) data to perform portfolio optimization, backtesting, and performance evaluation. It aims to simulate various trading strategies and provide a web dashboard for live tracking.

## Features

- **Data Collection:** Fetches historical price and fundamental data from Yahoo Finance.
- **Financial Metrics Calculation:** Calculates key financial metrics such as returns, volatility, Sharpe ratio, Sortino ratio, alpha, beta, and max drawdown.
- **Portfolio Optimization:** Implements Modern Portfolio Theory (MPT) and Black-Litterman model for portfolio optimization.
- **Trading Strategies:** Supports multiple trading strategies, including:
    - Long/Short Equity
    - Pairs Trading
    - Factor Investing
    - Trend-Following
- **Backtesting:** Simulates trades over historical data, avoiding lookahead bias and considering transaction costs.
- **Performance Evaluation:** Evaluates strategy performance using metrics like CAGR, Sharpe ratio, Sortino ratio, alpha, beta, and max drawdown.
- **AI-Driven Optimization:** Integrates a reinforcement learning model (DQN/PPO) to dynamically adjust portfolio allocations.
- **Web Dashboard:** Features a web dashboard (Streamlit/Dash) for live tracking of portfolio performance.

## Project Structure

```
OptiHedge/
├── src/
│   ├── data/
│   │   └── data_handler.py       # Handles data fetching and preprocessing
│   ├── strategies/
│   │   ├── strategy_long_short.py  # Implements Long/Short Equity strategy
│   │   ├── strategy_pairs_trading.py # Implements Pairs Trading strategy
│   │   ├── strategy_factor_investing.py # Implements Factor Investing strategy
│   │   └── strategy_trend_following.py # Implements Trend Following strategy
│   ├── optimization/
│   │   ├── optimizer_mpt.py        # Implements Modern Portfolio Theory (MPT) optimization
│   │   └── optimizer_black_litterman.py # Implements Black-Litterman model optimization
│   ├── backtesting/
│   │   └── backtester.py           # Handles backtesting of trading strategies
│   ├── performance/
│   │   └── performance_metrics.py  # Calculates performance metrics
│   ├── dashboard/
│   │   └── dashboard.py            # Creates a web dashboard
│   └── main.py                   # Main application entry point
├── configs/
│   ├── config.py               # Configuration file (API keys, data paths, etc.)
│   └── logging_config.py       # Configuration file for logging
├── tests/                      # Unit tests
├── data/                       # Data storage
├── logs/                       # Log files
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
├── setup.py                    # Setup file for packaging and distribution
└── .gitignore                  # Specifies intentionally untracked files that Git should ignore
```

## Setup Instructions

1.  **Clone the repository:**

    ```bash
    git clone [repository_url]
    cd OptiHedge
    ```

2.  **Create a virtual environment:**

    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**

    -   **Linux/macOS:**

        ```bash
        source venv/bin/activate
        ```

    -   **Windows:**

        ```bash
        venv\Scripts\activate
        ```

4.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

5.  **Configure the application:**

    -   Create a `.env` file in the project root directory.
    -   Define the necessary environment variables, such as:

        ```
        TICKERS=AAPL,MSFT
        START_DATE=2023-01-01
        END_DATE=2023-01-31
        INITIAL_CAPITAL=100000
        ```

    -   Alternatively, you can set these environment variables directly in your system.

## Running the Application

1.  **Run the main script:**

    ```bash
    python src/main.py
    ```

2.  **Access the web dashboard:**

    -   Run the dashboard script:

        ```bash
        streamlit run src/dashboard/dashboard.py
        ```

    -   Open your web browser and navigate to the address provided by Streamlit (usually `http://localhost:8501`).

## Contributing

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Implement your changes.
4.  Write unit tests for your changes.
5.  Submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
