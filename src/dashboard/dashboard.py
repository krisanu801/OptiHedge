import streamlit as st
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logger = logging.getLogger(__name__)

def create_dashboard(portfolio: pd.DataFrame, metrics: dict) -> None:
    """
    Creates a web dashboard using Streamlit to visualize portfolio performance.

    Args:
        portfolio (pd.DataFrame): A Pandas DataFrame containing the portfolio value over time.
                                  The DataFrame should have a 'total' column representing the portfolio value.
        metrics (dict): A dictionary containing performance metrics calculated for the portfolio.
    """
    try:
        st.title("Algorithmic Trading Dashboard")

        # Display portfolio performance metrics
        st.header("Performance Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Return", f"{metrics.get('Total Return', 0):.2%}")
        col2.metric("CAGR", f"{metrics.get('CAGR', 0):.2%}")
        col3.metric("Sharpe Ratio", f"{metrics.get('Sharpe Ratio', 0):.2f}")

        col4, col5 = st.columns(2)
        col4.metric("Sortino Ratio", f"{metrics.get('Sortino Ratio', 0):.2f}")
        col5.metric("Max Drawdown", f"{metrics.get('Max Drawdown', 0):.2%}")

        # Display portfolio value over time
        st.header("Portfolio Value Over Time")
        if not portfolio.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.lineplot(x=portfolio.index, y=portfolio['total'], ax=ax)
            ax.set_xlabel("Date")
            ax.set_ylabel("Portfolio Value")
            ax.set_title("Portfolio Value Over Time")
            st.pyplot(fig)
        else:
            st.warning("No portfolio data to display.")

        # Display returns distribution
        st.header("Returns Distribution")
        if 'returns' in portfolio.columns:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(portfolio['returns'], ax=ax, kde=True)
            ax.set_xlabel("Daily Returns")
            ax.set_ylabel("Frequency")
            ax.set_title("Distribution of Daily Returns")
            st.pyplot(fig)
        else:
            st.warning("No returns data to display.")

        logger.info("Dashboard created successfully.")

    except Exception as e:
        logger.error(f"Error creating dashboard: {e}")
        st.error(f"An error occurred while creating the dashboard: {e}")


if __name__ == "__main__":
    # Example Usage
    # Create sample portfolio data
    data = {
        'total': [100000, 101000, 102000, 101500, 103000],
        'returns': [0.0, 0.01, 0.0099, -0.0049, 0.0148]
    }
    portfolio = pd.DataFrame(data)
    portfolio.index = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])

    # Create sample performance metrics
    metrics = {
        'Total Return': 0.03,
        'CAGR': 0.02,
        'Sharpe Ratio': 1.5,
        'Sortino Ratio': 2.0,
        'Max Drawdown': -0.01
    }

    # Create the dashboard
    create_dashboard(portfolio, metrics)

    # To run this dashboard:
    # 1. Save the code as dashboard.py
    # 2. Open your terminal and navigate to the directory where you saved the file.
    # 3. Run the command: streamlit run dashboard.py
    # 4. Streamlit will open the dashboard in your web browser.