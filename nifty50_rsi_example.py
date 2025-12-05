#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example of running RSI strategy on Nifty 50 data.
This standalone script demonstrates how to use a single trading strategy
with the Nifty 50 data fetcher.
"""

import os
import sys
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the Nifty data fetcher
from src.utils.nifty_data_fetcher import NiftyDataFetcher

# Import RSI strategy
from src.strategies.rsi_strategy import RSIStrategy


def main():
    """
    Main function to run the RSI strategy on Nifty 50 data.
    """
    # Fetch Nifty 50 data
    print("Fetching Nifty 50 data...")
    fetcher = NiftyDataFetcher()
    # data = fetcher.fetch_nifty_data(period="10y", ticker_symbol="BAJAJFINSV.NS")
    data = fetcher.fetch_ticker_data_from_csv(ticker_symbol="TATAMOTORS.NS")

    # data = NiftyDataFetcher.fetch_data_from_csv(
    #     '/Users/neelansh/Desktop/Projects/My Projects/Stock Market Data/TATAMOTORS_till_13June2025.csv')

    # data['Date'] = data.to_datetime(data['Date'])

    # Reset index if you want clean index after sort
    # data = data.reset_index()

    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")

    # Initialize RSI strategy
    print("\nInitializing RSI strategy...")
    rsi_strategy = RSIStrategy(
        data=data,
        rsi_period=14,
        config_path='config/config.json'
    )

    # Run backtest
    print("Running backtest...")
    results = rsi_strategy.run_backtest()

    # Print performance metrics
    print("\nPerformance Metrics:")
    for metric, value in results["metrics"].items():
        if isinstance(value, float):
            if "return" in metric or "drawdown" in metric or "rate" in metric:
                print(f"  {metric}: {value:.2%}")
            else:
                print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")

    # Plot results
    print("\nPlotting results...")
    fig = rsi_strategy.plot_results()

    # Show plot
    fig.tight_layout()
    fig.show()

    return results


if __name__ == "__main__":
    main()
