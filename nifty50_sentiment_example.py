#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import json

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.nifty_data_fetcher import NiftyDataFetcher
from src.strategies.sentiment_strategy import SentimentStrategy
from src.utils.sentiment_analyzer import SentimentAnalyzer
from src.utils.news_fetcher import NewsFetcher

def main():
    """
    Example script demonstrating the sentiment-based trading strategy.
    """
    print("Nifty 50 Sentiment Analysis Strategy Example")
    print("===========================================")

    # Load configuration
    config_path = 'config/config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Check if API keys are set
    if not config.get('sentiment', {}).get('llm_api_key'):
        print("\nWarning: LLM API key not set in config. Using mock sentiment data.")
        print("To use real sentiment analysis, add your API key to config/config.json")

    if not config.get('sentiment', {}).get('news_api_key'):
        print("\nWarning: News API key not set in config. Using mock news data.")
        print("To use real news data, add your API key to config/config.json")

    # Fetch Nifty 50 data
    print("\nFetching Nifty 50 data...")
    data_fetcher = NiftyDataFetcher(config_path)
    nifty_data = data_fetcher.fetch_nifty_data(period='1y')

    # Check if data is empty
    if len(nifty_data) == 0:
        print("No data fetched. Generating sample data...")
        # Get start and end dates for the sample data
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1 year of data
        nifty_data = data_fetcher.generate_sample_data(start_date, end_date)

    print(f"Data fetched: {len(nifty_data)} days from {nifty_data.index[0]} to {nifty_data.index[-1]}")

    # Initialize sentiment strategy
    print("\nInitializing sentiment strategy...")
    sentiment_strategy = SentimentStrategy(
        data=nifty_data,
        config_path=config_path,
        symbol='NIFTY 50',  # Use NIFTY 50 to match the data we fetched
        sentiment_threshold=config['strategies']['sentiment_strategy']['sentiment_threshold'],
        market_sentiment_weight=config['strategies']['sentiment_strategy']['market_sentiment_weight'],
        use_technical_indicators=config['strategies']['sentiment_strategy']['use_technical_indicators'],
        days=config['strategies']['sentiment_strategy']['days'],
        max_news=config['strategies']['sentiment_strategy']['max_news']
    )

    # Fetch and analyze news
    print("\nFetching and analyzing news sentiment...")

    # Run backtest
    print("\nRunning backtest...")
    results = sentiment_strategy.run_backtest()

    # Print performance metrics
    print("\nPerformance Metrics:")
    metrics = results['metrics']
    print(f"Total Return: {metrics.get('total_return', 0):.2%}")
    print(f"Annual Return: {metrics.get('annual_return', 0):.2%}")
    print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    print(f"Win Rate: {metrics.get('win_rate', 0):.2%}")

    # Plot results
    print("\nPlotting results...")
    fig = sentiment_strategy.plot_results()

    # Save results
    print("\nSaving results...")
    sentiment_strategy.save_results()

    plt.show()

    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()
