#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the Nifty data fetcher
from src.utils.nifty_data_fetcher import NiftyDataFetcher

# Import trading strategies
from src.strategies.moving_average_crossover import MovingAverageCrossover
from src.strategies.rsi_strategy import RSIStrategy
from src.strategies.macd_strategy import MACDStrategy
from src.strategies.bollinger_bands import BollingerBands

# Import ML/DL strategies if available
try:
    from src.strategies.ml_strategy import MLStrategy
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: ML strategy not available")

try:
    from src.strategies.lstm_strategy import LSTMStrategy
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    print("Warning: LSTM strategy not available")

# Import visualization tools
from src.visualization.compare_strategies import compare_strategies

def run_nifty50_strategies(config_path, period="5y", index_name="NIFTY 50"):
    """
    Run trading strategies on Nifty 50 data.
    
    Args:
        config_path (str): Path to config file
        period (str): Period for data ('1y', '2y', '5y', etc.)
        index_name (str): Index name (default: "NIFTY 50")
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Fetch Nifty 50 data
    # fetcher = NiftyDataFetcher(config_path)
    # data = fetcher.fetch_nifty_data(period=period, index_name=index_name)
    data = NiftyDataFetcher.fetch_data_from_csv(
        '/Users/neelansh/Desktop/Projects/My Projects/Stock Market Data/TATAMOTORS_till_13June2025.csv')

    # Determine if we're using sample data
    using_sample_data = '_sample' in str(data.index.name) if hasattr(data.index, 'name') else False
    if not using_sample_data:
        # Check the first few rows to see if they match our sample data pattern
        if len(data) > 0 and data.iloc[0]['Close'] > 18000 and data.iloc[0]['Close'] < 18200:
            # This looks like our sample data
            using_sample_data = True
    
    print(f"\nRunning trading strategies on {index_name} data...")
    
    # Initialize strategies
    strategies = {}
    results = {}
    
    # 1. Moving Average Crossover
    print("\n1. Running Moving Average Crossover strategy...")
    ma_strategy = MovingAverageCrossover(
        data=data,
        config=config,
        short_window=20,
        long_window=50
    )
    ma_results = ma_strategy.run_backtest()
    strategies["Moving Average Crossover"] = ma_strategy
    results["Moving Average Crossover"] = ma_results
    
    # 2. RSI Strategy
    print("\n2. Running RSI strategy...")
    rsi_strategy = RSIStrategy(
        data=data,
        config=config,
        rsi_period=14,
        overbought=70,
        oversold=30
    )
    rsi_results = rsi_strategy.run_backtest()
    strategies["RSI"] = rsi_strategy
    results["RSI"] = rsi_results
    
    # 3. MACD Strategy
    print("\n3. Running MACD strategy...")
    macd_strategy = MACDStrategy(
        data=data,
        config=config
    )
    macd_results = macd_strategy.run_backtest()
    strategies["MACD"] = macd_strategy
    results["MACD"] = macd_results
    
    # 4. Bollinger Bands Strategy
    print("\n4. Running Bollinger Bands strategy...")
    bb_strategy = BollingerBands(
        data=data,
        config=config
    )
    bb_results = bb_strategy.run_backtest()
    strategies["Bollinger Bands"] = bb_strategy
    results["Bollinger Bands"] = bb_results
    
    # 5. ML Strategy (if available and not using sample data)
    # if ML_AVAILABLE and not using_sample_data:
    #     print("\n5. Running ML strategy...")
    #     ml_strategy = MLStrategy(
    #         data=data,
    #         config=config,
    #         model_type='logistic'
    #     )
    #     ml_results = ml_strategy.run_backtest()
    #     strategies["ML"] = ml_strategy
    #     results["ML"] = ml_results
    # else:
    #     print("\n5. Skipping ML strategy (not available or using sample data)")
    
    # 6. LSTM Strategy (if available and not using sample data)
    if LSTM_AVAILABLE and not using_sample_data:
        print("\n6. Running LSTM strategy...")
        lstm_strategy = LSTMStrategy(
            data=data,
            config=config
        )
        lstm_results = lstm_strategy.run_backtest()
        strategies["LSTM"] = lstm_strategy
        results["LSTM"] = lstm_results
    else:
        print("\n6. Skipping LSTM strategy (not available or using sample data)")
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'nifty50')
    os.makedirs(results_dir, exist_ok=True)
    
    # Compare strategies
    print("\nComparing all strategies...")
    compare_results = compare_strategies(results)
    
    # Save comparison results
    comparison_path = os.path.join(results_dir, 'strategy_comparison.csv')
    comparison_df = pd.DataFrame(compare_results)
    comparison_df.to_csv(comparison_path)
    print(f"Comparison results saved to {comparison_path}")
    
    # Visualize results
    visualize_results(strategies, results, results_dir)
    
    return results

def visualize_results(strategies, results, output_dir):
    """
    Visualize the results of all strategies.
    
    Args:
        strategies (dict): Dictionary of strategy objects
        results (dict): Dictionary of strategy results
        output_dir (str): Directory to save visualizations
    """
    # Create a directory for visualizations
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Plot individual strategy results
    for name, strategy in strategies.items():
        print(f"Plotting results for {name} strategy...")
        fig = strategy.plot_results()
        fig_path = os.path.join(viz_dir, f"{name.lower().replace(' ', '_')}_plot.png")
        fig.savefig(fig_path)
        plt.close(fig)
        print(f"Plot saved to {fig_path}")
    
    # Plot cumulative returns comparison
    plt.figure(figsize=(16, 10))
    
    # Buy and Hold baseline
    if 'Moving Average Crossover' in results:  # Use any strategy data for buy and hold
        data = strategies['Moving Average Crossover'].data
        returns = data['Close'].pct_change().fillna(0)
        cum_returns = (1 + returns).cumprod() - 1
        plt.plot(cum_returns.index, cum_returns, label='Buy & Hold', linestyle='--')
    
    # Plot each strategy's cumulative returns
    for name, result in results.items():
        plt.plot(result['cum_returns'].index, result['cum_returns'], label=name)
    
    plt.title(f"Cumulative Returns Comparison - Nifty 50")
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    
    # Save the comparison plot
    comparison_path = os.path.join(viz_dir, 'cumulative_returns_comparison.png')
    plt.savefig(comparison_path)
    plt.close()
    print(f"Comparison plot saved to {comparison_path}")
    
    # Create a performance metrics table
    metrics = ['total_return', 'annualized_return', 'sharpe_ratio', 
              'sortino_ratio', 'maximum_drawdown', 'win_rate']
    
    metrics_data = []
    for name, result in results.items():
        strategy_metrics = [name]
        for metric in metrics:
            value = result['metrics'][metric]
            strategy_metrics.append(value)
        metrics_data.append(strategy_metrics)
    
    metrics_df = pd.DataFrame(metrics_data, columns=['Strategy'] + metrics)
    
    # Format percentage values
    for col in ['total_return', 'annualized_return', 'maximum_drawdown', 'win_rate']:
        metrics_df[col] = metrics_df[col].apply(lambda x: f"{x:.2%}")
    
    # Format other values
    for col in ['sharpe_ratio', 'sortino_ratio']:
        metrics_df[col] = metrics_df[col].apply(lambda x: f"{x:.4f}")
    
    # Save metrics table
    metrics_path = os.path.join(output_dir, 'performance_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Performance metrics saved to {metrics_path}")
    
    return metrics_df

if __name__ == "__main__":
    # Get config path
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'config.json')
    
    # Get period if provided
    period = "5y"
    if len(sys.argv) > 2:
        period = sys.argv[2]
    
    # Get index name if provided
    index_name = "NIFTY 50"
    if len(sys.argv) > 3:
        index_name = sys.argv[3]
    
    # Run strategies
    results = run_nifty50_strategies(config_path, period, index_name) 