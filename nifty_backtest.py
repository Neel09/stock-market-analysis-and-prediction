#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from src.backtest.run_backtest import BacktestRunner

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Nifty 50 Trading Backtest System')
    parser.add_argument('--symbol', default='NIFTY 50', help='Symbol to backtest (default: NIFTY 50)')
    parser.add_argument('--period', default='1y', help='Period to fetch (e.g., 1y, 5y, default: 1y)')
    parser.add_argument('--interval', default='1d', help='Data interval (e.g., 1d, 1h, default: 1d)')
    parser.add_argument('--strategies', help='Comma-separated list of strategies to run')
    parser.add_argument('--list-strategies', action='store_true', help='List available strategies')
    parser.add_argument('--config', help='Path to configuration file')
    return parser.parse_args()

def list_available_strategies():
    """List all available strategies."""
    print("Available strategies:")
    print("  - moving_average_crossover: Simple moving average crossover strategy")
    print("  - rsi_strategy: Relative Strength Index strategy")
    print("  - macd_strategy: Moving Average Convergence Divergence strategy")
    print("  - bollinger_bands: Bollinger Bands strategy")
    print("  - ml_strategy: Machine Learning based strategy (if available)")
    print("  - lstm_strategy: LSTM deep learning strategy (if available)")

def main():
    """Main function."""
    args = parse_args()
    
    # List strategies if requested
    if args.list_strategies:
        list_available_strategies()
        return
    
    print(f"Nifty 50 Trading Backtest System")
    print(f"================================")
    print(f"Symbol: {args.symbol}")
    print(f"Period: {args.period}")
    print(f"Interval: {args.interval}")
    
    # Initialize backtest runner
    runner = BacktestRunner(args.config)
    
    # Fetch data
    print(f"\nFetching data for {args.symbol}...")
    data = runner.fetch_data(args.symbol, args.period, args.interval)
    
    if data is None or data.empty:
        print("Error: No data available. Exiting.")
        return
    
    print(f"Data fetched successfully. Shape: {data.shape}")
    
    # Add strategies
    if args.strategies:
        strategies = args.strategies.split(',')
        print(f"\nRunning the following strategies: {', '.join(strategies)}")
    else:
        strategies = list(runner.strategy_classes.keys())
        print(f"\nRunning all available strategies: {', '.join(strategies)}")
    
    for strategy in strategies:
        try:
            runner.add_strategy(strategy)
        except ValueError as e:
            print(f"Error adding strategy {strategy}: {e}")
    
    # Run backtests
    print("\nRunning backtests...")
    results = runner.run_backtest()
    
    # Plot comparison
    runner.plot_comparison()
    
    # Print results summary
    print("\nBacktest Results Summary")
    print("=======================")
    
    comparison = runner.compare_strategies()
    
    # Create a more readable format for the metrics
    for column in comparison.columns:
        metrics = comparison[column]
        print(f"\n{column}:")
        for metric, value in metrics.items():
            if metric in ['total_return', 'annualized_return', 'maximum_drawdown', 'win_rate']:
                print(f"  {metric}: {value:.2%}")
            else:
                print(f"  {metric}: {value:.4f}")
    
    print("\nBacktest completed successfully!")
    print(f"Results saved to: {os.path.abspath('results/backtest')}")
    
    return results

if __name__ == '__main__':
    main() 