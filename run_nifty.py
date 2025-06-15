#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """
    Main function to run the Nifty 50 trading system.
    """
    parser = argparse.ArgumentParser(description='Nifty 50 Trading System')
    parser.add_argument('--config', default='config/config.json', help='Path to config file')
    parser.add_argument('--period', default='5y', help='Period for data (e.g., 1y, 2y, 5y)')
    parser.add_argument('--index', default='NIFTY 50', help='Index name (e.g., NIFTY 50, NIFTY BANK)')
    parser.add_argument('--mode', choices=['fetch', 'run', 'all'], default='all', 
                       help='Mode: fetch data only, run strategies only, or both')
    
    args = parser.parse_args()
    
    # Import modules here to avoid slow startup
    if args.mode in ['fetch', 'all']:
        from src.utils.nifty_data_fetcher import NiftyDataFetcher
        
        print(f"Fetching {args.index} data for period {args.period}...")
        fetcher = NiftyDataFetcher(args.config)
        data = fetcher.fetch_nifty_data(period=args.period, index_name=args.index)
        
        print(f"\nData shape: {data.shape}")
        print("\nSample data:")
        print(data.head())
    
    if args.mode in ['run', 'all']:
        from src.nifty50_trading import run_nifty50_strategies
        
        print(f"\nRunning trading strategies on {args.index} data...")
        results = run_nifty50_strategies(args.config, args.period, args.index)
        
        print("\nStrategy performance summary:")
        if 'Moving Average Crossover' in results:
            print("Moving Average Crossover:")
            for metric, value in results['Moving Average Crossover']['metrics'].items():
                if isinstance(value, float):
                    if 'return' in metric or 'drawdown' in metric:
                        print(f"  {metric}: {value:.2%}")
                    else:
                        print(f"  {metric}: {value:.4f}")

if __name__ == '__main__':
    main() 