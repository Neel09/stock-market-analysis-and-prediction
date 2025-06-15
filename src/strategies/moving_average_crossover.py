#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.strategies.base_strategy import BaseStrategy

class MovingAverageCrossover(BaseStrategy):
    """
    Moving Average Crossover Strategy.
    
    This strategy generates buy signals when the short-term moving average crosses above 
    the long-term moving average, and sell signals when the short-term moving average 
    crosses below the long-term moving average.
    """
    
    def __init__(self, data=None, config=None, config_path=None, short_window=None, long_window=None):
        """
        Initialize the Moving Average Crossover strategy.
        
        Args:
            data (pandas.DataFrame, optional): The OHLCV dataframe
            config (dict, optional): Configuration parameters
            config_path (str, optional): Path to configuration file
            short_window (int, optional): Short-term moving average window
            long_window (int, optional): Long-term moving average window
        """
        super().__init__('moving_average_crossover', data, config, config_path)
        
        # Set strategy parameters
        self.short_window = short_window or self.strategy_config.get('short_window', 20)
        self.long_window = long_window or self.strategy_config.get('long_window', 50)
        
    def generate_signals(self):
        """
        Generate trading signals based on moving average crossovers.
        
        Returns:
            pandas.Series: Series of trading signals (1: long, 0: no position)
        """
        if self.data is None:
            raise ValueError("Data must be set before generating signals")
        
        # Make a copy of the data
        data = self.data.copy()
        
        # Create moving averages
        data['short_ma'] = data['Close'].rolling(window=self.short_window, min_periods=1).mean()
        data['long_ma'] = data['Close'].rolling(window=self.long_window, min_periods=1).mean()
        
        # Store indicators for plotting
        self.indicators = {
            'short_ma': data['short_ma'],
            'long_ma': data['long_ma']
        }
        
        # Generate signals - identify crossover points only
        signals = pd.Series(0, index=data.index)
        
        # Buy signal: short MA crosses above long MA
        buy_signals = (data['short_ma'].shift(1) <= data['long_ma'].shift(1)) & (data['short_ma'] > data['long_ma'])
        signals[buy_signals] = 1
        
        # Sell signal: short MA crosses below long MA
        sell_signals = (data['short_ma'].shift(1) >= data['long_ma'].shift(1)) & (data['short_ma'] < data['long_ma'])
        signals[sell_signals] = -1
        
        # Store signals (these are the entry/exit points)
        self.signals = signals
        
        # Generate positions (1 for long, 0 for no position)
        # Start with 0 positions
        positions = pd.Series(0, index=data.index)
        
        # Set position to 1 (long) when short MA > long MA
        positions[data['short_ma'] > data['long_ma']] = 1
        
        # Store positions
        self.positions = positions
        
        return signals

if __name__ == '__main__':
    # Example usage
    import sys
    import os
    import json
    
    # Add the parent directory to the path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    from src.utils.data_fetcher import DataFetcher
    from src.utils.data_processor import DataProcessor
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                              'config', 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Fetch data
    fetcher = DataFetcher(config_path)
    data = fetcher.fetch_data('AAPL', period='2y')
    
    # Initialize strategy
    strategy = MovingAverageCrossover(data=data, config=config)
    
    # Run backtest
    results = strategy.run_backtest()
    
    # Print results
    print(f"Strategy: {results['strategy_name']}")
    for metric, value in results['metrics'].items():
        if isinstance(value, (int, float)):
            print(f"{metric}: {value:.4f}")
    
    # Plot results
    strategy.plot_results()
    plt.show()
    
    # Save results
    strategy.save_results() 