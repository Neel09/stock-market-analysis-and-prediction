#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.strategies.base_strategy import BaseStrategy

class MACDStrategy(BaseStrategy):
    """
    Moving Average Convergence Divergence (MACD) Strategy.
    
    This strategy generates buy signals when the MACD line crosses above the signal line,
    and sell signals when the MACD line crosses below the signal line.
    """
    
    def __init__(self, data=None, config=None, config_path=None, 
                 fast_period=None, slow_period=None, signal_period=None):
        """
        Initialize the MACD strategy.
        
        Args:
            data (pandas.DataFrame, optional): The OHLCV dataframe
            config (dict, optional): Configuration parameters
            config_path (str, optional): Path to configuration file
            fast_period (int, optional): Fast EMA period
            slow_period (int, optional): Slow EMA period
            signal_period (int, optional): Signal line period
        """
        super().__init__('macd', data, config, config_path)
        
        # Set strategy parameters
        self.fast_period = fast_period or self.strategy_config.get('fast_period', 12)
        self.slow_period = slow_period or self.strategy_config.get('slow_period', 26)
        self.signal_period = signal_period or self.strategy_config.get('signal_period', 9)
        
    def calculate_macd(self, prices):
        """
        Calculate MACD.
        
        Args:
            prices (pandas.Series): Series of prices
            
        Returns:
            tuple: (macd, signal, histogram)
        """
        # Calculate EMAs
        fast_ema = prices.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = prices.ewm(span=self.slow_period, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
        
    def generate_signals(self):
        """
        Generate trading signals based on MACD.
        
        Returns:
            pandas.Series: Series of trading signals (1: long, 0: no position, -1: short)
        """
        if self.data is None:
            raise ValueError("Data must be set before generating signals")
        
        # Make a copy of the data
        data = self.data.copy()
        
        # Calculate MACD
        macd_line, signal_line, histogram = self.calculate_macd(data['Close'])
        
        # Add to dataframe
        data['macd'] = macd_line
        data['signal'] = signal_line
        data['histogram'] = histogram
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        
        # Buy signal: MACD line crosses above signal line
        signals[(data['macd'] > data['signal']) & (data['macd'].shift(1) <= data['signal'].shift(1))] = 1
        
        # Sell signal: MACD line crosses below signal line
        signals[(data['macd'] < data['signal']) & (data['macd'].shift(1) >= data['signal'].shift(1))] = -1
        
        # Fill in the gaps with previous signal (hold the position)
        signals = signals.replace(0, np.nan).ffill().fillna(0)
        
        # Store the signals
        self.signals = signals
        
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
    strategy = MACDStrategy(data=data, config=config)
    
    # Run backtest
    results = strategy.run_backtest()
    
    # Print results
    print(f"Strategy: {results['strategy_name']}")
    for metric, value in results['metrics'].items():
        if isinstance(value, (int, float)):
            print(f"{metric}: {value:.4f}")
    
    # Plot results
    import matplotlib.pyplot as plt
    strategy.plot_results()
    plt.show()
    
    # Save results
    strategy.save_results() 