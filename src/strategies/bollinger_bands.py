#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.strategies.base_strategy import BaseStrategy

class BollingerBands(BaseStrategy):
    """
    Bollinger Bands Strategy.
    
    This strategy generates buy signals when the price touches or crosses below the lower band,
    and sell signals when the price touches or crosses above the upper band.
    """
    
    def __init__(self, data=None, config=None, config_path=None, window=None, num_std=None):
        """
        Initialize the Bollinger Bands strategy.
        
        Args:
            data (pandas.DataFrame, optional): The OHLCV dataframe
            config (dict, optional): Configuration parameters
            config_path (str, optional): Path to configuration file
            window (int, optional): Window for moving average
            num_std (float, optional): Number of standard deviations for bands
        """
        super().__init__('bollinger_bands', data, config, config_path)
        
        # Set strategy parameters
        self.window = window or self.strategy_config.get('window', 20)
        self.num_std = num_std or self.strategy_config.get('num_std', 2)
        
    def calculate_bollinger_bands(self, prices):
        """
        Calculate Bollinger Bands.
        
        Args:
            prices (pandas.Series): Series of prices
            
        Returns:
            tuple: (upper_band, middle_band, lower_band)
        """
        # Calculate middle band (simple moving average)
        middle_band = prices.rolling(window=self.window).mean()
        
        # Calculate standard deviation
        rolling_std = prices.rolling(window=self.window).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (rolling_std * self.num_std)
        lower_band = middle_band - (rolling_std * self.num_std)
        
        return upper_band, middle_band, lower_band
        
    def generate_signals(self):
        """
        Generate trading signals based on Bollinger Bands.
        
        Returns:
            pandas.Series: Series of trading signals (1: long, 0: no position, -1: short)
        """
        if self.data is None:
            raise ValueError("Data must be set before generating signals")
        
        # Make a copy of the data
        data = self.data.copy()
        
        # Calculate Bollinger Bands
        upper_band, middle_band, lower_band = self.calculate_bollinger_bands(data['Close'])
        
        # Add to dataframe
        data['upper_band'] = upper_band
        data['middle_band'] = middle_band
        data['lower_band'] = lower_band
        
        # Calculate bandwidth and %B
        data['bandwidth'] = (data['upper_band'] - data['lower_band']) / data['middle_band']
        data['percent_b'] = (data['Close'] - data['lower_band']) / (data['upper_band'] - data['lower_band'])
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        
        # Buy signal: Price crosses below lower band
        signals[data['Close'] <= data['lower_band']] = 1
        
        # Sell signal: Price crosses above upper band
        signals[data['Close'] >= data['upper_band']] = -1
        
        # Additional signal: Mean reversion when price is extreme
        # Buy when very oversold (percent_b < 0)
        signals[data['percent_b'] < 0] = 1
        
        # Sell when very overbought (percent_b > 1)
        signals[data['percent_b'] > 1] = -1
        
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
    strategy = BollingerBands(data=data, config=config)
    
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