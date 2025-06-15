#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.strategies.base_strategy import BaseStrategy

class RSIStrategy(BaseStrategy):
    """
    Relative Strength Index (RSI) Strategy.
    
    This strategy generates buy signals when the RSI falls below the oversold level,
    and sell signals when the RSI rises above the overbought level.
    """
    
    def __init__(self, data=None, config=None, config_path=None, 
                 rsi_period=None, overbought=None, oversold=None):
        """
        Initialize the RSI strategy.
        
        Args:
            data (pandas.DataFrame, optional): The OHLCV dataframe
            config (dict, optional): Configuration parameters
            config_path (str, optional): Path to configuration file
            rsi_period (int, optional): RSI calculation period
            overbought (int, optional): Overbought threshold
            oversold (int, optional): Oversold threshold
        """
        super().__init__('rsi', data, config, config_path)
        
        # Set strategy parameters
        self.rsi_period = rsi_period or self.strategy_config.get('rsi_period', 14)
        self.overbought = overbought or self.strategy_config.get('overbought', 70)
        self.oversold = oversold or self.strategy_config.get('oversold', 30)
        
    def calculate_rsi(self, prices, period=14):
        """
        Calculate Relative Strength Index.
        
        Args:
            prices (pandas.Series): Series of prices
            period (int): Period for RSI calculation
            
        Returns:
            pandas.Series: Series of RSI values
        """
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.copy()
        losses = delta.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = -losses
        
        # Calculate simple averages
        avg_gain = gains.rolling(window=period).mean()
        avg_loss = losses.rolling(window=period).mean()
        
        # Calculate RS
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    def generate_signals(self):
        """
        Generate trading signals based on RSI values.
        
        Returns:
            pandas.Series: Series of trading signals (1: long, 0: no position, -1: short)
        """
        if self.data is None:
            raise ValueError("Data must be set before generating signals")
        
        # Make a copy of the data
        data = self.data.copy()
        
        # Calculate RSI
        data['rsi'] = self.calculate_rsi(data['Close'], self.rsi_period)
        
        # Store indicator values for visualization
        self.indicators = {
            'rsi': data['rsi']
        }
        
        # Generate basic signals based on RSI thresholds
        signals = pd.Series(0, index=data.index)
        
        # Simple approach: Buy when RSI is below oversold, sell when above overbought
        signals[data['rsi'] < self.oversold] = 1  # Buy signal
        signals[data['rsi'] > self.overbought] = -1  # Sell signal
        
        # Store the signals (entry/exit points)
        self.signals = signals
        
        # Generate positions (holding status)
        # Start with 0 position everywhere
        positions = pd.Series(0, index=data.index)
        
        # Set position to 1 (long) after buy signal until sell signal
        in_position = False
        for i in range(len(signals)):
            date = signals.index[i]
            signal = signals.iloc[i]
            
            if signal == 1 and not in_position:  # Buy signal and not already in position
                positions.iloc[i:] = 1  # Set position to 1 from this point forward
                in_position = True
            elif signal == -1 and in_position:  # Sell signal and in position
                positions.iloc[i:] = 0  # Exit position
                in_position = False
        
        # Store the positions (holding status)
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
    strategy = RSIStrategy(data=data, config=config)
    
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