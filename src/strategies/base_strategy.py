#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from abc import ABC, abstractmethod
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.performance_metrics import PerformanceMetrics

class BaseStrategy(ABC):
    """
    Base class for all trading strategies.
    """
    
    def __init__(self, name, data=None, config=None, config_path=None):
        """
        Initialize the strategy.
        
        Args:
            name (str): The name of the strategy
            data (pandas.DataFrame, optional): The OHLCV dataframe
            config (dict, optional): Configuration parameters
            config_path (str, optional): Path to configuration file
        """
        self.name = name
        self.data = data
        self.signals = None
        self.positions = None
        self.returns = None
        self.cum_returns = None
        self.metrics = None
        
        # Load configuration
        if config is not None:
            self.config = config
        elif config_path is not None:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {}
        
        # Load strategy-specific config if available
        if 'strategies' in self.config and name in self.config['strategies']:
            self.strategy_config = self.config['strategies'][name]
        else:
            self.strategy_config = {}
            
        # Initialize backtest parameters
        self.initial_capital = self.config.get('backtest', {}).get('initial_capital', 100000)
        self.commission = self.config.get('backtest', {}).get('commission', 0.001)
        self.slippage = self.config.get('backtest', {}).get('slippage', 0.001)
        self.position_size = self.config.get('backtest', {}).get('position_size', 0.2)
        self.risk_free_rate = self.config.get('backtest', {}).get('risk_free_rate', 0.02)
    
    def set_data(self, data):
        """
        Set the data for the strategy.
        
        Args:
            data (pandas.DataFrame): The OHLCV dataframe
        """
        self.data = data
        
    @abstractmethod
    def generate_signals(self):
        """
        Generate trading signals.
        
        Returns:
            pandas.Series: Series of trading signals (1: long, 0: no position, -1: short)
        """
        pass
    
    def generate_positions(self, signals=None):
        """
        Generate positions from signals.
        
        Args:
            signals (pandas.Series, optional): Series of signals
            
        Returns:
            pandas.Series: Series of positions
        """
        if signals is None:
            if self.signals is None:
                self.generate_signals()
            signals = self.signals
            
        # Generate positions (assume positions are already derived from signals in the strategy)
        # If positions are already set by the strategy class, use those
        if hasattr(self, 'positions') and self.positions is not None:
            return self.positions
            
        # Otherwise, derive them from signals
        # Long position when signal is 1, no position when signal is 0 or -1
        self.positions = signals.copy()
        
        # Convert signals to positions (holding status)
        # Fill forward - maintain position until new signal
        self.positions = self.positions.replace(0, np.nan).ffill().fillna(0)
        
        return self.positions
    
    def calculate_returns(self, positions=None, price_column='Close'):
        """
        Calculate strategy returns.
        
        Args:
            positions (pandas.Series, optional): Series of positions
            price_column (str): Column to use for price data
            
        Returns:
            pandas.Series: Series of strategy returns
        """
        if positions is None:
            if self.positions is None:
                self.generate_positions()
            positions = self.positions
        
        # Calculate returns
        prices = self.data[price_column]
        self.returns = PerformanceMetrics.calculate_returns(prices, positions)
        
        # Calculate cumulative returns
        self.cum_returns = PerformanceMetrics.calculate_cumulative_returns(self.returns)
        
        return self.returns
    
    def calculate_metrics(self, returns=None):
        """
        Calculate performance metrics.
        
        Args:
            returns (pandas.Series, optional): Series of returns
            
        Returns:
            dict: Dictionary of performance metrics
        """
        if returns is None:
            if self.returns is None:
                self.calculate_returns()
            returns = self.returns
            
        # Calculate metrics
        self.metrics = PerformanceMetrics.calculate_all_metrics(
            returns, 
            risk_free_rate=self.risk_free_rate
        )
        
        return self.metrics
    
    def run_backtest(self):
        """
        Run backtest from start to finish.
        
        Returns:
            dict: Dictionary containing backtest results
        """
        # Generate signals
        signals = self.generate_signals()
        
        # Generate positions
        positions = self.generate_positions(signals)
        
        # Calculate returns
        returns = self.calculate_returns(positions)
        
        # Calculate metrics
        metrics = self.calculate_metrics(returns)
        
        # Prepare results
        results = {
            'strategy_name': self.name,
            'signals': signals,
            'positions': positions,
            'returns': returns,
            'cum_returns': self.cum_returns,
            'metrics': metrics
        }
        
        return results
    
    def plot_results(self, benchmark_returns=None):
        """
        Plot strategy results.
        
        Args:
            benchmark_returns (pandas.Series, optional): Series of benchmark returns
        """
        if self.cum_returns is None:
            self.calculate_returns()
            
        # Create figure - adjust to have 3 subplots if indicators are available
        if hasattr(self, 'indicators') and self.indicators:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), 
                                             gridspec_kw={'height_ratios': [3, 1, 1.5]})
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                        gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price and cumulative returns
        ax1.plot(self.cum_returns.index, self.cum_returns, label=f'{self.name} Strategy')
        
        # Plot benchmark if provided
        if benchmark_returns is not None:
            benchmark_cum_returns = PerformanceMetrics.calculate_cumulative_returns(benchmark_returns)
            ax1.plot(benchmark_cum_returns.index, benchmark_cum_returns, label='Benchmark')
        
        # Plot entry/exit points
        try:
            if self.signals is not None and self.cum_returns is not None:
                # Find actual buy signals (entry points)
                buy_signals = self.signals[self.signals == 1].index
                
                # Get cumulative returns at signal points - safely check if all points exist in cum_returns
                if len(buy_signals) > 0:
                    # Filter to ensure we only use buy signals that exist in cum_returns
                    valid_buy_signals = buy_signals.intersection(self.cum_returns.index)
                    if len(valid_buy_signals) > 0:
                        buy_y = self.cum_returns.loc[valid_buy_signals]
                        if len(valid_buy_signals) == len(buy_y):  # Double check sizes match
                            ax1.scatter(valid_buy_signals, buy_y, color='green', marker='^', s=100, label='Buy Signal')
                
                # Find actual sell signals (exit points)
                sell_signals = self.signals[self.signals == -1].index
                
                # Get cumulative returns at signal points - safely check if all points exist in cum_returns
                if len(sell_signals) > 0:
                    # Filter to ensure we only use sell signals that exist in cum_returns
                    valid_sell_signals = sell_signals.intersection(self.cum_returns.index)
                    if len(valid_sell_signals) > 0:
                        sell_y = self.cum_returns.loc[valid_sell_signals]
                        if len(valid_sell_signals) == len(sell_y):  # Double check sizes match
                            ax1.scatter(valid_sell_signals, sell_y, color='red', marker='v', s=100, label='Sell Signal')
        except Exception as e:
            print(f"Warning: Could not plot buy/sell signals: {e}")
        
        ax1.set_title(f'{self.name} Strategy Performance')
        ax1.set_ylabel('Cumulative Returns')
        ax1.legend()
        ax1.grid(True)
        
        # Plot daily returns
        ax2.plot(self.returns.index, self.returns, label='Daily Returns', color='gray', alpha=1.0)
        ax2.set_ylabel('Daily Returns')
        ax2.grid(True)
        
        # Plot indicators if available
        if hasattr(self, 'indicators') and self.indicators:
            # Plot first indicator (usually the main one like RSI, MACD, etc.)
            if 'rsi' in self.indicators:
                ax3.plot(self.indicators['rsi'].index, self.indicators['rsi'], label='RSI', color='purple')
                # Add overbought/oversold lines if RSI
                if hasattr(self, 'overbought') and hasattr(self, 'oversold'):
                    ax3.axhline(y=self.overbought, color='r', linestyle='--', alpha=0.5)
                    ax3.axhline(y=self.oversold, color='g', linestyle='--', alpha=0.5)
            elif 'macd' in self.indicators:
                ax3.plot(self.indicators['macd'].index, self.indicators['macd'], label='MACD', color='blue')
                ax3.plot(self.indicators['signal'].index, self.indicators['signal'], label='Signal', color='red')
                ax3.bar(self.indicators['histogram'].index, self.indicators['histogram'], label='Histogram', color='green', alpha=0.5)
            elif 'upper_band' in self.indicators:
                ax3.plot(self.data.index, self.data['Close'], label='Price', color='black', alpha=0.7)
                ax3.plot(self.indicators['upper_band'].index, self.indicators['upper_band'], label='Upper Band', color='red')
                ax3.plot(self.indicators['middle_band'].index, self.indicators['middle_band'], label='Middle Band', color='blue')
                ax3.plot(self.indicators['lower_band'].index, self.indicators['lower_band'], label='Lower Band', color='green')
            elif 'short_ma' in self.indicators:
                # For moving average strategies, show price and MAs on the same plot
                ax1.plot(self.data.index, self.data['Close'], label='Price', color='black', alpha=0.5)
                ax1.plot(self.indicators['short_ma'].index, self.indicators['short_ma'], label=f'{self.short_window}-day MA', color='blue')
                ax1.plot(self.indicators['long_ma'].index, self.indicators['long_ma'], label=f'{self.long_window}-day MA', color='red')
                # Don't use a third subplot for moving averages
                ax3.set_visible(False)
            
            ax3.set_ylabel('Indicator')
            ax3.legend()
            ax3.grid(True)
        
        # Show metrics in a text box
        if self.metrics is not None:
            metrics_text = '\n'.join([
                f'Total Return: {self.metrics["total_return"]:.2%}',
                f'Annualized Return: {self.metrics["annualized_return"]:.2%}',
                f'Sharpe Ratio: {self.metrics["sharpe_ratio"]:.2f}',
                f'Max Drawdown: {self.metrics["maximum_drawdown"]:.2%}',
                f'Win Rate: {self.metrics["win_rate"]:.2%}'
            ])
            
            ax1.text(
                0.02, 0.95, metrics_text,
                transform=ax1.transAxes,
                verticalalignment='top',
                bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5}
            )
        
        try:
            plt.tight_layout()
        except:
            print("Warning: Could not apply tight layout")
        
        return fig
    
    def save_results(self, results_dir='../../results'):
        """
        Save backtest results.
        
        Args:
            results_dir (str): Directory to save results
        """
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Run backtest if not already run
        if self.metrics is None:
            self.run_backtest()
        
        # Save metrics to JSON
        metrics_path = os.path.join(results_dir, f'{self.name}_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        # Save returns to CSV
        returns_path = os.path.join(results_dir, f'{self.name}_returns.csv')
        self.returns.to_csv(returns_path)
        
        # Save cumulative returns to CSV
        cum_returns_path = os.path.join(results_dir, f'{self.name}_cum_returns.csv')
        self.cum_returns.to_csv(cum_returns_path)
        
        # Save signals to CSV
        signals_path = os.path.join(results_dir, f'{self.name}_signals.csv')
        self.signals.to_csv(signals_path)
        
        # Save plot
        plot_path = os.path.join(results_dir, f'{self.name}_plot.png')
        fig = self.plot_results()
        fig.savefig(plot_path)
        plt.close(fig)
        
        print(f"Results saved to {results_dir}")
        
    def __str__(self):
        """String representation of the strategy."""
        return f"{self.name} Strategy"
        
    def __repr__(self):
        """Representation of the strategy."""
        return self.__str__() 