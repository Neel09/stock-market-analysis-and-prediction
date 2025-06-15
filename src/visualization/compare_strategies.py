#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import json
import argparse
from tabulate import tabulate

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.data_fetcher import DataFetcher
from src.utils.data_processor import DataProcessor
from src.strategies.moving_average_crossover import MovingAverageCrossover
from src.strategies.rsi_strategy import RSIStrategy
from src.strategies.macd_strategy import MACDStrategy
from src.strategies.bollinger_bands import BollingerBands
from src.strategies.ml_strategy import MLStrategy
from src.strategies.lstm_strategy import LSTMStrategy

class StrategyComparison:
    """
    Compare multiple trading strategies.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the strategy comparison.
        
        Args:
            config_path (str, optional): Path to configuration file
        """
        # Load configuration
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                    'config', 'config.json')
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize data fetcher and processor
        self.fetcher = DataFetcher(config_path)
        self.processor = DataProcessor()
        
        # Dictionary to map strategy names to strategy classes
        self.strategy_classes = {
            'moving_average_crossover': MovingAverageCrossover,
            'rsi': RSIStrategy,
            'macd': MACDStrategy,
            'bollinger_bands': BollingerBands,
            'ml_logistic': lambda data, config: MLStrategy(data, config, model_type='logistic'),
            'ml_linear': lambda data, config: MLStrategy(data, config, model_type='linear'),
            'lstm': LSTMStrategy
        }
        
        # Initialize attributes
        self.data = None
        self.strategies = {}
        self.results = {}
        
    def fetch_data(self, symbol='AAPL', period='2y', interval='1d'):
        """
        Fetch data for the specified symbol.
        
        Args:
            symbol (str): The stock symbol
            period (str): Period to fetch
            interval (str): Data interval
            
        Returns:
            pandas.DataFrame: The fetched data
        """
        self.data = self.fetcher.fetch_data(symbol, period=period, interval=interval)
        return self.data
    
    def add_strategy(self, strategy_name, strategy_instance=None, **kwargs):
        """
        Add a strategy to the comparison.
        
        Args:
            strategy_name (str): Name of the strategy
            strategy_instance (BaseStrategy, optional): Strategy instance
            **kwargs: Additional parameters for the strategy
            
        Returns:
            BaseStrategy: The added strategy
        """
        if strategy_instance is not None:
            # Use provided strategy instance
            self.strategies[strategy_name] = strategy_instance
        elif strategy_name in self.strategy_classes:
            # Create a new strategy instance
            if callable(self.strategy_classes[strategy_name]):
                # Handle custom lambda functions for strategies with special initialization
                self.strategies[strategy_name] = self.strategy_classes[strategy_name](self.data, self.config, **kwargs)
            else:
                # Standard strategy initialization
                self.strategies[strategy_name] = self.strategy_classes[strategy_name](
                    data=self.data, config=self.config, **kwargs
                )
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        return self.strategies[strategy_name]
    
    def run_backtest(self, strategy_names=None):
        """
        Run backtest for the specified strategies.
        
        Args:
            strategy_names (list, optional): List of strategy names to run
            
        Returns:
            dict: Dictionary of backtest results
        """
        if strategy_names is None:
            strategy_names = list(self.strategies.keys())
        
        # Run backtest for each strategy
        for name in strategy_names:
            if name in self.strategies:
                print(f"Running backtest for {name}...")
                self.results[name] = self.strategies[name].run_backtest()
            else:
                print(f"Strategy {name} not found!")
        
        return self.results
    
    def compare_metrics(self, metrics=None):
        """
        Compare metrics across strategies.
        
        Args:
            metrics (list, optional): List of metrics to compare
            
        Returns:
            pandas.DataFrame: Comparison of metrics
        """
        if not self.results:
            raise ValueError("No backtest results available. Run 'run_backtest()' first.")
        
        # Default metrics to compare
        if metrics is None:
            metrics = [
                'total_return', 'annualized_return', 'annualized_volatility',
                'sharpe_ratio', 'sortino_ratio', 'maximum_drawdown',
                'win_rate', 'profit_factor'
            ]
        
        # Create comparison dataframe
        comparison = {}
        
        for name, result in self.results.items():
            comparison[name] = {metric: result['metrics'][metric] for metric in metrics if metric in result['metrics']}
        
        return pd.DataFrame(comparison)
    
    def plot_cumulative_returns(self, benchmark=None, figsize=(12, 8)):
        """
        Plot cumulative returns for all strategies.
        
        Args:
            benchmark (str, optional): Benchmark symbol (e.g., 'SPY')
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: Cumulative returns plot
        """
        if not self.results:
            raise ValueError("No backtest results available. Run 'run_backtest()' first.")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot cumulative returns for each strategy
        for name, result in self.results.items():
            ax.plot(result['cum_returns'], label=name)
        
        # Add benchmark if provided
        if benchmark is not None:
            # Fetch benchmark data
            benchmark_data = self.fetcher.fetch_data(benchmark, period='2y')
            
            # Calculate benchmark returns
            benchmark_returns = benchmark_data['Close'].pct_change().fillna(0)
            
            # Calculate cumulative returns
            benchmark_cum_returns = (1 + benchmark_returns).cumprod() - 1
            
            # Plot benchmark
            ax.plot(benchmark_cum_returns, label=benchmark, linestyle='--')
        
        # Add labels and legend
        ax.set_title('Cumulative Returns Comparison')
        ax.set_ylabel('Cumulative Returns')
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def plot_metrics_comparison(self, metrics=None, figsize=(15, 10)):
        """
        Plot metrics comparison across strategies.
        
        Args:
            metrics (list, optional): List of metrics to compare
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: Metrics comparison plot
        """
        # Get metrics comparison
        comparison = self.compare_metrics(metrics)
        
        # Default metrics to compare
        if metrics is None:
            metrics = comparison.index.tolist()
        
        # Create figure
        fig, axes = plt.subplots(len(metrics), 1, figsize=figsize)
        
        # Adjust for single metric case
        if len(metrics) == 1:
            axes = [axes]
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            ax = axes[i]
            comparison.loc[metric].plot(kind='bar', ax=ax)
            ax.set_title(f'{metric}')
            ax.set_ylabel(metric)
            ax.grid(True)
        
        plt.tight_layout()
        
        return fig
    
    def plot_drawdowns(self, figsize=(12, 8)):
        """
        Plot drawdowns for all strategies.
        
        Args:
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: Drawdowns plot
        """
        if not self.results:
            raise ValueError("No backtest results available. Run 'run_backtest()' first.")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate and plot drawdowns for each strategy
        for name, result in self.results.items():
            # Calculate drawdowns
            cum_returns = result['cum_returns']
            running_max = cum_returns.cummax()
            drawdown = (cum_returns - running_max) / (1 + running_max)
            
            # Plot drawdown
            ax.plot(drawdown, label=name)
        
        # Add labels and legend
        ax.set_title('Drawdowns Comparison')
        ax.set_ylabel('Drawdown')
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def print_metrics_table(self, metrics=None):
        """
        Print a table of metrics for all strategies.
        
        Args:
            metrics (list, optional): List of metrics to include
        """
        # Get metrics comparison
        comparison = self.compare_metrics(metrics)
        
        # Format the table for printing
        comparison = comparison.T  # Transpose for better readability
        
        # Format the values (percentage for returns, etc.)
        formatted = comparison.copy()
        
        for col in formatted.columns:
            if col in ['total_return', 'annualized_return', 'maximum_drawdown', 'win_rate']:
                formatted[col] = formatted[col].apply(lambda x: f"{x:.2%}")
            elif col in ['sharpe_ratio', 'sortino_ratio', 'profit_factor', 'calmar_ratio']:
                formatted[col] = formatted[col].apply(lambda x: f"{x:.2f}")
            elif col in ['annualized_volatility']:
                formatted[col] = formatted[col].apply(lambda x: f"{x:.2%}")
        
        # Print table
        print("\nStrategy Performance Comparison:")
        print(tabulate(formatted, headers='keys', tablefmt='pretty'))
    
    def plot_combined_analysis(self, benchmark=None, figsize=(15, 18)):
        """
        Create a combined analysis plot with multiple subplots.
        
        Args:
            benchmark (str, optional): Benchmark symbol (e.g., 'SPY')
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: Combined analysis plot
        """
        if not self.results:
            raise ValueError("No backtest results available. Run 'run_backtest()' first.")
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(4, 1, figsize=figsize)
        
        # 1. Cumulative Returns
        ax1 = axes[0]
        for name, result in self.results.items():
            ax1.plot(result['cum_returns'], label=name)
        
        # Add benchmark if provided
        if benchmark is not None:
            # Fetch benchmark data
            benchmark_data = self.fetcher.fetch_data(benchmark, period='2y')
            
            # Calculate benchmark returns
            benchmark_returns = benchmark_data['Close'].pct_change().fillna(0)
            
            # Calculate cumulative returns
            benchmark_cum_returns = (1 + benchmark_returns).cumprod() - 1
            
            # Plot benchmark
            ax1.plot(benchmark_cum_returns, label=benchmark, linestyle='--')
        
        ax1.set_title('Cumulative Returns Comparison')
        ax1.set_ylabel('Cumulative Returns')
        ax1.legend()
        ax1.grid(True)
        
        # 2. Drawdowns
        ax2 = axes[1]
        for name, result in self.results.items():
            # Calculate drawdowns
            cum_returns = result['cum_returns']
            running_max = cum_returns.cummax()
            drawdown = (cum_returns - running_max) / (1 + running_max)
            
            # Plot drawdown
            ax2.plot(drawdown, label=name)
        
        ax2.set_title('Drawdowns Comparison')
        ax2.set_ylabel('Drawdown')
        ax2.legend()
        ax2.grid(True)
        
        # 3. Rolling Sharpe Ratio (252-day window)
        ax3 = axes[2]
        for name, result in self.results.items():
            # Calculate rolling Sharpe ratio
            returns = result['returns']
            rolling_return = returns.rolling(window=252).mean() * 252
            rolling_vol = returns.rolling(window=252).std() * np.sqrt(252)
            rolling_sharpe = rolling_return / rolling_vol
            
            # Plot rolling Sharpe ratio
            ax3.plot(rolling_sharpe, label=name)
        
        ax3.set_title('Rolling Sharpe Ratio (252-day window)')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.legend()
        ax3.grid(True)
        
        # 4. Performance Metrics Heatmap
        ax4 = axes[3]
        comparison = self.compare_metrics()
        
        # Normalize metrics for better visualization
        normalized = comparison.copy()
        for metric in normalized.index:
            if metric in ['maximum_drawdown']:  # Metrics where lower is better
                normalized.loc[metric] = (normalized.loc[metric] - normalized.loc[metric].max()) / (normalized.loc[metric].min() - normalized.loc[metric].max())
            else:  # Metrics where higher is better
                normalized.loc[metric] = (normalized.loc[metric] - normalized.loc[metric].min()) / (normalized.loc[metric].max() - normalized.loc[metric].min())
        
        # Create heatmap
        sns.heatmap(normalized, annot=True, cmap='RdYlGn', ax=ax4, fmt='.2f')
        ax4.set_title('Performance Metrics Comparison (Normalized)')
        
        plt.tight_layout()
        
        return fig
    
    def save_results(self, output_dir='../../results'):
        """
        Save comparison results to files.
        
        Args:
            output_dir (str): Output directory
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics comparison
        comparison = self.compare_metrics()
        comparison.to_csv(os.path.join(output_dir, 'metrics_comparison.csv'))
        
        # Save cumulative returns plot
        fig = self.plot_cumulative_returns()
        fig.savefig(os.path.join(output_dir, 'cumulative_returns.png'))
        plt.close(fig)
        
        # Save drawdowns plot
        fig = self.plot_drawdowns()
        fig.savefig(os.path.join(output_dir, 'drawdowns.png'))
        plt.close(fig)
        
        # Save combined analysis plot
        fig = self.plot_combined_analysis()
        fig.savefig(os.path.join(output_dir, 'combined_analysis.png'))
        plt.close(fig)
        
        # Save metrics comparison plot
        fig = self.plot_metrics_comparison()
        fig.savefig(os.path.join(output_dir, 'metrics_comparison.png'))
        plt.close(fig)
        
        print(f"Results saved to {output_dir}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compare trading strategies')
    parser.add_argument('--strategies', type=str, help='Comma-separated list of strategies to compare')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol to use')
    parser.add_argument('--period', type=str, default='2y', help='Time period to fetch')
    parser.add_argument('--interval', type=str, default='1d', help='Data interval')
    parser.add_argument('--benchmark', type=str, default='SPY', help='Benchmark symbol')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--output', type=str, default='../../results', help='Output directory')
    return parser.parse_args()

def compare_strategies(results_dict):
    """
    Compare multiple trading strategies.
    
    Args:
        results_dict (dict): Dictionary of strategy results
        
    Returns:
        dict: Dictionary of comparison results
    """
    # Check if results are available
    if not results_dict:
        print("No results available for comparison")
        return {}
    
    # Get metrics from the first strategy to determine available metrics
    first_strategy = list(results_dict.keys())[0]
    available_metrics = list(results_dict[first_strategy]['metrics'].keys())
    
    # Initialize results dictionary
    comparison_results = {
        'strategy_names': list(results_dict.keys())
    }
    
    # Add metrics to results
    for metric in available_metrics:
        comparison_results[metric] = []
        for strategy_name, result in results_dict.items():
            comparison_results[metric].append(result['metrics'][metric])
    
    return comparison_results

def plot_cumulative_returns(results_dict, save_path=None):
    """
    Plot cumulative returns for multiple strategies.
    
    Args:
        results_dict (dict): Dictionary of strategy results
        save_path (str, optional): Path to save the plot
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Check if results are available
    if not results_dict:
        print("No results available for plotting")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Plot each strategy's cumulative returns
    for strategy_name, result in results_dict.items():
        ax.plot(result['cum_returns'].index, result['cum_returns'], label=strategy_name)
    
    # Set plot labels
    ax.set_title('Cumulative Returns Comparison')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Returns')
    ax.legend()
    ax.grid(True)
    
    # Save the plot if a path is provided
    if save_path:
        fig.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    return fig

def create_metrics_table(results_dict, metrics=None, format_as_percentage=None):
    """
    Create a table of performance metrics for multiple strategies.
    
    Args:
        results_dict (dict): Dictionary of strategy results
        metrics (list, optional): List of metrics to include
        format_as_percentage (list, optional): List of metrics to format as percentages
        
    Returns:
        pandas.DataFrame: DataFrame containing the metrics table
    """
    # Check if results are available
    if not results_dict:
        print("No results available for creating table")
        return None
    
    # Get metrics from the first strategy to determine available metrics
    first_strategy = list(results_dict.keys())[0]
    available_metrics = list(results_dict[first_strategy]['metrics'].keys())
    
    # Use provided metrics or all available metrics
    if metrics is None:
        metrics = available_metrics
    
    # Default percentage format for common metrics
    if format_as_percentage is None:
        format_as_percentage = ['total_return', 'annualized_return', 'maximum_drawdown', 'win_rate']
    
    # Create data for the table
    data = []
    for strategy_name, result in results_dict.items():
        row = [strategy_name]
        for metric in metrics:
            if metric in result['metrics']:
                row.append(result['metrics'][metric])
            else:
                row.append(np.nan)
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=['Strategy'] + metrics)
    
    # Format percentage values
    for col in format_as_percentage:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.2%}" if isinstance(x, (int, float)) else x)
    
    # Format other values
    for col in [m for m in metrics if m not in format_as_percentage]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
    
    return df

if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()
    
    # Initialize strategy comparison
    comparison = StrategyComparison(args.config)
    
    # Fetch data
    data = comparison.fetch_data(args.symbol, args.period, args.interval)
    
    # Add strategies
    if args.strategies:
        strategies = args.strategies.split(',')
    else:
        # Default strategies to compare
        strategies = [
            'moving_average_crossover',
            'rsi',
            'macd',
            'bollinger_bands',
            'ml_logistic'
        ]
    
    for strategy in strategies:
        comparison.add_strategy(strategy)
    
    # Run backtest
    comparison.run_backtest()
    
    # Print metrics table
    comparison.print_metrics_table()
    
    # Plot results
    comparison.plot_cumulative_returns(args.benchmark)
    plt.figure()
    comparison.plot_drawdowns()
    plt.figure()
    comparison.plot_metrics_comparison()
    plt.figure()
    comparison.plot_combined_analysis(args.benchmark)
    
    # Save results
    comparison.save_results(args.output)
    
    # Show plots
    plt.show() 