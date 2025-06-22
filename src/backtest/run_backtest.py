#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.data_fetcher import DataFetcher
from src.strategies.moving_average_crossover import MovingAverageCrossover
from src.strategies.rsi_strategy import RSIStrategy
from src.strategies.macd_strategy import MACDStrategy
from src.strategies.bollinger_bands import BollingerBands
from src.backtest.backtest_engine import BacktestEngine

# Optional imports for ML and LSTM strategies
try:
    from src.strategies.ml_strategy import MLStrategy

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: ML strategy not available.")

try:
    from src.strategies.lstm_strategy import LSTMStrategy

    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    print("Warning: LSTM strategy not available.")


class BacktestRunner:
    """
    Class to run backtests for multiple strategies.
    """

    def __init__(self, config_path=None):
        """
        Initialize the backtest runner.
        
        Args:
            config_path (str): Path to configuration file
        """
        # Resolve config path
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                       'config', 'config.json')

        # Load configuration
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            print(f"Config file not found at {config_path}, using default settings")
            self.config = {
                'backtest': {
                    'initial_capital': 100000,
                    'commission': 0.001,
                    'slippage': 0.001,
                    'risk_free_rate': 0.02
                }
            }

        # Initialize data fetcher
        self.fetcher = DataFetcher(config_path)

        # Dictionary to map strategy names to strategy classes
        self.strategy_classes = {
            'moving_average_crossover': MovingAverageCrossover,
            'rsi_strategy': RSIStrategy,
            'macd_strategy': MACDStrategy,
            'bollinger_bands': BollingerBands
        }

        # Add ML and LSTM strategies if available
        if ML_AVAILABLE:
            self.strategy_classes['ml_strategy'] = MLStrategy

        if LSTM_AVAILABLE:
            self.strategy_classes['lstm_strategy'] = LSTMStrategy

        # Initialize attributes
        self.data = None
        self.strategies = {}
        self.results = {}

    def fetch_data(self, symbol, period='1y', interval='1d', use_sample_data=True):
        """
        Fetch data for the specified symbol.
        
        Args:
            symbol (str): The stock symbol
            period (str): Period to fetch
            interval (str): Data interval
            
        Returns:
            pandas.DataFrame: The fetched data
        """
        self.data = self.fetcher.fetch_data(symbol, period=period, interval=interval, use_sample_data=use_sample_data)
        print(
            f"Fetched data for {symbol} from {self.data['Date'][0].strftime('%Y-%m-%d')} to {self.data['Date'].iloc[-1].strftime('%Y-%m-%d')}"
        )
        return self.data

    def add_strategy(self, strategy_name, **kwargs):
        """
        Add a strategy to the backtest.
        
        Args:
            strategy_name (str): Name of the strategy
            **kwargs: Additional parameters for the strategy
            
        Returns:
            object: The strategy instance
        """
        if strategy_name not in self.strategy_classes:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        # Create strategy instance
        strategy_class = self.strategy_classes[strategy_name]
        strategy = strategy_class(data=self.data, config=self.config, **kwargs)

        # Add to strategies dictionary
        self.strategies[strategy_name] = strategy

        return strategy

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

                # Generate signals
                strategy = self.strategies[name]
                signals = strategy.generate_signals()

                # Initialize backtest engine
                backtest = BacktestEngine(self.data, self.config)

                # Run backtest
                results = backtest.run(signals)

                # Save results
                results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                           'results', 'backtest')
                os.makedirs(results_dir, exist_ok=True)

                save_paths = backtest.save_results(name, results_dir)

                # Add results to dictionary
                self.results[name] = results

                # Print summary
                print(f"Results saved to {results_dir}")
                print("Performance summary:")
                for metric, value in results['metrics'].items():
                    if isinstance(value, float):
                        if metric in ['total_return', 'annualized_return', 'maximum_drawdown']:
                            print(f"  {metric}: {value:.2%}")
                        else:
                            print(f"  {metric}: {value:.4f}")
            else:
                print(f"Strategy {name} not found!")

        return self.results

    def compare_strategies(self, strategy_names=None, metrics=None):
        """
        Compare performance of multiple strategies.
        
        Args:
            strategy_names (list, optional): List of strategy names to compare
            metrics (list, optional): List of metrics to compare
            
        Returns:
            pandas.DataFrame: Comparison of metrics
        """
        if not self.results:
            raise ValueError("No backtest results available. Run 'run_backtest()' first.")

        # Default to all strategies if not specified
        if strategy_names is None:
            strategy_names = list(self.results.keys())

        # Default metrics to compare
        if metrics is None:
            metrics = [
                'total_return', 'annualized_return', 'sharpe_ratio',
                'sortino_ratio', 'maximum_drawdown', 'win_rate'
            ]

        # Create comparison dataframe
        comparison = {}

        for name in strategy_names:
            if name in self.results:
                comparison[name] = {metric: self.results[name]['metrics'][metric] for metric in metrics}

        return pd.DataFrame(comparison)

    def plot_comparison(self, strategy_names=None, figsize=(16, 10)):
        """
        Plot comparison of cumulative returns for multiple strategies.
        
        Args:
            strategy_names (list, optional): List of strategy names to compare
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: Comparison plot
        """
        if not self.results:
            raise ValueError("No backtest results available. Run 'run_backtest()' first.")

        # Default to all strategies if not specified
        if strategy_names is None:
            strategy_names = list(self.results.keys())

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Buy and Hold
        if self.data is not None:
            returns = self.data['Close'].pct_change().fillna(0)
            cum_returns = (1 + returns).cumprod() - 1
            ax.plot(cum_returns.index, cum_returns, label='Buy & Hold', linestyle='--')

        # Plot each strategy's cumulative returns
        for name in strategy_names:
            if name in self.results:
                cum_returns = self.results[name]['cum_returns']
                ax.plot(cum_returns.index, cum_returns, label=name)

        # Add labels and legend
        ax.set_title('Cumulative Returns Comparison')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Returns')
        ax.legend()
        ax.grid(True)

        # Save figure
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                   'results', 'backtest')
        os.makedirs(results_dir, exist_ok=True)

        plot_path = os.path.join(results_dir, 'strategy_comparison.png')
        fig.savefig(plot_path)
        print(f"Comparison plot saved to {plot_path}")

        return fig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run backtests for trading strategies')
    parser.add_argument('--config', help='Path to config file')
    parser.add_argument('--symbol', default='NIFTY 50', help='Stock symbol to use')
    parser.add_argument('--period', default='1y', help='Period to fetch (e.g., 1y, 5y)')
    parser.add_argument('--interval', default='1d', help='Data interval (e.g., 1d, 1h)')
    parser.add_argument('--strategies', help='Comma-separated list of strategies to run')
    return parser.parse_args()


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()

    # Initialize backtest runner
    runner = BacktestRunner(args.config)

    # Fetch data
    data = runner.fetch_data(args.symbol, args.period, args.interval)

    # Add strategies
    if args.strategies:
        strategies = args.strategies.split(',')
    else:
        strategies = list(runner.strategy_classes.keys())

    for strategy in strategies:
        try:
            runner.add_strategy(strategy)
        except ValueError as e:
            print(f"Error adding strategy {strategy}: {e}")

    # Run backtests
    results = runner.run_backtest()

    # Plot comparison
    runner.plot_comparison()

    # Print comparison table
    comparison = runner.compare_strategies()
    print("\nStrategy Comparison:")

    # Format table for better readability
    try:
        # Check if the DataFrame is in the expected format
        formatted = comparison.copy()

        # Create a more readable format
        for column in formatted.columns:
            metrics = formatted[column]
            print(f"\n{column}:")
            for metric, value in metrics.items():
                if metric in ['total_return', 'annualized_return', 'maximum_drawdown', 'win_rate']:
                    print(f"  {metric}: {value:.2%}")
                else:
                    print(f"  {metric}: {value:.4f}")
    except Exception as e:
        print(f"Error formatting comparison table: {e}")
        print("Raw comparison data:")
        print(comparison)

    plt.show()
