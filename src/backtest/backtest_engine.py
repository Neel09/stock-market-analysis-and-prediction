#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json

class BacktestEngine:
    """
    A simple backtest engine for trading strategies.
    """
    
    def __init__(self, data, config=None, initial_capital=100000, commission=0.001, slippage=0.001):
        """
        Initialize the backtest engine.
        
        Args:
            data (pandas.DataFrame): OHLCV data
            config (dict, optional): Configuration dictionary
            initial_capital (float): Initial capital
            commission (float): Commission rate
            slippage (float): Slippage rate
        """
        self.data = data
        
        # Load configuration if provided
        if config is not None:
            self.config = config
            
            # Get backtest parameters from config
            backtest_config = config.get('backtest', {})
            initial_capital = backtest_config.get('initial_capital', initial_capital)
            commission = backtest_config.get('commission', commission)
            slippage = backtest_config.get('slippage', slippage)
        else:
            self.config = {}
        
        # Set backtest parameters
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        # Initialize backtest results
        self.positions = None
        self.portfolio_value = None
        self.returns = None
        self.metrics = None
    
    def run(self, signals):
        """
        Run backtest with given trading signals.
        
        Args:
            signals (pandas.Series): Trading signals
            
        Returns:
            dict: Backtest results
        """
        # Ensure data and signals have the same index
        if not signals.index.equals(self.data.index):
            raise ValueError("Signals index must match data index")
        
        # Calculate positions (1: long, 0: no position, -1: short)
        self.positions = signals.copy()
        
        # Calculate asset price changes
        price_changes = self.data['Close'].pct_change().fillna(0)
        
        # Apply transaction costs (commission and slippage)
        # Only apply costs when position changes
        position_changes = self.positions.diff().fillna(0).abs()
        transaction_costs = position_changes * (self.commission + self.slippage)
        
        # Calculate strategy returns
        strategy_returns = self.positions.shift(1) * price_changes - transaction_costs
        self.returns = strategy_returns.fillna(0)
        
        # Calculate cumulative returns
        cum_returns = (1 + self.returns).cumprod() - 1
        
        # Calculate portfolio value
        self.portfolio_value = self.initial_capital * (1 + cum_returns)
        
        # Calculate performance metrics
        self.metrics = self.calculate_metrics()
        
        # Prepare results
        results = {
            'positions': self.positions,
            'returns': self.returns,
            'cum_returns': cum_returns,
            'portfolio_value': self.portfolio_value,
            'metrics': self.metrics
        }
        
        return results
    
    def calculate_metrics(self):
        """
        Calculate performance metrics.
        
        Returns:
            dict: Performance metrics
        """
        if self.returns is None:
            raise ValueError("No backtest results available. Run backtest first.")
        
        # Get returns
        returns = self.returns
        
        # Convert returns to numpy array for calculations
        if hasattr(returns, 'values'):
            # Handle pandas Series
            returns_array = returns.values
        else:
            # Already numpy array or list
            returns_array = np.array(returns)
        
        # Ensure we're working with a 1D array
        returns_array = returns_array.ravel()
        
        # Total return
        total_return = np.prod(1 + returns_array) - 1
        
        # Annualized return
        n_periods = len(returns_array)
        periods_per_year = 252  # Trading days in a year
        annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
        
        # Annualized volatility
        annualized_volatility = np.std(returns_array) * np.sqrt(periods_per_year)
        
        # Sharpe ratio
        risk_free_rate = self.config.get('backtest', {}).get('risk_free_rate', 0.02)
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility != 0 else 0
        
        # Sortino ratio (downside risk)
        downside_returns = returns_array[returns_array < 0]
        downside_volatility = np.std(downside_returns) * np.sqrt(periods_per_year) if len(downside_returns) > 0 else 0
        sortino_ratio = (annualized_return - risk_free_rate) / downside_volatility if downside_volatility != 0 else 0
        
        # Maximum drawdown
        cum_returns = np.cumprod(1 + returns_array) - 1
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / (1 + running_max)
        max_drawdown = np.min(drawdown)
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate
        win_rate = np.sum(returns_array > 0) / len(returns_array) if len(returns_array) > 0 else 0
        
        # Profit factor
        gross_profit = np.sum(returns_array[returns_array > 0]) if np.any(returns_array > 0) else 0
        gross_loss = abs(np.sum(returns_array[returns_array < 0])) if np.any(returns_array < 0) else 0
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        # Average win/loss
        avg_win = np.mean(returns_array[returns_array > 0]) if np.any(returns_array > 0) else 0
        avg_loss = np.mean(returns_array[returns_array < 0]) if np.any(returns_array < 0) else 0
        
        # Expectancy
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        # Metrics dictionary
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'maximum_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'expectancy': expectancy
        }
        
        return metrics
    
    def plot_results(self, benchmark=None, figsize=(15, 10)):
        """
        Plot backtest results.
        
        Args:
            benchmark (pandas.Series, optional): Benchmark returns
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        if self.returns is None:
            raise ValueError("No backtest results available. Run backtest first.")
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # Plot portfolio value
        ax1 = axes[0]
        ax1.plot(self.portfolio_value, label='Portfolio Value')
        ax1.set_title('Backtest Results')
        ax1.set_ylabel('Portfolio Value')
        ax1.legend()
        ax1.grid(True)
        
        # Plot returns
        ax2 = axes[1]
        cum_returns = (1 + self.returns).cumprod() - 1
        ax2.plot(cum_returns, label='Strategy')
        
        # Plot benchmark if provided
        if benchmark is not None:
            benchmark_cum_returns = (1 + benchmark).cumprod() - 1
            ax2.plot(benchmark_cum_returns, label='Benchmark')
            
        ax2.set_ylabel('Cumulative Returns')
        ax2.legend()
        ax2.grid(True)
        
        # Plot positions
        ax3 = axes[2]
        ax3.plot(self.positions, label='Position', color='green')
        ax3.set_ylabel('Position')
        ax3.set_xlabel('Date')
        ax3.set_yticks([-1, 0, 1])
        ax3.set_yticklabels(['Short', 'No Position', 'Long'])
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        
        return fig
    
    def save_results(self, strategy_name, results_dir='../../results'):
        """
        Save backtest results to files.
        
        Args:
            strategy_name (str): Name of the strategy
            results_dir (str): Results directory path
            
        Returns:
            dict: Paths to saved files
        """
        if self.returns is None:
            raise ValueError("No backtest results available. Run backtest first.")
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Save metrics
        metrics_path = os.path.join(results_dir, f"{strategy_name}_metrics.json")
        with open(metrics_path, 'w') as f:
            # Convert NumPy types to Python types to make it JSON serializable
            metrics_serializable = {}
            for key, value in self.metrics.items():
                if isinstance(value, (np.floating, np.integer)):
                    metrics_serializable[key] = float(value)
                else:
                    metrics_serializable[key] = value
            
            json.dump(metrics_serializable, f, indent=4)
        
        # Save returns
        returns_path = os.path.join(results_dir, f"{strategy_name}_returns.csv")
        self.returns.to_csv(returns_path)
        
        # Save cumulative returns
        cum_returns = (1 + self.returns).cumprod() - 1
        cum_returns_path = os.path.join(results_dir, f"{strategy_name}_cum_returns.csv")
        cum_returns.to_csv(cum_returns_path)
        
        # Save portfolio value
        portfolio_path = os.path.join(results_dir, f"{strategy_name}_portfolio.csv")
        self.portfolio_value.to_csv(portfolio_path)
        
        # Save positions
        positions_path = os.path.join(results_dir, f"{strategy_name}_positions.csv")
        self.positions.to_csv(positions_path)
        
        # Save plot
        plot_path = os.path.join(results_dir, f"{strategy_name}_plot.png")
        fig = self.plot_results()
        fig.savefig(plot_path)
        plt.close(fig)
        
        return {
            'metrics': metrics_path,
            'returns': returns_path,
            'cum_returns': cum_returns_path,
            'portfolio': portfolio_path,
            'positions': positions_path,
            'plot': plot_path
        }

if __name__ == '__main__':
    # Example usage
    import sys
    
    # Add the project root to the Python path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    from src.utils.data_fetcher import DataFetcher
    
    # Fetch data
    fetcher = DataFetcher()
    data = fetcher.fetch_data('NIFTY 50', period='1y')
    
    # Create random signals for demonstration
    np.random.seed(42)
    signals = pd.Series(np.random.choice([-1, 0, 1], size=len(data)), index=data.index)
    
    # Initialize backtest engine
    backtest = BacktestEngine(data)
    
    # Run backtest
    results = backtest.run(signals)
    
    # Print metrics
    print("Backtest Results:")
    for key, value in results['metrics'].items():
        print(f"{key}: {value:.4f}")
    
    # Plot results
    backtest.plot_results()
    plt.show() 