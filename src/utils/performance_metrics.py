#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix
)

class PerformanceMetrics:
    """
    A class to calculate performance metrics for trading strategies.
    """
    
    @staticmethod
    def calculate_returns(prices, positions):
        """
        Calculate strategy returns based on positions.
        
        Args:
            prices (pandas.Series): Series of prices
            positions (pandas.Series): Series of positions
            
        Returns:
            pandas.Series: Series of strategy returns
        """
        if len(positions) == 0 or len(prices) == 0:
            # Return empty Series if there's no data
            return pd.Series(dtype=float)
            
        # Calculate price changes
        price_changes = prices.pct_change()
        
        # Shift positions to reflect that we trade based on yesterday's signal
        # and get return for the next day
        shifted_positions = positions.shift(1)
        
        # Calculate strategy returns
        # For each day, we get the return based on our position from the previous day
        strategy_returns = price_changes * shifted_positions
        
        # Drop first row which will be NaN due to pct_change and shift
        strategy_returns = strategy_returns.dropna()
        
        # If we have no returns (maybe all positions were 0), return zeros
        if len(strategy_returns) == 0:
            return pd.Series(0, index=prices.index[1:])
            
        return strategy_returns
    
    @staticmethod
    def calculate_cumulative_returns(returns):
        """
        Calculate cumulative returns from a series of returns.
        
        Args:
            returns (pandas.Series): Series of returns
            
        Returns:
            pandas.Series: Series of cumulative returns
        """
        return (1 + returns).cumprod() - 1
    
    @staticmethod
    def calculate_annualized_return(returns, periods_per_year=252):
        """
        Calculate annualized return.
        
        Args:
            returns (pandas.Series): Series of returns
            periods_per_year (int): Number of periods in a year
            
        Returns:
            float: Annualized return
        """
        total_return = (1 + returns).prod() - 1
        n_periods = len(returns)
        
        # Handle division by zero or very small n_periods
        if n_periods <= 1:
            return total_return  # Just return the total return if we don't have enough data
            
        return (1 + total_return) ** (periods_per_year / n_periods) - 1
    
    @staticmethod
    def calculate_annualized_volatility(returns, periods_per_year=252):
        """
        Calculate annualized volatility.
        
        Args:
            returns (pandas.Series): Series of returns
            periods_per_year (int): Number of periods in a year
            
        Returns:
            float: Annualized volatility
        """
        return returns.std() * np.sqrt(periods_per_year)
    
    @staticmethod
    def calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
        """
        Calculate Sharpe ratio.
        
        Args:
            returns (pandas.Series): Series of returns
            risk_free_rate (float): Risk-free rate
            periods_per_year (int): Number of periods in a year
            
        Returns:
            float: Sharpe ratio
        """
        excess_returns = returns - (risk_free_rate / periods_per_year)
        ann_return = PerformanceMetrics.calculate_annualized_return(returns, periods_per_year)
        ann_volatility = PerformanceMetrics.calculate_annualized_volatility(returns, periods_per_year)
        
        if ann_volatility == 0:
            return 0
            
        return (ann_return - risk_free_rate) / ann_volatility
    
    @staticmethod
    def calculate_sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252, target_return=0.0):
        """
        Calculate Sortino ratio.
        
        Args:
            returns (pandas.Series): Series of returns
            risk_free_rate (float): Risk-free rate
            periods_per_year (int): Number of periods in a year
            target_return (float): Target return
            
        Returns:
            float: Sortino ratio
        """
        # Calculate excess returns
        excess_returns = returns - (risk_free_rate / periods_per_year)
        
        # Calculate downside deviation
        downside_returns = excess_returns[excess_returns < target_return]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0
        
        downside_deviation = downside_returns.std() * np.sqrt(periods_per_year)
        
        # Calculate annualized return
        ann_return = PerformanceMetrics.calculate_annualized_return(returns, periods_per_year)
        
        return (ann_return - risk_free_rate) / downside_deviation
    
    @staticmethod
    def calculate_maximum_drawdown(returns):
        """
        Calculate maximum drawdown.
        
        Args:
            returns (pandas.Series): Series of returns
            
        Returns:
            float: Maximum drawdown
        """
        # Calculate cumulative returns
        cum_returns = PerformanceMetrics.calculate_cumulative_returns(returns)
        
        # Calculate drawdowns
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / (1 + running_max)
        
        return drawdown.min()
    
    @staticmethod
    def calculate_calmar_ratio(returns, periods_per_year=252):
        """
        Calculate Calmar ratio.
        
        Args:
            returns (pandas.Series): Series of returns
            periods_per_year (int): Number of periods in a year
            
        Returns:
            float: Calmar ratio
        """
        max_drawdown = abs(PerformanceMetrics.calculate_maximum_drawdown(returns))
        ann_return = PerformanceMetrics.calculate_annualized_return(returns, periods_per_year)
        
        if max_drawdown == 0:
            return 0
            
        return ann_return / max_drawdown
    
    @staticmethod
    def calculate_win_rate(returns):
        """
        Calculate win rate.
        
        Args:
            returns (pandas.Series): Series of returns
            
        Returns:
            float: Win rate
        """
        if len(returns) == 0:
            return 0
            
        wins = (returns > 0).sum()
        return wins / len(returns)
    
    @staticmethod
    def calculate_profit_factor(returns):
        """
        Calculate profit factor.
        
        Args:
            returns (pandas.Series): Series of returns
            
        Returns:
            float: Profit factor
        """
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        
        if losses == 0:
            return float('inf') if gains > 0 else 0
            
        return gains / losses
    
    @staticmethod
    def calculate_expectancy(returns):
        """
        Calculate expectancy.
        
        Args:
            returns (pandas.Series): Series of returns
            
        Returns:
            float: Expectancy
        """
        win_rate = PerformanceMetrics.calculate_win_rate(returns)
        
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(wins) == 0:
            avg_win = 0
        else:
            avg_win = wins.mean()
            
        if len(losses) == 0:
            avg_loss = 0
        else:
            avg_loss = losses.mean()
            
        return (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    
    @staticmethod
    def calculate_ml_metrics(y_true, y_pred, y_prob=None):
        """
        Calculate machine learning metrics for classification.
        
        Args:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
            y_prob (array-like, optional): Predicted probabilities
            
        Returns:
            dict: Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            
        return metrics
    
    @staticmethod
    def calculate_all_metrics(returns, risk_free_rate=0.0, periods_per_year=252):
        """
        Calculate all performance metrics.
        
        Args:
            returns (pandas.Series): Series of returns
            risk_free_rate (float): Risk-free rate
            periods_per_year (int): Number of periods in a year
            
        Returns:
            dict: Dictionary of metrics
        """
        # Create default response with zeros
        default_metrics = {
            'total_return': 0,
            'annualized_return': 0,
            'annualized_volatility': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'maximum_drawdown': 0,
            'calmar_ratio': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'expectancy': 0
        }
        
        # Check if returns is empty or too small for meaningful calculations
        if returns is None or len(returns) <= 1:
            return default_metrics
            
        # Filter out NaN values
        returns = returns.dropna()
        
        # Check again after removing NaN values
        if len(returns) <= 1:
            return default_metrics
        
        try:
            metrics = {
                'total_return': float((1 + returns).prod() - 1),
                'annualized_return': float(PerformanceMetrics.calculate_annualized_return(returns, periods_per_year)),
                'annualized_volatility': float(PerformanceMetrics.calculate_annualized_volatility(returns, periods_per_year)),
                'sharpe_ratio': float(PerformanceMetrics.calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)),
                'sortino_ratio': float(PerformanceMetrics.calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)),
                'maximum_drawdown': float(PerformanceMetrics.calculate_maximum_drawdown(returns)),
                'calmar_ratio': float(PerformanceMetrics.calculate_calmar_ratio(returns, periods_per_year)),
                'win_rate': float(PerformanceMetrics.calculate_win_rate(returns)),
                'profit_factor': float(PerformanceMetrics.calculate_profit_factor(returns)),
                'expectancy': float(PerformanceMetrics.calculate_expectancy(returns))
            }
            return metrics
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return default_metrics

if __name__ == '__main__':
    # Example usage
    import yfinance as yf
    
    # Fetch data
    data = yf.download('AAPL', period='2y')
    
    # Calculate simple returns
    returns = data['Close'].pct_change().dropna()
    
    # Create a simple moving average crossover strategy
    short_ma = data['Close'].rolling(20).mean()
    long_ma = data['Close'].rolling(50).mean()
    
    # Generate signals (1 when short_ma > long_ma, 0 otherwise)
    signals = pd.Series(0, index=data.index)
    signals[short_ma > long_ma] = 1
    
    # Calculate strategy returns
    strategy_returns = PerformanceMetrics.calculate_returns(data['Close'], signals)
    
    # Calculate performance metrics
    metrics = PerformanceMetrics.calculate_all_metrics(strategy_returns)
    
    # Print results
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}") 