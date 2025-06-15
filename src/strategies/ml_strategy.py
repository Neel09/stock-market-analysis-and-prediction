#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sys
import os
import joblib

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.strategies.base_strategy import BaseStrategy
from src.models.ml_linear_model import MLLinearModel
from src.utils.data_processor import DataProcessor

class MLStrategy(BaseStrategy):
    """
    Machine Learning-based Trading Strategy.
    
    This strategy uses machine learning models to generate trading signals.
    """
    
    def __init__(self, data=None, config=None, config_path=None, model_type='logistic',
                features=None, target_column='return_1d', train_size=0.7, threshold=0.5):
        """
        Initialize the ML strategy.
        
        Args:
            data (pandas.DataFrame, optional): The OHLCV dataframe
            config (dict, optional): Configuration parameters
            config_path (str, optional): Path to configuration file
            model_type (str): Type of model ('logistic', 'linear', etc.)
            features (list, optional): List of features to use
            target_column (str): Target column for prediction
            train_size (float): Proportion of data to use for training
            threshold (float): Threshold for generating signals
        """
        strategy_name = f"ml_{model_type}"
        super().__init__(strategy_name, data, config, config_path)
        
        # Set strategy parameters
        self.model_type = model_type
        self.features = features or self.strategy_config.get('features', None)
        self.target_column = target_column or self.strategy_config.get('target_column', 'return_1d')
        self.train_size = train_size or self.strategy_config.get('train_size', 0.7)
        self.threshold = threshold or self.strategy_config.get('threshold', 0.5)
        
        # Initialize ML model
        self.ml_model = None
        self.processor = DataProcessor()
        self.processed_data = None
        
    def preprocess_data(self):
        """
        Preprocess data for ML model.
        
        Returns:
            pandas.DataFrame: Processed dataframe
        """
        if self.data is None:
            raise ValueError("Data must be set before preprocessing")
        
        # Add technical indicators
        self.processed_data = self.processor.add_technical_indicators(self.data)
        
        return self.processed_data
    
    def train_model(self):
        """
        Train the ML model.
        
        Returns:
            self: Trained model
        """
        if self.processed_data is None:
            self.preprocess_data()
        
        # Initialize model
        self.ml_model = MLLinearModel(model_type=self.model_type)
        
        # Prepare features
        X, y, feature_names = self.ml_model.prepare_features(
            self.processed_data,
            target_column=self.target_column,
            features=self.features
        )
        
        # Split data for training
        train_size = int(len(X) * self.train_size)
        X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
        
        # Train model
        self.ml_model.train(X_train, y_train)
        
        return self
    
    def generate_signals(self):
        """
        Generate trading signals based on ML model predictions.
        
        Returns:
            pandas.Series: Series of trading signals (1: long, 0: no position, -1: short)
        """
        if self.data is None:
            raise ValueError("Data must be set before generating signals")
        
        # Preprocess data if not already done
        if self.processed_data is None:
            self.preprocess_data()
        
        # Train model if not already trained
        if self.ml_model is None:
            self.train_model()
        
        # Prepare features for the entire dataset
        X, _, _ = self.ml_model.prepare_features(
            self.processed_data,
            target_column=self.target_column,
            features=self.features
        )
        
        # Make predictions
        predictions = self.ml_model.predict(X)
        
        # Convert predictions to signals
        signals = pd.Series(0, index=self.data.index)
        
        if self.model_type == 'logistic':
            # For logistic regression, we use probability threshold
            signals[predictions > self.threshold] = 1  # Buy signal
            signals[predictions < (1 - self.threshold)] = -1  # Sell signal
        else:
            # For linear regression, we predict returns
            signals[predictions > 0] = 1  # Buy signal
            signals[predictions < 0] = -1  # Sell signal
        
        # Store the signals
        self.signals = signals
        
        return signals
    
    def save_model(self, model_dir='../../models'):
        """
        Save the trained model.
        
        Args:
            model_dir (str): Directory to save the model
        """
        if self.ml_model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, f"{self.name}_model.joblib")
        self.ml_model.save_model(model_path)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """
        Load a trained model.
        
        Args:
            model_path (str): Path to the model file
            
        Returns:
            self: Strategy with loaded model
        """
        # Load model
        self.ml_model = MLLinearModel.load_model(model_path)
        
        # Update strategy parameters
        self.model_type = self.ml_model.model_type
        self.features = self.ml_model.feature_names
        
        return self
    
    def evaluate_model(self):
        """
        Evaluate the ML model.
        
        Returns:
            dict: Model evaluation metrics
        """
        if self.ml_model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        if self.processed_data is None:
            self.preprocess_data()
        
        # Prepare features
        X, y, _ = self.ml_model.prepare_features(
            self.processed_data,
            target_column=self.target_column,
            features=self.features
        )
        
        # Split data
        train_size = int(len(X) * self.train_size)
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        
        # Evaluate model
        train_metrics = self.ml_model.evaluate(X_train, y_train)
        test_metrics = self.ml_model.evaluate(X_test, y_test)
        
        # Cross-validation
        cv_results = self.ml_model.cross_validate(X, y)
        
        # Feature importance
        importance = self.ml_model.feature_importance()
        
        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_results': cv_results,
            'feature_importance': importance
        }

if __name__ == '__main__':
    # Example usage
    import sys
    import os
    import json
    import matplotlib.pyplot as plt
    
    # Add the parent directory to the path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    from src.utils.data_fetcher import DataFetcher
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                              'config', 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Fetch data
    fetcher = DataFetcher(config_path)
    data = fetcher.fetch_data('AAPL', period='2y')
    
    # Initialize strategy
    strategy = MLStrategy(data=data, config=config, model_type='logistic')
    
    # Run backtest
    results = strategy.run_backtest()
    
    # Print results
    print(f"Strategy: {results['strategy_name']}")
    for metric, value in results['metrics'].items():
        if isinstance(value, (int, float)):
            print(f"{metric}: {value:.4f}")
    
    # Evaluate model
    model_eval = strategy.evaluate_model()
    
    print("\nModel Evaluation:")
    print("Training metrics:")
    for metric, value in model_eval['train_metrics'].items():
        print(f"{metric}: {value:.4f}")
    
    print("\nTest metrics:")
    for metric, value in model_eval['test_metrics'].items():
        print(f"{metric}: {value:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    model_eval['feature_importance'].head(10).plot(kind='bar')
    plt.title('Top 10 Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    
    # Plot results
    strategy.plot_results()
    plt.show()
    
    # Save results
    strategy.save_results()
    
    # Save model
    strategy.save_model() 