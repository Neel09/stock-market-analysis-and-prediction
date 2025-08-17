#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

from src.utils.nifty_data_fetcher import NiftyDataFetcher

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.strategies.base_strategy import BaseStrategy
from src.models.dl_lstm_model import LSTMModel
from src.utils.data_processor import DataProcessor


class LSTMStrategy(BaseStrategy):
    """
    LSTM-based Trading Strategy.
    
    This strategy uses LSTM (Long Short-Term Memory) networks to generate trading signals.
    """

    def __init__(self, data=None, config=None, config_path=None,
                 sequence_length=None, features=None, target_column='return_1d',
                 train_size=0.7, threshold=0.5, neurons=None, dropout=None,
                 epochs=None, batch_size=None):
        """
        Initialize the LSTM strategy.
        
        Args:
            data (pandas.DataFrame, optional): The OHLCV dataframe
            config (dict, optional): Configuration parameters
            config_path (str, optional): Path to configuration file
            sequence_length (int, optional): Length of input sequences
            features (list, optional): List of features to use
            target_column (str): Target column for prediction
            train_size (float): Proportion of data to use for training
            threshold (float): Threshold for generating signals
            neurons (list, optional): List of neurons in LSTM layers
            dropout (float, optional): Dropout rate
            epochs (int, optional): Number of training epochs
            batch_size (int, optional): Batch size for training
        """
        super().__init__('lstm', data, config, config_path)

        # Set strategy parameters
        self.sequence_length = sequence_length or self.strategy_config.get('sequence_length', 60)
        self.features = features or self.strategy_config.get('features', None)
        self.target_column = target_column or self.strategy_config.get('target_column', 'return_1d')
        self.train_size = train_size or self.strategy_config.get('train_size', 0.7)
        self.threshold = threshold or self.strategy_config.get('threshold', 0.5)
        self.neurons = neurons or self.strategy_config.get('neurons', [64, 32])
        self.dropout = dropout or self.strategy_config.get('dropout', 0.2)
        self.epochs = epochs or self.strategy_config.get('epochs', 50)
        self.batch_size = batch_size or self.strategy_config.get('batch_size', 32)

        # Initialize LSTM model
        self.lstm_model = None
        self.processor = DataProcessor()
        self.processed_data = None
        self.X_seq = None
        self.y_seq = None

    def preprocess_data(self):
        """
        Preprocess data for LSTM model.
        
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
        Train the LSTM model.
        
        Returns:
            self: Trained model
        """
        if self.processed_data is None:
            self.preprocess_data()

        # Initialize model
        self.lstm_model = LSTMModel(
            sequence_length=self.sequence_length,
            neurons=self.neurons,
            dropout=self.dropout,
            epochs=self.epochs,
            batch_size=self.batch_size
        )

        # Prepare sequences
        self.X_seq, self.y_seq = self.lstm_model.prepare_sequences(
            self.processed_data,
            target_column=self.target_column,
            features=self.features
        )

        # Split data for training
        train_size = int(len(self.X_seq) * self.train_size)
        X_train, y_train = self.X_seq[:train_size], self.y_seq[:train_size]

        # Train model
        self.lstm_model.train(X_train, y_train)

        return self

    def generate_signals(self):
        """
        Generate trading signals based on LSTM model predictions.
        
        Returns:
            pandas.Series: Series of trading signals (1: long, 0: no position, -1: short)
        """
        if self.data is None:
            raise ValueError("Data must be set before generating signals")

        # Preprocess data if not already done
        if self.processed_data is None:
            self.preprocess_data()

        # Train model if not already trained
        if self.lstm_model is None:
            self.train_model()

        # Make predictions for the entire sequence
        predictions = self.lstm_model.predict(self.X_seq)

        # Create signals DataFrame with the same index as the original data
        # Note: We need to account for the sequence_length offset
        signals = pd.Series(0, index=self.data.index)

        # We can only make predictions after sequence_length data points
        pred_index = self.data.index[self.sequence_length:self.sequence_length + len(predictions)]

        # Convert predictions to signals
        pred_signals = pd.Series(0, index=pred_index)
        pred_signals[predictions.flatten() > self.threshold] = 1  # Buy signal
        pred_signals[predictions.flatten() < (1 - self.threshold)] = -1  # Sell signal

        # Update signals with predictions
        signals.loc[pred_index] = pred_signals

        # Fill the first sequence_length positions with 0 (no position)
        signals.iloc[:self.sequence_length] = 0

        # Store the signals
        self.signals = signals

        return signals

    def save_model(self, model_dir='../../models'):
        """
        Save the trained model.
        
        Args:
            model_dir (str): Directory to save the model
        """
        if self.lstm_model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        # Save model
        model_path = os.path.join(model_dir, f"{self.name}_model.h5")
        self.lstm_model.save_model(model_path)

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
        self.lstm_model = LSTMModel.load_model(model_path)

        # Update strategy parameters
        self.sequence_length = self.lstm_model.sequence_length
        self.features = self.lstm_model.feature_names
        self.neurons = self.lstm_model.neurons
        self.dropout = self.lstm_model.dropout

        return self

    def evaluate_model(self):
        """
        Evaluate the LSTM model.
        
        Returns:
            dict: Model evaluation metrics
        """
        if self.lstm_model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        if self.X_seq is None or self.y_seq is None:
            if self.processed_data is None:
                self.preprocess_data()
            self.X_seq, self.y_seq = self.lstm_model.prepare_sequences(
                self.processed_data,
                target_column=self.target_column,
                features=self.features
            )

        # Split data
        train_size = int(len(self.X_seq) * self.train_size)
        X_train, X_test = self.X_seq[:train_size], self.X_seq[train_size:]
        y_train, y_test = self.y_seq[:train_size], self.y_seq[train_size:]

        # Evaluate model
        train_metrics = self.lstm_model.evaluate(X_train, y_train)
        test_metrics = self.lstm_model.evaluate(X_test, y_test)

        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        }

    def plot_model_history(self):
        """
        Plot the model training history.
        
        Returns:
            matplotlib.figure.Figure: Training history plot
        """
        if self.lstm_model is None or self.lstm_model.history is None:
            raise ValueError("Model not trained. Call train_model() first.")

        return self.lstm_model.plot_training_history()


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
    # fetcher = DataFetcher(config_path)
    # data = fetcher.fetch_data('AAPL', period='2y')

    data = NiftyDataFetcher.fetch_data_from_csv(
        '/Users/neelansh/Desktop/Projects/My Projects/Stock Market Data/TATAMOTORS_till_13June2025.csv')
    #     '/Users/neelansh/Desktop/Projects/My Projects/stock-market-analysis-and-prediction/src/data/ADANIENT.NS_1d_10y.csv')
    # fetcher = NiftyDataFetcher()
    # data = fetcher.fetch_ticker_data_from_csv(ticker_symbol="TATAMOTORS.NS")
    # Initialize strategy
    strategy = LSTMStrategy(data=data, config=config)

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
        if metric != 'confusion_matrix':
            print(f"{metric}: {value:.4f}")

    print(f"Confusion matrix:\n{model_eval['train_metrics']['confusion_matrix']}")

    print("\nTest metrics:")
    for metric, value in model_eval['test_metrics'].items():
        if metric != 'confusion_matrix':
            print(f"{metric}: {value:.4f}")

    print(f"Confusion matrix for Training Data:\n{model_eval['test_metrics']['confusion_matrix']}")

    # Plot model history
    strategy.plot_model_history()

    # Plot results
    strategy.plot_results()
    plt.show()

    # Save results
    strategy.save_results()

    # Save model
    strategy.save_model()
