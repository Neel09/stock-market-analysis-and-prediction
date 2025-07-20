#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.src.layers import Bidirectional, BatchNormalization
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import os
import sys
import joblib
import matplotlib.pyplot as plt

from src.utils.nifty_data_fetcher import NiftyDataFetcher

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.data_processor import DataProcessor


class LSTMModel:
    """
    LSTM (Long Short-Term Memory) Model for time series prediction.
    
    This class implements LSTM networks for predicting stock price movements.
    """

    def __init__(self, sequence_length=60, neurons=None, dropout=0.2,
                 epochs=50, batch_size=32, learning_rate=0.001):
        """
        Initialize the LSTM Model.
        
        Args:
            sequence_length (int): Length of input sequences
            neurons (list, optional): List of neurons in LSTM layers
            dropout (float): Dropout rate
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate for optimizer
        """
        self.sequence_length = sequence_length
        self.neurons = neurons or [64, 32]
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_names = None
        self.history = None

    def build_model(self, input_shape):
        """
        Build the LSTM model architecture.
        
        Args:
            input_shape (tuple): Shape of input data (sequence_length, n_features)
            
        Returns:
            tensorflow.keras.models.Sequential: The LSTM model
        """

        # *******************************Model-1*******************************************************
        model = Sequential()

        # First LSTM layer
        model.add(LSTM(
            units=self.neurons[0],
            return_sequences=len(self.neurons) > 1,
            input_shape=input_shape
        ))

        # Additional LSTM layers
        for i in range(1, len(self.neurons)):
            model.add(LSTM(
                units=self.neurons[i],
                return_sequences=i < len(self.neurons) - 1
            ))

        model.add(Dense(25))

        # Output layer
        model.add(Dense(units=1, activation='sigmoid'))

        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mean_squared_error',
            metrics=['accuracy']
        )

        # *******************************Model-2*******************************************************

        print(model.summary())

        return model

    def prepare_sequences(self, data, target_column='return_1d', features=None):
        """
        Prepare sequences for LSTM model.
        
        Args:
            data (pandas.DataFrame): DataFrame with technical indicators
            target_column (str): Target column name
            features (list): List of feature column names
            
        Returns:
            tuple: (X, y, scaler)
        """
        # Default features if none provided
        if features is None:
            features = [
                'Close', 'Volume', 'rsi', 'macd', 'macd_signal', 'macd_hist',
                'bb_width', 'ema_12', 'ema_26', 'atr', 'adx'
            ]

        # Filter features that exist in the dataframe
        available_features = [f for f in features if f in data.columns]

        # Store feature names
        self.feature_names = available_features

        # Extract features and target
        X = data[available_features].values

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        threshold = 0.01  # 1% movement

        # For binary classification, convert returns to binary signals
        if target_column == 'return_1d':
            y = (data[target_column] > 0).astype(int).values
        else:
            y = data[target_column].values

        print(pd.Series(y).value_counts(normalize=True))

        # Create sequences
        X_seq, y_seq = [], []

        for i in range(len(X_scaled) - self.sequence_length):
            X_seq.append(X_scaled[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])

        return np.array(X_seq), np.array(y_seq)

    def train(self, X, y, validation_split=0.2, early_stopping=True,
              checkpoint_path=None, verbose=1):
        """
        Train the LSTM model.
        
        Args:
            X (numpy.ndarray): Input sequences
            y (numpy.ndarray): Target values
            validation_split (float): Fraction of data to use for validation
            early_stopping (bool): Whether to use early stopping
            checkpoint_path (str, optional): Path to save model checkpoints
            verbose (int): Verbosity level
            
        Returns:
            self: Trained model
        """
        # Build model if not already built
        if self.model is None:
            input_shape = (X.shape[1], X.shape[2])
            self.model = self.build_model(input_shape)

        # Callbacks
        callbacks = []

        if early_stopping:
            callbacks.append(EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ))

        if checkpoint_path:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            callbacks.append(ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_loss',
                save_best_only=True
            ))

        # Train model
        self.history = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )

        return self

    def predict(self, X):
        """
        Make predictions with the model.
        
        Args:
            X (numpy.ndarray): Input sequences
            
        Returns:
            numpy.ndarray: Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.predict(X)

    def evaluate(self, X, y):
        """
        Evaluate the model.
        
        Args:
            X (numpy.ndarray): Input sequences
            y (numpy.ndarray): True target values
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Evaluate model
        loss, accuracy = self.model.evaluate(X, y, verbose=0)

        # Make predictions
        y_pred_proba = self.model.predict(X)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        # Calculate precision, recall, f1
        from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        cm = confusion_matrix(y, y_pred)

        return {
            'loss': loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }

    def save_model(self, filepath):
        """
        Save the model to a file.
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save model
        self.model.save(filepath)

        # Save metadata
        metadata_path = f"{filepath}_metadata.joblib"
        joblib.dump({
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'sequence_length': self.sequence_length,
            'neurons': self.neurons,
            'dropout': self.dropout
        }, metadata_path)

        print(f"Model saved to {filepath}")
        print(f"Metadata saved to {metadata_path}")

    @classmethod
    def load_model(cls, filepath):
        """
        Load a model from a file.
        
        Args:
            filepath (str): Path to the model file
            
        Returns:
            LSTMModel: Loaded model
        """
        # Load keras model
        keras_model = load_model(filepath)

        # Load metadata
        metadata_path = f"{filepath}_metadata.joblib"
        metadata = joblib.load(metadata_path)

        # Create model instance
        model_instance = cls(
            sequence_length=metadata['sequence_length'],
            neurons=metadata['neurons'],
            dropout=metadata['dropout']
        )

        model_instance.model = keras_model
        model_instance.scaler = metadata['scaler']
        model_instance.feature_names = metadata['feature_names']

        return model_instance

    def plot_training_history(self):
        """
        Plot training history.
        
        Returns:
            matplotlib.figure.Figure: Training history plot
        """
        if self.history is None:
            raise ValueError("Model not trained. Call train() first.")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Loss')
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        ax1.grid(True)

        # Plot accuracy
        ax2.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax2.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Accuracy')
        ax2.set_ylabel('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        return fig

    def use_ensemble_approach(self):

        data_directory = '/Users/neelansh/Desktop/Projects/My Projects/stock-market-analysis-and-prediction/src/data';

        data = NiftyDataFetcher.fetch_data_from_csv(
            '/Users/neelansh/Desktop/Projects/My Projects/Stock Market Data/TATAMOTORS_till_13June2025.csv')


if __name__ == '__main__':
    # Example usage
    import sys
    import os

    # Add the parent directory to the path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    from src.utils.data_fetcher import DataFetcher
    from src.utils.data_processor import DataProcessor

    # Load configuration
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                               'config', 'config.json')

    # Fetch data
    # fetcher = DataFetcher(config_path)
    # data = fetcher.fetch_data('AAPL', period='2y')

    data = NiftyDataFetcher.fetch_data_from_csv(
        # '/Users/neelansh/Desktop/Projects/My Projects/Stock Market Data/TATAMOTORS_till_13June2025.csv')
        '/Users/neelansh/Desktop/Projects/My Projects/stock-market-analysis-and-prediction/src/data/ADANIENT.NS_1d_10y.csv')

    # Process data
    processor = DataProcessor()
    processed_data = processor.add_technical_indicators(data)

    # Create LSTM model
    lstm = LSTMModel(sequence_length=60, neurons=[64, 32], dropout=0.2, epochs=50, batch_size=32)

    # Prepare sequences
    X, y = lstm.prepare_sequences(processed_data)

    # Split data
    train_size = int(0.7 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Train model
    lstm.train(X_train, y_train)

    # Evaluate model
    metrics = lstm.evaluate(X_test, y_test)

    print("Test metrics:")
    for metric, value in metrics.items():
        if metric != 'confusion_matrix':
            print(f"{metric}: {value:.4f}")
    print(f"Confusion matrix:\n{metrics['confusion_matrix']}")

    # Plot training history
    lstm.plot_training_history()
    plt.show()

    # Save model
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                              'models', 'lstm_model')
    lstm.save_model(f"{model_path}.h5")
