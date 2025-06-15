#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import sys
import os
import joblib

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.data_processor import DataProcessor

class MLLinearModel:
    """
    Machine Learning Linear Model for trading.
    
    This class implements linear models (Linear Regression, Logistic Regression)
    for predicting price movements.
    """
    
    def __init__(self, model_type='logistic', cv_folds=5):
        """
        Initialize the ML Linear Model.
        
        Args:
            model_type (str): Type of model ('logistic' or 'linear')
            cv_folds (int): Number of folds for cross-validation
        """
        self.model_type = model_type
        self.cv_folds = cv_folds
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def prepare_features(self, data, target_column='return_1d', features=None):
        """
        Prepare features for the model.
        
        Args:
            data (pandas.DataFrame): DataFrame with technical indicators
            target_column (str): Target column name
            features (list): List of feature column names
            
        Returns:
            tuple: (X, y, feature_names)
        """
        # Default features if none provided
        if features is None:
            features = [
                'Close', 'Volume', 'rsi', 'macd', 'macd_signal', 'macd_hist',
                'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
                'ema_12', 'ema_26', 'atr', 'cci', 'stoch_k', 'stoch_d',
                'adx', 'obv', 'willr'
            ]
        
        # Filter features that exist in the dataframe
        available_features = [f for f in features if f in data.columns]
        
        # Extract features and target
        X = data[available_features].copy()
        
        # For classification, we convert returns to binary signals
        if self.model_type == 'logistic':
            # Up (1) or Down (0)
            y = (data[target_column] > 0).astype(int)
        else:
            # For regression, we use the actual return
            y = data[target_column]
        
        # Store feature names
        self.feature_names = available_features
        
        return X, y, available_features
    
    def train(self, X, y):
        """
        Train the model.
        
        Args:
            X (pandas.DataFrame): Feature matrix
            y (pandas.Series): Target vector
            
        Returns:
            self: Trained model
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize model
        if self.model_type == 'logistic':
            self.model = LogisticRegression(
                C=1.0,                      # Regularization strength
                penalty='l2',               # L2 regularization
                solver='liblinear',         # Solver
                max_iter=1000,              # Maximum iterations
                random_state=42             # Random seed for reproducibility
            )
        else:  # linear regression
            self.model = LinearRegression()
        
        # Train model
        self.model.fit(X_scaled, y)
        
        return self
    
    def predict(self, X):
        """
        Make predictions with the model.
        
        Args:
            X (pandas.DataFrame): Feature matrix
            
        Returns:
            numpy.ndarray: Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        if self.model_type == 'logistic':
            # For logistic regression, predict probabilities
            return self.model.predict_proba(X_scaled)[:, 1]
        else:
            # For linear regression, predict values
            return self.model.predict(X_scaled)
    
    def evaluate(self, X, y):
        """
        Evaluate the model.
        
        Args:
            X (pandas.DataFrame): Feature matrix
            y (pandas.Series): True target values
            
        Returns:
            dict: Evaluation metrics
        """
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        if self.model_type == 'logistic':
            y_prob = self.model.predict_proba(X_scaled)[:, 1]
            y_pred = (y_prob > 0.5).astype(int)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, zero_division=0),
                'recall': recall_score(y, y_pred, zero_division=0),
                'f1': f1_score(y, y_pred, zero_division=0)
            }
        else:
            y_pred = self.model.predict(X_scaled)
            
            # Calculate metrics
            metrics = {
                'mse': np.mean((y - y_pred) ** 2),
                'rmse': np.sqrt(np.mean((y - y_pred) ** 2)),
                'mae': np.mean(np.abs(y - y_pred)),
                'r2': self.model.score(X_scaled, y)
            }
        
        return metrics
    
    def cross_validate(self, X, y):
        """
        Perform time series cross-validation.
        
        Args:
            X (pandas.DataFrame): Feature matrix
            y (pandas.Series): Target vector
            
        Returns:
            dict: Cross-validation results
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        
        # Initialize model for cross-validation
        if self.model_type == 'logistic':
            cv_model = LogisticRegression(
                C=1.0, penalty='l2', solver='liblinear', max_iter=1000, random_state=42
            )
            # Use accuracy for classification
            scoring = 'accuracy'
        else:
            cv_model = LinearRegression()
            # Use negative mean squared error for regression
            scoring = 'neg_mean_squared_error'
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            cv_model, X_scaled, y, cv=tscv, scoring=scoring
        )
        
        # Return results
        return {
            'cv_scores': cv_scores,
            'mean_cv_score': np.mean(cv_scores),
            'std_cv_score': np.std(cv_scores)
        }
    
    def feature_importance(self):
        """
        Get feature importance.
        
        Returns:
            pandas.Series: Feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if self.model_type == 'logistic':
            importance = pd.Series(
                np.abs(self.model.coef_[0]),
                index=self.feature_names
            )
        else:
            importance = pd.Series(
                np.abs(self.model.coef_),
                index=self.feature_names
            )
        
        # Sort by importance
        importance = importance.sort_values(ascending=False)
        
        return importance
    
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
        
        # Save model and scaler
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_names': self.feature_names
        }, filepath)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """
        Load a model from a file.
        
        Args:
            filepath (str): Path to the model file
            
        Returns:
            MLLinearModel: Loaded model
        """
        # Load model and scaler
        data = joblib.load(filepath)
        
        # Create model instance
        model_instance = cls(model_type=data['model_type'])
        model_instance.model = data['model']
        model_instance.scaler = data['scaler']
        model_instance.feature_names = data['feature_names']
        
        return model_instance

if __name__ == '__main__':
    # Example usage
    import sys
    import os
    import matplotlib.pyplot as plt
    
    # Add the parent directory to the path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    from src.utils.data_fetcher import DataFetcher
    from src.utils.data_processor import DataProcessor
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                              'config', 'config.json')
    
    # Fetch data
    fetcher = DataFetcher(config_path)
    data = fetcher.fetch_data('AAPL', period='2y')
    
    # Process data
    processor = DataProcessor()
    processed_data = processor.add_technical_indicators(data)
    
    # Create model
    model = MLLinearModel(model_type='logistic')
    
    # Prepare features
    X, y, feature_names = model.prepare_features(processed_data)
    
    # Split data
    train_size = int(0.7 * len(X))
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    # Train model
    model.train(X_train, y_train)
    
    # Evaluate model
    train_metrics = model.evaluate(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)
    
    print("Training metrics:")
    for metric, value in train_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nTest metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Cross-validation
    cv_results = model.cross_validate(X, y)
    print(f"\nCross-validation scores: {cv_results['cv_scores']}")
    print(f"Mean CV score: {cv_results['mean_cv_score']:.4f}")
    
    # Feature importance
    importance = model.feature_importance()
    print("\nFeature importance:")
    print(importance)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    importance.plot(kind='bar')
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.show()
    
    # Save model
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                             'models', 'linear_model.joblib')
    model.save_model(model_path) 