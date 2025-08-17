#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from src.utils.sentiment_analyzer import SentimentAnalyzer
from src.utils.news_fetcher import NewsFetcher

class DataProcessor:
    """
    A class to process and transform financial data for machine learning models.
    """
    def __init__(self):
        """
        Initialize the DataProcessor.
        """
        # Initialize scalers dictionary
        self.scalers = {}

    """
    Feature Engineering / Technical Indicators
    """
    def add_technical_indicators(self, data):
        """
        Add technical indicators to the dataframe.

        Args:
            data (pandas.DataFrame): The OHLCV dataframe

        Returns:
            pandas.DataFrame: Dataframe with technical indicators
        """
        # Make a copy to avoid modifying the original dataframe
        df = data.copy()

        # Ensure the dataframe has the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")

        # === Trend Indicators ===
        # Moving Averages
        df['sma_20'] = ta.sma(df['Close'], length=20)
        df['sma_50'] = ta.sma(df['Close'], length=50)
        df['sma_200'] = ta.sma(df['Close'], length=200)
        df['ema_12'] = ta.ema(df['Close'], length=12)
        df['ema_26'] = ta.ema(df['Close'], length=26)

        # MACD
        macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        df['macd_hist'] = macd['MACDh_12_26_9']

        # Parabolic SAR
        psar = ta.psar(df['High'], df['Low'], df['Close'])
        df['psar'] = psar['PSARl_0.02_0.2']

        # ADX (Average Directional Index)
        adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
        df['adx'] = adx['ADX_14']
        df['di_plus'] = adx['DMP_14']
        df['di_minus'] = adx['DMN_14']

        # === Momentum Indicators ===
        # RSI (Relative Strength Index)
        df['rsi'] = ta.rsi(df['Close'], length=14)

        # Stochastic Oscillator
        stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3)
        df['stoch_k'] = stoch['STOCHk_14_3_3']
        df['stoch_d'] = stoch['STOCHd_14_3_3']

        # CCI (Commodity Channel Index)
        df['cci'] = ta.cci(df['High'], df['Low'], df['Close'], length=20)

        # Williams %R
        df['willr'] = ta.willr(df['High'], df['Low'], df['Close'], length=14)

        # === Volatility Indicators ===
        # Bollinger Bands
        bbands = ta.bbands(df['Close'], length=20, std=2)
        df['bb_upper'] = bbands['BBU_20_2.0']
        df['bb_middle'] = bbands['BBM_20_2.0']
        df['bb_lower'] = bbands['BBL_20_2.0']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        # ATR (Average True Range)
        df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

        # === Volume Indicators ===
        # OBV (On-Balance Volume)
        df['obv'] = ta.obv(df['Close'], df['Volume'])

        # Volume SMA
        df['volume_sma'] = ta.sma(df['Volume'], length=20)
        df['volume_ratio'] = df['Volume'] / df['volume_sma']

        # === Price Transformations ===
        # Log returns
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

        # Simple returns
        df['return_1d'] = df['Close'].pct_change(1)
        df['return_5d'] = df['Close'].pct_change(5)
        df['return_20d'] = df['Close'].pct_change(20)

        # Price relative to moving averages
        df['close_sma20_ratio'] = df['Close'] / df['sma_20']
        df['close_sma50_ratio'] = df['Close'] / df['sma_50']
        df['close_sma200_ratio'] = df['Close'] / df['sma_200']

        # Drop NaN values
        df = df.dropna()

        return df

    def create_sequences(self, data, sequence_length, target_column='return_1d', features=None):
        """
        Create sequences for time series models like LSTM.

        Args:
            data (pandas.DataFrame): The processed dataframe
            sequence_length (int): Length of the sequence
            target_column (str): Target column for prediction
            features (list): List of feature columns to use

        Returns:
            tuple: (X, y) where X is the sequences and y is the targets
        """
        # Default features if none provided
        if features is None:
            features = [
                'Close', 'Volume', 'rsi', 'macd', 'bb_width', 'atr', 'cci',
                'obv', 'close_sma20_ratio', 'close_sma50_ratio'
            ]

        # Ensure all requested features exist in the dataframe
        for feature in features:
            if feature not in data.columns:
                raise ValueError(f"Feature '{feature}' not found in dataframe")

        # Extract features and target
        df_features = data[features].copy()
        df_target = data[target_column].copy()

        # Scale features
        scaler = MinMaxScaler()
        df_features_scaled = pd.DataFrame(
            scaler.fit_transform(df_features),
            columns=df_features.columns,
            index=df_features.index
        )
        self.scalers['features'] = scaler

        # Create sequences
        X, y = [], []
        for i in range(len(df_features_scaled) - sequence_length):
            X.append(df_features_scaled.iloc[i:i+sequence_length].values)
            y.append(df_target.iloc[i+sequence_length])

        return np.array(X), np.array(y)

    def prepare_data(self, data, target_column='return_1d', features=None, sequence_length=None, train_size=0.7):
        """
        Prepare data for training ML/DL models.

        Args:
            data (pandas.DataFrame): The processed dataframe
            target_column (str): Target column for prediction
            features (list): List of feature columns to use
            sequence_length (int): Length of the sequence for time series models
            train_size (float): Proportion of data to use for training

        Returns:
            dict: Dictionary containing prepared data
        """
        # Default features if none provided
        if features is None:
            features = [
                'Close', 'Volume', 'rsi', 'macd', 'bb_width', 'atr', 'cci',
                'obv', 'close_sma20_ratio', 'close_sma50_ratio'
            ]

        # Ensure all requested features exist in the dataframe
        for feature in features:
            if feature not in data.columns:
                raise ValueError(f"Feature '{feature}' not found in dataframe")

        # Extract features and target
        df_features = data[features].copy()
        df_target = data[target_column].copy()

        # For classification tasks, convert target to categorical
        if target_column == 'return_1d':
            # Convert returns to binary signals (1 for positive return, 0 for negative)
            df_target = (df_target > 0).astype(int)

        # Scale features
        scaler = MinMaxScaler()
        df_features_scaled = pd.DataFrame(
            scaler.fit_transform(df_features),
            columns=df_features.columns,
            index=df_features.index
        )
        self.scalers['features'] = scaler

        # Split data
        train_size = int(len(df_features_scaled) * train_size)

        # For time series models
        if sequence_length is not None:
            X, y = self.create_sequences(data, sequence_length, target_column, features)

            # Split into train and test
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'feature_names': features,
                'scaler': self.scalers['features']
            }

        # For traditional ML models
        else:
            X_train = df_features_scaled.iloc[:train_size]
            X_test = df_features_scaled.iloc[train_size:]
            y_train = df_target.iloc[:train_size]
            y_test = df_target.iloc[train_size:]

            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'feature_names': features,
                'scaler': self.scalers['features']
            }

    def add_sentiment_data(self, data, symbol, config=None, days=7, max_news=10):
        """
        Add sentiment analysis data to the dataframe.

        Args:
            data (pandas.DataFrame): The OHLCV dataframe
            symbol (str): The stock symbol
            config (dict): Configuration for news and sentiment APIs
            days (int): Number of days to look back for news
            max_news (int): Maximum number of news articles to analyze

        Returns:
            pandas.DataFrame: Dataframe with sentiment indicators
        """
        # Make a copy to avoid modifying the original dataframe
        df = data.copy()

        print(f"Adding sentiment data for symbol: {symbol}")

        # Initialize news fetcher and sentiment analyzer
        news_fetcher = NewsFetcher(config)
        sentiment_analyzer = SentimentAnalyzer(config)

        # Get the latest date in the dataframe
        if len(df.index) > 0:
            latest_date = df.index[-1]
            print(f"Latest date in dataframe: {latest_date}")
        else:
            print("Warning: Empty dataframe provided to add_sentiment_data")
            return df

        # Fetch news for the symbol
        print(f"Fetching news for {symbol}...")
        news_articles = news_fetcher.fetch_news(symbol, days=days, max_results=max_news)
        print(f"Fetched {len(news_articles)} news articles")

        if len(news_articles) == 0:
            print("Warning: No news articles found. Using neutral sentiment.")
            df.loc[latest_date, 'sentiment_score'] = 0
            df.loc[latest_date, 'sentiment_signal'] = 0
        else:
            # Analyze sentiment for each news article
            print("Analyzing sentiment for news articles...")
            # sentiment_results = sentiment_analyzer.analyze_news_batch([
            #     f"{article['title']}. {article['description']}"
            #     for article in news_articles
            # ])
            # print(f"Sentiment analysis results: {sentiment_results}")

            # Calculate aggregate sentiment
            # aggregate_sentiment = sentiment_analyzer.get_aggregate_sentiment(sentiment_results)
            aggregate_sentiment = sentiment_analyzer.get_mocked_aggeregate_sentiment()
            print(f"Aggregate sentiment: {aggregate_sentiment}")

            # Add sentiment data to the dataframe
            # We'll add it to the latest date in the dataframe
            df.loc[latest_date, 'sentiment_score'] = aggregate_sentiment['aggregate_score']
            df.loc[latest_date, 'sentiment_signal'] = aggregate_sentiment['sentiment_signal']

        # Fill NaN values with 0 (for dates without sentiment data)
        df['sentiment_score'] = df['sentiment_score'].fillna(0)
        df['sentiment_signal'] = df['sentiment_signal'].fillna(0)

        # Create a rolling average of sentiment to smooth the signal
        df['sentiment_score_ma5'] = df['sentiment_score'].rolling(window=5, min_periods=1).mean()

        print("Sentiment data added successfully")
        return df

    def add_market_sentiment_data(self, data, market="India", config=None, days=7, max_news=10):
        """
        Add market-wide sentiment analysis data to the dataframe.

        Args:
            data (pandas.DataFrame): The OHLCV dataframe
            market (str): The market name (e.g., "India", "Nifty")
            config (dict): Configuration for news and sentiment APIs
            days (int): Number of days to look back for news
            max_news (int): Maximum number of news articles to analyze

        Returns:
            pandas.DataFrame: Dataframe with market sentiment indicators
        """
        # Make a copy to avoid modifying the original dataframe
        df = data.copy()

        print(f"Adding market sentiment data for market: {market}")

        # Initialize news fetcher and sentiment analyzer
        news_fetcher = NewsFetcher(config)
        sentiment_analyzer = SentimentAnalyzer(config)

        # Get the latest date in the dataframe
        if len(df.index) > 0:
            latest_date = df.index[-1]
            print(f"Latest date in dataframe: {latest_date}")
        else:
            print("Warning: Empty dataframe provided to add_market_sentiment_data")
            return df

        # Fetch market news
        print(f"Fetching news for market: {market}...")
        market_news = news_fetcher.fetch_market_news(market, days=days, max_results=max_news)
        print(f"Fetched {len(market_news)} market news articles")

        if len(market_news) == 0:
            print("Warning: No market news articles found. Using neutral sentiment.")
            df.loc[latest_date, 'market_sentiment_score'] = 0
            df.loc[latest_date, 'market_sentiment_signal'] = 0
        else:
            # Analyze sentiment for each news article
            print("Analyzing sentiment for market news articles...")
            # sentiment_results = sentiment_analyzer.analyze_news_batch([
            #     f"{article['title']}. {article['description']}"
            #     for article in market_news
            # ])
            # print(f"Market sentiment analysis results: {sentiment_results}")

            # Calculate aggregate sentiment
            # aggregate_sentiment = sentiment_analyzer.get_aggregate_sentiment(sentiment_results)
            aggregate_sentiment = sentiment_analyzer.get_mocked_aggeregate_sentiment()
            print(f"Aggregate market sentiment: {aggregate_sentiment}")

            # Add market sentiment data to the dataframe
            # We'll add it to the latest date in the dataframe
            df.loc[latest_date, 'market_sentiment_score'] = aggregate_sentiment['aggregate_score']
            df.loc[latest_date, 'market_sentiment_signal'] = aggregate_sentiment['sentiment_signal']

        # Fill NaN values with 0 (for dates without sentiment data)
        df['market_sentiment_score'] = df['market_sentiment_score'].fillna(0)
        df['market_sentiment_signal'] = df['market_sentiment_signal'].fillna(0)

        # Create a rolling average of sentiment to smooth the signal
        df['market_sentiment_score_ma5'] = df['market_sentiment_score'].rolling(window=5, min_periods=1).mean()

        print("Market sentiment data added successfully")
        return df

    def prepare_data_with_sentiment(self, data, symbol, config=None, target_column='return_1d', 
                                   features=None, sentiment_features=True, sequence_length=None, 
                                   train_size=0.7, days=7, max_news=10):
        """
        Prepare data for training ML/DL models with sentiment features.

        Args:
            data (pandas.DataFrame): The processed dataframe
            symbol (str): The stock symbol
            config (dict): Configuration for news and sentiment APIs
            target_column (str): Target column for prediction
            features (list): List of feature columns to use
            sentiment_features (bool): Whether to include sentiment features
            sequence_length (int): Length of the sequence for time series models
            train_size (float): Proportion of data to use for training
            days (int): Number of days to look back for news
            max_news (int): Maximum number of news articles to analyze

        Returns:
            dict: Dictionary containing prepared data
        """
        # Add technical indicators
        df = self.add_technical_indicators(data)

        # Add sentiment data if requested
        if sentiment_features:
            df = self.add_sentiment_data(df, symbol, config, days, max_news)
            df = self.add_market_sentiment_data(df, "India", config, days, max_news)

            # Add sentiment features to the default feature list
            if features is None:
                features = [
                    'Close', 'Volume', 'rsi', 'macd', 'bb_width', 'atr', 'cci',
                    'obv', 'close_sma20_ratio', 'close_sma50_ratio',
                    'sentiment_score', 'sentiment_signal', 'sentiment_score_ma5',
                    'market_sentiment_score', 'market_sentiment_signal', 'market_sentiment_score_ma5'
                ]

        # Prepare data using the standard method
        return self.prepare_data(df, target_column, features, sequence_length, train_size)

if __name__ == '__main__':
    # Test the data processor
    import sys
    import os

    # Add the project root to the Python path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    # Import DataFetcher with the correct path
    from src.utils.data_fetcher import DataFetcher

    # Initialize data processor and fetcher
    processor = DataProcessor()
    fetcher = DataFetcher('../config/config.json')

    # Fetch data
    data = fetcher.fetch_data('AAPL', period='2y')

    # Process data
    processed_data = processor.add_technical_indicators(data)

    # Print results
    print(f"Original data shape: {data.shape}")
    print(f"Processed data shape: {processed_data.shape}")
    print("Available features:")
    print(processed_data.columns.tolist())

    # Prepare data for ML model
    ml_data = processor.prepare_data(processed_data)

    # Prepare data for DL model with sequences
    dl_data = processor.prepare_data(processed_data, sequence_length=60)

    print(f"ML data shapes: X_train {ml_data['X_train'].shape}, y_train {ml_data['y_train'].shape}")
    print(f"DL data shapes: X_train {dl_data['X_train'].shape}, y_train {dl_data['y_train'].shape}") 
