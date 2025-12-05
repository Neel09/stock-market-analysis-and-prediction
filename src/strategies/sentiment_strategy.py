import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.strategies.base_strategy import BaseStrategy
from src.utils.data_processor import DataProcessor
from src.utils.sentiment_analyzer import SentimentAnalyzer
from src.utils.news_fetcher import NewsFetcher

class SentimentStrategy(BaseStrategy):
    """
    A trading strategy based on sentiment analysis of financial news using LLM.
    """
    def __init__(self, data=None, config=None, config_path=None, symbol=None, 
                 sentiment_threshold=0.2, market_sentiment_weight=0.4, 
                 use_technical_indicators=True, days=7, max_news=10):
        """
        Initialize the SentimentStrategy.

        Args:
            data (pandas.DataFrame): OHLCV data
            config (dict): Configuration dictionary
            config_path (str): Path to configuration file
            symbol (str): Stock symbol
            sentiment_threshold (float): Threshold for sentiment signals
            market_sentiment_weight (float): Weight for market sentiment vs. stock-specific sentiment
            use_technical_indicators (bool): Whether to use technical indicators alongside sentiment
            days (int): Number of days to look back for news
            max_news (int): Maximum number of news articles to analyze
        """
        super().__init__("Sentiment Strategy", data, config, config_path)

        self.symbol = symbol or self.config.get('data', {}).get('default_symbols', ['NIFTY 50'])[0]
        self.sentiment_threshold = sentiment_threshold
        self.market_sentiment_weight = market_sentiment_weight
        self.use_technical_indicators = use_technical_indicators
        self.days = days
        self.max_news = max_news

        # Initialize data processor and sentiment components
        self.data_processor = DataProcessor()
        self.sentiment_analyzer = SentimentAnalyzer(self.config.get('sentiment', {}))
        self.news_fetcher = NewsFetcher(self.config.get('sentiment', {}))

        # Process data if provided
        if self.data is not None:
            self.preprocess_data()

    def preprocess_data(self):
        """
        Preprocess the data by adding technical indicators and sentiment data.
        """
        # Add technical indicators
        if self.use_technical_indicators:
            self.data = self.data_processor.add_technical_indicators(self.data)

        try:
            # Add sentiment data
            self.data = self.data_processor.add_sentiment_data(
                self.data, 
                self.symbol, 
                self.config.get('sentiment', {}), 
                self.days, 
                self.max_news
            )

            # Add market sentiment data
            self.data = self.data_processor.add_market_sentiment_data(
                self.data, 
                "India", 
                self.config.get('sentiment', {}), 
                self.days, 
                self.max_news
            )
        except Exception as e:
            print(f"Warning: Error adding sentiment data: {e}")
            print("Using default neutral sentiment.")
            # Ensure sentiment columns exist with default values
            if 'sentiment_signal' not in self.data.columns:
                self.data['sentiment_signal'] = 0
            if 'market_sentiment_signal' not in self.data.columns:
                self.data['market_sentiment_signal'] = 0

        # Calculate combined sentiment signal
        self.calculate_combined_sentiment()

    def calculate_combined_sentiment(self):
        """
        Calculate a combined sentiment signal using stock-specific and market sentiment.
        """
        # Check if sentiment columns exist
        if 'sentiment_signal' not in self.data.columns or 'market_sentiment_signal' not in self.data.columns:
            print("Warning: Sentiment data not found in dataframe. Adding default neutral sentiment.")
            # Add default neutral sentiment
            self.data['sentiment_signal'] = 0
            self.data['market_sentiment_signal'] = 0

        # Calculate combined sentiment
        stock_weight = 1 - self.market_sentiment_weight
        self.data['combined_sentiment'] = (
            stock_weight * self.data['sentiment_signal'] + 
            self.market_sentiment_weight * self.data['market_sentiment_signal']
        )

        # Create a smoothed version using moving average
        self.data['combined_sentiment_ma5'] = self.data['combined_sentiment'].rolling(window=5, min_periods=1).mean()

    def generate_signals(self):
        """
        Generate trading signals based on sentiment analysis.

        Returns:
            pandas.Series: Series with signals
        """
        # Ensure data is preprocessed
        if 'combined_sentiment' not in self.data.columns:
            print("Combined sentiment not found in dataframe. Running preprocess_data()...")
            self.preprocess_data()

        # Create a copy of the data
        signals = self.data.copy()

        # Initialize signal column
        signals['signal'] = 0

        print(f"Generating signals with sentiment threshold: {self.sentiment_threshold}")

        # Check if we have sentiment data
        if 'combined_sentiment' in signals.columns:
            print(f"Combined sentiment values: {signals['combined_sentiment'].values}")

            # Count how many rows exceed the threshold
            positive_sentiment = (signals['combined_sentiment'] > self.sentiment_threshold).sum()
            negative_sentiment = (signals['combined_sentiment'] < -self.sentiment_threshold).sum()
            print(f"Rows with positive sentiment > threshold: {positive_sentiment}")
            print(f"Rows with negative sentiment < -threshold: {negative_sentiment}")

            # If no sentiment signals exceed the threshold, use technical indicators as fallback
            if positive_sentiment == 0 and negative_sentiment == 0:
                print("No sentiment signals exceed threshold. Using technical indicators as fallback.")
                if self.use_technical_indicators and 'rsi' in signals.columns:
                    # Generate signals based on RSI
                    signals.loc[signals['rsi'] < 30, 'signal'] = 1  # Buy when RSI is oversold
                    signals.loc[signals['rsi'] > 70, 'signal'] = -1  # Sell when RSI is overbought
                    print(f"Generated {(signals['signal'] != 0).sum()} signals based on RSI")
                else:
                    # Generate some basic signals based on price movements
                    print("Using price movements as fallback")
                    signals.loc[signals['Close'] > signals['sma_20'], 'signal'] = 1  # Buy when price above SMA20
                    signals.loc[signals['Close'] < signals['sma_20'], 'signal'] = -1  # Sell when price below SMA20
                    print(f"Generated {(signals['signal'] != 0).sum()} signals based on price movements")
            else:
                # Generate signals based on sentiment
                signals.loc[signals['combined_sentiment'] > self.sentiment_threshold, 'signal'] = 1
                signals.loc[signals['combined_sentiment'] < -self.sentiment_threshold, 'signal'] = -1

                # If using technical indicators, incorporate them into the signal
                if self.use_technical_indicators and 'rsi' in signals.columns:
                    print("Using RSI as confirmation indicator")
                    # Use RSI as a confirmation indicator
                    # Buy only if RSI is not overbought and sentiment is positive
                    signals.loc[(signals['combined_sentiment'] > self.sentiment_threshold) & 
                                (signals['rsi'] > 70), 'signal'] = 0

                    # Sell only if RSI is not oversold and sentiment is negative
                    signals.loc[(signals['combined_sentiment'] < -self.sentiment_threshold) & 
                                (signals['rsi'] < 30), 'signal'] = 0
        else:
            print("Warning: No combined sentiment data found in dataframe. Using technical indicators as fallback.")
            if self.use_technical_indicators and 'rsi' in signals.columns:
                # Generate signals based on RSI
                signals.loc[signals['rsi'] < 30, 'signal'] = 1  # Buy when RSI is oversold
                signals.loc[signals['rsi'] > 70, 'signal'] = -1  # Sell when RSI is overbought
                print(f"Generated {(signals['signal'] != 0).sum()} signals based on RSI")
            else:
                # Generate some basic signals based on price movements
                print("Using price movements as fallback")
                signals.loc[signals['Close'] > signals['sma_20'], 'signal'] = 1  # Buy when price above SMA20
                signals.loc[signals['Close'] < signals['sma_20'], 'signal'] = -1  # Sell when price below SMA20
                print(f"Generated {(signals['signal'] != 0).sum()} signals based on price movements")

        # Count final signals
        buy_signals = (signals['signal'] == 1).sum()
        sell_signals = (signals['signal'] == -1).sum()
        no_signals = (signals['signal'] == 0).sum()
        print(f"Final signal counts: Buy: {buy_signals}, Sell: {sell_signals}, No position: {no_signals}")

        # Store signals
        self.signals = signals

        # Return just the signal column as a Series to match the expected format
        return signals['signal']

    def plot_sentiment(self, ax=None):
        """
        Plot sentiment data.

        Args:
            ax (matplotlib.axes.Axes): Axes to plot on

        Returns:
            matplotlib.axes.Axes: The axes with the plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        # Plot stock-specific sentiment
        if 'sentiment_signal' in self.data.columns:
            ax.plot(self.data.index, self.data['sentiment_signal'], 
                    label='Stock Sentiment', alpha=0.7, color='blue')

        # Plot market sentiment
        if 'market_sentiment_signal' in self.data.columns:
            ax.plot(self.data.index, self.data['market_sentiment_signal'], 
                    label='Market Sentiment', alpha=0.7, color='green')

        # Plot combined sentiment
        if 'combined_sentiment' in self.data.columns:
            ax.plot(self.data.index, self.data['combined_sentiment'], 
                    label='Combined Sentiment', linewidth=2, color='red')

        # Add threshold lines
        ax.axhline(y=self.sentiment_threshold, color='green', linestyle='--', alpha=0.5)
        ax.axhline(y=-self.sentiment_threshold, color='red', linestyle='--', alpha=0.5)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.2)

        # Add labels and legend
        ax.set_title('Sentiment Analysis')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sentiment Score')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def plot_results(self, benchmark_returns=None):
        """
        Plot strategy results including sentiment data.

        Args:
            benchmark_returns (pandas.Series): Benchmark returns for comparison
        """
        # Call the parent class plot_results method
        fig = super().plot_results(benchmark_returns)

        # Add a new subplot for sentiment
        fig.set_size_inches(12, 12)
        sentiment_ax = fig.add_subplot(4, 1, 4)
        self.plot_sentiment(sentiment_ax)

        # Adjust layout
        plt.tight_layout()

        return fig
