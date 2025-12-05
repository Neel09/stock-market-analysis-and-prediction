#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Support for Indian market indices
try:
    from src.utils.nifty_data_fetcher import NiftyDataFetcher

    NIFTY_AVAILABLE = True
except ImportError:
    NIFTY_AVAILABLE = False


class DataFetcher:
    """
    A class to fetch and preprocess stock market data.
    Supports both general market data via yfinance and Indian market indices.
    """

    def __init__(self, config_path=None):
        """
        Initialize the DataFetcher with configuration.
        
        Args:
            config_path (str): Path to the configuration file
        """
        # Load configuration
        if config_path is None:
            # Use default config path
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                       'config', 'config.json')

        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            print(f"Config file not found at {config_path}, using default settings")
            self.config = {'data': {}}

        # Ensure data directory exists
        self.data_dir = os.path.abspath(
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         'data'))
        os.makedirs(self.data_dir, exist_ok=True)

        # Set default parameters
        self.default_symbols = self.config.get('data', {}).get('default_symbols', ['AAPL'])
        self.default_timeframe = self.config.get('data', {}).get('default_timeframe', '1d')
        self.default_period = self.config.get('data', {}).get('default_period', '1y')

        # Initialize Nifty data fetcher if available
        self.nifty_fetcher = NiftyDataFetcher(config_path) if NIFTY_AVAILABLE else None

        # List of Indian market symbols that should use the Nifty fetcher
        self.indian_symbols = ['NIFTY', 'NIFTY50', 'NIFTY 50', 'NIFTY BANK', 'NIFTY NEXT 50', 'BANKNIFTY', 'SENSEX']

    def is_indian_market(self, symbol):
        """
        Check if the symbol is from the Indian market.
        
        Args:
            symbol (str): The stock/index symbol
            
        Returns:
            bool: True if it's an Indian market symbol
        """
        return (symbol.upper() in [s.upper() for s in self.indian_symbols] or
                symbol.startswith('NSE:') or
                'INDIA' in symbol.upper())

    def fetch_data(self, symbol, start_date=None, end_date=None, period=None, interval=None, use_sample_data=True):
        """
        Fetch data for a given symbol.
        Automatically detects Indian market symbols and uses the appropriate fetcher.
        
        Args:
            symbol (str): The stock symbol to fetch
            start_date (str, optional): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format
            period (str, optional): Period to fetch (e.g., '1y', '5y')
            interval (str, optional): Data interval (e.g., '1d', '1h')
            
        Returns:
            pandas.DataFrame: The fetched data
        """

        """
        Return saved data
        """
        if use_sample_data:
            return self.nifty_fetcher.fetch_ticker_data_from_csv(symbol, period='10y', interval='1d')

        # Set default values if not provided
        interval = interval or self.default_timeframe
        period = period or self.default_period

        # Check if this is an Indian market symbol
        if self.is_indian_market(symbol) and self.nifty_fetcher is not None:
            print(f"Using Nifty data fetcher for {symbol}")

            # Format the symbol for nifty_data_fetcher
            if symbol.upper() in ['NIFTY', 'NIFTY50', 'NIFTY 50']:
                nifty_symbol = "NIFTY 50"
            elif symbol.upper() in ['BANKNIFTY', 'NIFTY BANK']:
                nifty_symbol = "NIFTY BANK"
            else:
                nifty_symbol = symbol

            # Use nifty_data_fetcher
            return self.nifty_fetcher.fetch_nifty_data(period=period, ticker_symbol=nifty_symbol)

        # Use yfinance for other symbols
        print(f"Using yfinance for {symbol}")

        # If start_date and end_date are provided, use them
        if start_date and end_date:
            data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
        # Otherwise use period
        else:
            data = yf.download(symbol, period=period, interval=interval)

        # Check if data is empty
        if data.empty:
            print(f"No data found for {symbol}")
            return None

        # Basic preprocessing
        data.index = pd.to_datetime(data.index)
        data = data.dropna()

        # Save data
        csv_path = os.path.join(self.data_dir, f"{symbol}_{interval}.csv")
        data.to_csv(csv_path)

        print(f"Data for {symbol} saved to {csv_path}")
        return data

    def fetch_multiple(self, symbols=None, start_date=None, end_date=None, period=None, interval=None):
        """
        Fetch data for multiple symbols.
        
        Args:
            symbols (list, optional): List of stock symbols to fetch
            start_date (str, optional): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format
            period (str, optional): Period to fetch (e.g., '1y', '5y')
            interval (str, optional): Data interval (e.g., '1d', '1h')
            
        Returns:
            dict: Dictionary mapping symbols to their data frames
        """
        symbols = symbols or self.default_symbols
        result = {}

        for symbol in symbols:
            data = self.fetch_data(symbol, start_date, end_date, period, interval)
            if data is not None:
                result[symbol] = data

        return result


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Fetch stock market data')
    parser.add_argument('--symbol', type=str, help='Stock symbol to fetch', default='AAPL')
    parser.add_argument('--symbols', type=str, help='Comma-separated list of stock symbols')
    parser.add_argument('--start', type=str, help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end', type=str, help='End date in YYYY-MM-DD format')
    parser.add_argument('--period', type=str, help='Period to fetch (e.g., 1y, 5y)')
    parser.add_argument('--interval', type=str, help='Data interval (e.g., 1d, 1h)')
    parser.add_argument('--config', type=str, help='Path to config file',
                        default='../../config/config.json')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    fetcher = DataFetcher(args.config)

    if args.symbols:
        symbols = args.symbols.split(',')
        fetcher.fetch_multiple(symbols, args.start, args.end, args.period, args.interval)
    else:
        fetcher.fetch_data(args.symbol, args.start, args.end, args.period, args.interval)
