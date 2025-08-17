#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import os
import sys
from datetime import datetime, timedelta
import numpy as np

# Try to import nsepy and investpy
try:
    import nsepy

    USE_NSEPY = False
except ImportError:
    USE_NSEPY = False
    print("Warning: nsepy not found. Please install using: pip install nsepy")

try:
    import investpy

    USE_INVESTPY = False
except ImportError:
    USE_INVESTPY = False
    print("Warning: investpy not found. Please install using: pip install investpy")

try:
    import yfinance as yf

    USE_YFINANCE = True
except ImportError:
    USE_YFINANCE = False
    print("Warning: yfinance not found. Please install using: pip install yfinance")

nifty_tickers = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BEL.NS", "BHARTIARTL.NS",
    "CIPLA.NS", "COALINDIA.NS", "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS",
    "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS",
    "HAL.NS", "HINDUNILVR.NS", "ICICIBANK.NS",
    "INDUSINDBK.NS", "INFY.NS", "ITC.NS",
    "JIOFIN.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS", "M&M.NS",
    "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS",
    "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", "SHRIRAMFIN.NS", "SUNPHARMA.NS",
    "TCS.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TECHM.NS",
    "TITAN.NS", "TRENT.NS", "ULTRACEMCO.NS", "WIPRO.NS"
]


class NiftyDataFetcher:
    """
    Data fetcher specifically for Nifty 50 and other Indian market indices.
    """

    def __init__(self, config_path=None):
        """
        Initialize the Nifty data fetcher.
        
        Args:
            config_path (str, optional): Path to configuration file
        """
        self.config_path = config_path
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)

    def generate_sample_data(self, start_date, end_date, index_name="NIFTY 50"):
        """
        Generate sample data if fetching from external sources fails.
        
        Args:
            start_date (datetime.date): Start date for the data
            end_date (datetime.date): End date for the data
            index_name (str): Index name
            
        Returns:
            pandas.DataFrame: DataFrame containing sample data
        """
        print(f"Generating sample data for {index_name}...")

        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days

        # Create random price data
        np.random.seed(42)  # For reproducibility
        n_days = len(date_range)

        # Start with an initial price of 18000 for Nifty 50
        initial_price = 18000
        if 'BANK' in index_name:
            initial_price = 40000  # Nifty Bank is usually higher

        # Generate daily returns with a slight upward bias
        daily_returns = np.random.normal(0.0003, 0.012, n_days)

        # Calculate prices
        prices = [initial_price]
        for ret in daily_returns:
            prices.append(prices[-1] * (1 + ret))
        prices = prices[1:]  # Remove the initial price

        # Create sample data
        sample_data = pd.DataFrame({
            'Open': [price * (1 - np.random.uniform(0, 0.01)) for price in prices],
            'High': [price * (1 + np.random.uniform(0, 0.015)) for price in prices],
            'Low': [price * (1 - np.random.uniform(0, 0.015)) for price in prices],
            'Close': prices,
            'Volume': [int(np.random.uniform(10000000, 100000000)) for _ in range(n_days)]
        }, index=date_range)

        # Round prices to 2 decimal places
        for col in ['Open', 'High', 'Low', 'Close']:
            sample_data[col] = sample_data[col].round(2)

        print(f"Sample data generated successfully. Shape: {sample_data.shape}")

        # Save the sample data
        file_name = f"{index_name.replace(' ', '_')}_1d_sample.csv"
        file_path = os.path.join(self.data_dir, file_name)
        sample_data.to_csv(file_path)
        print(f"Sample data saved to {file_path}")

        return sample_data

    @staticmethod
    def get_start_end_date(period):
        # Calculate start and end dates based on period
        end_date = datetime.now().date()
        if period.endswith('y'):
            years = int(period[:-1])
            start_date = end_date - timedelta(days=365 * years)
        elif period.endswith('m'):
            months = int(period[:-1])
            start_date = end_date - timedelta(days=30 * months)
        elif period.endswith('d'):
            days = int(period[:-1])
            start_date = end_date - timedelta(days=days)
        else:
            # Default to 1 year
            start_date = end_date - timedelta(days=365)

            # Fix for fetch date range - ensure dates are in the past not future
        if start_date > end_date:
            start_date, end_date = end_date, start_date
        return end_date, start_date

    def fetch_all_nifty_data(self, period='5y', interval='1d'):

        # nifty_tickers = ["ADANIENT.NS"]

        print(self.data_dir)
        directory = f"{self.data_dir}/{interval}/{period}"

        if not os.path.exists(directory):
            os.makedirs(directory)

        for ticker_symbol in nifty_tickers:
            end_date, start_date = self.get_start_end_date(period)
            df = yf.download(ticker_symbol, period=period, interval=interval,
                             start=start_date, end=end_date)

            df = df.reset_index()
            df = df.sort_values(by='Date', ascending=True)

            file_name = f"{ticker_symbol.replace(' ', '_')}_{interval}_{period}.csv"
            file_path = os.path.join(directory, file_name)

            df.columns = df.columns.get_level_values(0)
            df.to_csv(file_path, index=False)
            print(f"Data saved to {file_path}")

    @staticmethod
    def fetch_data_from_csv(file_path):
        return pd.read_csv(
            file_path,
            parse_dates=['Date'],  # parse the Date column properly
            dtype={  # specify dtypes
                'Close': float,
                'High': float,
                'Low': float,
                'Open': float,
                'Volume': int
            }
        )

    def fetch_ticker_data_from_csv(self, ticker_symbol, period='10y', interval='1d'):
        file_name = f"{ticker_symbol.replace(' ', '_')}_{interval}_{period}.csv"
        file_path = os.path.join(self.data_dir, file_name)
        return self.fetch_data_from_csv(file_path)

    def fetch_nifty_data(self, period='5y', interval='1d', ticker_symbol="NIFTY 50", save=True,
                         use_sample_if_needed=True):
        """
        Fetch Nifty 50 historical data.
        
        Args:
            period (str): Period for data ('1y', '2y', '5y', etc.)
            ticker_symbol (str): Index name (default: "NIFTY 50")
            save (bool): Whether to save the data to a CSV file

        Returns:
            pandas.DataFrame: DataFrame containing the historical data
        """

        end_date, start_date = self.get_start_end_date(period)

        print(f"Fetching {ticker_symbol} data from {start_date} to {end_date}")

        # Try to fetch real data
        try:

            if USE_YFINANCE:
                try:
                    print(f"Using yfinance to fetch data from {start_date} to {end_date}")

                    df = yf.download(ticker_symbol, period=period, interval=interval,
                                     start=start_date, end=end_date)

                    print(f"Data fetched successfully using Yahoo finance. Shape: {df.shape}")
                    print(df.info())

                    df = df.reset_index()
                    df = df.sort_values(by='Date', ascending=True)
                    print(df.info())

                    # Save the data if requested
                    if save and len(df) > 0:
                        file_name = f"{ticker_symbol.replace(' ', '_')}_{interval}_{period}.csv"
                        file_path = os.path.join(self.data_dir, file_name)

                        # To Get rid of multi-index column headers
                        df.columns = df.columns.get_level_values(0)
                        df.to_csv(file_path, index=False)

                        print(f"Data saved to {file_path}")

                        df = self.fetch_data_from_csv(file_path)

                    return df

                except Exception as e:
                    print(f"Error fetching data with Yahoo finance: {e}")

            # Try nsepy first
            if USE_NSEPY:
                try:
                    print(f"Using nsepy to fetch data from {start_date} to {end_date}")
                    data = nsepy.get_history(symbol=ticker_symbol,
                                             start=start_date,
                                             end=end_date,
                                             index=True)

                    print(f"Raw data from nsepy: {data.shape} rows")

                    # Rename columns to match our system's expected format if needed
                    if len(data) > 0:
                        if 'Close' not in data.columns and 'CLOSE' in data.columns:
                            data = data.rename(columns={
                                'OPEN': 'Open',
                                'HIGH': 'High',
                                'LOW': 'Low',
                                'CLOSE': 'Close',
                                'VOLUME': 'Volume'
                            })

                    # If still no data, try with today's date as end_date
                    if len(data) == 0:
                        yesterday = datetime.now().date() - timedelta(days=1)
                        print(f"Still no data, trying with yesterday's date: {yesterday}")
                        data = nsepy.get_history(symbol=ticker_symbol,
                                                 start=start_date,
                                                 end=yesterday,
                                                 index=True)
                        print(f"Yesterday end date result: {data.shape} rows")

                    if len(data) > 0:
                        print(f"Data fetched successfully using nsepy. Shape: {data.shape}")

                        # Save the data if requested
                        if save and len(data) > 0:
                            file_name = f"{ticker_symbol.replace(' ', '_')}_{interval}.csv"
                            file_path = os.path.join(self.data_dir, file_name)
                            data.to_csv(file_path)
                            print(f"Data saved to {file_path}")

                        return data

                except Exception as e:
                    print(f"Error fetching data with nsepy: {e}")

            # Try using investpy as a fallback
            if USE_INVESTPY:
                try:
                    print(f"Trying investpy to fetch data from {start_date} to {end_date}")

                    # Convert dates to string format
                    start_str = start_date.strftime('%d/%m/%Y')
                    end_str = end_date.strftime('%d/%m/%Y')

                    # Map index names to investpy format if needed
                    index_map = {
                        "NIFTY 50": "Nifty 50",
                        "NIFTY BANK": "Nifty Bank",
                        "NIFTY NEXT 50": "Nifty Next 50"
                    }

                    investpy_index = index_map.get(ticker_symbol, ticker_symbol)
                    print(f"Using investpy index name: {investpy_index}")

                    data = investpy.get_index_historical_data(
                        index=investpy_index,
                        country='India',
                        from_date=start_str,
                        to_date=end_str
                    )

                    if len(data) > 0:
                        print(f"Data fetched successfully using investpy. Shape: {data.shape}")

                        # Save the data if requested
                        if save and len(data) > 0:
                            file_name = f"{ticker_symbol.replace(' ', '_')}_1d.csv"
                            file_path = os.path.join(self.data_dir, file_name)
                            data.to_csv(file_path)
                            print(f"Data saved to {file_path}")

                        return data

                except Exception as e:
                    print(f"Error fetching data with investpy: {e}")

            # If we reach here, both methods failed
            if use_sample_if_needed:
                print("Both nsepy and investpy failed to fetch data. Generating sample data instead.")
                return self.generate_sample_data(start_date, end_date, ticker_symbol)
            else:
                raise ValueError("Failed to fetch data from any source.")

        except Exception as e:
            if use_sample_if_needed:
                print(f"Error fetching data: {e}. Generating sample data instead.")
                return self.generate_sample_data(start_date, end_date, ticker_symbol)
            else:
                raise

    def load_saved_data(self, index_name="NIFTY 50"):
        """
        Load previously saved data for the index.
        
        Args:
            index_name (str): Index name (default: "NIFTY 50")
            
        Returns:
            pandas.DataFrame: DataFrame containing the historical data
        """
        file_name = f"{index_name.replace(' ', '_')}_1d.csv"
        file_path = os.path.join(self.data_dir, file_name)

        if os.path.exists(file_path):
            data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            print(f"Loaded data from {file_path}. Shape: {data.shape}")
            return data
        else:
            print(f"No saved data found at {file_path}")
            return None


# import urllib3
# import yfinance as yf
# import requests as requests
# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


if __name__ == "__main__":
    fetcher = NiftyDataFetcher()
    # data = fetcher.fetch_nifty_data(period="5y")

    fetcher.fetch_all_nifty_data(period="5y", interval='1d')

    fetcher.fetch_ticker_data_from_csv("ADANIENT.NS")

    # data = NiftyDataFetcher.fetch_data_from_csv(
    #     '/Users/neelansh/Desktop/Projects/My Projects/Stock Market Data/Nifty_50_till_13June2025.csv')
    #
    # # Print basic statistics
    # print("\nBasic information about the data:")
    # print(data.info())
    #
    # print("\nSample data:")
    # print(data.head())
    #
    # # Calculate and print basic return statistics
    # if 'Close' in data.columns:
    #     returns = data['Close'].pct_change().dropna()
    #     print("\nReturn statistics:")
    #     print(returns.describe())
    #
    #     annualized_return = ((1 + returns.mean()) ** 252) - 1
    #     annualized_volatility = returns.std() * (252 ** 0.5)
    #     sharpe_ratio = annualized_return / annualized_volatility
    #
    #     print(f"\nAnnualized Return: {annualized_return:.2%}")
    #     print(f"Annualized Volatility: {annualized_volatility:.2%}")
    #     print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
