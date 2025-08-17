import os
import pandas as pd
import requests
import json
from datetime import datetime, timedelta
import time

class NewsFetcher:
    """
    A class to fetch financial news for stocks and markets.
    """
    def __init__(self, config=None):
        """
        Initialize the NewsFetcher.

        Args:
            config (dict): Configuration dictionary containing API settings
        """
        self.config = config or {}
        self.api_key = self.config.get('news_api_key', os.environ.get('NEWS_API_KEY', ''))
        self.default_sources = self.config.get('news_sources', 'bloomberg,financial-times,the-wall-street-journal,fortune,business-insider')
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 2)

        if not self.api_key:
            print("Warning: No API key provided for news API. News fetching will use mock data.")

    def fetch_news(self, symbol, days=7, max_results=10):
        """
        Fetch news articles for a specific stock symbol.

        Args:
            symbol (str): The stock symbol to fetch news for
            days (int): Number of days to look back
            max_results (int): Maximum number of results to return

        Returns:
            list: List of news articles with title, description, and published date
        """
        if not self.api_key:
            return self._get_mock_news(symbol, max_results)

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Format dates for API
        from_date = start_date.strftime('%Y-%m-%d')
        to_date = end_date.strftime('%Y-%m-%d')

        # Prepare query
        company_name = self._get_company_name(symbol)
        query = f"{company_name} OR {symbol} stock"

        # NewsAPI endpoint
        url = "https://newsapi.org/v2/everything"

        params = {
            'q': query,
            'from': from_date,
            'to': to_date,
            'sortBy': 'relevancy',
            'language': 'en',
            'sources': self.default_sources,
            'apiKey': self.api_key,
            'pageSize': max_results
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()

                data = response.json()
                articles = data.get('articles', [])

                # Format the results
                results = []
                for article in articles:
                    results.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'content': article.get('content', ''),
                        'published_at': article.get('publishedAt', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'url': article.get('url', '')
                    })

                return results

            except requests.exceptions.RequestException as e:
                print(f"News API request failed (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff

        # If all retries fail, return mock data
        return self._get_mock_news(symbol, max_results)

    def fetch_market_news(self, market="India", days=7, max_results=10):
        """
        Fetch general market news.

        Args:
            market (str): Market to fetch news for (e.g., "India", "Nifty", "BSE")
            days (int): Number of days to look back
            max_results (int): Maximum number of results to return

        Returns:
            list: List of news articles
        """
        if not self.api_key:
            return self._get_mock_market_news(market, max_results)

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Format dates for API
        from_date = start_date.strftime('%Y-%m-%d')
        to_date = end_date.strftime('%Y-%m-%d')

        # Prepare query
        query = f"{market} stock market OR Nifty OR Sensex"

        # NewsAPI endpoint
        url = "https://newsapi.org/v2/everything"

        params = {
            'q': query,
            'from': from_date,
            'to': to_date,
            'sortBy': 'relevancy',
            'language': 'en',
            'sources': self.default_sources,
            'apiKey': self.api_key,
            'pageSize': max_results
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()

                data = response.json()
                articles = data.get('articles', [])

                # Format the results
                results = []
                for article in articles:
                    results.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'content': article.get('content', ''),
                        'published_at': article.get('publishedAt', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'url': article.get('url', '')
                    })

                return results

            except requests.exceptions.RequestException as e:
                print(f"News API request failed (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff

        # If all retries fail, return mock data
        return self._get_mock_market_news(market, max_results)

    def _get_company_name(self, symbol):
        """
        Get the company name from a stock symbol.

        Args:
            symbol (str): Stock symbol

        Returns:
            str: Company name
        """
        # This is a simplified mapping for Nifty 50 stocks
        # In a real implementation, this would be more comprehensive or use an API
        symbol_to_name = {
            'RELIANCE.NS': 'Reliance Industries',
            'TCS.NS': 'Tata Consultancy Services',
            'HDFCBANK.NS': 'HDFC Bank',
            'INFY.NS': 'Infosys',
            'HINDUNILVR.NS': 'Hindustan Unilever',
            'ICICIBANK.NS': 'ICICI Bank',
            'SBIN.NS': 'State Bank of India',
            'BHARTIARTL.NS': 'Bharti Airtel',
            'ITC.NS': 'ITC Limited',
            'KOTAKBANK.NS': 'Kotak Mahindra Bank',
            'NIFTY 50': 'Nifty 50 Index'
        }

        return symbol_to_name.get(symbol, symbol.replace('.NS', ''))

    def _get_mock_news(self, symbol, count=5):
        """
        Generate mock news for testing when API key is not available.

        Args:
            symbol (str): Stock symbol
            count (int): Number of mock articles to generate

        Returns:
            list: List of mock news articles
        """
        company_name = self._get_company_name(symbol)

        mock_templates = [
            {"title": "{company} Reports Strong Quarterly Earnings", 
             "description": "{company} exceeded analyst expectations with quarterly revenue growth of 15%.", 
             "sentiment": "positive"},

            {"title": "{company} Announces New Product Launch", 
             "description": "{company} is set to launch a new product line next month, expanding its market reach.", 
             "sentiment": "positive"},

            {"title": "{company} Faces Regulatory Scrutiny", 
             "description": "Regulators are investigating {company} over potential compliance issues.", 
             "sentiment": "negative"},

            {"title": "{company} Stock Downgraded by Analysts", 
             "description": "Several analysts have downgraded {company} stock citing concerns about future growth.", 
             "sentiment": "negative"},

            {"title": "{company} Maintains Stable Outlook Despite Market Volatility", 
             "description": "{company} executives reaffirmed their annual guidance during the investor call.", 
             "sentiment": "neutral"},

            {"title": "{company} Announces Partnership with Tech Giant", 
             "description": "A new strategic partnership between {company} and a major tech firm was announced today.", 
             "sentiment": "positive"},

            {"title": "Investors Cautious About {company}'s Expansion Plans", 
             "description": "Market watchers express mixed opinions about {company}'s recent expansion strategy.", 
             "sentiment": "neutral"},

            {"title": "{company} Cuts Forecast Amid Economic Uncertainty", 
             "description": "{company} has reduced its annual forecast citing macroeconomic headwinds.", 
             "sentiment": "negative"}
        ]

        # Generate random dates within the last week
        now = datetime.now()

        results = []
        for i in range(min(count, len(mock_templates))):
            template = mock_templates[i]
            days_ago = i % 7
            mock_date = (now - timedelta(days=days_ago)).isoformat()

            results.append({
                'title': template['title'].format(company=company_name),
                'description': template['description'].format(company=company_name),
                'content': template['description'].format(company=company_name) + " Industry experts are closely monitoring developments.",
                'published_at': mock_date,
                'source': 'Mock Financial News',
                'url': 'https://example.com/mock-news'
            })

        return results

    def _get_mock_market_news(self, market="India", count=5):
        """
        Generate mock market news for testing when API key is not available.

        Args:
            market (str): Market name
            count (int): Number of mock articles to generate

        Returns:
            list: List of mock market news articles
        """
        mock_templates = [
            {"title": "{market} Market Rallies on Positive Economic Data", 
             "description": "The {market} stock market saw gains following better-than-expected economic indicators.", 
             "sentiment": "positive"},

            {"title": "Inflation Concerns Weigh on {market} Stocks", 
             "description": "Rising inflation figures have created downward pressure on the {market} market.", 
             "sentiment": "negative"},

            {"title": "{market} Market Remains Flat Amid Mixed Signals", 
             "description": "Investors in the {market} market are cautious as mixed economic signals emerge.", 
             "sentiment": "neutral"},

            {"title": "Foreign Investors Increase Stakes in {market} Equities", 
             "description": "Foreign institutional investors have increased their allocation to {market} stocks.", 
             "sentiment": "positive"},

            {"title": "Central Bank Policy Shift Impacts {market} Market", 
             "description": "Recent monetary policy changes have created volatility in the {market} stock market.", 
             "sentiment": "neutral"},

            {"title": "{market} Market Hits New Record High", 
             "description": "The {market} stock index reached an all-time high today, driven by tech and financial sectors.", 
             "sentiment": "positive"},

            {"title": "Analysts Predict Correction in Overheated {market} Market", 
             "description": "Several market analysts are warning of a potential correction in the {market} stock market.", 
             "sentiment": "negative"},

            {"title": "Economic Growth Forecast Revised for {market}", 
             "description": "Economists have revised growth projections for the {market} economy, affecting market sentiment.", 
             "sentiment": "neutral"}
        ]

        # Generate random dates within the last week
        now = datetime.now()

        results = []
        for i in range(min(count, len(mock_templates))):
            template = mock_templates[i]
            days_ago = i % 7
            mock_date = (now - timedelta(days=days_ago)).isoformat()

            results.append({
                'title': template['title'].format(market=market),
                'description': template['description'].format(market=market),
                'content': template['description'].format(market=market) + " Analysts continue to monitor global factors affecting the market.",
                'published_at': mock_date,
                'source': 'Mock Market News',
                'url': 'https://example.com/mock-market-news'
            })

        return results
