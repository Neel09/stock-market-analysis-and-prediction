import os
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import time

class SentimentAnalyzer:
    """
    A class to analyze sentiment from financial news and social media using LLM.
    """
    def __init__(self, config=None):
        """
        Initialize the SentimentAnalyzer.

        Args:
            config (dict): Configuration dictionary containing LLM API settings
        """
        self.config = config or {}
        self.api_key = self.config.get('llm_api_key', os.environ.get('LLM_API_KEY', ''))
        self.api_url = self.config.get('llm_api_url', 'https://api.openai.com/v1/chat/completions')
        self.model = self.config.get('llm_model', 'gpt-3.5-turbo')
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 2)

        # Print configuration for debugging
        print(f"SentimentAnalyzer initialized with API URL: {self.api_url}, Model: {self.model}")
        print(f"API Key present: {'Yes' if self.api_key else 'No'}")

        if not self.api_key:
            print("Warning: No API key provided for LLM. Sentiment analysis will not work.")

    def analyze_text(self, text):
        """
        Analyze the sentiment of a given text using LLM.

        Args:
            text (str): The text to analyze

        Returns:
            dict: A dictionary containing sentiment scores and analysis
        """
        if not self.api_key:
            return {'score': 0, 'label': 'neutral', 'confidence': 0}

        prompt = f"""
        Analyze the sentiment of the following financial news or social media post about stocks or markets.
        Rate the sentiment on a scale from -1 (very negative) to 1 (very positive), where 0 is neutral.
        Also provide a sentiment label (negative, neutral, positive) and a confidence score (0-1).

        Text: {text}

        Return only a JSON object with the following format:
        {{
            "score": (float between -1 and 1),
            "label": (string: "negative", "neutral", or "positive"),
            "confidence": (float between 0 and 1)
        }}
        """

        return self._query_llm(prompt)

    def analyze_news_batch(self, news_list):
        """
        Analyze sentiment for a batch of news articles.

        Args:
            news_list (list): List of news article texts

        Returns:
            list: List of sentiment analysis results
        """
        results = []
        for news in news_list:
            result = self.analyze_text(news)
            results.append(result)
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
        return results


    def get_mocked_aggregate_sentiment(self):
        return {'aggregate_score': -0.08833819241982506, 'sentiment_signal': -0.08833819241982506}

    def get_aggregate_sentiment(self, sentiment_results):
        """
        Calculate aggregate sentiment from multiple results.

        Args:
            sentiment_results (list): List of sentiment analysis results

        Returns:
            dict: Aggregated sentiment metrics
        """
        if not sentiment_results:
            return {'aggregate_score': 0, 'sentiment_signal': 0}

        # Weight each score by its confidence
        weighted_scores = [r['score'] * r['confidence'] for r in sentiment_results]
        confidence_sum = sum(r['confidence'] for r in sentiment_results)

        if confidence_sum == 0:
            return {'aggregate_score': 0, 'sentiment_signal': 0}

        aggregate_score = sum(weighted_scores) / confidence_sum

        # Convert to a trading signal between -1 and 1
        sentiment_signal = aggregate_score

        return {
            'aggregate_score': aggregate_score,
            'sentiment_signal': sentiment_signal
        }

    def _query_llm(self, prompt):
        """
        Query the LLM API with retry logic.

        Args:
            prompt (str): The prompt to send to the LLM

        Returns:
            dict: The parsed response from the LLM
        """
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }

        data = {
            'model': self.model,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.3,
            'max_tokens': 150
        }

        print(f"Sending request to {self.api_url} with model {self.model}")

        for attempt in range(self.max_retries):
            try:
                response = requests.post(self.api_url, headers=headers, json=data)
                print(f"Response status code: {response.status_code}")

                if response.status_code != 200:
                    print(f"Error response: {response.text}")

                response.raise_for_status()

                response_data = response.json()
                content = response_data['choices'][0]['message']['content']
                print(f"Received response: {content[:100]}...")  # Print first 100 chars

                # Parse the JSON response
                try:
                    # Remove any backticks and json tags that might be in the response
                    cleaned_content = content
                    if '```json' in cleaned_content:
                        cleaned_content = cleaned_content.replace('```json', '')
                    if '```' in cleaned_content:
                        cleaned_content = cleaned_content.replace('```', '')

                    # Try to parse the JSON
                    result = json.loads(cleaned_content)
                    return result
                except json.JSONDecodeError:
                    print(f"Failed to parse LLM response as JSON: {content}")
                    # Try to extract JSON using regex as a fallback
                    import re
                    json_pattern = r'{\s*"score"\s*:\s*(-?\d+\.?\d*)\s*,\s*"label"\s*:\s*"(\w+)"\s*,\s*"confidence"\s*:\s*(\d+\.?\d*)\s*}'
                    match = re.search(json_pattern, content)
                    if match:
                        score = float(match.group(1))
                        label = match.group(2)
                        confidence = float(match.group(3))
                        print(f"Extracted values using regex: score={score}, label={label}, confidence={confidence}")
                        return {'score': score, 'label': label, 'confidence': confidence}

                    return {'score': 0, 'label': 'neutral', 'confidence': 0}

            except requests.exceptions.RequestException as e:
                print(f"API request failed (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff

        # If all retries fail, return a neutral sentiment
        print("All API request attempts failed. Returning neutral sentiment.")
        return {'score': 0, 'label': 'neutral', 'confidence': 0}
