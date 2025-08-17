import os
import pandas as pd
import numpy as np
import time
from src.utils.strategy_explainer import StrategyExplainer

class ChatbotInterface:
    """
    A class to provide chatbot functionality for the trading system UI.
    """
    def __init__(self, config=None):
        """
        Initialize the ChatbotInterface.

        Args:
            config (dict): Configuration dictionary containing LLM API settings
        """
        self.config = config or {}
        self.strategy_explainer = StrategyExplainer(config)
        self.conversation_history = []
        self.context = {}
        
    def add_context(self, key, value):
        """
        Add context information that will be used to answer queries.
        
        Args:
            key (str): Context key (e.g., 'strategies', 'results')
            value: Context value
        """
        self.context[key] = value
        
    def clear_context(self):
        """Clear all context information."""
        self.context = {}
        
    def add_strategies_context(self, strategies_results):
        """
        Add strategies results as context.
        
        Args:
            strategies_results (dict): Dictionary containing results for each strategy
        """
        # Extract key metrics for each strategy
        strategies_context = {}
        
        for strategy_id, results in strategies_results.items():
            if 'metrics' in results:
                metrics = results['metrics']
                strategies_context[strategy_id] = {
                    'total_return': metrics.get('total_return', 0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'max_drawdown': metrics.get('maximum_drawdown', 0),
                    'win_rate': metrics.get('win_rate', 0)
                }
        
        self.add_context('strategies_results', strategies_context)
    
    def add_message(self, message, is_user=True):
        """
        Add a message to the conversation history.
        
        Args:
            message (str): The message content
            is_user (bool): Whether the message is from the user (True) or the system (False)
        """
        self.conversation_history.append({
            'content': message,
            'is_user': is_user,
            'timestamp': time.time()
        })
    
    def get_conversation_history(self, max_messages=10):
        """
        Get the conversation history.
        
        Args:
            max_messages (int): Maximum number of messages to return
            
        Returns:
            list: List of message dictionaries
        """
        return self.conversation_history[-max_messages:]
    
    def clear_conversation(self):
        """Clear the conversation history."""
        self.conversation_history = []
    
    def get_response(self, query):
        """
        Get a response to a user query.
        
        Args:
            query (str): The user's question
            
        Returns:
            str: The response to the query
        """
        # Add user message to history
        self.add_message(query, is_user=True)
        
        # Generate response using strategy explainer
        response = self.strategy_explainer.answer_query(query, self.context)
        
        # Add system response to history
        self.add_message(response, is_user=False)
        
        return response
    
    def get_strategy_explanation(self, comparison_data, strategy_names):
        """
        Get an explanation of strategy comparison.
        
        Args:
            comparison_data (dict): Dictionary containing comparison metrics for each strategy
            strategy_names (dict): Dictionary mapping strategy IDs to display names
            
        Returns:
            str: A detailed explanation of the strategy comparison
        """
        explanation = self.strategy_explainer.explain_strategy_comparison(comparison_data, strategy_names)
        
        # Add explanation to conversation history
        self.add_message("Please explain the strategy comparison results.", is_user=True)
        self.add_message(explanation, is_user=False)
        
        return explanation
    
    def get_signals_explanation(self, signals_data, strategy_name, price_data=None):
        """
        Get an explanation of trading signals.
        
        Args:
            signals_data (pandas.DataFrame): DataFrame containing trading signals
            strategy_name (str): Name of the strategy
            price_data (pandas.DataFrame, optional): Price data corresponding to the signals
            
        Returns:
            str: An explanation of the trading signals
        """
        explanation = self.strategy_explainer.explain_trading_signals(signals_data, strategy_name, price_data)
        
        # Add explanation to conversation history
        self.add_message(f"Please explain the trading signals for {strategy_name}.", is_user=True)
        self.add_message(explanation, is_user=False)
        
        return explanation
    
    def get_welcome_message(self):
        """
        Get a welcome message for the chatbot.
        
        Returns:
            str: Welcome message
        """
        welcome_message = """
        # Welcome to the Trading Strategy Assistant!
        
        I can help you understand your trading strategy results and answer questions about:
        
        - Strategy performance and comparisons
        - Trading signals and their interpretation
        - Technical indicators and their meanings
        - Risk management and portfolio optimization
        - General questions about algorithmic trading
        
        What would you like to know about your trading strategies?
        """
        
        return welcome_message
    
    def get_suggested_questions(self):
        """
        Get a list of suggested questions for the user.
        
        Returns:
            list: List of suggested questions
        """
        general_questions = [
            "Which strategy performed best in terms of returns?",
            "Which strategy has the best risk-adjusted performance?",
            "How do these strategies perform in different market conditions?",
            "What are the main differences between these strategies?",
            "How can I combine these strategies for better performance?"
        ]
        
        # Add strategy-specific questions if we have context
        if 'strategies_results' in self.context:
            strategies = list(self.context['strategies_results'].keys())
            if strategies:
                strategy_questions = [
                    f"Why did {strategies[0]} perform better/worse than others?",
                    f"What are the strengths and weaknesses of {strategies[0]}?",
                    "What risk management techniques would you recommend for these strategies?"
                ]
                return general_questions + strategy_questions
        
        return general_questions