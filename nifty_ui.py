#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from datetime import datetime, timedelta
import time
import json

# To Avoid decompression bomb
Image.MAX_IMAGE_PIXELS = None

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.backtest.run_backtest import BacktestRunner
from src.utils.strategy_explainer import StrategyExplainer
from src.utils.chatbot_interface import ChatbotInterface
from src.strategies.sentiment_strategy import SentimentStrategy

nifty_tickers = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BEL.NS", "BHARTIARTL.NS",
    "TATAMOTORS.NS"
    # "CIPLA.NS", "COALINDIA.NS", "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS",
    # "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS",
    # "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS", "INFY.NS", "ITC.NS",
    # "JIOFIN.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS", "M&M.NS",
    # "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS",
    # "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", "SHRIRAMFIN.NS", "SUNPHARMA.NS",
    # "TCS.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TECHM.NS",
    # "TITAN.NS", "TRENT.NS", "ULTRACEMCO.NS", "WIPRO.NS"
]

# Set page configuration
st.set_page_config(
    page_title="Stock Market Analysis & Prediction System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 30px;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #42A5F5;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 5px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 10px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666666;
    }
</style>
""", unsafe_allow_html=True)


def load_results(strategy_name):
    """Load backtest results for a specific strategy."""
    metrics_path = os.path.join('results', 'backtest', f"{strategy_name}_metrics.json")
    positions_path = os.path.join('results', 'backtest', f"{strategy_name}_positions.csv")
    returns_path = os.path.join('results', 'backtest', f"{strategy_name}_returns.csv")
    portfolio_path = os.path.join('results', 'backtest', f"{strategy_name}_portfolio.csv")
    plot_path = os.path.join('results', 'backtest', f"{strategy_name}_plot.png")

    try:
        with open(metrics_path) as f:
            metrics_dict = json.load(f)

        # make it a one-row DataFrame so it's consistent
        metrics = pd.DataFrame([metrics_dict])
        positions = pd.read_csv(positions_path, index_col=0)
        returns = pd.read_csv(returns_path, index_col=0)
        portfolio = pd.read_csv(portfolio_path, index_col=0)
        plot_img = Image.open(plot_path)

        return {
            'metrics': metrics,
            'positions': positions,
            'returns': returns,
            'portfolio': portfolio,
            'plot_img': plot_img
        }
    except Exception as e:
        st.error(f"Error loading results for {strategy_name}: {e}")
        return None


def format_metrics(metrics):
    """Format metrics for display."""
    if isinstance(metrics, pd.DataFrame):
        metrics = metrics.to_dict()

    if isinstance(metrics, dict):
        # Check if metrics is a nested dictionary or flat
        if any(isinstance(v, dict) for v in metrics.values()):
            # Convert to flat dictionary
            flat_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        flat_metrics[k2] = v2
                else:
                    flat_metrics[k] = v
            metrics = flat_metrics

    formatted = {}
    for key, value in metrics.items():
        if key in ['total_return', 'annualized_return', 'maximum_drawdown', 'win_rate']:
            formatted[key] = f"{value * 100:.2f}%"
        elif key in ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio']:
            formatted[key] = f"{value:.4f}"
        else:
            formatted[key] = f"{value}"

    return formatted


def run_backtest(symbol, period, interval, selected_strategies, strategy_params=None, config=None,
                 use_sample_data=False):
    """Run backtest and return results."""
    try:
        runner = BacktestRunner(config_path=None)

        # If custom config provided, update runner config
        if config:
            runner.config.update(config)

        data = runner.fetch_data(symbol, period, interval, use_sample_data=use_sample_data)

        if data is None or data.empty:
            st.error("No data available for the selected symbol and period.")
            return None

        # Add strategies with parameters
        for strategy in selected_strategies:
            params = strategy_params.get(strategy, {}) if strategy_params else {}
            runner.add_strategy(strategy, **params)

        # Run backtests
        results = runner.run_backtest()

        # Generate comparison
        comparison = runner.compare_strategies()

        # Plot comparison
        fig = runner.plot_comparison()

        return {
            'results': results,
            'comparison': comparison,
            'comparison_fig': fig
        }
    except Exception as e:
        st.error(f"Error running backtest: {e}")
        return None


def main():
    """Main function."""
    # Sidebar
    st.sidebar.markdown("## Stock Market Analysis Tool")
    st.sidebar.markdown("Configure your backtest parameters below:")

    # Parameters
    # symbol = st.sidebar.text_input("Symbol", value="NIFTY 50", key="symbol_input")
    symbol = st.sidebar.selectbox("Select a ticker symbol", options=nifty_tickers)

    period = st.sidebar.selectbox("Period", options=["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3,
                                  key="period_select")
    interval = st.sidebar.selectbox("Interval", options=["1d", "1wk", "1mo"], index=0, key="interval_select")

    # Strategies with configurable parameters
    st.sidebar.markdown("## Select Strategies")
    strategies = {
        "moving_average_crossover": "Moving Average Crossover",
        "rsi_strategy": "RSI Strategy",
        "lstm_strategy": "LSTM Strategy",
        "ml_strategy": "ML Strategy",
        "sentiment_strategy": "Sentiment Strategy"
        # "macd_strategy": "MACD Strategy",
        # "bollinger_bands": "Bollinger Bands"
    }

    # Dynamic strategy parameters
    strategy_params = {}
    selected_strategies = []

    # Expandable sections for each strategy
    for strategy_id, strategy_name in strategies.items():
        with st.sidebar.expander(f"{strategy_name}", expanded=False):
            use_strategy = st.checkbox("Use this strategy", value=False, key=f"use_{strategy_id}")

            if use_strategy:
                selected_strategies.append(strategy_id)

                # Strategy-specific parameters
                if strategy_id == "moving_average_crossover":
                    short_window = st.slider("Short Window", min_value=5, max_value=50, value=20, step=1,
                                             key=f"short_window_{strategy_id}")
                    long_window = st.slider("Long Window", min_value=20, max_value=200, value=50, step=5,
                                            key=f"long_window_{strategy_id}")
                    strategy_params[strategy_id] = {"short_window": short_window, "long_window": long_window}

                elif strategy_id == "rsi_strategy":
                    rsi_period = st.slider("RSI Period", min_value=5, max_value=30, value=14, step=1,
                                           key=f"rsi_period_{strategy_id}")
                    overbought = st.slider("Overbought Level", min_value=50, max_value=90, value=70, step=5,
                                           key=f"overbought_{strategy_id}")
                    oversold = st.slider("Oversold Level", min_value=10, max_value=50, value=30, step=5,
                                         key=f"oversold_{strategy_id}")
                    strategy_params[strategy_id] = {"rsi_period": rsi_period, "overbought": overbought,
                                                    "oversold": oversold}
                elif strategy_id == "lstm_strategy":
                    lstm_period = st.slider("LSTM Period", min_value=5, max_value=30, value=14, step=1,
                                           key=f"lstm_period_{strategy_id}")
                    lstm_window = st.slider("LSTM Window", min_value=5, max_value=50, value=20, step=1,
                                            key=f"lstm_window_{strategy_id}")
                    strategy_params[strategy_id] = {}
                elif strategy_id == "sentiment_strategy":
                    sentiment_threshold = st.slider("Sentiment Threshold", min_value=0.0, max_value=1.0, value=0.2, step=0.05,
                                                  key=f"sentiment_threshold_{strategy_id}")
                    market_sentiment_weight = st.slider("Market Sentiment Weight", min_value=0.0, max_value=1.0, value=0.4, step=0.05,
                                                      key=f"market_sentiment_weight_{strategy_id}")
                    use_technical_indicators = st.checkbox("Use Technical Indicators", value=True,
                                                         key=f"use_technical_indicators_{strategy_id}")
                    days = st.slider("Days to Look Back for News", min_value=1, max_value=30, value=7, step=1,
                                   key=f"days_{strategy_id}")
                    max_news = st.slider("Maximum News Articles to Analyze", min_value=1, max_value=50, value=10, step=1,
                                       key=f"max_news_{strategy_id}")
                    strategy_params[strategy_id] = {
                        "sentiment_threshold": sentiment_threshold,
                        "market_sentiment_weight": market_sentiment_weight,
                        "use_technical_indicators": use_technical_indicators,
                        "days": days,
                        "max_news": max_news
                    }
                # elif strategy_id == "macd_strategy":
                #     fast_period = st.slider("Fast Period", min_value=5, max_value=20, value=12, step=1,
                #                             key=f"fast_period_{strategy_id}")
                #     slow_period = st.slider("Slow Period", min_value=10, max_value=50, value=26, step=1,
                #                             key=f"slow_period_{strategy_id}")
                #     signal_period = st.slider("Signal Period", min_value=5, max_value=20, value=9, step=1,
                #                               key=f"signal_period_{strategy_id}")
                #     strategy_params[strategy_id] = {"fast_period": fast_period, "slow_period": slow_period,
                #                                     "signal_period": signal_period}
                #
                # elif strategy_id == "bollinger_bands":
                #     window = st.slider("Window", min_value=5, max_value=50, value=20, step=1,
                #                        key=f"window_{strategy_id}")
                #     num_std_dev = st.slider("Standard Deviations", min_value=1.0, max_value=4.0, value=2.0, step=0.1,
                #                             key=f"num_std_{strategy_id}")
                #     strategy_params[strategy_id] = {"window": window, "num_std_dev": num_std_dev}

    # Advanced options
    with st.sidebar.expander("Advanced Options"):
        initial_capital = st.number_input("Initial Capital", min_value=1000, max_value=1000000, value=100000,
                                          step=10000, key="initial_capital")
        commission = st.slider("Commission (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.05,
                               key="commission") / 100
        slippage = st.slider("Slippage (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.05, key="slippage") / 100
        risk_free_rate = st.slider("Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.5,
                                   key="risk_free_rate") / 100

        # Update config with advanced options
        advanced_config = {
            'backtest': {
                'initial_capital': initial_capital,
                'commission': commission,
                'slippage': slippage,
                'risk_free_rate': risk_free_rate
            }
        }


    # Create radio buttons in the sidebar
    test_option = st.sidebar.radio(
        "Select Run Mode",
        options=["Run with Sample Data", "Run with Latest Data"],
        index=0,  # default selected option
        key="run_mode_radio"
    )
    # You can now check which option was selected
    if test_option == "Run with Sample Data":
        use_sample_data = True
    else:
        use_sample_data = False

    # Action buttons with visual styling
    col1, col2 = st.sidebar.columns(2)
    with col1:
        run_button = st.button("🚀 Run Backtest", type="primary", key="run_backtest_button")
    with col2:
        load_existing = st.checkbox("📊 Load Results", value=True, key="load_existing_checkbox")

    # Main content
    st.markdown("<h1 class='main-header'>Stock Market Analysis & Prediction System</h1>", unsafe_allow_html=True)

    # Create main tabs
    main_tab1, main_tab2 = st.tabs(["📊 Backtest", "💬 AI Assistant"])

    with main_tab2:
        # AI Assistant
        st.markdown("<h2 class='sub-header'>AI Assistant</h2>", unsafe_allow_html=True)
        st.markdown("Ask questions about trading strategies, technical indicators, or market analysis.")

        # Initialize chatbot if not already in session state
        if 'standalone_chatbot' not in st.session_state:
            st.session_state.standalone_chatbot = ChatbotInterface()

        # Add backtest results to chatbot context if available
        if 'backtest_results' in st.session_state and st.session_state.backtest_results:
            st.session_state.standalone_chatbot.add_strategies_context(st.session_state.backtest_results['results'])

        # Display welcome message if conversation is empty
        if not hasattr(st.session_state.standalone_chatbot, 'conversation_history') or not st.session_state.standalone_chatbot.conversation_history:
            welcome_message = st.session_state.standalone_chatbot.get_welcome_message()
            st.markdown(welcome_message)

        # Display conversation history
        for message in st.session_state.standalone_chatbot.get_conversation_history():
            if message['is_user']:
                st.markdown(f"<div style='background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>You:</strong> {message['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='background-color: #e1f5fe; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>Assistant:</strong> {message['content']}</div>", unsafe_allow_html=True)

        # User input
        col1, col2 = st.columns([3, 1])
        with col1:
            user_query = st.text_input("Ask a question about trading strategies:", key="standalone_user_query")
        with col2:
            submit_button = st.button("Submit", key="standalone_submit_button")

        # Suggested questions
        st.markdown("### Suggested Questions")
        suggested_questions = st.session_state.standalone_chatbot.get_suggested_questions()

        # Create columns for suggested questions
        question_cols = st.columns(2)
        for i, question in enumerate(suggested_questions[:6]):  # Limit to 6 questions
            col_idx = i % 2
            with question_cols[col_idx]:
                if st.button(question, key=f"standalone_suggested_q_{i}"):
                    # Set the query but don't submit automatically
                    st.session_state.standalone_user_query = question
                    st.rerun()

        # Process user query when submit button is clicked
        if submit_button and user_query:
            with st.spinner("Generating response..."):
                # Store the query in session state to avoid infinite loop
                if 'last_standalone_query' not in st.session_state or st.session_state.last_standalone_query != user_query:
                    st.session_state.last_standalone_query = user_query
                    response = st.session_state.standalone_chatbot.get_response(user_query)
                    # Force a rerun to display the new messages
                    st.rerun()

        # Add buttons to clear conversation or context
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Conversation", key="standalone_clear_conversation"):
                st.session_state.standalone_chatbot.clear_conversation()
                st.rerun()
        with col2:
            if st.button("Reset Assistant", key="standalone_reset_assistant"):
                st.session_state.standalone_chatbot.clear_conversation()
                st.session_state.standalone_chatbot.clear_context()
                st.rerun()

    with main_tab1:
        # Dashboard Overview
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<div class='highlight'>", unsafe_allow_html=True)
            st.markdown("### Symbol")
            st.markdown(f"<h2 style='text-align: center;'>{symbol}</h2>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='highlight'>", unsafe_allow_html=True)
            st.markdown("### Period")
            st.markdown(f"<h2 style='text-align: center;'>{period}</h2>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            st.markdown("<div class='highlight'>", unsafe_allow_html=True)
            st.markdown("### Selected Strategies")
            st.markdown(f"<h2 style='text-align: center;'>{len(selected_strategies)}</h2>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with main_tab1:
        # Run backtest or load existing results
        if run_button:
            # Create a progress bar
            progress_bar = st.progress(0)

            # Status indicator
            status = st.empty()

            # Update status messages with animation effect
            for i, msg in enumerate(
                    ["Fetching market data...", "Processing indicators...", "Generating signals...", "Running backtests...",
                     "Calculating metrics..."]):
                status.markdown(f"<h3 style='color: #1E88E5;'>{msg}</h3>", unsafe_allow_html=True)
                progress_bar.progress((i + 1) / 5)
                time.sleep(0.5)  # Simulate processing time

            with st.spinner("Finalizing results..."):
                # Run backtest with parameters
                backtest_results = run_backtest(symbol, period, interval, selected_strategies,
                                                strategy_params=strategy_params, config=advanced_config,
                                                use_sample_data=use_sample_data)

                if backtest_results:
                    # Store backtest results in session state for access by standalone AI Assistant
                    st.session_state.backtest_results = backtest_results

                    progress_bar.progress(100)
                    status.success("✅ Backtest completed successfully!")

                    # Create tabs for results
                    tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance Comparison", "🔍 Detailed Metrics", "📊 Trading Signals", "💬 AI Assistant"])

                    with tab1:
                        # Display comparison chart
                        st.markdown("<h2 class='sub-header'>Strategy Comparison</h2>", unsafe_allow_html=True)
                        st.pyplot(backtest_results['comparison_fig'])

                        # Add interactive chart for cumulative returns
                        if 'results' in backtest_results:
                            # Prepare data for Altair chart
                            chart_data = pd.DataFrame()

                            for strategy in selected_strategies:
                                if strategy in backtest_results['results']:
                                    cum_returns = backtest_results['results'][strategy]['cum_returns']
                                    chart_data[strategies[strategy]] = cum_returns.values

                            chart_data['Date'] = cum_returns.index
                            chart_data = pd.melt(chart_data, id_vars=['Date'], var_name='Strategy', value_name='Return')

                            # Create interactive chart
                            import altair as alt

                            chart = alt.Chart(chart_data).mark_line().encode(
                                x='Date:T',
                                y=alt.Y('Return:Q', scale=alt.Scale(domain=[-0.3, 0.1])),
                                color='Strategy:N',
                                tooltip=['Date:T', 'Return:Q', 'Strategy:N']
                            ).properties(
                                width=800,
                                height=400,
                                title='Cumulative Returns Over Time (Interactive)'
                            ).interactive()

                            st.altair_chart(chart, use_container_width=True)

                            # Add LLM-based strategy comparison explanation
                            st.markdown("<h2 class='sub-header'>Strategy Analysis</h2>", unsafe_allow_html=True)

                            # Create a dictionary of strategy metrics for explanation
                            comparison_data = {}
                            strategy_names = {}

                            for strategy in selected_strategies:
                                if strategy in backtest_results['results']:
                                    comparison_data[strategy] = backtest_results['results'][strategy]['metrics']
                                    strategy_names[strategy] = strategies[strategy]

                            # Initialize strategy explainer
                            strategy_explainer = StrategyExplainer()
                            with st.spinner("Generating strategy analysis..."):
                                explanation = strategy_explainer.explain_strategy_comparison(comparison_data, strategy_names)
                                st.markdown(explanation)

                    with tab2:
                        # Display metrics in a more interactive way
                        st.markdown("<h2 class='sub-header'>Performance Metrics</h2>", unsafe_allow_html=True)

                        # Create a table of all metrics
                        metrics_df = pd.DataFrame()

                        for strategy in selected_strategies:
                            if strategy in backtest_results['results']:
                                result = backtest_results['results'][strategy]
                                metrics = result['metrics']
                                metrics_df[strategies[strategy]] = pd.Series(metrics)

                        # Format metrics table
                        formatted_df = metrics_df.copy()
                        for col in formatted_df.columns:
                            for idx in formatted_df.index:
                                if idx in ['total_return', 'annualized_return', 'maximum_drawdown', 'win_rate']:
                                    formatted_df.at[idx, col] = f"{formatted_df.at[idx, col] * 100:.2f}%"
                                elif idx in ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio']:
                                    formatted_df.at[idx, col] = f"{formatted_df.at[idx, col]:.4f}"

                        # Display metrics table
                        st.dataframe(formatted_df, use_container_width=True)

                        # Create metrics cards
                        cols = st.columns(len(selected_strategies))

                        for i, strategy in enumerate(selected_strategies):
                            with cols[i]:
                                if strategy in backtest_results['results']:
                                    result = backtest_results['results'][strategy]
                                    metrics = format_metrics(result['metrics'])

                                    st.markdown(f"<h3 style='text-align: center;'>{strategies[strategy]}</h3>",
                                                unsafe_allow_html=True)

                                    # Create metrics display
                                    for metric_name, metric_value in metrics.items():
                                        if metric_name in ['total_return', 'sharpe_ratio', 'maximum_drawdown', 'win_rate']:
                                            # Color code based on value
                                            color = "#4CAF50"  # Green for positive
                                            if metric_name in ['total_return', 'sharpe_ratio'] and float(
                                                    metric_value.strip('%')) < 0:
                                                color = "#F44336"  # Red for negative

                                            st.markdown(
                                                f"""
                                                <div class='metric-card'>
                                                    <div class='metric-value' style='color: {color};'>{metric_value}</div>
                                                    <div class='metric-label'>{metric_name.replace('_', ' ').title()}</div>
                                                </div>
                                                <br>
                                                """,
                                                unsafe_allow_html=True
                                            )

                    with tab3:
                        # Display trading signals
                        st.markdown("<h2 class='sub-header'>Trading Signals</h2>", unsafe_allow_html=True)

                        # Create a dropdown to select strategy
                        signal_strategy = st.selectbox(
                            "Select Strategy to View Signals",
                            options=[strategies[s] for s in selected_strategies],
                            index=0,
                            key="signal_strategy_select"
                        )

                        # Reverse lookup to get strategy_id
                        strategy_id = next((s for s in selected_strategies if strategies[s] == signal_strategy), None)

                        if strategy_id and strategy_id in backtest_results['results']:
                            result = backtest_results['results'][strategy_id]

                            # Get positions
                            positions = result['positions']

                            # Count signal types
                            buy_signals = (positions > 0).sum()
                            sell_signals = (positions < 0).sum()
                            no_signals = (positions == 0).sum()

                            # Display signal counts
                            signal_cols = st.columns(3)
                            with signal_cols[0]:
                                st.metric("Buy Signals", buy_signals, delta=None)
                            with signal_cols[1]:
                                st.metric("Sell Signals", sell_signals, delta=None)
                            with signal_cols[2]:
                                st.metric("No Position Days", no_signals, delta=None)

                            # Show signal table with filter
                            st.subheader("Signal Table")
                            signal_filter = st.multiselect(
                                "Filter Signals",
                                options=["Buy (1)", "Sell (-1)", "No Position (0)"],
                                default=["Buy (1)", "Sell (-1)"],
                                key=f"signal_filter_{strategy_id}"
                            )

                            filter_values = []
                            if "Buy (1)" in signal_filter:
                                filter_values.append(1)
                            if "Sell (-1)" in signal_filter:
                                filter_values.append(-1)
                            if "No Position (0)" in signal_filter:
                                filter_values.append(0)

                            # Filter positions
                            filtered_positions = positions[positions.isin(filter_values)]

                            # Enhance display
                            if not filtered_positions.empty:
                                # Add signal type column
                                display_positions = filtered_positions.copy()
                                display_positions['Signal Type'] = display_positions.map({
                                    1: "Buy", -1: "Sell", 0: "No Position"
                                })

                                # Rename columns for better display
                                # display_positions = display_positions.rename(
                                #     columns={'Index': "Signal Value"})

                                # Convert index to datetime if it's not already
                                # if not isinstance(display_positions.index, pd.DatetimeIndex):
                                #     display_positions.index = pd.to_datetime(display_positions.index)
                                #
                                # # Sort by date (most recent first)
                                # display_positions = display_positions.sort_index(ascending=False)

                                # Show table with pagination
                                st.dataframe(display_positions, use_container_width=True)

                                # Download button for signals
                                csv = display_positions.to_csv()
                                st.download_button(
                                    label="Download Signals CSV",
                                    data=csv,
                                    file_name=f"{strategy_id}_signals.csv",
                                    mime="text/csv",
                                )

                                # Add LLM-based signal explanation
                                st.subheader("Signal Analysis")

                                strategy_explainer = StrategyExplainer()
                                with st.spinner("Analyzing trading signals..."):
                                    # Get price data if available
                                    price_data = None
                                    if 'data' in result:
                                        price_data = result['data']

                                    explanation = strategy_explainer.explain_trading_signals(
                                        positions, signal_strategy, price_data)
                                    st.markdown(explanation)
                            else:
                                st.info("No signals match the selected filters.")

                    with tab4:
                        # AI Assistant / Chatbot Interface
                        st.markdown("<h2 class='sub-header'>Trading Strategy Assistant</h2>", unsafe_allow_html=True)

                        # Initialize chatbot if not already in session state
                        if 'chatbot' not in st.session_state:
                            st.session_state.chatbot = ChatbotInterface()

                        # Add strategy results to chatbot context
                        if 'results' in backtest_results:
                            st.session_state.chatbot.add_strategies_context(backtest_results['results'])

                        # Display welcome message if conversation is empty
                        if not st.session_state.chatbot.conversation_history:
                            welcome_message = st.session_state.chatbot.get_welcome_message()
                            st.markdown(welcome_message)

                        # Display conversation history
                        for message in st.session_state.chatbot.get_conversation_history():
                            if message['is_user']:
                                st.markdown(f"<div style='background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>You:</strong> {message['content']}</div>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<div style='background-color: #e1f5fe; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>Assistant:</strong> {message['content']}</div>", unsafe_allow_html=True)

                        # User input
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            user_query = st.text_input("Ask a question about your trading strategies:", key="user_query")
                        with col2:
                            submit_button = st.button("Submit", key="tab4_submit_button")

                        # Suggested questions
                        st.markdown("### Suggested Questions")
                        suggested_questions = st.session_state.chatbot.get_suggested_questions()

                        # Create columns for suggested questions
                        question_cols = st.columns(2)
                        for i, question in enumerate(suggested_questions[:6]):  # Limit to 6 questions
                            col_idx = i % 2
                            with question_cols[col_idx]:
                                if st.button(question, key=f"suggested_q_{i}"):
                                    # Set the query but don't submit automatically
                                    st.session_state.user_query = question
                                    st.rerun()

                        # Process user query when submit button is clicked
                        if submit_button and user_query:
                            with st.spinner("Generating response..."):
                                # Store the query in session state to avoid infinite loop
                                if 'last_query_tab4' not in st.session_state or st.session_state.last_query_tab4 != user_query:
                                    st.session_state.last_query_tab4 = user_query
                                    response = st.session_state.chatbot.get_response(user_query)
                                    # Force a rerun to display the new messages
                                    st.rerun()

                        # Add buttons to clear conversation or context
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Clear Conversation", key="clear_conversation"):
                                st.session_state.chatbot.clear_conversation()
                                st.rerun()
                        with col2:
                            if st.button("Reset Assistant", key="reset_assistant"):
                                st.session_state.chatbot.clear_conversation()
                                st.session_state.chatbot.clear_context()
                                st.rerun()

        elif load_existing and os.path.exists(os.path.join('results', 'backtest')):
            st.markdown("<h2 class='sub-header'>Existing Backtest Results</h2>", unsafe_allow_html=True)

            # Load comparison plot
            comparison_path = os.path.join('results', 'backtest', 'strategy_comparison.png')
            if os.path.exists(comparison_path):
                st.image(comparison_path, caption="Strategy Comparison", use_container_width=True)

            # Load individual strategy results
            st.markdown("<h2 class='sub-header'>Strategy Details</h2>", unsafe_allow_html=True)

            available_strategies = []
            for strategy in strategies.keys():
                if os.path.exists(os.path.join('results', 'backtest', f"{strategy}_metrics.json")):
                    available_strategies.append(strategy)

            # Create tabs for strategies and AI Assistant
            if available_strategies:
                # Add AI Assistant tab to the list of strategy tabs
                tab_names = [strategies[s] for s in available_strategies] + ["💬 AI Assistant"]
                tabs = st.tabs(tab_names)

                # Process strategy tabs
                for i, strategy in enumerate(available_strategies):
                    with tabs[i]:
                        results = load_results(strategy)

                        if results:
                            col1, col2 = st.columns([2, 1])

                            with col1:
                                st.image(results['plot_img'], caption=f"{strategies[strategy]} Performance",
                                         use_container_width=True)

                            with col2:
                                metrics = format_metrics(results['metrics'])

                                st.markdown("### Key Metrics")
                                for metric_name, metric_value in metrics.items():
                                    if metric_name in ['total_return', 'annualized_return', 'sharpe_ratio', 'sortino_ratio',
                                                       'maximum_drawdown', 'win_rate', 'profit_factor']:
                                        # Color code based on value
                                        color = "#4CAF50"  # Green for positive
                                        if metric_name in ['total_return', 'sharpe_ratio'] and metric_value.startswith('-'):
                                            color = "#F44336"  # Red for negative

                                        st.markdown(
                                            f"""
                                            <div style='margin-bottom: 10px;'>
                                                <span style='font-weight: bold;'>{metric_name.replace('_', ' ').title()}:</span> <span style='color: {color};'>{metric_value}</span>
                                            </div>
                                            """,
                                            unsafe_allow_html=True
                                        )

                                # Show signal counts
                                positions = results['positions']
                                buy_signals = (positions[positions.columns[0]] > 0).sum()
                                sell_signals = (positions[positions.columns[0]] < 0).sum()

                                st.markdown("### Trading Signals")
                                signal_cols = st.columns(2)
                                with signal_cols[0]:
                                    st.metric("Buy Signals", buy_signals, delta=None)
                                with signal_cols[1]:
                                    st.metric("Sell Signals", sell_signals, delta=None)

                            # Add expander for trading signals
                            with st.expander("View Trading Signals"):
                                # Add signal type column
                                display_positions = positions.copy()
                                display_positions['Signal Type'] = display_positions[display_positions.columns[0]].map({
                                    1: "Buy", -1: "Sell", 0: "No Position"
                                })

                                # Filter out no-position days
                                display_positions = display_positions[display_positions[display_positions.columns[0]] != 0]

                                # Rename columns for better display
                                display_positions = display_positions.rename(
                                    columns={display_positions.columns[0]: "Signal Value"})

                                # Show table with pagination
                                st.dataframe(display_positions, use_container_width=True)

                            # Add LLM-based signal explanation
                            with st.expander("Signal Analysis"):
                                strategy_explainer = StrategyExplainer()
                                with st.spinner("Analyzing trading signals..."):
                                    explanation = strategy_explainer.explain_trading_signals(
                                        positions, strategies[strategy])
                                    st.markdown(explanation)

                # AI Assistant tab (last tab)
                with tabs[-1]:
                    # AI Assistant / Chatbot Interface
                    st.markdown("<h2 class='sub-header'>Trading Strategy Assistant</h2>", unsafe_allow_html=True)

                    # Initialize chatbot if not already in session state
                    if 'chatbot' not in st.session_state:
                        st.session_state.chatbot = ChatbotInterface()

                    # Add strategy results to chatbot context
                    strategies_results = {}
                    for strategy in available_strategies:
                        results = load_results(strategy)
                        if results and 'metrics' in results:
                            strategies_results[strategy] = {'metrics': results['metrics']}

                    if strategies_results:
                        # Store strategies results in session state for access by standalone AI Assistant
                        st.session_state.backtest_results = {'results': strategies_results}

                        st.session_state.chatbot.add_strategies_context(strategies_results)

                    # Display welcome message if conversation is empty
                    if not st.session_state.chatbot.conversation_history:
                        welcome_message = st.session_state.chatbot.get_welcome_message()
                        st.markdown(welcome_message)

                    # Display conversation history
                    for message in st.session_state.chatbot.get_conversation_history():
                        if message['is_user']:
                            st.markdown(f"<div style='background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>You:</strong> {message['content']}</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div style='background-color: #e1f5fe; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>Assistant:</strong> {message['content']}</div>", unsafe_allow_html=True)

                    # User input
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        user_query = st.text_input("Ask a question about your trading strategies:", key="user_query_existing")
                    with col2:
                        submit_button = st.button("Submit", key="existing_submit_button")

                    # Suggested questions
                    st.markdown("### Suggested Questions")
                    suggested_questions = st.session_state.chatbot.get_suggested_questions()

                    # Create columns for suggested questions
                    question_cols = st.columns(2)
                    for i, question in enumerate(suggested_questions[:6]):  # Limit to 6 questions
                        col_idx = i % 2
                        with question_cols[col_idx]:
                            if st.button(question, key=f"suggested_q_existing_{i}"):
                                # Set the query but don't submit automatically
                                st.session_state.user_query_existing = question
                                st.rerun()

                    # Process user query when submit button is clicked
                    if submit_button and user_query:
                        with st.spinner("Generating response..."):
                            # Store the query in session state to avoid infinite loop
                            if 'last_query_existing' not in st.session_state or st.session_state.last_query_existing != user_query:
                                st.session_state.last_query_existing = user_query
                                response = st.session_state.chatbot.get_response(user_query)
                                # Force a rerun to display the new messages
                                st.rerun()

                    # Add buttons to clear conversation or context
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Clear Conversation", key="clear_conversation_existing"):
                            st.session_state.chatbot.clear_conversation()
                            st.rerun()
                    with col2:
                        if st.button("Reset Assistant", key="reset_assistant_existing"):
                            st.session_state.chatbot.clear_conversation()
                            st.session_state.chatbot.clear_context()
                            st.rerun()
            else:
                st.warning("No existing backtest results found. Please run a backtest first.")
        else:
            st.info("Configure your backtest parameters in the sidebar and click 'Run Backtest' to start.")


    # Instructions
    with st.expander("How to Use", expanded=False):
        st.markdown("""
        1. **Configure Parameters**: Set the symbol, time period, and data interval
        2. **Select Strategies**: Choose which trading strategies to evaluate
        3. **Customize Strategy Parameters**: Adjust parameters for each selected strategy
        4. **Set Advanced Options**: Configure capital, commission, and risk parameters
        5. **Run Backtest**: Click the Run Backtest button to execute
        6. **Analyze Results**: Compare performance across strategies using the interactive charts and tables
        7. **Review AI Analysis**: Read the AI-generated explanations of strategy performance and trading signals
        8. **Use the AI Assistant**: Ask questions about your strategies in the AI Assistant tab
        9. **Export Data**: Download metrics and signals for further analysis

        **AI Features**:
        - **Strategy Analysis**: Get detailed explanations of strategy performance comparisons
        - **Signal Analysis**: Understand the patterns and implications of trading signals
        - **AI Assistant**: Ask questions about your strategies and get personalized answers
        - **Suggested Questions**: Click on suggested questions to quickly get insights
        """)


if __name__ == "__main__":
    main()
