# Chapter 3: Trading Strategies

## 3.1 Strategy Framework

The Nifty 50 Algorithmic Trading System implements a flexible strategy framework that allows for the development and testing of diverse trading approaches. At the core of this framework is the `BaseStrategy` abstract class, which defines the common interface and functionality that all trading strategies must implement.

### 3.1.1 Base Strategy Design

The `BaseStrategy` class, located in `src/strategies/base_strategy.py`, provides:

1. **Common Interface**: A standardized interface for strategy initialization, signal generation, and performance evaluation.
2. **Data Handling**: Methods for accessing and manipulating market data.
3. **Position Management**: Functionality for tracking and managing trading positions.
4. **Parameter Management**: Mechanisms for configuring strategy parameters.
5. **Performance Tracking**: Basic metrics calculation for strategy evaluation.

All specific strategy implementations inherit from this base class, ensuring consistency in how strategies interact with the rest of the system while allowing for unique trading logic.

## 3.2 Technical Analysis Strategies

The system implements several traditional technical analysis strategies that rely on mathematical calculations based on price and volume data.

### 3.2.1 Moving Average Crossover

**Theoretical Background**:
The Moving Average Crossover strategy is based on the interaction between short-term and long-term moving averages. When the short-term moving average crosses above the long-term moving average, it generates a buy signal, indicating an uptrend. Conversely, when the short-term moving average crosses below the long-term moving average, it generates a sell signal, indicating a downtrend.

**Implementation Details**:
- Located in `src/strategies/moving_average_crossover.py`
- Calculates simple moving averages (SMA) or exponential moving averages (EMA)
- Default parameters: short_window=20, long_window=50
- Generates binary signals (1 for buy, -1 for sell, 0 for hold)

**Performance Characteristics**:
- Tends to perform well in trending markets
- May generate false signals in choppy or sideways markets
- Typically has a moderate number of trades with medium holding periods

### 3.2.2 Relative Strength Index (RSI)

**Theoretical Background**:
The RSI strategy is based on the Relative Strength Index, a momentum oscillator that measures the speed and change of price movements. RSI oscillates between 0 and 100, with values above 70 typically considered overbought and values below 30 considered oversold. The strategy generates buy signals when RSI moves from below 30 to above 30, and sell signals when RSI moves from above 70 to below 70.

**Implementation Details**:
- Located in `src/strategies/rsi_strategy.py`
- Calculates RSI using a default period of 14 days
- Configurable overbought (default: 70) and oversold (default: 30) thresholds
- Includes optional signal smoothing to reduce false signals

**Performance Characteristics**:
- Often effective in range-bound markets
- May miss extended trends
- Generally has a higher frequency of trades with shorter holding periods

### 3.2.3 Moving Average Convergence Divergence (MACD)

**Theoretical Background**:
The MACD strategy uses the Moving Average Convergence Divergence indicator, which is calculated by subtracting the 26-period EMA from the 12-period EMA. A 9-period EMA of the MACD, called the "signal line," is then plotted on top of the MACD. Buy signals are generated when the MACD crosses above the signal line, and sell signals when it crosses below.

**Implementation Details**:
- Located in `src/strategies/macd_strategy.py`
- Uses standard parameters: fast_period=12, slow_period=26, signal_period=9
- Includes histogram calculation (MACD - signal line) for signal strength assessment
- Provides options for additional filters based on MACD histogram values

**Performance Characteristics**:
- Combines trend-following and momentum characteristics
- Moderate number of trades with varying holding periods
- Often used as a confirmation indicator alongside other strategies

### 3.2.4 Bollinger Bands

**Theoretical Background**:
Bollinger Bands consist of a middle band (typically a 20-period moving average) and two outer bands placed at a standard deviation distance from the middle band. The strategy generates buy signals when the price touches or crosses below the lower band and then moves back inside, indicating a potential reversal from oversold conditions. Similarly, sell signals are generated when the price touches or crosses above the upper band and then moves back inside.

**Implementation Details**:
- Located in `src/strategies/bollinger_bands.py`
- Default parameters: window=20, num_std_dev=2
- Includes band width calculation for volatility assessment
- Provides options for "squeeze" detection (when bands narrow significantly)

**Performance Characteristics**:
- Adapts to changing market volatility
- Effective in identifying potential price reversals
- Generally has fewer trades with potentially higher accuracy

## 3.3 Advanced Strategies

Beyond traditional technical analysis, the system implements more sophisticated strategies that leverage machine learning and deep learning techniques.

### 3.3.1 Machine Learning Strategy

**Theoretical Background**:
The Machine Learning strategy uses supervised learning algorithms to predict price movements based on historical data. The strategy treats the prediction problem as a classification task (predicting whether the price will go up or down) or a regression task (predicting the actual price change).

**Implementation Details**:
- Located in `src/strategies/ml_strategy.py`
- Uses scikit-learn for model implementation
- Supports multiple algorithms: Random Forest, Support Vector Machines, Gradient Boosting
- Features include technical indicators, price patterns, and market statistics
- Implements cross-validation to prevent overfitting

**Performance Characteristics**:
- Potentially captures complex, non-linear relationships in data
- Performance heavily dependent on feature selection and model tuning
- Requires substantial historical data for effective training

### 3.3.2 LSTM Deep Learning Strategy

**Theoretical Background**:
The Long Short-Term Memory (LSTM) strategy uses a specialized type of recurrent neural network designed to recognize patterns in sequence data. LSTM networks are particularly well-suited for time series prediction tasks like stock market forecasting because they can capture long-term dependencies and remember information for extended periods.

**Implementation Details**:
- Located in `src/strategies/lstm_strategy.py`
- Implemented using TensorFlow/Keras
- Network architecture: Input layer, 1-2 LSTM layers, dropout layers, dense output layer
- Uses sequence data with configurable lookback period
- Implements early stopping and learning rate reduction to prevent overfitting

**Performance Characteristics**:
- Potentially captures complex temporal patterns in market data
- Requires significant computational resources for training
- Performance can be sensitive to hyperparameter selection
- Generally requires more historical data than traditional strategies

## 3.4 Strategy Comparison Framework

To facilitate objective comparison between different trading strategies, the system includes a strategy comparison framework that:

1. **Standardizes Inputs**: Ensures all strategies receive identical market data for fair comparison.
2. **Normalizes Parameters**: Adjusts strategy parameters to ensure comparable risk levels.
3. **Calculates Metrics**: Computes a comprehensive set of performance metrics for each strategy.
4. **Visualizes Results**: Generates comparative charts and tables to highlight differences in performance.

This framework allows users to identify which strategies perform best under specific market conditions and helps inform the selection of appropriate strategies for different trading objectives.

## 3.5 Strategy Implementation Progress

As of the mid-semester point, all six planned trading strategies have been implemented and integrated into the system:

1. Moving Average Crossover: Complete with parameter optimization
2. RSI Strategy: Complete with signal filtering options
3. MACD Strategy: Complete with histogram analysis
4. Bollinger Bands: Complete with squeeze detection
5. Machine Learning Strategy: Basic implementation complete, feature engineering ongoing
6. LSTM Deep Learning Strategy: Basic implementation complete, hyperparameter tuning ongoing

The focus for the remainder of the project will be on refining the machine learning and deep learning strategies, which have shown promising initial results but require further optimization to maximize their predictive power.