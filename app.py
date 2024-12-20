import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import time
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

# Load the saved model
model = joblib.load('stock_trading_model.pkl')  # Ensure you have your trained model saved here

# Function to fetch live stock data for a single ticker
def get_live_data(symbol):
    # Fetch 1-minute data for real-time stock price updates
    data = yf.download(symbol, period="1d", interval="1m")
    data.reset_index(inplace=True)
    data = data[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
    data.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    return data

# Feature engineering function for model prediction
def feature_engineering(df):
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['RSI'] = 100 - (100 / (1 + df['close'].pct_change().rolling(window=14).mean() / df['close'].pct_change().rolling(window=14).std()))
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    df = df.dropna()
    return df

# Model prediction function to decide Buy/Sell/Hold
def execute_trade(model, df):
    last_data = df.iloc[-1]
    X_latest = last_data[['SMA_20', 'SMA_50', 'RSI']].values.reshape(1, -1)
    action = model.predict(X_latest)[0]
    
    if action == 1:
        return 'Buy'
    elif action == 0:
        return 'Sell'
    return 'Hold'

# Function to update the Streamlit app with real-time data, predictions, and ticker table
def real_time_trading():
    st.title("Real-Time Stock Trading Bot")

    # User input for a single stock symbol
    symbol = st.text_input("Enter Stock Symbol", "AAPL").upper()

    # Frequency of updates (seconds)
    update_frequency = st.slider("Update Frequency (seconds)", 30, 300, 60)  # 30s to 5min interval
    
    # Placeholder for real-time chart updates
    chart_placeholder = st.empty()
    
    # Display real-time trading for the selected stock ticker
    while True:
        try:
            # Fetch live data for the stock symbol
            data = get_live_data(symbol)
            data = feature_engineering(data)  # Perform feature engineering

            # Make a trading decision (Buy/Sell/Hold)
            action = execute_trade(model, data)
            
            # Show the latest decision
            st.write(f"Latest decision for {symbol} at {data['timestamp'].iloc[-1]}: {action}")
            
            # Show the table with key details about the ticker
            ticker_info = pd.DataFrame({
                'Stock Symbol': [symbol],
                'Latest Price': [data['close'].iloc[-1]],
                'SMA_20': [data['SMA_20'].iloc[-1]],
                'SMA_50': [data['SMA_50'].iloc[-1]],
                'RSI': [data['RSI'].iloc[-1]],
                'Action': [action]
            })
            st.table(ticker_info)

            # Plot interactive chart using Plotly
            fig = go.Figure()

            # Add Close Price trace
            fig.add_trace(go.Scatter(x=data['timestamp'], y=data['close'], mode='lines', name=f'{symbol} Close Price', line=dict(color='blue')))
            
            # Add Buy markers
            buy_dates = data[data['target'] == 1]
            fig.add_trace(go.Scatter(x=buy_dates['timestamp'], y=buy_dates['close'], mode='markers', name='Buy', marker=dict(symbol='triangle-up', color='green', size=10)))
            
            # Add Sell markers
            sell_dates = data[data['target'] == 0]
            fig.add_trace(go.Scatter(x=sell_dates['timestamp'], y=sell_dates['close'], mode='markers', name='Sell', marker=dict(symbol='triangle-down', color='red', size=10)))

            # Update layout for interactivity
            fig.update_layout(
                title=f"Real-Time Stock Price for {symbol}",
                xaxis_title='Timestamp',
                yaxis_title='Price',
                template="plotly_dark",
                hovermode='closest'
            )

            # Display the interactive chart
            chart_placeholder.plotly_chart(fig)
        
        except Exception as e:
            st.write(f"Error fetching data for {symbol}: {e}")
        
        # Wait for the specified interval before fetching new data
        time.sleep(update_frequency)  # Wait before next update

# Run the real-time trading function
if __name__ == "__main__":
    real_time_trading()
