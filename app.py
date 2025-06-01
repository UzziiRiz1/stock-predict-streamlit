import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import math
import datetime

# Load the trained model
model = load_model('stock_price_prediction_model.h5')

# Function to get stock data
def get_stock_data(ticker, start_date, end_date):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if stock_data.empty:
            st.error(f"Data for {ticker} could not be retrieved. Please check the ticker symbol.")
            return None
        return stock_data
    except Exception as e:
        st.error(f"An error occurred while fetching data for {ticker}: {str(e)}")
        return None

# Function to preprocess the data
def preprocess_data(stock_data):
    features = stock_data[['Open', 'High', 'Low', 'Volume', 'Close']]
    target = stock_data['Close']
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler.fit_transform(features)
    
    return features_scaled, target, scaler

def predict_stock_price(model, features_scaled):
    prediction = model.predict(features_scaled)
    return prediction

def plot_stock_data(stock_data):
    stock_data['MA30'] = stock_data['Close'].rolling(window=30).mean()
    stock_data['MA100'] = stock_data['Close'].rolling(window=100).mean()
    
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(stock_data['Close'], label='Close Price', color='blue')
    ax.plot(stock_data['MA30'], label='30 Day MA', color='red')
    ax.plot(stock_data['MA100'], label='100 Day MA', color='green')
    ax.set_title('Stock Closing Price with Moving Averages')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

# Function to rescale predictions
def inverse_transform_prediction(prediction, scaler, features_shape):
    dummy_features = np.zeros((prediction.shape[0], features_shape))
    dummy_features[:, -1] = prediction.flatten()
    predicted_rescaled = scaler.inverse_transform(dummy_features)
    return predicted_rescaled[:, -1]  # Return 1D array

# ---------------------- Streamlit Web App ---------------------- #

# Web app layout
st.title('📈 Stock Price Prediction Web App')

# Input field to enter a stock ticker symbol
ticker = st.text_input("Enter a company ticker symbol:", "AAPL")

# Input field for number of years to look back
years = st.number_input("Enter number of years you want record:", min_value=1, max_value=20, value=5, step=1)

# Calculate date range
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=years*365)

# Fetch stock data
with st.spinner('Fetching stock data....'):
    stock_data = get_stock_data(ticker, start_date, end_date)

if stock_data is not None:
    # Display raw data and basic info
    st.subheader('📄 Raw Stock Data')
    st.write(stock_data)

    # Perform data preprocessing
    features_scaled, target, scaler = preprocess_data(stock_data)

    # Show EDA visualizations
    st.subheader('📊 Stock Data Analysis')
    plot_stock_data(stock_data)

    # Predict stock price (using the most recent data)
    last_data = features_scaled[-1:].reshape(1, -1)
    prediction = predict_stock_price(model, last_data)

    # Rescale the predicted prices
    predicted_price = inverse_transform_prediction(prediction, scaler, 5)

    # Display the predicted price
    st.subheader('🔮 Predicted Closing Price for Next Day')
    st.success(f"The predicted closing price for {ticker} is: **${predicted_price[0]:.2f}**")

    # Get predicted prices for all data points
    predicted_prices = model.predict(features_scaled)

    # Rescale the predicted prices and ensure 1D array
    predicted_prices_rescaled = inverse_transform_prediction(predicted_prices, scaler, 5)
    predicted_prices_flat = np.squeeze(predicted_prices_rescaled)

    # Calculate and display error metrics
    mse = mean_squared_error(stock_data['Close'], predicted_prices_flat)
    rmse = math.sqrt(mse)
    r2 = r2_score(stock_data['Close'], predicted_prices_flat)

    st.subheader('🧮 Model Error Metrics')
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
    st.write(f"**R-squared (R²):** {r2:.2f}")

    # Add visualizations for model performance
    st.subheader('📈 Model Performance Visualization')
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(stock_data.index, stock_data['Close'], label='Real Price', color='blue')
    ax.plot(stock_data.index, predicted_prices_flat, label='Predicted Price', color='red')
    ax.set_title('Real vs Predicted Stock Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)
