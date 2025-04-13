# 📈 Stock Price Prediction App

This is a **Streamlit web application** that uses a pre-trained **LSTM (Long Short-Term Memory)** deep learning model to predict future stock prices based on historical data. It supports fetching real-time stock data, visualization, and investment suggestions.

---

## 🚀 Features

- 📊 Visualize historical stock prices and moving averages (MA50, MA100, MA250)
- 🧠 Predict stock prices using an LSTM model
- 📈 Display predicted vs. actual stock prices
- 📉 Interactive candlestick chart using Plotly
- 💡 Investment decision suggestion (Buy / Don't Buy)
- ⏩ Forecast the next 50 days of stock prices

---

## 📂 Project Structure

├── Latest_stock_price.keras # Pre-trained LSTM model ├── app.py # Main Streamlit app script ├── AAPL.csv # Example CSV output (auto-generated) └── README.md # This file


---

## 🛠️ Requirements

Install dependencies with:

```bash
pip install -r requirements.txt

Make sure you are using Python 3.8 to 3.11 for best compatibility with Keras and TensorFlow.


🧪 How It Works
The user enters a stock ticker (e.g., AAPL, GOOG, MSFT).

Historical data for the last 20 years is downloaded using yfinance.

The model predicts stock prices based on the last 100 days of data.

Visualization includes:

Original prices + moving averages

Predicted vs actual prices

Candlestick chart

Next 50-day forecast

A simple logic decides whether it's a good time to buy the stock.

**Running the App
bash
Copy
Edit
streamlit run app.py
Ensure Latest_stock_price.keras (your trained LSTM model) is in the same directory as app.py.**


Let me know if you'd like me to generate the `requirements.txt` or help you tweak the README for a GitHub repo!
