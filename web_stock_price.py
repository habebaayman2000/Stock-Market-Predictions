import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
import plotly.graph_objects as go

st.title('Stock Price Prediction')
st.subheader('Predict future stock prices using LSTM')
st.write('This app uses a Long Short-Term Memory (LSTM) model to predict stock prices based on historical data.')
st.write('You can enter the stock ticker symbol and the number of days to predict.')

stock=st.text_input("Enter stock ID (e.g., AAPL, MSFT):","GOOG")

end=datetime.datetime.now()
start=datetime.datetime(end.year-20,end.month,end.day)

google_data=yf.download(stock,start,end)

model=load_model('Latest_stock_price.keras')

st.subheader('Stock Price Data')
st.write(google_data)

splitting_len=int(len(google_data)*0.7)
x_test=pd.DataFrame(google_data['Close'][splitting_len:])

def plot_graph(figsize,values,data, extra_data=0,extra_dataset=None):
    fig= plt.figure(figsize=figsize)
    plt.plot(values,'Orange')
    plt.plot(data.Close,'b')
    if extra_data:
        plt.plot(extra_dataset,"r")
    plt.legend()    
    return fig



st.subheader('Original Close Price and MA for 250 days')
google_data['MA250']=google_data['Close'].rolling(250).mean()
st.pyplot(plot_graph((15,6),google_data['MA250'],google_data,0,))


st.subheader('Original Close Price and MA for 50 days')
google_data['MA50']=google_data['Close'].rolling(50).mean()
st.pyplot(plot_graph((15,6),google_data['MA50'],google_data,0))


st.subheader('Original Close Price and MA for 100 days')
google_data['MA100']=google_data['Close'].rolling(100).mean()
st.pyplot(plot_graph((15,6),google_data['MA100'],google_data,0))


st.subheader('Original Close Price and MA for 100 days and MA for 50 days')
#google_data['MA100']=google_data['Close'].rolling(100).mean()
st.pyplot(plot_graph((15,6),google_data['MA100'],google_data,1,google_data['MA50']))

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(x_test.values.reshape(-1,1))

x_data=[]
y_data=[]

for i in range(100,len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])


x_data,y_data=np.array(x_data),np.array(y_data)

predicted_data=model.predict(x_data)

inv_pre=scaler.inverse_transform(predicted_data)

inv_y=scaler.inverse_transform(y_data)

plotting_data=pd.DataFrame(
    {
        'Actual':inv_y.reshape(-1),
        'Predicted':inv_pre.reshape(-1)
    },index= google_data.index[splitting_len+100:]
)
st.subheader('Predicted vs Actual Stock Price')
st.write(plotting_data)

st.subheader('Original Close Price vs Predicted Close Price')
fig= plt.figure(figsize=(15,6))
plt.plot(pd.concat([google_data.Close[:splitting_len+100],plotting_data],axis=0))
plt.legend(['Data- not used','Orignal Test data',"Predicted Test data"])
st.pyplot(fig)

st.subheader('Candlestick Chart')

google_data.to_csv("AAPL.csv")
data01=pd.read_csv("AAPL.csv")
fig = go.Figure(data=[go.Candlestick(x=data01.index,
                                     open=data01['Open'],
                                     high=data01['High'],
                                     low=data01['Low'],
                                     close=data01['Close'])])
fig.update_layout(xaxis_rangeslider_visible=False, title="Candlestick Chart", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig)


# Use last 60 days to predict next 50
last_60 = scaled_data[-60:]
input_seq = last_60.reshape(1, 60, 1)

# Predict 50 future values step-by-step
predictions = []
current_input = input_seq.copy()

for _ in range(50):
    next_pred = model.predict(current_input)[0][0]
    predictions.append(next_pred)
    
    current_input = np.append(current_input[:, 1:, :], [[[next_pred]]], axis=1)

# Inverse scale the predictions
predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Simple investment logic
if predicted_prices[-1][0] > google_data['Close'].iloc[-1].item()  * 1.05:
    decision = "Buy ✅"
else:
    decision = "Don't Buy ❌"

# Ensure both are 1D
historical = google_data['Close'].values.flatten()
forecast = predicted_prices.reshape(-1)

# Create full series with correct length
historical_series = pd.Series(historical, name='Historical')
forecast_series = pd.Series(forecast, index=range(len(historical), len(historical) + len(forecast)), name='Forecast')

st.subheader('Historical vs Forecasted Stock Prices')
# Combine into a DataFrame for charting
chart_df = pd.DataFrame({
    'Historical': historical_series,
    'Forecast': forecast_series
})
# Plot historical and forecasted prices with a title



st.line_chart(chart_df)
st.subheader('Investment Decision')
st.write(f"Based on the prediction, you should: {decision}")


