import yfinance as yf
import pandas as pd
import joblib


model = joblib.load('day_trading_model.pkl')

features = ['Close', 'Volume', 'SMA_10', 'RSI', 'Hour']

def calculate_rsi(data, period=14):
    """
    Calculate the Relative Strength Index (RSI) for a given DataFrame.
    """
    delta = data[('Close', 'PETR4.SA')].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


ticker = "PETR4.SA"
interval = "5m"
real_time_data = yf.download(ticker, period="1d", interval=interval)



real_time_data[('SMA_10', 'PETR4.SA')] = real_time_data[('Close', 'PETR4.SA')].rolling(window=10).mean()
real_time_data['RSI'] = calculate_rsi(real_time_data, period=14)
real_time_data['MACD'] = real_time_data[('Close', 'PETR4.SA')].ewm(span=12, adjust=False).mean() - real_time_data[('Close', 'PETR4.SA')].ewm(span=26, adjust=False).mean()
real_time_data['Hour'] = real_time_data.index.hour


latest_data = real_time_data.iloc[[-1]][features]  


predicted_change = model.predict(latest_data)[0]
print("Predicted Price Change:", predicted_change)