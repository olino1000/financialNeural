import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib


ticker = "PETR4.SA"
start_date = "2025-01-11"
end_date = "2025-03-11"
interval = "5m"
data = yf.download(ticker, start=start_date, end=end_date, interval=interval)


data.index = data.index.tz_convert('America/Sao_Paulo')
data = data.between_time('10:00', '21:00')

data.dropna(subset=[('Close', 'PETR4.SA')], inplace=True)

data[('SMA_10', 'PETR4.SA')] = data[('Close', 'PETR4.SA')].rolling(window=10).mean()

def calculate_rsi(data, period=14):
    delta = data[('Close', 'PETR4.SA')].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['RSI'] = calculate_rsi(data, period=14)

data['Hour'] = data.index.hour
data['Minute'] = data.index.minute
data['DayOfWeek'] = data.index.dayofweek


data['Target'] = data[('Close', 'PETR4.SA')].shift(-1) - data[('Close', 'PETR4.SA')]
data.dropna(inplace=True)

features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'RSI', 'Hour', 'Minute', 'DayOfWeek']
X = data[features] 
y = data['Target']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

feature_importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)


top_features = feature_importance_df['Feature'].head(5).tolist()

X_train_selected = X_train[top_features]
X_test_selected = X_test[top_features]

model_selected = RandomForestRegressor(n_estimators=100, random_state=42)
model_selected.fit(X_train_selected, y_train)

y_pred_selected = model_selected.predict(X_test_selected)
mae_selected = mean_absolute_error(y_test, y_pred_selected)
rmse_selected = mean_squared_error(y_test, y_pred_selected) ** 0.5

print("MAE:", mae_selected)
print("RMSE:", rmse_selected)

joblib.dump(model_selected, 'day_trading_model.pkl')


