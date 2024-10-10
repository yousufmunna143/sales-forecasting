import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load sales data
data = pd.read_csv('Assignment-3-ML-Sales_Transactions_Dataset_Weekly.csv')

# Extract relevant columns (Weekly Sales Data)
weekly_sales_data = data[[f'W{i}' for i in range(52)]]

# Split data into training and testing sets (80% training, 20% testing)
train_data, test_data = train_test_split(weekly_sales_data, test_size=0.2, random_state=42)

# Define SARIMA model function
def sarima_model(train_data):
    model = SARIMAX(train_data.mean(axis=0), order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
    model_fit = model.fit(disp=False)
    return model_fit

# Train and predict with SARIMA
sarima_model_fit = sarima_model(train_data)
sarima_predictions = sarima_model_fit.forecast(steps=52)

# Prepare data for Prophet
def prepare_prophet_data(data):
    dates = pd.date_range(start='1/1/2020', periods=52, freq='W')
    df = pd.DataFrame({
        'ds': dates,
        'y': data.mean(axis=0).values
    })
    return df

# Define Prophet model function
def prophet_model(train_data):
    model = Prophet()
    df = prepare_prophet_data(train_data)
    model.fit(df)
    return model

# Train and predict with Prophet
prophet_model_fit = prophet_model(train_data)
future_dates = pd.DataFrame({'ds': pd.date_range(start='1/1/2021', periods=52, freq='W')})
prophet_forecast = prophet_model_fit.predict(future_dates)
prophet_predictions = prophet_forecast['yhat'].values

# Prepare data for LSTM model
train_values = train_data.mean(axis=0).values.reshape(-1, 1)
test_values = test_data.mean(axis=0).values.reshape(-1, 1)

# Define LSTM model function
def lstm_model(train_values):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(50, input_shape=(train_values.shape[1], 1)))
    model.add(tf.keras.layers.Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(train_values, train_values, epochs=50, batch_size=32, verbose=0)
    return model

# Train and predict with LSTM
lstm_model_fit = lstm_model(train_values)
lstm_predictions = lstm_model_fit.predict(test_values)

# Collect all models' predictions
models = {'SARIMA': sarima_predictions, 'Prophet': prophet_predictions, 'LSTM': lstm_predictions}
results = []

# Evaluate metrics
for model_name, predictions in models.items():
    actual = test_data.mean(axis=0).values
    mae = mean_absolute_error(actual, predictions)
    mse = mean_squared_error(actual, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predictions)
    mape = np.mean(np.abs((actual - predictions) / actual)) * 100
    
    results.append({'Model': model_name, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R²': r2, 'MAPE': mape})

results_df = pd.DataFrame(results)
print(results_df)