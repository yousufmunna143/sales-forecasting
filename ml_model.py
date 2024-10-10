import pandas as pd
from sklearn.model_selection import train_test_split
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load sales data
data = pd.read_csv('sales_data.csv')

# Extract relevant columns (Weekly Sales Data)
weekly_sales_data = data[[f'W{i}' for i in range(52)]]

# Split data into training and testing sets (80% training, 20% testing)
train_data, test_data = train_test_split(weekly_sales_data, test_size=0.2, random_state=42)

# Prepare data for Prophet
def prepare_prophet_data(data):
    dates = pd.date_range(start='1/1/2024', periods=52, freq='W')
    df = pd.DataFrame({
        'ds': dates,
        'y': data.mean(axis=0).values
    })
    return df

# Prepare training data
train_df = prepare_prophet_data(train_data)

# Define Prophet model function
def prophet_model(train_df):
    model = Prophet()
    model.fit(train_df)
    return model

# Train the model
prophet_model_fit = prophet_model(train_df)

# Create future dataframe for prediction for 2025 (12 months)
future_dates_2025 = pd.DataFrame({'ds': pd.date_range(start='1/1/2025', periods=12, freq='M')})

# Predict future values for 2025
prophet_forecast_2025 = prophet_model_fit.predict(future_dates_2025)

# Extract predictions for 2025
prophet_predictions_2025 = prophet_forecast_2025[['ds', 'yhat']]

# Create future dataframe for prediction for 2026 (12 months)
future_dates_2026 = pd.DataFrame({'ds': pd.date_range(start='1/1/2026', periods=12, freq='M')})

# Predict future values for 2026
prophet_forecast_2026 = prophet_model_fit.predict(future_dates_2026)

# Extract predictions for 2026
prophet_predictions_2026 = prophet_forecast_2026[['ds', 'yhat']]

# Prepare test data for evaluation
evaluation_dates = pd.date_range(start='1/1/2021', periods=12, freq='M')
test_forecast = prophet_model_fit.predict(pd.DataFrame({'ds': evaluation_dates}))
actual_values = test_forecast['yhat'].values
predicted_values = prophet_forecast_2025['yhat'].values[:len(actual_values)]

# Evaluate metrics
mae = mean_absolute_error(actual_values, predicted_values)
mse = mean_squared_error(actual_values, predicted_values)
rmse = np.sqrt(mse)
r2 = r2_score(actual_values, predicted_values)
mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100

# Print results
print("Evaluation Metrics for Prophet Model:")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")
print(f"MAPE: {mape}%")

# Save predictions to CSV
prophet_predictions_2025.to_csv('prophet_predictions_2025.csv', index=False)
prophet_predictions_2026.to_csv('prophet_predictions_2026.csv', index=False)