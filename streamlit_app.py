import streamlit as st
import pandas as pd
import requests

st.title('Sales Prediction Dashboard')

# Buttons for user selection
year = st.selectbox("Select Year", [2025, 2026])

# Fetch data from API when button is pressed
if st.button('Get Predictions'):
    response = requests.get(f"http://127.0.0.1:8000/predictions/{year}/")
    if response.status_code == 200:
        data = pd.DataFrame(response.json())
        data['ds'] = pd.to_datetime(data['ds'])
        data['month_year'] = data['ds'].dt.strftime('%Y-%m')

        st.subheader(f"Predicted Sales for {year} (Bar Chart)")
        st.bar_chart(data.set_index('month_year')['predicted_sales'])

        st.subheader(f"Line Chart of Predicted Sales for {year}")
        st.line_chart(data.set_index('month_year')['predicted_sales'])
    else:
        st.error("Failed to fetch data")
