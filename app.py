# import streamlit as st
# from prophet import Prophet
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # Load your dataset
# df = pd.read_csv('USD_INR Historical Data.csv')
# df.rename(columns={'Date': 'ds', 'Price': 'y'}, inplace=True)
#
# # Initialize the model
# model = Prophet(
#     growth='linear',
#     changepoint_prior_scale=0.5,  # flexibility of trend change points
#     seasonality_mode='additive',  # type of seasonality
#     yearly_seasonality=True,
#     weekly_seasonality=False,
#     daily_seasonality=False
# )
#
# # Fit the model
# model.fit(df)
#
# # Streamlit app
# st.title('Price Prediction App')
#
# # Input fields for day, month, and year
# day = st.number_input('Enter day', min_value=1, max_value=31, step=1)
# month = st.number_input('Enter month', min_value=1, max_value=12, step=1)
# year = st.number_input('Enter year', min_value=2000, max_value=2100, step=1)
#
# if st.button('Predict Price'):
#     # Create the specific date
#     specific_date = f"{year}-{month:02d}-{day:02d}"
#
#     # Create a dataframe for the specific date
#     future_dates = pd.DataFrame({'ds': [specific_date]})
#
#     # Predict future prices
#     forecast = model.predict(future_dates)
#
#     # Extract the predicted value
#     predicted_price = forecast['yhat'].iloc[0]
#
#     # Convert to integer
#     predicted_price_int = int(round(predicted_price))
#
#     st.write(f"The predicted price on {specific_date} is {predicted_price_int}")
#
#     # Plot the forecast
#     fig1 = model.plot(forecast)
#     st.pyplot(fig1)
#
#     # Plot the components (trend, weekly, yearly seasonality)
#     fig2 = model.plot_components(forecast)
#     st.pyplot(fig2)

import streamlit as st
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('USD_INR Historical Data.csv')
df.rename(columns={'Date': 'ds', 'Price': 'y'}, inplace=True)

# Initialize the model
model = Prophet(
    growth='linear',
    changepoint_prior_scale=0.5,  # flexibility of trend change points
    seasonality_mode='additive',  # type of seasonality
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False
)

# Fit the model
model.fit(df)

# Streamlit app
st.title('Price Prediction App')

# Input fields for day, month, and year
day = st.number_input('Enter day', min_value=1, max_value=31, step=1)
month = st.number_input('Enter month', min_value=1, max_value=12, step=1)
year = st.number_input('Enter year', min_value=2000, max_value=2100, step=1)

if st.button('Predict Price'):
    # Create the specific date
    specific_date = f"{year}-{month:02d}-{day:02d}"

    # Create a dataframe for the specific date
    future_dates = pd.DataFrame({'ds': [specific_date]})

    # Predict future prices
    forecast = model.predict(future_dates)

    # Extract the predicted value
    predicted_price = forecast['yhat'].iloc[0]

    # Convert to integer
    predicted_price_int = int(round(predicted_price))

    st.write(f"The predicted price on {specific_date} is {predicted_price_int}")

    # Plot the forecast
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    # Plot the components (trend, weekly, yearly seasonality)
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

