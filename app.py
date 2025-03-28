

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.express as px

# Custom UI Styling
st.markdown("""
    <style>
        body { background-color: #1e1e2f; color: white; }
        .stApp { background-color: #1e1e2f; }
        .sidebar .sidebar-content { background-color: #252532; color: white; }
        h1, h2, h3, h4, h5 { color: #ffffff; }
        .stButton>button { background-color: #1f77b4; color: white; border-radius: 5px; font-size: 18px; }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("flight_data_large_cleaned.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.dropna(subset=["Date"], inplace=True)
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    return df

df = load_data()

# Label Encoding
encoders = {}
for col in ["StartCountry", "DestinationCountry", "Weather", "FlightName", "Airline", "SeatClass"]:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col].astype(str))

# Sidebar Navigation
st.sidebar.title("🔹 Navigation")
menu = st.sidebar.radio("", ["🏠 Home", "🔍 Prediction", "📊 Analytics", "📈 Trends"])

# 🏠 Home Page
if menu == "🏠 Home":
    st.title("✈️ Flight Prediction & Analytics")
    st.subheader("📌 About the Project")
    st.write("""
        This platform provides advanced **flight cost predictions**, **weather forecasts**, and **seat availability** insights. 
        It also offers detailed analytics on airline pricing trends and passenger traffic.
    """)
    st.image("https://source.unsplash.com/800x400/?airplane,travel", use_column_width=True)
    
    st.markdown("### 📌 Features")
    st.markdown("- ✈️ **Flight Cost & Weather Prediction**")
    st.markdown("- 📊 **Real-time Analytics** on flight trends")
    st.markdown("- 📈 **Visualizations for Decision Making**")
    
    st.sidebar.success("Select a page above.")

# 🔍 Prediction Page
elif menu == "🔍 Prediction":
    st.title("🔍 Flight Prediction")
    st.sidebar.subheader("User Input")

    start_country_input = st.sidebar.text_input("Starting Country:")
    destination_country_input = st.sidebar.text_input("Destination Country:")
    airline_input = st.sidebar.text_input("Airline Name:")
    seat_class_input = st.sidebar.selectbox("Seat Class:", ["Economy", "Business", "First Class"])
    passengers = st.sidebar.number_input("Passengers:", min_value=1, max_value=500, value=1)
    travel_date = st.sidebar.date_input("Travel Date:")
    flight_name = st.sidebar.text_input("Flight Name:")

    if st.sidebar.button("Predict"):
        try:
            encoded_values = {
                col: encoders[col].transform([val])[0]
                for col, val in zip(["StartCountry", "DestinationCountry", "FlightName", "Airline", "SeatClass"],
                                    [start_country_input, destination_country_input, flight_name, airline_input, seat_class_input])
            }

            flight_data = df[
                (df["StartCountry"] == encoded_values["StartCountry"]) &
                (df["DestinationCountry"] == encoded_values["DestinationCountry"]) &
                (df["FlightName"] == encoded_values["FlightName"]) &
                (df["Airline"] == encoded_values["Airline"]) &
                (df["SeatClass"] == encoded_values["SeatClass"])
            ]

            if flight_data.empty:
                st.error("No data available for this route and airline.")
            else:
                cost_series = flight_data["Cost"].astype(float)
                predicted_cost = ARIMA(cost_series, order=(2, 1, 2)).fit().forecast(steps=1).iloc[0] if len(cost_series) > 3 else cost_series.mean()

                weather_series = flight_data["Weather"].astype(float)
                predicted_weather_index = int(round(SARIMAX(weather_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit().forecast(steps=1).iloc[0])) if len(weather_series) > 3 else 0
                predicted_weather = encoders["Weather"].inverse_transform([predicted_weather_index])[0] if predicted_weather_index in range(len(encoders["Weather"].classes_)) else "Unknown"

                available_seats = flight_data["AvailableSeats"].mean()

                st.subheader("🛫 Flight Prediction")
                st.write(f"### Predicted Cost: **${predicted_cost:.2f}**")
                st.write(f"### Predicted Weather: **{predicted_weather}**")
                st.write(f"### Available Seats: **{int(available_seats)}**")
                st.write(f"### Travel Date: **{travel_date}**")

        except Exception as e:
            st.error(f"Prediction error: {e}")

