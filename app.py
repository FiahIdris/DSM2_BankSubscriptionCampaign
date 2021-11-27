import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.header('FTDS Model Deployment')
st.write("""
Created by FTDS Curriculum Team

Use the sidebar to select input features.
""")


@st.cache
def fetch_data():
    df = pd.read_csv(
        '/Users/nurfiahidris/Desktop/DS-Hacktiv8/Hacktiv8-Assignment/P1/DSM2_BankSubscriptionCampaign/bank-additional-full.csv', delimiter=";")
    return df


# df = fetch_data()
# st.write(df)

st.sidebar.header('User Input Features')


# def user_input():
#     longitude = st.sidebar.number_input('Longitude', value=-121.89, )
#     latitude = st.sidebar.number_input('Latitude', value=37.29)
#     housing_median_age = st.sidebar.number_input(
#         'Housing Median Age', 0.0, value=38.0)
#     total_rooms = st.sidebar.number_input('Total Rooms', 0.0, value=1568.0)
#     total_bedrooms = st.sidebar.number_input(
#         'Total Bedrooms', 0.0, value=351.0)
#     population = st.sidebar.number_input('Population', 0.0, value=710.0)
#     households = st.sidebar.number_input('Households', 0.0, value=339.0)
#     median_income = st.sidebar.number_input('Median Income', 0.0, value=2.7042)
#     ocean_proximity = st.sidebar.selectbox(
#         'Ocean Proximity', df['ocean_proximity'].unique())
#
#     data = {
#         'longitude': longitude,
#         'latitude': latitude,
#         'housing_median_age': housing_median_age,
#         'total_rooms': total_rooms,
#         'total_bedrooms': total_bedrooms,
#         'population': population,
#         'households': households,
#         'median_income': median_income,
#         'ocean_proximity': ocean_proximity
#     }
#     features = pd.DataFrame(data, index=[0])
#     return features


# input = user_input()
#
# st.subheader('User Input')
# st.write(input)
#
# load_model = joblib.load("my_model.pkl")
# predicton = load_model.predict(input)
#
# st.write('Based on user input, the housing model predicted: ')
# st.write(predicton)
#
# st.subheader('Exploratory Data Analysis')
#
# scatterDF = px.scatter(df, x="longitude", y="latitude")
# st.plotly_chart(scatterDF)
# st.write("""
# This is a Placeholder
# For insights or analysis
# """)
#
# medianIncome = px.scatter(df, x="median_income", y="median_house_value")
# st.plotly_chart(medianIncome)
# st.write("""
# This is a Placeholder
# For insights or analysis
# """)
