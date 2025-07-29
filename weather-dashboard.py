import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page Title
st.set_page_config(page_title="Weather Data Analysis Dashboard", layout="wide")

# Upload CSV
uploaded_file = st.file_uploader("Upload your weather CSV file", type=["csv"])

if uploaded_file is not None:
    # Load data
    data = pd.read_csv(uploaded_file)
    
    # Parse dates
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date')

    # Clean column names
    data = data.rename(columns={
        "Temperature (°C)": "Temperature",
        "Humidity (%)": "Humidity",
        "Wind Speed (km/h)": "Wind Speed",
        "Rainfall (mm)": "Rainfall"
    })

    # All Records for weather
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Records", len(data))
        st.metric("Average Temperature (°C)", f"{data['Temperature'].mean():.2f}")
        st.metric("Average Humidity (%)", f"{data['Humidity'].mean():.2f}")
        
    with col2:
        st.metric("Average Wind Speed (km/h)", f"{data['Wind Speed'].mean():.2f}")
        st.metric("Max Wind Speed (km/h)", f"{data['Wind Speed'].max():.2f}")
        st.metric("Total Rainfall (mm)", f"{data['Rainfall'].sum():.2f}")

    # Hottest and Coolest Days
    with st.expander("Top 5 Hottest & Coolest Days"):
        hottest_days = data.nlargest(5, 'Temperature')[['Date', 'Temperature']]
        coolest_days = data.nsmallest(5, 'Temperature')[['Date', 'Temperature']]
        col_hot, col_cool = st.columns(2)
        with col_hot:
            st.write("**Top 5 Hottest Days**")
            st.dataframe(hottest_days.reset_index(drop=True))
        with col_cool:
            st.write("**Top 5 Coolest Days**")
            st.dataframe(coolest_days.reset_index(drop=True))

    # Rainfall Trends Overtime
    with st.expander("Rainfall Trend"):
        fig_rain, ax_rain = plt.subplots(figsize=(10, 4))
        ax_rain.plot(data['Date'], data['Rainfall'], marker='s', color='blue', label='Rainfall')
        ax_rain.set_title('Rainfall Trend Over Time')
        ax_rain.set_xlabel('Date')
        ax_rain.set_ylabel('Rainfall (mm)')
        ax_rain.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig_rain)

    # Temperature Trends
    with st.expander("Temperature Trends Over a Month", expanded=True):
        fig_trend, ax_trend = plt.subplots(figsize=(10, 5))
        ax_trend.plot(data['Date'], data['Temperature'], marker='o', linestyle='-', color='teal')
        ax_trend.set_title('Temperature Trends')
        ax_trend.set_xlabel('Date')
        ax_trend.set_ylabel('Temperature (°C)')
        ax_trend.grid(True, linestyle='--', alpha=0.5)
        plt.xticks(rotation=45)
        st.pyplot(fig_trend)

    # Scatter Plot: Temperature vs Humidity
    with st.expander("Scatter Plot: Temperature vs Humidity"):
        fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=data, x='Temperature', y='Humidity', ax=ax_scatter, color='purple')
        sns.regplot(data=data, x='Temperature', y='Humidity', ax=ax_scatter, scatter=False, color='black')
        ax_scatter.set_title('Temperature vs Humidity')
        ax_scatter.set_xlabel('Temperature (°C)')
        ax_scatter.set_ylabel('Humidity (%)')
        st.pyplot(fig_scatter)

    # Histogram: Temperature Distribution
    with st.expander("Histogram: Temperature Distribution"):
        fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
        sns.histplot(data['Temperature'], bins=20, kde=True, color='orange', ax=ax_hist)
        ax_hist.set_title('Temperature Distribution')
        ax_hist.set_xlabel('Temperature (°C)')
        ax_hist.set_ylabel('Frequency')
        st.pyplot(fig_hist)

    # Heatmap: Correlation Matrix
    with st.expander("Correlation Heatmap"):
        fig_heat, ax_heat = plt.subplots(figsize=(6, 4))
        corr_matrix = data[['Temperature', 'Humidity']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_heat)
        ax_heat.set_title('Correlation Heatmap')
        st.pyplot(fig_heat)

    # Barplot: Monthly Average Temperature
    with st.expander("Monthly Average Temperature"):
        monthly_avg = data.groupby(data['Date'].dt.to_period('M'))['Temperature'].mean().reset_index()
        monthly_avg['Date'] = monthly_avg['Date'].dt.to_timestamp()
        fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
        sns.barplot(data=monthly_avg, x='Date', y='Temperature', hue='Date', palette='viridis', ax=ax_bar, legend=False)
        ax_bar.set_title('Monthly Average Temperature')
        plt.xticks(rotation=45)
        st.pyplot(fig_bar)

else:
    st.info("Please upload your weather CSV file to start the analysis.")
