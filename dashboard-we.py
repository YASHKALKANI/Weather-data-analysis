# # import streamlit as st
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import seaborn as sns

# # # Page Config
# # st.set_page_config(page_title="🌦️ Weather Data Dashboard", layout="wide")

# # # Title
# # st.title("🌦️ Weather Data Analysis Dashboard")
# # st.caption("Built with ❤️ using Streamlit | Your Weather Insights")

# # # Theme Toggle
# # theme = st.selectbox("Select Theme", ["Light", "Dark"])
# # if theme == "Dark":
# #     sns.set_theme(style="darkgrid")
# # else:
# #     sns.set_theme(style="whitegrid")

# # # Upload CSV
# # uploaded_file = st.file_uploader("Upload your weather CSV file", type=["csv"])

# # if uploaded_file is not None:
# #     data = pd.read_csv(uploaded_file)
# #     data['Date'] = pd.to_datetime(data['Date'])
# #     data = data.sort_values('Date')

# #     data = data.rename(columns={
# #         "Temperature (°C)": "Temperature",
# #         "Humidity (%)": "Humidity",
# #         "Wind Speed (km/h)": "Wind Speed",
# #         "Rainfall (mm)": "Rainfall"
# #     })

# #     # Date Range Filter
# #     min_date = data['Date'].min()
# #     max_date = data['Date'].max()
# #     start_date, end_date = st.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
# #     filtered_data = data[(data['Date'] >= pd.to_datetime(start_date)) & (data['Date'] <= pd.to_datetime(end_date))]

# #     st.download_button("📥 Download Filtered Data", filtered_data.to_csv(index=False), "filtered_weather_data.csv", "text/csv")

# #     # Metrics
# #     col1, col2, col3 = st.columns(3)
# #     with col1:
# #         st.metric("🌡️ Average Temperature (°C)", f"{filtered_data['Temperature'].mean():.2f}")
# #     with col2:
# #         st.metric("💧 Average Humidity (%)", f"{filtered_data['Humidity'].mean():.2f}")
# #     with col3:
# #         st.metric("☔ Total Rainfall (mm)", f"{filtered_data['Rainfall'].sum():.2f}")

# #     # Combined Min, Max, Avg Temperature Chart
# #     with st.expander("🌡️ Temperature Trends (Min, Max, Avg)", expanded=True):
# #         temp_df = filtered_data.groupby('Date')['Temperature'].agg(['min', 'max', 'mean']).reset_index()
# #         fig_temp, ax_temp = plt.subplots(figsize=(10, 4))
# #         ax_temp.plot(temp_df['Date'], temp_df['min'], label='Min Temp', color='skyblue')
# #         ax_temp.plot(temp_df['Date'], temp_df['max'], label='Max Temp', color='tomato')
# #         ax_temp.plot(temp_df['Date'], temp_df['mean'], label='Avg Temp', color='seagreen')
# #         ax_temp.set_title('Min, Max, Avg Temperature Trends')
# #         ax_temp.set_xlabel('Date')
# #         ax_temp.set_ylabel('Temperature (°C)')
# #         ax_temp.legend()
# #         plt.xticks(rotation=45)
# #         st.pyplot(fig_temp)

# #     # Rainfall vs Temperature with Humidity Hue
# #     with st.expander("🌧️ Rainfall vs Temperature with Humidity"):
# #         fig_scatter, ax_scatter = plt.subplots(figsize=(8, 5))
# #         scatter = ax_scatter.scatter(filtered_data['Temperature'], filtered_data['Rainfall'], c=filtered_data['Humidity'], cmap='viridis', s=60)
# #         ax_scatter.set_title('Rainfall vs Temperature (Color: Humidity)')
# #         ax_scatter.set_xlabel('Temperature (°C)')
# #         ax_scatter.set_ylabel('Rainfall (mm)')
# #         cbar = fig_scatter.colorbar(scatter, ax=ax_scatter)
# #         cbar.set_label('Humidity (%)')
# #         st.pyplot(fig_scatter)

# #     # Correlation Heatmap
# #     with st.expander("🧩 Correlation Heatmap"):
# #         fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
# #         corr_matrix = filtered_data[['Temperature', 'Humidity', 'Wind Speed', 'Rainfall']].corr()
# #         sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
# #         ax_corr.set_title('Correlation Heatmap')
# #         st.pyplot(fig_corr)

# #     # Footer Branding
# #     st.markdown("---")
# #     st.markdown("Made with ❤️ by Your Name | Data Analysis Simplified")

# # else:
# #     st.info("👆 Please upload your weather CSV file to start the analysis.")



# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # from sklearn.linear_model import LinearRegression
# # from sklearn.preprocessing import OrdinalEncoder

# # # Page Config
# # st.set_page_config(page_title="🌦️ Weather Data Dashboard", layout="wide")

# # # Sidebar Branding
# # with st.sidebar:
# #     st.title("🌦️ Weather Dashboard")
# #     st.markdown("Created with ❤️ by Your Name")
# #     st.markdown("[Visit Portfolio](https://your-portfolio-link.com)")

# # # Header
# # st.markdown("""
# # # 🌤️ **Interactive Weather Data Analysis & Prediction Dashboard**
# # Upload your weather CSV, analyze trends, and get a **7-day temperature prediction** based on your data.
# # ---
# # """)

# # # Upload CSV
# # uploaded_file = st.file_uploader("Upload your weather CSV file", type=["csv"])

# # if uploaded_file is not None:
# #     with st.spinner('Processing your data...'):
# #         data = pd.read_csv(uploaded_file)
# #         data['Date'] = pd.to_datetime(data['Date'])
# #         data = data.sort_values('Date')

# #         data = data.rename(columns={
# #             "Temperature (°C)": "Temperature",
# #             "Humidity (%)": "Humidity",
# #             "Wind Speed (km/h)": "Wind Speed",
# #             "Rainfall (mm)": "Rainfall"
# #         })

# #         # Date Range Filter
# #         min_date = data['Date'].min()
# #         max_date = data['Date'].max()
# #         start_date, end_date = st.date_input("Select Date Range for Analysis:", [min_date, max_date], min_value=min_date, max_value=max_date)
# #         filtered_data = data[(data['Date'] >= pd.to_datetime(start_date)) & (data['Date'] <= pd.to_datetime(end_date))]

# #         st.download_button("📥 Download Filtered Data", filtered_data.to_csv(index=False), "filtered_weather_data.csv", "text/csv")

# #         # Metrics
# #         col1, col2, col3 = st.columns(3)
# #         with col1:
# #             st.metric("🌡️ Avg Temperature (°C)", f"{filtered_data['Temperature'].mean():.2f}")
# #         with col2:
# #             st.metric("💧 Avg Humidity (%)", f"{filtered_data['Humidity'].mean():.2f}")
# #         with col3:
# #             st.metric("☔ Total Rainfall (mm)", f"{filtered_data['Rainfall'].sum():.2f}")

# #         st.markdown("---")

# #         # Temperature Trends
# #         with st.expander("🌡️ Temperature Trends (Min, Max, Avg)", expanded=True):
# #             temp_df = filtered_data.groupby('Date')['Temperature'].agg(['min', 'max', 'mean']).reset_index()
# #             fig_temp, ax_temp = plt.subplots(figsize=(10, 4))
# #             ax_temp.plot(temp_df['Date'], temp_df['min'], label='Min Temp', color='skyblue')
# #             ax_temp.plot(temp_df['Date'], temp_df['max'], label='Max Temp', color='tomato')
# #             ax_temp.plot(temp_df['Date'], temp_df['mean'], label='Avg Temp', color='seagreen')
# #             ax_temp.set_title('Min, Max, Avg Temperature Trends')
# #             ax_temp.set_xlabel('Date')
# #             ax_temp.set_ylabel('Temperature (°C)')
# #             ax_temp.legend()
# #             plt.xticks(rotation=45)
# #             st.pyplot(fig_temp)

# #         # Rainfall vs Temperature Scatter
# #         with st.expander("🌧️ Rainfall vs Temperature (Colored by Humidity)"):
# #             fig_scatter, ax_scatter = plt.subplots(figsize=(8, 5))
# #             scatter = ax_scatter.scatter(filtered_data['Temperature'], filtered_data['Rainfall'], c=filtered_data['Humidity'], cmap='viridis', s=60)
# #             ax_scatter.set_title('Rainfall vs Temperature (Color: Humidity)')
# #             ax_scatter.set_xlabel('Temperature (°C)')
# #             ax_scatter.set_ylabel('Rainfall (mm)')
# #             cbar = fig_scatter.colorbar(scatter, ax=ax_scatter)
# #             cbar.set_label('Humidity (%)')
# #             st.pyplot(fig_scatter)

# #         # Correlation Heatmap
# #         with st.expander("🧩 Correlation Heatmap"):
# #             fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
# #             corr_matrix = filtered_data[['Temperature', 'Humidity', 'Wind Speed', 'Rainfall']].corr()
# #             sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
# #             ax_corr.set_title('Correlation Heatmap')
# #             st.pyplot(fig_corr)

# #         # Prediction Section
# #         st.markdown("---")
# #         st.subheader("🔮 7-Day Temperature Prediction")
# #         if len(data) >= 30:
# #             last_30_days = data.tail(30)
# #             X = np.arange(len(last_30_days)).reshape(-1, 1)
# #             y = last_30_days['Temperature'].values
# #             model = LinearRegression()
# #             model.fit(X, y)
# #             X_future = np.arange(len(last_30_days), len(last_30_days) + 7).reshape(-1, 1)
# #             y_pred = model.predict(X_future)
# #             future_dates = pd.date_range(start=data['Date'].max() + pd.Timedelta(days=1), periods=7)

# #             pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Temperature (°C)': y_pred})
# #             st.dataframe(pred_df)

# #             fig_pred, ax_pred = plt.subplots(figsize=(10, 4))
# #             ax_pred.plot(data['Date'], data['Temperature'], label='Historical Temperature', color='gray')
# #             ax_pred.plot(pred_df['Date'], pred_df['Predicted Temperature (°C)'], label='Predicted Temperature', color='red', linestyle='--', marker='o')
# #             ax_pred.set_title('7-Day Temperature Prediction')
# #             ax_pred.set_xlabel('Date')
# #             ax_pred.set_ylabel('Temperature (°C)')
# #             ax_pred.legend()
# #             plt.xticks(rotation=45)
# #             st.pyplot(fig_pred)
# #         else:
# #             st.info("📊 Need at least 30 days of data for prediction.")

# #         st.markdown("---")
# #         st.success("✅ Dashboard Loaded Successfully!")

# # else:
# #     st.info("👆 Please upload your weather CSV file to start the analysis.")



# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.linear_model import LinearRegression

# # Page Config
# st.set_page_config(page_title="🌦️ Weather Data Dashboard", layout="wide")

# # Sidebar Branding
# with st.sidebar:
#     st.title("🌦️ Weather Dashboard")
#     st.markdown("Created with ❤️ by Your Name")
#     st.markdown("[Visit Portfolio](https://your-portfolio-link.com)")

# # Header
# st.markdown("""
# # 🌤️ **Interactive Weather Data Analysis & Prediction Dashboard**
# Upload your weather CSV, analyze trends, explore rainfall, humidity, and temperature patterns, and get a **7-day temperature prediction** based on your uploaded data.
# ---
# """)

# # Upload CSV
# uploaded_file = st.file_uploader("📂 **Upload your weather CSV file**", type=["csv"])

# if uploaded_file is not None:
#     with st.spinner('Processing your data...'):
#         data = pd.read_csv(uploaded_file)
#         data['Date'] = pd.to_datetime(data['Date'])
#         data = data.sort_values('Date')

#         data = data.rename(columns={
#             "Temperature (°C)": "Temperature",
#             "Humidity (%)": "Humidity",
#             "Wind Speed (km/h)": "Wind Speed",
#             "Rainfall (mm)": "Rainfall"
#         })

#         # Date Range Filter
#         min_date = data['Date'].min()
#         max_date = data['Date'].max()
#         start_date, end_date = st.date_input(
#             "📅 **Select Date Range for Analysis:**", 
#             [min_date, max_date], 
#             min_value=min_date, 
#             max_value=max_date
#         )
#         filtered_data = data[(data['Date'] >= pd.to_datetime(start_date)) & (data['Date'] <= pd.to_datetime(end_date))]

#         st.download_button("📥 Download Filtered Data", filtered_data.to_csv(index=False), "filtered_weather_data.csv", "text/csv")

#         # Metrics Display
#         col1, col2, col3, col4 = st.columns(4)
#         with col1:
#             st.metric("🌡️ Avg Temperature", f"{filtered_data['Temperature'].mean():.2f} °C")
#         with col2:
#             st.metric("💧 Avg Humidity", f"{filtered_data['Humidity'].mean():.2f} %")
#         with col3:
#             st.metric("☔ Total Rainfall", f"{filtered_data['Rainfall'].sum():.2f} mm")
#         with col4:
#             st.metric("💨 Avg Wind Speed", f"{filtered_data['Wind Speed'].mean():.2f} km/h")

#         st.markdown("---")

#         # Smoothed Temperature Trends
#         with st.expander("🌡️ **Smoothed Temperature Trends with Rolling Average (7 days)**", expanded=True):
#             fig_temp, ax_temp = plt.subplots(figsize=(10, 4))
#             filtered_data['Temp_Rolling'] = filtered_data['Temperature'].rolling(window=7).mean()
#             ax_temp.plot(filtered_data['Date'], filtered_data['Temperature'], label='Daily Temperature', alpha=0.5, color='skyblue')
#             ax_temp.plot(filtered_data['Date'], filtered_data['Temp_Rolling'], label='7-Day Rolling Avg', color='darkblue')
#             ax_temp.set_title('Smoothed Temperature Trends')
#             ax_temp.set_xlabel('Date')
#             ax_temp.set_ylabel('Temperature (°C)')
#             ax_temp.legend()
#             ax_temp.grid(True, linestyle='--', alpha=0.3)
#             plt.xticks(rotation=45)
#             st.pyplot(fig_temp)

#         # Humidity Trend
#         with st.expander("💧 **Humidity Trend Over Time**"):
#             fig_hum, ax_hum = plt.subplots(figsize=(10, 4))
#             ax_hum.plot(filtered_data['Date'], filtered_data['Humidity'], label='Humidity (%)', color='seagreen')
#             ax_hum.set_title('Humidity Trend Over Time')
#             ax_hum.set_xlabel('Date')
#             ax_hum.set_ylabel('Humidity (%)')
#             ax_hum.grid(True, linestyle='--', alpha=0.3)
#             plt.xticks(rotation=45)
#             st.pyplot(fig_hum)

#         # Rainfall vs Temperature Scatter
#         with st.expander("🌧️ **Rainfall vs Temperature Scatter (Color: Humidity)**"):
#             fig_scatter, ax_scatter = plt.subplots(figsize=(8, 5))
#             scatter = ax_scatter.scatter(
#                 filtered_data['Temperature'], 
#                 filtered_data['Rainfall'], 
#                 c=filtered_data['Humidity'], 
#                 cmap='viridis', s=60, alpha=0.7
#             )
#             ax_scatter.set_title('Rainfall vs Temperature (Color = Humidity)')
#             ax_scatter.set_xlabel('Temperature (°C)')
#             ax_scatter.set_ylabel('Rainfall (mm)')
#             cbar = fig_scatter.colorbar(scatter, ax=ax_scatter)
#             cbar.set_label('Humidity (%)')
#             st.pyplot(fig_scatter)

#         with st.expander("📈 **Monthly Average Temperature Trends**"):
#             monthly_avg = filtered_data.resample('M', on='Date')['Temperature'].mean().reset_index()
#             fig_month, ax_month = plt.subplots(figsize=(10, 4))
#             sns.barplot(
#                 data=monthly_avg,
#                 x='Date',
#                 y='Temperature',
#                 hue='Date',
#                 palette='Blues',
#                 ax=ax_month,
#                 dodge=False,
#                 legend=False
#             )
#             ax_month.set_title('Monthly Average Temperature')
#             ax_month.set_xlabel('Month')
#             ax_month.set_ylabel('Temperature (°C)')
#             plt.xticks(rotation=45)
#             st.pyplot(fig_month)


#         # Correlation Heatmap
#         with st.expander("🧩 **Correlation Heatmap**"):
#             fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
#             corr_matrix = filtered_data[['Temperature', 'Humidity', 'Wind Speed', 'Rainfall']].corr()
#             sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
#             ax_corr.set_title('Correlation Heatmap')
#             st.pyplot(fig_corr)

#         st.markdown("---")
#         st.subheader("🔮 **7-Day Temperature Prediction**")
#         if len(data) >= 30:
#             last_30_days = data.tail(30)
#             X = np.arange(len(last_30_days)).reshape(-1, 1)
#             y = last_30_days['Temperature'].values
#             model = LinearRegression()
#             model.fit(X, y)
#             X_future = np.arange(len(last_30_days), len(last_30_days) + 7).reshape(-1, 1)
#             y_pred = model.predict(X_future)
#             future_dates = pd.date_range(start=data['Date'].max() + pd.Timedelta(days=1), periods=7)

#             pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Temperature (°C)': y_pred})
#             st.dataframe(pred_df)

#             st.download_button("📥 Download Predictions", pred_df.to_csv(index=False), "predicted_temperatures.csv", "text/csv")

#             fig_pred, ax_pred = plt.subplots(figsize=(10, 4))
#             ax_pred.plot(data['Date'], data['Temperature'], label='Historical Temperature', color='gray')
#             ax_pred.plot(pred_df['Date'], pred_df['Predicted Temperature (°C)'], label='Predicted Temperature', color='red', linestyle='--', marker='o')
#             ax_pred.set_title('7-Day Temperature Prediction')
#             ax_pred.set_xlabel('Date')
#             ax_pred.set_ylabel('Temperature (°C)')
#             ax_pred.legend()
#             plt.xticks(rotation=45)
#             st.pyplot(fig_pred)
#         else:
#             st.info("📊 Need at least 30 days of data for prediction.")

#         st.success("✅ Dashboard Loaded Successfully!")

# else:
#     st.info("👆 Please upload your weather CSV file to start the analysis.")

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
import requests
import json

# Page Config
st.set_page_config(page_title="🌦️ Weather Data Dashboard", layout="wide")

# CSS for gradient header and rounded metric boxes
st.markdown("""
<style>
.gradient-text {
    background: linear-gradient(to right, #36D1DC, #5B86E5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-box {
    padding: 10px;
    border-radius: 15px;
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(5px);
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Sidebar Branding with Lottie
with st.sidebar:
    st.markdown("<h2 class='gradient-text'>🌦️ Weather Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("Created with ❤️ by **Your Name**")
    st.markdown("[🌐 Visit Portfolio](https://your-portfolio-link.com)")
    st.image("https://assets10.lottiefiles.com/packages/lf20_iwmd6pyr.json", width=200, caption="Weather Animation")

    st.markdown("---")
    st.info("Upload weather CSV to begin your analysis and prediction journey 🚀.")

# Upload CSV
uploaded_file = st.file_uploader("📂 **Upload your weather CSV file**", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.rename(columns={
        "Temperature (°C)": "Temperature",
        "Humidity (%)": "Humidity",
        "Wind Speed (km/h)": "Wind Speed",
        "Rainfall (mm)": "Rainfall"
    })
    data = data.sort_values('Date')

    st.markdown("---")
    st.markdown("<h2 class='gradient-text'>📊 Data Overview & Metrics</h2>", unsafe_allow_html=True)

    min_date = data['Date'].min()
    max_date = data['Date'].max()
    start_date, end_date = st.date_input("📅 Select Date Range:", [min_date, max_date], min_value=min_date, max_value=max_date)
    filtered_data = data[(data['Date'] >= pd.to_datetime(start_date)) & (data['Date'] <= pd.to_datetime(end_date))]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='metric-box'><h3>🌡️ Avg Temp</h3><h2>{filtered_data['Temperature'].mean():.1f} °C</h2></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-box'><h3>💧 Avg Humidity</h3><h2>{filtered_data['Humidity'].mean():.1f} %</h2></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-box'><h3>🌬️ Avg Wind</h3><h2>{filtered_data['Wind Speed'].mean():.1f} km/h</h2></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='metric-box'><h3>☔ Total Rainfall</h3><h2>{filtered_data['Rainfall'].sum():.1f} mm</h2></div>", unsafe_allow_html=True)

    st.markdown("---")

    # Interactive Line Chart: Temperature Trends
    st.subheader("📈 Temperature Trends Over Time")
    fig_temp = px.line(filtered_data, x='Date', y='Temperature', markers=True, line_shape='linear', color_discrete_sequence=["#5B86E5"])
    fig_temp.update_layout(hovermode='x unified', xaxis_title='Date', yaxis_title='Temperature (°C)')
    st.plotly_chart(fig_temp, use_container_width=True)

    # Rainfall Pie Chart
    st.subheader("☔ Rainfall Distribution")
    # rain_bins = pd.cut(filtered_data['Rainfall'], bins=[-0.1, 0, 5, 20, filtered_data['Rainfall'].max()], labels=["No Rain", "Light", "Moderate", "Heavy"])


    # Determine unique, sorted bins safely
    max_rainfall = filtered_data['Rainfall'].max()
    bins = [-0.1, 0, 5, 20]

    # Add max_rainfall only if it is greater than 20
    if max_rainfall > 20:
        bins.append(max_rainfall)

    # Remove duplicate edges if any
    bins = sorted(list(set(bins)))

    rain_bins = pd.cut(
        filtered_data['Rainfall'],
        bins=bins,
        labels=["No Rain", "Light", "Moderate", "Heavy"][:len(bins)-1],
        include_lowest=True
    )

    rain_counts = rain_bins.value_counts().sort_index()
    fig_pie = px.pie(values=rain_counts.values, names=rain_counts.index, color_discrete_sequence=px.colors.sequential.Blues, hole=0.4)
    fig_pie.update_traces(textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

    # Correlation Heatmap
    st.subheader("🧩 Correlation Heatmap")
    corr = filtered_data[['Temperature', 'Humidity', 'Wind Speed', 'Rainfall']].corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu', aspect='auto', title="Correlation Matrix")
    st.plotly_chart(fig_corr, use_container_width=True)

    # 7-Day Temperature Prediction
    st.subheader("🔮 7-Day Temperature Prediction")
    if len(filtered_data) >= 30:
        last_30 = filtered_data.tail(30)
        X = np.arange(len(last_30)).reshape(-1, 1)
        y = last_30['Temperature'].values
        model = LinearRegression().fit(X, y)
        X_future = np.arange(len(last_30), len(last_30) + 7).reshape(-1, 1)
        y_future = model.predict(X_future)
        future_dates = pd.date_range(start=filtered_data['Date'].max() + pd.Timedelta(days=1), periods=7)
        pred_df = pd.DataFrame({"Date": future_dates, "Predicted Temperature (°C)": y_future.round(2)})

        st.dataframe(pred_df)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=filtered_data['Date'], y=filtered_data['Temperature'], mode='lines+markers', name='Historical', line=dict(color='gray')))
        fig_pred.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Predicted Temperature (°C)'], mode='lines+markers', name='Predicted', line=dict(color='red', dash='dash')))
        fig_pred.update_layout(title="7-Day Temperature Prediction", xaxis_title="Date", yaxis_title="Temperature (°C)", hovermode='x unified')
        st.plotly_chart(fig_pred, use_container_width=True)

    else:
        st.info("📊 Need at least 30 days of data for prediction.")

    # Download button
    st.download_button("📥 Download Filtered Data as CSV", filtered_data.to_csv(index=False), "filtered_weather_data.csv", "text/csv")

    st.success("✅ Dashboard Loaded Successfully!")

else:
    st.info("👆 Upload your weather CSV to begin analysis.")

