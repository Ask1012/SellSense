# frontend/app.py

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Import our modules
from scripts.data_loader import load_sales_data
from scripts.eda import sales_trend, top_categories, customer_behavior
from scripts.model import prepare_data, train_linear_regression, train_random_forest, predict_future_sales

st.set_page_config(page_title="SellSense - Sales Prediction", layout="wide")

st.title("📊 SellSense – AI Powered Sales Prediction System")

# --- Upload CSV ---
st.sidebar.header("Upload Sales Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = load_sales_data(uploaded_file)
    
    if df is not None:
        st.success("Data loaded successfully!")
        st.write("Preview of your data:")
        st.dataframe(df.head())

        # --- Filters ---
        st.sidebar.header("Filters")
        min_date = df['Date'].min()
        max_date = df['Date'].max()
        date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])
        
        categories = df['Category'].unique().tolist()
        selected_categories = st.sidebar.multiselect("Select Categories", categories, default=categories)

        # Apply filters
        filtered_df = df[(df['Date'] >= pd.to_datetime(date_range[0])) & 
                         (df['Date'] <= pd.to_datetime(date_range[1])) &
                         (df['Category'].isin(selected_categories))]

        # --- Sales Trend ---
        st.subheader("📈 Sales Trend Over Time")
        trend_df = sales_trend(filtered_df)
        fig, ax = plt.subplots()
        sns.lineplot(data=trend_df, x='Date', y='Revenue', marker='o', ax=ax)
        ax.set_ylabel("Revenue")
        st.pyplot(fig)

        # --- Top Categories ---
        st.subheader("🏆 Top 3 Performing Categories")
        top_cat_df = top_categories(filtered_df)
        st.bar_chart(top_cat_df.set_index('Category'))

        # --- Customer Behavior ---
        st.subheader("🛒 Top Products by Quantity Sold")
        top_prod_df = customer_behavior(filtered_df)
        st.table(top_prod_df)

        # --- Sales Prediction ---
        st.subheader("🔮 Future Sales Prediction")
        X, y = prepare_data(filtered_df)
        model_choice = st.selectbox("Select ML Model", ["Linear Regression", "Random Forest"])
        
        if st.button("Predict Next 30 Days"):
            future_dates = pd.date_range(filtered_df['Date'].max() + timedelta(1), 
                                         periods=30).to_frame(index=False, name='Date')
            future_dates['Day'] = future_dates['Date'].dt.day
            future_dates['Month'] = future_dates['Date'].dt.month
            future_dates['Year'] = future_dates['Date'].dt.year
            X_future = future_dates[['Day', 'Month', 'Year']]
            
            if model_choice == "Linear Regression":
                model, rmse = train_linear_regression(X, y)
            else:
                model, rmse = train_random_forest(X, y)
            
            predictions = predict_future_sales(model, X_future)
            future_dates['Predicted_Revenue'] = predictions
            
            st.write(f"Model RMSE: {rmse:.2f}")
            fig2, ax2 = plt.subplots()
            sns.lineplot(data=future_dates, x='Date', y='Predicted_Revenue', marker='o', ax=ax2)
            ax2.set_ylabel("Predicted Revenue")
            st.pyplot(fig2)
