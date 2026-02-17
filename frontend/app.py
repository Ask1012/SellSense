# frontend/app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import our modules
from scripts.data_loader import load_sales_data
from scripts.eda import top_products, top_categories, purchases_by_gender, purchases_by_city, purchases_by_age
from scripts.model import prepare_data, train_linear_regression, train_random_forest, predict_sales

st.set_page_config(page_title="SellSense - Sales Prediction", layout="wide")

st.title("📊 SellSense – AI Powered Sales Analysis & Prediction")

# --- Upload CSV ---
st.sidebar.header("Upload Sales Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = load_sales_data(uploaded_file)
    st.write("Columns in the dataset:", df.columns.tolist())

    if df is not None:
        st.success("Data loaded successfully!")
        st.write("Preview of your data:")
        st.dataframe(df.head())

        # --- Top Products ---
        st.subheader("🏆 Top Products by Purchase")
        top_prod_df = top_products(df)
        st.bar_chart(top_prod_df.set_index('Product_ID'))

        # --- Top Categories ---
        st.subheader("📦 Top Product Categories")
        top_cat_df = top_categories(df)
        st.bar_chart(top_cat_df.set_index('Product_Category_1'))

        # --- Purchases by Gender ---
        st.subheader("👥 Purchases by Gender")
        gender_df = purchases_by_gender(df)
        st.bar_chart(gender_df.set_index('Gender'))

        # --- Purchases by City ---
        st.subheader("🏙️ Purchases by City Category")
        city_df = purchases_by_city(df)
        st.bar_chart(city_df.set_index('City_Category'))

        # --- Purchases by Age ---
        st.subheader("🎂 Purchases by Age Group")
        age_df = purchases_by_age(df)
        st.bar_chart(age_df.set_index('Age'))

        # --- Sales Prediction ---
        st.subheader("🔮 Predict Purchase Amounts")
        X, y = prepare_data(df)
        model_choice = st.selectbox("Select ML Model", ["Linear Regression", "Random Forest"])
        
        if st.button("Train & Predict"):
            if model_choice == "Linear Regression":
                model, rmse = train_linear_regression(X, y)
            else:
                model, rmse = train_random_forest(X, y)
            
            st.write(f"Model RMSE: {rmse:.2f}")
            
            # Predict on the same dataset (or new inputs)
            predictions = predict_sales(model, X)
            df['Predicted_Purchase'] = predictions
            st.subheader("Predicted Purchase Amounts (sample)")
            st.dataframe(df[['User_ID', 'Product_ID', 'Purchase', 'Predicted_Purchase']].head(20))
