# scripts/eda.py

import pandas as pd

def sales_trend(df, date_col='Date', value_col='Revenue'):
    """
    Aggregate sales by date to show trends over time.
    Returns a DataFrame with 'Date' and 'Total_Sales'
    """
    trend = df.groupby(date_col)[value_col].sum().reset_index()
    trend = trend.sort_values(date_col)
    return trend

def top_categories(df, category_col='Category', value_col='Revenue', top_n=3):
    """
    Find top N product categories based on total revenue.
    Returns a DataFrame with 'Category' and 'Total_Revenue'
    """
    category_perf = df.groupby(category_col)[value_col].sum().reset_index()
    category_perf = category_perf.sort_values(value_col, ascending=False).head(top_n)
    return category_perf

def customer_behavior(df, product_col='Product', value_col='Quantity', top_n=5):
    """
    Identify most frequently purchased products.
    Returns a DataFrame with 'Product' and 'Total_Quantity'
    """
    behavior = df.groupby(product_col)[value_col].sum().reset_index()
    behavior = behavior.sort_values(value_col, ascending=False).head(top_n)
    return behavior

# eda.py

def top_products(data, n=10):
    return data['product'].value_counts().head(n)

def purchases_by_gender(df, gender_col='Gender', value_col='Quantity'):
    """Aggregate purchases by gender"""
    return df.groupby(gender_col)[value_col].sum().reset_index()

def purchases_by_city(df, city_col='City', value_col='Quantity'):
    """Aggregate purchases by city"""
    return df.groupby(city_col)[value_col].sum().reset_index()

def purchases_by_age(df, age_col='Age', value_col='Quantity', bins=None):
    """Aggregate purchases by age groups"""
    if bins is None:
        bins = [0, 18, 25, 35, 50, 100]
    labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]
    df['AgeGroup'] = pd.cut(df[age_col], bins=bins, labels=labels, right=False)
    return df.groupby('AgeGroup')[value_col].sum().reset_index()

# Example usage:
# if __name__ == "__main__":
#     import scripts.data_loader as dl
#     df = dl.load_sales_data("../data/sales_data.csv")
#     print(sales_trend(df).head())
#     print(top_categories(df))
#     print(customer_behavior(df))
