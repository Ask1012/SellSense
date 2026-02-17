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

# Example usage:
# if __name__ == "__main__":
#     import scripts.data_loader as dl
#     df = dl.load_sales_data("../data/sales_data.csv")
#     print(sales_trend(df).head())
#     print(top_categories(df))
#     print(customer_behavior(df))
