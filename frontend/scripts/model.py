# scripts/model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def prepare_data(df, date_col='Date', target_col='Revenue'):
    """
    Prepares features for time-series like prediction.
    Uses 'Day', 'Month', 'Year' as simple features.
    """
    df = df.copy()
    df['Day'] = df[date_col].dt.day
    df['Month'] = df[date_col].dt.month
    df['Year'] = df[date_col].dt.year
    X = df[['Day', 'Month', 'Year']]
    y = df[target_col]
    return X, y

def train_linear_regression(X, y):
    """Train Linear Regression model and return model and RMSE"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return model, rmse

def train_random_forest(X, y, n_estimators=100):
    """Train Random Forest model and return model and RMSE"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return model, rmse

def predict_future_sales(model, future_dates):
    """
    future_dates: pandas DataFrame with 'Day', 'Month', 'Year'
    Returns predicted sales
    """
    predictions = model.predict(future_dates)
    return predictions

# Example usage:
# if __name__ == "__main__":
#     import scripts.data_loader as dl
#     df = dl.load_sales_data("../data/sales_data.csv")
#     X, y = prepare_data(df)
#     model, rmse = train_linear_regression(X, y)
#     print(f"Linear Regression RMSE: {rmse}")
