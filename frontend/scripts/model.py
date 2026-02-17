# scripts/model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def prepare_data(df, target_col='Purchase'):
    """
    Prepare features for regression.
    Uses numeric encoding for categorical columns.
    """
    df = df.copy()
    
    # Fill missing values (just in case)
    df = df.fillna(0)
    
    # Select features (all except target)
    X = df.drop(columns=[target_col])
    
    # Encode categorical features
    cat_cols = X.select_dtypes(include='object').columns
    for col in cat_cols:
        X[col] = X[col].astype('category').cat.codes  # simple label encoding
    
    y = df[target_col]
    return X, y

def train_linear_regression(X, y):
    """Train Linear Regression model and return model and RMSE"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    
    return model, rmse

def train_random_forest(X, y, n_estimators=100):
    """Train Random Forest model and return model and RMSE"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return model, rmse

def predict_sales(model, input_data):
    """
    Predict sales for new input data.
    input_data: pandas DataFrame with same feature columns as training data
    """
    # Encode categorical features
    cat_cols = input_data.select_dtypes(include='object').columns
    for col in cat_cols:
        input_data[col] = input_data[col].astype('category').cat.codes
    
    predictions = model.predict(input_data)
    return predictions

# Example usage:
# if __name__ == "__main__":
#     import scripts.data_loader as dl
#     df = dl.load_sales_data("../data/sales_data.csv")
#     X, y = prepare_data(df)
#     model, rmse = train_random_forest(X, y)
#     print(f"Random Forest RMSE: {rmse}")
