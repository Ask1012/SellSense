# scripts/data_loader.py

import pandas as pd

def load_sales_data(file_path):
    """
    Load and preprocess sales data CSV without requiring a 'Date' column.
    Expected columns: ['User_ID', 'Product_ID', 'Gender', 'Age', 'Occupation',
                       'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status',
                       'Product_Category_1', 'Product_Category_2', 'Product_Category_3', 'Purchase']
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        
        # Fill missing numerical values with 0
        numeric_cols = df.select_dtypes(include='number').columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Fill missing categorical values with 'Unknown'
        cat_cols = df.select_dtypes(include='object').columns
        df[cat_cols] = df[cat_cols].fillna('Unknown')
        
        # Optional: convert 'Age' or 'Stay_In_Current_City_Years' to int if needed
        if 'Age' in df.columns:
            df['Age'] = pd.to_numeric(df['Age'], errors='coerce').fillna(0).astype(int)
        if 'Stay_In_Current_City_Years' in df.columns:
            df['Stay_In_Current_City_Years'] = pd.to_numeric(df['Stay_In_Current_City_Years'], errors='coerce').fillna(0).astype(int)
        
        return df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Example usage:
# if __name__ == "__main__":
#     df = load_sales_data("../data/sales_data.csv")
#     print(df.head())
