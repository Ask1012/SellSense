# scripts/data_loader.py

import pandas as pd

def load_sales_data(file_path):
    """
    Load and preprocess sales data CSV.
    Expected columns: ['Date', 'Product', 'Category', 'Quantity', 'Revenue', ...]
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        
        # Convert 'Date' column to datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        else:
            raise KeyError("CSV must have a 'Date' column")
        
        # Fill missing numerical values with 0
        numeric_cols = df.select_dtypes(include='number').columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Fill missing categorical values with 'Unknown'
        cat_cols = df.select_dtypes(include='object').columns
        df[cat_cols] = df[cat_cols].fillna('Unknown')
        
        return df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Example usage:
# if __name__ == "__main__":
#     df = load_sales_data("../data/sales_data.csv")
#     print(df.head())
