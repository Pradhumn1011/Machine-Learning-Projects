import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from pymongo import MongoClient
import time

# Connect to MongoDB Atlas with proper error handling
try:
    # Use a complete connection string and add timeout parameters
    client = MongoClient('mongodb+srv://lakshyamutha04:FYXav3yWFFD3x4ee@cluster0.k68hm.mongodb.net/?retryWrites=true&w=majority', 
                         serverSelectionTimeoutMS=5000)
    
    # Test the connection
    client.server_info()
    print("Successfully connected to MongoDB Atlas")
    
    db = client['test']
    collection = db['pricepredicition']
    
    # Fetch data from MongoDB with error handling
    mongo_data = list(collection.find({}))
    
    if not mongo_data:
        raise Exception("No data found in the collection")
        
    print(f"Successfully retrieved {len(mongo_data)} documents from MongoDB")
    
    # Convert to DataFrame
    df = pd.DataFrame(mongo_data)
    
    # Drop MongoDB's _id field if it exists
    if '_id' in df.columns:
        df = df.drop('_id', axis=1)
    
    # Print DataFrame schema and sample data for debugging
    print("DataFrame columns:", df.columns.tolist())
    print("\nSample data (first 3 rows):")
    print(df.head(3))
    
    # Features and target
    X = df[['product', 'height', 'length', 'breadth', 'industry']]
    y = df['cost']
    
    # One-hot encode product and industry
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded = encoder.fit_transform(X[['product', 'industry']])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['product', 'industry']))
    
    # Combine numeric features and encoded features
    X = pd.concat([X[['height', 'length', 'breadth']], encoded_df], axis=1)
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate and print model accuracy
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"\nModel R² score on training data: {train_score:.4f}")
    print(f"Model R² score on test data: {test_score:.4f}")
    
    # Prediction function
    def predict_cost(product, height, length, breadth):
        industries = ['Industry 1', 'Industry 2', 'Industry 3']
        predictions = {}
        
        for industry in industries:
            # Prepare input features for prediction
            input_data = pd.DataFrame({
                'height': [height],
                'length': [length],
                'breadth': [breadth],
                'product_table': [1 if product == 'table' else 0],
                'industry_Industry 2': [1 if industry == 'Industry 2' else 0],
                'industry_Industry 3': [1 if industry == 'Industry 3' else 0]
            })
            
            # Predict cost
            predicted_cost = model.predict(input_data)[0]
            predictions[industry] = round(predicted_cost, 2)
        
        return predictions
    
    # Example usage
    if __name__ == "__main__":
        while True:
            try:
                product = input("\nEnter product (jute bag/table): ").strip().lower()
                if product not in ['jute bag', 'table']:
                    print("Invalid product. Please enter 'jute bag' or 'table'.")
                    continue
                    
                height = float(input("Enter height (in cm): "))
                length = float(input("Enter length (in cm): "))
                breadth = float(input("Enter breadth (in cm): "))
                
                predicted_prices = predict_cost(product, height, length, breadth)
                
                print("\nPredicted Prices:")
                for industry, price in predicted_prices.items():
                    print(f"{industry}: ₹{price}")
                    
                another = input("\nPredict another price? (yes/no): ").lower()
                if another != 'yes':
                    break
            except ValueError:
                print("Please enter valid numeric values for dimensions.")
            except Exception as e:
                print(f"An error occurred: {e}")
                break

except Exception as e:
    print(f"Error connecting to MongoDB Atlas: {e}")
    
    # Fallback to CSV if MongoDB connection fails
    try:
        print("Falling back to CSV data source...")
        # Load dataset from CSV if available
        df = pd.read_csv('product_cost_prediction_100_entries_decimal.csv')
        print("Successfully loaded data from CSV")
        
        # Continue with the rest of the code
        # (This would be a repeat of the machine learning pipeline)
        
    except Exception as csv_e:
        print(f"Error loading CSV data: {csv_e}")
        print("Could not load data from any source. Please check your connection and data files.")