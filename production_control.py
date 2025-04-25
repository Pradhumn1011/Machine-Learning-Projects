"""This is the start of the project for the toynic company created by pradhumn singh
this problem statement require us to manage the industry's data and basically predict demand and help them in production control
and inventory management"""


#first and foremost importing libaries that are to be used
import pandas as pd
import numpy as np
from pymongo import MongoClient,collection
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
# The second step is to access the data from the database. Right now we are taking mongodb but will tweak the code in the future
client = MongoClient('mongodb://localhost:27017/')  # Update with your connection string
db = client['industry_db']
demand_collection = db['demand_data']
inventory_collection = db['inventory_data']

cursor = demand_collection.find({})
df = pd.DataFrame(list(cursor))

if '_id' in df.columns:
    df.drop('_id', axis=1, inplace=True)

print("Data Sample:")
print(df.head())

#The next step is to preprocess the data that we are getting from the dataset. Right now we are just filling out the
df.fillna(method='ffill', inplace=True)

categorical_columns = df.select_dtypes(include=['object']).columns

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


X = df.drop(columns=['Demand'])
y = df['Demand']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
lr_model = LinearRegression()

rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)

# Ensemble prediction
rf_pred = rf_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)
lr_pred = lr_model.predict(X_test)
ensemble_pred = (rf_pred + xgb_pred + lr_pred) / 3

# Evaluate (optional)
mae = mean_absolute_error(y_test, ensemble_pred)
rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
print(f"Model Performance - MAE: {mae:.2f}, RMSE: {rmse:.2f}\n")


# Fetch all inventory batches (with date)
def get_inventory_batches(product, region,inventory_collection=None):
    batches = list(inventory_collection.find({'product': product, 'region': region}))
    if not batches:
        return [], 0

    total_inventory = sum(batch.get('current_inventory', 0) for batch in batches)
    # Sort by entry_date (oldest first)
    for batch in batches:
        batch['entry_date'] = datetime.strptime(batch['entry_date'], '%Y-%m-%d')

    batches = sorted(batches, key=lambda x: x['entry_date'])
    return batches, total_inventory


# Production decision logic
def should_increase_production(predicted_demand, current_inventory):
    buffer = 0.1 * predicted_demand
    if current_inventory < (predicted_demand - buffer):
        return "YES - Increase Production"
    else:
        return "NO - Do Not Increase Production"


# Print decisions and inventory management recommendations
print("Production Decisions and Inventory Recommendations:")

for idx, demand in enumerate(ensemble_pred):
    product = df.iloc[X_test.index[idx]]['product']  # Adjust these to your actual column names
    region = df.iloc[X_test.index[idx]]['region']

    batches, total_inventory = get_inventory_batches(product, region)
    decision = should_increase_production(demand, total_inventory)

    print(f"\nProduct: {product}, Region: {region}")
    print(f"Predicted Demand: {demand:.2f}")
    print(f"Current Inventory (all batches): {total_inventory}")
    print(f"Production Decision: {decision}")

    if batches:
        print("Inventory Management (Sell these batches first - oldest to newest):")
        for batch in batches:
            date_str = batch['entry_date'].strftime('%Y-%m-%d')
            print(f"  Batch ID: {batch['batch_id']}, Entry Date: {date_str}, Inventory: {batch['current_inventory']}")

    else:
        print("No inventory available for this product in this region.")

print("\nProcessing complete.")
