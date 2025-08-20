# Importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset (replace with your dataset file path if available)
# Example: df = pd.read_csv("house_price_dataset.csv")
df = pd.read_csv("house_price_dataset.csv")

# Display first few rows
print(df.head())

# Selecting relevant features
X = df[['square_feet', 'bedrooms', 'bathrooms']]
y = df['price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Example prediction
example = [[2000, 3, 2]]  # 2000 sq ft, 3 bedrooms, 2 bathrooms
predicted_price = model.predict(example)
print("Predicted Price for example house:", predicted_price[0])
    
