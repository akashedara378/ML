import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load Data
# You can replace this with pd.read_csv('your_data.csv') or another data loading method
data = {
    'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'feature2': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    'target': [1.1, 2.1, 3.0, 4.1, 4.9, 5.8, 7.1, 8.2, 9.0, 10.1]
}
df = pd.DataFrame(data)

# Step 2: Explore Data
print("Data Head:\n", df.head())
print("\nData Description:\n", df.describe())
print("\nMissing Values:\n", df.isnull().sum())

# Visualize the data relationships
sns.pairplot(df)
plt.show()

# Step 3: Prepare Data
X = df[['feature1', 'feature2']]
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Step 7: Visualize the Results
plt.scatter(y_test, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Values vs Predictions')
plt.show()

sns.regplot(x=y_test, y=y_pred, line_kws={"color": "red"})
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Regression Line')
plt.show()
