import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import matplotlib.pyplot as plt

# Load your CSV data 
# into a pandas DataFrame
df = pd.read_csv('ganga_statewise_2017-22.csv')

# Convert non-numeric values to NaN
df = df.apply(pd.to_numeric, errors='coerce')

# Impute missing values (replace with mean for simplicity)
df.fillna(df.mean(), inplace=True)

# Select relevant features for prediction
features = ['DO (mg/L)', 'BOD (mg/L)', 'FC (MPN/100ml)', 'FS (MPN/100ml)', 'pH']
X = df[features]
print("Contaminates list\n1. DO (mg/L)\n2. BOD (mg/L)\n3. FC (MPN/100ml)\n4. FS (MPN/100ml)\n5. pH")
contaminate = int(input("Enter contaminate for plotting RANDOM FOREST:"))
y = df[features[contaminate-1]] 

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=0, max_depth=5, min_samples_split=2)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Calculate Mean Absolute Error (MAE)
mae = metrics.mean_absolute_error(y_test, y_pred)

# Calculate accuracy percentage
accuracy_percentage = (1 - mae / y_test.mean()) * 100

# Print the accuracy percentage
print('Accuracy Percentage:', accuracy_percentage)

# Visualize the results
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Water Contamination Level')
plt.ylabel('Predicted Water Contamination Level')
plt.title('Actual vs Predicted Water Contamination Level (Random Forest)')
plt.show()
