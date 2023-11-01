import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Load and preprocess the 2017-2021 dataset
df = pd.read_csv('ganga_statewise_2017-22.csv')
df = df.apply(pd.to_numeric, errors='coerce')
df.fillna(df.mean(), inplace=True)

# Features for prediction
features = ['DO (mg/L)', 'BOD (mg/L)', 'FC (MPN/100ml)', 'FS (MPN/100ml)', 'pH']
X = df[features]

# Prompt user for the contaminant to predict
print("Contaminants list:")
for idx, feature in enumerate(features, start=1):
    print(f"{idx}. {feature}")

contaminant_choice = int(input("Enter the number of the contaminant for plotting linear regression:"))

# Select the corresponding contaminant column
contaminant_column = features[contaminant_choice - 1]

# Train the Linear Regression model
model = LinearRegression()
model.fit(X, df[contaminant_column])

# Load and preprocess the 2022 dataset
df_2022 = pd.read_csv('ganga_statewise_2022.csv')
df_2022 = df_2022.apply(pd.to_numeric, errors='coerce')
df_2022.fillna(df_2022.mean(), inplace=True)

# Use the trained model to make predictions for 2022
predictions_2022 = model.predict(df_2022[features])

# Calculate mean absolute error (MAE)
mae = mean_absolute_error(df_2022[contaminant_column], predictions_2022)

# Calculate accuracy percentage based on MAE
accuracy_percentage = 100 - (mae / np.mean(df_2022[contaminant_column])) * 100

# Print MAE and accuracy percentage
print(f'Mean Absolute Error: {mae:.2f}')
print(f'Accuracy Percentage: {accuracy_percentage:.2f}%')

# Plotting the actual vs predicted values for 2022
plt.scatter(df_2022[contaminant_column], predictions_2022)
plt.xlabel(f'Actual {contaminant_column} (2022)')
plt.ylabel(f'Predicted {contaminant_column} (2022)')
plt.title(f'Actual vs Predicted {contaminant_column} (2022)')
plt.show()
