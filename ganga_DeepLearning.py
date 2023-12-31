import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('surfacewq_1996-2006.csv', encoding='ISO-8859-1')

# Remove columns with more than 40% null values
threshold = 0.6 * len(df)
df = df.dropna(thresh=threshold, axis=1)

# Convert non-numeric values to NaN
df = df.apply(pd.to_numeric, errors='coerce')

# Impute missing values (replace with mean for simplicity)
df.fillna(df.mean(), inplace=True)

contaminants = df.columns[10:].tolist()

# Prompt the user to select the contaminant for prediction
print("Available Contaminants:")
for idx, contaminate in enumerate(contaminants, start=1):
    print(f"{idx}. {contaminate}")

contaminate_idx = int(input("Enter the number corresponding to the contaminant for prediction:")) - 1
selected_contaminant = contaminants[contaminate_idx]
print(selected_contaminant)

# Select relevant features for prediction (excluding the first 10 features)
features = [col for col in contaminants if col != selected_contaminant]
X = df[features]
y = df[selected_contaminant].astype(np.float32)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

# Create a deep learning model
model = Sequential()
model.add(Dense(units=120, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(units=1))  # Output layer with 1 neuron for regression

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Make predictions on the test set
y_pred = model.predict(X_test).flatten()
tolerance = 1.5
# Calculate Root Mean Squared Error
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

# Calculate R-squared
r_squared = metrics.r2_score(y_test, y_pred)

# Print metrics
print('Root Mean Squared Error (RMSE):', rmse)
print('R-squared:', r_squared)


# Calculate the percentage of predictions within the tolerance level
percentage = (np.sum(np.abs(y_pred - y_test)<= tolerance) / len(y_test)) * 100

# Print accuracy percentage
print('Accuracy Percentage' ':', percentage, '%')

# Visualize the results
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Water Contamination Level')
plt.ylabel('Predicted Water Contamination Level')
plt.title('Actual vs Predicted Water Contamination Level (Deep Learning)')
plt.show()
