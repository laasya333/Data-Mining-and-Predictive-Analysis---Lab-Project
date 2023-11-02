import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn import metrics
import matplotlib.pyplot as plt

# Load your CSV data into a pandas DataFrame
df = pd.read_csv('ganga_statewise_2017-21.csv')

# Convert non-numeric values to NaN
df = df.apply(pd.to_numeric, errors='coerce')

# Impute missing values (replace with mean for simplicity)
df.fillna(df.mean(), inplace=True)

# Select relevant features for prediction
features = ['DO (mg/L)', 'BOD (mg/L)', 'FC (MPN/100ml)', 'FS (MPN/100ml)', 'pH']
X = df[features]
print("Contaminates list\n1. DO (mg/L)\n2. BOD (mg/L)\n3. FC (MPN/100ml)\n4. FS (MPN/100ml)\n5. pH")
contaminate = int(input("Enter contaminate for plotting DEEP LEARNING:"))
y = df[features[contaminate-1]] 

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a deep learning model
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(units=1))  # Output layer with 1 neuron for regression

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Make predictions on the test set
y_pred = model.predict(X_test).flatten()

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
plt.title('Actual vs Predicted Water Contamination Level (Deep Learning)')
plt.show()
