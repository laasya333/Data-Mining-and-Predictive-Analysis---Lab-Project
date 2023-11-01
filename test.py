import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import VarianceThreshold

# Load your dataset into a DataFrame (replace 'your_dataset.csv' with your actual file)
df = pd.read_csv('surfacewq_1996-2006.csv')

# Filter out columns with high missing values
threshold_missing = 0.5
df = df.dropna(thresh=len(df) * threshold_missing, axis=1)

# Filter out low variance columns
threshold_variance = 0.1
selector = VarianceThreshold(threshold=threshold_variance)
df_filtered = selector.fit_transform(df)
selected_columns = df.columns[selector.get_support()].tolist()
df = df[selected_columns]

# Get user input for the target column from the available columns
print("Available columns for target variable:", df.columns)
target_column = input("Enter the name of the target column: ")

# Check if the provided target column is in the DataFrame
if target_column not in df.columns:
    print("Invalid target column!")
else:
    # Prepare the data for modeling
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Calculate the mean squared error
    mse = mean_squared_error(y_test, predictions)
    print("Mean Squared Error:", mse)
