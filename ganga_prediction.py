import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

print("\nPrediction of Contamination Level for Year 2022 in Ganga River using Linear Regression\n")

df = pd.read_csv('ganga_statewise_2017-21.csv')
df = df.apply(pd.to_numeric, errors='coerce')
df.fillna(df.mean(), inplace=True)
grouped_df = df.groupby(['Station Code', 'Year'], as_index=False).mean()

features = ['DO (mg/L)', 'BOD (mg/L)', 'FC (MPN/100ml)', 'FS (MPN/100ml)']
X = df[features]
print("Contaminants list\n1. DO (mg/L)\n2. BOD (mg/L)\n3. FC (MPN/100ml)\n4. FS (MPN/100ml)")
contaminate = int(input("Enter contaminant for plotting linear regression:"))

predictions_list = []

for station_code in grouped_df['Station Code'].unique():
    station_data = grouped_df[grouped_df['Station Code'] == station_code]

    year_to_predict = station_data['Year'].max() + 1

    X_train, y_train = station_data[features], station_data[features[contaminate-1]]
    model = LinearRegression()
    model.fit(X_train, y_train)

    features_next_year = df[df['Station Code'] == station_code].tail(1)[features]
    predicted_contamination_level = model.predict(features_next_year)

    predictions_list.append({
        'Station Code': station_code,
        'PredictedContaminationLevel': predicted_contamination_level[0]
    })

predictions_df = pd.DataFrame(predictions_list)
predictions_df.to_csv('prediction_2022.csv', index=False)

# Read the actual values for 2022
actual_df = pd.read_csv('ganga_statewise_2022.csv')
actual_df = actual_df[actual_df['Year'] == 2022][['Station Code', features[contaminate-1]]]
actual_df.rename(columns={features[contaminate-1]: 'ActualContaminationLevel'}, inplace=True)

# Merge the predicted and actual DataFrames
final_df = predictions_df.merge(actual_df, on='Station Code')

# Calculate the percentage difference
final_df['PercentageDifference'] = ((final_df['ActualContaminationLevel'] - final_df['PredictedContaminationLevel']) / final_df['ActualContaminationLevel']) * 100

# Save the final DataFrame to a new CSV file
final_df.to_csv('final_predictions_2022.csv', index=False)

# Calculate the mean of all percentage differences
mean_percentage_difference = final_df['PercentageDifference'].mean()

# Calculate the accuracy of the model
accuracy = 100 - abs(mean_percentage_difference)

print("Mean Percentage Difference:", mean_percentage_difference)
print("Model Accuracy:", accuracy, "%")
