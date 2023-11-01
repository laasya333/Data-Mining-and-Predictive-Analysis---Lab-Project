import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

print("\nPrediction of Contamination Level for Year 2023 in Ganga River using Linear Regression\n")

df = pd.read_csv('ganga_statewise_2017-22.csv')
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

print(predictions_df)

