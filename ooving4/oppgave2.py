import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt  # Not used in this script
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score  # Not used in this script

# Step 1: Load and preprocess the data
data = pd.read_csv('dataset/life-expectancy-data.csv')

# Ensure 'Year', 'Schooling', and 'Life expectancy ' are numeric
data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
data['Schooling'] = pd.to_numeric(data['Schooling'], errors='coerce')
data['Life expectancy '] = pd.to_numeric(
    data['Life expectancy '], errors='coerce')

# Drop rows with missing values in critical columns
data = data.dropna(subset=['Life expectancy ', 'Schooling', 'Year'])

# Step 2: Feature selection
features = ['Schooling']
target = 'Life expectancy '

# Step 3: Model building
X = data[features]
y = data[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Ensure X_train and X_test are DataFrames with proper column names
X_train = pd.DataFrame(X_train, columns=features)
X_test = pd.DataFrame(X_test, columns=features)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Forecast predictors to 2020 using Linear Extrapolation for all countries
countries = data['Country'].unique()
predictions = []

for country in countries:
    country_data = data[data['Country'] == country]
    schooling_values = country_data['Schooling'].values
    years = country_data['Year'].values

    # Ensure years are numeric
    years = pd.to_numeric(years, errors='coerce')
    max_year = years.max()

    # If we have data for 2020, use it
    if 2020 in years:
        schooling_2020 = country_data.loc[country_data['Year']
                                          == 2020, 'Schooling'].values[0]
    else:
        # Use linear extrapolation to predict 'Schooling' for 2020
        if len(schooling_values) >= 2:
            # Fit a linear model on years vs. schooling_values
            years_df = pd.DataFrame({'Year': years})
            reg = LinearRegression()
            reg.fit(years_df, schooling_values)
            future_year = pd.DataFrame({'Year': [2020]})
            schooling_2020 = reg.predict(future_year)[0]
        else:
            # Not enough data points, use the last known value
            schooling_2020 = schooling_values[-1]

    # Step 5: Predict life expectancy for 2020
    schooling_df = pd.DataFrame({'Schooling': [schooling_2020]})
    life_expectancy_2020 = model.predict(schooling_df)[0]
    predictions.append({
        'Country': country,
        'Predicted Life Expectancy': life_expectancy_2020
    })

# Step 6: Determine the country with the highest life expectancy
predictions_df = pd.DataFrame(predictions)
top_country = predictions_df.loc[predictions_df['Predicted Life Expectancy'].idxmax(
)]

print(
    f"The country predicted to have the highest life expectancy in 2020 is {top_country['Country']} with a life expectancy of {top_country['Predicted Life Expectancy']:.2f} years.")
