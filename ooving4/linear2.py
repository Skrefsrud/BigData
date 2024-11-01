import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

# Step 1: Load and preprocess the data
data = pd.read_csv('dataset/life-expectancy-data.csv')
data = data.dropna(subset=['Life expectancy ', 'Schooling'])

# Step 2: Feature selection
features = ['Schooling']
target = 'Life expectancy '

# Step 3: Model building
X = data[features]
y = data[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Forecast predictors to 2020
countries = data['Country'].unique()
predictions = []

for country in countries:
    country_data = data[data['Country'] == country]
    schooling_values = country_data['Schooling'].values
    years = country_data['Year'].values

    # Ensure years are numeric
    years = pd.to_numeric(years, errors='coerce')
    max_year = years.max()

    steps = int(2020 - max_year)  # steps = 5

    # If we have data for 2020, use it
    if 2020 in years:
        schooling_2020 = country_data.loc[country_data['Year']
                                          == 2020, 'Schooling'].values[0]
    else:
        if steps <= 0:
            # Use the last available value
            schooling_2020 = schooling_values[-1]
        else:
            if len(schooling_values) < 10:
                # Not enough data to fit ARIMA, use linear extrapolation
                # Fit a linear model on years vs. schooling_values
                years_df = pd.DataFrame({'Year': years})
                reg = LinearRegression()
                reg.fit(years_df, schooling_values)
                future_year = pd.DataFrame({'Year': [2020]})
                schooling_2020 = reg.predict(future_year)[0]
            else:
                try:
                    # Forecast schooling to 2020 using ARIMA
                    model_arima = ARIMA(schooling_values, order=(1, 1, 1))
                    model_fit = model_arima.fit()
                    forecast = model_fit.forecast(steps=steps)
                    schooling_2020 = forecast[-1]
                except Exception as e:
                    print(f"ARIMA model failed for {country} with error: {e}")
                    print("Using linear extrapolation instead.")
                    # Fit a linear model on years vs. schooling_values
                    reg = LinearRegression()
                    reg.fit(years.reshape(-1, 1), schooling_values)
                    schooling_2020 = reg.predict([[2020]])[0]

    # Step 5: Predict life expectancy for 2020
    life_expectancy_2020 = model.predict([[schooling_2020]])
    predictions.append({
        'Country': country,
        'Predicted Life Expectancy': life_expectancy_2020[0]
    })

# Step 6: Determine the country with the highest life expectancy
predictions_df = pd.DataFrame(predictions)
top_country = predictions_df.loc[predictions_df['Predicted Life Expectancy'].idxmax(
)]

print(
    f"The country predicted to have the highest life expectancy in 2020 is {top_country['Country']} with a life expectancy of {top_country['Predicted Life Expectancy']:.2f} years.")
