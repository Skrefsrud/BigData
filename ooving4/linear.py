import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_csv('dataset/life-expectancy-data.csv')

# Strip whitespace from column names to avoid KeyError issues
df.columns = df.columns.str.strip()

# Drop rows with missing values in 'Life expectancy' or the predictors
data = df[['Country', 'Year', 'Life expectancy',
           'Schooling', 'BMI', 'GDP']].dropna()

# Prepare data for linear regression with time (using 'Year' as a predictor)
X = data[['Year']]
y = data['Life expectancy']

# Split the data into training and testing sets (train on years before 2015)
X_train = X[X['Year'] < 2015]
y_train = y[X['Year'] < 2015]
X_test = pd.DataFrame({'Year': [2020]})

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict life expectancy for the year 2020
predicted_life_expectancy_2020 = model.predict(X_test)[0]
print(
    f"Predicted average life expectancy in 2020: {predicted_life_expectancy_2020:.2f}")

# Predict life expectancy for each country in 2020 based on trends
predictions = []
for country in data['Country'].unique():
    country_data = data[data['Country'] == country]
    if not country_data.empty:
        # Train a separate model for each country
        X_country = country_data[['Year']]
        y_country = country_data['Life expectancy']
        model.fit(X_country, y_country)
        predicted_life_expectancy = model.predict(X_test)[0]
        predictions.append((country, predicted_life_expectancy))

# Sort predictions to get the top 10 countries with the highest predicted life expectancy
predictions.sort(key=lambda x: x[1], reverse=True)
top_10_countries = predictions[:10]

# Plot the trend for the top 10 countries
plt.figure(figsize=(15, 10))
for country, _ in top_10_countries:
    country_data = data[data['Country'] == country]
    X_country = country_data[['Year']]
    y_country = country_data['Life expectancy']
    model.fit(X_country, y_country)
    years = np.arange(2000, 2021).reshape(-1, 1)
    country_life_expectancy_trend = model.predict(years)
    plt.plot(years, country_life_expectancy_trend,
             linestyle='--', label=f'{country} Trend')

# Find the country with the highest predicted life expectancy in 2020
if top_10_countries:
    best_country = top_10_countries[0]
    # Display the results
    print(
        f"\nCountry likely to have the best life expectancy in 2020: {best_country[0]} with predicted life expectancy: {best_country[1]:.2f}")
else:
    print("\nNo predictions available for the year 2020.")

# Finalize and save the plot
plt.xlabel('Year')
plt.ylabel('Life Expectancy')
plt.title('Predicted Life Expectancy Trends for Top 10 Countries (2000-2020)')
plt.legend()
plt.savefig('life_expectancy_trend_top_10_countries.png')
print("Life expectancy trend plot saved as 'life_expectancy_trend_top_10_countries.png'")
