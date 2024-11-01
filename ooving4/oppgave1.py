import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the dataset
df = pd.read_csv('dataset/life-expectancy-data.csv')

# Map Status column to 0 for Developing and 1 for Developed
df['Status'] = df['Status'].map({'Developing': 0, 'Developed': 1})

# List of predictors we want to test
predictors = ['Alcohol', 'percentage expenditure', ' BMI ', 'Schooling', 'GDP']
results = {}

# Loop through each predictor, create a linear regression model, and evaluate it
for predictor in predictors:
    # Drop rows with missing values in 'Life expectancy ' or the predictor
    data = df[['Life expectancy ', predictor]].dropna()

    # Prepare data for linear regression
    X = data[[predictor]]
    y = data['Life expectancy ']

    # Create and fit the model
    model = LinearRegression()
    model.fit(X, y)

    # Predict and calculate R-squared score
    y_pred = model.predict(X)
    r_squared = r2_score(y, y_pred)
    results[predictor] = r_squared

# Determine the best predictor
best_predictor = max(results, key=results.get)
best_score = results[best_predictor]

# Display the results
results, best_predictor, best_score

print("R-squared values for each predictor:")
for predictor, r2 in results.items():
    print(f"{predictor}: {r2:.4f}")

print(
    f"\nBest predictor: {best_predictor} with R-squared value: {best_score:.4f}")


# Predict life expectancy in 2020 using the best predictor
# Filter data for the best predictor and drop missing values
data = df[['Country', 'Year', 'Life expectancy ', best_predictor]].dropna()

# Use data from years before 2020 to train the model
train_data = data[data['Year'] < 2020]
X_train = train_data[[best_predictor]]
y_train = train_data['Life expectancy ']

# Create and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict life expectancy for each country in 2020
countries_2020 = data[data['Year'] == 2019]['Country'].unique()
predictions = []
for country in countries_2020:
    country_data = data[(data['Country'] == country) & (data['Year'] == 2019)]
    if not country_data.empty:
        X_pred = country_data[[best_predictor]]
        predicted_life_expectancy = model.predict(X_pred)[0]
        predictions.append((country, predicted_life_expectancy))

# Check if predictions list is not empty before finding the country with the highest predicted life expectancy
if predictions:
    best_country = max(predictions, key=lambda x: x[1])
    # Display the results
    print(
        f"\nCountry likely to have the best life expectancy in 2020: {best_country[0]} with predicted life expectancy: {best_country[1]:.2f}")
else:
    print("\nNo predictions available for the year 2020.")
