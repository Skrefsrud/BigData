import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn import tree
import os
import matplotlib
import numpy as np

# Set backend for matplotlib to avoid Qt errors
matplotlib.use('Agg')

# Load the dataset
df = pd.read_csv('dataset/life-expectancy-data.csv')

# Strip whitespace from column names to avoid KeyError issues
df.columns = df.columns.str.strip()

# List of predictors we want to use based on previous analysis
predictors = ['Schooling', 'BMI', 'GDP']

# Drop rows with missing values in 'Life expectancy' or the predictors
data = df[['Country', 'Year', 'Life expectancy'] + predictors].dropna()

# Prepare data for training the decision tree model
X = data[predictors]
y = data['Life expectancy']

# Split the data into training and testing sets
X_train = data[data['Year'] < 2015][predictors]
y_train = data[data['Year'] < 2015]['Life expectancy']

# Create and fit the decision tree model
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict life expectancy for each country in 2020 (using 2015 data as reference)
countries_2015 = data[data['Year'] == 2015]['Country'].unique()
predictions = []
for country in countries_2015:
    country_data = data[(data['Country'] == country) & (data['Year'] == 2015)]
    if not country_data.empty:
        X_pred = country_data[predictors]
        predicted_life_expectancy = model.predict(X_pred)[0]
        predictions.append((country, predicted_life_expectancy))

# Find the country with the highest predicted life expectancy in 2020
if predictions:
    best_country = max(predictions, key=lambda x: x[1])
    # Display the results
    print(
        f"\nCountry likely to have the best life expectancy in 2020: {best_country[0]} with predicted life expectancy: {best_country[1]:.2f}")
else:
    print("\nNo predictions available for the year 2020.")

# Save the decision tree plot to a file
plt.figure(figsize=(20, 10))
tree.plot_tree(model, feature_names=predictors, filled=True, rounded=True)
plt.title('Decision Tree for Predicting Life Expectancy')
plt.savefig('decision_tree_plot.png')
print("Decision tree plot saved as 'decision_tree_plot.png'")
