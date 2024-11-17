import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# cleaning of the data

# Load the dataset
file_path = 'employee-status.csv'
data = pd.read_csv(file_path)

# Step 1: Remove irrelevant columns
data_cleaned = data.drop(
    columns=['last_evaluation', 'n_projects', 'recently_promoted'])

# Step 2: Drop rows with missing values in critical columns
data_cleaned = data_cleaned.dropna(
    subset=['satisfaction', 'avg_monthly_hrs', 'department'])

# Step 3: Standardize department names
data_cleaned['department'] = data_cleaned['department'].replace({
    'IT': 'information_technology',
    'admin': 'administration'
})

# Step 4: Remove duplicate rows
data_cleaned = data_cleaned.drop_duplicates()

# Step 5: Detect and handle outliers in numerical data using the IQR method


def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


data_cleaned = remove_outliers(data_cleaned, 'avg_monthly_hrs')
data_cleaned = remove_outliers(data_cleaned, 'tenure')

# Step 6: Handle missing values in 'filed_complaint'
data_cleaned['filed_complaint'] = data_cleaned['filed_complaint'].fillna(0)

# Step 7: Ensure 'status' has consistent encoding
data_cleaned['status'] = data_cleaned['status'].map({'Left': 0, 'Employed': 1})

# Step 8: Verify numeric ranges and ensure satisfaction is within [0, 1]
data_cleaned = data_cleaned[(data_cleaned['satisfaction'] >= 0) & (
    data_cleaned['satisfaction'] <= 1)]

# Step 9: Verify valid categories in categorical columns
valid_departments = [
    'engineering', 'support', 'sales', 'information_technology',
    'product', 'marketing', 'procurement', 'finance', 'management', 'administration'
]
data_cleaned = data_cleaned[data_cleaned['department'].isin(valid_departments)]

valid_salaries = ['low', 'medium', 'high']
data_cleaned = data_cleaned[data_cleaned['salary'].isin(valid_salaries)]

# --------------------------------------------------------------------
# task a)

# Filter employees who work more than 200 hours per month
filtered_satisfaction_data = data_cleaned[data_cleaned['avg_monthly_hrs'] > 200]

# Calculate the mean satisfaction score for these employees
mean_satisfaction = filtered_satisfaction_data['satisfaction'].mean()

print(
    f"Mean satisfaction score for employees working more than 200 hours/month: {mean_satisfaction:.4f}")
# Mean satisfaction score for employees working more than 200 hours/month: 0.6281

# --------------------------------------------------------------------
# task b)

# Create a histogram for the distribution of avg_monthly_hrs
plt.figure(figsize=(10, 6))  # Set the figure size for the plot
plt.hist(
    data_cleaned['avg_monthly_hrs'],  # The data column to plot
    bins=30,  # Number of bins to divide the range of hours
    edgecolor='black',  # Add black edges around bins for clarity
    alpha=0.7  # Set transparency level for the bars
)
plt.title('Distribution of Average Monthly Hours Worked',
          fontsize=16)  # Add title to the histogram
plt.xlabel('Average Monthly Hours', fontsize=14)  # Label for the x-axis
plt.ylabel('Frequency', fontsize=14)  # Label for the y-axis
# Add grid lines for the y-axis with dashed style
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()  # Display the histogram

# --------------------------------------------------------------------
# task c)

# Step 1: Visualize the relationship with a scatter plot
plt.figure(figsize=(10, 6))  # Set the figure size
sns.scatterplot(
    x=data_cleaned['avg_monthly_hrs'],  # X-axis: Average monthly hours worked
    y=data_cleaned['satisfaction'],  # Y-axis: Employee satisfaction score
    alpha=0.7  # Set transparency for points to handle overlapping
)
plt.title('Scatter Plot of Avg Monthly Hours vs Satisfaction',
          fontsize=16)  # Add title
plt.xlabel('Average Monthly Hours', fontsize=14)  # Label for x-axis
plt.ylabel('Satisfaction Score', fontsize=14)  # Label for y-axis
plt.grid(True)  # Add gridlines for better visualization
plt.show()  # Display the plot

# Step 2: Calculate Pearson correlation coefficient
# Compute the correlation matrix and extract the value between the two variables
correlation = np.corrcoef(
    data_cleaned['avg_monthly_hrs'], data_cleaned['satisfaction'])[0, 1]

# Print the correlation coefficient
print(f"Pearson Correlation Coefficient: {correlation:.4f}")
# Pearson Correlation Coefficient: -0.0325
# - Value ranges:
#    - `+1`: Perfect positive linear relationship.
#    - `1`: Perfect negative linear relationship.
#    - `0`: No linear relationship.
# near zero values indicate a weak or no relationship


# --------------------------------------------------------------------
# task d)
# Step 1: Select features and target for the decision tree

features = data_cleaned[['avg_monthly_hrs', 'filed_complaint',
                         'satisfaction', 'tenure']]  # Predictor variables

target = data_cleaned['status']  # Target variable (0 = Left, 1 = Employed)


# Step 2: Split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(

    features, target, test_size=0.2, random_state=42, stratify=target

)


# Step 3: Create the Decision Tree Classifier

decision_tree = DecisionTreeClassifier(
    random_state=42, min_impurity_decrease=0.01)


# Train the decision tree

decision_tree.fit(X_train, y_train)


# Step 4: Visualize the Decision Tree

plt.figure(figsize=(12, 8))  # Set figure size for better readability

plot_tree(

    decision_tree,

    feature_names=features.columns,  # Column names as feature labels

    class_names=['Left', 'Employed'],  # Class labels for the target

    filled=True,  # Fill nodes with colors to indicate class probabilities

    rounded=True  # Rounded corners for better aesthetics

)

plt.title('Decision Tree for Employee Status',
          fontsize=16)  # Add a title to the plot

plt.show()


# --------------------------------------------------------------------
# Task e) Decision Tree Evaluation

# Evaluate accuracy on training and testing sets for the Decision Tree
train_accuracy = decision_tree.score(X_train, y_train)
test_accuracy = decision_tree.score(X_test, y_test)

print(f"Decision Tree Training Accuracy: {train_accuracy:.4f}")
print(f"Decision Tree Testing Accuracy: {test_accuracy:.4f}")
# Training Accuracy: 0.8965
# Testing Accuracy: 0.8901
# Since the training accuracy is comparable to the testing accuracy, we can assume that the model is not overfitted.


# --------------------------------------------------------------------
# Task f) KNN Classifier Evaluation against descision tree

# KNN

# Step 1: Create and train the KNN classifier with K=8
knn_classifier = KNeighborsClassifier(
    n_neighbors=8)  # Initialize KNN with 8 neighbors
# Train the KNN classifier on the training data
knn_classifier.fit(X_train, y_train)

# Step 2: Make predictions on the test set
y_pred_knn = knn_classifier.predict(X_test)

# Step 3: Compute the confusion matrix for KNN
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)

# Step 4: Manually calculate the F1-score for KNN
tn_knn, fp_knn, fn_knn, tp_knn = conf_matrix_knn.ravel()
precision_knn = tp_knn / (tp_knn + fp_knn)
recall_knn = tp_knn / (tp_knn + fn_knn)
f1_knn_sklearn = f1_score(y_test, y_pred_knn)

print("\nKNN Results:")
print(f"Confusion Matrix:\n{conf_matrix_knn}")
print(f"Sklearn F1-Score: {f1_knn_sklearn:.4f}")
print(f"KNN Accuracy: {knn_classifier.score(X_test, y_test):.4f}")


# Decision Tree

# Compute confusion matrix for the Decision Tree
y_pred_tree = decision_tree.predict(X_test)
conf_matrix_tree = confusion_matrix(y_test, y_pred_tree)

# Calculate F1-Score for Decision Tree
tn_tree, fp_tree, fn_tree, tp_tree = conf_matrix_tree.ravel()
precision_tree = tp_tree / (tp_tree + fp_tree)
recall_tree = tp_tree / (tp_tree + fn_tree)

f1_tree_sklearn = f1_score(y_test, y_pred_tree)

print("\nDecision Tree Results:")
print(f"Confusion Matrix:\n{conf_matrix_tree}")
print(f"Sklearn F1-Score: {f1_tree_sklearn:.4f}")
print(f"Decision Tree Accuracy: {test_accuracy:.4f}")

# --------------------------------------------------------------------
# Comparison Analysis

print("\nComparison of Decision Tree and KNN Classifier:")
print(f"Decision Tree Accuracy: {test_accuracy:.4f}")
print(f"Decision Tree F1-Score: {f1_tree_sklearn:.4f}")
print(f"KNN Accuracy: {knn_classifier.score(X_test, y_test):.4f}")
print(f"KNN F1-Score: {f1_knn_sklearn:.4f}")

if f1_knn_sklearn > f1_tree_sklearn:
    print("KNN outperforms the Decision Tree based on F1-Score.")
else:
    print("Decision Tree outperforms the KNN based on F1-Score.")


# --------------------------------------------------------------------
# task g)
# Reload the dataset
file_path = 'employee-status.csv'
data = pd.read_csv(file_path)

# Clean the data for regression tree analysis, the last cleaning removed the now nessecary n_project that was specified as not relevant in the exam task.
# Step 1: Drop rows with missing values in relevant columns
data_cleaned_reg = data.dropna(
    subset=['satisfaction', 'n_projects', 'tenure', 'avg_monthly_hrs'])

# Step 2: Ensure satisfaction is within the valid range [0, 1]
data_cleaned_reg = data_cleaned_reg[(data_cleaned_reg['satisfaction'] >= 0) & (
    data_cleaned_reg['satisfaction'] <= 1)]

# Step 3: Remove outliers in numerical columns using the IQR method


def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


data_cleaned_reg = remove_outliers(data_cleaned_reg, 'n_projects')
data_cleaned_reg = remove_outliers(data_cleaned_reg, 'tenure')
data_cleaned_reg = remove_outliers(data_cleaned_reg, 'avg_monthly_hrs')

# Step 4: Select features and target for the regression tree
# Predictor variables
features_reg = data_cleaned_reg[['n_projects', 'tenure', 'avg_monthly_hrs']]
# Target variable (satisfaction score)
target_reg = data_cleaned_reg['satisfaction']

# Step 5: Split the dataset into training and testing sets
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    features_reg, target_reg, test_size=0.2, random_state=42
)

# Step 6: Create the Regression Tree
reg_tree = DecisionTreeRegressor(random_state=42, min_impurity_decrease=0.0001)

# Train the regression tree
reg_tree.fit(X_train_reg, y_train_reg)

# Step 7: Find parameters of those most satisfied
predicted_satisfaction = reg_tree.predict(features_reg)
# Find the index of the highest satisfaction
most_satisfied_index = np.argmax(predicted_satisfaction)
# Get the corresponding parameters
most_satisfied_params = features_reg.iloc[most_satisfied_index]

# Visualize the regression tree
plt.figure(figsize=(12, 8))
plot_tree(
    reg_tree,
    feature_names=features_reg.columns,
    filled=True,
    rounded=True
)
plt.title('Regression Tree for Employee Satisfaction', fontsize=16)
plt.show()

# Output the parameters of the most satisfied employee
print("Parameters of the most satisfied employee:", most_satisfied_params)
# Parameters of the most satisfied employee:
# n_projects           4.0
# tenure               5.0
# avg_monthly_hrs    221.0
