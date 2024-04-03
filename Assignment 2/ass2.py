# Sherif Bakr               COMP4432                Assignment 2

from sklearn.datasets import load_diabetes
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import numpy as np
import joblib



diabetes = load_diabetes()
print(diabetes.DESCR)
diabetes_df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
diabetes_df['progression'] = diabetes.target
print(diabetes_df.info())
print(diabetes_df.describe())
sns.set()
diabetes_df.hist(figsize=(15, 10), bins=20)
plt.tight_layout()
plt.show()
train_df, test_df = train_test_split(diabetes_df, test_size=0.2, random_state=42)
correlation_matrix = train_df.corr()
print(correlation_matrix['progression'].sort_values(ascending=False)) # Display  correlation values with target variable 'progression'
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()

columns_of_interest = ['bmi', 's5', 'bp', 'progression']

# Use a Seaborn pairplot to look at the scatter plots of these columns
sns.pairplot(train_df[columns_of_interest])
plt.show()

# Prepare a feature set by dropping the target from your training dataframe
X_train = train_df.drop('progression', axis=1)

# Copy your training target into a new dataframe
y_train = train_df['progression']

# Check
# print(X_train.head())
# print(y_train.head())

# Train a linear regression model using the training set
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Predictions on the training set
y_train_pred = lin_reg.predict(X_train)

# Calculate RMSE for the linear regression model on the training set
lin_reg_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
print("Linear Regression RMSE on Training Set:", lin_reg_rmse)

# Implement cross_val_score on a decision tree regressor using the training set
decision_tree_reg = DecisionTreeRegressor(random_state=42)
dt_scores = cross_val_score(decision_tree_reg, X_train, y_train, 
                            scoring='neg_mean_squared_error', cv=10)
dt_rmse_scores = np.sqrt(-dt_scores)
dt_rmse_mean = dt_rmse_scores.mean()
dt_rmse_std = dt_rmse_scores.std()
print("Decision Tree Regressor CV RMSE Mean:", dt_rmse_mean)
print("Decision Tree Regressor CV RMSE Std:", dt_rmse_std)

# Implement cross_val_score on a RandomForestRegressor using the training set
random_forest_reg = RandomForestRegressor(random_state=42)
rf_scores = cross_val_score(random_forest_reg, X_train, y_train, 
                            scoring='neg_mean_squared_error', cv=10)
rf_rmse_scores = np.sqrt(-rf_scores)
rf_rmse_mean = rf_rmse_scores.mean()
rf_rmse_std = rf_rmse_scores.std()
print("Random Forest Regressor CV RMSE Mean:", rf_rmse_mean)
print("Random Forest Regressor CV RMSE Std:", rf_rmse_std)

# Record which model performs better
best_model = "Linear Regression" if lin_reg_rmse < min(dt_rmse_mean, rf_rmse_mean) else "Random Forest" if rf_rmse_mean < dt_rmse_mean else "Decision Tree"
print("Best performing model:", best_model)

####################

print("Parameters of Random Forest Model:", random_forest_reg.get_params())

# Grid search cross-validation parameters
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
]

# Perform grid search cross-validation
grid_search = GridSearchCV(random_forest_reg, param_grid, cv=10,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(X_train, y_train)

# Print out the best parameters and the best performing model based on this grid search
print("Best Parameters:", grid_search.best_params_)
print("Best Estimator:", grid_search.best_estimator_)

# Using the cv_results dictionary, print out the RMSE of each feature combination for comparison
cv_results = grid_search.cv_results_
for mean_score, params in zip(cv_results["mean_test_score"], cv_results["params"]):
    print(np.sqrt(-mean_score), params)

# Print out the feature importances of the best performing grid search model
best_model = grid_search.best_estimator_
print("Feature Importances:", best_model.feature_importances_)

print("\nCorrelation Matrix Analysis:")
print("Feature importances in the RandomForestRegressor reflect how significantly each feature influences the model's predictive outcomes. The correlation matrix illustrates the direct linear relationships between each feature and the 'progression' target variable.")
print("A high importance of a feature in the model doesn't always correspond to a strong linear correlation, since random forests can detect non-linear patterns.")


X_test = test_df.drop('progression', axis=1)
y_test = test_df['progression']
y_test_pred = lin_reg.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("Linear Regression RMSE on Test Set:", rmse_test)

# Save the model for future use
joblib.dump(lin_reg, 'linear_regression_model.pkl')

