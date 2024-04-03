import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.linear_model import SGDRegressor, Lasso, Ridge, ElasticNet
from prettytable import PrettyTable
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint as sp_randint
import os

# Ensure you have seaborn installed: pip install seaborn

def main():
    num_cpus= os.cpu_count()
    print(f"Number of CPUs: {num_cpus}")

    # Load the dataset
    df = pd.read_csv('c:\\Users\\Sheri\\Documents\\COMP4432\\Assignment 4\\bike_share_hour.csv')

    # Convert categorical columns to "category" dtype
    categorical_columns = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
    for col in categorical_columns:
        df[col] = df[col].astype('category')

    # Check for non-null values in the dataset
    print("Non-null counts in each column:")
    print(df.notnull().sum())

    # Descriptive analysis of numeric columns
    print("\nDescriptive analysis of numeric columns:")
    print(df.describe())

    # Implement a bar plot of cnt versus season
    plt.figure(figsize=(10, 6))
    sns.barplot(x='season', y='cnt', data=df, estimator=sum)
    plt.title('Total Bike Rides per Season')
    plt.xlabel('Season')
    plt.ylabel('Total Count of Bike Rides')
    plt.xticks(ticks=[0, 1, 2, 3], labels=['Spring', 'Summer', 'Fall', 'Winter'])
    plt.show()
    # The fall has the most bike rides, and the spring has the least. 

    # Implement a bar chart for working day versus count
    plt.figure(figsize=(10, 6))
    sns.barplot(x='workingday', y='cnt', data=df, estimator=sum)
    plt.title('Bike Rides Distribution Across Working Days')
    plt.xlabel('Working Day')
    plt.ylabel('Count of Bike Rides')
    plt.show()
    #1 The  bike rides are distributed as the below:
    #1 (True): Represents a working day, which is any day that is not a weekend or holiday. 
    #0 (False): Represents a non-working day, which includes weekends (Saturday and Sunday) and holidays as defined by the dataset's readme file.

    # Implement a bar chart for month versus count
    plt.figure(figsize=(12, 6))
    sns.barplot(x='mnth', y='cnt', data=df, estimator=sum)
    plt.title('Total Bike Rides per Month')
    plt.xlabel('Month')
    plt.ylabel('Total Count of Bike Rides')
    plt.show()
    # August has the most bike rides. 

    # Implement a bar plot of weathersit versus cnt
    plt.figure(figsize=(10, 6))
    sns.barplot(x='weathersit', y='cnt', data=df, estimator=sum)
    plt.title('Total Bike Rides per Weather Situation')
    plt.xlabel('Weather Situation')
    plt.ylabel('Total Count of Bike Rides')
    plt.show()
    #4: Weather situation 4 (Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog) has the least bike rentals


    # Implement a point plot of weathersit on the x-axis, count on the y-axis, and the season as the hue
    plt.figure(figsize=(12, 6))
    sns.pointplot(x='weathersit', y='cnt', hue='season', data=df, estimator=sum)
    plt.title('Count of Bike Rides by Weather Situation and Season')
    plt.xlabel('Weather Situation')
    plt.ylabel('Total Count of Bike Rides')
    plt.show()
    # In all four seasons, the number of rides decreases as the weather situation changes from 1 to 4. 

    # Implement a bar plot of hour versus count
    plt.figure(figsize=(12, 6))
    sns.barplot(x='hr', y='cnt', data=df, estimator=sum)
    plt.title('Total Bike Rides per Hour')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Total Count of Bike Rides')
    plt.show()
    # Yes, during the times when employees are going to work (7-8 PM) and after the working hours (5-6 PM) are busy times. 

    # Implement a bar plot of hour versus count on weekends and holidays (when workingday = 0)
    df_nonworking = df[df['workingday'] == 0]
    plt.figure(figsize=(12, 6))
    sns.barplot(x='hr', y='cnt', data=df_nonworking, estimator=sum)
    plt.title('Total Bike Rides per Hour on Non-Working Days')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Total Count of Bike Rides')
    plt.show()
    # Yes! The hourly trend changes on the weekend, with the busiest times are from 11 AM to 3 PM. 


    ######## PART 2 ##

     # Correlation matrix of numeric features
    numeric_features = ['temp', 'atemp', 'hum', 'windspeed', 'cnt']
    correlation_matrix = df[numeric_features].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix of Numeric Features')
    plt.show()

    # Scale numerical features
    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    # Drop specified columns
    df.drop(columns=['casual', 'registered', 'dteday', 'instant'], inplace=True)

    # Histogram of the count column
    plt.figure(figsize=(10, 6))
    sns.histplot(df['cnt'], bins=30, kde=True)
    plt.title('Distribution of Bike Rental Counts')
    plt.xlabel('Normalized Count of Bike Rentals')
    plt.ylabel('Frequency')
    plt.show()
    # That the frequency polynomially decreases as the count of bike rentals increaes.


    # Split data into features and target
    X = df.drop('cnt', axis=1)
    y = df['cnt']

    # Convert categorical columns to dummies
    X = pd.get_dummies(X, drop_first=True)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Baseline linear regression algorithm
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    # Cross-validation scores
    cv_r2_scores = cross_val_score(lr, X_train, y_train, cv=5, scoring='r2')
    cv_mse_scores = cross_val_score(lr, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    
    print(f"Average R2 across 5 folds: {np.mean(cv_r2_scores):.4f}")
    print(f"Average MSE across 5 folds: {np.mean(cv_mse_scores):.4f}")
    print(f"RMSE: {np.sqrt(-np.mean(cv_mse_scores)):.4f}")

    ######### Part 3 ##
    results_table = PrettyTable()
    results_table.field_names = ["Model", "R2", "MSE", "RMSE"]

    # Models to evaluate
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=0),
        "Random Forest": RandomForestRegressor(random_state=0, n_estimators=30),
        "SGD Regressor": SGDRegressor(max_iter=1000, tol=1e-3),
        "Lasso": Lasso(alpha=0.1),
        "ElasticNet": ElasticNet(random_state=0),
        "Ridge": Ridge(alpha=0.5),
        "Bagging Regressor": BaggingRegressor(random_state=0)
    }

    # Evaluate each model
    for name, model in models.items():
        evaluate_model(model, X_train, y_train, X_test, y_test, name, results_table)

    # Print the results
    print(results_table)
    
    ########### Part 4 ##
    top_models = {
        "Decision Tree": DecisionTreeRegressor(random_state=0),
        "Random Forest": RandomForestRegressor(random_state=0, n_estimators=30),
        "Bagging Regressor": BaggingRegressor(random_state=0)
    }

    for name, model in top_models.items():
        print(f"Evaluating {name} with cross-validation...")
        cv_r2_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        cv_mse_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-cv_mse_scores)
        print(f"{name} - Average R2: {np.mean(cv_r2_scores):.4f}, Average RMSE: {np.mean(rmse_scores):.4f}")

    # Hyperparameter tuning 
    param_dist = {
        "bootstrap": [True, False],
        "max_depth": sp_randint(10, 110),
        "max_features": [1.0, 'sqrt'],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "n_estimators": sp_randint(200, 2000)
    }

    random_search = RandomizedSearchCV(RandomForestRegressor(random_state=0),
                                        param_distributions=param_dist,
                                        n_iter=20, cv=3, random_state=0, n_jobs=11)
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    print(f"Best parameters: {random_search.best_params_}")

    # Evaluate the best model from RandomizedSearchCV
    best_model_cv_r2_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2')
    best_model_cv_mse_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    best_model_rmse_scores = np.sqrt(-best_model_cv_mse_scores)
    print(f"Best Model - Average R2: {np.mean(best_model_cv_r2_scores):.4f}, Average RMSE: {np.mean(best_model_rmse_scores):.4f}")

    # Predictions on the test set with the best model
    y_pred = best_model.predict(X_test)
    final_r2 = r2_score(y_test, y_pred)
    final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Test Set - R2 Score: {final_r2:.4f}, RMSE: {final_rmse:.4f}")


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name, results_table):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = model.score(X_test, y_test)
    
    # Add results to the table
    results_table.add_row([model_name, f"{r2:.4f}", f"{mse:.4f}", f"{rmse:.4f}"])


if __name__ == "__main__":
    main()
