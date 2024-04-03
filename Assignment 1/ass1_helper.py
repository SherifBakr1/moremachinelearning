from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import numpy as np
import matplotlib.pyplot as plt

diabetes_data = load_diabetes()

# Set up one plot
fig, ax = plt.subplots(figsize=(15, 10))
fig.suptitle("Linear Regression on Diabetes Dataset")

for selectedFeature in range(len(diabetes_data.feature_names)):
    X = diabetes_data.data[:, selectedFeature].reshape(-1, 1)
    y = diabetes_data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)

    yTrainPred = model.predict(X_train)

    rmse_train = np.sqrt(mean_squared_error(y_train, yTrainPred))

    ax.scatter(X_train, y_train, label=f"Feature: {diabetes_data.feature_names[selectedFeature]} - RMSE: {rmse_train:.2f}")
    ax.plot(X_train, yTrainPred, linewidth=2)

ax.set_xlabel("Features")
ax.set_ylabel("Disease Progression")
ax.legend()
plt.show()
