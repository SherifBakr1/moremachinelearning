# Sherif Bakr                       COMP 4432                   Assignment 1

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score

import numpy as np
import matplotlib.pyplot as plt

diabetes_data = load_diabetes()

selectedFeature = 2
print("Feature Name:", diabetes_data.feature_names[2]) #prints bmi 

# I selected body mass index because it has the lowest RMSE among all the features.
# I compared bmi amongst the features age, sex, bp. The plot can be seen in the other file
# ass1_helper.py. The results for the RMSE were as follows:
# age: 75.72
# sex: 77.07
# bmi: 62.89
# bp: 69.35

X = diabetes_data.data[:, selectedFeature].reshape(-1, 1)
y = diabetes_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, shuffle=False)

model = LinearRegression()              #instance of the lin reg model
model.fit(X_train, y_train)             # training using training sets 
print("Coefficient: ", model.coef_[0])
print("Intercept: ", model.intercept_)

yTrainPred = model.predict(X_train)             # predictions

print("First 10 predictions on the training set :")                         #Printing the first 10 predictions
for i in range(10):
    print(f"Sample {i + 1}: Predicted={yTrainPred[i]:.2f}, Actual={y_train[i]}")

print("\n")          
print("Feature Coefficient:", model.coef_[0])       # printing feature coeff. 
print("Intercept:", model.intercept_)               # printing intercept

rmse_train = np.sqrt(mean_squared_error(y_train, yTrainPred))    # printing rmse
print("\nRoot Mean Squared Error:", rmse_train)         

plt.scatter(X_train, y_train, color='blue', label='Actual Progression (true)')                  # Actual data
plt.plot(X_train, yTrainPred, color='red', linewidth=2, label='Regression Line (pred)')         # Predictions

plt.xlabel(f"Feature: {diabetes_data.feature_names[selectedFeature]}")
plt.ylabel("Disease Progression")
plt.title("Linear Regression on Diabetes Dataset")

plt.legend()
plt.show()



