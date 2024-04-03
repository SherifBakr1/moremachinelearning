# Sherif Bakr                # COMP4432          #  02/11/2024

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import numpy as np

#Loading the Titanic dataset
titanic = sns.load_dataset('titanic')

missing_data = titanic.isnull().sum()
print("Missing data count per column:\n", missing_data)

categorical_columns = titanic.select_dtypes(include=['object', 'category']).columns
print("\nCategorical columns:\n", categorical_columns)

plt.figure(figsize=(10, 6))
sns.heatmap(titanic.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Data Heatmap")
plt.show()

deaths_by_gender = titanic[titanic['survived'] == 0]['sex'].value_counts()

survival_by_class = titanic.groupby('pclass')['survived'].mean()

plt.figure(figsize=(10, 6))
sns.histplot(titanic['fare'], kde=True, bins=30)
plt.title("Distribution of Fare")
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(titanic[titanic['age'].notnull()]['age'], kde=True, bins=30)
plt.title("Distribution of Age")
plt.show()

median_age_by_class = titanic.groupby('pclass')['age'].median()

plt.figure(figsize=(10, 6))
sns.boxplot(x='pclass', y='age', data=titanic)
plt.title("Median Age by Passenger Class")
plt.show()

print("Deaths by gender:\n", deaths_by_gender)
print("\nSurvival rate by passenger class:\n", survival_by_class)
print("\nMedian age by passenger class:\n", median_age_by_class)

#### Part 2 ####

titanic.drop('deck', axis=1, inplace=True)

# Function to impute age
def impute_age(cols):
    age = cols.iloc[0]
    pclass = cols.iloc[1]
    
    if pd.isnull(age):
        if pclass == 1:
            return median_age_by_class.loc[1]
        elif pclass == 2:
            return median_age_by_class.loc[2]
        else:
            return median_age_by_class.loc[3]
    else:
        return age


titanic['age'] = titanic[['age', 'pclass']].apply(impute_age, axis=1)

titanic.dropna(inplace=True)

assert titanic.isnull().sum().sum() == 0

categorical_cols = ['sex', 'embarked', 'class', 'who', 'adult_male', 'embark_town', 'alive', 'alone']
titanic = pd.get_dummies(titanic, columns=categorical_cols, drop_first=True)

X = titanic.drop('survived', axis=1)

# Implementing a label dataframe
y = titanic['survived']

# Splitting into training and test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


########### Part 3 ###


log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)

svc = SVC(probability=True)
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)

sgd = SGDClassifier(loss='log_loss')  # loss parameter
sgd.fit(X_train, y_train)
y_pred_sgd = sgd.predict(X_test)

# Evaluation
models = [('Logistic Regression', log_reg, y_pred_log_reg),
          ('Support Vector Classifier', svc, y_pred_svc),
          ('Stochastic Gradient Descent', sgd, y_pred_sgd)]

for name, model, y_pred in models:
    print(f"{name} - Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"{name} - Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # ROC and AUC
    if name == 'Stochastic Gradient Descent':
        y_scores = cross_val_predict(model, X_test, y_test, cv=3, method="decision_function")
    else:
        y_probas = model.predict_proba(X_test)[:, 1]
        y_scores = y_probas  # Use probability estimates for ROC curve for logistic regression and SVC
    
    roc_auc = roc_auc_score(y_test, y_scores)
    print(f"{name} - ROC AUC Score:", roc_auc)
    
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    plt.plot(fpr, tpr, label=f"{name} (auc = {roc_auc:.2f})")
    
plt.plot([0, 1], [0, 1], 'k--')  
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()

##### Part 4 ##

# Define the pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(probability=True))
])

param_grid = {
    'svc__kernel': ['rbf'],
    'svc__gamma': [0.0001, 0.001, 0.01, 0.1, 1],
    'svc__C': [1, 10, 50, 100, 200, 300]
}

# Grid search
grid_search = GridSearchCV(pipe, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Printing the best estimator, its parameters, and the resulting score
print("Best Estimator:", grid_search.best_estimator_)
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Applying the best estimator to the test set
y_pred_best_svc = grid_search.predict(X_test)
print("Classification Report for Best Estimator:")
print(classification_report(y_test, y_pred_best_svc))

train_sizes, train_scores, test_scores = learning_curve(
    grid_search.best_estimator_, X_train, y_train, cv=5, n_jobs=-1, 
    train_sizes=np.linspace(.1, 1.0, 5))

# Calculate mean and standard deviation for training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Calculate mean and standard deviation for test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot learning curve
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="r", alpha=0.1)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="g", alpha=0.1)
plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")

plt.title("Learning Curve")
plt.xlabel("Training Examples")
plt.ylabel("Score")
plt.legend(loc="best")
plt.show()

# What does this learning curve tell you?

## The learning curve shows how model performance varies with more training data. 
# High and converging training and validation scores imply good generalization. 
# A big gap between these scores indicates overfitting, while low scores for both suggest underfitting.





