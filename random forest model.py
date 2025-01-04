#https://www.datacamp.com/tutorial/random-forests-classifier-python

# Data Processing
import pandas as pd
import numpy as np

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.neighbors import KNeighborsClassifier


# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz
import pandas

data_file = "train_var.csv"
data_good = pd.read_csv(data_file)
print(data_good.head())

test_file = 'test_data_variables.csv'
test_good = pd.read_csv(data_file)


def read_from_file(file):
    f = open(file, "r")
    a = f.read()
    b = a.rsplit()
    return b

features = read_from_file("Important_variables.txt")

X = data_good[features] # Features
y = data_good.label # Target variable

X_testdata= test_good[features]
Y_testdata = test_good.label

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,20)}

# Create a random forest classifier
rf = RandomForestClassifier()

# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(rf, 
                                 param_distributions = param_dist, 
                                 n_iter=5, 
                                 cv=5)

# Fit the random search object to the data
rand_search.fit(X_train, y_train)


# Create a variable for the best model
best_rf = rand_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  rand_search.best_params_)

model = RandomForestClassifier(n_estimators = 400, max_depth=12)
model.fit(X_train, y_train)
val_preds = model.predict(X_testdata)

test_preds = model.predict('target_feature')

submission = pd.DataFrame({
    'Unique_ID': test_data.loc[valid_test_indices, 'Unique_ID'].reset_index(drop = True),
    'target_feature': pd.Series(test_preds, name='target_feature')
})

submission_path = 'submission.csv'
submission.to_csv(submission_path, index = False, sep = ',' )
# y_pred=rand_search(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)

'''knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# Create a series containing feature importances from the model and feature names from the training data
feature_importances = pd.Series(best_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# Plot a simple bar chart
feature_importances.plot.bar()'''