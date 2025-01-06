# Data Processing
import pandas as pd

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Read the train data
data_file = "train_var.csv"
data_good = pd.read_csv(data_file)

# Read the test data
test_file = 'test_data_variables.csv'
test_good = pd.read_csv(test_file)

# Function to read a file
def read_from_file(file):
    f = open(file, "r")
    a = f.read()
    b = a.rsplit()
    return b

# Most important variables of the molecules
features = read_from_file("Important_variables.txt")


X = data_good[features] # Features
y = data_good['label'] # Target variable

X_testdata= test_good[features] #Features


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# The Random Forest model 
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

# The accuracy of the model
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
test_preds = model.predict(X_testdata)
print(test_preds)


# Save the result to a csv file
Unique_ID = list(range(1,6234))
Submission_forest = pd.DataFrame(test_preds, index=Unique_ID, columns=['target_feature'])
Submission_forest.index.name = 'Unique_ID'
submission_path = 'submission.csv'
Submission_forest.to_csv(submission_path)
print(f"Submission file created: {submission_path}")
