#example logistic regression
import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

data_file = "train_var.csv"
data = pd.read_csv(data_file)
print(data.head())

def read_from_file(file):
    f = open(file, "r")
    a = f.read()
    b = a.rsplit()
    return b

features = read_from_file("Important_variables.txt")

X = data[features] # Features
y = data.label # Target variable


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)
print(X_train.head())
'''
logr = linear_model.LogisticRegression()
logr.fit(X,y)

#predict if tumor is cancerous where the size is 3.46mm:
predicted = logr.predict(numpy.array([3.46]).reshape(-1,1))
print(predicted)
'''


