#example logistic regression
import numpy
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

def balanced_accuracy(cnf_matrix):
    TN = int(cnf_matrix[0][0])
    FP = int(cnf_matrix[0][1])
    FN = int(cnf_matrix[1][0])
    TP = int(cnf_matrix[1][1])
    Sensitivity = TP/(TP+FN)
    Specificity = TN/(TN+FP)
    balanced_accuracy = (Sensitivity+Specificity)/2
    return balanced_accuracy


data_file = "train_var.csv"
data = pd.read_csv(data_file)
data = data.drop(["Unnamed: 0"], axis=1)
print(data.head())

scaler = MinMaxScaler() 
scaled = scaler.fit_transform(data)
scaled_data = pd.DataFrame(scaled, columns=data.columns)

def read_from_file(file):
    f = open(file, "r")
    a = f.read()
    b = a.rsplit()
    return b



def Logistic_Regression(data):
    features = read_from_file("Important_variables.txt")
    X = data[features] # Features
    y = data.label # Target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

    # instantiate the model (using the default parameters)
    logreg = LogisticRegression(random_state=16)

    # fit the model with data
    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)

    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print(balanced_accuracy(cnf_matrix))

Logistic_Regression(data)
Logistic_Regression(scaled_data)



