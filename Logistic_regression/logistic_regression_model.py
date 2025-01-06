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

final_testing_data_file = "test_data_variables.csv"
final_testing_data = pd.read_csv(final_testing_data_file)
final_testing_data = final_testing_data.drop(["Unnamed: 0"], axis=1)
print(final_testing_data.head())

scaler = MinMaxScaler() 
scaled = scaler.fit_transform(data)
scaled_data = pd.DataFrame(scaled, columns=data.columns)

def read_from_file(file):
    f = open(file, "r")
    a = f.read()
    b = a.rsplit()
    return b


def Logistic_Regression(data, testing_data):
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

    logregfinal = LogisticRegression(random_state=16)
    logregfinal.fit(X,y)

    final_X = testing_data[features]
    final_y_pred = logregfinal.predict(final_X)
    return final_y_pred

Result_LogReg = Logistic_Regression(data, final_testing_data)
print(Result_LogReg)
#Logistic_Regression(scaled_data)

Unique_ID = list(range(1,6234))

Submission_LogReg = pd.DataFrame(Result_LogReg, index=Unique_ID, columns=['target_feature'])
Submission_LogReg.index.name = 'Unique_ID'

print(Submission_LogReg.head())

Submission_LogReg.to_csv("submission_LogReg.csv")