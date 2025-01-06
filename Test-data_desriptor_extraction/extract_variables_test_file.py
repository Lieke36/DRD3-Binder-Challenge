#gets values of important variables of the test file (to use for the prediction)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors

# read and save data from csv file, do not forget to change datafile_name
data_file = "test.csv"
data = pd.read_csv(data_file)
data.head() # shows top 5 rows of data, use this to check that the data loaded correctly
print(data.head())

#get all smiles in a list:
smiles_column_list = data["SMILES_canonical"].tolist()

def dict_from_smiles(smiles_column_list):
    feature_dict = {}
    for smiles in smiles_column_list:
        mol = Chem.MolFromSmiles(smiles)
        mol_describtors = rdkit.Chem.Descriptors.CalcMolDescriptors(mol, missingVal=None, silent=True)
        #feature_list.append(mol_describtors)
        feature_dict[smiles] = mol_describtors
    return feature_dict

feature_dict = dict_from_smiles(smiles_column_list)
feature_with_mol_df = pd.DataFrame.from_dict(feature_dict).transpose()
variables = feature_with_mol_df.columns

def read_from_file(file):
    f = open(file, "r")
    a = f.read()
    b = a.rsplit()
    return b

def remove_unwanted_columns_from_data(var_list, wanted_var_list,Data_Frame):
    not_wanted = []
    for var in var_list:
        if var not in wanted_var_list:
            not_wanted.append(var)

    df = Data_Frame.drop(not_wanted, axis=1)
    return df

important_variables_list = read_from_file("Important_variables.txt")
new_test_data_df = remove_unwanted_columns_from_data(variables, important_variables_list, feature_with_mol_df)
new_test_data_df.to_csv("test_data_variables.csv")
