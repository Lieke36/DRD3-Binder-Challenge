import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors

###read and save data from csv file, do not forget to change datafile_name
data_file = "train.csv"
data = pd.read_csv(data_file)
data.head() # shows top 5 rows of data, use this to check that the data loaded correctly
print(data.head())

###shows missing data
"""
missing_data = data.isna()
Amount_Missing_Data = data.isna().sum().sum()
print(Amount_Missing_Data)
"""

###checks for duplication and the amount of duplication:
"""
duplication = data.duplicated()
duplicated_rows = []
for i in range(len(duplication)): 
    if duplication[i] == True:
        duplicated_rows.append(duplication[i])

print(len(duplicated_rows))
if len(duplicated_rows) > 0:
    print(duplicated_rows)
"""

smiles_column_list = data["SMILES_canonical"].tolist()
target_feature_column_list = data["target_feature"].tolist()
#feature_list = []

def dict_from_smiles(smiles_column_list):
    feature_dict = {}
    for smiles in smiles_column_list:
        mol = Chem.MolFromSmiles(smiles)
        mol_describtors = rdkit.Chem.Descriptors.CalcMolDescriptors(mol, missingVal=None, silent=True)
        #feature_list.append(mol_describtors)
        feature_dict[smiles] = mol_describtors
    return feature_dict

feature_dict = dict_from_smiles(smiles_column_list)
#print(smiles_column_list)
#print(feature_dict)

#feature_df = pd.DataFrame(feature_list)
feature_with_mol_df = pd.DataFrame.from_dict(feature_dict).transpose()
print(feature_with_mol_df.head())

"""
PCA + number of components needed for 90% of varience:
"""

data_for_pca = feature_with_mol_df
scaler = MinMaxScaler() 
scaled = scaler.fit_transform(data_for_pca)
scaled_data = pd.DataFrame(scaled, columns=data_for_pca.columns)

#print(scaled[10:20])

###PCA
pca = PCA()
pca.fit(scaled_data)

###find the variance ratio for each component
explained_variance = pca.explained_variance_ratio_

###Find the cumulative variance
cumulative_variance = np.cumsum(explained_variance)

###print the amount of componends with at least 90% of the dataset varience
num_components = np.argmax(cumulative_variance >= 0.90) + 1
print(f"The amount of components with at least 90% variance: {num_components}")


###Plot the varience
plt.figure()
plt.plot(np.arange(1, len(cumulative_variance) + 1), cumulative_variance, color='red')
plt.title('Cumulative Variance by PCA')
plt.xlabel('Amount of Principal Components')
plt.ylabel('Cumulative Variance')
plt.axhline(y=0.90, color='black', linestyle='--', label='90% Variance Threshold')
plt.legend()

###Show plot
plt.tight_layout()
plt.show()


"""
Find variables with most effect on PCA vectors
"""
###get the principal component vectors
pc_components = pca.components_ 

###get the first and second principal component
pc1_component = pc_components[0]
pc2_component = pc_components[1]
variables = data_for_pca.columns

#print(max(pc1_component))
#index = list(pc1_component).index(max(list(pc1_component)))
#print(index)
#print(variables[index])


###prints most important variables?
def important_variables(pc1_component, pc2_component,variables):
    high_variables = []
    for i in range(len(pc1_component)):
        if pc1_component[i]>0.12 or pc2_component[i]>0.12:
            high_variables.append(variables[i])

    print(high_variables, len(high_variables))

#important_variables(pc1_component, pc2_component,variables)

###plot the two components over the variables
plt.figure()
plt.plot(variables, pc1_component, label = 'pc1', color = 'blue')
plt.plot(variables, pc2_component, label = 'pc2', color = 'black')
plt.legend()
plt.show()


###writes most important variables to a txt-file
def list_to_file(file, l):
    f = open(file, "a")
    for variables in l:
        f.write('%s\n' %variables)
    f.close()

#list_to_file("Important_variables.txt", high_variables)

###reads important variables file
def read_from_file(file):
    f = open(file, "r")
    a = f.read()
    b = a.rsplit()
    return b

important_variables_list = read_from_file("Important_variables.txt")
print(type(variables))
print(type(important_variables_list))
print(type(feature_with_mol_df))



def remove_unwanted_columns_from_data(var_list, wanted_var_list,Data_Frame):
    not_wanted = []
    for var in var_list:
        if var not in wanted_var_list:
            not_wanted.append(var)

    df = Data_Frame.drop(not_wanted, axis=1)
    return df

#feature_with_target_df = feature_with_mol_df.set_axis(target_feature_column_list,axis=0)
new_data_df = remove_unwanted_columns_from_data(variables, important_variables_list, feature_with_mol_df)
new_data_df['label']=target_feature_column_list
new_data_df.to_csv("train_var.csv")
