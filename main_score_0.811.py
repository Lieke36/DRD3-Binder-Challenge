import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

#%%
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

#%%
def compute_descriptors(smiles_list):
    descriptors = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            descriptor = [Descriptors.MolWt(mol),
                          Descriptors.TPSA(mol),
                          Descriptors.MolLogP(mol),
                          Descriptors.NumHAcceptors(mol),
                          Descriptors.NumHDonors(mol)]
            descriptors.append(descriptor)
        else:
            descriptors.append([np.nan] * 5)
    return pd.DataFrame(descriptors, columns=["MolWt", "TPSA", "MolLogP", "NumHAcceptors", "NumHDonors"])

#%%
train_descriptors = compute_descriptors(train_data['SMILES_canonical'])
test_descriptors = compute_descriptors(test_data['SMILES_canonical'])

#%%
valid_train_indices = train_descriptors.dropna().index
train_descriptors = train_descriptors.dropna()
train_labels = train_data.loc[valid_train_indices, 'target_feature']

valid_test_indices = test_descriptors.dropna().index
test_descriptors = test_descriptors.dropna()

#%%
X_train, X_val, y_train, y_val = train_test_split(train_descriptors, train_labels, test_size = 0.2, random_state = 42)

model = RandomForestClassifier(n_estimators = 100, random_state = 42)
model.fit(X_train, y_train)

val_preds = model.predict(X_val)
val_score = balanced_accuracy_score(y_val, val_preds)

test_preds = model.predict(test_descriptors)

#%%
submission = pd.DataFrame({
    'Unique_ID': test_data.loc[valid_test_indices, 'Unique_ID'].reset_index(drop = True),
    'target_feature': pd.Series(test_preds, name='target_feature')
})

submission_path = 'submission.csv'
submission.to_csv(submission_path, index = False, sep = ',' )

print(f"Submission file created: {submission_path}")
