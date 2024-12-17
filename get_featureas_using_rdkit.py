#get featureas using rdkit
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors


smiles = "CCO"
mol = Chem.MolFromSmiles(smiles)
mol_describtors = rdkit.Chem.Descriptors.CalcMolDescriptors(mol, missingVal=None, silent=True)
#rdkit.Chem.Descriptors.CalcMolDescriptors(mol, missingVal=None, silent=True)
"""
    calculate the full set of descriptors for a molecule

    Parameters:
    mol (RDKit molecule) –

    missingVal (float, optional) – This will be used if a particular descriptor cannot be calculated

    silent (bool, optional) – if True then exception messages from descriptors will be displayed

    Returns:
    A dictionary with decriptor names as keys and the descriptor values as values

    Return type:
    dict
"""
print(mol_describtors)
list_dict_molecules = []
df_features = pd.DataFrame(list_dict_molecules) #puts the features in the dictionary of the molecule in a pandas table
