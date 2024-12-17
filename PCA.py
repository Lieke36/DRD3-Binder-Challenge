#packages for PCA:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

#packages for data:
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors


"""
PCA + number of components needed for 90% of varience
"""

#data = panda frame () #does the pandaframe have the target values?
scaler = MinMaxScaler() 
scaled = scaler.fit_transform(data)
scaled_data = pd.DataFrame(scaled, columns=data.columns)

#PCA
pca = PCA()
pca.fit(scaled_data)

#find the variance ratio for each component
explained_variance = pca.explained_variance_ratio_

#Find the cumulative variance
cumulative_variance = np.cumsum(explained_variance)

#print the amount of componends with at least 90% of the dataset varience
num_components = np.argmax(cumulative_variance >= 0.90) + 1
print(f"The amount of components with at least 90% variance: {num_components}")

# Plot the varience
plt.figure()
plt.plot(np.arange(1, len(cumulative_variance) + 1), cumulative_variance, color='red')
plt.title('Cumulative Variance by PCA')
plt.xlabel('Amount of Principal Components')
plt.ylabel('Cumulative Variance')
plt.axhline(y=0.90, color='black', linestyle='--', label='90% Variance Threshold')
plt.legend()

# Show plot
plt.tight_layout()
plt.show()
#-----------------------------------------------------------------------------------------
"""
Find variables with most effect on PCA vectors
"""
#get the principal component vectors
pc_components = pca.components_ 

#get the first and second principal component
pc1_component = pc_components[0]
pc2_component = pc_components[1]
variables = data.columns

#plot the two components over the variables
plt.figure()
plt.plot(variables, pc1_component, label = 'pc1', color = 'blue')
plt.plot(variables, pc2_component, label = 'pc2',color = 'black')
plt.legend()
plt.show