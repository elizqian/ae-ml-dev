import pandas as pd 
import numpy as np 

# Loading dataframe 
n_samples = 10000
df = pd.read_csv("CDRSamples.csv")
X = df[['A', 'E', 'Ti', 'To', 'phi']].to_numpy()[:n_samples,:].T
Y = df[['T_max']].to_numpy()[:n_samples]

# Converting to numpy matrix (columns are snapshots of data vector)
np.random.seed(42) 
beta = np.random.randn(X.shape[0])*1e-1
Y = X.T @ beta + np.random.randn(Y.shape[0]) * 1


# Creating orthogonal basis 
high_dim = 1000
A = np.random.randn(high_dim, high_dim)
print("Forming QR")
Q, _ = np.linalg.qr(A)

# Only taking the first d components of Q to form a projection up to the high-dimensional space
Q = Q[:,:X.shape[0]]

# Projecting the inputs into high-dimensional space 
X_high = (Q @ X).T 

# Creating X and Y dataframes 
np.save("X.npy", X_high)
np.save("Y.npy", Y)
