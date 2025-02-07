import pandas as pd 
import numpy as np 


def get_data(N, random_seed = 42):
    np.random.seed(random_seed)

    g = 9.81 

    P1 = np.random.uniform(0.1, 1e5, size=(N,1))
    rho = np.random.uniform(0.1, 2.0, size=(N,1))
    v1 = np.random.uniform(0.1, 100.0,size=(N,1))
    v2 = np.random.uniform(0.1, 100.0, size=(N,1))
    dh = np.random.uniform(-1.0, 1.0, size=(N,1))

    P2 = (P1 - 0.5*rho*(v1**2 - v2**2) + rho * g * dh + 5e3*np.random.randn(P1.shape[0], 1))/1e3

    df= pd.DataFrame(data=np.hstack((P2, P1, rho, v1, v2)))

    df.columns = ['P2', 'P1', 'rho', 'v1', 'v2']

    return df


N_datapoints = 50
df = get_data(N_datapoints, random_seed = 42)

