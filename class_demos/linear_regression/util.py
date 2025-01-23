import numpy as np 
import math 
import pandas as pd
from datetime import datetime, timedelta 

def get_data(N, random_seed = 42, noise_var = 0.0):
    np.random.seed(random_seed)

    def Cd(alpha):
        return 0.1 * np.exp(math.pi / 180.0 * alpha)

    def Fd(alpha, rho, v):
        C_d = Cd(alpha)
        A = 1.2 # m^2 
        return 0.5*rho*alpha*v**2*C_d +1

    alpha = np.random.rand(N)*45
    rho = np.random.randn(N)*0.3 + 1.293 
    v = np.random.rand(N)*100
    F_d = Fd(alpha, rho, v) + 0*np.random.randn(N)

    df_dict = {
        "alpha":alpha,
        "rho":rho,
        "velocity":v, 
        "f_drag":F_d
    }

    return pd.DataFrame(df_dict)
