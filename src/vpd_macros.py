import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vpd import _cpp

def create_heat_random_fourier_features(n, axis_dim, resolution, R=100, tau=1, mask=None, seed=42):
    return _cpp.Heat_RFF(n,axis_dim,resolution,R,tau,mask,seed)

if __name__ == "__main__":
    print("Python is searching in these folders:", sys.path)
    print(create_heat_random_fourier_features(n=2, axis_dim=1, resolution=100, R=20, tau=1, mask=None, seed=42))
