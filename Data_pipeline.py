import os
import tensorflow as tf
import h5py
import numpy as np

def preprocess(X):
    return X.astype(np.float32) / 255

def load_data(data_file):
    with h5py.File(data_file, 'r') as f:
        key = list(f.keys())[0]
        X = f[key][:]
    return X
  
def load_source(source_file):
    with h5py.File(source_file, 'r') as f:
        key = list(f.keys())[0]
        sources = f[key][:]
    return sources
