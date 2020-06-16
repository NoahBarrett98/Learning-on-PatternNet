import scipy.io
import pandas as pd
import sys
import numpy as np
"""
Script to convert .mat fikles to csv
saves to same dir
"""

TO_CONVERT = sys.argv[1]
if TO_CONVERT == "":
    raise ValueError

def convert():
    mat = scipy.io.loadmat(TO_CONVERT)
    #mat = {k:v for k, v in mat.items() if k[0] != '_'}
    #data = pd.DataFrame({k: pd.Series(v[0]) for k, v in zip(mat.keys(), mat.items())})
    #data.to_csv(TO_CONVERT[:-4] +".csv")
    #for k in mat.keys():
      # print(k, len(mat[k]))
    print(np.array(mat['paviaU']).shape())
convert()