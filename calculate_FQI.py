import numpy as np
import argparse
from glob import glob
import os

parser = argparse.ArgumentParser()

parser.add_argument('-data_path', '--data_path', required=True, type=str,
                    help='Data path')

# Parse arguments
args = parser.parse_args()

data_path = args.data_path

print("data_path: ", data_path)
        
# Calculate R2 and FQI
def calculateFQI(im, Errmap):
    SS_im_mean = np.mean(im, axis=0)
    SS_total = np.sum(np.square(im - SS_im_mean), axis=0)
    SS_res = np.square(Errmap)
    R2_map = 1 - SS_res/SS_total
    return np.nanmean(R2_map)

files_path = glob(os.path.join(data_path, "*"))

for file_path in files_path:
    file = np.load(file_path)
    name = file_path.split("\\")[-1].split(".npz")[0]
    
    FQI = calculateFQI(file["im"], file["Errmap"])
    
    print(f'{name} FQI: {FQI}')