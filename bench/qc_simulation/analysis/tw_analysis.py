import pandas as pd
import glob
import sys
import json
import numpy as np
import pathlib

def main():
    fname_pattern = sys.argv[1]
    fnames = sorted(glob.glob(fname_pattern))
    for fname in fnames:
        data = json.load(open(fname))
        width = data['width']
        slices = np.round(np.log2(data['slices']))
        info = f"{width} +{slices}"
        print(pathlib.Path(fname).name,'\t',  info)

if __name__=="__main__":
    main()

