import numpy as np
import pandas as pd

import os, sys, random, glob
import IPython

in_file, out_file = sys.argv[1], sys.argv[2]
df = pd.read_csv(in_file, sep='\t')
files_list = glob.glob("../files/*")
files_list = [dir_file[9:] for dir_file in files_list]

df = df[~df['id'].isin(files_list)]

df.to_csv(out_file, sep='\t', header=True, index=False, index_label=False)


