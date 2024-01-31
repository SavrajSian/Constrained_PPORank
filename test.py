import numpy as np
import os
import pandas as pd

##### print npz to see whats in it #######################################################################

currentdir = os.path.dirname(os.path.realpath(__file__))
npz_path = os.path.join(currentdir, 'preprocess', 'GDSC_ALL')

#read npz file
data = np.load(npz_path + '/GDSC_GEX.npz')

names = {'X': 'GEX', 'Y': 'IC50', 'cell_ids': 'GEX_cell_ids', 'cell_names': 'GEX_cell_names', 'drug_ids': 'IC50_drug_ids', 'drug_names': 'IC50_drug_names', 'GEX_gene_symbols': 'GEX_gene_symbols'}

#read data
for key in data.keys():
    array = data[key]
    df = pd.DataFrame(array)
    print(names[key])
    print(df)

############################################################################################################


