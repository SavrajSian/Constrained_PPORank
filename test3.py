import numpy as np
import pandas as pd
file = 'temp2/ppo_0.npz'

data = np.load(file)
Y_true = data['Y_true']
Y_pred = data['Y_pred']


true_df = pd.DataFrame(Y_true)
pred_df = pd.DataFrame(Y_pred)
