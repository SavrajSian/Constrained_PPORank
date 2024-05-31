import pandas as pd
import numpy as np

drugs = pd.read_csv('GDSC_ALL/severity_scores.csv')

normalised_scores = drugs['normalised']

ordered_scores = normalised_scores.sort_values(ascending=True)

# lambda pen function:
indices = np.arange(1, len(ordered_scores) + 1, dtype=float)
pen_func = lambda x: x * (-10*np.log(indices) + 40)

penalties = pen_func(ordered_scores)

total_penalty = penalties.sum()
normalised_total_penalty = total_penalty / len(ordered_scores)

print('total penalty:', total_penalty)
print('normalised total penalty:', normalised_total_penalty)

print('end')
