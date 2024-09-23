from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from lassonet import LassoNetRegressorCV 
from lassonet import LassoNetRegressor
from sklearn.datasets import load_diabetes
import itertools
import sys

# Get index, p, and m from command line arguments
index = str(sys.argv[1])  # seed index
p = int(sys.argv[2])  # number of functional covariates (number of splits)
m = int(sys.argv[3])  # number of columns in each group

file_path = f'~/Desktop/Functional_graphical/X{index}.csv'
save_path = f'~/Desktop/Functional_graphical/results{index}.csv'
df = pd.read_csv(file_path, index_col=0)

# Initialize an empty dictionary to store the split DataFrames
data_frames = {}

# Split the DataFrame into p smaller DataFrames with m columns each
for i in range(p):
    start_col = i * m
    end_col = start_col + m
    data_frames[f'X{i+1}'] = df.iloc[:, start_col:end_col]
    data_frames[f'X{i+1}'] = scale(data_frames[f'X{i+1}'])

lam = np.arange(0, 20000, 5)

# Define groups for consecutive m columns up to p * m (adjusted based on p and m)
groups = [list(range(i, i + m)) for i in range(0, (p - 1) * m, m)]

model = LassoNetRegressor(
    hidden_dims=(300,),
    verbose=True,
    n_iters=(3000, 300),
    patience=(300, 30),
    dropout=0.4,
    lambda_seq=lam,
    groups=groups,
    M=100,  )

results = []

# Loop over the range 1 to p (instead of hardcoded 100 or 101)
for i in range(1, p + 1):
    combined_data = pd.concat([data_frames[f'X{j}'] for j in range(1, p + 1) if j != i], axis=1)
    path = model.path(combined_data, data_frames[f'X{i}'])
    results.append((i, [save.selected.sum() for save in path], [save.lambda_ for save in path]))

results_df = pd.DataFrame(results, columns=['i', 'selected', 'lambda'])

# Save results
results_df.to_csv(save_path, index=False)
