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

index = str(sys.argv[1])
file_path = f'~/Fun_LassoNet/modelA1/X{index}.csv'
save_path = f'~/Fun_LassoNet/modelA1/results{index}.csv'
df = pd.read_csv(file_path, index_col=0)

# Initialize an empty dictionary to store the split DataFrames
data_frames = {}

# Split the DataFrame into 30 smaller DataFrames with 7 columns each
for i in range(30):
    start_col = i * 7
    end_col = start_col + 7
    data_frames[f'X{i+1}'] = df.iloc[:, start_col:end_col]
    data_frames[f'X{i+1}'] = scale(data_frames[f'X{i+1}'])


lam = np.arange(0, 20000, 5)

model = LassoNetRegressor(
    hidden_dims=(30,),
    verbose=True,
    n_iters=(3000, 300),
        patience=(300, 30),
    dropout=0.4,
    lambda_seq = lam,
    #groups=[[0, 1, 2]],
    M=100,
)

results = []

for i in range(1, 31):
    for j in range(1, 31):
        path = model.path(data_frames[f'X{i}'],data_frames[f'X{j}'])
        results.append((i, j, [save.selected.sum() for save in path], [save.lambda_ for save in path]))

results_df = pd.DataFrame(results, columns=['i', 'j', 'selected', 'lambda'])

results_df.to_csv(save_path, index=False)


