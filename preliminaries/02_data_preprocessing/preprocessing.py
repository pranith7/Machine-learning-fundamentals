import os
import torch
import pandas as pd

os.makedirs(os.path.join('/teamspace/studios/this_studio/preliminaries/02_data_preprocessing/', 'data'), exist_ok=True)
data_file = os.path.join('/teamspace/studios/this_studio/preliminaries/02_data_preprocessing/', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('''
                NumRooms,RoofType,Price
                NA,NA,127500
                2,NA,106000
                4,Slate,178100
                NA,NA,140000
            '''
            )


data = pd.read_csv(data_file)
print(data)

print("Preparing our data \n")
inputs, targets = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

print("Replace NaN values with Mean of the corresponding column \n")
inputs = inputs.fillna(inputs.mean())
print(inputs)

print("Conversion into Tensor Format. \n")

X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(targets.to_numpy(dtype=float))
print(X, y)
