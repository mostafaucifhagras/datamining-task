import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

dataset_path = '/path/to/your/dataset.csv'
api.download_dataset('username/dataset-name', path=dataset_path)

data = pd.read_csv(dataset_path)

imputer = SimpleImputer(strategy='mean')
data = imputer.fit_transform(data)

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)


data_scaled = pd.DataFrame(data_scaled, columns=data.columns)

print(data_scaled.head())
