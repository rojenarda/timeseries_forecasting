# Runs everyday at 10.00 and predicts the next day's Day-ahead prices

import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('.')
sys.path.append('./epftoolbox')

from epftoolbox.data import DataScaler
from epftoolbox.data import read_data
from epftoolbox.models import DNN
from epftoolbox.models import hyperparameter_optimizer, evaluate_dnn_in_test_dataset

from electricity_data_fetching_tr import GetData, UpdateData

path_datasets = 'data_fetching/datasets'
path_hyperparameter_folder = 'demo/hyperparameters'
dataset = os.listdir('data_fetching/datasets')[0].split('.')[0]
years_test = 1
experiment_id = 'live_model_exp_01'
shuffle_train = False
calibration_window = 3 # Use 3 years of data for recalibration
max_evals = 20
nlayers = 2
path_datasets_folder = path_datasets
path_recalibration_folder = 'demo/recalibration'
new_recalibration = 0
data_augmentation = 0
new_hyperopt = 0

# Update the dataset with the latest data
data = UpdateData(dataset)
data.get_data(replace_last_day=True)

df_train, df_test = read_data(dataset=dataset, years_test=years_test, path=path_datasets_folder)

forecast_file_name = 'fc_nl' + str(nlayers) + \
'_YT' + str(years_test) + '_SF' + str(shuffle_train) + \
'_DA' * data_augmentation + '_CW' + str(calibration_window) + \
'_' + str(experiment_id) + '.csv'

forecast_file_path = os.path.join(path_recalibration_folder, forecast_file_name)

data_available = pd.concat([df_train, df_test], axis=0)
forecast_date = data_available.index[-1].replace(hour=0)

# Rename the hyperparameters file

files = os.listdir(path_hyperparameter_folder)
for file in files:
    if 'hyperparameters' in file:
        hyperparameter_file = file
        break

hyperparameter_file_path = os.path.join(path_hyperparameter_folder, hyperparameter_file)
new_name = \
            'DNN_hyperparameters_nl' + str(nlayers) + \
            '_dat' + str(dataset) + '_YT' + str(years_test) + \
            '_SF' * (shuffle_train) + '_DA' * (data_augmentation) + \
            '_CW' + str(calibration_window) + '_' + str(experiment_id)

new_hyperparameter_file_path = os.path.join(path_hyperparameter_folder, new_name)
os.rename(hyperparameter_file_path, new_hyperparameter_file_path)


model = DNN(
    dataset=dataset,
    experiment_id=experiment_id,
    path_hyperparameter_folder=path_hyperparameter_folder,
    nlayers=nlayers,
    years_test=years_test,
    shuffle_train=shuffle_train,
    data_augmentation=data_augmentation,
    calibration_window=calibration_window
)

Yp = model.recalibrate_and_forecast_next_day(df=data_available, next_day_date=forecast_date)

forecast = pd.DataFrame(index=[forecast_date], columns=['h' + str(k) for k in range(24)])
forecast.iloc[0, :] = Yp

if os.path.exists(forecast_file_path):
    forecast.to_csv(forecast_file_path, mode='a', header=False)
else:
    forecast.to_csv(forecast_file_path)
