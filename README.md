# Electricity Price Forecasting
Past experiments made with the goal of forecasting electricity prices. The data used in these experiments is the the Day Ahead Prices of Turkey Market, fetched using [Electricity Price Fetcher](https://github.com/rojenarda/electricity_data_fetching_tr) from the EPIAS API.

## chronos.ipynb    
AutoGluon Chronos, pretreined timeseries forecasting model. Complete example.

## deepar.ipynb
DeepAR PyTorch implementation from GluonTS. Complete example.

## epftoolbox_dnn_live_model.py
Live model training and forecasting using [EPFToolBox's DNN Model](https://github.com/jeslago/epftoolbox). Install the EPFToolBox library, then set a cronjob to run this script every day. Complete example.

## pytorch_forecasting.ipynb
Temporal Fusion Transformer and NBeats implementation from PyTorch Forecasting. Incomplete example.

## Data
Example data can be found in the `data` directory. The data is in the format of a CSV. You should use Electricity Price Fetcher to get the latest data.
