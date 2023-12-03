import pandas as pd
import numpy as np
import statsmodels as sm
import statsmodels.regression.linear_model as srl
from statsmodels import api as a
import matplotlib.pyplot as plt
from matplotlib.dates import drange
import datetime as dt

# set pandas print display options
pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', False)

# import data from WRDS mutual fund monthly returns
def read_WRDS():
    data = pd.read_csv("test_data.csv",skiprows=0).dropna(how='any')
    return data

# rename and drop columns in WRDS data
def rename_WRDS(data):
    data = data.copy()
    data = data.rename(
        columns={
            "caldt": "date",
            "mtna": "total_net_assets",
            "mret": "total_returns",
            "mnav": "net_asset_value",
        })
    data = data.drop(columns=["crsp_fundno"])
    return data

# remove invalid rows in WRDS data
def remove_R_WRDS(data):
    data = data.copy()

    # remove rows with 'R' in total_returns
    data = data[data.total_returns != 'R']

    # reset index
    data = data.reset_index(drop=True)
    return data

# convert date column into datetime format in WRDS data
def convert_date_WRDS(data):
    data = data.copy()
    data['date'] = pd.to_datetime(data['date'], format='%m/%d/%Y')
    return data

def get_WRDS():
    data = read_WRDS()
    data = rename_WRDS(data)
    data = remove_R_WRDS(data)
    data = convert_date_WRDS(data)
    print(data)
    return data

get_WRDS()