import pandas as pd
import numpy as np
import statsmodels as sm
import statsmodels.regression.linear_model as srl
from statsmodels import api as a
import matplotlib.pyplot as plt
from matplotlib.dates import drange
import datetime as dt

# convert columns in mutual fund historical data from codes to names
MUTUAL_FUND_DATA_COLUMNS = {
    "mret": "return", # Total Return per Share as of Month End
    "mnav": "net_asset_value", #  Monthly Net Asset Value per Share
    "mtna": "total_net_assets" # Total Net Assets as of Month End
}

# tickers of largest largest active mutual fund by AUM for each morningstar category
MUTUAL_FUND_CATEGORIES = {
    ("US Equity", "Large Value"): ["AMRMX"],
    ("US Equity", "Large Blend"): ["AWSHX"],
    ("US Equity", "Large Growth"): ["AGTHX"],
    ("US Equity", "Mid Value"): ["FLPSX"],
    ("US Equity", "Mid Blend"): ["EAASX"],
    ("US Equity", "Mid Growth"): ["RPMGX"],
    ("US Equity", "Small Value"): ["UBVAX"],
    ("US Equity", "Small Blend"): ["PRSVX"],
    ("US Equity", "Small Growth"): ["VEXPX"],
    ("International Equity", "Foreign Large Value"): ["OAYIX"],
    ("International Equity", "Foreign Large Blend"): ["MDIDX"],
    ("International Equity", "Foreign Large Growth"): ["AEPGX"],
    ("International Equity", "Foreign Small/Mid Value"): ["OAYEX"],
    ("International Equity", "Foreign Small/Mid Blend"): ["FISMX"],
    ("International Equity", "Foreign Small/Mid Growth"): ["PRIDX"],
    ("International Equity", "Diversified Emerging Markets"): ["NEWFX"],
    ("US Fixed Income", "Long-Term Bond"): ["VWESX"],
    ("US Fixed Income", "Intermediate Core Bond"): ["ABNDX"],
    ("US Fixed Income", "Intermediate Core-Plus Bond"): ["MWTIX"],
    ("US Fixed Income", "Short-Term Bond"): ["VFSTX"],
}

# reverse dictionary so that tickers are keys and categories are values
MUTUAL_FUND_TICKERS = {}
for category, tickers in MUTUAL_FUND_CATEGORIES.items():
    for ticker in tickers:
        MUTUAL_FUND_TICKERS[ticker] = category

# set print display options
pd.set_option('display.max_columns', 0)
pd.set_option('expand_frame_repr', True)

# import data from WRDS mutual fund monthly returns
def read_mutual_fund_data():
    data = pd.read_csv("unused_data\largest_mutual_fund_every_category.csv",skiprows=0).dropna(how='any')
    return data

# rename and drop columns in mutual fund data
def rename_mutual_fund_data(data):
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

# remove invalid rows in mutual fund data
def remove_rows_mutual_fund_data(data):
    data = data.copy()

    # remove rows with 'R' in total_returns
    data = data[data.total_returns != 'R']

    # reset index
    data = data.reset_index(drop=True)
    return data

# convert date column into datetime format in mutual fund data
def convert_date_mutual_fund_data(data):
    data = data.copy()
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
    data['date'] = data['date'] + pd.offsets.MonthEnd(0)
    return data

# split mutual fund dataframe by ticker
def split_mutual_fund_data(data):
    data = data.copy()
    
    split_data = {}

    total_rows = 0
    for ticker, category in MUTUAL_FUND_TICKERS.items():

        # filter data by ticker
        ticker_data = data[data['ticker'] == ticker].copy()
        ticker_data = ticker_data.reset_index(drop=True)

        # change dates to be sorted properly
        ticker_data = ticker_data.sort_values(by='date', axis=0)
        ticker_data = ticker_data.reset_index().drop('index', axis=1)

        # add col nav return to find returns of the nav
        ticker_data['nav_return'] = ticker_data['net_asset_value'].pct_change()

        split_data[(ticker, category)] = ticker_data

        # update total rows
        total_rows += len(ticker_data)

        # print warning if less than 5 years of data
        if len(ticker_data) < 60:
            print("WARNING: Less than 5 years of data for", ticker, category, "-", len(ticker_data), "months")

    print("Total mutual fund categories:", len(split_data))
    print("Total mutual funds:", len(MUTUAL_FUND_TICKERS))
    print("Total number of rows:", total_rows)
    return split_data

# get and process mutual fund data
def get_mutual_fund_data():
    print("\nMutual Fund Data")
    data = read_mutual_fund_data()
    data = rename_mutual_fund_data(data)
    data = remove_rows_mutual_fund_data(data)
    data = convert_date_mutual_fund_data(data)
    data = split_mutual_fund_data(data)
    return data

# import data from WRDS treasury and inflation monthly returns
def read_bond_data():
    data = pd.read_csv("data\\bond_data.csv",skiprows=0).dropna(how='any')
    return data

# rename columns in bond data
def rename_bond_data(data):
    data = data.copy()
    data = data.rename(
        columns={
            "caldt": "date",
            "b30ret": "30 Year Bond Return",
            "b30ind": "30 Year Bond Index Level",
            "b20ret": "20 Year Bond Return",
            "b20ind": "20 Year Bond Index Level",
            "b10ret": "10 Year Bond Return",
            "b10ind": "10 Year Bond Index Level",
            "b7ret": "7 Year Bond Return",
            "b7ind": "7 Year Bond Index Level",
            "b5ret": "5 Year Bond Return",
            "b5ind": "5 Year Bond Index Level",
            "b2ret": "2 Year Bond Return",
            "b2ind": "2 Year Bond Index Level",
            "b1ret": "1 Year Bond Return",
            "b1ind": "1 Year Bond Index Level",
            "t90ret": "90 Day Bond Return",
            "t90ind": "90 Day Bond Index Level",
            "t30ret": "30 Day Bond Return",
            "t30ind": "30 Day Bond Index Level",
            "cpiret": "CPI Return",
            "cpiind": "CPI Index Level",
        })
    data = data.reset_index(drop=True)
    return data

# convert date column into datetime format in bond data
def convert_date_bond_data(data):
    data = data.copy()
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
    data['date'] = data['date'] + pd.offsets.MonthEnd(0)
    return data

# get bond data
def get_bond_data():
    print("\nBond Data")
    data = read_bond_data()
    data = rename_bond_data(data)
    data = convert_date_bond_data(data)
    print(data)
    return data

def read_ff_data():
    data = pd.read_csv("data\F-F_Research_Data_5_Factors_2x3.csv")
    data = data.rename(columns={'Unnamed: 0': 'date'})
    return data

def convert_date_ff_data(data):
    data = data.copy()
    data['date'] = pd.to_datetime(data['date'], format='%Y%m')
    data['date'] = data['date'] + pd.offsets.MonthEnd(0)
    print(data)
    return data

def get_ff_data():
    print("\nFF Data")
    data = read_ff_data()
    return convert_date_ff_data(data)

get_mutual_fund_data()
get_bond_data()
get_ff_data()