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

# convert between morningstar categories and ticker of that category's largest active mutual fund
ALLOCATION_CATEGORIES = {
    "GWPAX": "Aggressive Allocation",
    "FASIX": "Conservative Allocation",
    "FCVSX": "Convertibles",
    "CAIBX": "Global Allocation",
    "ABALX": "Moderate Allocation",
    "AMECX": "Moderately Aggressive Allocation",
    "VWINX": "Moderately Conservative Allocation",
    "PASAX": "Tactical Allocation",
    "AAATX": "Target-Date 2000-2010",
    "FFVFX": "Target-Date 2015",
    "AACTX": "Target-Date 2020",
    "AADTX": "Target-Date 2025",
    "AAETX": "Target-Date 2030",
    "AAFTX": "Target-Date 2035",
    "AAGTX": "Target-Date 2040",
    "AAHTX": "Target-Date 2045",
    "AALTX": "Target-Date 2050",
    "AAMTX": "Target-Date 2055",
    "AANTX": "Target-Date 2060",
    "AAOTX": "Target-Date 2065+",
    "FFFAX": "Target-Date Retirement"
}
ALTERNATIVE_CATEGORIES = {
    "BTCFX": "Digital Assets",
    "CBHAX": "Equity Market Neutral",
    "BILPX": "Event Driven",
    "MBXAX": "Macro Trading",
    "BIMBX": "Multistrategy",
    "JHQAX": "Options Trading",
    "CVSIX": "Relative Value Arbitrage",
    "AHLAX": "Systematic Trend"
}
COMMODITY_CATEGORIES = {
    "PCRAX": "Commodities Broad Basket",
    "QGLCX": "Commodities Focused"
}
INTERNATIONAL_CATEGORIES = {
    "FHKCX": "China Region",
    "NEWFX": "Diversified Emerging Mkts",
    "MIPIX": "Diversified Pacific/Asia",
    "PRESX": "Europe Stock",
    "MDIDX": "Foreign Large Blend",
    "AEPGX": "Foreign Large Growth",
    "OAYIX": "Foreign Large Value",
    "FISMX": "Foreign Small/Mid Blend",
    "PRIDX": "Foreign Small/Mid Growth",
    "OAYEX": "Foreign Small/Mid Value",
    "CWGIX": "Global Large-Stock Blend",
    "ANWPX": "Global Large-Stock Growth",
    "PRAFX": "Global Large-Stock Value",
    "SMCWX": "Global Small/Mid Stock",
    "MIDNX": "India Equity",
    "FJPNX": "Japan Stock",
    "PRLAX": "Latin America Stock",
    "FICDX": "Miscellaneous Region",
    "MIPTX": "Pacific/Asia ex-Japan Stk"
}
MISCELLANEOUS_CATEGORIES = {
    "COMVX": "Trading--Inverse Equity"
}
MONEY_MARKET_CATEGORIES = {
    "FTCXX": "Money Market-Tax-Free",
    "SPAXX": "Money Market-Taxable",
    "FMPXX": "Prime Money Market"
}
MUNICIPAL_BOND_CATEGORIES = {
    "NHMAX": "High Yield Muni",
    "VCAIX": "Muni California Intermediate",
    "FTFQX": "Muni California Long",
    "VMATX": "Muni Massachusetts",
    "FMNQX": "Muni Minnesota",
    "VWITX": "Muni National Interm",
    "VWLTX": "Muni National Long",
    "VMLTX": "Muni National Short",
    "VNJTX": "Muni New Jersey",
    "DRNYX": "Muni New York Intermediate",
    "VNYTX": "Muni New York Long",
    "VOHIX": "Muni Ohio",
    "VPAIX": "Muni Pennsylvania",
    "KYTFX": "Muni Single State Interm",
    "MDXBX": "Muni Single State Long",
    "LTNYX": "Muni Single State Short",
    "FIMSX": "Muni Target Maturity"
}
NONTRADITIONAL_EQUITY_CATEGORIES = {
    "JEPAX": "Derivative Income",
    "NLSAX": "Long-Short Equity"
}
SECTOR_EQUITY_CATEGORIES = {
    "PRMTX": "Communications",
    "FSRPX": "Consumer Cyclical",
    "FDFAX": "Consumer Defensive",
    "MLPDX": "Energy Limited Partnership",
    "VGENX": "Equity Energy",
    "SGGDX": "Equity Precious Metals",
    "PRISX": "Financial",
    "MGLAX": "Global Real Estate",
    "VGHCX": "Health",
    "FSDAX": "Industrials",
    "GLIFX": "Infrastructure",
    "WWWFX": "Miscellaneous Sector",
    "PRNEX": "Natural Resources",
    "CSRIX": "Real Estate",
    "FSPTX": "Technology",
    "FKUQX": "Utilities"
}
TAXABLE_BOND_CATEGORIES = {
    "FFRHX": "Bank Loan",
    "VFICX": "Corporate Bond",
    "MEDAX": "Emerging Markets Bond",
    "PELAX": "Emerging-Markets Local-Currency Bond",
    "CWBFX": "Global Bond",
    "PFOAX": "Global Bond-USD Hedged",
    "VWEHX": "High Yield Bond",
    "VIPSX": "Inflation-Protected Bond",
    "ABNDX": "Intermediate Core Bond",
    "MWTIX": "Intermediate Core-Plus Bond",
    "AMUSX": "Intermediate Government",
    "VUSTX": "Long Government",
    "VWESX": "Long-Term Bond",
    "FXIDX": "Miscellaneous Fixed Income",
    "PONAX": "Multisector Bond",
    "BSIIX": "Nontraditional Bond",
    "CPXAX": "Preferred Stock",
    "VFISX": "Short Government",
    "VFSTX": "Short-Term Bond",
    "VUBFX": "Ultrashort Bond"
}
US_EQUITY_CATEGORIES = {
    "AWSHX": "Large Blend",
    "AGTHX": "Large Growth",
    "AMRMX": "Large Value",
    "EAASX": "Mid-Cap Blend",
    "RPMGX": "Mid-Cap Growth",
    "FLPSX": "Mid-Cap Value",
    "PRSVX": "Small Blend",
    "VEXPX": "Small Growth",
    "UBVAX": "Small Value"
}

# combine all morningstar mutual fund categories into one dictionary
ALL_CATEGORIES = {}
for ticker, category in ALLOCATION_CATEGORIES.items():
    ALL_CATEGORIES[ticker] = "Allocation: " + category
for ticker, category in ALTERNATIVE_CATEGORIES.items():
    ALL_CATEGORIES[ticker] = "Alternative: " + category
for ticker, category in COMMODITY_CATEGORIES.items():
    ALL_CATEGORIES[ticker] = "Commodity: " + category
for ticker, category in INTERNATIONAL_CATEGORIES.items():
    ALL_CATEGORIES[ticker] = "International: " + category
for ticker, category in MISCELLANEOUS_CATEGORIES.items():
    ALL_CATEGORIES[ticker] = "Miscellaneous: " + category
for ticker, category in MONEY_MARKET_CATEGORIES.items():
    ALL_CATEGORIES[ticker] = "Money Market: " + category
for ticker, category in MUNICIPAL_BOND_CATEGORIES.items():
    ALL_CATEGORIES[ticker] = "Municipal Bond: " + category
for ticker, category in NONTRADITIONAL_EQUITY_CATEGORIES.items():
    ALL_CATEGORIES[ticker] = "Nontraditional Equity: " + category
for ticker, category in SECTOR_EQUITY_CATEGORIES.items():
    ALL_CATEGORIES[ticker] = "Sector Equity: " + category
for ticker, category in TAXABLE_BOND_CATEGORIES.items():
    ALL_CATEGORIES[ticker] = "Taxable Bond: " + category
for ticker, category in US_EQUITY_CATEGORIES.items():
    ALL_CATEGORIES[ticker] = "US Equity: " + category

# set print display options
pd.set_option('display.max_columns', 0)
pd.set_option('expand_frame_repr', True)

# import data from WRDS mutual fund monthly returns
def read_mutual_fund_data():
    data = pd.read_csv("mutual_fund_data.csv",skiprows=0).dropna(how='any')
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
    return data

# split mutual fund dataframe by ticker
def split_mutual_fund_data(data):
    data = data.copy()
    
    split_data = {}

    total_rows = 0
    for ticker, category in ALL_CATEGORIES.items():

        # filter data by ticker
        ticker_data = data[data['ticker'] == ticker].copy()
        ticker_data = ticker_data.reset_index(drop=True)
        split_data[category] = ticker_data

        # update total rows
        total_rows += len(ticker_data)

        # print warning if less than 5 years of data
        if len(ticker_data) < 60:
            print("WARNING: Less than 5 years of data for ", ticker, ", ", category, "(", len(ticker_data), " months)")

    print("Total mutual fund categories: ", len(split_data))
    print("Total number of rows: ", total_rows)
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
    data = pd.read_csv("bond_data.csv",skiprows=0).dropna(how='any')
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
    return data

# get bond data
def get_bond_data():
    print("\nBond Data")
    data = read_bond_data()
    data = rename_bond_data(data)
    data = convert_date_bond_data(data)
    print(data)
    return data

get_mutual_fund_data()
get_bond_data()