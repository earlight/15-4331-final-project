import pandas as pd

# tickers of largest largest active mutual fund by AUM for each morningstar category
MUTUAL_FUND_CATEGORIES= {
    ("US Equity", "Large Value"): [],
    ("US Equity", "Large Blend"): [],
    ("US Equity", "Large Growth"): [],
    ("US Equity", "Mid-Cap Value"): [],
    ("US Equity", "Mid-Cap Blend"): [],
    ("US Equity", "Mid-Cap Growth"): [],
    ("US Equity", "Small Value"): [],
    ("US Equity", "Small Blend"): [],
    ("US Equity", "Small Growth"): [],
    ("International Equity", "Foreign Large Value"): [],
    ("International Equity", "Foreign Large Blend"): [],
    ("International Equity", "Foreign Large Growth"): [],
    ("International Equity", "Foreign SmallMid Value"): [],
    ("International Equity", "Foreign SmallMid Blend"): [],
    ("International Equity", "Foreign SmallMid Growth"): [],
    ("International Equity", "Diversified Emerging Mkts"): [],
    ("US Fixed Income", "Long-Term Bond"): [],
    ("US Fixed Income", "Intermediate Core Bond"): [],
    ("US Fixed Income", "Intermediate Core-Plus Bond"): [],
    ("US Fixed Income", "Short-Term Bond"): [],
}

# reversed dictionary so that tickers are keys and categories are values
MUTUAL_FUND_TICKERS = {}

# convert benchmark index ticker to morningstar category
BENCHMARK_INDEX_CATEGORIES = {
    "RU10VATR": ("US Equity", "Large Value", "Russell 1000 Value TR USD"),
    "RUITR": ("US Equity", "Large Blend", "Russell 1000 TR USD"),
    "RU10GRTR": ("US Equity", "Large Growth", "Russell 1000 Growth TR USD"),
    "RUMCVATR": ("US Equity", "Mid-Cap Value", "Russell Mid Cap Value TR USD"),
    "RMCTR": ("US Equity", "Mid-Cap Blend", "Russell Mid Cap TR USD"),
    "RUMCGRTR": ("US Equity", "Mid-Cap Growth", "Russell Mid Cap Growth TR USD"),
    "RUJTR": ("US Equity", "Small Value", "Russell 2000 Value TR USD"),
    "RUTTR": ("US Equity", "Small Blend", "Russell 2000 TR USD"),
    "RUOTR": ("US Equity", "Small Growth", "Russell 2000 Growth TR USD"),
    "MGCUWXUN": ("International Equity", "Foreign Large Value", "MSCI ACWI ex USA Value NR USD"),
    "M1WDU": ("International Equity", "Foreign Large Blend", "MSCI ACWI ex USA NR USD"),
    "M1WDU00G": ("International Equity", "Foreign Large Growth", "MSCI ACWI ex USA Growth NR USD"),
    "M1WDU009": ("International Equity", "Foreign SmallMid Value", "MSCI ACWI ex USA SMID Value NR USD"),
    "M1WDUSM": ("International Equity", "Foreign SmallMid Blend", "MSCI ACWI ex USA SMID NR USD"),
    "M1WDU00Z": ("International Equity", "Foreign SmallMid Growth", "MSCI ACWI ex USA SMID Growth NR USD"),
    "M1EF": ("International Equity", "Diversified Emerging Mkts", "MSCI EM NR USD"),
    "BFALTRUU": ("US Fixed Income", "Long-Term Bond", "Bloomberg US Government/Credit Long TR USD"),
    "LBUSTRUU": ("US Fixed Income", "Intermediate Core Bond", "Bloomberg US Aggregate Bond TR Index"),
    "LC07TRUU": ("US Fixed Income", "Intermediate Core-Plus Bond", "Bloomberg US Universal Bond TR Index"),
    "LGC3TRUU": ("US Fixed Income", "Short-Term Bond", "Bloomberg US Government/Credit 1-3 Year TR USD"),
}

# set print display options
pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', False)

# import fidelity data of top 100 active funds by AUM for each category
def read_fidelity_data():
    
    fidelity_data = dict()
    count = 0
    for asset_class, category in MUTUAL_FUND_CATEGORIES.keys():
        count += 1
        data = pd.read_excel("data/mutual_funds/category_largest/" + category + ".xlsx")
        fidelity_data[(asset_class, category)] = data

    return fidelity_data

# remove last 21 rows of every dataframe
def remove_rows_fidelity_data(fidelity_data):
    output_data = dict()

    for key, data in fidelity_data.items():
        asset_class, category = key
        data = data.copy()
        data = data.iloc[:-21]
        output_data[(asset_class, category)] = data

    return output_data

# add tickers from fidelity data to the MUTUAL_FUND_CATEGORIES dictionary
def add_tickers_fidelity_data(fidelity_data):
    for key, data in fidelity_data.items():
        asset_class, category = key
        names = data["Name"].tolist()
        
        # find the ticker inside parentheses
        tickers = [ name[name.find("(")+1:name.find(")")] for name in names ]
        MUTUAL_FUND_CATEGORIES[(asset_class, category)] = tickers
    
    # reversed dictionary so that tickers are keys and categories are values
    for category, tickers in MUTUAL_FUND_CATEGORIES.items():
        for ticker in tickers:
            MUTUAL_FUND_TICKERS[ticker] = category

    print("Total number of categories", len(MUTUAL_FUND_CATEGORIES))
    print("Total number of funds:", len(MUTUAL_FUND_TICKERS))

def create_ticker_file_fidelity_data(fidelity_data):
    file = open("data/mutual_funds/all_tickers.txt", "w")

    for key, data in fidelity_data.items():
        asset_class, category = key
        names = data["Name"].tolist()
        
        # find the ticker inside parentheses
        tickers = [ name[name.find("(")+1:name.find(")")] for name in names ]
        for ticker in tickers:
            file.write(ticker + "\n")

    file.close()

    file = open("data/mutual_funds/all_tickers.txt", "r")
    print("Created ticker file with", len(file.readlines()), "tickers")
    file.close()

# get fidelity data
def get_fidelity_data(create_ticker_file=False):
    fidelity_data = read_fidelity_data()
    fidelity_data = remove_rows_fidelity_data(fidelity_data)
    add_tickers_fidelity_data(fidelity_data)
    if create_ticker_file:
        create_ticker_file_fidelity_data(fidelity_data)

# import data from WRDS mutual fund monthly returns
def read_mutual_fund_data():
    # ticker, crsp_fundno, caldt are strings. mtna, mret, mnav are floats
    data = pd.read_csv("data/mutual_funds/mutual_fund_data.csv",skiprows=0).dropna(how='any')
    return data

# rename and drop columns in mutual fund data
def rename_mutual_fund_data(data):
    data = data.copy()
    data = data.rename(
        columns={
            "caldt": "date",
            "mtna": "total_net_assets", # Total Net Assets as of Month End
            "mret": "total_returns", # Total Return per Share as of Month End
            "mnav": "net_asset_value", # Monthly Net Asset Value per Share
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
    empty_tickers = []
    young_tickers = []
    for ticker, ticker_info in MUTUAL_FUND_TICKERS.items():
        asset_class, category = ticker_info

        # filter data by ticker
        ticker_data = data[data['ticker'] == ticker].copy()
        ticker_data = ticker_data.reset_index(drop=True)

        # change dates to be sorted properly
        ticker_data = ticker_data.sort_values(by='date', axis=0)
        ticker_data = ticker_data.reset_index().drop('index', axis=1)

        # add col nav return to find returns of the nav
        ticker_data['nav_return'] = ticker_data['net_asset_value'].astype(float).pct_change()

        # ignore tickers with no data
        if len(ticker_data) == 0:
            empty_tickers.append((ticker, asset_class, category))

        # ignore tickers with less than 5 years of data
        elif len(ticker_data) < 60:
            young_tickers.append((ticker, asset_class, category))

        # add ticker data to split data dictionary and update total rows
        else:
            split_data[(ticker, asset_class, category)] = ticker_data
            total_rows += len(ticker_data)
    
    print("Total number of rows:", total_rows)
    print("Total number of funds with enough data:", len(split_data))
    print("Funds with no data:", len(empty_tickers))
    print("Funds with less than 5 years of data:", len(young_tickers))
    print("Columns:", split_data[next(iter(split_data))].columns)
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
    data = pd.read_csv("data/bond_data.csv",skiprows=0).dropna(how='any')
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
    print("Columns:", data.columns)
    return data

# import data from fama french monthly returns
def read_ff_data():
    data = pd.read_csv("data/F-F_Research_Data_5_Factors_2x3.csv")
    data = data.rename(columns={'Unnamed: 0': 'date'})
    return data

# convert date column into datetime format in fama french data
def convert_date_ff_data(data):
    data = data.copy()
    data['date'] = pd.to_datetime(data['date'], format='%Y%m')
    data['date'] = data['date'] + pd.offsets.MonthEnd(0)
    print(data)
    return data

# get fama french data
def get_ff_data():
    print("\nFF Data")
    data = read_ff_data()
    return convert_date_ff_data(data)

# import data from bloomberg benchmark index monthly returns
def read_index_data():
    all_index_data = dict()
    count = 0
    for ticker, ticker_info in BENCHMARK_INDEX_CATEGORIES.items():
        count += 1
        asset_class, category, name = ticker_info
        if asset_class == "US Fixed Income":
            index_data = pd.read_excel("data/representative_benchmarks/" + ticker + ".xlsx", skiprows=5)
        else:
            index_data = pd.read_excel("data/representative_benchmarks/" + ticker + ".xlsx", skiprows=6)

        # Reorganizing index data
        index_data['Date'] = pd.to_datetime(index_data['Date'], format='%Y-%m-%d')
        index_data['Date'] = index_data['Date'] + pd.offsets.MonthEnd(0)
        index_data = index_data.sort_values(by='Date', axis=0)
        index_data = index_data.reset_index().drop('index', axis=1)

        all_index_data[(ticker, asset_class, category, name)] = index_data
    return all_index_data

# rename and drop columns in index data
def rename_index_data(all_index_data):
    for index, data in all_index_data.items():
        ticker, asset_class, category, name = index
        # drop columns
        data = data.drop(columns=["PX_VOLUME", "Change.1", "% Change.1"]).dropna(how='any')
        print(category, ":", len(data), "months")
    print("Columns:", data.columns)
    return all_index_data

def get_index_data():
    print("\nIndex Data")
    data = read_index_data()
    data = rename_index_data(data)
    return data

get_fidelity_data()
mutual_fund_data = get_mutual_fund_data()
# bond_data = get_bond_data()
# ff_data = get_ff_data()
# index_data = get_index_data()