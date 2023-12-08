from tools import get_mutual_fund_data, get_bond_data, get_ff_data, get_index_data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from data_tools import capm, ff_3, ff_5, reg_date_range, capm_index, corr_index

mf_dict = get_mutual_fund_data()
ff_df = get_ff_data()
bond_df = get_bond_data()
index_dict = get_index_data()

mf_types = list(mf_dict.keys())
us_eq_data = {k[2]:[] for k in mf_dict.keys() if 'US Equity' in k}

for mf_key, mf_val in mf_dict.items():
    if mf_key[2] in us_eq_data.keys():
        us_eq_data[mf_key[2]].append(mf_val)

us_eq_alphas_c = []
us_eq_alphas_std_c = []
us_eq_betas_c = []
for strat in us_eq_data.keys():
    temp_alpha=[]
    temp_beta=[]
    for ticker in us_eq_data[strat]:
        result = capm(ticker, ff_df)
        if result == None:
            continue
        temp_alpha.append(result['const'])
        temp_beta.append(result['Mkt-RF'])
    us_eq_alphas_c.append(np.mean(temp_alpha))
    us_eq_alphas_std_c.append(np.std(temp_alpha))
    us_eq_betas_c.append(np.mean(temp_beta))

us_index = {k[2]: v for k, v in index_dict.items() if 'US Equity' in k}

us_idx_alphas_c = []
us_idx_betas_c = []
us_idx_alphas_std_c = []
for strat in us_eq_data.keys():
    temp_alpha=[]
    temp_beta=[]
    for ticker in us_eq_data[strat]:
        start_date = ff_df['date'][0] if ff_df['date'][0] > ticker['date'][1] else ticker['date'][1]
        start_date = start_date if start_date > us_index[strat]['Date'][1] else us_index[strat]['Date'][1]
        end_date = ticker['date'].iloc[-1]
        result = capm_index(ticker, ff_df, us_index[strat], start_date, end_date)
        if result == None:
            continue
        temp_alpha.append(result['const'])
        temp_beta.append(result['beta'])
    us_idx_alphas_c.append(np.mean(temp_alpha))
    us_idx_betas_c.append(np.mean(temp_beta))
    us_idx_alphas_std_c.append(np.std(temp_alpha))

ind_alphas_c = []
ind_betas_c = []
fund_tickers = []

for strat in us_eq_data.keys():
    temp_alpha=[]
    temp_beta=[]
    temp_tickers = []
    for ticker in us_eq_data[strat]:
        result = capm(ticker, ff_df)
        if result == None:
            continue
        temp_alpha.append(result['const'])
        temp_beta.append(result['Mkt-RF'])
        temp_tickers.append(ticker['ticker'].iloc[0])
    ind_alphas_c.append(temp_alpha)
    ind_betas_c.append(temp_beta)
    fund_tickers.append(temp_tickers)

ind_idx_alphas_c = []
ind_idx_betas_c = []
fund_tickers_idx = []
ind_corr = []
for strat in us_eq_data.keys():
    temp_alpha=[]
    temp_beta=[]
    temp_tickers=[]
    temp_corr=[]
    for ticker in us_eq_data[strat]:
        start_date = ff_df['date'][0] if ff_df['date'][0] > ticker['date'][1] else ticker['date'][1]
        start_date = start_date if start_date > us_index[strat]['Date'][1] else us_index[strat]['Date'][1]
        end_date = ticker['date'].iloc[-1]
        result = capm_index(ticker, ff_df, us_index[strat], start_date, end_date)
        if result == None:
            continue
        temp_alpha.append(result['const'])
        temp_beta.append(result['beta'])
        temp_tickers.append(ticker['ticker'].iloc[0])
        temp_corr.append(corr_index(ticker, us_index[strat], start_date, end_date))
    ind_idx_alphas_c.append(temp_alpha)
    ind_idx_betas_c.append(temp_beta)
    fund_tickers_idx.append(temp_tickers)
    ind_corr.append(temp_corr)


def data_analyze_strat_base(strat, strat_name):
    fig = plt.figure(figsize=(15,6))

    plt.axhspan(us_eq_alphas_c[strat]-us_eq_alphas_std_c[strat], us_eq_alphas_c[strat]+us_eq_alphas_std_c[strat], facecolor='r', alpha=0.5)
    plt.bar(fund_tickers[strat], ind_alphas_c[strat])
    plt.xticks(rotation=90)
    plt.show()

    n=5
    largest_index = sorted(range(len(ind_alphas_c[strat])), key = lambda sub: ind_alphas_c[strat][sub])[-n:]
    best_alpha_mf = [fund_tickers[strat][i] for i in largest_index]
    print(f'The largest alpha funds by CAPM marked to whole market are {best_alpha_mf}.  Note that the best one is last.')
    above_std = [idx for idx, val in enumerate(ind_alphas_c[strat]) if val > us_eq_alphas_c[strat]+us_eq_alphas_std_c[strat]]
    above_std_mf = [fund_tickers[strat][i] for i in above_std]
    print(f'The mutual funds that are 1 stdev above mean with benchmark alpha are {above_std_mf}')

    below_std = [idx for idx, val in enumerate(ind_alphas_c[strat]) if val < us_eq_alphas_c[strat]-us_eq_alphas_std_c[strat]]
    below_std_mf = [fund_tickers[strat][i] for i in below_std]
    print(f'The mutual funds that are 1 stdev below mean with benchmark alpha are {below_std_mf}')

    below_std_beta = [ind_betas_c[strat][i] for i in below_std]
    above_std_beta = [ind_betas_c[strat][i] for i in above_std]
    print(f'Average beta of below stdev alpha is {np.mean(below_std_beta)} with a stdev on beta of {np.std(below_std_beta)}')
    print(f'Average beta of above stdev alpha is {np.mean(above_std_beta)} with a stdev on beta of {np.std(above_std_beta)}')

    plt.bar(below_std_mf, below_std_beta, label='below')
    plt.bar(above_std_mf, above_std_beta, label='above')
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()

def data_analyze_strat_bench(strat, strat_name):
    fig = plt.figure(figsize=(15,6))

    plt.axhspan(us_idx_alphas_c[strat]-us_idx_alphas_std_c[strat], us_idx_alphas_c[strat]+us_idx_alphas_std_c[strat], facecolor='r', alpha=0.5)
    plt.bar(fund_tickers[strat], ind_idx_alphas_c[strat])
    plt.xticks(rotation=90)
    plt.show()

    n=5
    largest_index = sorted(range(len(ind_idx_alphas_c[strat])), key = lambda sub: ind_idx_alphas_c[strat][sub])[-n:]
    best_alpha_mf = [fund_tickers_idx[strat][i] for i in largest_index]
    print(f'The largest alpha funds by CAPM marked to benchmask are {best_alpha_mf}.  Note last one is best one')
    above_std_index = [idx for idx, val in enumerate(ind_idx_alphas_c[strat]) if val > us_idx_alphas_c[strat]+us_idx_alphas_std_c[strat]]
    above_std_mf = [fund_tickers_idx[strat][i] for i in above_std_index]
    print(f'The mutual funds that are 1 stdev above mean with benchmark alpha are {above_std_mf}')
    below_std_index = [idx for idx, val in enumerate(ind_idx_alphas_c[strat]) if val < us_idx_alphas_c[strat]-us_idx_alphas_std_c[strat]]
    below_std_mf = [fund_tickers_idx[strat][i] for i in below_std_index]
    print(f'The mutual funds that are 1 stdev above mean with benchmark alpha are {below_std_mf}')
    below_std_beta = [ind_idx_betas_c[strat][i] for i in below_std_index]
    above_std_beta = [ind_idx_betas_c[strat][i] for i in above_std_index]
    print(f'Average beta of below stdev alpha is {np.mean(below_std_beta)} with a stdev on beta of {np.std(below_std_beta)}')
    print(f'Average beta of above stdev alpha is {np.mean(above_std_beta)} with a stdev on beta of {np.std(above_std_beta)}')

    plt.bar(below_std_mf, below_std_beta, label='below')
    plt.bar(above_std_mf, above_std_beta, label='above')
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()

    below_std_corr = [ind_corr[strat][i] for i in below_std_index]
    above_std_corr = [ind_corr[strat][i] for i in above_std_index]
    print(f'Average beta of below stdev alpha is {np.mean(below_std_corr)} with a stdev on beta of {np.std(below_std_corr)}')
    print(f'Average beta of above stdev alpha is {np.mean(above_std_corr)} with a stdev on beta of {np.std(above_std_corr)}')

    plt.bar(below_std_mf, below_std_corr, label='below')
    plt.bar(above_std_mf, above_std_corr, label='above')
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()