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

for idx, val in enumerate(us_eq_data['Mid-Cap Growth']):
    if val['ticker'].iloc[0] == 'DEEVX':
        del us_eq_data['Mid-Cap Growth'][idx]

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


ind_alphas_5 = []
ind_betas_5 = []
ind_smbs_5 = []
ind_hmls_5 = []
ind_rmws_5 = []
ind_cmas_5 = []
for strat in us_eq_data.keys():
    temp_alpha=[]
    temp_beta=[]
    temp_smb = []
    temp_hml = []
    temp_rmw = []
    temp_cma = []
    for ticker in us_eq_data[strat]:
        start_date = ff_df['date'][0] if ff_df['date'][0] > ticker['date'][1] else ticker['date'][1]
        end_date = ticker['date'].iloc[-1]
        result = reg_date_range(ticker, ff_df, ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'], start_date, end_date)
        if result == None:
            continue
        temp_alpha.append(result['const'])
        temp_beta.append(result['Mkt-RF'])
        temp_smb.append(result['SMB'])
        temp_hml.append(result['HML'])
        temp_rmw.append(result['RMW'])
        temp_cma.append(result['CMA'])
    ind_alphas_5.append(temp_alpha)
    ind_betas_5.append(temp_beta)
    ind_smbs_5.append(temp_smb)
    ind_hmls_5.append(temp_hml)
    ind_rmws_5.append(temp_rmw)
    ind_cmas_5.append(temp_cma)


def data_analyze_strat_base(strat, strat_name):
    print('CAPM base measurement')
    fig = plt.figure(figsize=(15,4))

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
    print('\nCAPM benchmark measurement')
    fig = plt.figure(figsize=(15,3))

    plt.axhspan(us_idx_alphas_c[strat]-us_idx_alphas_std_c[strat], us_idx_alphas_c[strat]+us_idx_alphas_std_c[strat], facecolor='r', alpha=0.5)
    plt.bar(fund_tickers_idx[strat], ind_idx_alphas_c[strat])
    plt.xticks(rotation=90)
    plt.title(strat_name + ' alpha under CAPM benchmark')
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
    print(f'Average corr of below stdev alpha is {np.mean(below_std_corr)} with a stdev on corr of {np.std(below_std_corr)}')
    print(f'Average corr of above stdev alpha is {np.mean(above_std_corr)} with a stdev on corr of {np.std(above_std_corr)}')

    plt.bar(below_std_mf, below_std_corr, label='below')
    plt.bar(above_std_mf, above_std_corr, label='above')
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()

def data_analyze_strat_5(strat, strat_name):
    print('\n5-factor measurement')
    fig = plt.figure(figsize=(15,6))

    plt.axhspan(np.mean(ind_alphas_5[strat])-np.std(ind_alphas_5[strat]), np.mean(ind_alphas_5[strat])+np.std(ind_alphas_5[strat]), facecolor='r', alpha=0.5)
    plt.bar(fund_tickers[strat], ind_alphas_5[strat])
    plt.xticks(rotation=90)
    plt.show()

    n=5
    largest_index = sorted(range(len(ind_alphas_5[strat])), key = lambda sub: ind_alphas_5[strat][sub])[-n:]
    best_alpha_mf = [fund_tickers[strat][i] for i in largest_index]
    print(f'The largest alpha funds by 5-factor are {best_alpha_mf}.  Note last one is best one')

    above_std_5 = [idx for idx, val in enumerate(ind_alphas_5[strat]) if val > np.mean(ind_alphas_5[strat])+np.std(ind_alphas_5[strat])]
    above_std_mf = [fund_tickers[strat][i] for i in above_std_5]
    print(f'The mutual funds that are 1 stdev above mean with 5-factor alpha are {above_std_mf}')

    below_std_5 = [idx for idx, val in enumerate(ind_alphas_5[strat]) if val < np.mean(ind_alphas_5[strat])-np.std(ind_alphas_5[strat])]
    below_std_mf = [fund_tickers[strat][i] for i in below_std_5]
    print(f'The mutual funds that are 1 stdev above mean with 5-factor alpha are {below_std_mf}')

    below_std_beta = [ind_betas_5[strat][i] for i in below_std_5]
    above_std_beta = [ind_betas_5[strat][i] for i in above_std_5]
    print(f'Average beta of below stdev alpha is {np.mean(below_std_beta)} with a stdev on beta of {np.std(below_std_beta)}')
    print(f'Average beta of above stdev alpha is {np.mean(above_std_beta)} with a stdev on beta of {np.std(above_std_beta)}')
    print('\n')

    below_std_smb = [ind_smbs_5[strat][i] for i in below_std_5]
    above_std_smb = [ind_smbs_5[strat][i] for i in above_std_5]
    print(f'Average SMB of below stdev alpha is {np.mean(below_std_smb)} with a stdev on SMB of {np.std(below_std_smb)}')
    print(f'Average SMB of above stdev alpha is {np.mean(above_std_smb)} with a stdev on SMB of {np.std(above_std_smb)}')
    print('\n')

    below_std_hml = [ind_hmls_5[strat][i] for i in below_std_5]
    above_std_hml = [ind_hmls_5[strat][i] for i in above_std_5]
    print(f'Average HML of below stdev alpha is {np.mean(below_std_hml)} with a stdev on HML of {np.std(below_std_hml)}')
    print(f'Average HML of above stdev alpha is {np.mean(above_std_hml)} with a stdev on HML of {np.std(above_std_hml)}')
    print('\n')

    below_std_rmw = [ind_rmws_5[strat][i] for i in below_std_5]
    above_std_rmw = [ind_rmws_5[strat][i] for i in above_std_5]
    print(f'Average RMW of below stdev alpha is {np.mean(below_std_rmw)} with a stdev on RMW of {np.std(below_std_rmw)}')
    print(f'Average RMW of above stdev alpha is {np.mean(above_std_rmw)} with a stdev on RMW of {np.std(above_std_rmw)}')
    print('\n')

    below_std_cma = [ind_cmas_5[strat][i] for i in below_std_5]
    above_std_cma = [ind_cmas_5[strat][i] for i in above_std_5]
    print(f'Average CMA of below stdev alpha is {np.mean(below_std_cma)} with a stdev on CMA of {np.std(below_std_cma)}')
    print(f'Average CMA of above stdev alpha is {np.mean(above_std_cma)} with a stdev on CMA of {np.std(above_std_cma)}')


def data_analyze_top(strat, strat_name):
    print('\nTop Mutual Fund')
    n=1
    largest_index = sorted(range(len(ind_idx_alphas_c[strat])), key = lambda sub: ind_idx_alphas_c[strat][sub])[-n:]
    best_alpha_mf = [fund_tickers_idx[strat][i] for i in largest_index] 
    for mf in us_eq_data[strat_name]:
        if mf['ticker'].iloc[0] in best_alpha_mf:
            best_data = mf

    print(f'Best Mutual Fund for {strat_name} is {best_alpha_mf[0]}')
    start_date = ff_df['date'][0] if ff_df['date'][0] > best_data['date'][1] else best_data['date'][1]
    start_date = start_date if start_date > us_index[strat_name]['Date'][1] else us_index[strat_name]['Date'][1]
    end_date = best_data['date'].iloc[-1]
    capm_result = capm(best_data, ff_df)
    bench_result = capm_index(best_data, ff_df, us_index[strat_name], start_date, end_date)
    three_result = reg_date_range(best_data, ff_df, ['Mkt-RF', 'SMB', 'HML'], start_date, end_date)
    five_result = reg_date_range(best_data, ff_df, ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'], start_date, end_date)   
    print(f'CAPM regression result is {capm_result}')
    print(f'Benchmark CAPM regression result is {bench_result}')
    print(f'3-Factor regression result is {three_result}')
    print(f'5-Factor regression result is {five_result}')

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,2))
    ax1.bar(['CAPM', 'Benchmark CAPM', '3-Factor', '5-Factor'], [capm_result['const'], bench_result['const'], three_result['const'], five_result['const']])
    ax2.bar(['CAPM', 'Benchmark CAPM', '3-Factor', '5-Factor'], [capm_result['Mkt-RF'], bench_result['beta'], three_result['Mkt-RF'], five_result['Mkt-RF']])
    fig.autofmt_xdate(rotation=20)
    ax1.set_title('Alpha')
    ax2.set_title('Beta')
    fig.suptitle(f'{strat_name} Top Performer: {best_alpha_mf[0]}', y=1.05)
    plt.show()




