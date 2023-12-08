import pandas as pd
import numpy as np
import statsmodels.api as sm


def ff_3(eq_data, ff_df):
    alphas = []
    betas = []
    smbs = []
    hmls = []
    eq_names = []

    for eq_type, type_data in eq_data.items():
        start_date = ff_df['date'][0] if ff_df['date'][0] > type_data['date'][1] else type_data['date'][1]
        end_date = type_data['date'].iloc[-1]
        temp_ff = ff_df[(ff_df['date'] >= start_date) & (ff_df['date'] <= end_date)].reset_index().drop('index', axis=1)
        type_data = type_data[type_data['date'] >= start_date].reset_index().drop('index', axis=1)

        x = sm.add_constant(temp_ff[['Mkt-RF', 'HML', 'SMB']])
        y = type_data['nav_return']*100 - temp_ff['RF']
        model = sm.OLS(y, x).fit(cov_type='HC0')

        alphas.append(model.params['const'])
        betas.append(model.params['Mkt-RF'])
        smbs.append(model.params['SMB'])
        hmls.append(model.params['HML'])
        eq_names.append(eq_type)
    
    return alphas, betas, eq_names, smbs, hmls

def ff_5(eq_data, ff_df):
    alphas = []
    betas = []
    smbs = []
    hmls = []
    rmws = []
    cmas = []
    eq_names = []

    for eq_type, type_data in eq_data.items():
        start_date = ff_df['date'][0] if ff_df['date'][0] > type_data['date'][1] else type_data['date'][1]
        end_date = type_data['date'].iloc[-1]
        temp_ff = ff_df[(ff_df['date'] >= start_date) & (ff_df['date'] <= end_date)].reset_index().drop('index', axis=1)
        type_data = type_data[type_data['date'] >= start_date].reset_index().drop('index', axis=1)

        x = sm.add_constant(temp_ff[['Mkt-RF', 'HML', 'SMB', 'RMW', 'CMA']])
        y = type_data['nav_return']*100 - temp_ff['RF']
        model = sm.OLS(y, x).fit(cov_type='HC0')


        alphas.append(model.params['const'])
        betas.append(model.params['Mkt-RF'])
        smbs.append(model.params['SMB'])
        hmls.append(model.params['HML'])
        rmws.append(model.params['RMW'])
        cmas.append(model.params['CMA'])
        eq_names.append(eq_type)
    
    return alphas, betas, eq_names, smbs, hmls, rmws, cmas

def capm(eq_data, ff_df):

    start_date = ff_df['date'][0] if ff_df['date'][0] > eq_data['date'][1] else eq_data['date'][1]
    end_date = eq_data['date'].iloc[-1]

    return reg_date_range(eq_data, ff_df, ['Mkt-RF'], start_date, end_date)

def reg_date_range(eq_data, ff_df, ff_factors, start_date, end_date):
    '''
    eq_data: df
    ff_df: df
    ff_factors: list of strings
    start_date: date in string
    end_date: date in string
    '''
    results = {}
    type_data = eq_data.copy()
    temp_ff = ff_df[(ff_df['date'] >= start_date) & (ff_df['date'] <= end_date)].reset_index().drop('index', axis=1)
    type_data = type_data[(type_data['date'] >= start_date) & (type_data['date'] <= end_date)].reset_index().drop('index', axis=1)

    x = sm.add_constant(temp_ff[ff_factors])
    y = type_data['nav_return']*100 - temp_ff['RF']
    if len(temp_ff) == len(type_data):
        model = sm.OLS(y, x).fit(cov_type='HC0')

        ff_factors.insert(0, 'const')
        for factor in ff_factors:
            results[factor]=model.params[factor]
        return results
    return None

def capm_index(eq_data, ff_df, index_df, start_date, end_date):
    '''
    eq_data: df
    ff_df: df
    index_df: list of strings
    start_date: date in string
    end_date: date in string
    '''
    type_data = eq_data.copy()
    temp_ff = ff_df[(ff_df['date'] >= start_date) & (ff_df['date'] <= end_date)].reset_index().drop('index', axis=1)
    type_data = type_data[(type_data['date'] >= start_date) & (type_data['date'] <= end_date)].reset_index().drop('index', axis=1)
    index_data = index_df[(index_df['Date'] >= start_date) & (index_df['Date'] <= end_date)].reset_index().drop('index', axis=1)
    
    if len(temp_ff) == len(type_data) == len(index_data):
        x = sm.add_constant(index_data['% Change'] - temp_ff['RF'])
        y = type_data['nav_return']*100 - temp_ff['RF']
        model = sm.OLS(y, x).fit(cov_type='HC0')

        results = {'const': model.params['const'], 'beta': model.params[0]}
        return results
    return None

def corr_index(eq_data, index_df, start_date, end_date):
    '''
    eq_data: df
    ff_df: df
    index_df: list of strings
    start_date: date in string
    end_date: date in string
    '''
    type_data = eq_data.copy()
    type_data = type_data[(type_data['date'] >= start_date) & (type_data['date'] <= end_date)].reset_index().drop('index', axis=1)
    index_data = index_df[(index_df['Date'] >= start_date) & (index_df['Date'] <= end_date)].reset_index().drop('index', axis=1)

    index_pct_change = index_data['% Change']
    mf_pct_change = type_data['nav_return']*100

    return np.corrcoef(index_pct_change, mf_pct_change)[0][1]
