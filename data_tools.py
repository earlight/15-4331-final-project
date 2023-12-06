import pandas as pd
import numpy as np
import statsmodels.api as sm

def capm(eq_data, ff_df):
    alphas = []
    betas = []
    eq_names = []

    for eq_type, type_data in eq_data.items():
        start_date = ff_df['date'][0] if ff_df['date'][0] > type_data['date'][1] else type_data['date'][1]
        end_date = type_data['date'].iloc[-1]
        temp_ff = ff_df[(ff_df['date'] >= start_date) & (ff_df['date'] <= end_date)].reset_index().drop('index', axis=1)
        type_data = type_data[type_data['date'] >= start_date].reset_index().drop('index', axis=1)

        x = sm.add_constant(temp_ff['Mkt-RF'])
        y = type_data['nav_return']*100 - temp_ff['RF']
        model = sm.OLS(y, x).fit(cov_type='HC0')
        # print(f'Below are the CAPM results for {eq_type}')
        # print(f"Alpha: {model.params['const']}")
        # print(f"Beta: {model.params['Mkt-RF']}")
        # print('\n')

        alphas.append(model.params['const'])
        betas.append(model.params['Mkt-RF'])
        eq_names.append(eq_type)
    
    return alphas, betas, eq_names