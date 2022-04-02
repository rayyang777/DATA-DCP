import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import dags.bingo.ltv.cfit_tools as cftool
import os
import datetime

def func_arppu(x, a, b):
    return a*np.power(x+0.5, b)

def func_prate(x, a, b, c):
    return a/(b+np.power(x+0.5, c))
    
def fit_func( data):
    if 'country' in data.columns and 'campaign' in data.columns:
        result = pd.DataFrame(columns=('country','avg_cpi', 'campaign', 'users', 'analysis_day', 'iap', 'iap_users', 'arppu', 'prate', 'real_ltv', 'nday', 'arppu_y2', 'arppu_r2', 'prate_y2', 'prate_r2', 'ltv', 'ltv_low', 'ltv_high', 'r2'))
        grouped = data.groupby(['country','campaign'],as_index=False).apply(lambda x:x if x['arppu'].mean()!=0 else None).dropna().groupby(['country','campaign'],as_index=False)
    else:
        result = pd.DataFrame(columns=('campaign','avg_cpi', 'users', 'analysis_day', 'iap', 'iap_users', 'arppu', 'prate', 'real_ltv', 'nday', 'arppu_y2', 'arppu_r2', 'prate_y2', 'prate_r2', 'ltv', 'ltv_low', 'ltv_high', 'r2'))
        grouped = data.groupby(['campaign']).apply(lambda x:x if x['arppu'].mean()!=0 else None).dropna().groupby(['campaign'])
    
    total = len(grouped)
    num = 0
    for index,groups in grouped:
        
        x = groups['nday'].values.tolist()
        arppu_y = groups['arppu'].values.tolist()
        iap_rate_y = groups['pay_rate'].values.tolist()
        

        if len(x) == 0:
            continue
        
        try:
            arppu_popt, arppu_pcov = curve_fit(func_arppu, x, arppu_y, bounds=(0, [200, 1.]), maxfev=1000)
            iap_rate_popt, iap_rate_pcov = curve_fit(func_prate, x, iap_rate_y, bounds=([0, 0, -np.inf], [np.inf, np.inf, -0.83]), maxfev=1000)
            x2 = np.arange(0,180,dtype=np.float)

            arppu_ci, arppu_pi = cftool.get_interval(func_arppu, np.array(x), np.array(arppu_y), arppu_popt, arppu_pcov, x2)
            arppu_y2 = func_arppu(x2, *arppu_popt)
            arppu_r2 = cftool.get_r2(func_arppu, np.array(x), np.array(arppu_y), arppu_popt)

                
            
            iap_rate_ci, iap_rate_pi = cftool.get_interval(func_prate, np.array(x), np.array(iap_rate_y), iap_rate_popt, iap_rate_pcov, x2)
            iap_rate_y2 = func_prate(x2, *iap_rate_popt)
            iap_rate_r2 = cftool.get_r2(func_prate, np.array(x), np.array(iap_rate_y), iap_rate_popt)

            max_nday = int(max(groups['nday'].values.tolist()))
            
            # discount
            discount = 1. / func_prate(x2, *iap_rate_popt) * func_prate(max_nday, *iap_rate_popt)

            # ltv cal
            ltv = arppu_y2 * iap_rate_y2 * discount
            ltv_down = (arppu_y2 - arppu_pi) * (iap_rate_y2 - iap_rate_pi) * discount
            ltv_up = (arppu_y2 + arppu_pi) * (iap_rate_y2 + iap_rate_pi) * discount

            ltv_real = np.array(arppu_y) * np.array(iap_rate_y)
        except Exception as e:
            print("skip", index, '没有拟合出最佳参数', e)
            continue

        analysis_day = int(max(groups['analysis_day'].values.tolist()))

        result_dict = {}
        if 'country' in data.columns:
            result_dict['country'] = [index[0] for i in range(0, len(x2))]
            result_dict['campaign'] = [index[1] for i in range(0, len(x2))]
        else:
            result_dict['campaign'] = [index for i in range(0, len(x2))]
        
        none_data = np.zeros(len(x2) - max_nday - 1)
        result_dict['users'] = np.append(groups['users'], none_data)
        result_dict['analysis_day'] = np.append(groups['analysis_day'], none_data)
        result_dict['iap'] = np.append(groups['iap'], none_data)
        result_dict['avg_cpi'] = np.append(groups['avg_cpi'],none_data)
                
        result_dict['iap_users'] = np.append(groups['iap_users'], none_data)
        result_dict['arppu'] = np.append(groups['arppu'], none_data)
        result_dict['prate'] = np.append(groups['pay_rate'], none_data)
        result_dict['real_ltv'] = np.append(ltv_real, none_data)

        result_dict['nday'] = x2.tolist()
        result_dict['arppu_y2'] = arppu_y2.tolist()
        result_dict['arppu_r2'] = [arppu_r2 for i in range(len(x2))]
        result_dict['prate_y2'] = iap_rate_y2.tolist()
        result_dict['prate_r2'] = [iap_rate_r2 for i in range(len(x2))]
        result_dict['ltv'] = ltv.tolist()
        result_dict['ltv_low'] = ltv_down.tolist()
        result_dict['ltv_high'] = ltv_up.tolist()
        result_dict['r2'] = [arppu_r2 * iap_rate_r2 for i in range(len(x2))]

        result = result.append(pd.DataFrame(data=result_dict), ignore_index=True)
        num = num + 1
    print(f'总共 {total}, 成功{num}')
    return result     


# def start():
#     with open("ltv_campaign_country.csv", "groups") as f:
#         data = pd.read_csv(f)
#         country_campaign_groups = data.groupby(['country', 'campaign']).groups
#         country_campaign_data = fit_func(country_campaign_groups, data)
#         country_campaign_data.to_csv('country_campaign_ltv.csv')
# start()

# with open("ltv_campaign.csv", 'groups') as f:
#     data = pd.read_csv(f)
#     campaign_groups = data.groupby(['campaign']).groups
#     campaign_data = fit_func(campaign_groups, data)
#     campaign_data.to_csv('./campaign_ltv.csv')