#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from dateutil.relativedelta import relativedelta

import warnings
warnings.filterwarnings('ignore')

import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.rcParams['font.family'] = 'Yu Mincho'
from IPython.core.display import display
get_ipython().run_line_magic('matplotlib', 'inline')

# specify the place of data
WORK_DIR = ''
DATA_DIR = ''

data_for_analysis_loaded = pd.read_pickle(
    f'{DATA_DIR}data_for_analysis_fin.pickle'
)


# In[ ]:


independent_variables_names = [
    'market_beta', "market cap", 'PBR', 'leverage',
    'Dummy', 'MA25', 'PER'
]
columns_to_use = [
    'Sector', 'Return', "Exc Return", 'Ear Rate', 'Market Rate'
] + independent_variables_names

data_for_analysis=data_for_analysis_loaded.assign(
    Excess_Return=lambda x: x["Exc Return"].groupby(level=0).shift(-1),
    Return_dat =lambda x: x['Return'].groupby(level=0).shift(-1)  
)[columns_to_use]


# In[ ]:


# cofirm the calculation with 5 tickers each
data_for_analysis.dropna(
    subset=independent_variables_names
).groupby(level=0).head()


# In[ ]:


exclude_fin = data_for_analysis[
    data_for_analysis['Sector'].apply(
        lambda x: x not in ['Bank', 'Security・Forward', 'Insurance', 'Others']
    )
]


# In[ ]:


data_tmp = exclude_fin.dropna(
    subset=independent_variables_names
).xs(pd.datetime(2016,6,7), level=1)  
exog = sm.add_constant(data_tmp[independent_variables_names])
endog = data_tmp["Exc return"]
sm.OLS(endog, exog).fit().summary()


# In[ ]:


data_tmp = exclude_fin.dropna(
    subset=independent_variables_names
).xs(pd.datetime(2017,7,6), level=1)  
exog = sm.add_constant(data_tmp[independent_variables_names])
endog = data_tmp['Nex Exc return']
sm.OLS(endog, exog).fit().summary()


# In[ ]:


def cross_sectional_regression_overtime(
    data_with_excess_returns,
    endog_name, exog_names
):
    group_by_date = data_with_excess_returns.groupby('Time')
    
    results = []
    for date_point, values in tqdm(group_by_date):
        
        result = cross_sectional_regression(
            values,
            endog_name,
            exog_names
        )
        if result is None:
            continue
        results.append(result)
    
    results = pd.concat(results)
    return results


# In[ ]:


def cross_sectional_regression(data, endog_name, exog_names):
    data = data.reset_index()
    data = data.dropna(subset=endog_name + exog_names)

    if data.shape[0] < 1:  # ignore empty dataframe
        return None

    end_date = data["time"].max()

    endog = data[endog_name]

    exog = data[exog_names]
    exog = exog.assign(constant=1)

    model = sm.OLS(endog, exog)

    result = model.fit()
    betas = result.params.rename(end_date)

    result = pd.DataFrame(betas).T

    return result


# In[ ]:


def calculate_mean_value_of_coefficients(coefficients):
    mean = coefficients.mean().rename('mean')
    std_err = (
        coefficients.std() / np.sqrt(coefficients.shape[0])
    ).rename('std err')
    t_stat = (mean / std_err).rename('t-stat')
    
    result = pd.concat([mean, std_err, t_stat], axis=1)
    
    return result


# In[ ]:


coefficients_excluding_fin = cross_sectional_regression_overtime(
    exclude_fin, endog_name=['Nex Exc return'],
    exog_names=independent_variables_names
)
display(coefficients_excluding_fin.head())  


# In[ ]:


result = calculate_mean_value_of_coefficients(
    coefficients_excluding_fin
)

display(result)


# In[ ]:


result[['mean']].applymap(lambda x: '{:.2%}'.format(x * 250))  


# In[ ]:


window_size = 50
rolling_coefficients = coefficients_excluding_fin.rolling(
    window_size
).mean()

rolling_std_err = (
    coefficients_excluding_fin.rolling(window_size).std()
    / np.sqrt(window_size)
)

rolling_coefficients.index.rename('日付', inplace=True)

fig, axes=plt.subplots(4, 2, figsize=(12, 12))

for col, ax in zip(rolling_coefficients.columns, axes.flatten()):
    sns.lineplot(data=rolling_coefficients[col], ax=ax)
    sns.lineplot(
        data=rolling_coefficients[col] + 2 * rolling_std_err[col],
        ax=ax,
        color='gray'
    )
    sns.lineplot(
        data=rolling_coefficients[col] - 2 * rolling_std_err[col],
        ax=ax,
        color='gray'
    )
    ax.hlines( 
        0, *ax.get_xlim(), 
        linestyles=':', alpha=.5
    )    
    for line in ax.get_lines()[1:]:
        line.set_linestyle('--')
    ax.set_title(col if col != 'PER' else 'E/P ratio')

fig.tight_layout() 


# In[ ]:


def create_portfolio_by_one_variable(
    data,
    sort_by,
    q,
    labels=None,
    group_name=None
):
    group_by_date = data.groupby('time')
    
    if isinstance(q, int) and labels is None:
        labels = range(q)

    values = []
    for date, value in group_by_date:
        if value[sort_by].isnull().all(): # ignore empty dataframe
            continue
            
        value = value.assign(
            quantile=lambda x: pd.qcut(
                x[sort_by], q, labels=labels
            )
        )
        
        if group_name is not None:
            value.rename(columns={'quantile':group_name}, inplace=True)
            
        values.append(value)
    
    return pd.concat(values)


# In[ ]:


portfolio_by_mv = create_portfolio_by_one_variable(
    data=exclude_fin,
    sort_by="MA25",
    q=5,
    labels=range(5),
    group_name='MV_quantile'
)


# In[ ]:


portfolio_by_mv.head()


# In[ ]:


portfolio_returns = portfolio_by_mv.groupby(
    ['MV_quantile', 'time']
)['return'].mean()


# In[ ]:


market_return = exclude_fin.groupby(
    'time'
)['return'].mean().rename("ave return")


# In[ ]:


fig, ax = plt.subplots(figsize=(12, 5))

to_plot=portfolio_returns.unstack().T.filter(
    items=[0, 4]
).rename(
    columns={0:'deviation rate low', 4:'deviation rate high'}
).join(
    market_return
).dropna(
    how='all'
).apply(
    lambda column: np.log((column + 1).cumprod())
)
to_plot.columns.rename('portfolio', inplace=True)

sns.lineplot(
    data=to_plot.stack().reset_index(name='cum return'),
    x='date',
    y='cum return',
    hue='portfolio',
    style='portfolo',
    palette='gray',
    ax=ax
)


# In[ ]:


# construct portfolio
portfolio_by_per = create_portfolio_by_one_variable(
    data=exclude_fin,
    sort_by='PER',
    q=5,
    labels=range(5),
    group_name='PER_quantile'
)

# culculate cumulative return
portfolio_returns = portfolio_by_per.groupby(
    ['PER_quantile', 'date']
)['return'].mean()

to_plot=portfolio_returns.unstack().T.filter(
    items=[0, 4]
).rename(
    columns={0:'lowE/P ratio', 4:'highE/P ratio'}
).join(
    market_return
).dropna(
    how='all'
).apply(
    lambda column: np.log((column + 1).cumprod())
)
to_plot.columns.rename('portfolio', inplace=True)
_, ax=plt.subplots(figsize=(12, 5))
sns.lineplot(
    data=to_plot.stack().reset_index(name='cum return'),
    x='date',
    y="cum return",
    hue='portfolio',
    style='portfolio',
    palette='gray',
    ax=ax
)


# In[ ]:


fig, ax = plt.subplots(figsize=(12, 5))
handles, labels = [], []
hml = portfolio_returns.loc[4].rolling(25).sum() - portfolio_returns.loc[0].rolling(25).sum()
ax=hml.rename('undervalue・growth gap).plot(color='k', ax=ax)
handle, label= ax.get_legend_handles_labels()
handles.extend(handle)
labels.extend(label)

ax=(np.log(market_return.apply(lambda x:x**2).rolling(10).sum()[hml.index])).plot(
    ax=ax, secondary_y=True, linestyle=':', color='gray'
)
handle, label= ax.get_legend_handles_labels()
handles.extend(handle)
labels.extend(label)

ax.set_title('undervalue・gap of performance and volatility')
ax.legend(handles, labels, bbox_to_anchor=(.5,-.2), ncol=2, loc='upper center')


# In[ ]:


def calculate_predicted_values(
    exog,
    shifted_coefficients,
    predicted_value_label='predicted_value'
):
    predicted_returns=[]
    
    group_by_date = exog.groupby(level=1)
    for date, group in tqdm(group_by_date):
        try:
            coefficients_ = shifted_coefficients.xs(date).T

            predicted = np.dot(group.values, coefficients_)
            predicted_returns.append(
                pd.DataFrame(
                    predicted,
                    columns=[predicted_value_label],
                    index=group.index
                )
            )
        except KeyError:
            continue

    predicted_returns = pd.concat(predicted_returns).sort_index()

    return predicted_returns


# In[ ]:


X = exclude_fin.assign(constant=1)[coefficients_excluding_fin.columns]


# In[ ]:


predicted = calculate_predicted_values(
    X,
    coefficients_excluding_fin.shift(),
    'expect return'
)

data_with_predicted_values = predicted.join(exclude_fin)

data_with_predicted_values.head()


# In[ ]:


portfolio_by_predicted = create_portfolio_by_one_variable(
    data=data_with_predicted_values,
    sort_by='expect return',
    q=5,
    labels=range(5),
    group_name='predicted_quantile'
)


# In[ ]:


#calculate cum return
portfolio_returns = portfolio_by_predicted.groupby(
    ['predicted_quantile', 'date']
)['return'].mean()

fig, ax = plt.subplots(figsize=(12, 5))

to_plot=portfolio_returns.unstack().T.filter(
    items=[0, 4]
).rename(
    columns={0:'prediction low', 4:'prediction high'}
).join(
    market_return
).dropna(
    how='all'
).apply(
    lambda column: np.log((column + 1).cumprod())
)
to_plot.columns.rename('portfolio', inplace=True)
sns.lineplot(
    data=to_plot.stack().reset_index(name='cum return'),
    x='date',
    y='cum return',
    hue='portfolio',
    style='portfolio',
    palette='gray',
    ax=ax
)


ax.set_title('portfolio return based on prediction')
ax.legend(loc='upper left')


# In[ ]:


fig, ax = plt.subplots(figsize=(12, 5))

pos_ = portfolio_by_predicted[
    (portfolio_by_predicted['exp return'] > 0)
].groupby('date')['return'].mean() + 1

neg_ = portfolio_by_predicted[
    (portfolio_by_predicted['exp return']<=0)
].groupby('date')['return'].mean()+1

to_plot = pd.concat(
    [
        np.log(pos_).cumsum().rename('prediction pos'),
        np.log(neg_).cumsum().rename('prediction neg'),
        np.log(market_return[neg_.index] + 1).cumsum()
    ],
    axis=1
)
to_plot.columns.rename('portfolio', inplace=True)
sns.lineplot(
    data=to_plot.stack().reset_index(name='cum return'),
    x='date',
    y='cum return',
    hue='portfolio',
    style='portfolio',
    palette='gray',
    ax=ax
)


# In[ ]:


test_predicted = cross_sectional_regression_overtime(
    data_with_predicted_values,
    endog_name=['exc return'],
    exog_names=['exp return']
)

calculate_mean_value_of_coefficients(test_predicted)


# In[ ]:


data_for_analysis_loaded

