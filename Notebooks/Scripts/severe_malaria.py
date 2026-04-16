import sys
sys.path.append('..')
from Edinburgh_Model.model import *
from Edinburgh_Model.simulate_single_vaccine import *

from . import expectation_model as em
from . import paper

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

default_pars = Parameters()


def reyburn_data():
    data = pd.read_csv('Data/rspb20142657supp2.csv')
    data['transmission'] = data['altitude band'].replace({1: 'high', 2: 'medium', 3: 'low'})
    data['died'] = data['died'].replace({0: 'no', 1: 'yes'})

    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.ecdfplot(data=data[data['age (years)'] < 40], x='age (years)', hue='transmission', ax=ax)
    ax.set_ylabel('Cumulative proportion of\nsevere malaria episodes');
    return fig, ax

def get_draws(draws, variables, groups=None, wide=False):

    if groups is not None:
        vars = ['draw']
        for v in variables:
            for p in range(1, len(groups)+1):
                vars += [f'{v}[{p}]']
    else:
        vars = ['draw'] + variables

    df = draws[vars].melt(id_vars='draw', var_name='tmp', value_name='value')
    df[['parameter', 'group']] = df['tmp'].str.extract(r'(?P<text>[^\[]+)\[(?P<digits>\d+)\]')
    df['parameter'] = df['parameter'].str.strip()

    if not wide:
        return df.drop(columns=['tmp'])

    # pivot so that parameter values become columns (one row per draw + group)
    df_wide = df.pivot_table(index=['group', 'draw'], columns='parameter', values='value').reset_index()
    df_wide.columns.name = None
    df_wide = df_wide.set_index('draw')
    return df_wide

def plot_model(pars1=None, pars2=None, legend=True):
    a = np.linspace(0, 10, 101)[:, np.newaxis] # age in years

    if pars1:
        l1, n1 = em.infections(a, pars1)
    if pars2:
        l2, n2 = em.infections(a, pars2)

    mosaic = """
        .ab
        cde
        fgh
    """
    fig, axs = plt.subplot_mosaic(mosaic, figsize=(9, 9))
    # fig, axs = plt.subplots(3, 5, figsize=(15, 3))
    # fig.subplots_adjust(wspace=0.4)
    ax = axs['a']
    if pars1:
        ax.plot(a, l1)
    if pars2:
        ax.plot(a, l2)
    ax.scatter(0, 0, color='w')  # to force y-axis to start at 0
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Blood infections per year')

    ax = axs['b']
    if pars1:
        ax.plot(a, n1)
    if pars2:
        ax.plot(a, n2)
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Cumulative number of infections')

    ax = axs['c']
    ni = np.arange(150)[:, np.newaxis]
    if pars1:
        ax.plot(ni, em.clinical_model(ni, pars1), label='Exponential')
    if pars2:
        ax.plot(ni, em.clinical_model(ni, pars2), label='Sigmoidal')
    ax.scatter(0, 0, color='w')  # to force y-axis to start at 0
    ax.set_xlabel('Number of blood infections')
    ax.set_ylabel('Risk of clinical episode');
    if legend:
        ax.legend()

    ax = axs['d']
    if pars1:
        ax.plot(a, em.clinical_cases(a, pars1), label='Exponential')
    if pars2:
        ax.plot(a, em.clinical_cases(a, pars2), label='Sigmoidal')
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Clinical episodes per year');
    if legend:
        ax.legend()

    ax = axs['e']
    if pars1:
        C = np.array([em.cumulative_clinical_cases(0, a1, pars1) for a1 in a]).reshape(len(a), -1)
        ax.plot(a, C, label='Exponential')
    if pars2:
        C = np.array([em.cumulative_clinical_cases(0, a1, pars2) for a1 in a]).reshape(len(a), -1)
        ax.plot(a, C, label='Sigmoidal')
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Cumulative clinical episodes');
    if legend:
        ax.legend()

    ax = axs['f']
    ni = np.arange(150)[:, np.newaxis]
    if pars1:
        ax.plot(ni, em.severe_model(ni, pars1), label='Exponential')
    if pars2:
        ax.plot(ni, em.severe_model(ni, pars2), label='Sigmoidal')
    ax.scatter(0, 0, color='w')  # to force y-axis to start at 0
    ax.set_xlabel('Number of blood infections')
    ax.set_ylabel('Risk of severe episode');
    if legend:
        ax.legend()

    ax = axs['g']
    if pars1:
        ax.plot(a, em.severe_cases(a, pars1), label='Exponential')
    if pars2:
        ax.plot(a, em.severe_cases(a, pars2), label='Sigmoidal')
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Clinical episodes per year');
    if legend:
        ax.legend()

    ax = axs['h']
    years = np.arange(11)
    if pars1:
        C = [em.cumulative_severe_cases(a1, a2, pars1) for a1, a2 in zip(years[:-1], years[1:])]
        ax.plot(years[:-1], C, label='Exponential')
    if pars2:
        C = [em.cumulative_severe_cases(a1, a2, pars2) for a1, a2 in zip(years[:-1], years[1:])]
        ax.plot(years[:-1], C, label='Exponential')
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Cumulative severe episodes');
    if legend:
        ax.legend()

    # for ax, i in zip(axs, 'ABCDE'):
    #     ax.set_title(i, loc='left')
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['top'].set_visible(False)

    return fig, axs

def severe_malaria_risk_plot():
    n = np.arange(1, 15, 1) # range of number of infections
    # risk of severe malaria on nth infection at age a
    y = lambda a, n: 0.042 * np.exp(-(0.12 + 0.05*a)*(n-1))

    fig, ax = plt.subplots(figsize=(3, 3))

    ax.plot(n, y(1, n), marker='o', lw=0.2, label=1)
    ax.plot(n, y(5, n), marker='s', lw=0.2, label=5)
    ax.plot(n, y(10, n), marker='^', lw=0.2, label=10)

    ax.set_xticks(n)
    ax.set_ylabel('Risk of severe malaria')
    ax.set_xlabel('Number of infections')
    ax.legend(title='Age (years)', loc='upper right')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return fig, ax

def ε_severe_calibration(fig_xscale=1, fig_yscale=1, legend_pos='last', legend_title='') -> 'tuple':
    model = Model()
    model.pars = Parameters(
        λ=1,
        study_months=20*12,
    )
    model.variables = Variables(['Severe malaria', 'Direct deaths'])
    model.measures = Measures(['cases', 'cdf'])
    config = {'time_scale':'Years', 'fig_yscale':fig_yscale, 'fig_xscale':fig_xscale}
    model.Config(**config)

    rs = []
    c = []
    ε_severes = np.arange(0, 0.12, 0.025)
    for ε_severe in ε_severes:
        c.append(f'{ε_severe:.3g}')
        model.pars.control = c[-1]
        model.pars.ε_severe = ε_severe
        model.pars.update_pars('ε_severe')
        rs.append(simulate_single_vaccine(model).set_index('ages'))

    df_joined = pd.concat(rs, axis=1).reset_index(drop=False)
    return paper.plot(model, c, sim_source=df_joined, legend_title=legend_title, legend_pos=legend_pos)

def validate_severe_malaria_risk():
    model = Model()
    model.pars = Parameters(
        λ=1,
        study_months=15*12,
    )
    model.variables = Variables(['Severe malaria'])
    model.measures = Measures(['cases'])
    config = {'time_scale':'Years'}
    model.Config(**config)
    model.Sources()

    r = simulate_single_vaccine(model)
    f = 'control Severe malaria cases'
    fig, axs = plt.subplots(2, 3, figsize=(7, 4), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.05, hspace=0.3)

    for λ, ax in zip([0.5, 1, 2, 4, 8, 12][::-1], axs.flatten()):
        model.pars.λ = λ
        model.pars.update_pars('λ')
        r = simulate_single_vaccine(model)
        r['percent'] = r[f] / r[f].sum()*100
        ax.bar(r.index, r['percent'], label=f'λ={λ}')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title(f'{λ} {"infection" if λ==1 else "infections"} per year', loc='left')

    for i in range(2):
        axs[i, 0].set_ylabel('Percentage of total\nsevere episodes')
    for i in range(3):
        axs[1, i].set_xlabel('Age (years)')
    return fig, axs

def imbert():
    model = Model()
    model.pars = Parameters(
        λ=0.05,
        study_months=31*12,
        control='0.05'
    )
    model.variables = Variables(['Severe malaria'])
    model.measures = Measures(['cases'])
    config = {'time_scale':'Years'}
    model.Config(**config)
    model.Sources()

    r = simulate_single_vaccine(model)

    fig, ax = paper.plot(model, [model.pars.control], r, legend_title='Infections per year')
    ax[0, 0].set_ylabel('Annual severe malaria\nepsiodes per 100,000')
    ax[0, 0].set_xlabel('Age (years)')
    return fig, ax
