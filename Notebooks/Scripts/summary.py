import sys
sys.path.append('..')
from Edinburgh_Model.model import *

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pars = Parameters()

def tanzania_all_cause_mortality():
    """
        All-cause death rates from Tanzania 2019
    """
    def extract_age_range(age_group):
        """ Extract age range from 'Age Group' and create 'start' and 'end' columns """
        if '<' in age_group:
            start = 0
            m = re.match(r'<(\d+)', age_group)
            if m is not None:
                end = int(m.group(1))-1
        elif '-' in age_group:
            m = re.match(r'(\d+)-(\d+)', age_group)
            if m is not None:
                start = int(m.group(1))
                end = int(m.group(2))
        elif '+' in age_group:
            m = re.match(r'(\d+)', age_group)
            if m is not None:
                start = int(m.group(1))
                end = 100
        return pd.Series({'start': start, 'end': end})

    lt = pd.read_csv('Data/life_table_tanzania.csv', header=[0, 1])
    lt = lt.rename(columns={'Unnamed: 1_level_0': ''})

    lt[['start', 'end']] = lt[('', 'Age Group')].apply(extract_age_range)
    r = lt[[('', 'Age Group'), ('2019', 'Both sexes'), ('start', ''), ('end', '')]].iloc[:19]
    r.columns = ['age', 'rate', 'start', 'end']
    r['start'] *= 52
    r['rate'] /= 52
    r = r.set_index('start')
    r = pd.concat([r, pd.DataFrame({'rate': 0.01}, index=[5200])])

    age = np.arange(5200)

    y = np.interp(np.arange(5200), r.index, r['rate'])
    pa = 1 - y

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(age/52, 100*pa.cumprod(), label='pa')
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Percent survival')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return fig, ax

def clinical_malaria_risk_plot():
    n = np.arange(1, 200, 10) # range of number of infections
    # risk of clinical malaria on nth infection
    y = lambda n:  pars.ρ_clinical * (exp(pars.δ_clinical*pars.γ_clinical) - 1) / (exp(pars.δ_clinical*pars.γ_clinical) - 2 + exp(pars.δ_clinical*(n-1)))

    fig, ax = plt.subplots(figsize=(3, 3))

    ax.plot(n, y(n), marker='o', lw=0.2)

    ax.set_ylabel('Risk of clinical malaria')
    ax.set_xlabel('Number of infections')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(visible=True, which='both', axis='both', color='lightgrey', linestyle='--', linewidth=0.5)

    return fig, ax

def imperial_vaccine_protection():
    a = np.arange(10*52, dtype=float)

    rtss = 100*Imperial_RTSS_vaccination(a, (1, 1, 1, 1))
    r21 = 100*Imperial_R21_vaccination(a, (1, 1, 1, 1))

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(a/52, rtss, label='RTS,S')
    ax.plot(a/52, r21, label='R21', ls='--')
    ax.set_xlabel('Years post last primary vaccination')
    ax.set_ylabel('Protection against\nblood infection (%)')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(visible=True, which='both', axis='both', color='lightgrey', linestyle='--', linewidth=0.5)
    ax.legend()

    return fig, ax

def SMC_protection():
    a = np.arange(52, dtype=float)

    smc_args = pars.smc_ramp, pars.smc_rounds, pars.smc_repeats, pars.smc_coverage
    smc_profile = 100*smc_protection(a, smc_args)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(a, smc_profile)
    ax.set_xlabel('Weeks')
    ax.set_ylabel('Protection against\nblood infection (%)')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(visible=True, which='both', axis='both', color='lightgrey', linestyle='--', linewidth=0.5)

    return fig, ax

def severe_malaria_risk_plot():
    n = np.arange(1, 15, 1) # range of number of infections
    # risk of severe malaria on nth infection at age a
    y = lambda a, n: pars.ρ_severe * np.exp(-(pars.δ_severe + pars.ε_severe*a)*(n-1))

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
