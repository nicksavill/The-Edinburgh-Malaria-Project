import pickle
import arviz as az
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from collections import namedtuple
from numpy import exp, log, quantile

import warnings
warnings.simplefilter("ignore")

Pars_sig = namedtuple('Parameters', 'λ ρ_elp τ_elp δ_clinical ρ_clinical γ_clinical')
Pars_exp = namedtuple('Parameters', 'λ ρ_elp τ_elp δ_clinical ρ_clinical')

mid_ages = [0.5, 2.5, 5.5, 8.5, 12.5, 20.5]
name_conversions = {
    'lambda':'λ (infections per child per year)',
    'rho_elp':'ρ_elp',
    'tau_elp':'τ_elp (years)',
    'delta_clinical':'δ_clinical (per infection)',
    'rho_clinical':'ρ_clinical',
    'gamma_clinical':'γ_clinical (infections)'
}
name_conversions_hierarchical = {
    'lambda':'λ (infections per year)',
    'tau_elp':'τ_elp (years)',
    'delta_clinical':'δ_clinical (per infection)',
    'rho_clinical':'ρ_clinical',
    'gamma_clinical':'γ_clinical (infections)'
}

###################################### INDIVIDUAL CASE DATA ###########################

def individual_episode_data(rates_file):
    # Read in the clinical cases by age category for each child from Trape et al. 2024 Supplementary Table 2

    rates_by_age = pd.read_csv(rates_file)
    rates_by_age = rates_by_age.drop(columns=['ID'])
    rates_by_age.columns = mid_ages + ['total']

    q = quantile(rates_by_age['total'], q=(0, 0.25, 0.5, 0.75, 1.0))
    labels = [f'{i:.0f}-{j:.0f} episodes' for i, j in zip(q[:-1], q[1:])]

    # Creates a new 'quartile' column using pd.qcut() which divides the 'total' column into 4 equal-sized groups
    rates_by_age['quartile'] = pd.qcut(rates_by_age['total'], q=4, labels=labels)

    rates_by_age_longform = rates_by_age.melt(value_vars=mid_ages, id_vars='quartile', var_name='age', value_name='rate', ignore_index=False)
    rates_by_age_longform = rates_by_age_longform.reset_index(drop=False)
    return rates_by_age, rates_by_age_longform

###################################### MODELS ###########################

def infection_rate(a, p):
    b = log(2) / p.τ_elp
    l = p.λ * (1 - p.ρ_elp*np.exp(-b*a))        # rate of blood infection at age a
    return l

def infections(a, p):
    b = log(2) / p.τ_elp
    l = p.λ * (1 - p.ρ_elp*np.exp(-b*a))        # rate of blood infection at age a
    n = p.λ * (a - p.ρ_elp*(1-np.exp(-b*a))/b)  # cumulative number of blood infections by age a
    return l, n

def sigmoidal_model(n, p):
    # risk of clinical episode on nth infection
    return p.ρ_clinical * (exp(p.δ_clinical*p.γ_clinical) - 1) / (exp(p.δ_clinical*p.γ_clinical) - 2 + exp(p.δ_clinical*n))

def exponential_model(n, p):
    # risk of clinical episode on nth infection
    return (p.ρ_clinical * exp(-p.δ_clinical*n))

def sigmoidal_clinical_cases(a, p):
    l, n = infections(a, p)
    return l * sigmoidal_model(n, p)  # rate of clinical episode at age a (episodes per unit time)

def exponential_clinical_cases(a, p):
    l, n = infections(a, p)
    return (l * exponential_model(n, p))  # rate of clinical episode at age a (episodes per unit time)

def sigmoidal_cumulative_clinical_cases(a1, a2, p):
    # number of clinical episodes between ages a1 and a2
    b = log(2) / p.τ_elp
    f = lambda a: 2 * exp(-p.δ_clinical * p.γ_clinical) - 1 - exp(-p.δ_clinical * (p.γ_clinical - p.λ * (a - p.ρ_elp / b * (1 - exp(-b * a)))))
    return p.ρ_clinical / p.δ_clinical * (1 - exp(-p.δ_clinical * p.γ_clinical)) / (1 - 2 * exp(-p.δ_clinical * p.γ_clinical)) * (p.δ_clinical * p.λ * p.ρ_elp / b * (exp(-b * a2) - exp(-b * a1)) - (a1 - a2) * p.δ_clinical * p.λ - log(f(a2) / f(a1)))

def exponential_cumulative_clinical_cases(a1, a2, p):
    # number of clinical episodes between ages a1 and a2
    b = log(2) / p.τ_elp
    f1 = exp(p.δ_clinical * p.λ * p.ρ_elp / b)
    f2 = exp(-p.δ_clinical * p.λ * (a1 + p.ρ_elp * exp(-a1 * b) / b))
    f3 = exp(-p.δ_clinical * p.λ * (a2 + p.ρ_elp * exp(-a2 * b) / b))
    return (p.ρ_clinical * f1 * (f2 - f3) / p.δ_clinical)

################# INDIVIDUAL LEVEL PLOTS ###########################

def plot_episodes_by_age(rates):
    g = sns.relplot(data=rates, x='age', y='rate', col='quartile', kind='line', legend=None, linewidth=3, color='C1', errorbar=None, height=3)

    for q, ax in g.axes_dict.items():
        lines = rates.query('quartile == @q')
        sns.lineplot(data=lines, x='age', y='rate', units='index', estimator=None, linewidth=0.5, color='C2', alpha=0.5, ax=ax, zorder=0)

    g.set_axis_labels("Age (years)", "Annual clinical episodes")
    g.tight_layout()
    g.fig.subplots_adjust(top=0.85)
    return g

def plot_bednet_scores(bednet_file, rates_by_age):
    # read in bednet scores
    bednet = pd.read_csv(bednet_file, header=None)
    bednet[0] = bednet[0].replace('?', pd.NA).astype('Int64')
    id = np.array(list(range(111))*3).reshape(3, 111).T.flatten()
    year = [1, 2, 3]*111
    bednet['id'] = id
    bednet['year'] = year
    bednet_score = bednet.groupby('id')[0].sum()

    r = pearsonr(bednet_score.astype(float), rates_by_age['total'].values)

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))

    sns.scatterplot(x=bednet_score.astype('float64'), y=rates_by_age['total'], ax=ax)
    ax.set_xlabel('Sum of bednet scores')
    ax.set_ylabel('Total clinical episodes')
    ax.set_title(f'Pearson r = {r.statistic:.2f}, p = {r.pvalue:.2f}')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return fig, ax, r

def plot_individual_risk_profiles(sig_file):
    import matplotlib as mpl

    def sigmoidal_model(n, p):
        # risk of clinical episode on nth infection
        return p.ρ_clinical * (exp(p.δ_clinical*p.γ_clinical) - 1) / (exp(p.δ_clinical*p.γ_clinical) - 2 + exp(p.δ_clinical*n))

    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    fig.subplots_adjust(wspace=0.4)
    n = np.arange(300)
    ages = np.linspace(0, 20, 100)
    df = pd.read_csv(sig_file)

    def get_pars(df, i):
        # get all parameter draws from the posterior for child i
        λ = np.array(df[f'lambda[{i}]'])[:, np.newaxis]
        # ρ_elp = np.array(df[f'rho_elp[{i}]'])[:, np.newaxis]
        ρ_elp = np.ones_like(λ)
        τ_elp = np.array(df[f'tau_elp[{i}]'])[:, np.newaxis]
        δ_clinical = np.array(df[f'delta_clinical[{i}]'])[:, np.newaxis]
        ρ_clinical = np.array(df[f'rho_clinical[{i}]'])[:, np.newaxis]
        γ_clinical = np.array(df[f'gamma_clinical[{i}]'])[:, np.newaxis]
        return Pars_sig(λ, ρ_elp, τ_elp, δ_clinical, ρ_clinical, γ_clinical)

    def get_individual_curves(f, x, df):
        y = np.array([f(x, get_pars(df, j)) for j in range(1, 112)]).T
        ym = y.mean(axis=1)
        ym_df = pd.DataFrame(ym, columns=[i for i in range(ym.shape[1])], index=x)
        ym_df['time'] = ym_df.index
        ym_df = ym_df.melt(id_vars='time', var_name='individual', value_name='rate')
        return ym_df

    ax = axs[0]
    y = get_individual_curves(infection_rate, ages, df)
    sns.lineplot(data=y, x='time', y='rate', hue='individual', estimator=None, linewidth=0.5, legend=False, alpha=0.5, ax=ax)
    ax.set_ylabel('Blood infections per year')
    ax.set_xlabel('Age (years)')

    ax = axs[1]
    y = get_individual_curves(sigmoidal_model, n, df)
    sns.lineplot(data=y, x='time', y='rate', hue='individual', estimator=None, linewidth=0.5, legend=False, alpha=0.5, ax=ax)
    ax.set_xlabel('Number of blood infections')
    ax.set_ylabel('Risk of clinical episode');

    ax = axs[2]
    y = get_individual_curves(sigmoidal_clinical_cases, ages, df)
    sns.lineplot(data=y, x='time', y='rate', hue='individual', estimator=None, linewidth=0.5, legend=False, alpha=0.5, ax=ax)
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Clinical episodes per year')

    # Create colorbar for individual representation
    sm = plt.cm.ScalarMappable(cmap=sns.cubehelix_palette(reverse=True, as_cmap=True), norm=mpl.colors.Normalize(vmin=7, vmax=110))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Total clinical episodes')

    ages = np.arange(16)
    y = []
    for j in range(1, 112):
        pars_sig = get_pars(df, j)
        C = [sigmoidal_cumulative_clinical_cases(a1, a2, pars_sig) for a1, a2 in zip(ages[:-1], ages[1:])]
        y.append(C)

    for ax, i in zip(axs, 'ABC'):
        ax.set_title(i, loc='left')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    return fig, axs

def model(pars_exp=None, pars_sig=None, legend=True):
    a = np.linspace(0, 20, 100)[:, np.newaxis] # age in years
    if pars_exp:
        l, n = infections(a, pars_exp)
    elif pars_sig:
        l, n = infections(a, pars_sig)

    fig, axs = plt.subplots(1, 5, figsize=(15, 3))
    fig.subplots_adjust(wspace=0.4)
    ax = axs[0]
    ax.plot(a, l)
    ax.scatter(0, 0, color='w')  # to force y-axis to start at 0
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Blood infections per year')

    ax = axs[1]
    ax.plot(a, n)
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Cumulative number of infections')

    ax = axs[2]
    ni = np.arange(150)[:, np.newaxis]
    if pars_exp:
        ax.plot(ni, exponential_model(ni, pars_exp), label='Exponential')
    if pars_sig:
        ax.plot(ni, sigmoidal_model(ni, pars_sig), label='Sigmoidal')
    ax.scatter(0, 0, color='w')  # to force y-axis to start at 0
    ax.set_xlabel('Number of blood infections')
    ax.set_ylabel('Risk of clinical episode');
    if legend:
        ax.legend()

    ax = axs[3]
    if pars_exp:
        ax.plot(a, exponential_clinical_cases(a, pars_exp), label='Exponential')
    if pars_sig:
        ax.plot(a, sigmoidal_clinical_cases(a, pars_sig), label='Sigmoidal')
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Clinical episodes per year');
    if legend:
        ax.legend()

    ax = axs[4]
    if pars_exp:
        C = np.array([exponential_cumulative_clinical_cases(0, a1, pars_exp) for a1 in a]).reshape(len(a), -1)
        ax.plot(a, C, label='Exponential')
    if pars_sig:
        C = np.array([sigmoidal_cumulative_clinical_cases(0, a1, pars_sig) for a1 in a]).reshape(len(a), -1)
        ax.plot(a, C, label='Sigmoidal')
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Cumulative clinical episodes');
    if legend:
        ax.legend()

    for ax, i in zip(axs, 'ABCDE'):
        ax.set_title(i, loc='left')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    return fig, axs

def prior_predictive_check(pars_exp=None, pars_sig=None):
    a = np.linspace(0, 20, 100)[:, np.newaxis] # age in years

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    if pars_exp:
        ax.plot(a, exponential_clinical_cases(a, pars_exp), label='Exponential')
    if pars_sig:
        ax.plot(a, sigmoidal_clinical_cases(a, pars_sig), label='Sigmoidal')

    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Clinical episodes per year');
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return fig, ax

def plot_fit_episodes_by_age(rates, exp_file=None, sig_file=None):
    def get_prediction(filename):
        df = pd.read_csv(filename)
        m = df.mean()

        scales = (1, 3, 3, 3, 5, 11)

        r = []
        for p in range(1, 7):
            rs = m.filter(regex=fr'r\[\d{{1,3}},{p}]').values
            r.append(rs)
        r = pd.DataFrame(r).T
        for c, s, a in zip(r.loc[:,0:5], scales, mid_ages):
            r[a] = r[c] / s

        r['quartile'] = rates['quartile']
        r = r.melt(value_vars=mid_ages, id_vars='quartile', var_name='age', value_name='rate', ignore_index=False)
        r = r.reset_index(drop=False)
        return r

    g = sns.relplot(data=rates, x='age', y='rate', col='quartile', kind='line', linewidth=3, color='C1', errorbar=None, height=4, label="Mean of individuals' data")

    if exp_file:
        r1 = get_prediction(exp_file)
    if sig_file:
        r2 = get_prediction(sig_file)

    for q, ax in g.axes_dict.items():
        data = rates.query('quartile == @q')
        sns.lineplot(data=data, x='age', y='rate', units='index', estimator=None, linewidth=0.5, color='C2', alpha=0.5, ax=ax, zorder=0)
        if exp_file:
            prediction = r1.query('quartile == @q')
            sns.lineplot(data=prediction, x='age', y='rate', linewidth=1, color='k', ls='--', errorbar=None, ax=ax, label='Exponential model')
        if sig_file:
            prediction = r2.query('quartile == @q')
            sns.lineplot(data=prediction, x='age', y='rate', linewidth=2, color='k', ls='--', errorbar=None, ax=ax, label='Sigmoidal model')

    g.set_axis_labels("Age (years)", "Clinical episodes per year")
    g.tight_layout();
    g.fig.subplots_adjust(top=0.85)
    return g

def parameter_relationships(rates_by_age, sig_file=None):
    priors = {
        'lambda': stats.norm(loc=15, scale=2),
        'rho_elp': stats.beta(5, 1),
        'tau_elp': stats.expon(scale=2),
        'delta_clinical': stats.norm(loc=0.01, scale=0.01),
        'gamma_clinical': stats.norm(loc=100, scale=50),
        'rho_clinical': stats.uniform()
    }

    # read in all posterior draws
    df = pd.read_csv(sig_file)

    fig, axs = plt.subplots(1, 6, figsize=(18, 4))

    # mean parameter estimates across draws for each child
    m = df.mean()
    pl = df.quantile(0.025)
    pu = df.quantile(0.975)

    for var, label, title, ax in zip(name_conversions.keys(), name_conversions.values(), 'ABCDEF', axs):
        try:
            m[var] = m.filter(like=f'{var}[').reset_index(drop=True)
            pl[var] = pl.filter(like=f'{var}[').reset_index(drop=True)
            pu[var] = pu.filter(like=f'{var}[').reset_index(drop=True)
        except:
            m[var] = m.filter(like=var).reset_index(drop=True)
            pl[var] = pl.filter(like=var).reset_index(drop=True)
            pu[var] = pu.filter(like=var).reset_index(drop=True)
        df = pd.DataFrame({
            'total clinical cases': rates_by_age['total'],
            'mean': m[var],
            'xerrl': m[var] - pl[var],
            'xerru': pu[var] - m[var]
        })
        ax.errorbar(df['mean'], df['total clinical cases'], xerr=np.array([df['xerrl'], df['xerru']]), fmt='o', alpha=0.5, color='C0')

        for pp in (0.05, 0.5, 0.95):
            ax.axvline(priors[var].ppf(pp), color='C1', lw=2, linestyle=':')
        ax.set_xlabel(label)
        ax.set_title(title, loc='left')
    axs[0].set_ylabel('Total clinical episodes')

    return fig, axs

def parameter_relationships_hierarchical(rates_by_age, sig_file=None):
    # read in all posterior draws
    df = pd.read_csv(sig_file)

    fig, axs = plt.subplots(1, 5, figsize=(13.5, 4))

    # mean parameter estimates across draws for each child
    m = df.mean()
    pl = df.quantile(0.025)
    pu = df.quantile(0.975)

    for var, label, title, ax in zip(name_conversions_hierarchical.keys(), name_conversions_hierarchical.values(), 'ABCDEFG', axs):
        try:
            m[var] = m.filter(like=f'{var}[').reset_index(drop=True)
            pl[var] = pl.filter(like=f'{var}[').reset_index(drop=True)
            pu[var] = pu.filter(like=f'{var}[').reset_index(drop=True)
        except:
            m[var] = m.filter(like=var).reset_index(drop=True)
            pl[var] = pl.filter(like=var).reset_index(drop=True)
            pu[var] = pu.filter(like=var).reset_index(drop=True)
        df = pd.DataFrame({
            'total clinical cases': rates_by_age['total'],
            'mean': m[var],
            'xerrl': m[var] - pl[var],
            'xerru': pu[var] - m[var]
        })
        ax.errorbar(df['mean'], df['total clinical cases'], xerr=np.array([df['xerrl'], df['xerru']]), fmt='o', alpha=0.5, color='C0')

        # for pp in (0.05, 0.5, 0.95):
        #     ax.axvline(d[var].ppf(pp), color='C1', lw=2, linestyle=':')
        ax.set_xlabel(label)
        ax.set_title(title, loc='left')
    axs[0].set_ylabel('Total clinical episodes')

    return fig, axs

def model_sensitivity():
    # parameters taken from mean posterior estimates for individual fits
    pars_sig = Pars_sig(
        16.1,     # infections per year
        1,        # relative protection of newborns
        0.54,     # half-life of protection in years
        0.026,    # decay rate of clinical risk per infection
        0.5,     # probability of clinical episode in non-immunes
        70,     # half-maximum number of infections for sigmoidal model
    )
    a = np.linspace(0, 20, 100) # age in years

    fig, axs = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey='col')
    fig.subplots_adjust(wspace=0.3)

    for ρ_clinical in [1.0, 0.5, 0.1]:
        p = pars_sig._replace(ρ_clinical=ρ_clinical)
        axs[0, 0].plot(a, sigmoidal_clinical_cases(a, p), label=f'{ρ_clinical:.1f}')
        axs[0, 1].plot(a, [sigmoidal_cumulative_clinical_cases(0, a1, p) for a1 in a])

    axs[0, 0].legend(title='ρ_clinical (γ_clinical=70)')

    for γ_clinical in np.arange(70, 160, 40):
        p = pars_sig._replace(γ_clinical=γ_clinical)
        axs[1, 0].plot(a, sigmoidal_clinical_cases(a, p), ls='--', label=f'{γ_clinical}')
        axs[1, 1].plot(a, [sigmoidal_cumulative_clinical_cases(0, a1, p) for a1 in a], ls='--')

    axs[1, 0].legend(title='γ_clinical (ρ_clinical=0.5)')
    axs[1, 0].set_xlabel('Age (years)')
    axs[1, 1].set_xlabel('Age (years)')
    axs[0, 0].set_ylabel('Clinical episodes per year');
    axs[1, 0].set_ylabel('Clinical episodes per year');
    axs[0, 1].set_ylabel('Cumulative clinical episodes');
    axs[1, 1].set_ylabel('Cumulative clinical episodes');

    for ax, i in zip(axs.flatten(), 'ABCD'):
        ax.set_title(i, loc='left')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    return fig, axs

################# POPULATION LEVEL PLOTS ###########################

def plot_population_episodes_by_age(rate_file):
    rates = pd.read_csv(rate_file)
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))

    sns.scatterplot(data=rates, x='age', y='rate', hue='site', ax=ax)
    sns.lineplot(data=rates, x='age', y='rate', hue='site', legend=False, ax=ax)
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Clinical episodes per year');
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend(title='Village')
    return fig, ax

def fit_pop_model(pars_dielmo_sig, pars_ndiop_sig, ind_model_file, rate_file):
    def mean_CI(f, pars):
        mean, lower, upper = [], [], []
        for a1, a2 in zip(periods[:-1], periods[1:]):
            rates = f(a1, a2, pars)
            mean.append(np.mean(rates) / (a2-a1))
            lower.append(np.percentile(rates, 97.5) / (a2-a1))
            upper.append(np.percentile(rates, 2.5) / (a2-a1))
        return mean, lower, upper

    # read in data
    cases_by_age = pd.read_csv(rate_file)
    # read in posterior parameter draws for all children
    periods = list(range(16)) + [25]

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3))
    fig.subplots_adjust(right=1)

    pars_ind = get_mean_pars_sig(None, ind_model_file)
    mean, lower, upper = mean_CI(sigmoidal_cumulative_clinical_cases, pars_ind)
    ax.plot(periods[:-1], mean, color='C3', ls='--', label='Mean individual fit')

    mean, lower, upper = mean_CI(sigmoidal_cumulative_clinical_cases, pars_dielmo_sig)
    ax.plot(periods[:-1], mean, color='C0')
    ax.fill_between(periods[:-1], lower, upper, color='C0', alpha=0.3)

    mean, lower, upper = mean_CI(sigmoidal_cumulative_clinical_cases, pars_ndiop_sig)
    ax.plot(periods[:-1], mean, color='C1')
    ax.fill_between(periods[:-1], lower, upper, color='C1', alpha=0.3)

    sns.scatterplot(data=cases_by_age, x='age', y='rate', hue='site', ax=ax)

    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Clinical episodes per year')
    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return fig, ax

def plot_pop_parameter_distributions(hetero_model1_file, hetero_model2_file, homo_model_file, ind_model_file):
    dielmo_rho_clinical_prior = stats.norm(loc=0.789, scale=0.015)
    ndiop_lambda_prior = stats.norm(loc=5, scale=2)
    priors = {
        'lambda': stats.norm(loc=15, scale=2),
        'rho_elp': stats.beta(5, 1),
        'tau_elp': stats.expon(scale=2),
        'delta_clinical': stats.norm(loc=0.01, scale=0.01),
        'rho_clinical': stats.uniform(),
        'gamma_clinical': stats.norm(loc=100, scale=50),
    }

    villages = ['Deilmo', 'Ndiop']

    df_hetero1 = pd.read_csv(hetero_model1_file).reset_index().rename(columns={'index': 'draw'})
    df_hetero2 = pd.read_csv(hetero_model2_file).reset_index().rename(columns={'index': 'draw'})
    df_homo = pd.read_csv(homo_model_file).reset_index().rename(columns={'index': 'draw'})
    df_homo['rho_elp[1]'] = df_homo['rho_elp']
    df_homo['rho_elp[2]'] = df_homo['rho_elp']
    df_homo['tau_elp[1]'] = df_homo['tau_elp']
    df_homo['tau_elp[2]'] = df_homo['tau_elp']
    df_homo['delta_clinical[1]'] = df_homo['delta_clinical']
    df_homo['delta_clinical[2]'] = df_homo['delta_clinical']
    df_homo['rho_clinical[1]'] = df_homo['rho_clinical']
    df_homo['rho_clinical[2]'] = df_homo['rho_clinical']
    df_homo['gamma_clinical[1]'] = df_homo['gamma_clinical']
    df_homo['gamma_clinical[2]'] = df_homo['gamma_clinical']
    df_homo = df_homo.drop(columns=['tau_elp', 'delta_clinical', 'rho_clinical', 'gamma_clinical'])

    model_hetero1 = get_draws(df_hetero1, variables=list(name_conversions.keys()), groups=villages)
    model_hetero1['Village'] = model_hetero1['group'].replace({str(i+1): g for i, g in enumerate(villages)})
    model_hetero2 = get_draws(df_hetero2, variables=list(name_conversions.keys()), groups=villages)
    model_hetero2['Village'] = model_hetero2['group'].replace({str(i+1): g for i, g in enumerate(villages)})
    model_homo = get_draws(df_homo, variables=list(name_conversions.keys()), groups=villages)
    model_homo['Village'] = model_homo['group'].replace({str(i+1): g for i, g in enumerate(villages)})

    fig, axs = plt.subplots(3, 6, figsize=(15, 6), sharex='col')

    for (var, label), ax in zip(name_conversions.items(), axs[0, :]):
        legend = True if var == 'tau_elp' else False
        sns.kdeplot(data=model_hetero1.query('parameter == @var'), x='value', hue='Village', ax=ax, common_norm=False, legend=legend, fill=True, clip=(0, None))
        ax.set_xlabel(f'{label}')
        ax.set_ylabel('')
        ax.set_yticks([])
    axs[0, 0].set_ylabel('Heterogeneous')

    for (var, label), ax in zip(name_conversions.items(), axs[1, :]):
        legend = True if var == 'tau_elp' else False
        sns.kdeplot(data=model_hetero2.query('parameter == @var'), x='value', hue='Village', ax=ax, common_norm=False, legend=False, fill=True, clip=(0, None))
        ax.set_xlabel(f'{label}')
        ax.set_ylabel('')
        ax.set_yticks([])
    axs[1, 0].set_ylabel('Heterogeneous\nconstrained ρ_clinical')

    for (var, label), ax in zip(name_conversions.items(), axs[2, :]):
        legend = True if var == 'tau_elp' else False
        sns.kdeplot(data=model_homo.query('parameter == @var'), x='value', hue='Village', ax=ax, common_norm=False, legend=False, fill=True, clip=(0, None))
        ax.set_xlabel(f'{label}')
        ax.set_ylabel('')
        ax.set_yticks([])
    axs[2, 0].set_ylabel('Homogeneous\nconstrained ρ_clinical')

    for i, (label, prior) in enumerate(priors.items()):
        x0, x1 = axs[0, i].get_xlim()
        x = np.linspace(x0, x1, 100)
        if label == 'rho_clinical':
            axs[0, i].plot(x, prior.pdf(x), ls=':', lw=0.5, color='k')
            axs[1, i].plot(x, dielmo_rho_clinical_prior.pdf(x), ls=':', lw=0.5, color='k')
            axs[2, i].plot(x, dielmo_rho_clinical_prior.pdf(x), ls=':', lw=0.5, color='k')
        else:
            axs[0, i].plot(x, prior.pdf(x), ls=':', lw=0.5, color='k')
            axs[1, i].plot(x, prior.pdf(x), ls=':', lw=0.5, color='k')
            axs[2, i].plot(x, prior.pdf(x), ls=':', lw=0.5, color='k')
        if label == 'lambda':
            axs[0, i].plot(x, ndiop_lambda_prior.pdf(x), ls=':', lw=0.5, color='k')
            axs[1, i].plot(x, ndiop_lambda_prior.pdf(x), ls=':', lw=0.5, color='k')
            axs[2, i].plot(x, ndiop_lambda_prior.pdf(x), ls=':', lw=0.5, color='k')

    pars_ind = get_mean_pars_sig(None, ind_model_file)
    for i, p in enumerate(pars_ind):
        axs[0, i].axvline(p, color='C3', ls='--')
        axs[1, i].axvline(p, color='C3', ls='--')
        axs[2, i].axvline(p, color='C3', ls='--')
    return fig, axs

def plot_marginal_posterior(hetero_model_file):
    draws = pd.read_csv(hetero_model_file)

    p = 1 # village index: 1 to 2
    vars = [f'lambda[{p}]', f'rho_elp[{p}]', f'tau_elp[{p}]', f'rho_clinical[{p}]', f'delta_clinical[{p}]', f'gamma_clinical[{p}]']

    draws = draws[vars]
    draws.columns = 'λ', 'ρ_elp', 'τ_elp', 'ρ_clinical', 'δ_clinical', 'γ_clinical'

    g = sns.pairplot(draws, kind='scatter', corner=True, plot_kws=dict(marker=".", linewidth=0, alpha=0.5), height=1)
    return g

######################### OTHER FUNCTIONS ###########################

def loocv(models, files):
    fits = []
    for f in files:
        with open(f, 'rb') as f:
            fits.append(az.from_cmdstanpy(pickle.load(f)))

    return az.compare({m:f for m, f in zip(models, fits)})[['rank', 'elpd_loo', 'p_loo']]

def get_estimates(filename):
    draws = pd.read_csv(filename)
    vars = ['lambda[1]', 'lambda[2]', 'rho_elp', 'tau_elp', 'delta_clinical', 'rho_clinical', 'gamma_clinical']
    means = draws[vars].mean()
    stds = draws[vars].std()
    return means, stds

def get_mean(filename):
    with open(filename, 'rb') as f:
        fit = az.from_cmdstanpy(pickle.load(f))
        vars = ['lambda', 'rho_elp', 'tau_elp', 'delta_clinical', 'rho_clinical', 'gamma_clinical']
        pars = az.extract(fit, var_names=vars).mean().to_array().data
    return Pars_sig(*pars)

def get_mean_pars_sig(i, filename, fix_ρ_elp=True):
    # parameters taken from mean posterior estimates for individual fits
    """
        select individual i from the sigmoidal model fit file
        if i is None use all individuals
        return mean parameter estimates as Pars_sig namedtuple
    """
    with open(filename, 'rb') as f:
        fit = az.from_cmdstanpy(pickle.load(f))

    if i is not None:
        fit = fit.isel(lambda_dim_0=i, rho_elp_dim_0=i, tau_elp_dim_0=i, delta_clinical_dim_0=i, rho_clinical_dim_0=i, gamma_clinical_dim_0=i)
    if fix_ρ_elp:
        vars = ['lambda', 'tau_elp', 'delta_clinical', 'rho_clinical', 'gamma_clinical']
        pars = az.extract(fit, var_names=vars).mean().to_array().data
        pars = np.insert(pars, 1, 1) # insert rho_elp=1 for sigmoidal model
    else:
        vars = ['lambda', 'rho_elp', 'tau_elp', 'delta_clinical', 'rho_clinical', 'gamma_clinical']
        pars = az.extract(fit, var_names=vars).mean().to_array().data
    return Pars_sig(*pars)

def get_mean_pars_exp(i, filename):
    """
        select individual i from the exponential model fit file
        if i is None use all individuals
        return mean parameter estimates as Pars_sig namedtuple
    """
    # parameters taken from mean posterior estimates for individual fits
    with open(filename, 'rb') as f:
        fit = az.from_cmdstanpy(pickle.load(f))

    if i is not None:
        fit = fit.isel(lambda_dim_0=i, rho_elp_dim_0=i, tau_elp_dim_0=i, delta_clinical_dim_0=i, rho_clinical_dim_0=i)
    vars = ['lambda', 'rho_elp', 'tau_elp', 'delta_clinical', 'rho_clinical']
    pars = az.extract(fit, var_names=vars).mean().to_array().data
    return Pars_exp(*pars)

def get_population_pars_hetero(filename):
    # get all parameter draws from the posterior
    df = pd.read_csv(filename)
    pars_dielmo = Pars_sig(
        df['lambda[1]'],
        df['rho_elp[1]'],
        df['tau_elp[1]'],
        df['delta_clinical[1]'],
        df['rho_clinical[1]'],
        df['gamma_clinical[1]'],
    )
    pars_ndiop = Pars_sig(
        df['lambda[2]'],
        df['rho_elp[2]'],
        df['tau_elp[2]'],
        df['delta_clinical[2]'],
        df['rho_clinical[2]'],
        df['gamma_clinical[2]'],
    )
    return pars_dielmo, pars_ndiop, df

def get_population_pars_homo(filename):
    # get all parameter draws from the posterior
    df = pd.read_csv(filename)
    pars_dielmo = Pars_sig(
        df['lambda[1]'],
        df['rho_elp'],
        df['tau_elp'],
        df['delta_clinical'],
        df['rho_clinical'],
        df['gamma_clinical'],
    )

    pars_ndiop = Pars_sig(
        df['lambda[2]'],
        df['rho_elp'],
        df['tau_elp'],
        df['delta_clinical'],
        df['rho_clinical'],
        df['gamma_clinical'],
    )
    return pars_dielmo, pars_ndiop, df

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

