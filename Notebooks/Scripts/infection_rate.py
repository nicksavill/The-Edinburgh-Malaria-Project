import sys
sys.path.append('..')
from Edinburgh_Model.model import *

from Scripts import paper, trape
from Scripts import trape

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from numpy import log10
from scipy.stats import norm
from multiprocessing import Pool
from collections import namedtuple
from statsmodels.formula.api import ols

# get the default parameters for the model
default_pars = Parameters()


def get_pfpr():
    map = pd.read_csv('Unpublished_data/pfpr.csv', low_memory=False)
    map['year_end'] = map['year_end'].astype('Int64')

    # Only use sites with positive PfPR values
    map = map[(map['pr'] > 0.0)]

    return map

def distribution_over_surveys(map):
    years = np.arange(1987, 2025, 5)
    colours = sns.color_palette("Spectral", n_colors=len(years), as_cmap=False)
    fig, ax = plt.subplots(1, 1)

    # KDE plots of the distribution of PfPR in a selection of survey years
    for c, year in enumerate(sorted(years)):
        sns.kdeplot(x='pr', cut=0, data=map[map['year_end'] == year], color=colours[c], alpha=1, lw=1, label=f'{year} ({len(map[map["year_end"] == year])})', ax=ax)


    # KDE plots of the distribution of PfPR using the most recent survey at each site from 2020 onwards
    year_start = 2020
    pr = map.groupby('site_id').last().reset_index()
    pr = pr[pr['year_end'] >= year_start]
    sns.kdeplot(x='pr', cut=0, data=pr, color='black', alpha=1, lw=2, label=f'{year_start}+ (most recent, {len(pr)})', ax=ax)
    ax.set_xlabel('PfPR in 2-10 year olds')
    ax.set_ylabel('Density');
    ax.legend(title='Year of survey (No. sites)');
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return fig, ax

def distribution_over_children():
    # PfPR values from the Malaria Atlas Project are discretised into 200 contiguous bins.
    # Each bin contains the number of children (by age category (0, 1, 2-5) and location (World or Africa))
    # experiencing that PfPR
    #
    with open('Data/pfpr_0-9.pickle', 'rb') as f:
        a = pickle.load(f)

    # counts of the number of children in a location with PfPR > 0.1% (ie children with PFPR <= 0.1%
    # are not in this data)
    (count_mean_total_world,
    count_mean_total_africa,
    count_mean_00_africa,
    count_mean_01_africa,
    count_mean_05_africa) = a
    nbins = len(count_mean_total_world) - 1

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    s = slice(0, int(60*nbins/100))

    ax = axs[0]
    ax.plot(np.linspace(0, 1, nbins+1)[s], count_mean_total_world[s] * nbins / 100 * 1e-6, label=f'World: {sum(count_mean_total_world):,.0f}')
    ax.plot(np.linspace(0, 1, nbins+1)[s], count_mean_total_africa[s] * nbins / 100 * 1e-6, label=f'Africa: {sum(count_mean_total_africa):,.0f}')
    ax.legend()
    ax.set_ylabel('Millions of children (per unit PfPR)')
    ax.set_xlabel('PfPR in ages 2-10')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title('A', loc='left')

    ax = axs[1]
    ax.plot(np.linspace(0, 1, nbins+1)[s], count_mean_00_africa[s] * nbins / 100 * 1e-6, label=f'Africa 0 yr: {sum(count_mean_00_africa):,.0f}')
    ax.plot(np.linspace(0, 1, nbins+1)[s], count_mean_01_africa[s] * nbins / 100 * 1e-6, label=f'Africa 1-4 yr: {sum(count_mean_01_africa):,.0f}')
    ax.plot(np.linspace(0, 1, nbins+1)[s], count_mean_05_africa[s] * nbins / 100 * 1e-6, label=f'Africa 5-9 yr: {sum(count_mean_05_africa):,.0f}')
    ax.legend()
    ax.set_ylabel('Millions of children (per unit PfPR)')
    ax.set_xlabel('PfPR in ages 2-10');
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title('B', loc='left')

    return fig, axs

def get_pfpr_vs_clinical():
    griffin = pd.read_csv('Data/griffin_2014_fig1.csv')
    trape = pd.read_csv('Data/trape_2014.csv')
    griffin = griffin.sort_values('pfpr')
    trape = trape.sort_values('pfpr')
    return griffin, trape

def plot_pfpr_vs_clinical(data):
    titles = 'A) Griffin et al. (2014)', 'B) Trape et al. (2014)'
    fig, axs = plt.subplots(1, 2, figsize=(7, 3), sharex=True, sharey=True)
    for d, title, ax in zip(data, titles, axs):
        sns.scatterplot(d, x='pfpr', y='clinical_rate', ax=ax)
        ax.set_xlabel('PfPR in 2-10 year olds')
        ax.set_ylabel('Average annual clinical episodes\nper child under five')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title(title)
        ax.grid(visible=True, which='both', axis='both', color='lightgrey', linestyle='--', linewidth=0.5);

    return fig, ax

def model_pfpr_clinical_relationship(data, missing=0):
    fits = []

    for d in data:
        d['log_clinical_missing'] = log10(d['clinical_rate'] / (1-missing))
        d['log_pfpr'] = log10(d['pfpr'])
        formula = 'log_clinical_missing ~ log_pfpr'
        fit = ols(formula, d).fit()

        sf = fit.get_prediction().summary_frame()
        fit.σ_c = (sf['obs_ci_upper'] - sf['obs_ci_lower']).mean() / (2*1.96)

        d['mean'] = 10**sf['mean']
        d['obs_ci_lower'] = 10**sf['obs_ci_lower']
        d['obs_ci_upper'] = 10**sf['obs_ci_upper']
        fits.append(fit)

    return fits, data

def plot_model_pfpr_clinical_relationship(fits, data):
    titles = 'A) Griffin et al. (2014)', 'B) Trape et al. (2014)'
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    for fit, d, title, ax in zip(fits, data, titles, axs):
        # randomly sample 1000 values from the estimated distribution
        Mu = fit.params
        Sigma = fit.cov_params()
        samples = np.random.multivariate_normal(Mu, Sigma, 1000)

        # construct lines using the samples
        β0 = samples[:, 0]
        β1 = samples[:, 1]
        r = np.linspace(log10(d['pfpr'].min()), log10(d['pfpr'].max()), 20)
        lines = 10**(β0+β1*r[:, np.newaxis])

        # construct 95% CI
        lower = np.percentile(lines, 2.5, axis=1)
        upper = np.percentile(lines, 97.5, axis=1)

        ax.scatter('pfpr', 'clinical_rate', data=d, label='Data', ec='C0', fc='w')
        ax.plot('pfpr', 'mean', data=d, color='C0', label='mean')
        ax.fill_between(10**r, lower, upper, color='C0', alpha=0.25, label='95% CI')
        ax.plot('pfpr', 'obs_ci_lower', data=d, ls='--', color='C0', label='95% Predictive interval')
        ax.plot('pfpr', 'obs_ci_upper', data=d, ls='--', color='C0', label='');
        ax.set_xlabel('PfPR in 2-10 year olds')
        ax.set_ylabel('Average annual clinical\nepisodes per child under five')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title(title)

    return fig, axs

def infection_rate_vs_clinical_rate(
        β=default_pars.β,
        ρ_elp=default_pars.ρ_elp,
        τ_elp=default_pars.τ_elp,
        ρ_clinical=default_pars.ρ_clinical,
        δ_clinical=default_pars.δ_clinical,
        γ_clinical=default_pars.γ_clinical,
        ):

    months = 60
    model = Model()
    model.pars = Parameters(
        study_months=months,
        β=β,
        ρ_elp=ρ_elp,
        τ_elp=τ_elp,
        ρ_clinical=ρ_clinical,
        δ_clinical=δ_clinical,
        γ_clinical=γ_clinical,
    )
    model.variables = Variables(['All clinical'])
    model.measures = Measures(['cdf'])
    model.Config({'time_scale':'Weeks'})

    λs = np.logspace(log10(0.01), log10(30), 50)

    clinical_rate_under_5 = []
    for λ in λs:
        model.pars.λ = λ
        model.pars.control = λ
        model.pars.update_pars(('λ', 'control'))
        r = paper.simulate_single_vaccine(model)

        # average number of clinical cases per child per year in first five years of life
        clinical_rate_under_5.append(r[f'{λ} All clinical cdf'].iloc[-1] / model.pars.popsize / (months/12))

    return λs, np.array(clinical_rate_under_5)

def infection_rate_vs_clinical_rate_sensitivity():
    λs = np.logspace(log10(0.01), log10(30), 50)

    fig, axs = plt.subplots(1, 6, figsize=(20, 3), sharey=True)

    n = 5

    ax = axs[0]
    for β in np.linspace(0, 1, n):
        _, rates = infection_rate_vs_clinical_rate(β=β)
        ax.plot(λs, rates)
    ax.set_title('β\n(0, 1)')
    ax.set_ylabel('Average annual clinical\nepisodes per child under five')

    ax = axs[1]
    for ρ_elp in np.linspace(0.5, 1, n):
        _, rates = infection_rate_vs_clinical_rate(ρ_elp=ρ_elp)
        ax.plot(λs, rates)
    ax.set_title('ρ_elp\n(0.5, 1)')

    ax = axs[2]
    for τ_elp in np.linspace(0.5, 2, n):
        _, rates = infection_rate_vs_clinical_rate(τ_elp=τ_elp)
        ax.plot(λs, rates)
    ax.set_title('τ_elp\n(0.5, 2) years')

    ax = axs[3]
    for ρ_clinical in np.linspace(0.5, 1.0, n):
        _, rates = infection_rate_vs_clinical_rate(ρ_clinical=ρ_clinical)
        ax.plot(λs, rates)
    ax.set_title('ρ_clinical\n(0.5, 1)')

    ax = axs[4]
    for δ_clinical in np.linspace(0.01, 0.04, n):
        _, rates = infection_rate_vs_clinical_rate(δ_clinical=δ_clinical)
        ax.plot(λs, rates)
    ax.set_title('δ_clinical\n(0.01, 0.04) per infection')

    ax = axs[5]
    for γ_clinical in np.linspace(50, 150, n):
        _, rates = infection_rate_vs_clinical_rate(γ_clinical=γ_clinical)
        ax.plot(λs, rates)
    ax.set_title('γ_clinical\n(50, 150) infections')

    for ax in axs:
        ax.set_xlabel('Annual blood infection rate')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(visible=True, which='both', axis='both', color='lightgrey', linestyle='--', linewidth=0.5);

    return fig, ax

def construct_distributions(map, count_mean_total_africa, fit, clinical_rate_under_5, λs):
    # extract estimates from fit
    β0 = fit.params['Intercept']
    β1 = fit.params['log_pfpr']
    σ_c_2 = fit.σ_c**2

    Sigma = fit.cov_params()
    σ_β0_2 = Sigma.iloc[0, 0]
    σ_β1_2 = Sigma.iloc[1, 1]
    σ_β0_β1 = Sigma.iloc[0, 1]

    max_log_clinical_rate = log10(clinical_rate_under_5.max())
    p_log_c_r = lambda r: norm.pdf(log_clinical_rate_space, loc=β0 + β1*log10(r), scale=np.sqrt(σ_β0_2 + σ_β1_2 * log10(r)**2 + 2 * σ_β0_β1 * log10(r) + σ_c_2))

    N = 100
    # The minimum PfPR we are considering is 0.1%. From the fit of the Trape data,
    # the minimum annual clinical rate in the under fives is above 0.001. We therefore chose this
    # as the minimum clinical rate to analyse. The annual blood infection rate at this clinical rate is ~0.001
    # which is well below 0.5 which we consider to be the minimum at which vaccination would happen
    min_log_clinical_rate = -3
    log_clinical_rate_space = np.linspace(min_log_clinical_rate, max_log_clinical_rate, N)
    # Transform to clinical rate space
    clinical_rate_space = 10**log_clinical_rate_space

    # transform clinical rate to blood infection rate
    # (range of λs is 0.01 - 30 which gives clinical rates from ~0.01 to 10)
    infection_rate_space = np.interp(clinical_rate_space, np.array(clinical_rate_under_5), λs)
    # calculate gradient of clinical rate against infection rate
    dc_by_df = np.gradient(clinical_rate_under_5, λs)
    dc_by_df = np.interp(infection_rate_space, λs, dc_by_df)

    # SURVEY SITES
    # Number of sites
    n = len(map)

    # Create an array of size N to hold the distribution of log-clinical episode rate for each site
    clinical_survey_pdfs = np.zeros(N*n).reshape(n, N)

    # for each value of pfpr in the survey data calculate its corresponding distribution of log-clinical episode rate and store each in a row of an array
    clinical_survey_pdfs = np.array([p_log_c_r(r) for r in map['pr']])

    # for each log-clinical episode rate, sum across all sites
    clinical_survey_pdf = clinical_survey_pdfs.sum(axis=0)
    # transform distribution to linear scale
    clinical_survey_pdf *= np.exp(-np.log(10) * log_clinical_rate_space) / np.log(10)
    # normalise distribution so the AUC=1
    clinical_survey_pdf /= np.trapz(clinical_survey_pdf, clinical_rate_space)
    # calculate the CDF for each log-clinical episode rate distribtion
    clinical_survey_cdf = np.array([np.trapz(clinical_survey_pdf[:i], clinical_rate_space[:i]) for i in range(len(clinical_survey_pdf))])

    # transform to infection rate
    infection_survey_pdf = clinical_survey_pdf * dc_by_df
    # normalise distribution so the AUC=1
    infection_survey_pdf /= np.trapz(infection_survey_pdf, infection_rate_space)
    # calculate the CDF
    infection_survey_cdf = np.array([np.trapz(infection_survey_pdf[:i], infection_rate_space[:i]) for i in range(len(infection_survey_pdf))])

    # NUMBER OF CHILDREN
    nbins = len(count_mean_total_africa)
    mid = 0.5/nbins
    pfpr_space = np.linspace(mid, 1-mid, nbins)
    # r is the midpoint of the bin
    clinical_children_pdfs = np.array([nkids*p_log_c_r(r) for r, nkids in zip(pfpr_space, count_mean_total_africa)])

    # for each log-clinical episode rate, sum across all sites
    clinical_children_pdf = clinical_children_pdfs.sum(axis=0)
    # transform distribution to linear scale
    clinical_children_pdf *= np.exp(-np.log(10) * log_clinical_rate_space) / np.log(10)
    # normalise distribution so the AUC=1
    clinical_children_pdf /= np.trapz(clinical_children_pdf, clinical_rate_space)
    # calculate the CDF for each log-clinical episode rate distribtion
    clinical_children_cdf = np.array([np.trapz(clinical_children_pdf[:i], clinical_rate_space[:i]) for i in range(len(clinical_survey_pdf))])

    # transform to infection rate
    infection_children_pdf = clinical_children_pdf * dc_by_df
    # normalise distribution so the AUC=1
    infection_children_pdf /= np.trapz(infection_children_pdf, infection_rate_space)
    # calculate the CDF
    infection_children_cdf = np.array([np.trapz(infection_children_pdf[:i], infection_rate_space[:i]) for i in range(len(infection_children_pdf))])

    return pd.DataFrame({
        'infection_rate': infection_rate_space,
        'clinical_rate': clinical_rate_space,
        'clinical_survey_pdf': clinical_survey_pdf,
        'clinical_survey_cdf': 100*clinical_survey_cdf,
        'clinical_children_pdf': clinical_children_pdf,
        'clinical_children_cdf': 100*clinical_children_cdf,
        'infection_survey_pdf': infection_survey_pdf,
        'infection_survey_cdf': 100*infection_survey_cdf,
        'infection_children_pdf': infection_children_pdf,
        'infection_children_cdf': 100*infection_children_cdf
    })

def plot_distributions(df, map, count_mean_total_africa):
    # plot up to 15 blood infections per year
    df = df[df['infection_rate'] < 15.1]

    fig, axs = plt.subplots(2, 3, figsize=(10, 6), sharex='col')

    nbins = len(count_mean_total_africa)
    mid = 0.5/nbins
    pfpr_space_children = np.linspace(mid, 1-mid, nbins)

    ax = axs[0, 0]
    sns.kdeplot(x='pr', cut=0, data=map, ax=ax, label='Surveys')
    ax.plot(pfpr_space_children, count_mean_total_africa/count_mean_total_africa.sum()*nbins, label='African children')
    # ax.set_xlabel('PfPR in 2-10 year olds')
    ax.set_ylabel('Density')
    # ax.set_title(f'Estimated mean PfPR in 2-10 year olds')
    ax.legend()

    ax = axs[0, 1]
    ax.plot('clinical_rate', 'clinical_survey_pdf', data=df, label='Surveys')
    ax.plot('clinical_rate', 'clinical_children_pdf', data=df, label='African children')
    # ax.set_xlabel('Episodes per child per year')
    # ax.set_title('Estimated average clinical\nepisode rates in 0-5 year olds')
    ax.set_xticks(np.arange(7))

    ax = axs[0, 2]
    ax.plot('infection_rate', 'infection_survey_pdf', data=df, label='Surveys')
    ax.plot('infection_rate', 'infection_children_pdf', data=df, label='African children')
    # ax.set_xlabel('Infections per child per year')
    # ax.set_title('Estimated blood infection rates')
    ax.set_xticks(np.arange(11))

    ax = axs[1, 0]
    sns.histplot(x='pr', data=map, cumulative=True, stat='percent', element='poly', bins=100, fill=False, ax=ax);
    ax.plot(pfpr_space_children, 100*(count_mean_total_africa/count_mean_total_africa.sum()).cumsum(), label='African children')
    ax.set_xlabel('PfPR in 2-10 year olds (%)')
    ax.set_ylabel('Percentage of surveyed\nsites/children under 10')
    ax.set_yticks(np.arange(0, 100.1, 10))

    ax = axs[1, 1]
    ax.plot('clinical_rate', 'clinical_survey_cdf', data=df, label='Surveys')
    ax.plot('clinical_rate', 'clinical_children_cdf', data=df, label='African children')
    ax.set_xlabel('Annual clinical episodes\nin the under fives')
    ax.set_xticks(np.arange(7))
    ax.set_yticks(np.arange(0, 100.1, 10))

    ax = axs[1, 2]
    ax.plot('infection_rate', 'infection_survey_cdf', data=df, label='Surveys')
    ax.plot('infection_rate', 'infection_children_cdf', data=df, label='African children')
    ax.set_xlabel('Annual blood infection rate')
    ax.set_xticks(np.arange(16))
    ax.set_yticks(np.arange(0, 100.1, 10))

    for ax, title in zip(axs.flatten(), 'ABCDEF'):
        ax.set_title(title, loc='left')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(visible=True, which='both', axis='both', color='lightgrey', linestyle='--', linewidth=0.5);

    return fig, axs

def trape_cumulative_episodes():
    episodes = pd.read_csv('Data/trape_supp_info_2.csv')
    episodes = trape.drop(columns=['ID'])


    cum_episodes = episodes.cumsum(axis=1)
    cum_episodes.columns = 0.5, 1, 4, 7, 10, 15, 20, 26
    cum_episodes = cum_episodes.melt(var_name='Age', value_name='Cumulative no. episodes')
    g = sns.displot(data=cum_episodes, col='Age', x='Cumulative no. episodes', common_bins=False, col_wrap=4, height=2.5)
    g.set_axis_labels("Cumulative no. episodes", "No. of people")
    g.set_titles("{col_name} years old");
    return g, cum_episodes

def distribution_sensitivity(fit, map, count_mean_total_africa):
    fig, axs = plt.subplots(2, 3, figsize=(12, 6), sharex='col', sharey='row')
    n = 5

    for ρ_elp in np.linspace(0.5, 1, n):
        λs, rates = infection_rate_vs_clinical_rate(ρ_elp=ρ_elp)
        df = construct_distributions(map, count_mean_total_africa, fit, rates, λs)
        df = df[df['infection_rate'] < 15.1]
        axs[0, 0].plot('infection_rate', 'infection_children_pdf', data=df)
        axs[1, 0].plot('infection_rate', 'infection_children_cdf', data=df)
        axs[0, 0].set_title('ρ_elp (0.5, 1)')

    for τ_elp in np.linspace(0.5, 2, n):
        λs, rates = infection_rate_vs_clinical_rate(τ_elp=τ_elp)
        df = construct_distributions(map, count_mean_total_africa, fit, rates, λs)
        df = df[df['infection_rate'] < 15.1]
        axs[0, 1].plot('infection_rate', 'infection_children_pdf', data=df)
        axs[1, 1].plot('infection_rate', 'infection_children_cdf', data=df)
        axs[0, 1].set_title('τ_elp (0.5, 2) years')

    for ρ_clinical in np.linspace(0.5, 1.0, n):
        λs, rates = infection_rate_vs_clinical_rate(ρ_clinical=ρ_clinical)
        df = construct_distributions(map, count_mean_total_africa, fit, rates, λs)
        df = df[df['infection_rate'] < 15.1]
        axs[0, 2].plot('infection_rate', 'infection_children_pdf', data=df)
        axs[1, 2].plot('infection_rate', 'infection_children_cdf', data=df)
        axs[0, 2].set_title('ρ_clinical (0.5, 1)')


    axs[0, 0].set_ylabel('Density')
    axs[1, 0].set_ylabel('Percentage of African\nchildren under 10')
    for ax in axs.flatten():
        ax.set_xlabel('Annual blood infection rate')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(visible=True, which='both', axis='both', color='lightgrey', linestyle='--', linewidth=0.5);

    return fig, axs

def lifetime_efficacies(
        ρ_elp=default_pars.ρ_elp,
        τ_elp=default_pars.τ_elp,
        ρ_clinical=default_pars.ρ_clinical,
        ε_severe=default_pars.ε_severe
        ):

    model = Model()
    model.pars = Parameters(
        lsv=True,
        β=0,
        ρ_elp=ρ_elp,
        τ_elp=τ_elp,
        ρ_clinical=ρ_clinical,
        ε_severe=ε_severe,
        study_months=60*12, # at worst, averted deaths at 30 years is about 1% lower than at 40 years for λ=0.5 and β=1
        season_width=0.11,
        nboosters_liver=1,
        vac_age_range=52
    )
    model.variables = Variables(['Direct deaths'])
    model.measures = Measures(['averted'])
    model.time_warning = True

    λs = np.logspace(log10(0.1), log10(25), 30)
    items = [[model, λ, True, False, None, 20*12 if λ > 0.5 else 60*12] for λ in λs]

    with Pool(12) as p:
        rs = p.starmap(paper.sim, items)

    # fill results with λ=0, averted=0 to get the origin
    results = {'λ':[0], 'averted':[0]}
    for predictions, i in zip(rs, items):
        results['λ'] += [i[1]]
        results['averted'] += [ predictions['Vaccine Direct deaths averted'].iloc[-1] ]
    r = pd.DataFrame(results)
    return r

def plot_weighted_averted_deaths(map, count_mean_total_africa):
    data = get_pfpr_vs_clinical()
    trape_fit = model_pfpr_clinical_relationship(data)[0][1]

    λs, rates = infection_rate_vs_clinical_rate()
    response = lifetime_efficacies()
    df = construct_distributions(map, count_mean_total_africa, trape_fit, rates, λs)

    λ = np.linspace(0, 15, 1000)
    p_λ = np.interp(λ, df['infection_rate'], df['infection_children_pdf'])
    f_λ = np.interp(λ, response['λ'], response['averted'])

    fig, axs = plt.subplots(1, 3, figsize=(11, 3))
    plt.subplots_adjust(wspace=0.4)

    ax = axs[0]
    ax.plot(λ, p_λ)
    ax = axs[1]
    ax.plot(λ, f_λ)
    ax = axs[2]
    ax.plot(λ, p_λ*f_λ)

    mask = f_λ < 0
    axs[1].annotate(r'$\lambda_\text{R21}$', (λ[mask][-1], 0), (λ[mask][-1]+5, 50), arrowprops=dict(arrowstyle="->"))
    axs[2].annotate(r'$\lambda_\text{R21}$', (λ[mask][-1], 0), (λ[mask][-1]+5, -5), arrowprops=dict(arrowstyle="->"))

    axs[1].axhline(0)
    axs[2].axhline(0)
    axs[0].set_ylabel('Density')
    axs[1].set_ylabel('Deaths averted per 100,000')
    axs[2].set_ylabel('Weighted deaths averted\nper 100,000')
    for ax, title in zip(axs.flatten(), 'ABC'):
        ax.set_xlabel('Annual blood infection rate')
        ax.set_title(title, loc='left')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(visible=True, which='both', axis='both', color='lightgrey', linestyle='--', linewidth=0.5);

    return fig, axs

def sensitivity_net_deaths(map, count_mean_total_africa):
    data = get_pfpr_vs_clinical()
    trape_fit = model_pfpr_clinical_relationship(data)[0][1]

    n = 5

    ρ_elp_results = []
    x_ρ_elp = np.linspace(0.5, 1, n)
    for ρ_elp in x_ρ_elp:
        λs, rates = infection_rate_vs_clinical_rate(ρ_elp=ρ_elp)
        response = lifetime_efficacies(ρ_elp=ρ_elp)
        df = construct_distributions(map, count_mean_total_africa, trape_fit, rates, λs)
        ρ_elp_results.append((response, df))

    τ_elp_results = []
    x_τ_elp = np.linspace(0.5, 2, n)
    for τ_elp in x_τ_elp:
        λs, rates = infection_rate_vs_clinical_rate(τ_elp=τ_elp)
        response = lifetime_efficacies(τ_elp=τ_elp)
        df = construct_distributions(map, count_mean_total_africa, trape_fit, rates, λs)
        τ_elp_results.append((response, df))

    ρ_clinical_results = []
    x_ρ_clinical = np.linspace(0.5, 1, n)
    for ρ_clinical in x_ρ_clinical:
        λs, rates = infection_rate_vs_clinical_rate(ρ_clinical=ρ_clinical)
        response = lifetime_efficacies(ρ_clinical=ρ_clinical)
        df = construct_distributions(map, count_mean_total_africa, trape_fit, rates, λs)
        ρ_clinical_results.append((response, df))

    ε_severe_results = []
    x_ε_severe = np.linspace(0.035, 0.075, n)
    for ε_severe in x_ε_severe:
        λs, rates = infection_rate_vs_clinical_rate()
        response = lifetime_efficacies(ε_severe=ε_severe)
        df = construct_distributions(map, count_mean_total_africa, trape_fit, rates, λs)
        ε_severe_results.append((response, df))

    r = (ρ_elp_results, τ_elp_results, ρ_clinical_results, ε_severe_results,
         x_ρ_elp, x_τ_elp, x_ρ_clinical, x_ε_severe)
    with open('Data/net_deaths.pickle', 'wb') as f:
        pickle.dump(r, f)

def plot_net_deaths():
    def plot(xs, results, axs, xlabel):
        λ = np.linspace(0, 25, 1000)
        lost = []
        gained = []
        net_averted = []
        net_gain = []

        for r in results:
            response, df = r
            p_λ = np.interp(λ, df['infection_rate'], df['infection_children_pdf'])
            p_λ /= np.trapz(p_λ, λ)
            f_λ = np.interp(λ, response['λ'], response['averted'])
            mask = f_λ < 0
            loss = -np.trapz(f_λ[mask]*p_λ[mask], λ[mask])
            gain = np.trapz(f_λ[~mask]*p_λ[~mask], λ[~mask])
            lost.append(loss)
            gained.append(gain)
            net_averted.append(gain - loss)
            net_gain.append(gain / loss)

        ax = axs[0]
        ax.plot(xs, lost)
        ax.scatter(xs[0], 0, color='w')
        ax = axs[1]
        ax.plot(xs, gained)
        ax.scatter(xs[0], 0, color='w')
        ax = axs[2]
        ax.plot(xs, net_averted)
        ax.scatter(xs[0], 0, color='w')
        ax = axs[3]
        ax.plot(xs, net_gain)
        ax.set_xlabel(xlabel)
        ax.scatter(xs[0], 0, color='w')

    with open('Data/net_deaths.pickle', 'rb') as f:
        (ρ_elp_results, τ_elp_results, ρ_clinical_results, ε_severe_results,
         x_ρ_elp, x_τ_elp, x_ρ_clinical, x_ε_severe) = pickle.load(f)

    fig, axs = plt.subplots(4, 4, figsize=(9, 9), sharex='col', sharey='row')

    plot(x_ρ_elp, ρ_elp_results, axs[:, 0], 'ρ_elp')
    plot(x_τ_elp, τ_elp_results, axs[:, 1], 'τ_elp')
    plot(x_ρ_clinical, ρ_clinical_results, axs[:, 2], 'ρ_clinical')
    plot(x_ε_severe, ε_severe_results, axs[:, 3], 'ε_severe')

    axs[0, 0].set_ylabel('Indirectly caused\ndeaths per 100,000')
    axs[1, 0].set_ylabel('Directly averted\ndeaths per 100,000')
    axs[2, 0].set_ylabel('Net deaths averted\nper 100,000')
    axs[3, 0].set_ylabel('Ratio of deaths averted\nto deaths caused')

    for ax in axs.flatten():
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(visible=True, which='both', axis='both', color='lightgrey', linestyle='--', linewidth=0.5);
    ymin = axs[0, 0].get_ylim()[0]
    for ax in axs.flatten()[:12]:
        ax.set_ylim(ymin, 100)

    return fig, axs

def test_min_study_months(λ=0.5, β=1):
    """
        test the minimum study months to get a 1% error in deaths averted
        worst case is lowest λ and highest β
        30 months appears to be the minimum
    """
    model = Model()
    model.pars = Parameters(
        lsv=True,
        β=β,
        λ=λ,
        season_width=0.11,
        nboosters_liver=1,
        vac_age_range=52
    )
    model.variables = Variables(['Direct deaths'])
    model.measures = Measures(['cdf', 'averted'])
    model.Config(time_warning=False)

    for y in np.arange(10, 90, 10):
        model.pars.study_months = 12 * y
        model.pars.update_pars('study_months')
        r = paper.simulation_single_weekly(model)
        print(y, r['Vaccine Direct deaths averted'].iloc[-1])
