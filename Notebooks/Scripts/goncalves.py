import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from numpy import ma
from pandas import read_csv
from bisect import bisect_left
from numpy.random import poisson
from collections import namedtuple

Pars_goncalves = namedtuple('Parameters', 'λ, μ, σ, aF, cF, aS, cS, bS, dS, P_fever, P_detect, P_fever_and_detect, P_fever_and_5000, I , P_severe')

# class RaisedCosine:
#     """ the pdf, derivative of the pdf and the cdf of the raised cosine distribution
#         on the domain (μ-s, μ+s) """
#     def __init__(self, μ, s):
#         self.μ = μ
#         self.s = s
#     def pdf(self, x):
#         z = (x - self.μ) / self.s
#         return (1 + np.cos(z*np.pi)) / (2*self.s)
#     def dpdf(self, x):
#         z = (x - self.μ) / self.s
#         return z/2 + np.sin(z*np.pi) / (2*np.pi)
#     def cdf(self, x):
#         z = (x - self.μ) / self.s
#         return (1 + z + np.sin(z*np.pi)/np.pi) / 2


# def simulate_infections():
#     df = pd.read_csv('../Data_Fitting/Goncalves/goncalves_draws.csv')
#     means = df.mean()
#     λ = means.loc['lambda']
#     μ = means.loc['mu']
#     s = means.loc['s']
#     aF = means.loc['aF']
#     cF = means.loc['cF']
#     aS = means.loc['aS']
#     cS = means.loc['cS']
#     bS = means.loc['bS']
#     dS = means.loc['dS']

#     dist = RaisedCosine(μ, s)
#     urng = np.random.default_rng()
#     raised_cosine = NumericalInverseHermite(dist, domain=(μ-s, μ+s), order=5, random_state=urng)

#     p_fever_given_x = lambda x: 1 / (1 + np.exp(-aF*(x - cF)))
#     p_SM_given_fever_n_x = lambda n, x: bS * np.exp(-dS*(n-1)) / (1 + np.exp(-aS*(x - cS)))
#     p_SM_given_n_x = lambda n, x: p_SM_given_fever_n_x(n, x) * p_fever_given_x(x)

#     infections = {'age':[], 'parasitaemia':[], 'SM':[]}
#     for t in np.arange(1, 200):
#         # number of kids drops off over time
#         C = 882 - 4*t
#         # number of infection this week
#         N = binomial(int(C), λ/52)
#         # parasitaemias >= 1, converted to int
#         X = 10**raised_cosine.rvs(size=N)
#         # X = 10**normal(μ, s, size=N)
#         X = X[X >= 1]
#         X = np.array([int(i) for i in X])

#         # expected cumulative exposure this week
#         mean_n = 1 + λ * t / 52
#         # randomly select SM episodes
#         SM = binomial(1, p_SM_given_n_x(mean_n, np.log10(X)))

#         infections['age'] += [t]*len(X)
#         infections['parasitaemia'] += list(X)
#         infections['SM'] += list(SM)

#     infections = pd.DataFrame(infections).sort_values('SM')
#     infections['Episode'] = infections['SM'].apply(lambda x: 'Severe Malaria' if x else 'Mild/Asymptomatic')
#     ax = sns.scatterplot(data=infections, x='age', y='parasitaemia', hue='Episode', palette=['C0', 'Red'], alpha=0.5, markers=('o', 's'), style='Episode')
#     ax.set_xlabel('Age (weeks)')
#     ax.set_ylabel('Parasitaemia (per 200 WBCs)');
#     ax.set_yscale('log')
#     ax.set_title(f"{len(infections['age']):,} simulated malaria episodes");


def figS2():
    goncalves_figS2 = read_csv('Data/goncalves_figS2.csv')
    goncalves_figS2.iloc[:, 1:] = 10**goncalves_figS2.iloc[:, 1:]
    goncalves_rows = goncalves_figS2.to_dict(orient='records')

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.bxp(goncalves_rows, patch_artist=True, boxprops={'facecolor': 'C0'}, showfliers=False);
    ax.set_yscale('log')
    ax.set_xticks((1, 2, 3), ['Severe', 'Mild', 'Asymptomatic'])
    ax.set_xlabel('Disease severity');
    ax.set_ylabel('Parasites per 200 WBC');
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return fig, ax

def fig2C():
    with open('../Data_Fitting/Goncalves/goncalves.data.json') as f:
        dj = json.load(f)
    sm_risks = pd.DataFrame({'n':dj['SM_risk'][0], 'm':dj['SM_risk'][1]})
    sm_risks['p'] = sm_risks['n'] / sm_risks['m'] * 100
    n = np.arange(1, len(sm_risks['p'])+1, 1)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.scatter(n, sm_risks['p'], color='C0', s=75, edgecolor='w')
    ax.set_ylabel('Risk of clinical malaria\nbecoming severe (%)')
    ax.set_xlabel('Number of bllod infections');
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for i, r in sm_risks.iterrows():
        ax.annotate(f"{r['n']:.0f}/{r['m']:.0f}", xy=(int(i)+1, r['p']+0.2), ha='center', va='bottom', fontsize=7)

    return fig, ax

def risk_fever_with_parasitaemia():
    with plt.xkcd():

        fig, ax = plt.subplots(figsize=(3, 3))
        a = 0.5
        d = 0.3

        x = np.linspace(-10, 10, 100)
        y = lambda x, n: np.exp(-d*(n-1)) / (1 + np.exp(-a*x))

        ax.plot(x, y(x, 1))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('log-Parasitaemia')
        ax.set_ylabel('Risk of blood infection\nbecoming clinical')
        ax.set_xticks([])
        ax.set_yticks([])
    return fig, ax

def risk_severe_with_parasitaemia():
    with plt.xkcd():

        fig, ax = plt.subplots(figsize=(3, 3))
        a = 0.5
        d = 0.3

        x = np.linspace(-10, 10, 100)
        y = lambda x, n: np.exp(-d*(n-1)) / (1 + np.exp(-a*x))

        ax.plot(x, y(x, 1), label='first infection')
        ax.plot(x, y(x, 2), label='second infection')
        ax.plot(x, y(x, 3), label='third infection')
        ax.text(10, 1.0, 'first infection', ha='right', color='C0')
        ax.text(10, 0.75, 'second infection', ha='right', color='C1')
        ax.text(10, 0.56, 'third infection', ha='right', color='C2')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('log-Parasitaemia')
        ax.set_ylabel('Risk of clinical malaria\nbecoming severe')
        ax.set_xticks([])
        ax.set_yticks([])
    return fig, ax

def distributions(filename):
    df = read_csv(filename)
    means = df.mean()
    μ = means.loc['mu']
    s = means.loc['s']
    aF = means.loc['aF']
    cF = means.loc['cF']
    aS = means.loc['aS']
    cS = means.loc['cS']
    bS = means.loc['bS']
    dS = means.loc['dS']

    x1 = μ - s
    x2 = μ + s

    x = np.linspace(x1, x2, 1000).astype(float)

    p_n = [0.166, 0.163, 0.156, 0.141, 0.119, 0.0926, 0.0657,
        0.04254, 0.0254, 0.0139, 0.00712, 0.00328, 0.00140, 0.000532, 0.000211];
    f = sum([p_n[n-1] * np.exp(-dS*(n-1)) for n in range(1, len(p_n)+1)])

    p_fever_given_x = lambda x: 1 / (1 + np.exp(-aF*(x - cF)))
    p_SM_given_fever_x = lambda x: bS * f / (1 + np.exp(-aS*(x - cS)))
    p_x = lambda x: (1+np.cos((x-μ)/s*np.pi)) / (2*s)
    p_fever_x = lambda x: p_fever_given_x(x) * p_x(x)
    p_SM_x = lambda x: p_SM_given_fever_x(x) * p_fever_given_x(x) * p_x(x)
    p_asym_x = lambda x: p_x(x) - p_fever_x(x)
    p_mild_x = lambda x: p_fever_x(x) - p_SM_x(x)

    dists = {'x':10**x,
            'p_x':p_x(x),
            'P_fever_given_x':100 * p_fever_given_x(x),
            'P_SM_given_x_fever':100 * p_SM_given_fever_x(x),
            'p_x_fever':p_fever_x(x),
            'p_x_no_fever':p_asym_x(x),
            'p_x_SM':p_SM_x(x),
            'p_x_mild':p_mild_x(x),
    }

    boxes = []

    def box_plot(px, boxes):
        q = 'whislo', 'q1', 'med', 'q3', 'whishi'
        v = 0.05, 0.25, 0.5, 0.75, 0.95
        cdf = px.cumsum() / px.sum()
        y = dict((i, 10**x[bisect_left(cdf, j)]) for i, j in zip(q, v))
        boxes.append(y)

    def truncate(p, mask):
        y = ma.array(data=p, mask=mask)
        return ma.filled(y, 0)

    # index in x closest to 0
    i0 = bisect_left(x, 0)
    # create a mask so that pdf(x) = 0 for x < 0
    mask = np.zeros_like(x, dtype=bool)
    mask[:i0] = True

    box_plot(truncate(p_SM_x(x), mask), boxes)
    box_plot(truncate(p_mild_x(x), mask), boxes)
    box_plot(truncate(p_asym_x(x), mask), boxes)

    goncalves_figS2 = read_csv('Data/goncalves_figS2.csv')
    goncalves_figS2.iloc[:, 1:] = 10**goncalves_figS2.iloc[:, 1:]
    goncalves_rows = goncalves_figS2.to_dict(orient='records')

    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    ax = axs[0]
    ax.plot(dists['x'], dists['p_x'], label='All')
    ax.plot(dists['x'], dists['p_x_no_fever'], label='Asymptomatic')
    ax.plot(dists['x'], dists['p_x_mild'], label='Mild')
    ax.plot(dists['x'], dists['p_x_SM'], label='Severe')
    ax.set_xlabel('Parasites per 200 WBC')
    ax.set_ylabel('Probability density')
    ax.set_xscale('log')
    ax.legend(loc='upper left')
    ax.set_title('A', loc='left');

    ax = axs[1]
    ax.bxp(goncalves_rows, positions=(0.8, 1.8, 2.8), patch_artist=True, boxprops={'facecolor': 'C0'}, showfliers=False);
    ax.bxp(boxes, positions=(1.2, 2.2, 3.2), showfliers=False);
    ax.set_yscale('log')
    ax.set_xticks((1, 2, 3), ['Severe', 'Mild', 'Asymptomatic'])
    ax.set_xlabel('Disease severity');
    ax.set_ylabel('Parasites per 200 WBC');
    ax.set_title('B', loc='left');

    for ax in axs.flatten():
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    return fig, axs


def risk_estimates(filename):
    def plot(ax, x, y, xlabel, ylabel, title, ccdf=False, color=None, label='', xaxis='log', yaxis='linear'):
        y_lower = np.percentile(y, 2.5, axis=0)
        y_upper = np.percentile(y, 97.5, axis=0)
        if color is None:
            ax.plot(x, np.mean(y, axis=0), label=label)
            ax.fill_between(x, y_lower, y_upper, alpha=0.5)
        else:
            ax.plot(x, np.mean(y, axis=0), label=label, color=color)
            ax.fill_between(x, y_lower, y_upper, alpha=0.5, color=color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        # ax.set_title(title)
        if xaxis == 'log':
            ax.set_xscale('log')
        if yaxis == 'log':
            ax.set_yscale('log')

    def ccdf(f, dx):
        return dx * np.cumsum(f[:,::-1], axis=1)[:,::-1]

    def convert(z):
        if isinstance(z, list):
            return [10**float(i) for i in z]
        else:
            return 10**float(z)

    with open('../Data_Fitting/Goncalves/goncalves.data.json') as f:
        dj = json.load(f)
    sm_risks = pd.DataFrame({'n':dj['SM_risk'][0], 'm':dj['SM_risk'][1]})
    sm_risks['p'] = sm_risks['n'] / sm_risks['m'] * 100
    df = pd.read_csv(filename)

    λ  = np.array(df['lambda'])[:, np.newaxis]
    μ  = np.array(df['mu'])[:, np.newaxis]
    s  = np.array(df['s'])[:, np.newaxis]
    aS = np.array(df['aS'])[:, np.newaxis]
    bS = np.array(df['bS'])[:, np.newaxis]
    cS = np.array(df['cS'])[:, np.newaxis]
    dS = np.array(df['dS'])[:, np.newaxis]
    aF = np.array(df['aF'])[:, np.newaxis]
    try:
        cF = np.array(df['cF'])[:, np.newaxis]
    except:
        cF = np.zeros_like(aF)
    I  = np.array(df['I'])[:, np.newaxis]
    Pd = np.array(df['P_detect'])[:, np.newaxis]
    P_SM = np.array(df['P_SM'])[:, np.newaxis]

    n = np.arange(1, len(sm_risks['p'])+1, 1)
    logx = np.linspace(-1.0, 5.0, 100)
    x = 10**logx
    dx = logx[1] - logx[0]

    p_n = dj['p_n'][3]
    f = sum([p_n[n-1] * np.exp(-dS*(n-1)) for n in range(1, len(p_n)+1)])

    p_fever_given_x = lambda x: 1 / (1 + np.exp(-aF*(x - cF)))
    p_SM_given_fever_n_x = lambda x, n: bS * np.exp(-dS*(n-1)) / (1 + np.exp(-aS*(x - cS)))
    p_SM_given_fever_x = lambda x: bS * f / (1 + np.exp(-aS*(x - cS)))
    p_x = lambda x: (1+np.cos((x-μ)/s*np.pi)) / (2*s)
    p_fever_x = lambda x: p_fever_given_x(x) * p_x(x)
    p_SM_x = lambda x: p_SM_given_fever_x(x) * p_fever_given_x(x) * p_x(x)
    p_asym_x = lambda x: p_x(x) - p_fever_x(x)
    p_mild_x = lambda x: p_fever_x(x) - p_SM_x(x)
    p_SM_given_fever_n = lambda i: I * bS * np.exp(-dS*(i-1)) / Pd

    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    plot(axs[0, 0], x, 100*p_fever_given_x(logx), 'Parasites per 200 WBC', 'Risk of blood infection\nbecoming clinical (%)', 'A) Estimated risk of fever\nwith parasitaemia')
    plot(axs[0, 1], x, 100*p_SM_given_fever_n_x(logx, 1), 'Parasites per 200 WBC', 'Risk of clinical malaria\nbecoming severe (%)', '', label='1st infection')
    plot(axs[0, 1], x, 100*p_SM_given_fever_n_x(logx, 5), 'Parasites per 200 WBC', 'Risk of clinical malaria\nbecoming severe (%)', '', label='5th infection')
    plot(axs[0, 1], x, 100*p_SM_given_fever_n_x(logx, 10), 'Parasites per 200 WBC', 'Risk of clinical malaria\nbecoming severe (%)', 'B) Estimated risk of severe malaria\nwith parasitaemia', label='10th infection')
    plot(axs[1, 0], n, 100*p_SM_given_fever_n(n), 'Number of blood infections', 'Risk of clinical malaria\nbecoming severe (%)', 'C) Fit of risk of severe malaria\nwith exposure', xaxis='linear')
    plot(axs[1, 1], x, λ * dj['child_years'] * ccdf(p_x(logx), dx), 'Parasites per 200 WBC', '', '', label='All')
    plot(axs[1, 1], x, λ * dj['child_years'] * ccdf(p_mild_x(logx), dx), 'Parasites per 200 WBC', '', '', label='Mild')
    plot(axs[1, 1], x, λ * dj['child_years'] * ccdf(p_asym_x(logx), dx), 'Threshold parasites per 200 WBC', 'Episodes above threshold parasitaemia', 'D) Fit of episodes\nabove threshold parasitaemia', label='Asymptomatic')
    plot(axs[1, 2], x, 122*ccdf(p_SM_x(logx) / P_SM, dx), 'Threshold parasites per 200 WBC', 'Episodes above threshold parasitaemia', 'E) Fit of severe malaria episodes\nabove threshold parasitaemia', color='C3', label='Severe')

    axs[0, 2].set_visible(False)
    axs[0, 0].set_ylim(-5, 100);
    axs[0, 1].legend(title='Infection number', loc='upper left')
    axs[1, 1].legend(title='Disease severity')
    axs[1, 2].legend(title='Disease severity')

    axs[1, 0].scatter(n, sm_risks['p'], color='C0', s=75, edgecolor='w')
    axs[1, 1].scatter(1, dj['cases'], color='C0', s=75, edgecolor='w')
    axs[1, 1].scatter(convert(dj['mild_logp']), dj['mild'], color='C1', s=75, edgecolor='w')
    axs[1, 1].scatter(convert(dj['asym_logp']), dj['asym'], color='C2', s=75, edgecolor='w')
    axs[1, 2].scatter(convert(dj['SM_logp']), np.array(dj['SM']), color='C3', s=75, edgecolor='w')

    axs[1, 0].set_xticks(np.arange(1, 15, 2))

    for ax, title in zip(axs.flatten(), 'AB CDE'):
        ax.set_title(title, loc='left')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    return fig, axs


def get_estimates(filename):
    df = pd.read_csv(filename)

    λ = np.array(df['lambda'])
    μ = np.array(df['mu'])
    σ = np.array(df['s'])
    aF = np.array(df['aF'])
    cF = np.array(df['cF'])
    aS = np.array(df['aS'])
    cS = np.array(df['cS'])
    bS = np.array(df['bS'])
    dS = np.array(df['dS'])
    P_fever = np.array(df['P_fever'])
    P_detect = np.array(df['P_detect'])
    P_fever_and_detect = np.array(df['P_fever_and_detect'])
    P_fever_and_5000 = np.array(df['P_fever_and_5000'])
    I  = np.array(df['I'])
    P_severe = I * bS / P_detect

    return Pars_goncalves(
        λ,
        μ,
        σ,
        aF,
        cF,
        aS,
        cS,
        bS,
        dS,
        P_fever,
        P_detect,
        P_fever_and_detect,
        P_fever_and_5000,
        I,
        P_severe,
    )


def simulate_p_n():
    c = 882 # number of children
    tbar = 1762.8/c # assume 2 years of follow-up
    df = pd.DataFrame()

    # examine infection rates from 2.0 to 3.0
    for λ in np.arange(2.0, 3.1, 0.2):
        for _ in range(100):
            Y = poisson(λ*tbar, size=c)
            ns = np.arange(1, Y.max()+1)

            M = Y.sum()
            p = [(Y >= n).sum() / M for n in ns]

            df = pd.concat((df, pd.DataFrame({'λ': [λ.round(1)]*len(ns), 'n': ns, 'p': p})))

    fig, ax = plt.subplots(figsize=(3, 2.5))
    ax = sns.lineplot(data=df, x='n', y='p', hue='λ', errorbar=('sd', 2), ax=ax)

    ax.set_xlabel('Number of blood infections')
    ax.set_ylabel(r'$p(n|\lambda\overline{t})$')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return fig, ax, df


def p_n_values(df):
    p_n = df.groupby(['λ', 'n']).mean().reset_index()
    p_pivot = p_n.pivot(index='n', columns='λ', values='p').sort_index()
    p_pivot = p_pivot.replace(np.nan, 0)
    return p_pivot.T.values.round(3)
