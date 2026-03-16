import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from pygam import LogisticGAM, s, f
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm

import sys
sys.path.append('..')
from Edinburgh_Model.model import *

data = pd.read_csv('Data/rspb20142657supp2.csv')
data['transmission'] = data['altitude band'].replace({1: 'high', 2: 'medium', 3: 'low'})
# subtract 1 from altitude band so that it's partial dependence can be plotted
data['altitude band'] -= 1

def median_ages():
    return data.groupby(['transmission'])['age (years)'].median(), \
        data.query('died == 1').groupby(['transmission'])['age (years)'].median()

def plot_raw_data():
    x = data.copy()
    x['died'] = x['died'].replace({0: 'survived', 1: 'died'})
    g = sns.FacetGrid(x, sharey='row', col='died', row='transmission', row_order=('high', 'medium', 'low'), height=1.75)
    g.map(sns.histplot, 'age (years)', bins=90 )
    g.set_titles(col_template='{col_name}', row_template='{row_name}')
    return g

def pool_data():
    # histograms of cases stratfied by age_categories
    age_categories = [0, 1, 2, 3, 4, 5, 10, 15, 20, 80]
    n = 9
    h_alive = np.histogram(data.query('died == 0')['age (years)'], bins=age_categories)
    h_dead = np.histogram(data.query('died == 1')['age (years)'], bins=age_categories)
    r = h_dead[0][:n] / h_alive[0][:n]
    return h_dead[1][:n], r

def plot_risk_pooled_data():
    h_dead, rate = pool_data()
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.scatter(h_dead, rate)
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Risk of severe malaria death');
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(visible=True, which='both', axis='both', color='lightgrey', linestyle='--', linewidth=0.5)
    return fig, ax

def fit_GAM1():
    # Use only up to 20 years old as cases become scarce after that
    d = data[data['age (years)'] <= 20]
    X = d[['age (years)']].values
    y = d['died'].values

    gam = LogisticGAM(s(0), fit_intercept=False).gridsearch(X, y, progress=False)
    return gam

def fit_GAM2():
    # Use only up to 20 years old as cases become scarce after that
    d = data[data['age (years)'] <= 20]
    X = d[['age (years)', 'altitude band']].values
    y = d['died'].values

    gam = LogisticGAM(s(0)+f(1), fit_intercept=False).gridsearch(X, y, progress=False)
    return gam

def plot_partial_dependence(gam):
    fig, axs = plt.subplots(1, 2, figsize=(7, 3))
    fig.subplots_adjust(wspace=0.3)
    for i, term in enumerate(gam.terms):
        ax = axs[i]
        XX = gam.generate_X_grid(term=i)
        pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)

        ax.plot(XX[:, term.feature], pdep)
        ax.plot(XX[:, term.feature], confi, c='r', ls='--')
        # ax.title(repr(term))
        if i == 0:
            ax.set_xlabel('Age (years)')
        elif i == 1:
            ax.set_xlabel('Transmission intensity')
            ax.set_xticks([0, 1, 2], ['high', 'medium', 'low'])
        ax.set_ylabel('Partial dependence')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    return fig, axs

def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs = {}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)

def death_rate_mean_CI(gam):
    """ from GAM calculate mean age-specific death rate and its 95% CI"""

    XX = gam.generate_X_grid(term=0, n=100) # age in years
    quantiles = np.linspace(0.025, 0.975, 3)
    cis = 1-2*abs(quantiles-0.5)

    pdep, confi = gam.partial_dependence(term=0, X=XX, quantiles=quantiles)
    f = lambda x: 1 / (1+np.exp(-x))
    fit = f(confi)
    death_rates = f(pdep)
    # death rates up to the maximum death rate
    idx = np.argmax(death_rates)+1
    death_rate_by_age = death_rates[:idx]
    # death rates up to the maximum and add the last value so that death rates plateau
    dr = lambda x: np.interp(x, np.concatenate((XX[:idx, 0], [100])), list(death_rate_by_age)+[death_rate_by_age[-1]])

    return XX, fit, dr, cis

def plot_GAM_mean(death_rates):
    fig, ax = plt.subplots()

    # Add the GAM fit
    x = np.arange(16)
    ax.plot(x, death_rates(x))

    ax.set_xlabel('Age (years)', fontsize=18)
    ax.set_ylabel('Risk of severe malaria death', fontsize=18);
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return fig, ax

def plot_GAM_mean_CI_data(XX, fit, death_rates, cis, arrows=None):
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    x = np.arange(21)

    # Add 95% CIs
    polygon = ax.fill_between(XX[:, 0], fit[:, 0], fit[:, -1], lw=0, color='none')
    verts = np.vstack([p.vertices for p in polygon.get_paths()])
    ymin, ymax = verts[:, 1].min(), verts[:, 1].max()
    z = np.array([np.interp(np.linspace(ymin, ymax, 200), y, cis/(y.max()-y.min())) for y in fit]).T
    gradient = ax.imshow(z, cmap='Blues', aspect='auto', origin='lower', extent=[x.min(), x.max(), ymin, ymax])
    gradient.set_clip_path(polygon.get_paths()[0], transform=ax.transData)

    # Add the GAM fit
    ax.plot(XX[:, 0], death_rates(XX[:, 0]), ls='--', lw=0.9, label='GAM fit')

    # Add the data
    d = data[data['age (years)'] <= 20]
    h_dead, rate = pool_data()
    ax.scatter(h_dead, rate, label=f'data, $n$={len(d)}', facecolor='w', color='C0', zorder=999)

    if arrows is not None:
        for arrow in arrows:
            ax.annotate("", xytext=arrow[0], xy=arrow[1], arrowprops=dict(arrowstyle="->", lw=2))

    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Risk of severe malaria death');
    ax.set_xlim(-1, 21)
    ax.set_ylim(0, 0.23)
    ax.set_xticks(np.arange(0, 21, 2))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend()

    return fig, ax

def plot_GAM_mean_CI_data_infection(fit, death_rates, cis, sm_age, λs):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    x = np.arange(21)
    XX = np.arange(0, 21*2, 0.1).reshape(2,210).T

    # Add 95% CIs
    polygon = ax.fill_between(XX[:, 0], fit[:, 0], fit[:, -1], lw=0, color='none')
    verts = np.vstack([p.vertices for p in polygon.get_paths()])
    ymin, ymax = verts[:, 1].min(), verts[:, 1].max()
    z = np.array([np.interp(np.linspace(ymin, ymax, 200), y, cis/(y.max()-y.min())) for y in fit]).T
    gradient = ax.imshow(z, cmap='Blues', aspect='auto', origin='lower', extent=[x.min(), x.max(), ymin, ymax])
    gradient.set_clip_path(polygon.get_paths()[0], transform=ax.transData)

    # Add the average age of SM
    lines = colored_line(sm_age, death_rates(sm_age), λs, ax, linewidth=6, cmap="jet",  norm=LogNorm(vmin=λs.min(), vmax=λs.max()))
    fig.colorbar(lines, format='%g', label='Blood infections per child per year')

    # Add the GAM fit
    ax.plot(x, death_rates(x), ls='--', lw=0.9, label='GAM fit')

    # Add the data
    d = data[data['age (years)'] < 15]
    h_dead, rate = pool_data()
    ax.scatter(h_dead, rate, label=f'data, $n$={len(d)}', facecolor='w', color='C0', zorder=999)

    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Risk of severe malaria death');
    ax.set_title('Age-specific, hospitalised, severe malaria\ncase fatality rates (Reyburn et al. 2005)')
    ax.set_xlim(-1, 21)
    ax.set_ylim(-0.02, 0.3)
    ax.set_xticks(np.arange(0, 21, 2))
    ax.legend()

    fig.savefig('Figures/reyburn_2015_mort_with_a_SM.svg', bbox_inches='tight')

def plot_infection(death_rates, sm_age, λs, filename, arrow=None):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    # Add the average age of SM
    lines = colored_line(sm_age, death_rates(sm_age), λs, ax, linewidth=6, cmap="jet",  norm=LogNorm(vmin=λs.min(), vmax=λs.max()))
    fig.colorbar(lines, format='%g', label='Blood infections per child per year')

    if arrow is not None:
        ax.annotate("", xytext=arrow[0], xy=arrow[1], arrowprops=dict(arrowstyle="->", lw=2))

    ax.set_xlabel('Average age of severe malaria (years)')
    ax.set_ylabel('Risk of severe malaria death')
    ax.set_title('Age-specific, hospitalised, severe malaria\ncase fatality rates')
    ax.set_ylim(0, 0.18)
    ax.set_xticks(np.arange(0, 13, 2))

    fig.savefig(f'Figures/{filename}', bbox_inches='tight')

def death_rates_with_arrows():
    gam = fit_GAM1()
    XX, fit, death_rates, cis = death_rate_mean_CI(gam)
    offset = 0.015
    a = (
        ( (0.25, death_rates(0.25)+offset), (2.25, death_rates(2.25)+offset) ),
        ( (0.25, death_rates(0.25)+offset), (8, death_rates(8)+offset) ),
        ( (5, death_rates(5)+offset), (7, death_rates(7)+offset) )
    )
    # plot_GAM_mean_CI_data(XX, fit, death_rates, cis, 'reyburn_2015_mort_1_arrow.svg', arrows=a[:1])
    plot_GAM_mean_CI_data(XX, fit, death_rates, cis, arrows=a)

