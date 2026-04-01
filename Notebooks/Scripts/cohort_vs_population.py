import sys
sys.path.append('..')
from Edinburgh_Model.model import *
from Edinburgh_Model.simulate_single_vaccine import *
from Scripts import paper

import pandas as pd
from itertools import product
from multiprocessing import Pool
import matplotlib.pyplot as plt

def single_infection_rate():
    model = Model()
    model.pars = Parameters(
        lsv=True,
        λ=0.5,
        study_months=22*12,
        min_vac_age=int(round(5*weeks_per_month)),
    )
    model.variables = Variables(['Direct deaths'])
    model.measures = Measures()


    model.pars.children = 'cohort'
    model.pars.update_pars('children')
    rc = simulate_single_vaccine(model)

    model.pars.children = 'population'
    model.pars.update_pars('children')
    rp = simulate_single_vaccine(model)

    rc['averted'] = rc['control Direct deaths cases'] - rc['Vaccine Direct deaths cases']
    rp['averted'] = rp['control Direct deaths cases'] - rp['Vaccine Direct deaths cases']

    return rc, rp

def plot_single_infection_rate(rc, rp):
    fig, ax = plt.subplots(1, 1)
    ax.bar(rc.index-0.2, rc['averted'], color=paper.purple, width=0.4, label=f'{-rc["averted"].sum():.0f} net deaths caused in cohort')
    ax.bar(rp.index+0.2, rp['averted'], color=paper.blue, width=0.4, label=f'{rp["averted"].sum():.0f} net deaths averted in population')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Years post vaccination (cohort)\nYears since start of vaccine campaign (population)')
    ax.set_ylabel(f'Annual averted deaths per 100,000\nfully vaccinated children')
    ax.legend()
    return fig, ax

def sim(model, λ, children):
    model.pars.children = children
    model.pars.λ = λ
    model.pars.update_pars(('children', 'λ'))
    return simulate_single_vaccine(model).set_index('ages')

def efficacies_plot(all_sims, fig_xscale, fig_yscale):
    """ Plot predictions """

    titles = 'Deaths averted per 100,000', 'Death efficacy (%)'

    ncols, nrows = 2, 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(3*ncols*fig_xscale, 3*nrows*fig_yscale), sharex=True, squeeze=False)
    fig.subplots_adjust(hspace=0.1)

    for row, var in enumerate(all_sims['case'].unique()):
        data = all_sims.query('case == @var')
        for col, (variable, title) in enumerate(zip(('averted', 'efficacy'), titles)):
            ax = axs[row, col]

            for n, c, s in zip(data['treatment'].unique(), [paper.purple, paper.blue, paper.turquoise], ('-', '-', '-')):
                d = data.query('treatment == @n')
                ax.plot('λ', variable, ls=s, color=c, data=d)
                ax.scatter(x=0, y=0, color='white') # forces origin to be plotted
                ax.text(d['λ'].iloc[-1]+0.2, d[variable].iloc[-1], n, color=c, fontsize=10, ha='left', va='center')
                ax.set_xlim(ax.get_xlim()[0], 14)

            ax.axhline(0, ls='--', color='DarkGrey')
            ax.set_xlabel('Blood infections per year')
            ax.set_ylabel(title)

    for ax in axs.flatten():
        ax.set_xticks(range(0, 13, 2))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(visible=True, which='both', axis='both', color='lightgrey', linestyle='--', linewidth=0.5)

    return fig, axs

def efficacies():
    model = Model()
    model.pars = Parameters(
        lsv=True,
        study_months=15*12,
        min_vac_age=int(round(5*weeks_per_month, 0)),
    )
    model.variables = Variables(['Direct deaths'])
    model.measures = Measures(['efficacy', 'averted'])

    λs = np.logspace(np.log10(0.5), np.log10(10), 20)
    treatments = ['cohort', 'population']
    items = [[model, λ, t] for λ, t in product(λs, treatments)]

    with Pool() as p:
        rs = p.starmap(sim, items)

    results = {'λ':[], 'treatment':[], 'case':[], 'efficacy':[], 'averted':[]}
    for predictions, i in zip(rs, items):
        results['λ'] += [i[1]]
        results['treatment'] += [i[2]]
        results['case'] += ['Direct deaths']
        results['efficacy'] += [
            predictions['Vaccine Direct deaths efficacy'].iloc[-1]
        ]
        results['averted'] += [
            predictions['Vaccine Direct deaths averted'].iloc[-1]
        ]
    r = pd.DataFrame(results)
    return r

