import sys
sys.path.append('..')
from Edinburgh_Model.model import *
from Edinburgh_Model.vaccination_types import *
from Edinburgh_Model.simulate_single_vaccine import *

from . import paper
from . import reyburn

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import to_hex
from itertools import product
from multiprocessing import Pool


mpl.rcParams['xtick.major.size'] = 0
mpl.rcParams['ytick.major.size'] = 0
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['grid.alpha'] = 0.5

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica Neue']

pink = to_hex(np.array([255, 94, 94])/255)
purple = to_hex(np.array([51, 0, 153])/255)
blue = to_hex(np.array([0, 153, 255])/255)
turquoise = to_hex(np.array([39, 255, 183])/255)

weeks_per_month = 52/12
vars = ['All infection', 'All clinical', 'Severe malaria', 'Direct deaths']
titles = 'Blood infection', 'Clinical malaria', 'Severe malaria', 'Death'

"""
    In seaonal sites we assume a 3.3 month season length (season_width=1, estimated form Dicko 2024).
    Vaccinate children from 5 to 17 months (so last primary dose is 7 to 19 months)
    Assuming 1 booster and 1 repeat of SMC when used
    To achieve maximum RTS,S and R21 efficacy, the last primary dose must be given 6 weeks before the peak of the malaria season (season_peak=6).
    To achieve maximum RH5 efficacy, the last primary dose must be given 5 weeks before the peak of the malaria season (season_peak=4).
    With RTS,S and R21, SMC is given 3 weeks before the last primary dose for maximum clinical efficacy (smc_offset=-3).
    With RH5, SMC is given 4 weeks before the last primary dose for maximum clinical efficacy (smc_offset=-4).
    (max clinical efficacy and max severe efficacy do not coincide for the same value of smc_offset, max efficacy is insensitive for +/- a week of two)

"""
season_peak_liver = 6
season_peak_blood = 4
smc_offset_liver = -3
smc_offset_blood = -4
min_vac_age_liver = int(7*weeks_per_month)
min_vac_age_blood = int(10*weeks_per_month)
vac_age_range = 52

pars = Parameters()

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap

def sim(model, λ, lsv, bsv, parameter, value):
    model.pars.λ = λ
    model.pars.lsv = lsv
    model.pars.bsv = bsv
    model.pars.study_months = 20*12 if λ > 0.5 else 60*12

    if lsv and bsv:
        model.pars.season_peak = min_vac_age_blood - min_vac_age_liver + season_peak_blood # time peak season relative to last primary dose of RH5 which occurs 3 months after last primary dose of R21
        model.pars.min_vac_age = min_vac_age_liver
        model.pars.offset_blood = min_vac_age_blood - min_vac_age_liver # difference of 3 months between last R21 dose and last RH5 dose
    elif lsv:
        model.pars.season_peak = season_peak_liver
        model.pars.min_vac_age = min_vac_age_liver
    elif bsv:
        model.pars.season_peak = season_peak_blood
        model.pars.min_vac_age = min_vac_age_blood

    model.pars.update_pars(('λ', 'lsv', 'bsv', 'study_months', 'season_peak', 'min_vac_age', 'offset_blood'))

    if parameter == 'ρ_elp':
        model.pars.ρ_elp = value
        model.pars.update_pars('ρ_elp')
    elif parameter == 'τ_elp':
        model.pars.τ_elp = value
        model.pars.update_pars('τ_elp')
    elif parameter == 'ρ_clinical':
        model.pars.ρ_clinical = value
        model.pars.update_pars('ρ_clinical')
    elif parameter == 'δ_clinical':
        model.pars.δ_clinical = value
        model.pars.update_pars('δ_clinical')
    elif parameter == 'γ_clinical':
        model.pars.γ = value
        model.pars.update_pars('γ_clinical')
    elif parameter == 'ρ_severe':
        model.pars.ρ_severe = value
        model.pars.update_pars('ρ_severe')
    elif parameter == 'δ_severe':
        model.pars.δ_severe = value
        model.pars.update_pars('δ_severe')
    elif parameter == 'ε_severe':
        model.pars.ε_severe = value
        model.pars.update_pars('ε_severe')
    elif parameter == 'β':
        model.pars.β = value
        model.pars.update_pars('β')
    elif parameter == 'ν_liver':
        model.pars.ν_liver = value
        model.pars.update_pars('ν_liver')
    elif parameter == 'τ_liver':
        model.pars.τ_liver = value
        model.pars.update_pars('τ_liver')
    elif parameter == 'nboosters_liver':
        model.pars.nboosters_liver = value
        model.pars.update_pars('nboosters_liver')
    elif parameter == 'ν_blood':
        model.pars.ν_blood = value
        model.pars.update_pars('ν_blood')
    elif parameter == 'τ_blood':
        model.pars.τ_blood = value
        model.pars.update_pars('τ_blood')
    elif parameter == 'nboosters_blood':
        model.pars.nboosters_blood = value
        model.pars.update_pars('nboosters_blood')
    elif parameter == 'ω_infection':
        model.pars.ω_infection = value
        model.pars.update_pars('ω_infection')
    elif parameter == 'ω_clinical':
        model.pars.ω_clinical = value
        model.pars.update_pars('ω_clinical')
    elif parameter == 'ω_severe':
        model.pars.ω_severe = value
        model.pars.update_pars('ω_severe')
    elif parameter == 'death_rate_modifier':
        model.pars.death_rate_modifier = value
        model.pars.update_pars('death_rate_modifier')

    return simulate_single_vaccine(model)

def plot(sim, default_value=None):
    from scipy.optimize import elementwise

    def get_root(x, y):
        f = lambda xp: np.interp(xp, x, y)
        return elementwise.find_root(f, (x[0], x[-1]))

    """ Plot predictions """
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.5))
    values = sim['value'].unique()
    nv = max(1, len(values) // 2)
    l = None

    cmap = plt.get_cmap('plasma')
    new_cmap = truncate_colormap(cmap, 0, 0.8)

    for col, measure in enumerate(['averted', 'efficacy']):
        ax = axs[col]

        xi, yi = [], []
        xr21, yr21 = [], []
        for i, v in enumerate(values):
            for t, c in zip(sim['treatment'].unique(), [purple, blue, turquoise]):
                d = sim.query('treatment == @t and value == @v')
                ax.plot('λ', measure, color=c, data=d, lw=1-abs(i-nv)/nv+0.2)
                # ax.plot('λ', measure, color=c, data=d, lw=1)
                if i == nv:
                    ax.text(d['λ'].iloc[-1]+0.2, d[measure].iloc[-1], t, color=c, fontsize=10, ha='left', va='center')

            # find root of R21 line
            dr21 = sim.query('treatment == "R21" and value == @v').set_index('λ')
            root = get_root(dr21.index, dr21[measure])
            if not root.success:
                raise UserWarning(f'root not found: {measure}, R21, {v}')
            xr21.append(root.x)
            yr21.append(np.interp(root.x, dr21.index, dr21[measure]))

            # find intersection of RH5 and RH5+R21 lines
            drh5 = sim.query('treatment == "RH5" and value == @v').set_index('λ')
            drh5r21 = sim.query('treatment == "R21+RH5" and value == @v').set_index('λ')
            y = drh5[measure] - drh5r21[measure]
            root = get_root(y.index, y)
            if not root.success:
                raise UserWarning(f'root not found: {measure}, RH5 and RH5+R21, {v}')
            xi.append(root.x)
            yi.append(np.interp(root.x, drh5.index, drh5[measure]))


        if max(xi)-min(xi) < 0.1 and max(yi)-min(yi) < 2:
            ax.scatter(xi, yi, color='k', zorder=99)
        else:
            l = reyburn.colored_line(xi, yi, values, ax, linewidth=6, cmap=new_cmap)
        if max(xr21)-min(xr21) < 0.1 and max(yr21)-min(yr21) < 2:
            ax.scatter(xr21, yr21, color='k', zorder=99)
        else:
            l = reyburn.colored_line(xr21, yr21, values, ax, linewidth=6, cmap=new_cmap)

    if l is not None:
        cb = fig.colorbar(l, ax=axs[1])
        if default_value is not None:
            cb.ax.axhline(default_value, color='w', lw=2)

    axs[0].set_ylabel('Lifetime death averted per 100,000')
    axs[1].set_ylabel('Lifetime death efficacy (%)')

    for ax in axs:
        ax.set_xlabel('Blood infections per person per year')
        ax.axhline(0, ls='--', color='DarkGrey', zorder=0)
        ax.set_xticks(range(0, 8, 1))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(visible=True, which='both', axis='both', color='lightgrey', linestyle='--', linewidth=0.5)

    return fig, axs

def sensitivity(parameter, values, β=0, season_width=1, boosters=1, protection_blood=False, code='C'):
    model = Model()
    model.pars = Parameters(
        lsv=True,
        β=β,
        study_months=12*20,
        season_width=season_width,
        nboosters_liver=boosters,
        nboosters_blood=boosters,
        smc_repeats=boosters,
        protection_blood=protection_blood,
        vac_age_range=vac_age_range,
        recording=min_vac_age_liver + vac_age_range # start recording at the same time as R21 for proper comparison of vaccines

    )
    model.variables = Variables(['Direct deaths'])
    model.measures = Measures(['efficacy', 'averted'])
    model.Config(code=code)

    λs = np.logspace(np.log10(0.5), np.log10(5), 19)
    treatments = {(True, False):'R21', (False, True):'RH5', (True, True):'R21+RH5'}
    items = [[model, λ, lsv, bsv, parameter, v] for λ, (lsv, bsv), v in product(λs, treatments.keys(), values)]

    results = {'λ':[], 'treatment':[], 'value':[], 'case':[], 'averted':[], 'efficacy':[]}
    with Pool(12) as p:
        rs = p.starmap(sim, items)

    for predictions, i in zip(rs, items):
        results['λ'] += [i[1]]
        results['treatment'].append(treatments[(i[2], i[3])])
        results['value'] += [i[5]]
        results['case'] += ['Direct deaths']
        results['efficacy'] += [ predictions['Vaccine Direct deaths efficacy'].iloc[-1] ]
        results['averted'] += [ predictions['Vaccine Direct deaths averted'].iloc[-1] ]

    r = pd.DataFrame(results)
    return r

def death_rate_modifier():
    from Scripts import reyburn

    gam = reyburn.fit_GAM()
    XX, fit, death_rates, cis = reyburn.death_rate_mean_CI(gam)
    fig, ax = reyburn.plot_GAM_mean_CI_data(XX, fit, death_rates, cis)
    ax.annotate('death rate modifier = 1', xy=(11, 0.15), color='C0')

    pp = Parameters(study_months=20*12, death_rate_modifier=0.55)
    duration = int(weeks_per_month*pp.study_months) + pp.min_vac_age + pp.vac_age_range
    ages = np.arange(duration, dtype=float) # ages in weeks

    ax.plot(ages/52, pp.direct_deaths[:duration])
    ax.annotate('death rate modifier = 0.55', xy=(10, 0.115), color='C1')

    pp = Parameters(study_months=20*12, death_rate_modifier=0)
    duration = int(weeks_per_month*pp.study_months) + pp.min_vac_age + pp.vac_age_range
    ages = np.arange(duration, dtype=float) # ages in weeks

    ax.plot(ages/52, pp.direct_deaths[:duration])
    ax.annotate('death rate modifier = 0', xy=(11, 0.07), color='C2')

    return fig, ax