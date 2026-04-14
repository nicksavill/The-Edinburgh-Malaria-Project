import sys
sys.path.append('..')
from Edinburgh_Model.model import *
from Edinburgh_Model.visualisation_utils import *
from Edinburgh_Model.simulate_single_vaccine import *

from . import infection_rate

import pickle
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
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
    To achieve maximum RH5 efficacy, the last primary dose must be given 4 weeks before the peak of the malaria season (season_peak=4).
    With RTS,S and R21, SMC is given 4 weeks before the last primary dose for maximum clinical efficacy (smc_offset=-4).
    With RH5, SMC is given 4 weeks before the last primary dose for maximum clinical efficacy (smc_offset=-4).
    (max clinical efficacy and max severe efficacy do not coincide for the same value of smc_offset, max efficacy is insensitive for +/- a week or two)

"""
season_peak_liver = 6
season_peak_blood = 4
smc_offset_liver = -3
smc_offset_blood = -4
min_vac_age_liver = int(round(7*weeks_per_month, 0))
min_vac_age_blood = int(round(10*weeks_per_month, 0))
vac_age_range = 52

pars = Parameters()

######################################## MODEL SCHEMATIC PLOTS #################################################

def severe_malaria_risk_plot(fontsize=12):
    n = np.arange(1, 15, 1) # range of number of infections
    # risk of severe malaria on nth infection at age a
    y = lambda a, n: pars.ρ_severe * np.exp(-(pars.δ_severe + pars.ε_severe*a)*(n-1))

    fig, ax = plt.subplots()

    ax.plot(n, y(1, n), marker='o', lw=0.2, label=1)
    ax.plot(n, y(5, n), marker='s', lw=0.2, label=5)
    ax.plot(n, y(10, n), marker='^', lw=0.2, label=10)

    ax.set_xticks(n)
    ax.set_ylabel('Risk of severe malaria', fontsize=fontsize)
    ax.set_xlabel('Number of infections', fontsize=fontsize)
    ax.legend(title='Age (years)', title_fontsize=0.75*fontsize, fontsize=0.75*fontsize, loc='upper right')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return fig, ax

def clinical_malaria_risk_plot(fontsize=12):
    n = np.arange(1, 200, 10) # range of number of infections
    # risk of clinical malaria on nth infection
    y = lambda n:  pars.ρ_clinical * (exp(pars.δ_clinical*pars.γ_clinical) - 1) / (exp(pars.δ_clinical*pars.γ_clinical) - 2 + exp(pars.δ_clinical*(n-1)))

    fig, ax = plt.subplots()

    ax.plot(n, y(n), marker='o', lw=0.2)

    ax.set_ylabel('Risk of clinical malaria', fontsize=fontsize)
    ax.set_xlabel('Number of infections', fontsize=fontsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(visible=True, which='both', axis='both', color='lightgrey', linestyle='--', linewidth=0.5)

    return fig, ax

def early_life_protection_plot(fontsize=12):
    a = np.linspace(0, 6, 100) # range of age in years
    # relative risk of blood infection at age a
    l = lambda a: (1 - pars.ρ_elp * np.exp(-np.log(2) / pars.τ_elp * a))

    fig, ax = plt.subplots()

    ax.plot(a, l(a))
    ax.scatter(0, 0, color='w') # force origin to be plotted

    ax.set_xlabel('Age (years)', fontsize=fontsize)
    ax.set_ylabel('Relative risk of infection', fontsize=fontsize)
    ax.text(4, 0.6, f'Relative risk doubles\nevery {52*pars.τ_elp:.0f} weeks', fontsize=fontsize, ha='center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return fig, ax


######################################## TIME SERIES PLOTS #################################################

def plot(model, cohorts, sim_source=None, legend_title=None, legend_pos='last'):
    if sim_source is None:
        sim_source = model.simulation_fn(model)

    if model.show_vac or model.show_smc or model.show_season or model.show_cfr:
        for c in [str(c) for c in cohorts]:
            if 'R21' in c or 'RTS,S' in c:
                model.pars.lsv = True
            if 'RH5' in c:
                model.pars.bsv = True
            if 'SMC' in c:
                model.pars.smc = True
        panel_source = {f:data for f, data in model.panels_fn(model).items()}
        ncols = len(model.variables.display_vars) + 1
    else:
        panel_source = None
        ncols = len(model.variables.display_vars)

    pars = model.pars
    display_vars = model.variables.display_vars
    display_measures = model.measures.display_measures

    colours = [pink, purple, blue, turquoise, 'C1']
    color  = {i:j for i, j in zip(cohorts, colours)}

    title_dict = {k:v for k, v in zip(vars, titles)}

    nrows = len(display_measures)
    fig, axs = plt.subplots(nrows, ncols, figsize=(3*ncols*model.fig_xscale, 3*nrows*model.fig_yscale), squeeze=False)
    fig.subplots_adjust(wspace=0.25)

    for j, variable in enumerate(display_vars):
        for i, measure in enumerate(display_measures):
            ax = axs[i, j]
            f = f'{variable} {measure}'
            if measure != 'efficacy':
                ax.yaxis.set_major_formatter(mt.ScalarFormatter(useMathText=True))
                ax.ticklabel_format(style="sci", axis="y", scilimits=(0,2))

            # Turn off right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.grid(visible=True, which='both', axis='both', color='lightgrey', linestyle='--', linewidth=0.5)
            ax.set_xticks(range(0, 26, 5))

            for c in cohorts:
                ax.scatter(x=0, y=0, color='white') # forces origin to be plotted
                if not ((measure == 'efficacy' or measure == 'averted') and c == pars.control):
                    try:
                        # plot control and treatment for episodes and cdf but only treatment for efficacy and averted
                        ax.plot(sim_source['ages'], sim_source[f'{c} {f}'], color=color[c], label=c)
                        if variable == 'Direct deaths' and min(sim_source[f'{c} {f}']) < 0 and (measure == 'averted' or measure == 'efficacy'):
                            ax.axhline(0, color='black', ls=':')
                    except:
                        pass

            # put legend only in one axis
            if i == 0:
                if legend_pos == 'last' and j == len(display_vars)-1:
                    ax.legend(loc='upper right', title=legend_title, fontsize=8, title_fontsize=8)
                elif legend_pos == 'first' and j == 0:
                    ax.legend(loc='lower right', title=legend_title, fontsize=8, title_fontsize=8)

            if i == 0:
                # title only in row 1
                ax.set_title(title_dict[variable], loc='right', fontsize=14)
            elif i == len(display_measures)-1:
                # x-axis label only in last row
                if pars.lsv or pars.bsv:
                    ax.set_xlabel(f'{model.time_scale} post-vaccination')
                else:
                    ax.set_xlabel(f'{model.time_scale}')
            if i < len(display_measures)-1:
                ax.set_xticklabels([])

            if j == 0:
                n = pars.popsize
                if pars.children == 'population':
                    n *= pars.study_months * weeks_per_month

                # y-axis label only in column 1
                if measure == 'cases':
                    if model.time_scale == 'Years':
                        ax.set_ylabel(f'Annual cases\nper {n:,.0f}')
                    elif model.time_scale == 'Months':
                        ax.set_ylabel(f'Monthly cases\nper {n:,.0f}')
                    elif model.time_scale == 'Weeks':
                        ax.set_ylabel(f'Weekly cases\nper {n:,.0f}')
                elif measure == 'cdf':
                    ax.set_ylabel(f'Cumulative cases\nper {n:,.0f}')
                elif measure == 'efficacy':
                    ax.set_ylabel('Efficacy (%)')
                elif measure == 'averted':
                    ax.set_ylabel(f'Cases averted\nper {n:,.0f}')
                elif measure == 'Kaplan-Meier':
                    ax.set_ylabel(f'Proportion of\nchildren surviving')

    ######################### DATA ###########################
    if model.data:
        season = model.data.seasonality
        case_def = model.data.case_def
        if model.time_scale == 'Years':
            scale_time = 1/12
        elif model.time_scale == 'Months':
            scale_time = 1
        elif model.time_scale == 'Weeks':
            scale_time = weeks_per_month

        for j, (variable, title) in enumerate(zip(display_vars, titles)):
            for i, measure in enumerate(display_measures):
                ax = axs[i, j]
                source = model.data.dataframe.query(f'variable == @variable and measure == @measure and seasonality == @season and case_def == @case_def')
                if len(source) > 0:
                    for dx, c in enumerate(cohorts):
                        data = source.query(f'group == @c')
                        if measure == 'efficacy':
                            # efficacy is plotted as a scatter plot with upper and lower bounds

                            ax.errorbar(data['month'] * scale_time, 100*data['value'], yerr=[100*(data['value']-data['lower']), 100*(data['upper']-data['value'])], color=color[c], ls='', marker='o', markerfacecolor='w', capsize=5)
                        elif measure == 'cases' or measure == 'cdf':
                            # cases is plotted as a line with points
                            # ax.plot(data['month'], data['value'], color=color[c], ls=':', marker='o', markerfacecolor='w', label=c)
                            if len(data) > 10:
                                ax.scatter(data['month'] * scale_time, data['value'], color=color[c], marker='.', label=c, alpha=0.25)
                            else:
                                ax.scatter(data['month'] * scale_time, data['value'], color=color[c], marker='o', facecolor='w', label=c)

    ######################### PANELS ###########################
    if panel_source is not None:
        row = 0
        if model.show_vac or model.show_season:
            ax = axs[row, 4]
            ax.set_xlabel(model.time_scale)
            ax.set_ylabel(f'{pars.treatment} protection')
            if model.show_vac:
                if pars.lsv:
                    ax.plot(panel_source['liver_vac_prot']['x'], panel_source['liver_vac_prot']['y'], color='C2', label='R21')
                if pars.bsv:
                    ax.plot(panel_source['blood_vac_prot']['x'], panel_source['blood_vac_prot']['y'], color='C3', label='RH5')
                # if pars.smc:
                #     ax.plot(panel_source['smc']['x'], panel_source['smc']['y'], color='C0', label='SMC')

            if model.show_season:
                ax.plot(panel_source['season']['x'], panel_source['season']['y'], color='C1', label='Season')
            ax.legend()
            # Turn off right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            # Make minor tick marks visible on x-axis
            ax.xaxis.set_minor_locator(mt.AutoMinorLocator())
            ax.tick_params(axis='x', which='minor', length=4, color='grey')
            row += 1

        if model.show_cfr:
            ax = axs[row, 4]
            ax.set_xlabel('Age (years)')
            ax.set_ylabel('Risk of death')
            ax.scatter(x=0, y=0, color='white') # forces origin to
            ax.plot(panel_source['cfr']['x'], panel_source['cfr']['SM'], color='Red')
            max_x = 10
            # for c in cohorts:
            #     max_x = max(max_x, sim_source[f'{c} SM mean age x'].max())
            #     ax.plot(sim_source[f'{c} SM mean age x'], sim_source[f'{c} SM mean age y'], color=color[c], ls=line[c], label=c)

            # ax.legend(title='Infections per year', loc='upper left', fontsize=8, title_fontsize=8)
            ax.set_title('Risk of death by age\nand at mean age of SM', loc='left', fontsize=14)
            ax.set_xlim(-1, max_x*1.1)
            # Turn off right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            # Make minor tick marks visible on x-axis
            ax.xaxis.set_minor_locator(mt.AutoMinorLocator())
            ax.tick_params(axis='x', which='minor', length=4, color='grey')
            row += 1

        for i in range(row, nrows):
            axs[i, 4].set_visible(False)  # hide empty panel
    return fig, axs

def unvaccinated(λs=(10, 5, 1, 0.5), popsize=100000, β=0, season_width=1, ε_severe=0.05, years=20, vars=['All infection', 'All clinical'], fig_xscale=1, fig_yscale=1, plot_figure=True, legend_pos='last') -> 'tuple':
    model = Model()
    model.pars = Parameters(
        study_months=years*12,
        popsize=popsize,
        season_width=season_width,
        season_peak=26,
        β=β,
        ε_severe=ε_severe,
    )
    model.variables = Variables(vars + ['Severe malaria', 'Direct deaths'])
    model.measures = Measures(['cases', 'cdf'])
    config = {'fig_yscale':fig_yscale, 'fig_xscale':fig_xscale}
    model.Config(**config)
    model.Sources()

    rs = []
    for λ in λs:
        model.pars.λ = λ
        model.pars.control = λ
        model.pars.update_pars(('control', 'λ'))
        rs.append(simulate_single_vaccine(model).set_index('ages'))

    df_joined = pd.concat(rs, axis=1).reset_index(drop=False)
    if plot_figure:
        return plot(model, λs, sim_source=df_joined, legend_title='Infections per year', legend_pos=legend_pos)
    else:
        return df_joined

def RTSS_validate_ts(λ=1, β=0, years=20, fig_xscale=1, fig_yscale=1, legend_pos='last') -> 'tuple':
    """ Children begin vaccination at 5-17 months, so last primary dose is at 7-19 months"""
    model = Model()
    model.pars = Parameters(
        lsv=True,
        λ=λ,
        β=β,
        # primary case definition of clinical malaria is fever and > 5000 parasites/uL.
        # Using estimates from Goncalves, 43% of blood infections result in fever and >5000 parasites/uL,
        # Assuming 80% risk of fever, then prob of >5000 parasites/uL given fever is 0.43/0.8=0.54
        case_def_clinical=0.54,
        study_months=years*12,
        min_vac_age=min_vac_age_liver,
        vac_age_range=int(round(12*weeks_per_month, 0)),
        vac_profile_liver='Imperial RTS,S'
    )
    model.variables = Variables(['All clinical', 'Severe malaria'])
    model.measures = Measures(['cases', 'cdf', 'averted', 'efficacy'])
    model.data = Data(datafile='Data/rtss_efficacy_raw.csv')

    config = {'fig_yscale':fig_yscale, 'fig_xscale':fig_xscale}
    model.Config(**config)
    model.Sources(data_fn=fig_data)

    rs, treatments = [], ['control']
    model.pars.treatment = 'no booster'
    model.pars.nboosters_liver = 0
    model.pars.update_pars(('treatment', 'nboosters_liver'))
    treatments.append(model.pars.treatment)
    rs.append(simulate_single_vaccine(model).set_index('ages'))

    model.pars.treatment = 'one booster'
    model.pars.nboosters_liver = 1
    model.pars.update_pars(('treatment', 'nboosters_liver'))
    treatments.append(model.pars.treatment)
    rs.append(simulate_single_vaccine(model).set_index('ages'))

    # Remove columns with "control" in the column name from rs[0]
    for i in range(len(rs)-1):
        rs[i] = rs[i].loc[:, ~rs[i].columns.str.contains("control")]

    df_joined = pd.concat(rs, axis=1).reset_index(drop=False)
    return plot(model, treatments, sim_source=df_joined, legend_pos=legend_pos)

def R21_validate_ts(fig_xscale=1, fig_yscale=1) -> 'tuple':
    model = Model()
    model.pars = Parameters(
        study_months=22,
        popsize=100,
        λ=7.,
        # primary case definition of clinical malaria is fever and > 5000 parasites/uL.
        # Using estimates from Goncalves, 43% of blood infections result in fever and >5000 parasites/uL,
        # Assuming 80% risk of fever, then prob of >5000 parasites/uL given fever is 0.43/0.8=0.54
        case_def_clinical=0.54,
        # season length and relative timing of last primary dose to peak of season is estimated from
        # Kaplan-Meier estimates of the time to first episode of clinical malaria (Datoo 2024, fig2)
        season_peak=11,
        season_width=0.143,
        lsv=True,
        vac_profile_liver='Imperial R21',
        nboosters_liver=1,
        min_vac_age=int(round(5*weeks_per_month, 0)),
        vac_age_range=int(round(36*weeks_per_month, 0)) - int(round(5*weeks_per_month, 0)),
        smc=True,
        smc_control=True, # the control arm uses SMC
        smc_offset=0,
        smc_rounds=7,
        # SMC coverage 54% = (1×13.6+2×11.3+3×12.2+4×36.1)/(4×100) (Table 1 from Datoo 2024)
        smc_coverage=0.54,
        recording='Last primary dose'
    )

    model.variables = Variables(['All clinical'])
    model.measures = Measures(['cases', 'cdf', 'averted', 'efficacy'])
    model.data = Data(case_def='primary', seasonality='seasonal', datafile='Data/R21.csv')
    config = {'fig_yscale':fig_yscale, 'fig_xscale':fig_xscale, 'time_scale':'Weeks'}
    model.Config(**config)
    model.Sources(data_fn=fig_data)

    rs = []
    model.pars.treatment = 'R21'
    rs.append(simulate_single_vaccine(model).set_index('ages'))

    df_joined = pd.concat(rs, axis=1).reset_index(drop=False)
    return plot(model, ('control', 'R21'), sim_source=df_joined)

def R21_or_R21_and_SMC_ts(λ=1, β=0, years=20, season_width=1, fig_xscale=1, fig_yscale=1, legend_pos='last') -> 'tuple':
    model = Model()
    model.pars = Parameters(
        lsv=True,
        smc=False,
        λ=λ,
        β=β,
        study_months=years*12,
        season_peak=season_peak_liver,
        season_width=season_width,
        min_vac_age=min_vac_age_liver,
        vac_age_range=vac_age_range,
        smc_offset=smc_offset_liver,
        recording=min_vac_age_liver + vac_age_range
    )
    model.variables = Variables(['All infection', 'All clinical', 'Severe malaria', 'Direct deaths'])
    model.measures = Measures(['cases', 'cdf', 'averted', 'efficacy'])
    config = {'fig_yscale':fig_yscale, 'fig_xscale':fig_xscale}
    model.Config(**config)
    model.Sources()

    rs, treatments = [], ['control']
    model.pars.treatment = 'R21 (2 yr)'
    model.pars.nboosters_liver = 1
    model.pars.update_pars(('treatment', 'nboosters_liver'))
    treatments.append(model.pars.treatment)
    rs.append(simulate_single_vaccine(model).set_index('ages'))

    model.pars.treatment = 'R21+SMC (2 yr)'
    model.pars.smc = True
    model.pars.nboosters_liver = 1
    model.pars.smc_repeats = 1
    model.pars.update_pars(('treatment', 'smc', 'smc_repeats', 'nboosters_liver'))
    treatments.append(model.pars.treatment)
    rs.append(simulate_single_vaccine(model).set_index('ages'))

    model.pars.treatment = 'R21+SMC (5 yr)'
    model.pars.smc = True
    model.pars.nboosters_liver = 4
    model.pars.smc_repeats = 4
    model.pars.update_pars(('treatment', 'smc', 'smc_repeats', 'nboosters_liver'))
    treatments.append(model.pars.treatment)
    rs.append(simulate_single_vaccine(model).set_index('ages'))

    # Remove columns with "control" in the column name from rs[0]
    for i in range(len(rs)-1):
        rs[i] = rs[i].loc[:, ~rs[i].columns.str.contains("control")]

    df_joined = pd.concat(rs, axis=1).reset_index(drop=False)
    return plot(model, treatments, sim_source=df_joined, legend_pos=legend_pos)

def RTSS_or_R21_ts(λ=1, β=0, years=20, season_width=1, fig_xscale=1, fig_yscale=1) -> 'tuple':
    model = Model()
    model.pars = Parameters(
        lsv=True,
        smc=False,
        λ=λ,
        β=β,
        study_months=years*12,
        nboosters_liver=1,
        season_peak=season_peak_liver,
        season_width=season_width,
        min_vac_age=min_vac_age_liver,
        vac_age_range=vac_age_range,
        smc_offset=smc_offset_liver,
        recording=min_vac_age_liver + vac_age_range
    )
    model.variables = Variables(['All infection', 'All clinical', 'Severe malaria', 'Direct deaths'])
    model.measures = Measures(['cases', 'cdf', 'averted', 'efficacy'])
    config = {'fig_yscale':fig_yscale, 'fig_xscale':fig_xscale}
    model.Config(**config)
    model.Sources()

    rs, treatments = [], ['control']
    model.pars.treatment = 'R21 (2 yr)'
    model.pars.vac_profile_liver = 'Imperial R21'
    model.pars.update_pars(('treatment', 'vac_profile_liver'))
    treatments.append(model.pars.treatment)
    rs.append(simulate_single_vaccine(model).set_index('ages'))

    model.pars.treatment = 'RTS,S (2 yr)'
    model.pars.vac_profile_liver = 'Imperial RTS,S'
    model.pars.update_pars(('treatment', 'vac_profile_liver'))
    treatments.append(model.pars.treatment)
    rs.append(simulate_single_vaccine(model).set_index('ages'))

    # Remove columns with "control" in the column name from rs[0]
    for i in range(len(rs)-1):
        rs[i] = rs[i].loc[:, ~rs[i].columns.str.contains("control")]

    df_joined = pd.concat(rs, axis=1).reset_index(drop=False)
    return plot(model, treatments, sim_source=df_joined)

def RH5_ts(λ=1, β=0, years=20, season_width=1, protection_blood=False, fig_xscale=1, fig_yscale=1, legend_pos='last') -> 'tuple':
    model = Model()
    model.pars = Parameters(
        bsv=True,
        λ=λ,
        β=β,
        study_months=years*12,
        season_peak=season_peak_blood,
        season_width=season_width,
        min_vac_age=min_vac_age_blood,
        vac_age_range=vac_age_range,
        protection_blood=protection_blood,
        recording=min_vac_age_liver + vac_age_range # start recording at the same time as R21 for proper comparison of vaccines
    )
    model.variables = Variables(['All clinical', 'Severe malaria', 'Direct deaths'])
    model.measures = Measures(['cases', 'cdf', 'averted', 'efficacy'])
    config = {'fig_yscale':fig_yscale, 'fig_xscale':fig_xscale}
    model.Config(**config)
    model.Sources()

    rs, treatments = [], ['control']
    model.pars.treatment = 'RH5 (2 yr)'
    model.pars.nboosters_blood = 1
    model.pars.update_pars(('treatment', 'nboosters_blood'))
    treatments.append(model.pars.treatment)
    rs.append(simulate_single_vaccine(model).set_index('ages'))

    model.pars.treatment = 'RH5 (2 yr), x2 half-life'
    model.pars.τ_blood = 2
    model.pars.nboosters_blood = 1
    model.pars.update_pars(('treatment', 'τ_blood', 'nboosters_blood'))
    treatments.append(model.pars.treatment)
    rs.append(simulate_single_vaccine(model).set_index('ages'))

    model.pars.treatment = 'RH5 (5 yr), x2 half-life'
    model.pars.τ_blood = 2
    model.pars.nboosters_blood = 4
    model.pars.update_pars(('treatment', 'τ_blood', 'nboosters_blood'))
    treatments.append(model.pars.treatment)
    rs.append(simulate_single_vaccine(model).set_index('ages'))

    # Remove columns with "control" in the column name from rs[0]
    for i in range(len(rs)-1):
        rs[i] = rs[i].loc[:, ~rs[i].columns.str.contains("control")]

    df_joined = pd.concat(rs, axis=1).reset_index(drop=False)
    return plot(model, treatments, sim_source=df_joined, legend_pos=legend_pos)

def RH5_arm_ts(λ=1, β=0, years=20, season_width=1, fig_xscale=1, fig_yscale=1, legend_pos='last') -> 'tuple':
    model = Model()
    model.pars = Parameters(
        bsv=True,
        λ=λ,
        β=β,
        study_months=years*12,
        season_peak=season_peak_blood,
        season_width=season_width,
        vac_age_range=vac_age_range,
        nboosters_blood=1,
        recording=min_vac_age_liver + vac_age_range # start recording at the same time as R21 for proper comparison of vaccines
    )
    model.variables = Variables(['All clinical', 'Severe malaria', 'Direct deaths'])
    model.measures = Measures(['cases', 'cdf', 'averted', 'efficacy'])
    config = {'fig_yscale':fig_yscale, 'fig_xscale':fig_xscale, 'logfile':True}
    model.Config(**config)
    model.Sources()

    rs, treatments = [], ['control']
    model.pars.treatment = 'RH5 delayed, no infection'
    model.pars.protection_blood = False
    model.pars.vac_profile_blood = 'RH5 delayed'
    model.pars.min_vac_age = int(round(10*weeks_per_month, 0))
    model.pars.update_pars(('treatment', 'protection_blood', 'vac_profile_blood', 'min_vac_age'))
    treatments.append(model.pars.treatment)
    rs.append(simulate_single_vaccine(model).set_index('ages'))

    model.pars.treatment = 'RH5 delayed, infection'
    model.pars.protection_blood = True
    model.pars.vac_profile_blood = 'RH5 delayed'
    model.pars.min_vac_age = int(round(10*weeks_per_month, 0))
    model.pars.update_pars(('treatment', 'protection_blood', 'vac_profile_blood', 'min_vac_age'))
    treatments.append(model.pars.treatment)
    rs.append(simulate_single_vaccine(model).set_index('ages'))

    model.pars.treatment = 'RH5 monthly, no infection'
    model.pars.protection_blood = None
    model.pars.vac_profile_blood = 'RH5 monthly'
    model.pars.min_vac_age = int(round(7*weeks_per_month, 0))
    model.pars.ω_infection = 0
    model.pars.ω_clinical = 0.1
    model.pars.ω_severe = 0.33
    model.pars.update_pars(('treatment', 'protection_blood', 'vac_profile_blood', 'min_vac_age'))
    model.pars.update_pars(('ω_infection', 'ω_clinical', 'ω_severe'))
    treatments.append(model.pars.treatment)
    rs.append(simulate_single_vaccine(model).set_index('ages'))

    # Remove columns with "control" in the column name from rs[0]
    for i in range(len(rs)-1):
        rs[i] = rs[i].loc[:, ~rs[i].columns.str.contains("control")]

    df_joined = pd.concat(rs, axis=1).reset_index(drop=False)
    return plot(model, treatments, sim_source=df_joined, legend_pos=legend_pos)

def RH5_R21_ts(λ=1, β=0, years=20, season_width=1, boosters=4, fig_xscale=1, fig_yscale=1, legend_pos='last') -> 'tuple':
    model = Model()
    model.pars = Parameters(
        lsv=True,
        smc=False,
        λ=λ,
        β=β,
        study_months=years*12,
        season_width=season_width,
        nboosters_liver=boosters,
        nboosters_blood=boosters,
        vac_age_range=vac_age_range,
        recording=min_vac_age_liver + vac_age_range # start recording at the same time as R21 for proper comparison of vaccines
    )
    model.variables = Variables(['All clinical', 'Severe malaria', 'Direct deaths'])
    model.measures = Measures(['cases', 'cdf', 'averted', 'efficacy'])
    config = {'fig_yscale':fig_yscale, 'fig_xscale':fig_xscale}
    model.Config(**config)
    model.Sources()

    rs, treatments = [], ['control']
    model.pars.treatment = f'R21 ({boosters+1} yr)'
    model.pars.lsv = True
    model.pars.bsv = False
    model.pars.season_peak = season_peak_liver
    model.pars.min_vac_age = min_vac_age_liver
    model.pars.update_pars(('treatment', 'lsv', 'bsv', 'season_peak', 'min_vac_age'))
    treatments.append(model.pars.treatment)
    rs.append(simulate_single_vaccine(model).set_index('ages'))

    model.pars.treatment = f'RH5 ({boosters+1} yr)'
    model.pars.lsv = False
    model.pars.bsv = True
    model.pars.season_peak = season_peak_blood
    model.pars.min_vac_age = min_vac_age_blood
    model.pars.update_pars(('treatment', 'lsv', 'bsv', 'season_peak', 'min_vac_age'))
    treatments.append(model.pars.treatment)
    rs.append(simulate_single_vaccine(model).set_index('ages'))

    model.pars.treatment = f'RH5+R21 ({boosters+1} yr)'
    model.pars.lsv = True
    model.pars.bsv = True
    model.pars.season_peak = min_vac_age_blood - min_vac_age_liver + season_peak_blood # time peak season relative to last primary dose of RH5 which occurs 3 months after last primary dose of R21
    model.pars.min_vac_age = min_vac_age_liver
    model.pars.offset_blood = min_vac_age_blood - min_vac_age_liver # difference of 3 months between last R21 dose and last RH5 dose
    model.pars.update_pars(('treatment', 'lsv', 'bsv', 'season_peak', 'min_vac_age', 'offset_blood'))
    treatments.append(model.pars.treatment)
    rs.append(simulate_single_vaccine(model).set_index('ages'))

    # Remove columns with "control" in the column name from rs[0]
    for i in range(len(rs)-1):
        rs[i] = rs[i].loc[:, ~rs[i].columns.str.contains("control")]

    df_joined = pd.concat(rs, axis=1).reset_index(drop=False)
    return plot(model, treatments,  sim_source=df_joined, legend_pos=legend_pos)



################################ TRANSMISSION INTENSITY PLOTS #############################################

def sim(model, λ, lsv, bsv, blood_boosters=None, study_months=90*12):
    model.pars.λ = λ
    model.pars.lsv = lsv
    model.pars.bsv = bsv
    model.pars.study_months = study_months

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

    if blood_boosters is not None:
        model.pars.nboosters_blood = blood_boosters

    model.pars.update_pars(('λ', 'lsv', 'bsv', 'study_months', 'season_peak', 'min_vac_age', 'offset_blood', 'nboosters_blood'))
    return simulate_single_vaccine(model).set_index('ages')

def lifetime_efficacies_plot(all_sims, fig_xscale, fig_yscale, surveyed=True):
    """ Plot predictions """

    ylabels = 'Clinical malaria', 'Severe malaria', 'Death'
    titles = 'Lifetime cases averted per 100,000', 'Lifetime efficacy (%)'

    ncols, nrows = 2, 3+int(surveyed)
    fig, axs = plt.subplots(nrows, ncols, figsize=(3*ncols*fig_xscale, 3*nrows*fig_yscale), sharex=True, squeeze=False)
    fig.subplots_adjust(hspace=0.1)

    for row, var in enumerate(all_sims['case'].unique()):
        data = all_sims.query('case == @var')
        for col, (variable, title) in enumerate(zip(('averted', 'efficacy'), titles)):
            ax = axs[row, col]

            for n, c, s in zip(data['treatment'].unique(), [purple, blue, turquoise], ('-', '-', '-')):
                d = data.query('treatment == @n')
                ax.plot('λ', variable, ls=s, color=c, data=d)
                ax.scatter(x=0, y=0, color='white') # forces origin to be plotted
                ax.text(d['λ'].iloc[-1]+0.2, d[variable].iloc[-1], n, color=c, fontsize=10, ha='left', va='center')
                ax.set_xlim(ax.get_xlim()[0], 14)

            if row == 2:
                ax.axhline(0, ls='--', color='DarkGrey')
                ax.set_xlabel('Blood infections per year')
            elif row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(ylabels[row])

    if surveyed:
        dist = pd.read_csv('Data/infection_rate_distribution.csv')
        for col in range(2):
            ax = axs[3, col]
            ax.plot('inf_rate', 'percentage', color='k', data=dist)
            ax.fill_between(dist['inf_rate'], 0, 100*dist['density'] / dist['density'].max(), color='LightGrey', alpha=0.5)
            ax.set_xlabel('Blood infections per year')
            ax.set_xlim(*axs[2, 0].get_xlim())
            if col == 0:
                ax.set_ylabel('Percentage of surveyed\nsites since 2020')

    for ax in axs.flatten():
        ax.set_xticks(range(0, 13, 2))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(visible=True, which='both', axis='both', color='lightgrey', linestyle='--', linewidth=0.5)

    return fig, axs

def lifetime_efficacies(β=0,
        season_width=1,
        boosters=0,
        years=90,
        ν_liver=1,
        τ_blood=1,
        smc=False,
        protection_blood=False,
        vac_profile_blood='RH5 delayed',
        ω_infection=0,
        ω_clinical=0,
        ω_severe=0,
        logfile=False):

    model = Model()
    model.pars = Parameters(
        lsv=True,
        β=β,
        smc=smc,
        study_months=years*12,
        season_width=season_width,
        nboosters_liver=boosters,
        nboosters_blood=boosters,
        smc_repeats=boosters,
        τ_blood=τ_blood,
        vac_profile_blood=vac_profile_blood,
        protection_blood=protection_blood,
        ω_infection=ω_infection,
        ω_clinical=ω_clinical,
        ω_severe=ω_severe,
        ν_liver=ν_liver,
        vac_age_range=vac_age_range,
        recording=min_vac_age_liver + vac_age_range # start recording at the same time as R21 for proper comparison of vaccines
    )
    model.variables = Variables(['All clinical', 'Severe malaria', 'Direct deaths'])
    model.measures = Measures(['efficacy', 'averted'])
    model.Config(logfile=logfile)

    λs = np.logspace(np.log10(0.5), np.log10(10), 20)
    treatments = {(True, False):'R21', (False, True):'RH5', (True, True):'R21+RH5'}
    items = [[model, λ, lsv, bsv, None, years*12] for λ, (lsv, bsv) in product(λs, treatments.keys())]

    with Pool(12) as p:
        rs = p.starmap(sim, items)

    results = {'λ':[], 'treatment':[], 'case':[], 'efficacy':[], 'averted':[]}
    for predictions, i in zip(rs, items):
        results['λ'] += [i[1]] * 3
        results['treatment'] += [treatments[(i[2], i[3])]] * 3
        results['case'] += ['All clinical', 'Severe malaria', 'Direct deaths']
        results['efficacy'] += [
            predictions['Vaccine All clinical efficacy'].iloc[-1],
            predictions['Vaccine Severe malaria efficacy'].iloc[-1],
            predictions['Vaccine Direct deaths efficacy'].iloc[-1]
        ]
        results['averted'] += [
            predictions['Vaccine All clinical averted'].iloc[-1],
            predictions['Vaccine Severe malaria averted'].iloc[-1],
            predictions['Vaccine Direct deaths averted'].iloc[-1]
        ]
    r = pd.DataFrame(results)
    return r


################################ R21 RESCUE PLOTS #############################################

def rescue_sim(model, λ, lsv, bsv, blood_boosters=None):
    model.pars.λ = λ
    model.pars.lsv = lsv
    model.pars.bsv = bsv

    if blood_boosters is not None:
        model.pars.nboosters_blood = blood_boosters

    model.pars.update_pars(('λ', 'lsv', 'bsv', 'nboosters_blood'))

    return simulate_single_vaccine(model).set_index('ages')

def R21_rescue_plot(sims, fig_xscale, fig_yscale, surveyed=False):
    """ Plot predictions """

    all_sims = pd.concat(sims)
    n = len(all_sims['vaccine'].unique())

    ncols, nrows = n+int(surveyed), 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(3*ncols*fig_xscale, 3*nrows*fig_yscale))
    fig.subplots_adjust(wspace=0.3)

    for col, vac in enumerate(all_sims['vaccine'].unique()):
        ax = axs[col+int(surveyed)]

        data = all_sims.query('vaccine == @vac')
        for n, c in zip(data['treatment'].unique(), [purple, blue, turquoise]):
            d = data.query('treatment == @n')
            ax.plot('λ', 'efficacy', color=c, data=d)
            ax.scatter(x=0, y=0, color='white') # forces origin to be plotted
            ax.text(d['λ'].iloc[-1]+0.2, d['efficacy'].iloc[-1], n, color=c, fontsize=10, ha='left', va='center')
            ax.set_xlim(ax.get_xlim()[0], 6.5)

        ax.axhline(0, ls='--', color='DarkGrey')
        ax.set_title(vac)
        ax.set_xlabel('Blood infections per year')

        if col == 0:
            ax.set_ylabel('Lifetime death efficacy (%)')

        ax.set_xticks(range(0, 7, 1))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(visible=True, which='both', axis='both', color='lightgrey', linestyle='--', linewidth=0.5)

    ylims = [ax.get_ylim() for ax in axs[1:]]
    for ax in axs[1:]:
        ax.set_ylim(min(list(zip(*ylims))[0]), max(list(zip(*ylims))[1]))

    if surveyed:
        data = infection_rate.get_pfpr_vs_clinical()
        λs, clinical_rate_under_5 = infection_rate.infection_rate_vs_clinical_rate()
        (_, trape_fit), data = infection_rate.model_pfpr_clinical_relationship(data)
        map = infection_rate.get_pfpr()
        with open('Data/pfpr_0-9.pickle', 'rb') as file:
            a = pickle.load(file)
        count_mean_total_africa = a[1]

        dists = infection_rate.construct_distributions(map, count_mean_total_africa, trape_fit, clinical_rate_under_5, λs)
        dists['infection_children_pdf'] *= 100 / dists['infection_children_pdf'].max()

        ax = axs[0]
        ax.plot('infection_rate', 'infection_children_cdf', data=dists, color='k')
        ax.fill_between(dists['infection_rate'], 0, 100*dists['infection_children_pdf'] / dists['infection_children_pdf'].max(), color='LightGrey', alpha=0.5)
        # ax.plot('infection_rate', 'infection_children_pdf', data=dists, color='k')
        # ax.fill_between(dists['infection_rate'], 0, 100*dists['infection_children_cdf'] / dists['infection_children_cdf'].max(), color='LightGrey', alpha=0.5)
        ax.grid(visible=True, which='both', axis='both', color='lightgrey', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Blood infections per year')
        ax.set_xlim(-0.5, 13)
        ax.set_ylim(0, 105)
        ax.set_ylabel('Percentage of\nchildren under 10')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    for ax, t in zip(axs, 'ABCD'):
        ax.set_title(f'{t}', loc='left')

    return fig, axs

def R21_rescue(treatment, β=0, season_width=1, boosters=0, years=30, ν_liver=1, τ_blood=1, smc=False, protection_blood=False):
    model = Model()
    model.pars = Parameters(
        lsv=True,
        smc=smc,
        β=β,
        study_months=years*12,
        season_width=season_width,
        season_peak=season_peak_liver,
        min_vac_age = min_vac_age_liver,
        vac_age_range=vac_age_range,
        offset_blood=(boosters + 1)*52 + season_peak_liver - season_peak_blood, # time blood vaccine rescue relative to peak season
        nboosters_liver=boosters,
        smc_repeats=boosters,
        smc_offset=smc_offset_liver,
        τ_blood=τ_blood,
        protection_blood=protection_blood,
        ν_liver=ν_liver,
        recording=min_vac_age_liver + vac_age_range # start recording at the same time as R21 for proper comparison of vaccines
    )
    model.variables = Variables(['Direct deaths'])
    model.measures = Measures(['efficacy'])

    λs = np.logspace(np.log10(0.5), np.log10(3), 20)
    treatments = {(True, False, None):'R21', (True, True, 1):'R21 then 2xRH5', (True, True, 3):'R21 then 4xRH5'}
    items = [[model, λ, lsv, bsv, b] for λ, (lsv, bsv, b) in product(λs, treatments.keys())]

    with Pool(12) as p:
        rs = p.starmap(rescue_sim, items)

    results = {'λ':[], 'treatment':[], 'case':[], 'efficacy':[]}
    for predictions, i in zip(rs, items):
        results['λ'] += [i[1]]
        results['treatment'] += [treatments[(i[2], i[3], i[4])]]
        results['case'] += ['Direct deaths']
        results['efficacy'] += [
            predictions['Vaccine Direct deaths efficacy'].iloc[-1]
        ]
    r = pd.DataFrame(results)
    r['vaccine'] = treatment
    return r


################################ ADDITIONAL #############################################


def find_closest_percentage(dist, inf_rate_value):
    """Find the closest percentage value for a given infection rate."""
    closest_idx = (dist['inf_rate'] - inf_rate_value).abs().idxmin()
    return dist.loc[closest_idx, 'percentage']

def infection_rate_distribution():
    dist = pd.read_csv('Data/infection_rate_distribution.csv')
    dist = dist[dist['inf_rate'] < 12]
    fig, ax = plt.subplots(figsize=(5, 4))

    x = 2
    y = find_closest_percentage(dist, x)
    ax.plot([x, x], [0, y], color='C0', ls='--')
    ax.plot([0, x], [y, y], color='C0', ls='--')

    ax.plot('inf_rate', 'percentage', color='k', data=dist)
    ax.fill_between(dist['inf_rate'], 0, 100*dist['density'] / dist['density'].max(), color='Grey', alpha=0.5)
    ax.set_xlabel('Blood infections per year')
    ax.set_ylabel('Percentage of surveyed sites since 2020')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 100)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return fig, ax