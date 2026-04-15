import sys
sys.path.append('..')
from Edinburgh_Model.model import *
from Edinburgh_Model.visualisation_utils import *

import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from copy import deepcopy

def monthly_predictions(model, cohorts):
    pars = model.pars

    # switch these off for control cohort
    pars.lsv = False
    pars.smc = False

    # treated and control cohorts are the same until first dose
    cohorts['control'] = Cohort(pars.max_weeks, pars.vac_age_range, pars)
    init_unvaccinated_cohort(cohorts['control'], None, pars)

    # copy control cohort into treated cohorts
    cohorts['rtss'] = deepcopy(cohorts['control'])
    cohorts['smc'] = deepcopy(cohorts['control'])
    cohorts['rtss_smc'] = deepcopy(cohorts['control'])

    # control cohort
    sim_one_cohort_through_time(cohorts['control'], pars.max_vac_age, pars.max_weeks, None, pars)

    # rtss cohort
    pars.lsv = True
    pars.smc = False
    pars.update_pars(('lsv', 'smc'))
    cohorts['rtss'].liver_stage_vaccinate(pars.offset_liver)
    sim_one_cohort_through_time(cohorts['rtss'], pars.max_vac_age, pars.max_weeks, None, pars)

    # SMC cohort
    pars.lsv = False
    pars.smc = True
    pars.update_pars(('lsv', 'smc'))
    cohorts['smc'].SMC(pars.smc_offset)
    sim_one_cohort_through_time(cohorts['smc'], pars.max_vac_age, pars.max_weeks, None, pars)

    # rtss_smc cohort
    pars.lsv = True
    pars.smc = True
    pars.update_pars(('lsv', 'smc'))
    cohorts['rtss_smc'].liver_stage_vaccinate(pars.offset_liver)
    cohorts['rtss_smc'].SMC(pars.smc_offset)
    sim_one_cohort_through_time(cohorts['rtss_smc'], pars.max_vac_age, pars.max_weeks, None, pars)

    return cohorts


def dicko_simulation(model):
    pars = model.pars
    display_vars = model.variables.display_vars
    display_measures = model.measures.display_measures
    # duration of simulation from birth of oldest vaccinated cohort to end of study

    cohorts = monthly_predictions(model, {})

    ############################ construct data for dicko_calibration_plots
    y = get_results(cohorts, None, pars, display_vars)

    ages = np.arange(pars.max_weeks - pars.max_vac_age)
    if model.time_scale == 'Months':
        result = {'ages':ages/weeks_per_month}
    else:
        result = {'ages':ages/52}

    # data for control and vaccinated cohorts (measure=cases, cdf)
    # scale to cases per month
    for c in cohorts.keys():
        for variable in display_vars:
            for measure, scale in zip(['cases', 'cdf'], (weeks_per_month, 1)):
                result[f'{c} {variable} {measure}'] = y[c][f'{variable} {measure}'] * scale

    # construct efficacies (measure=ve)
    if 'efficacy' in display_measures:
        for c in ('smc', 'rtss'):
            for variable in display_vars:
                result[f'{c} {variable} ve'] = 1 - divide(y['rtss_smc'][f'{variable} cdf'], y[c][f'{variable} cdf'])

    return result


################################# CLINICAL CASE CALIBRATION ####################################


def dicko_clinical_episode_calibration():
    """ calibrate the following model parameters with clinical case data from Dicko et al (2024)
        infection rate: λ
        timing of malaria season: season_peak
        strength of seasonality: season_width
        timing of RTS,S delivery relative to malaria season: offset_liver
        ramping up of SMC protective profile: smc_ramp
        timing of SMC delivery relative to malaria season: smc_offset
    """

    def observed_and_predicted(season_peak=26, season_width=0.094, offset_liver=17, smc_ramp=0.19, smc_offset=14, smc_rounds=4, λ=2.3):
        model = Model()
        model.pars = Parameters(
            lsv=True,
            smc=True,
            study_months=60,
            min_vac_age=int(round(5*weeks_per_month, 0)),
            vac_age_range=int(round((17-5)*weeks_per_month, 0)),
            vac_profile_liver='Imperial RTS,S',
            popsize=1000,
            season_peak=int(round(season_peak, 0)),
            season_width=season_width,
            case_def_clinical=0.43 / 0.8,
            nboosters_liver=4,
            offset_liver=int(round(offset_liver, 0)),
            smc_ramp=smc_ramp,
            smc_rounds=smc_rounds,
            smc_repeats=4,
            smc_offset=int(round(smc_offset, 0)),
            smc_coverage=0.9,
            λ=λ
        )
        model.variables = Variables(['All clinical'])
        model.measures = Measures(['cases'])
        model.Config(time_scale='Months')

        predictions = dicko_simulation(model)

        # read in the observed SM cases and deaths
        dicko_clinical = pd.read_csv('Data/dicko_2024_fig3b.csv')

        table = {'group':[], 'observed':[], 'predicted':[], 'month':[]}
        for c in ('smc', 'rtss', 'rtss_smc'):
            cases = predictions[f'{c} All clinical cases']
            ages = predictions['ages']
            observed = dicko_clinical.query('group == @c')
            for _, r in observed.iterrows():
                # for each month in observed find nearest month in predicted
                m = (np.abs(ages - r['month'])).argmin()

                table['group'].append(c)
                table['predicted'].append(cases[m])
                table['observed'].append(r['value'])
                table['month'].append(r['month'])

        table = pd.DataFrame(table)
        return table

    def clinical_episode_difference(x):
        """ sum of absolute difference between observed and predicted clinical cases """
        table = observed_and_predicted(season_peak=x[0], season_width=x[1], offset_liver=x[2], smc_ramp=x[3], smc_offset=x[4], smc_rounds=x[5], λ=x[6])
        return (abs(table['predicted'] - table['observed'])).sum()

    # # minimise the sum of the absolute differences between observed and predicted SM cases to estimate risk of SM on first infection
    # and decay in risk with exposure
    reseason_width = minimize(clinical_episode_difference, x0=(27, 0.1, 16, 0.2, 15, 4, 2.3), method='nelder-mead')
    print('Parameter values that minimise the sum of the absolute differences between observed and predicted clinical cases')
    print(f'  season_peak = {reseason_width.x[0]:.2g}')
    print(f'  season_width = {reseason_width.x[1]:.2g}')
    print(f'  offset_liver = {reseason_width.x[2]:.2g}')
    print(f'  smc_ramp = {reseason_width.x[3]:.2g}')
    print(f'  smc_offset = {reseason_width.x[4]:.2g}')
    print(f'  smc_rounds = {reseason_width.x[5]:.2g}')
    print(f'  λ = {reseason_width.x[6]:.2g}')
    return reseason_width.x


################################# CLINICAL CASE PREDICTION #####################################

def dicko_clinical_episode_plot(model, show_panels):
    result = model.simulation_fn(model)
    model_source = pd.DataFrame(result)
    panel_source = pd.DataFrame({f:data for f, data in model.panels_fn(model).items()})

    cohorts = {'rtss':'RTS,S-only', 'smc':'SMC-only', 'rtss_smc':'RTS,S+SMC'}
    color  = {i:j for i, j in zip(cohorts, ('C2', 'C0', 'C1'))}

    if show_panels:
        layout = """ad
                    be
                    c."""
    else:
        layout = """a
                    b
                    c"""
    fig, axs = plt.subplot_mosaic(layout, figsize=(3*2*model.fig_xscale, 3*4*model.fig_yscale))
    fig.subplots_adjust(hspace=0.3)
    fig.subplots_adjust(wspace=0.25)

    # ######################### CASES ###########################
    for (c, l), f in zip(cohorts.items(), 'abc'):
        axs[f].plot('ages', f'{c} All clinical cases', '', data=model_source, color=color[c], lw=2, label='Model')
        axs[f].set_xlabel('Years')
        axs[f].set_ylabel('Clinical cases per 1,000 per month')
        axs[f].set_title(f'{f.upper()}: {cohorts[c]}', fontsize=16, loc='left')
        axs[f].set_ylim(-5, 155)

    # ######################### EFFICACY ###########################
    if show_panels:
        for c in list(cohorts.keys())[:2]:
            axs['e'].plot('ages', f'{c} All clinical ve', data=model_source, color=color[c], lw=2, label=f'RTS,S+SMC vs. {cohorts[c]}')
            axs['e'].set_xlabel('Years')
            axs['e'].set_ylabel('Efficacy')
            axs['e'].set_title('E', fontsize=16, loc='left')

            axs['e'].legend()

    # ######################### PROTECTION ###########################
    if show_panels:
        axs['d'].plot('x', 'y', '', data=panel_source['liver_vac_prot'], color='C2', lw=2, label='RTS,S')
        axs['d'].plot('x', 'y', '', data=panel_source['smc'], color='C0', lw=2, label='SMC')
        axs['d'].plot('x', 'y', '', data=panel_source['season'], color='Red', lw=1, label='Season')
        axs['d'].set_xlim(axs['d'].get_xlim()[0], axs['a'].get_xlim()[1])
        axs['d'].set_xlabel('Years')
        axs['d'].set_ylabel('Protection')
        axs['d'].set_title('D', fontsize=16, loc='left')
        axs['d'].legend()

    # ######################### DATA ###########################
    for (c, l), f in zip(cohorts.items(), 'abc'):
        data = model.data.dataframe.query('measure == "cases" and group == @c')
        axs[f].scatter(data['month']/12, data['value'], color='k', marker='.', label='Data')
        axs[f].plot(data['month']/12, data['value'], color='k', ls='--', lw=0.5)
        axs[f].legend(loc='upper left')

    if show_panels:
        for c in list(cohorts.keys())[:2]:
            data = model.data.dataframe.query('measure == "efficacy" and group == @c')
            axs['e'].errorbar(data['month']/12, data['value'], yerr=(data['value']-data['lower'], data['upper']-data['value']), color=color[c], ls='', marker='.')

    for ax in axs.values():
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    return fig, axs


def dicko_clinical_episode_prediction(estimates, fig_xscale=1.5, fig_yscale=1.5, show_panels=True):
    """ plot observed and predicted clinical cases of calibrated model with predicted efficacies """
    model = Model()
    model.pars = Parameters(
        lsv=True,
        smc=True,
        study_months=60,
        min_vac_age=int(round(5*weeks_per_month, 0)),
        vac_age_range=int(round((17-5)*weeks_per_month, 0)),
        vac_profile_liver='Imperial RTS,S',
        popsize=1000,
        season_peak=int(round(estimates[0], 0)),
        season_width=estimates[1],
        case_def_clinical=0.43/0.8,
        nboosters_liver=4,
        offset_liver=int(round(estimates[2], 0)),
        smc_ramp=estimates[3],
        smc_rounds=estimates[5],
        smc_repeats=4,
        smc_offset=int(round(estimates[4], 0)),
        smc_coverage=0.9,
        λ=estimates[6]
    )
    model.variables = Variables(['All clinical'])
    model.measures = Measures(['cases', 'efficacy'])
    model.data = Data(case_def='primary', seasonality='seasonal', datafile='Data/dicko_2024_fig3b.csv')

    model.Config(show_season=True, show_smc=True, show_vac=True, time_scale='Years', fig_xscale=fig_xscale, fig_yscale=fig_yscale)
    model.Sources(simulation_fn=dicko_simulation, data_fn=fig_data, panels_fn=panels)

    return dicko_clinical_episode_plot(model, show_panels=show_panels)
