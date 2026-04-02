from Edinburgh_Model.model import *

import pandas as pd
from copy import deepcopy

def simulate_single_vaccine(model):
    """
        simulate one control cohort and one vaccine cohort and aggregate results into weeks or years for plotting
        return a pandas DataFrame of the required outputs
    """
    pars = model.pars
    display_vars = model.variables.display_vars
    display_measures = model.measures.display_measures

    # if first infections or first clinical cases are displayed then record these otherwise save time by not recording them
    if 'First infection' in display_vars or 'First clinical' in display_vars:
        t_record_first = pars.t_first
    else:
        t_record_first = None

    # run a single unvaccinated, multi-age cohort until the end of the last primary vaccine dose
    # note there is no vaccine induced protection until the last primary dose is given
    cohorts = {}
    cohorts[pars.control] = Cohort(pars.max_weeks, pars, time_warning=model.time_warning)

    # initialise SMC in the control cohort if it is given
    if pars.smc_control:
        cohorts[pars.control].SMC(pars.smc_offset)

    init_unvaccinated_cohort(cohorts[pars.control], t_record_first, pars, model.logfile, code='C')

    # copy the unvaccinated cohort into the vaccinated cohort on last primary dose
    if pars.lsv or pars.bsv or pars.smc:
        cohorts[pars.treatment] = deepcopy(cohorts[pars.control])

        # set the intervention type(s) for the treated cohort
        # interventions can be offset if they do not coincide
        if pars.lsv:
            cohorts[pars.treatment].liver_stage_vaccinate(pars.offset_liver)
        if pars.bsv:
            cohorts[pars.treatment].blood_stage_vaccinate(pars.offset_blood)
        if pars.smc:
            cohorts[pars.treatment].SMC(pars.smc_offset)

        # run the vaccinated cohort to the end of the study
        sim_one_cohort_through_time(cohorts[pars.treatment], pars.max_vac_age, pars.max_weeks, t_record_first, pars, code='C')

    # simulate control cohort
    sim_one_cohort_through_time(cohorts[pars.control], pars.max_vac_age, pars.max_weeks, t_record_first, pars, code='C')

    ############################ construct data for plots
    y = get_results(cohorts, t_record_first, pars, display_vars, display_measures, model.show_death_rates)

    if model.time_scale == 'Years':
        dt = 52
    elif model.time_scale == 'Months':
        dt = 4 # okay, so 13 4-week periods in a year
    else: # 'Weeks':
        dt = 1

    result = {}

    for c in cohorts.keys():
        for variable in display_vars:
            z = y[c][f'{variable} cdf']

            # construct cases
            if model.time_scale == 'Weeks':
                result[f'{c} {variable} cases'] = y[c][f'{variable} cases']
            else:
                result[f'{c} {variable} cases'] = np.concatenate((z[dt-1:dt], z[2*dt-1::dt] - z[dt-1:-dt:dt]))
                if model.time_scale == 'Months':
                    # scale cases because we are summing over 28 days rather than 30.4 days
                    result[f'{c} {variable} cases'] *= weeks_per_month / 4.

            # construct cumulative cases
            result[f'{c} {variable} cdf'] = z[dt-1::dt]


    for c in list(cohorts.keys())[1:]:
        for variable in display_vars:
            f = f'{variable} cdf'

            # construct efficacies
            if 'efficacy' in display_measures:
                result[f'{c} {variable} efficacy'] = 100*(1 - divide(y[c][f], y[pars.control][f])[dt-1::dt])

            # construct cases averted
            if 'averted' in display_measures:
                result[f'{c} {variable} averted'] = (y[pars.control][f] - y[c][f])[dt-1::dt]

    result['ages'] = np.arange(len(result[f'{c} {variable} cases']), dtype=float)
    if model.time_scale == 'Months':
        # scale times because we are summing over 28 days rather than 30.4 days
        result['ages'] *= 4. / weeks_per_month

    # mean age of severe malaria episodes
    if model.show_death_rates:
        l = len(result['ages'])
        for c in cohorts.keys():
            result[f'{c} severe mean age x'] = y[c]['severe_mean_age'] * np.ones(l) / dt
            if model.time_scale == 'Months':
                result[f'{c} severe mean age x'] *= 4. / weeks_per_month
            result[f'{c} severe mean age y'] = np.linspace(0, 0.15, l)

    return pd.DataFrame(result)
