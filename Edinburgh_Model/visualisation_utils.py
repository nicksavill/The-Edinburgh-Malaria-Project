from Edinburgh_Model.vaccination_types import *


def panels(model, update='all'):
    pars = model.pars
    duration = int(weeks_per_month*pars.study_months) + pars.min_vac_age + pars.vac_age_range
    ages = np.arange(duration, dtype=float) # ages in weeks
    panel = {'smc':{}, 'liver_vac_prot':{}, 'blood_vac_prot':{}, 'season':{}, 'cfr':{}, 'min_vac_age':{}}

    if model.time_scale == 'Years':
        scale = 52
    elif model.time_scale == 'Months':
        scale = weeks_per_month
    else:
        scale = 1

    # update only what is necessary
    if update == 'all':
        update = list(panel)
    elif isinstance(update, str):
        update = [update]

    if model.show_vac and model.show_smc and 'smc' in update:
        if pars.smc:
            smc_args = pars.smc_ramp, pars.smc_rounds, pars.smc_repeats, pars.smc_coverage
            pars.smc_protection = smc_protection(ages, smc_args)
            panel['smc'] = {'x':(ages + pars.smc_offset)/scale, 'y':pars.smc_protection}
        else:
            panel['smc'] = {'x':[0], 'y':[0]}

    if model.show_vac and 'liver_vac_prot' in update:
        if pars.lsv:
            liver_vac_args = pars.ν_liver, pars.τ_liver, pars.period_liver, pars.nboosters_liver
            panel['liver_vac_prot'] = {
                'x':(ages)/scale,
                'y':np.concatenate((np.zeros(pars.offset_liver), pars.liver_vac_fn(ages[:len(ages)-pars.offset_liver], liver_vac_args)))}
        else:
            panel['liver_vac_prot'] = {'x':[0], 'y':[0]}

    if model.show_vac and 'blood_vac_prot' in update:
        if pars.bsv:
            blood_vac_args = 0, pars.τ_blood, pars.period_blood, pars.nboosters_blood
            panel['blood_vac_prot'] = {
                'x':(ages)/scale,
                'y':pars.α_severe * np.concatenate((np.zeros(pars.offset_blood), pars.blood_vac_fn(ages[:len(ages)-pars.offset_blood], blood_vac_args)))}
        else:
            panel['blood_vac_prot'] = {'x':[0], 'y':[0]}

    if model.show_season and 'season' in update:
        panel['season'] = {'x':ages/scale, 'y':np.concatenate([model.pars.malaria_season / model.pars.malaria_season.max()]*(duration//52+1))[:duration]}

    if model.show_cfr and ('cfr' in update or 'min_vac_age' in update):
        panel['cfr'] = {'x':ages[:duration]/scale, 'y':pars.direct_deaths[:duration]}
        if pars.lsv or pars.bsv:
            panel['min_vac_age'] = {'left':[pars.min_vac_age/scale], 'right':[(pars.min_vac_age+pars.vac_age_range)/scale], 'bottom':[0], 'top':[0.15]}

    return panel


def fig_data(model):
    """
        Parse the observed data in the pandas DataFrame model.data.dataframe into a form that can be plotted.
    """
    df = model.data.dataframe
    # extract the correct seasonality and case definition data
    s = model.data.seasonality
    c = model.data.case_def

    if model.time_scale == 'Years':
        df['month'] /= 12

    # each figure has its own data source returned in a dictionary
    data = {}
    ve_mask = (df['measure'] == 'efficacy')

    # efficacy data
    variables = df[ve_mask]['variable'].unique()
    measures = df[ve_mask]['measure'].unique()
    groups = df[ve_mask]['group'].unique()
    for v in set(variables).intersection(set(model.variables.display_vars)):
        for g in groups:
            if 'infection' in v:
                x = df.query('variable == @v and measure == "efficacy" and seasonality == @s and group == @g')
            else:
                x = df.query('variable == @v and measure == "efficacy" and seasonality == @s and group == @g and case_def == @c')
            data[(f'{v} efficacy', g)] = {'month':x['month'].values, 'value':x['value'].values, 'lower':x['lower'].values, 'upper':x['upper'].values}

    # other case data
    variables = df[~ve_mask]['variable'].unique()
    measures = df[~ve_mask]['measure'].unique()
    groups = df[~ve_mask]['group'].unique()
    for v in set(variables).intersection(set(model.variables.display_vars)):
        for m in set(measures).intersection(set(model.measures.display_measures) - set(['efficacy'])):
            for g in groups:
                if 'infection' in v:
                    x = df.query('variable == @v and measure == @m and seasonality == @s and group == @g')
                else:
                    x = df.query('variable == @v and measure == @m and seasonality == @s and group == @g and case_def == @c')
                data[(f'{v} {m}', g)] = {'month':x['month'].values, 'value':x['value'].values}

    if model.show_vac:
        x = df.query('variable == "efficacy" and seasonality == @s')
        if len(x):
            data[('vaccine_prot', 'efficacy')] = {'month':x['month'].values, 'value':x['value'].values, 'lower':x['lower'].values, 'upper':x['upper'].values}

    return data


