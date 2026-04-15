import sys
sys.path.append('..')
from Edinburgh_Model.model import *
from Edinburgh_Model.interaction import *
from Edinburgh_Model.vaccination_types import *

from itertools import product
from copy import deepcopy
from multiprocessing import Pool

from bokeh.layouts import column, row, gridplot
from bokeh.models import NumeralTickFormatter, ColumnDataSource, HoverTool, Span, CrosshairTool, Div
from bokeh.plotting import figure, curdoc
from bokeh.palettes import Category10 as c10

def simulation(model):
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
    cohorts[pars.control] = Cohort(pars.max_weeks, pars.vac_age_range, pars, time_warning=model.time_warning)

    # initialise SMC in the control cohort if it is given
    if pars.smc_control:
        cohorts[pars.control].SMC(pars.smc_offset)

    init_unvaccinated_cohort(cohorts[pars.control], t_record_first, pars, model.logfile, code=model.code, fast=model.fast)

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
        sim_one_cohort_through_time(cohorts[pars.treatment], pars.max_vac_age, pars.max_weeks, t_record_first, pars, code=model.code, fast=model.fast)

    # simulate control cohort
    sim_one_cohort_through_time(cohorts[pars.control], pars.max_vac_age, pars.max_weeks, t_record_first, pars, code=model.code, fast=model.fast)

    ############################ construct data for plots
    y = get_results(cohorts, t_record_first, pars, display_vars, display_measures, model.show_cfr)

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
    if model.show_cfr:
        l = len(result['ages'])
        for c in cohorts.keys():
            result[f'{c} severe mean age x'] = y[c]['severe_mean_age'] * np.ones(l) / dt
            if model.time_scale == 'Months':
                result[f'{c} severe mean age x'] *= 4. / weeks_per_month
            result[f'{c} severe mean age y'] = np.linspace(0, 0.15, l)

    return result

def plot(doc):
    model.sim_source = ColumnDataSource(simulation(model))
    if model.show_vac or model.show_cfr:
        assert model.panels_fn is not None, 'Set model.Sources(panel_fn=panels)'
        model.panel_source = {f:ColumnDataSource(data=data) for f, data in model.panels_fn(model).items()}
    if model.data:
        model.data_source = {f:ColumnDataSource(data=data) for f, data in model.data_fn(model).items()}

    pars = model.pars
    display_vars = model.variables.display_vars
    display_measures = model.measures.display_measures

    cohorts = [pars.control] + ([pars.treatment] if pars.lsv or pars.bsv or pars.smc else [])
    color  = {i:j for i, j in zip(cohorts, ['Pink', 'Blue'])}
    titles = {'All infection':'Infections', 'All clinical':'Clinical', 'First clinical':'First clinical episode', 'Severe malaria':'Severe', 'Direct deaths':'Deaths'}
    fig = {}

    width = Span(dimension="width", line_width=0)
    height = Span(dimension="height", line_dash="dotted", line_width=2)

    for i, variable in enumerate(display_vars):
        for j, measure in enumerate(display_measures):
            f = f'{variable} {measure}'
            if ('infection' in variable or 'clinical' in variable) and measure == 'mean age':
                # no mean age for infections of clinical cases
                fig[f] = None
                continue

            fig[f] = figure(tools='hover', toolbar_location=None)
            fig[f].scatter(x=0, y=0, color='white') # forces origin to be plotted

            if measure == 'efficacy':
                fig[f].yaxis[0].formatter = NumeralTickFormatter(format='0.00')
            else:
                if model.pars.popsize > 1000:
                    fig[f].yaxis[0].formatter = NumeralTickFormatter()

            fig[f].outline_line_color = None
            for c in cohorts:
                if measure == 'efficacy':
                    fig[f].scatter(x=0, y=0, color='white') # forces origin to be plotted
                if variable == 'Direct deaths' and (measure == 'averted' or measure == 'efficacy'):
                    fig[f].add_layout(Span(location=0, dimension='width', line_color='Red'))

                if not ((measure == 'efficacy' or measure == 'averted') and c == pars.control):
                    # plot control and treatment for cases, cdf and mean age but only treatment for VE
                    fig[f].line('ages', f'{c} {f}', color=color[c], width=3, legend_label=c, source=model.sim_source)
                    fig[f].add_tools(CrosshairTool(overlay=[width, height]))


                    # put legend only in row 1, column 1
                    # if i == 0 and j == len(display_measures)-1:
                    if 'deaths' in variable and j == 0:
                        fig[f].legend.location = 'top_right'
                    else:
                        fig[f].legend.visible = False
            if j == 0:
                # title only in row 1
                fig[f].title.text = titles[variable]
                fig[f].title.text_font_size = "18pt"
            elif j == len(display_measures)-1:
                if pars.lsv or pars.bsv:
                    fig[f].xaxis.axis_label = f'{model.time_scale} post vaccination'
                else:
                    fig[f].xaxis.axis_label = f'{model.time_scale}'

            if i == 0:
                n = pars.popsize
                if pars.children == 'population':
                    n *= pars.study_months * weeks_per_month

                # y-axis label only in column 1
                if measure == 'cases':
                    if model.time_scale == 'Years':
                        fig[f].yaxis.axis_label = f'Annual cases\nper {n:,.0f}'
                    elif model.time_scale == 'Months':
                        fig[f].yaxis.axis_label = f'Monthly cases\nper {n:,.0f}'
                    elif model.time_scale == 'Weeks':
                        fig[f].yaxis.axis_label = f'Weekly cases\nper {n:,.0f}'

                elif measure == 'cdf':
                    fig[f].yaxis.axis_label = f'Cumulative cases\nper {n:,.0f}'
                elif measure == 'efficacy':
                    fig[f].yaxis.axis_label = 'Efficacy (%)'
                elif measure == 'averted':
                    fig[f].yaxis.axis_label = f'Cases averted\nper {n:,.0f}'
                elif measure == 'Kaplan-Meier':
                    fig[f].yaxis.axis_label = f'Proportion of\nchildren surviving'

    ######################### PANELS ###########################
    if model.show_vac or model.show_season:
        vaccine_prot = figure(x_axis_label=f'{model.time_scale} post vaccination', y_axis_label=f'{pars.treatment} protection', toolbar_location=None)
        vaccine_prot.add_tools(CrosshairTool(overlay=[width, height]))

        if model.show_vac:
            vaccine_prot.line('x', 'y', color=c10[4][2], width=3, source=model.panel_source['liver_vac_prot'])
            vaccine_prot.line('x', 'y', color=c10[4][3], width=3, source=model.panel_source['blood_vac_prot'])

        if model.show_smc:
            vaccine_prot.line('x', 'y', color=c10[4][0], width=3, source=model.panel_source['smc'])

        if model.show_season:
            vaccine_prot.line('x', 'y', color='DarkBlue', width=1, source=model.panel_source['season'])

    if model.show_cfr:
        cfr = figure(x_axis_label=f'Age ({model.time_scale})', y_axis_label='Risk of death', toolbar_location=None)

        cfr.scatter(x=0, y=0, color='white') # forces origin to be plotted
        cfr.line('x', 'y', color='Red', width=3, source=model.panel_source['cfr'])

        if pars.lsv or pars.bsv:
            l = cfr.quad(top='top', bottom='bottom', left='left', right='right', color='Grey', alpha=0.5, source=model.panel_source['min_vac_age'])
            cfr.add_tools(HoverTool(renderers=[l], attachment='above',
                                            tooltips=[
                                                ('Minimum vaccination age', '@left'),
                                                ('Maximum vaccination age', '@right'),
                                            ]))

        for c in cohorts:
            l = cfr.line(f'{c} severe mean age x', f'{c} severe mean age y', color=color[c], width=3, source=model.sim_source)
            cfr.add_tools(HoverTool(renderers=[l], attachment='above',
                                            tooltips=[
                                                (f'Mean age of severe episodes\nin {c} cohort', '@{'+f"{c} severe mean age x"+'}{0.1f}')
                                            ]))

    ######################### LAYOUT ###########################

    sliders_group1 = [
        [
            '',
            'study_months',
            'λ',
            # 'β',
        ],
        [
            'Malaria season',
            'season_peak',
            'season_width',
        ],
        [
            'Early life protection',
            'ρ_elp',
            'τ_elp',
        ],
        [
            'Seasonal malaria chemoprevention',
            'smc_coverage',
            'smc_offset',
            # 'smc_ramp',
            # 'smc_rounds',
            'smc_repeats',
        ]
    ]
    sliders_group2 = [
        [
            '',
            'min_vac_age',
            'vac_age_range',
        ],
        [
            'Pre-erythrocytic vaccine',
            'ν_liver',
            'τ_liver',
            # 'period_liver',
            'offset_liver',
            'nboosters_liver',
        ],
        [
            'Blood-stage vaccine',
            # 'ν_blood',
            'τ_blood',
            # 'period_blood',
            'offset_blood',
            'nboosters_blood',
            'ω_infection',
            'ω_clinical',
            'ω_severe',
        ]
    ]
    sliders_group3 = [
        [
            'Clinical risk',
            'ρ_clinical',
            'δ_clinical',
            'γ_clinical',
            # 'case_def_clinical',
        ],
        [
            'Severe risk',
            'ρ_severe',
            'δ_severe',
            'ε_severe',
            'cfr_modifier',
        ]
    ]

    buttons_group1 = [
        'time_scale',
        'season',
        'children',
        'recording',
        'cfr',
        'liver vaccine',
        'blood vaccine',
        'vaccines',
        'smc',
        # 'case definition',
    ]

    children = [[None for _ in range(7)] for _ in range(4)]

    for c, v in enumerate(display_vars):
        for r, m in enumerate(display_measures):
            children[r][c] = fig[f'{v} {m}']

    def make_slider_column(slider_group):
        S = model.sliders
        b = []
        for group in slider_group:
            c = [Div(text=f"<h3>{group[0]}</h3>")]
            c += [row(S[k][1], S[k][0]) for k in group[1:] if k in S]
            b.append(row(column(*c)))
        return column(*b)

    children[0][4] = make_slider_column(sliders_group1)
    children[0][5] = make_slider_column(sliders_group2)
    children[0][6] = make_slider_column(sliders_group3)

    M = model.buttons
    b = [row(M[k][1], M[k][0]) for k in buttons_group1 if k in M]
    b += [Div(text="<b>Use the browser's reload button to reset</b>")]
    children[2][6] = column(b)

    if model.show_vac or model.show_season:
        children[3][4] = vaccine_prot
    if model.show_cfr:
        children[3][5] = cfr

    gp = gridplot(
        children=children,
        width=int(250*model.fig_xscale),
        height=int(250*model.fig_yscale),
        toolbar_location=None
    )
    curdoc().add_root(gp)

model = Interaction()
model.measures = Measures(['cases', 'cdf', 'averted', 'efficacy'])
model.variables = Variables(['All infection', 'All clinical', 'Severe malaria', 'Direct deaths'])
model.Sources(simulation_fn=simulation)
config = {'show_season':True, 'show_vac':True, 'show_smc':True, 'show_cfr':True}
model.Config(**config)

model.pars = Parameters(
    treatment='vaccine',
    lsv=True,
    popsize=100000,
    λ=1,
    study_months=253,
    season_width=0.11,
    nboosters_blood=1,
    nboosters_liver=1,
    min_vac_age=int(round(7*weeks_per_month, 0)),
    season_peak=4,
)

model.Sliders('all')
model.Buttons('all')

plot(curdoc)