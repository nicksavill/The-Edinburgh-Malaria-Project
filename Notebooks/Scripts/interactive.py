import sys
sys.path.append('..')
from Edinburgh_Model.model import *
from Edinburgh_Model.interaction import *
from Edinburgh_Model.vaccination_types import *

from itertools import product
from copy import deepcopy
from multiprocessing import Pool

from bokeh.layouts import column, row, gridplot
from bokeh.models import NumeralTickFormatter, ColumnDataSource, Whisker, TeeHead, HoverTool, Span, CrosshairTool
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.palettes import Category10 as c10

season_peak_liver = 6
season_peak_blood = 4
smc_offset_liver = -3
smc_offset_blood = -4
min_vac_age_liver = int(round(7*weeks_per_month, 0))
min_vac_age_blood = int(round(10*weeks_per_month, 0))
vac_age_range = 52


def simulation(model):
    pars = model.pars
    display_vars = model.variables.display_vars
    display_measures = model.measures.display_measures

    # duration of simulation from birth of oldest vaccinated cohort to end of study
    duration = int(round(weeks_per_month*pars.study_months + pars.max_vac_age, 0))

    # if first infections or first clinical cases are displayed then record these otherwise save time by not recording them
    if 'First infection' in display_vars or 'First clinical' in display_vars:
        t_record_first = pars.t_first
    else:
        t_record_first = None

    # run a single unvaccinated, multi-age cohort until the end of the last primary vaccine dose
    # note there is no vaccine induced protection until the last primary dose is given
    cohorts = {}
    cohorts[pars.control] = Cohort(duration, pars, time_warning=model.time_warning)

    # initialise SMC in the control cohort if it is given
    if pars.smc_control:
        cohorts[pars.control].SMC(pars.smc_offset)

    init_unvaccinated_cohort(cohorts[pars.control], t_record_first, pars, model.logfile)

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
        sim_one_cohort_through_time(cohorts[pars.treatment], pars.max_vac_age, duration, t_record_first, pars)

    # simulate control cohort
    sim_one_cohort_through_time(cohorts[pars.control], pars.max_vac_age, duration, t_record_first, pars)

    ############################ construct data for plots
    y = get_results(cohorts, t_record_first, pars, display_vars, display_measures, model.show_death_rates)

    if model.time_scale == 'Years':
        dt = 52
    else:
        dt = 1

    result = {}

    for c in cohorts.keys():
        for variable in display_vars:
            z = y[c][f'{variable} cdf']

            # construct cases
            result[f'{c} {variable} cases'] = z[dt::dt] - z[:-dt:dt]

            # construct cumulative cases
            result[f'{c} {variable} cdf'] = z[dt::dt]


    for c in list(cohorts.keys())[1:]:
        for variable in display_vars:
            f = f'{variable} cdf'

            # construct efficacies
            if 'efficacy' in display_measures:
                result[f'{c} {variable} efficacy'] = 100*(1 - divide(y[c][f], y[pars.control][f])[dt::dt])

            # construct cases averted
            if 'averted' in display_measures:
                result[f'{c} {variable} averted'] = (y[pars.control][f] - y[c][f])[dt::dt]

    result['ages'] = np.arange(len(result[f'{c} {variable} cases']))

    # mean age of severe malaria episodes
    if model.show_death_rates:
        l = len(result['ages'])
        for c in cohorts.keys():
            mean_age = y[c]['severe_mean_age']
            result[f'{c} severe mean age x'] = mean_age * np.ones(l)
            result[f'{c} severe mean age y'] = np.linspace(0, pars.death_rate_multiplier*0.15, l)

    return result

def plot(doc):
    model.sim_source = ColumnDataSource(simulation(model))
    if model.show_vac or model.show_death_rates:
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

    tooltips = [("(x,y)", "($x, $y)")],
    # hover_tools = HoverTool(
    #     mode = 'vline'
    # )

    # width = Span(dimension="width", line_dash="dashed", line_width=2)
    # height = Span(dimension="height", line_dash="dotted", line_width=2)

    for i, variable in enumerate(display_vars):
        for j, measure in enumerate(display_measures):
            f = f'{variable} {measure}'
            if ('infection' in variable or 'clinical' in variable) and measure == 'mean age':
                # no mean age for infections of clinical cases
                fig[f] = None
                continue

            fig[f] = figure(tools='hover')
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
                if not ((measure == 'efficacy' or measure == 'averted') and c == pars.control):
                    # plot control and treatment for cases, cdf and mean age but only treatment for VE
                    fig[f].line('ages', f'{c} {f}', color=color[c], width=3, legend_label=c, source=model.sim_source)
                    # fig[f].add_tools(CrosshairTool(overlay=[width, height]))

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
            if variable == 'Severe malaria' and measure == 'mean age':
                fig[f].yaxis.axis_label = 'Mean age (years)'


    ######################### DATA ###########################
    # if model.data:
    #     data_color = {i:'DarkBlue' if i == pars.control else 'Orange' for i in model.data.dataframe['group'].unique()}
    #     for (f, group), source in model.data_source.items():
    #         if 'cases' in f and 'cases' in display_measures:
    #             fig[f].line('month', 'value', color=data_color[group], dash='dotted', width=2, source=source)
    #             fig[f].scatter('month', 'value', color=data_color[group], source=source)
    #         elif 'cdf' in f and 'cdf' in display_measures:
    #             fig[f].line('month', 'value', color=data_color[group], dash='dotted', width=2, source=source)
    #             fig[f].scatter('month', 'value', color=data_color[group], source=source)
    #         elif 'efficacy' in f and 'efficacy' in display_measures:
    #             fig[f].scatter('month', 'value', color=data_color[group], source=source)
    #             fig[f].scatter('month', 'upper', color='white', source=source) # scales y-axis so full extent of whiskers are plotted
    #             fig[f].scatter('month', 'lower', color='white', source=source) # scales y-axis so full extent of whiskers are plotted
    #             fig[f].add_layout(Whisker(base='month', upper='upper', lower='lower', line_color=data_color[group],
    #                                       upper_head=TeeHead(line_color=data_color[group]),
    #                                       lower_head=TeeHead(line_color=data_color[group]), level='annotation', source=source))

    ######################### PANELS ###########################
    if model.show_vac or model.show_season:
        vaccine_prot = figure(x_axis_label=model.time_scale, y_axis_label=f'{pars.treatment} protection')

        if model.show_vac:
            vaccine_prot.line('x', 'y', color=c10[4][2], width=3, source=model.panel_source['liver_vac_prot'])
            vaccine_prot.line('x', 'y', color=c10[4][3], width=3, source=model.panel_source['blood_vac_prot'])

        if model.show_smc:
            vaccine_prot.line('x', 'y', color=c10[4][0], width=3, legend_label='SMC', source=model.panel_source['smc'])

        if model.show_season:
            vaccine_prot.line('x', 'y', color='DarkBlue', width=1, legend_label='Season', source=model.panel_source['season'])

            # if model.data and ('vaccine_prot', 'efficacy') in model.data_source:
            #     vaccine_prot.scatter('month', 'upper', color='white', source=model.data_source[('vaccine_prot', 'efficacy')]) # scales y-axis so full extent of whiskers are plotted
            #     vaccine_prot.scatter('month', 'lower', color='white', source=model.data_source[('vaccine_prot', 'efficacy')]) # scales y-axis so full extent of whiskers are plotted
            #     vaccine_prot.add_layout(Whisker(base='month', upper='upper', lower='lower', line_color=c10[4][2],
            #                               upper_head=TeeHead(line_color=c10[4][2]),
            #                               lower_head=TeeHead(line_color=c10[4][2]), level='annotation', source=model.data_source[('vaccine_prot', 'efficacy')]))
        vaccine_prot.legend.visible = False

    if model.show_death_rates:
        if model.time_scale == 'Weeks':
            death_rates = figure(x_axis_label='Age (weeks)', y_axis_label='Risk of death')
        else:
            death_rates = figure(x_axis_label='Age (years)', y_axis_label='Risk of death')
        death_rates.scatter(x=0, y=0, color='white') # forces origin to be plotted
        death_rates.line('x', 'y', color='Red', width=3, source=model.panel_source['death rates'])
        if pars.lsv or pars.bsv:
            death_rates.quad(top='top', bottom='bottom', left='left', right='right', color='Grey', legend_label=f'{pars.treatment} ages', alpha=0.5, source=model.panel_source['min_vac_age'])
        for c in cohorts:
            death_rates.line(f'{c} severe mean age x', f'{c} severe mean age y', color=color[c], width=3, legend_label=f'{c}', source=model.sim_source)
        death_rates.legend.visible = False

    ######################### LAYOUT ###########################

    col1 = [
        'study_months',
        'λ',
        'β',
        'season_peak',
        'season_width',
        'ρ_elp',
        'τ_elp',
        'ρ_clinical',
        'δ_clinical',
        'γ_clinical',
        'case_def_clinical',
        'ρ_severe',
        'δ_severe',
        'ε_severe',
        'death_rate_multiplier',
    ]
    col2 = [
        'min_vac_age',
        'vac_age_range',
        'ν_liver',
        'τ_liver',
        'period_liver',
        'offset_liver',
        'nboosters_liver',
    ]
    col3 = [
        'ν_blood',
        'τ_blood',
        'period_blood',
        'offset_blood',
        'nboosters_blood',
        'ω_infection',
        'ω_clinical',
        'ω_severe',
        'smc_coverage',
        'smc_offset',
        'smc_ramp',
        'smc_rounds',
        'smc_repeats',
    ]

    children = [[None for _ in range(7)] for _ in range(4)]
    for col, v in enumerate(display_vars):
        for r, m in enumerate(display_measures):
            children[r][col] = fig[f'{v} {m}']

    children[0][4] = column([s for s in model.sliders if s.name in col1])
    children[0][5] = column([s for s in model.sliders if s.name in col2])
    children[0][6] = column([s for s in model.sliders if s.name in col3])
    if model.show_vac or model.show_season:
        children[3][4] = vaccine_prot
    if model.show_death_rates:
        children[3][5] = death_rates
    children[2][6] = column([row(button[1], button[0]) for button in model.buttons])
    gp = gridplot(
        children=children,
        width=int(300*model.fig_xscale),
        height=int(300*model.fig_yscale),
    )
    if doc is None:
        show(gp)
    else:
        doc.add_root(gp)

def run(port=None,
        simulation_fn=simulation,
        data_fn=None,
        variables=('All infection',),
        measures=('cases', 'cdf'),
        pars={},
        config={},
        data=None,
        buttons=[],
        sliders=[],
        ):
    global model
    model = Interaction()
    model.pars = Parameters(**pars)
    model.measures = Measures(measures)
    model.variables = Variables(variables)
    model.data = data

    # for the bokeh interactive plots we need to define these
    model.Sources(simulation_fn=simulation_fn, data_fn=data_fn)
    model.Config(**config)

    if port is None:
        plot(None)
    else:
        model.Sliders(sliders)
        model.Buttons(buttons)
        show(plot, notebook_url=f"http://localhost:{port}")
