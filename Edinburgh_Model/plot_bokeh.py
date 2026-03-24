from Edinburgh_Model.interaction import *

from bokeh.layouts import column, row, gridplot
from bokeh.models import NumeralTickFormatter, ColumnDataSource, HoverTool, Whisker, TeeHead, HoverTool, Span, CrosshairTool, Div
from bokeh.plotting import figure
from bokeh.palettes import Category10 as c10
from bokeh.io import show

model = Interaction()

def plot(doc):
    model.sim_source = ColumnDataSource(model.simulation_fn(model))
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
        vaccine_prot = figure(x_axis_label=f'{model.time_scale} post vaccination', y_axis_label=f'{pars.treatment} protection', toolbar_location=None)
        vaccine_prot.add_tools(CrosshairTool(overlay=[width, height]))

        if model.show_vac:
            vaccine_prot.line('x', 'y', color=c10[4][2], width=3, source=model.panel_source['liver_vac_prot'])
            vaccine_prot.line('x', 'y', color=c10[4][3], width=3, source=model.panel_source['blood_vac_prot'])

        if model.show_smc:
            vaccine_prot.line('x', 'y', color=c10[4][0], width=3, legend_label='SMC', source=model.panel_source['smc'])

        if model.show_season:
            vaccine_prot.line('x', 'y', color='DarkBlue', width=1, legend_label='Season', source=model.panel_source['season'])

        vaccine_prot.legend.visible = False

    if model.show_death_rates:
        death_rates = figure(x_axis_label=f'Age ({model.time_scale})', y_axis_label='Risk of death', toolbar_location=None)

        death_rates.scatter(x=0, y=0, color='white') # forces origin to be plotted
        death_rates.line('x', 'y', color='Red', width=3, source=model.panel_source['death rates'])

        if pars.lsv or pars.bsv:
            l = death_rates.quad(top='top', bottom='bottom', left='left', right='right', color='Grey', legend_label=f'{pars.treatment} ages', alpha=0.5, source=model.panel_source['min_vac_age'])
            death_rates.add_tools(HoverTool(renderers=[l], attachment='above',
                                            tooltips=[
                                                ('Minimum vaccination age', '@left'),
                                                ('Maximum vaccination age', '@right'),
                                            ]))

        for c in cohorts:
            l = death_rates.line(f'{c} severe mean age x', f'{c} severe mean age y', color=color[c], width=3, legend_label=f'{c}', source=model.sim_source)
            death_rates.add_tools(HoverTool(renderers=[l], attachment='above',
                                            tooltips=[
                                                (f'Mean age of severe episodes\nin {c} cohort', '@{'+f"{c} severe mean age x"+'}{0.1f}')
                                            ]))
        death_rates.legend.visible = False

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
        ],
        [
            'Severe risk',
            'ρ_severe',
            'δ_severe',
            'ε_severe',
            # 'SPACE',
            # 'case_def_clinical',
            # 'death_rate_multiplier',
        ]
    ]

    buttons_group1 = [
        'time_scale',
        'children',
        'recording',
        'death rate',
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
        colours = "#E4DFCE" "#DEE4CE" "#CEDEE4"
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
    children[2][6] = column([row(M[k][1], M[k][0]) for k in buttons_group1 if k in M])

    if model.show_vac or model.show_season:
        children[3][4] = vaccine_prot
    if model.show_death_rates:
        children[3][5] = death_rates
    gp = gridplot(
        children=children,
        width=int(300*model.fig_xscale),
        height=int(375*model.fig_yscale),
    )
    if doc is None:
        show(gp)
    else:
        doc.add_root(gp)

