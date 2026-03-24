from Edinburgh_Model.visualisation_utils import *

import matplotlib.pyplot as plt
import matplotlib.ticker as mt
from matplotlib.colors import to_hex

vars = 'All infection', 'All clinical', 'Severe malaria', 'Direct deaths'
titles = 'Infection', 'Clinical', 'Severe', 'Death'

pink = to_hex(np.array([255, 94, 94])/255)
purple = to_hex(np.array([51, 0, 153])/255)
blue = to_hex(np.array([0, 153, 255])/255)
turquoise = to_hex(np.array([39, 255, 183])/255)

def plot(model, cohorts, sim_source=None, legend_title=None, legend_pos='last'):
    if sim_source is None:
        # simulation has not been run so simulate it here
        sim_source = model.simulation_fn(model)

    if model.show_vac or model.show_smc or model.show_season or model.show_death_rates:
        assert model.panels_fn is not None, 'Set model.Sources(panel_fn=panels)'
        panel_source = {f:data for f, data in model.panels_fn(model).items()}
        ncols = len(model.variables.display_vars) + 1
    else:
        panel_source = None
        ncols = len(model.variables.display_vars)

    pars = model.pars
    display_vars = model.variables.display_vars
    display_measures = model.measures.display_measures

    colours = [pink, purple, blue, turquoise, turquoise ]
    color  = {i:j for i, j in zip(cohorts, colours)}
    title_dict = {k:v for k, v in zip(vars, titles)}

    nrows = max(len(display_measures), int(model.show_vac | model.show_season) + int(model.show_death_rates))

    fig, axs = plt.subplots(nrows, ncols, figsize=(3*ncols*model.fig_xscale, 3*nrows*model.fig_yscale), squeeze=False)
    fig.subplots_adjust(wspace=0.27, hspace=0.3)

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

            for c in cohorts:
                ax.scatter(x=0, y=0, color='white') # forces origin to be plotted
                if not ((measure == 'efficacy' or measure == 'averted') and c == pars.control):
                    try:
                        # plot control and treatment for cases and cdf but only treatment for efficacy and averted
                        ax.plot(sim_source['ages'], sim_source[f'{c} {f}'], color=color[c], label=c)

                        # put legend only in one axis
                        if i == 0:
                            if legend_pos == 'last' and j == len(display_vars)-1:
                                ax.legend(loc='upper right', title=legend_title, fontsize=8, title_fontsize=8)
                            elif legend_pos == 'first' and j == 0:
                                ax.legend(loc='lower right', title=legend_title, fontsize=8, title_fontsize=8)

                        if variable == 'Direct deaths' and min(sim_source[f'{c} {f}']) < 0 and (measure == 'averted' or measure == 'efficacy'):
                            ax.axhline(0, color='black', ls=':')
                    except:
                        pass

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
                            if len(data) > 10:
                                ax.scatter(data['month'] * scale_time, data['value'], color=color[c], marker='.', label=c, alpha=0.25)
                            else:
                                ax.scatter(data['month'] * scale_time, data['value'], color=color[c], marker='o', facecolor='w', label=c)

    ######################### PANELS ###########################
    if panel_source is not None:
        row = 0
        if model.show_vac or model.show_season:
            ax = axs[row, ncols-1]
            ax.set_xlabel(model.time_scale)
            ax.set_ylabel(f'{pars.treatment} protection')
            if model.show_vac:
                if pars.lsv:
                    ax.plot(panel_source['liver_vac_prot']['x'], panel_source['liver_vac_prot']['y'], color='C2', label='R21')
                if pars.bsv:
                    ax.plot(panel_source['blood_vac_prot']['x'], panel_source['blood_vac_prot']['y'], color='C3', label='RH5')
            if model.show_smc and pars.smc:
                ax.plot(panel_source['smc']['x'], panel_source['smc']['y'], color='C0', label='SMC')

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

        if model.show_death_rates:
            ax = axs[row, ncols-1]
            if model.time_scale == 'Weeks':
                ax.set_xlabel('Age (weeks)')
            else:
                ax.set_xlabel('Age (years)')

            ax.set_ylabel('Risk of death')
            ax.scatter(x=0, y=0, color='white') # forces origin to
            ax.plot(panel_source['death rates']['x'], panel_source['death rates']['y'], color='Red')
            max_x = 10
            for c in cohorts:
                max_x = max(max_x, sim_source[f'{c} severe mean age x'].max())
                ax.plot(sim_source[f'{c} severe mean age x'], sim_source[f'{c} severe mean age y'], color=color[c], label=c)

            # ax.legend(title='Infections per year', loc='upper left', fontsize=8, title_fontsize=8)
            ax.set_xlim(-1, max_x*1.1)
            # Turn off right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            # Make minor tick marks visible on x-axis
            ax.xaxis.set_minor_locator(mt.AutoMinorLocator())
            ax.tick_params(axis='x', which='minor', length=4, color='grey')
            row += 1

        for i in range(row, nrows):
            axs[i, ncols-1].set_visible(False)  # hide empty panel
    return fig, axs
