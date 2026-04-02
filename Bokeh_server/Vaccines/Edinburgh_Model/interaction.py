from Edinburgh_Model.model import *
from Edinburgh_Model.vaccination_types import *
from Edinburgh_Model.visualisation_utils import *

from bokeh.models import Slider, RadioButtonGroup, CustomJSTickFormatter, Tooltip, HelpButton
from bokeh.models.dom import HTML

class Interaction:
    def __init__(self):
        # Default interactions and outputs
        self.Config() # default panels to show are vaccine protection and season
        self.sliders = {} # default is no sliders
        self.buttons = {} # default is no buttons
        self.data = None

    def Sources(self, simulation_fn=None, data_fn=None, panels_fn=panels):
        """ set the sources of information for the plots """
        self.simulation_fn = simulation_fn
        self.data_fn = data_fn
        self.panels_fn = panels_fn

    def Config(self, show_smc=False, show_vac=False, show_season=False,
               show_death_rates=False, time_scale='Years',
               show_pre_vac=False, fig_xscale=1, fig_yscale=1, plotfile='', miscellaneous={},
               time_warning=0, logfile=False):

        assert isinstance(show_smc, bool), 'show_smc must be boolean'
        assert isinstance(show_vac, bool), 'show_vac must be boolean'
        assert isinstance(show_season, bool), 'show_season must be boolean'
        assert isinstance(show_death_rates, bool), 'show_death_rates must be boolean'
        assert time_scale in ['Years', 'Months', 'Weeks'], f'time_scale must be Years, Months or Weeks, got {time_scale}'
        assert fig_xscale > 0, 'fig_xscale must be positive'
        assert fig_yscale > 0, 'fig_yscale must be positive'
        assert isinstance(show_pre_vac, bool), 'show_pre_vac must be boolean'
        assert isinstance(plotfile, str), 'plotfile must be string'
        assert isinstance(miscellaneous, dict), 'miscellaneous must be dictionary'
        assert time_warning >= 0, f'time_warning must be non-negative, got {time_warning}'
        assert isinstance(logfile, bool), 'logfile must be boolean'

        """ plotting and other configuration variables """
        self.show_smc = show_smc  # show SMC protection in vaccination protection panel
        self.show_vac = show_vac  # show vaccine protection panel
        self.show_season = show_season  # show seasonality in vaccine protection panel
        self.show_death_rates = show_death_rates  # show death rate by age with vaccination age range overlaid
        self.show_pre_vac = show_pre_vac  # show infections, etc before last primary dose
        self.time_scale = time_scale  # time scale eg 'Months', 'Years'
        self.fig_xscale = fig_xscale  # x-scale of graphs
        self.fig_yscale = fig_yscale  # y-scale of graphs
        self.plotfile = plotfile # plot filename
        self.miscellaneous = miscellaneous # other miscellaneous user-supplied configuration variables in a dictionary
        self.time_warning = time_warning # if time_warning > 0 print a warning if the simulation is going to take a longer than time_warning seconds
        if logfile:
            self.logfile = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.log")
        else:
            self.logfile = ''

    def Buttons(self, button_types):
        if isinstance(button_types, str):
            button_types = [button_types]

        buttons = ['all', 'season', 'children', 'recording', 'death rate', 'case definition', 'liver vaccine', 'blood vaccine', 'vaccines', 'smc', 'time_scale']
        for b in button_types:
            assert b in buttons, f'Available buttons are {str(buttons)}, got {str(b)}'

        ######################### TIME SCALE BUTTON ###########################
        time_scale_labels = {"Weeks":0, "Months":1, "Years":2}
        def change_time_scale(attr, old, new):
            if new == 0:
                self.time_scale = list(time_scale_labels)[0]
            elif new == 1:
                self.time_scale = list(time_scale_labels)[1]
            else:
                self.time_scale = list(time_scale_labels)[2]

            self.sim_source.data = self.simulation_fn(self)
            if self.show_vac:
                s = self.panels_fn(self, ['season', 'liver_vac_prot', 'blood_vac_prot'])
                self.panel_source['season'].data = s['season']
                self.panel_source['liver_vac_prot'].data = s['liver_vac_prot']
                self.panel_source['blood_vac_prot'].data = s['blood_vac_prot']

            if self.show_death_rates:
                s = self.panels_fn(self, 'death rates')
                self.panel_source['death rates'].data = s['death rates']
                if self.pars.lsv or self.pars.bsv:
                    self.panel_source['min_vac_age'].data = s['min_vac_age']

        if 'all' in button_types or 'time_scale' in button_types:
            self.Time_scale_button = RadioButtonGroup(labels=list(time_scale_labels), active=time_scale_labels[self.time_scale])
            self.Time_scale_button.on_change('active', change_time_scale)
            tooltip = Tooltip(content=HTML("<center>Weekly, Monthly or Annual cases</center>"), position='top')
            self.buttons['time_scale'] = (self.Time_scale_button, HelpButton(tooltip=tooltip))

        ######################### COHORT BUTTON ###########################
        children_labels = {"cohort":0, "population":1}
        def change_children(attr, old, new):
            if new == 0:
                self.pars.children = list(children_labels)[0]
            else:
                self.pars.children = list(children_labels)[1]
            self.pars.update_pars('children')
            self.sim_source.data = self.simulation_fn(self)

        if 'all' in button_types or 'children' in button_types:
            self.Children_button = RadioButtonGroup(labels=list(children_labels), active=children_labels[self.pars.children])
            self.Children_button.on_change('active', change_children)
            tooltip = Tooltip(content=HTML("<center>Simulate a single, multi-age cohort<br>or a multi-cohort population</center>"), position='top')
            self.buttons['children'] = (self.Children_button, HelpButton(tooltip=tooltip))

        ######################### RECORDING BUTTON ###########################
        recording_labels = {'Birth':0, 'Last primary dose':1}
        def change_recording(attr, old, new):
            if new == 0:
                self.pars.recording = list(recording_labels)[0]
            else:
                self.pars.recording = list(recording_labels)[1]
            self.pars.update_pars('recording')
            self.sim_source.data = self.simulation_fn(self)

        if 'all' in button_types or 'recording' in button_types:
            self.Recording_button = RadioButtonGroup(labels=list(recording_labels), active=recording_labels[self.pars.recording])
            self.Recording_button.on_change('active', change_recording)
            tooltip = Tooltip(content=HTML("<center>Record cases from birth or<br>from last primary dose</center>"), position='top')
            self.buttons['recording'] = (self.Recording_button, HelpButton(tooltip=tooltip))

        ######################### DEATHS BUTTON ###########################
        death_labels = {'Reyburn':0, 'Flat':1}
        def change_deaths(attr, old, new):
            if new == 0:
                self.pars.deaths = list(death_labels)[0]
            else:
                self.pars.deaths = list(death_labels)[1]
            self.pars.update_pars('deaths')
            self.sim_source.data = self.simulation_fn(self)
            if self.show_death_rates:
                s = self.panels_fn(self, 'death rates')
                self.panel_source['death rates'].data = s['death rates']
                if self.pars.lsv or self.pars.bsv:
                    self.panel_source['min_vac_age'].data = s['min_vac_age']

        if 'all' in button_types or 'death rate' in button_types:
            self.Deaths_button = RadioButtonGroup(labels=list(death_labels), active=death_labels[self.pars.deaths])
            self.Deaths_button.on_change('active', change_deaths)
            tooltip = Tooltip(content=HTML("<center>Age-specific death rates from<br>Reyburn at al. (2005) or flat</center>"), position='top')
            self.buttons['death rate'] = (self.Deaths_button, HelpButton(tooltip=tooltip))

        ######################### DATA CASE DEFINITION BUTTON ###########################
        case_def_labels = {"primary":0, "secondary":1}
        def change_case_def(attr, old, new):
            if self.data is not None:
                if new == 0:
                    self.data.case_def = list(case_def_labels)[0]
                else:
                    self.data.case_def = list(case_def_labels)[1]
                for f, data in self.data_fn(self).items():
                    self.data_source[f].data = data

        if ('all' in button_types or 'case definition' in button_types) and self.data is not None:
            self.Case_def_button = RadioButtonGroup(labels=list(case_def_labels), active=case_def_labels[self.data.case_def])
            self.Case_def_button.on_change('active', change_case_def)
            tooltip = Tooltip(content=HTML("<center>Plot data from either<br>primary or secondary case defintion</center>"), position='top')
            self.buttons['case definition'] = (self.Case_def_button, HelpButton(tooltip=tooltip))

        ######################### LIVER VACCINATION BUTTON ##################################
        liver_vaccination_labels = {'RTS,S':0, 'R21':1}
        def change_liver_vaccination(attr, old, new):
            if new == 0:
                self.pars.vac_profile_liver = f'Imperial {list(liver_vaccination_labels)[0]}'
            elif new == 1:
                self.pars.vac_profile_liver = f'Imperial {list(liver_vaccination_labels)[1]}'

            self.pars.update_pars('vac_profile_liver')
            self.sim_source.data = self.simulation_fn(self)
            if self.show_vac and self.pars.lsv:
                s = self.panels_fn(self, ['liver_vac_prot'])
                self.panel_source['liver_vac_prot'].data = s['liver_vac_prot']

        if 'all' in button_types or 'liver vaccine' in button_types:
            self.Liver_vaccination_button = RadioButtonGroup(labels=list(liver_vaccination_labels),
                                                             active=liver_vaccination_labels[self.pars.vac_profile_liver.split()[1]])
            self.Liver_vaccination_button.on_change('active', change_liver_vaccination)
            tooltip = Tooltip(content=HTML("<center>Use Imperial's RTS,S or R21<br>vaccine protection profile</center>"), position='top')
            self.buttons['liver vaccine'] = (self.Liver_vaccination_button, HelpButton(tooltip=tooltip))

        ######################### BLOOD VACCINATION BUTTON ##################################
        blood_vaccination_labels = {'Yes':0, 'No':1}
        bv = {True:0, False:1}
        def change_blood_vaccination(attr, old, new):
            if new == 0:
                self.pars.protection_blood = list(bv)[0]
            elif new == 1:
                self.pars.protection_blood = list(bv)[1]
            self.pars.update_pars('protection_blood')
            self.sim_source.data = self.simulation_fn(self)
            if self.show_vac and self.pars.bsv:
                s = self.panels_fn(self, ['blood_vac_prot'])
                self.panel_source['blood_vac_prot'].data = s['blood_vac_prot']

        if 'all' in button_types or 'blood vaccine' in button_types:
            self.Blood_vaccination_button = RadioButtonGroup(labels=list(blood_vaccination_labels), active=bv[self.pars.protection_blood])
            self.Blood_vaccination_button.on_change('active', change_blood_vaccination)
            tooltip = Tooltip(content=HTML("""Whether RH5.1 protects against blood infection or not.<br>
                                           Yes: ω_infection=0.43, ω_clinical=0, ω=severe=0.29<br>
                                           No: ω_infection=0, ω_clinical=0.5, ω=severe=0.29<br>
                                           These values can also be adjusted with sliders"""), position='top')
            self.buttons['blood vaccine'] = (self.Blood_vaccination_button, HelpButton(tooltip=tooltip))

        ######################### SEASON BUTTON ##################################
        season_labels = {'Seasonal':0, 'Perennial':1}
        def change_season(attr, old, new):
            if new == 0:
                self.pars.season_width = 0.11
            elif new == 1:
                self.pars.season_width = 1
            self.pars.update_pars('season_width')
            self.sim_source.data = self.simulation_fn(self)
            if self.show_season:
                s = self.panels_fn(self, ['season'])
                self.panel_source['season'].data = s['season']

        if 'all' in button_types or 'season' in button_types:
            self.Season_button = RadioButtonGroup(labels=list(season_labels), active=0)
            self.Season_button.on_change('active', change_season)
            tooltip = Tooltip(content=HTML("""Seasonal or perennial transmission"""), position='top')
            self.buttons['season'] = (self.Season_button, HelpButton(tooltip=tooltip))

        ######################### VACCINATION BUTTON ##################################
        vaccination_labels = {'Liver':0, 'Blood':1, 'Both':2, 'None':3}
        vl = {(True, False):0, (False, True):1, (True, True):2, (False, False):3}
        def change_vaccination(attr, old, new):
            if new == 0:
                self.pars.lsv = True
                self.pars.bsv = False
            elif new == 1:
                self.pars.lsv = False
                self.pars.bsv = True
            elif new == 2:
                self.pars.lsv = True
                self.pars.bsv = True
            else:
                self.pars.lsv = False
                self.pars.bsv = False

            self.pars.update_pars(('lsv', 'bsv'))
            self.sim_source.data = self.simulation_fn(self)
            if self.show_vac:
                s = self.panels_fn(self, ['liver_vac_prot', 'blood_vac_prot', 'min_vac_age'])
                self.panel_source['liver_vac_prot'].data = s['liver_vac_prot']
                self.panel_source['blood_vac_prot'].data = s['blood_vac_prot']
            if self.show_death_rates:
                self.panel_source['min_vac_age'].data = self.panels_fn(self, 'death rates')['min_vac_age']

                self.panel_source['min_vac_age'].data = s['min_vac_age']

        if 'all' in button_types or 'vaccines' in button_types:
            self.Vaccination_button = RadioButtonGroup(labels=list(vaccination_labels), active=vl[(self.pars.lsv, self.pars.bsv)])
            self.Vaccination_button.on_change('active', change_vaccination)
            tooltip = Tooltip(content=HTML("<center>Which vaccines to use</center>"), position='top')
            self.buttons['vaccines'] = (self.Vaccination_button, HelpButton(tooltip=tooltip))

        ######################### SMC ##################################
        smc_labels = {'SMC':0, 'No SMC':1}
        vl = {True:0, False:1}
        def change_smc(attr, old, new):
            if new == 0:
                self.pars.smc = True
            elif new == 1:
                self.pars.smc = False
            self.pars.update_pars('smc')
            self.sim_source.data = self.simulation_fn(self)
            if self.show_vac:
                s = self.panels_fn(self, ['smc'])
                self.panel_source['smc'].data = s['smc']

        if 'all' in button_types or 'smc' in button_types:
            self.SMC_button = RadioButtonGroup(labels=list(smc_labels), active=vl[self.pars.smc])
            self.SMC_button.on_change('active', change_smc)
            tooltip = Tooltip(content=HTML("<center>Use SMC</center>"), position='top')
            self.buttons['smc'] = (self.SMC_button, HelpButton(tooltip=tooltip))

    def Sliders(self, slider_types, value='value'):
        if isinstance(slider_types, str):
            slider_types = [slider_types]

        def update_study_months(attr, old, new):
            self.pars.study_months = new
            self.pars.update_pars('study_months')
            self.sim_source.data = self.simulation_fn(self)
            if self.show_vac or self.show_smc or self.show_season:
                s = self.panels_fn(self, ['liver_vac_prot', 'blood_vac_prot', 'smc', 'season'])
                if self.pars.lsv:
                    self.panel_source['liver_vac_prot'].data = s['liver_vac_prot']
                if self.pars.bsv:
                    self.panel_source['blood_vac_prot'].data = s['blood_vac_prot']
                if self.pars.smc:
                    self.panel_source['smc'].data = s['smc']
                if self.show_season:
                    self.panel_source['season'].data = s['season']
            if self.show_death_rates:
                self.panel_source['death rates'].data = self.panels_fn(self, 'death rates')['death rates']

        def update_λ(attr, old, new):
            self.pars.λ = 10**new
            self.pars.update_pars('λ')
            self.sim_source.data = self.simulation_fn(self)

        def update_β(attr, old, new):
            self.pars.β = new
            self.pars.update_pars('β')
            self.sim_source.data = self.simulation_fn(self)

        def update_season_peak(attr, old, new):
            self.pars.season_peak = new
            self.pars.update_pars('season_peak')
            self.sim_source.data = self.simulation_fn(self)
            if self.show_season:
                self.panel_source['season'].data = self.panels_fn(self, 'season')['season']

        def update_season_width(attr, old, new):
            self.pars.season_width = new
            self.pars.update_pars('season_width')
            self.sim_source.data = self.simulation_fn(self)
            if self.show_season:
                self.panel_source['season'].data = self.panels_fn(self, 'season')['season']

        def update_ρ_elp(attr, old, new):
            self.pars.ρ_elp = new
            self.pars.update_pars('ρ_elp')
            self.sim_source.data = self.simulation_fn(self)

        def update_τ_elp(attr, old, new):
            self.pars.τ_elp = new
            self.pars.update_pars('τ_elp')
            self.sim_source.data = self.simulation_fn(self)

        def update_ρ_clinical(attr, old, new):
            self.pars.ρ_clinical = new
            self.pars.update_pars('ρ_clinical')
            self.sim_source.data = self.simulation_fn(self)

        def update_δ_clinical(attr, old, new):
            self.pars.δ_clinical = new
            self.pars.update_pars('δ_clinical')
            self.sim_source.data = self.simulation_fn(self)

        def update_γ_clinical(attr, old, new):
            self.pars.γ_clinical = new
            self.pars.update_pars('γ_clinical')
            self.sim_source.data = self.simulation_fn(self)

        def update_case_def_clinical(attr, old, new):
            self.pars.case_def_clinical = new
            self.pars.update_pars('case_def_clinical')
            self.sim_source.data = self.simulation_fn(self)

        def update_ρ_severe(attr, old, new):
            self.pars.ρ_severe = new
            self.pars.update_pars('ρ_severe')
            self.sim_source.data = self.simulation_fn(self)

        def update_δ_severe(attr, old, new):
            self.pars.δ_severe = new
            self.pars.update_pars('δ_severe')
            self.sim_source.data = self.simulation_fn(self)

        def update_ε_severe(attr, old, new):
            self.pars.ε_severe = new
            self.pars.update_pars('ε_severe')
            self.sim_source.data = self.simulation_fn(self)

        def update_death_rate_modifier(attr, old, new):
            self.pars.death_rate_modifier = new
            self.pars.update_pars('death_rate_modifier')
            self.sim_source.data = self.simulation_fn(self)
            if self.show_death_rates:
                self.panel_source['death rates'].data = self.panels_fn(self, 'death rates')['death rates']

        def update_min_vac_age(attr, old, new):
            self.pars.min_vac_age = new
            if self.pars.recording == 'Birth':
                self.pars.t_first = 1
            else:
                self.pars.t_first = self.pars.min_vac_age + self.pars.vac_age_range
            self.pars.update_pars('min_vac_age')
            self.sim_source.data = self.simulation_fn(self)
            if self.show_death_rates:
                self.panel_source['min_vac_age'].data = self.panels_fn(self, 'death rates')['min_vac_age']

        def update_vac_age_range(attr, old, new):
            self.pars.vac_age_range = new
            if self.pars.recording == 'Birth':
                self.pars.t_first = 1
            else:
                self.pars.t_first = self.pars.min_vac_age + self.pars.vac_age_range
            self.pars.update_pars('vac_age_range')
            self.sim_source.data = self.simulation_fn(self)
            if self.show_death_rates:
                self.panel_source['min_vac_age'].data = self.panels_fn(self, 'death rates')['min_vac_age']

        def update_nboosters(attr, old, new):
            self.pars.nboosters = new
            self.pars.update_pars('nboosters')
            self.sim_source.data = self.simulation_fn(self)
            if self.show_vac:
                if self.pars.bsv:
                    self.panel_source['blood_vac_prot'].data = self.panels_fn(self, 'blood_vac_prot')['blood_vac_prot']
                if self.pars.lsv:
                    self.panel_source['liver_vac_prot'].data = self.panels_fn(self, 'liver_vac_prot')['liver_vac_prot']

        def update_ν_liver(attr, old, new):
            self.pars.ν_liver = new
            self.pars.update_pars('ν_liver')
            self.sim_source.data = self.simulation_fn(self)
            if self.pars.lsv:
                if self.show_vac:
                    self.panel_source['liver_vac_prot'].data = self.panels_fn(self, 'liver_vac_prot')['liver_vac_prot']

        def update_τ_liver(attr, old, new):
            self.pars.τ_liver = new
            self.pars.update_pars('τ_liver')
            self.sim_source.data = self.simulation_fn(self)
            if self.pars.lsv:
                if self.show_vac:
                    self.panel_source['liver_vac_prot'].data = self.panels_fn(self, 'liver_vac_prot')['liver_vac_prot']

        def update_period_liver(attr, old, new):
            self.pars.period_liver = new
            self.pars.update_pars('period_liver')
            self.sim_source.data = self.simulation_fn(self)
            if self.pars.lsv:
                if self.show_vac:
                    self.panel_source['liver_vac_prot'].data = self.panels_fn(self, 'liver_vac_prot')['liver_vac_prot']

        def update_offset_liver(attr, old, new):
            self.pars.offset_liver = new
            self.pars.update_pars('offset_liver')
            self.sim_source.data = self.simulation_fn(self)
            if self.show_vac and self.pars.lsv:
                self.panel_source['liver_vac_prot'].data = self.panels_fn(self, 'liver_vac_prot')['liver_vac_prot']

        def update_nboosters_liver(attr, old, new):
            self.pars.nboosters_liver = new
            self.pars.update_pars('nboosters_liver')
            self.sim_source.data = self.simulation_fn(self)
            if self.pars.lsv:
                if self.show_vac:
                    self.panel_source['liver_vac_prot'].data = self.panels_fn(self, 'liver_vac_prot')['liver_vac_prot']

        def update_ν_blood(attr, old, new):
            self.pars.ν_blood = new
            self.pars.update_pars('ν_blood')
            self.sim_source.data = self.simulation_fn(self)
            if self.show_vac and self.pars.bsv:
                self.panel_source['blood_vac_prot'].data = self.panels_fn(self, 'blood_vac_prot')['blood_vac_prot']

        def update_τ_blood(attr, old, new):
            self.pars.τ_blood = new
            self.pars.update_pars('τ_blood')
            self.sim_source.data = self.simulation_fn(self)
            if self.show_vac and self.pars.bsv:
                self.panel_source['blood_vac_prot'].data = self.panels_fn(self, 'blood_vac_prot')['blood_vac_prot']

        def update_period_blood(attr, old, new):
            self.pars.period_blood = new
            self.pars.update_pars('period_blood')
            self.sim_source.data = self.simulation_fn(self)
            if self.show_vac and self.pars.bsv:
                self.panel_source['blood_vac_prot'].data = self.panels_fn(self, 'blood_vac_prot')['blood_vac_prot']

        def update_offset_blood(attr, old, new):
            self.pars.offset_blood = new
            self.pars.update_pars('offset_blood')
            self.sim_source.data = self.simulation_fn(self)
            if self.show_vac and self.pars.bsv:
                self.panel_source['blood_vac_prot'].data = self.panels_fn(self, 'blood_vac_prot')['blood_vac_prot']

        def update_nboosters_blood(attr, old, new):
            self.pars.nboosters_blood = new
            self.pars.update_pars('nboosters_blood')
            self.sim_source.data = self.simulation_fn(self)
            if self.show_vac and self.pars.bsv:
                self.panel_source['blood_vac_prot'].data = self.panels_fn(self, 'blood_vac_prot')['blood_vac_prot']

        def update_ω_infection(attr, old, new):
            self.pars.ω_infection = new
            self.pars.update_pars('ω_infection')
            self.sim_source.data = self.simulation_fn(self)

        def update_ω_clinical(attr, old, new):
            self.pars.ω_clinical = new
            self.pars.update_pars('ω_clinical')
            self.sim_source.data = self.simulation_fn(self)

        def update_ω_severe(attr, old, new):
            self.pars.ω_severe = new
            self.pars.update_pars('ω_severe')
            self.sim_source.data = self.simulation_fn(self)

        def update_smc_coverage(attr, old, new):
            self.pars.smc_coverage = new
            self.pars.update_pars('smc_coverage')
            self.sim_source.data = self.simulation_fn(self)
            if self.pars.smc and self.show_smc:
                self.panel_source['smc'].data = self.panels_fn(self, 'smc')['smc']

        def update_smc_offset(attr, old, new):
            self.pars.smc_offset = new
            self.pars.update_pars('smc_offset')
            self.sim_source.data = self.simulation_fn(self)
            if self.pars.smc and self.show_smc:
                self.panel_source['smc'].data = self.panels_fn(self, 'smc')['smc']

        def update_smc_ramp(attr, old, new):
            self.pars.smc_ramp = new
            self.pars.update_pars('smc_ramp')
            self.sim_source.data = self.simulation_fn(self)
            if self.pars.smc and self.show_smc:
                self.panel_source['smc'].data = self.panels_fn(self, 'smc')['smc']

        def update_smc_rounds(attr, old, new):
            self.pars.smc_rounds = new
            self.pars.update_pars('smc_rounds')
            self.sim_source.data = self.simulation_fn(self)
            if self.pars.smc and self.show_smc:
                self.panel_source['smc'].data = self.panels_fn(self, 'smc')['smc']

        def update_smc_repeats(attr, old, new):
            self.pars.smc_repeats = new
            self.pars.update_pars('smc_repeats')
            self.sim_source.data = self.simulation_fn(self)
            if self.pars.smc and self.show_smc:
                self.panel_source['smc'].data = self.panels_fn(self, 'smc')['smc']

        sw = int(175*self.fig_xscale)

        if 'core' in slider_types or 'all' in slider_types or 'study_months' in slider_types:
            self.sl_study_months = Slider(width=sw, start=1, end=50*12, value=self.pars.study_months, step=1, title='Study months', name='study_months')
            self.sl_study_months.on_change(value, update_study_months)
            tooltip = Tooltip(content=HTML("<center>Duration of simulation<br>from last primary dose</center>"), position='left')
            self.sliders['study_months'] = self.sl_study_months, HelpButton(tooltip=tooltip)

        if 'core' in slider_types or 'all' in slider_types or 'λ' in slider_types:
            self.sl_λ = Slider(width=sw, start=np.log10(0.1), end=np.log10(20), value=np.log10(self.pars.λ), step=0.01, format=CustomJSTickFormatter(code="return Math.pow(10, tick).toFixed(2)"), title='Infections per year', name='λ')
            self.sl_λ.on_change(value, update_λ)
            tooltip = Tooltip(content=HTML("<center>Annual average number of blood infections<br>per person per year</center>"), position='left')
            self.sliders['λ'] = self.sl_λ, HelpButton(tooltip=tooltip)

        if 'all' in slider_types or 'β' in slider_types:
            self.sl_β = Slider(width=sw, start=0, end=1, value=self.pars.β, step=0.1, title='Infection heterogeneity', name='β')
            self.sl_β.on_change(value, update_β)
            tooltip = Tooltip(content=HTML("<center>Heterogeneity in bite rate/susceptibility<br>0: homogeneous, 1: maximal heterogeneity<br>from 0 to two times blood infection rate</center>"), position='left')
            self.sliders['β'] = self.sl_β, HelpButton(tooltip=tooltip)

        if 'all' in slider_types or 'season_peak' in slider_types:
            self.sl_season_peak = Slider(width=sw, start=0, end=51, value=self.pars.season_peak, step=1, title='Peak week of malaria season', name='season_peak')
            self.sl_season_peak.on_change(value, update_season_peak)
            tooltip = Tooltip(content=HTML("<center>Peak of the malaria season<br>relative to the last primary dose</center>"), position='left')
            self.sliders['season_peak'] = self.sl_season_peak, HelpButton(tooltip=tooltip)

        if 'all' in slider_types or 'season_width' in slider_types:
            self.sl_season_width = Slider(width=sw, start=0.001, end=1, value=self.pars.season_width, step=0.001, format='0.000', title='Season duration', name='season_width')
            self.sl_season_width.on_change(value, update_season_width)
            tooltip = Tooltip(content=HTML("<center>Measure of season length</center>"), position='left')
            self.sliders['season_width'] = self.sl_season_width, HelpButton(tooltip=tooltip)

        if 'core' in slider_types or 'all' in slider_types or 'ρ_elp' in slider_types:
            self.sl_ρ_elp = Slider(width=sw, start=0, end=1, value=self.pars.ρ_elp, step=.01, title='Early-life protection', name='ρ_elp')
            self.sl_ρ_elp.on_change(value, update_ρ_elp)
            tooltip = Tooltip(content=HTML("<center>Relative protection from<br>blood infection at birth<br>compared to an adult</center>"), position='left')
            self.sliders['ρ_elp'] = self.sl_ρ_elp, HelpButton(tooltip=tooltip)

        if 'core' in slider_types or 'all' in slider_types or 'τ_elp' in slider_types:
            self.sl_τ_elp = Slider(width=sw, start=0, end=5, value=self.pars.τ_elp, step=0.1, title='Early-life protection half-life', name='τ_elp')
            self.sl_τ_elp.on_change(value, update_τ_elp)
            tooltip = Tooltip(content=HTML("<center>Half-life in years of<br>early life protection/center>"), position='left')
            self.sliders['τ_elp'] = self.sl_τ_elp, HelpButton(tooltip=tooltip)

        if 'core' in slider_types or 'all' in slider_types or 'ρ_clinical' in slider_types:
            self.sl_ρ_clinical = Slider(width=sw, start=0, end=1, value=self.pars.ρ_clinical, step=0.01, title='Clinical maximum risk', name='ρ_clinical')
            self.sl_ρ_clinical.on_change(value, update_ρ_clinical)
            tooltip = Tooltip(content=HTML("<center>Relative maximum risk of clinical malaria</center>"), position='left')
            self.sliders['ρ_clinical'] = self.sl_ρ_clinical, HelpButton(tooltip=tooltip)

        if 'core' in slider_types or 'all' in slider_types or 'δ_clinical' in slider_types:
            self.sl_δ_clinical = Slider(width=sw, start=0.001, end=0.1, value=self.pars.δ_clinical, step=0.001, format='0.000', title='Clinical risk decay', name='δ_clinical')
            self.sl_δ_clinical.on_change(value, update_δ_clinical)
            tooltip = Tooltip(content=HTML("<center>Rate of decay in risk of<br>clinical malaria at half-maximum with exposure</center>"), position='left')
            self.sliders['δ_clinical'] = self.sl_δ_clinical, HelpButton(tooltip=tooltip)

        if 'core' in slider_types or 'all' in slider_types or 'γ_clinical' in slider_types:
            self.sl_γ_clinical = Slider(width=sw, start=2, end=200, value=self.pars.γ_clinical, step=1, title='Clinical half-maximum exposure', name='γ_clinical')
            self.sl_γ_clinical.on_change(value, update_γ_clinical)
            tooltip = Tooltip(content=HTML("<center>Number of infections at which<br>clinical risk is half-maximum</center>"), position='left')
            self.sliders['γ_clinical'] = self.sl_γ_clinical, HelpButton(tooltip=tooltip)

        if 'all' in slider_types or 'case_def_clinical' in slider_types:
            self.sl_case_def_clinical = Slider(width=sw, start=0, end=1, value=self.pars.case_def_clinical, step=0.01, title='Proportion meeting case definition', name='case_def_clinical')
            self.sl_case_def_clinical.on_change(value, update_case_def_clinical)
            tooltip = Tooltip(content=HTML("<center>Proportion of fever episodes<br>that meet clinical case definition</center>"), position='left')
            self.sliders['case_def_clinical'] = self.sl_case_def_clinical, HelpButton(tooltip=tooltip)

        if 'core' in slider_types or 'all' in slider_types or 'ρ_severe' in slider_types:
            self.sl_ρ_severe = Slider(width=sw, start=0.0, end=0.2, value=self.pars.ρ_severe, step=0.001, format='0.000', title='Severe maximum risk', name='ρ_severe')
            self.sl_ρ_severe.on_change(value, update_ρ_severe)
            tooltip = Tooltip(content=HTML("<center>Relative maximum risk of severe malaria</center>"), position='left')
            self.sliders['ρ_severe'] = self.sl_ρ_severe, HelpButton(tooltip=tooltip)

        if 'core' in slider_types or 'all' in slider_types or 'δ_severe' in slider_types:
            self.sl_δ_severe = Slider(width=sw, start=0, end=1.0, value=self.pars.δ_severe, step=0.01, title='Severe risk decay (exposure)', name='δ_severe')
            self.sl_δ_severe.on_change(value, update_δ_severe)
            tooltip = Tooltip(content=HTML("<center>Rate of decay in risk of severe<br>malaria with exposure in newborns</center>"), position='left')
            self.sliders['δ_severe'] = self.sl_δ_severe, HelpButton(tooltip=tooltip)

        if 'core' in slider_types or 'all' in slider_types or 'ε_severe' in slider_types:
            self.sl_ε_severe = Slider(width=sw, start=0, end=0.1, value=self.pars.ε_severe, step=0.001, format='0.000', title='Severe risk decay (age)', name='ε_severe')
            self.sl_ε_severe.on_change(value, update_ε_severe)
            tooltip = Tooltip(content=HTML("<center>Decay rate age-modifier<br>in risk of severe malaria</center>"), position='left')
            self.sliders['ε_severe'] = self.sl_ε_severe, HelpButton(tooltip=tooltip)

        if 'all' in slider_types or 'death_rate_modifier' in slider_types:
            self.sl_death_rate_modifier = Slider(width=sw, start=0, end=1, value=self.pars.death_rate_modifier, step=0.01, title='Death rate modifier', name='death_rate_modifier')
            self.sl_death_rate_modifier.on_change(value, update_death_rate_modifier)
            tooltip = Tooltip(content=HTML("Modify severe-malaria associated age-specific death rate<br>0: No age specificity<br>1: As estimated from Reyburn et al. (2005)"), position='left')
            self.sliders['death_rate_modifier'] = self.sl_death_rate_modifier, HelpButton(tooltip=tooltip)

        if 'core' in slider_types or 'all' in slider_types or 'min_vac_age' in slider_types:
            self.sl_min_vac_age = Slider(width=sw, start=0, end=300, value=self.pars.min_vac_age, step=1, title='Min vaccination age (weeks)', name='min_vac_age')
            self.sl_min_vac_age.on_change(value, update_min_vac_age)
            tooltip = Tooltip(content=HTML("<center>Youngest age (in weeks) at which children<br>finish their last primary dose</center>"), position='left')
            self.sliders['min_vac_age'] = self.sl_min_vac_age, HelpButton(tooltip=tooltip)

        if 'core' in slider_types or 'all' in slider_types or 'vac_age_range' in slider_types:
            self.sl_vac_age_range = Slider(width=sw, start=1, end=300, value=self.pars.vac_age_range, step=4, title='Vaccination age range (weeks)', name='vac_age_range')
            self.sl_vac_age_range.on_change(value, update_vac_age_range)
            tooltip = Tooltip(content=HTML("<center>Range of ages (in weeks) of<br>children who are vaccinated</center>"), position='left')
            self.sliders['vac_age_range'] = self.sl_vac_age_range, HelpButton(tooltip=tooltip)

        if 'core' in slider_types or 'all' in slider_types or 'ν_liver' in slider_types:
            self.sl_ν_liver = Slider(width=sw, start=0, end=2, value=self.pars.ν_liver, step=0.01, title='Relative protective efficacy', name='ν_liver')
            self.sl_ν_liver.on_change(value, update_ν_liver)
            tooltip = Tooltip(content=HTML("<center>Factor multiplying the maximum<br>protective effect of a pre-erythrocytic vaccine</center>"), position='left')
            self.sliders['ν_liver'] = self.sl_ν_liver, HelpButton(tooltip=tooltip)

        if 'core' in slider_types or 'all' in slider_types or 'τ_liver' in slider_types:
            self.sl_τ_liver = Slider(width=sw, start=0.1, end=10, value=self.pars.τ_liver, step=0.1, title='Relative half-life', name='τ_liver')
            self.sl_τ_liver.on_change(value, update_τ_liver)
            tooltip = Tooltip(content=HTML("<center>Factor multiplying the half-life<br>of a pre-erythrocytic vaccine</center>"), position='left')
            self.sliders['τ_liver'] = self.sl_τ_liver, HelpButton(tooltip=tooltip)

        if 'all' in slider_types or 'period_liver' in slider_types:
            self.sl_period_liver = Slider(width=sw, start=1, end=24, value=self.pars.period_liver, step=1, title='Booster period (weeks)', name='period_liver')
            self.sl_period_liver.on_change(value, update_period_liver)
            tooltip = Tooltip(content=HTML("<center>Period (weeks) between<br>pre-erythrocytic boosters</center>"), position='left')
            self.sliders['period_liver'] = self.sl_period_liver, HelpButton(tooltip=tooltip)

        if 'all' in slider_types or 'offset_liver' in slider_types:
            self.sl_offset_liver = Slider(width=sw, start=0, end=500, value=self.pars.offset_liver, step=1, title='Offset in weeks', name='offset_liver')
            self.sl_offset_liver.on_change(value, update_offset_liver)
            tooltip = Tooltip(content=HTML("<center>Offset (in weeks) of the last primary<br>pre-erythrocytic vaccine dose</center>"), position='left')
            self.sliders['offset_liver'] = self.sl_offset_liver, HelpButton(tooltip=tooltip)

        if 'core' in slider_types or 'all' in slider_types or 'nboosters_liver' in slider_types:
            self.sl_nboosters_liver = Slider(width=sw, start=0, end=20, value=self.pars.nboosters_liver, step=1, title='No. of boosters', name='nboosters_liver')
            self.sl_nboosters_liver.on_change(value, update_nboosters_liver)
            tooltip = Tooltip(content=HTML("<center>Number of pre-erythrocytic<br>vaccine boosters</center>"), position='left')
            self.sliders['nboosters_liver'] = self.sl_nboosters_liver, HelpButton(tooltip=tooltip)

        if 'all' in slider_types or 'ν_blood' in slider_types:
            self.sl_ν_blood = Slider(width=sw, start=0, end=1, value=self.pars.ν_blood, step=0.01, title='Relative protective efficacy', name='ν_blood')
            self.sl_ν_blood.on_change(value, update_ν_blood)
            tooltip = Tooltip(content=HTML("<center>Factor multiplying the maximum<br>protective effect of a blood-stage vaccine vaccine</center>"), position='left')
            self.sliders['ν_blood'] = self.sl_ν_blood, HelpButton(tooltip=tooltip)

        if 'core' in slider_types or 'all' in slider_types or 'τ_blood' in slider_types:
            self.sl_τ_blood = Slider(width=sw, start=0.1, end=10, value=self.pars.τ_blood, step=0.1, title='Relative half-life', name='τ_blood')
            self.sl_τ_blood.on_change(value, update_τ_blood)
            tooltip = Tooltip(content=HTML("<center>Factor multiplying the half-life<br>of a blood-stage vaccine</center>"), position='left')
            self.sliders['τ_blood'] = self.sl_τ_blood, HelpButton(tooltip=tooltip)

        if 'all' in slider_types or 'period_blood' in slider_types:
            self.sl_period_blood = Slider(width=sw, start=1, end=24, value=self.pars.period_blood, step=1, title='Booster period (weeks)', name='period_blood')
            self.sl_period_blood.on_change(value, update_period_blood)
            tooltip = Tooltip(content=HTML("<center>Period (weeks) between<br>blood-stage boosters</center>"), position='left')
            self.sliders['period_blood'] = self.sl_period_blood, HelpButton(tooltip=tooltip)

        if 'all' in slider_types or 'offset_blood' in slider_types:
            self.sl_offset_blood = Slider(width=sw, start=0, end=500, value=self.pars.offset_blood, step=1, title='Offset in weeks', name='offset_blood')
            self.sl_offset_blood.on_change(value, update_offset_blood)
            tooltip = Tooltip(content=HTML("<center>Offset (in weeks) of the last primary<br>blood-stage vaccine dose</center>"), position='left')
            self.sliders['offset_blood'] = self.sl_offset_blood, HelpButton(tooltip=tooltip)

        if 'core' in slider_types or 'all' in slider_types or 'nboosters_blood' in slider_types:
            self.sl_nboosters_blood = Slider(width=sw, start=0, end=20, value=self.pars.nboosters_blood, step=1, title='No. of boosters', name='nboosters_blood')
            self.sl_nboosters_blood.on_change(value, update_nboosters_blood)
            tooltip = Tooltip(content=HTML("<center>Number of blood-stage<br>vaccine boosters</center>"), position='left')
            self.sliders['nboosters_blood'] = self.sl_nboosters_blood, HelpButton(tooltip=tooltip)

        if 'core' in slider_types or 'all' in slider_types or 'ω_infection' in slider_types:
            self.sl_ω_infection = Slider(width=sw, start=0, end=1, value=self.pars.ω_infection, step=0.01, title='Relative infection protection', name='ω_infection')
            self.sl_ω_infection.on_change(value, update_ω_infection)
            tooltip = Tooltip(content=HTML("<center>Relative protective effect<br>against blood infection</center>"), position='left')
            self.sliders['ω_infection'] = self.sl_ω_infection, HelpButton(tooltip=tooltip)

        if 'core' in slider_types or 'all' in slider_types or 'ω_clinical' in slider_types:
            self.sl_ω_clinical = Slider(width=sw, start=0, end=1, value=self.pars.ω_clinical, step=0.01, title='Relative clinical protection', name='ω_clinical')
            self.sl_ω_clinical.on_change(value, update_ω_clinical)
            tooltip = Tooltip(content=HTML("<center>Relative protective effect<br>against clinical malaria</center>"), position='left')
            self.sliders['ω_clinical'] = self.sl_ω_clinical, HelpButton(tooltip=tooltip)

        if 'core' in slider_types or 'all' in slider_types or 'ω_severe' in slider_types:
            self.sl_ω_severe = Slider(width=sw, start=0, end=1, value=self.pars.ω_severe, step=0.01, title='Relative severe protection', name='ω_severe')
            self.sl_ω_severe.on_change(value, update_ω_severe)
            tooltip = Tooltip(content=HTML("<center>Relative protective effect<br>against severe malaria</center>"), position='left')
            self.sliders['ω_severe'] = self.sl_ω_severe, HelpButton(tooltip=tooltip)

        if 'core' in slider_types or 'all' in slider_types or 'smc_coverage' in slider_types:
            self.sl_smc_coverage = Slider(width=sw, start=0, end=1, value=self.pars.smc_coverage, step=0.01, title='SMC coverage', name='smc_coverage')
            self.sl_smc_coverage.on_change(value, update_smc_coverage)
            tooltip = Tooltip(content=HTML("<center>Relative protective<br>effect of SMC in the cohort</center>"), position='left')
            self.sliders['smc_coverage'] = self.sl_smc_coverage, HelpButton(tooltip=tooltip)

        if 'core' in slider_types or 'all' in slider_types or 'smc_offset' in slider_types:
            self.sl_smc_offset = Slider(width=sw, start=-52, end=52, value=self.pars.smc_offset, step=1, title='SMC offset in weeks', name='smc_offset')
            self.sl_smc_offset.on_change(value, update_smc_offset)
            tooltip = Tooltip(content=HTML("<center>Offset (in weeks) of the first SMC dose<br>to the last primary vaccine dose</center>"), position='left')
            self.sliders['smc_offset'] = self.sl_smc_offset, HelpButton(tooltip=tooltip)

        if 'all' in slider_types or 'smc_ramp' in slider_types:
            self.sl_smc_ramp = Slider(width=sw, start=0, end=1, value=self.pars.smc_ramp, step=0.01, title='SMC rate of ramp-up', name='smc_ramp')
            self.sl_smc_ramp.on_change(value, update_smc_ramp)
            tooltip = Tooltip(content=HTML("<center>Ramping-up duration (weeks) of<br>the protective effect of SMC in the cohort</center>"), position='left')
            self.sliders['smc_ramp'] = self.sl_smc_ramp, HelpButton(tooltip=tooltip)

        if 'core' in slider_types or 'all' in slider_types or 'smc_rounds' in slider_types:
            self.sl_smc_rounds = Slider(width=sw, start=0, end=12, value=self.pars.smc_rounds, step=1, title='SMC rounds', name='smc_rounds')
            self.sl_smc_rounds.on_change(value, update_smc_rounds)
            tooltip = Tooltip(content=HTML("<center>Number of rounds of SMC<br>in a season (1 round given monthly)</center>"), position='left')
            self.sliders['smc_rounds'] = self.sl_smc_rounds, HelpButton(tooltip=tooltip)

        if 'core' in slider_types or 'all' in slider_types or 'smc_repeats' in slider_types:
            self.sl_smc_repeats = Slider(width=sw, start=0, end=20, value=self.pars.smc_repeats, step=1, title='SMC repeats', name='smc_repeats')
            self.sl_smc_repeats.on_change(value, update_smc_repeats)
            tooltip = Tooltip(content=HTML("<center>Number of years SMC is given</center>"), position='left')
            self.sliders['smc_repeats'] = self.sl_smc_repeats, HelpButton(tooltip=tooltip)
