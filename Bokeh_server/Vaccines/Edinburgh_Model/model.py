from Edinburgh_Model.vaccination_types import *

import numpy as np
from pandas import read_csv
from numpy import exp, log, ma
from scipy.stats import vonmises, triang
from datetime import datetime

variables = ['All infection', 'First infection', 'All clinical', 'First clinical', 'Severe malaria', 'Direct deaths', 'Indirect deaths', 'All deaths']
measures = ['cases', 'cdf', 'efficacy', 'averted']
weeks_per_month = 52./12.

class Model:
    def __init__(self):
        self.Config()
        self.Sources()
        self.data = None

    def Sources(self, simulation_fn=None, data_fn=None, panels_fn=None):
        """ set the sources of information for the plots """
        self.simulation_fn = simulation_fn
        self.data_fn = data_fn
        self.panels_fn = panels_fn

    def Config(self, show_smc=False, show_vac=False, show_season=False,
               show_death_rates=False, time_scale='Years',
               show_pre_vac=False, fig_xscale=1, fig_yscale=1, plotfile='', miscellaneous={},
               time_warning=0, logfile=False):
        assert time_warning >= 0, f'time_warning must be non-negative, got {time_warning}'

        assert isinstance(show_smc, bool), 'show_smc must be boolean'
        assert isinstance(show_vac, bool), 'show_vac must be boolean'
        assert isinstance(show_season, bool), 'show_season must be boolean'
        assert isinstance(show_death_rates, bool), 'show_death_rates must be boolean'
        assert time_scale in ['Years', 'Months', 'Weeks', 'Days'], f'time_scale must be Years, Months, Weeks or Days, got {time_scale}'
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


class Parameters:
    def __init__(self,
            # demographics
            children = 'cohort',                  # children are modelled as a 'cohort' or as a 'population'
            popsize = 100000,                     # number of fully vaccinated children at last primary dose
            study_months = 12,                    # number of months starting from last primary dose

            # blood infection rate
            λ = 1,                                # blood infections per child per year
            β = 0,                                # scale of heterogeneity of bite rate/susceptibility in the population, 0 is homogeneous, 1 is maximum heterogeneous (from Smith 2005)

            # seasonality
            season_peak = 0,                      # The peak of the season is at week season_peak relative to last primary vaccination
            season_width = 1,                     # Measure of duration of malaria season

            # early life protection
            ρ_elp = 0.71,                         # early life relative protection from infection at birth (from Trape 2024)
            τ_elp = 1.7,                          # half-life of early life protection against infection (years) (from Trape 2024)

            # clinical risk
            ρ_clinical = 0.79,                    # risk a blood infection becomes clinical on first infection (from Goncalves 2014 and Trape 2024)
            δ_clinical = 0.033,                   # rate of development of clinical immunity with number of infections (from Trape 2024)
            γ_clinical = 57,                      # half-maximum-risk exposure of clinical malaria (from Trape 2024)
            case_def_clinical = 1,                # proportion of clinical cases that meet case definition (eg Pr(>5000 parasites/μl given fever))

            # severe risk
            ρ_severe = 0.042,                     # risk of severe malaria on first infection (from Goncalves 2014)
            δ_severe = 0.12,                      # decay rate of risk of severe malaria with exposure (from Goncalves 2014)
            ε_severe = 0.05,                      # decay rate of risk of severe malaria with age (from Abduallah 2007)

            # deaths
            deaths = 'Reyburn',                   # age-specfic death rate function: 'Reyburn' or 'Flat'
            death_rate_multiplier = 1.,           # multiplier for death rate

            # general vaccination
            min_vac_age = 0,                      # minimum age at last primary dose (weeks)
            vac_age_range = 1,                    # age range of vaccinees (months)
            nboosters = None,                     # number of liver and blood vaccine boosters, if None then use vaccine specific nboosters

            # liver vaccination
            lsv = False,                          # whether liver stage vaccination is used
            vac_profile_liver = 'Imperial R21',   # liver vaccine profile function: 'Imperial RTS,S', 'Imperial R21', 'OpenMalaria', 'exponential' or 'ITN'
            ν_liver = 1,                          # scaling factor of vaccine liver stage profile
            τ_liver = 1,                          # half-life multiplier of liver stage vaccine
            period_liver = 1,                     # period between boosters (years)
            offset_liver = 0,                     # number of weeks to delay liver vaccination relative to last primary dose of blood stage vaccination
            nboosters_liver = 0,                  # number of boosters

            # blood vaccination
            bsv = False,                          # whether blood stage vaccination is used
            vac_profile_blood = 'RH5 delayed',    # blood vaccine profile function: 'RH5_Monthly' or 'RH5_Delayed'
            ν_blood = 1,                          # scaling factor of vaccine blood stage profile
            τ_blood = 1,                          # half-life multiplier of blood stage vaccine
            period_blood = 1 ,                    # period between boosters (years)
            offset_blood = 0,                     # number of weeks to delay blood vaccination relative to last primary dose of liver stage vaccination
            nboosters_blood = 0,                  # number of boosters
            protection_blood = False,             # None: ω_infection, ω_clinical and ω_severe are set independently (default values as for False),
                                                  # True: set ω_infection=0.43, ω_clinical=0 and ω_severe=0.29
                                                  # False: set ω_infection=0, ω_clinical=0.5 and ω_severe=0.29
            ω_infection = 0.0,                    # proportion of blood infections that a blood-stage vaccine prevents
            ω_clinical = 0.5,                     # proportion of blood infections that a blood-stage vaccine prevents from becoming clinical
            ω_severe = 0.29,                      # proportion of clinical cases that a blood-stage vaccine prevents from becoming severe

            # seasonal malaria chemoprevention
            smc_control = False,                  # whether the control cohort uses SMC
            smc = False,                          # whether seasonal malaria chemoprevention is used
            smc_coverage = 0.9,                   # coverage of SMC
            smc_offset = -4,                      # number of weeks to delay SMC relative to last primary dose of first intervention
            smc_ramp = 0.24,                      # ramp up  effectiveness of SMC (per week)
            smc_rounds = 4,                       # duration of SMC in months, one dose each month
            smc_repeats = 4,                      # number of annual repeats of SMC

            # miscellaneous
            control = 'control',                  # name of the control cohort
            treatment = 'Vaccine',                # name of the treatment cohort
            efficacy = 'Cumulative',              # how efficacy is calculated: 'Cumulative', 'Weekly'
            recording = 'Last primary dose',      # from when to record cumulative cases for vaccine efficacy: 'Birth' or 'Last primary dose'
        ):
        self.children = children
        self.popsize = popsize
        self.study_months = study_months

        self.β = β
        self.λ = λ

        self.season_peak = season_peak
        self.season_width = season_width

        self.ρ_elp = ρ_elp
        self.τ_elp = τ_elp

        self.ρ_clinical = ρ_clinical
        self.δ_clinical = δ_clinical
        self.γ_clinical = γ_clinical
        self.case_def_clinical = case_def_clinical

        self.ρ_severe = ρ_severe
        self.δ_severe = δ_severe
        self.ε_severe = ε_severe

        self.deaths = deaths
        self.death_rate_multiplier = death_rate_multiplier

        self.min_vac_age = min_vac_age
        self.vac_age_range = vac_age_range
        self.nboosters = nboosters

        self.lsv = lsv
        self.vac_profile_liver = vac_profile_liver
        self.ν_liver = ν_liver
        self.τ_liver = τ_liver
        self.period_liver = period_liver
        self.offset_liver = offset_liver
        self.nboosters_liver = nboosters_liver

        self.bsv = bsv
        self.vac_profile_blood = vac_profile_blood
        self.ν_blood = ν_blood
        self.τ_blood = τ_blood
        self.period_blood = period_blood
        self.offset_blood = offset_blood
        self.nboosters_blood = nboosters_blood
        self.protection_blood = protection_blood
        self.ω_infection = ω_infection
        self.ω_clinical = ω_clinical
        self.ω_severe = ω_severe

        self.smc_control = smc_control
        self.smc = smc
        self.smc_coverage = smc_coverage
        self.smc_offset = smc_offset
        self.smc_ramp = smc_ramp
        self.smc_rounds = smc_rounds
        self.smc_repeats = smc_repeats

        self.control = control
        self.treatment = treatment
        self.efficacy = efficacy
        self.recording = recording

        if lsv == False and bsv == False and vac_age_range != 1:
            print('Warning: no vaccination set, do you want vac_age_range != 1?')

        # age-specific all-cause death rates from Tanzania 2019 life tables
        # ages in weeks
        ages = [0, 52, 260, 520, 780, 1040, 1300, 1560, 1820, 2080, 2340, 2600, 2860, 3120, 3380, 3640, 3900, 4160, 4420, 5200]
        # corresponding death rates per week
        μ = [7.15e-04, 7.26e-05, 3.07e-05, 1.33e-05, 2.14e-05, 2.96e-05, 3.86e-05, 5.19e-05, 7.04e-05, 9.77e-05, 1.31e-04, 1.85e-04, 2.53e-04, 3.83e-04, 5.60e-04, 8.80e-04, 1.34e-03, 2.08e-03, 3.48e-03, 1e-2]
        self.μ = np.interp(np.arange(ages[-1]+1), ages, μ)
        self.survival = 1 - self.μ

        # parameter validation and setup all ancillary parameters and arrays
        self.update_pars('all')

    def update_pars(self, update):
        assert isinstance(update, (str, list, tuple)), f'update_pars() must be called with string, list or tuple, got {update}'
        if isinstance(update, str):
            update_set = set((update,))
        else:
            update_set = set(update)

        assert isinstance(self.study_months, int) and self.study_months > 0 and self.study_months < 98*52, f'study_months must be a positive integer and less than 98 years, got {self.study_months}'
        assert isinstance(self.min_vac_age, int) and self.min_vac_age >= 0, f'min_vac_age must be non-negative integer, got {self.min_vac_age}'
        assert isinstance(self.vac_age_range, int) and self.vac_age_range > 0, f'vac_age_range must be positive integer, got {self.vac_age_range}'

        self.max_vac_age = self.min_vac_age + self.vac_age_range
        # setup arrays of age and exposure
        maxweeks = self.study_months*weeks_per_month + 2*self.min_vac_age + self.vac_age_range+20
        _a = np.arange(maxweeks, dtype=float) # age
        _n = _a[1:] # exposure


        if set(('all', 'λ')).intersection(update_set):
            # convert λ to weekly rate
            assert self.λ > 0, f'λ must be positive, got {self.λ}'

            self.Λ = self.λ / 52

        if set(('all', 'season_width', 'season_peak')).intersection(update_set):
            # seasonal variation in force of infection
            assert 0 <= self.season_peak < 52 and isinstance(self.season_peak, int), f'season_peak must be an integer from 0 to 51, got {self.season_peak}'
            assert 0 < self.season_width <= 1, f'season_width must be 0 to 1, got {self.season_width}'

            if self.season_width == 0:
                # no seasonality
                self.malaria_season = np.ones(52)
            else:
                # seasonality modelled as a von Mises distribution scaled to 52 weeks
                # with peak at week season_peak relative to last primary vaccination
                x = np.arange(-np.pi, np.pi, 2*np.pi/52)
                self.malaria_season = vonmises.pdf(x, kappa=-log(self.season_width), loc=2*np.pi*(self.season_peak-26)/52)
                self.malaria_season *= 52 / self.malaria_season.sum()

        if set(('all', 'min_vac_age', 'vac_age_range', 'recording')).intersection(update_set):
            # record cases from birth or from end of primary vaccination or from a given age (in weeks)
            assert self.recording in ['Birth', 'Last primary dose'] or (isinstance(self.recording, int) and self.recording > 0), f'recording must be Birth, Last primary dose or positive integer, got {self.recording}'

            if self.recording == 'Birth':
                self.t_start_counting = 1
            elif self.recording == 'Last primary dose':
                self.t_start_counting = self.max_vac_age
            elif isinstance(self.recording, int) and self.recording > 0:
                self.t_start_counting = self.recording
            else:
                raise NotImplementedError("For recording use 'Birth', 'Last primary dose' or an integer")

        if set(('all', 'min_vac_age', 'vac_age_range', 'study_months', 'ρ_elp', 'τ_elp')).intersection(update_set):
            assert 0 <= self.ρ_elp <= 1, f'ρ_elp must be 0 to 1, got {self.ρ_elp}'
            assert self.τ_elp >= 0, f'τ_elp must be non-negative, got {self.τ_elp}'

            if self.τ_elp == 0:
                # no early life protection from infection
                self.elp = np.zeros_like(_a)
            else:
                # exponentially decaying early life protection from infection
                self.elp = self.ρ_elp * exp(-log(2)*_a/(52 * self.τ_elp))

        if set(('all', 'min_vac_age', 'vac_age_range', 'study_months', 'children', 'popsize')).intersection(update_set):
            # if counting all children born during the study then divide by the number of weeks in the study and only vaccinate at a single age
            assert self.children in ['cohort', 'population'], f'children must be cohort or population, got {self.children}'
            assert self.popsize > 0, f'popsize must be positive, got {self.popsize}'

            self.cohort_size_at_birth = self.popsize
            if self.children == 'population':
                self.cohort_size_at_birth /= (self.study_months * weeks_per_month)
                self.vac_age_range = 1

            # calculate cohort_size_at_birth for natural death to achieve "popsize" fully vaccinated children
            # this doesn't account for malaria-associated deaths before vaccination so the actual number of
            # fully vaccinated children will be slightly less than popsize
            prod = 1
            for age in range(self.min_vac_age + self.vac_age_range-1, self.min_vac_age, -1):
                prod = 1 + self.survival[age]*prod
            for age in range(self.min_vac_age):
                prod = self.survival[age]*prod

            self.cohort_size_at_birth /= prod

        if set(('all', 'min_vac_age', 'vac_age_range', 'study_months', 'ρ_clinical', 'δ_clinical', 'γ_clinical')).intersection(update_set):
            # risk of clinical symptoms with exposure
            assert 0 <= self.ρ_clinical <= 1, f'ρ_clinical must be 0 to 1, got {self.ρ_clinical}'
            assert self.δ_clinical > 0, f'δ_clinical must be positive, got {self.δ_clinical}'
            assert self.γ_clinical >= 2, f'γ_clinical must greater or equal to 2, got {self.γ_clinical}'

            self.clinical = self.ρ_clinical * (exp(self.δ_clinical*(self.γ_clinical-1)) - 1) / (exp(self.δ_clinical*(self.γ_clinical-1)) - 2 + exp(self.δ_clinical*(_n-1)))

        if set(('all', 'min_vac_age', 'vac_age_range', 'study_months', 'ρ_severe', 'δ_severe', 'ε_severe')).intersection(update_set):
            # risk of severe malaria with age-dependent exposure
            assert 0 <= self.ρ_severe <= 1, f'ρ_severe must be 0 to 1, got {self.ρ_severe}'
            assert self.δ_severe >= 0, f'δ_severe must be non-negative, got {self.δ_severe}'
            assert self.ε_severe >= 0, f'ε_severe must be non-negative, got {self.ε_severe}'

            self.severe_risk = self.ρ_severe * exp(-(self.δ_severe + self.ε_severe / 52. * _a[np.newaxis, :]) * (_n[:, np.newaxis]-1))

        if set(('all', 'min_vac_age', 'vac_age_range', 'study_months', 'deaths', 'death_rate_multiplier')).intersection(update_set):
            assert self.deaths in ['Reyburn', 'Flat'], f'deaths must be Reyburn or Flat, got {self.deaths}'
            assert self.death_rate_multiplier > 0, f'death_rate_multiplier must be positive, got {self.death_rate_multiplier}'

            if self.deaths == 'Reyburn':
                # this doesn't quite work when the study months is less than 8 years
                direct_death_rate_by_age = [0.082, 0.063, 0.051, 0.044, 0.046, 0.06, 0.086, 0.114, 0.137, 0.148, 0.148]
                if _a[-1] < len(direct_death_rate_by_age)*52:
                    self.direct_deaths = self.death_rate_multiplier*np.interp(_a, 52*_a[:len(direct_death_rate_by_age)], direct_death_rate_by_age)
                else:
                    self.direct_deaths = self.death_rate_multiplier*np.interp(_a, np.concatenate((52*_a[:len(direct_death_rate_by_age)-1], _a[-2:-1])), direct_death_rate_by_age)
            elif self.deaths == 'Flat':
                self.direct_deaths = 0.1*np.ones_like(_a)
            else:
                raise NotImplementedError("deaths must be Reyburn or Flat")
            # indirect death parameters of clinical cases from Ross et al 2006 model 2
            # QD = 0.019
            # aF = 0.012 * 52
            # self.indirect_deaths = QD / (1 + (a/aF))

        if set(('all', 'vac_profile_liver')).intersection(update_set):
            assert self.vac_profile_liver in ['Imperial RTS,S', 'Imperial R21', 'OpenMalaria', 'ITN', 'exponential'], f'vac_profile_liver must be Imperial RTS,S, Imperial R21, OpenMalaria, exponential or ITN, got {self.vac_profile_liver}'

            if self.vac_profile_liver == 'Imperial RTS,S':
                self.liver_vac_fn = Imperial_RTSS_vaccination
            elif self.vac_profile_liver == 'Imperial R21':
                self.liver_vac_fn = Imperial_R21_vaccination
            elif self.vac_profile_liver == 'OpenMalaria':
                self.liver_vac_fn = OpenMalaria_vaccination
            elif self.vac_profile_liver == 'ITN':
                self.liver_vac_fn = itn
            else:
                self.liver_vac_fn = exponential_vaccination

        if set(('all', 'vac_profile_blood')).intersection(update_set):
            assert self.vac_profile_blood in ['RH5 monthly', 'RH5 delayed', 'exponential'], f'vac_profile_blood must be RH5 monthly, RH5 delayed or exponential, got {self.vac_profile_blood}'

            if self.vac_profile_blood == 'RH5 monthly':
                self.blood_vac_fn = RH5_Monthly_vaccination
                # scaling parameters for proxy antibody protection
                self.α_clinical = 0.0045
                self.α_infection = 0.0045
                self.α_severe = 0.003
            elif self.vac_profile_blood == 'RH5 delayed':
                self.blood_vac_fn = RH5_Delayed_vaccination
                # scaling parameters for proxy antibody protection
                self.α_clinical = 0.002
                self.α_infection = 0.002
                self.α_severe = 0.0013
            else:
                self.blood_vac_fn = exponential_vaccination
                self.α_clinical = 1
                self.α_infection = 1
                self.α_severe = 1

        if set(('all', 'protection_blood')).intersection(update_set):
            assert self.protection_blood is None or isinstance(self.protection_blood, bool), f'protection_blood must be None or boolean, got {self.protection_blood}'

            if self.protection_blood is True:
                self.ω_infection = 0.43
                self.ω_clinical = 0.0
                self.ω_severe = 0.29
            elif self.protection_blood is False:
                self.ω_infection = 0.0
                self.ω_clinical = 0.5
                self.ω_severe = 0.29
            elif self.protection_blood is None:
                # use the values already set for ω_infection, ω_clinical and ω_severe
                pass
            else:
                raise NotImplementedError("protection_blood must be True, False or None")

        if set(('all', 'nboosters')).intersection(update_set):
            assert self.nboosters is None or isinstance(self.nboosters, int) and self.nboosters >= 0, f'nboosters must be None or non-negative integer, got {self.nboosters}'

            if self.nboosters is not None:
                # if nboosters is given then use this for both liver and blood boosters and SMC if they are not given
                if self.nboosters_liver == 0:
                    self.nboosters_liver = self.nboosters
                if self.nboosters_blood == 0:
                    self.nboosters_blood = self.nboosters
                if self.smc_repeats == 0:
                    self.smc_repeats = self.nboosters

        if set(('all', 'min_vac_age', 'vac_age_range', 'study_months', 'lsv', 'vac_profile_liver', 'ν_liver', 'τ_liver', 'period_liver', 'nboosters_liver', 'nboosters')).intersection(update_set):
            # initialise liver-stage vaccine protection
            assert isinstance(self.lsv, bool), f'lsv must be boolean, got {self.lsv}'
            assert self.ν_liver >= 0, f'ν_liver must be non-negative, got {self.ν_liver}'
            assert self.τ_liver > 0, f'τ_liver must be positive, got {self.τ_liver}'
            assert self.period_liver > 0, f'period_liver must be positive, got {self.period_liver}'
            assert self.nboosters_liver is None or isinstance(self.nboosters_liver, int) and self.nboosters_liver >= 0, f'nboosters_liver must be None or non-negative integer, got {self.nboosters_liver}'

            if self.lsv:
                liver_vac_args = self.ν_liver, self.τ_liver, self.period_liver, self.nboosters_liver
                self.liver_vac = self.liver_vac_fn(_a, liver_vac_args)

        if set(('all', 'min_vac_age', 'vac_age_range', 'study_months', 'bsv', 'vac_profile_blood', 'ν_blood', 'τ_blood', 'period_blood', 'nboosters_blood', 'nboosters')).intersection(update_set):
            # initialise blood-stage vaccine protection
            assert isinstance(self.bsv, bool), f'bsv must be boolean, got {self.bsv}'
            assert self.ν_blood > 0, f'ν_blood must be positive, got {self.ν_blood}'
            assert self.τ_blood > 0, f'τ_blood must be positive, got {self.τ_blood}'
            assert self.period_blood > 0, f'period_blood must be positive, got {self.period_blood}'
            assert self.nboosters_blood is None or isinstance(self.nboosters_blood, int) and self.nboosters_blood >= 0, f'nboosters_blood must be None or non-negative integer, got {self.nboosters_blood}'

            if self.bsv:
                blood_vac_args = self.ν_blood, self.τ_blood, self.period_blood, self.nboosters_blood
                self.blood_vac = self.blood_vac_fn(_a, blood_vac_args)

        if set(('all', 'min_vac_age', 'vac_age_range', 'study_months', 'smc', 'smc_ramp', 'smc_rounds', 'smc_repeats', 'nboosters')).intersection(update_set):
            # initialise seasonal malaria chemoprevention protection
            assert isinstance(self.smc, bool), f'smc must be boolean, got {self.smc}'
            assert 0 <= self.smc_coverage <= 1, f'smc_coverage must be 0 to 1, got {self.smc_coverage}'
            assert self.smc_ramp >= 0, f'smc_ramp must be non-negative, got {self.smc_ramp}'
            assert self.smc_rounds > 0, f'smc_rounds must be positive, got {self.smc_rounds}'
            assert isinstance(self.smc_repeats, int) and self.smc_repeats >= 0, f'smc_repeats must be non-negative integer, got {self.smc_repeats}'

            if self.smc:
                smc_args = self.smc_ramp, self.smc_rounds, self.smc_repeats
                self.smc_protection = smc_protection(_a, smc_args)

        if set(('all', 'case_def_clinical')).intersection(update_set):
            assert 0 <= self.case_def_clinical <= 1, f'case_def_clinical must be 0 to 1, got {self.case_def_clinical}'

        if set(('all', 'β')).intersection(update_set):
            assert 0 <= self.β <= 1, f'β must be 0 to 1, got {self.β}'

        if set(('all', 'offset_liver')).intersection(update_set):
            assert isinstance(self.offset_liver, int), f'offset_liver must be an integer, got {self.offset_liver}'

        if set(('all', 'offset_blood')).intersection(update_set):
            assert isinstance(self.offset_blood, int), f'offset_blood must be an integer, got {self.offset_blood}'

        if set(('all', 'ω_infection')).intersection(update_set):
            assert 0 <= self.ω_infection <= 1, f'ω_infection must be 0 to 1, got {self.ω_infection}'

        if set(('all', 'ω_clinical')).intersection(update_set):
            assert 0 <= self.ω_clinical <= 1, f'ω_clinical must be 0 to 1, got {self.ω_clinical}'

        if set(('all', 'ω_severe')).intersection(update_set):
            assert 0 <= self.ω_severe <= 1, f'ω_severe must be 0 to 1, got {self.ω_severe}'

        if set(('all', 'smc_control')).intersection(update_set):
            assert isinstance(self.smc_control, bool), f'smc_control must be boolean, got {self.smc_control}'

        if set(('all', 'smc_offset')).intersection(update_set):
            assert isinstance(self.smc_offset, int), f'smc_offset must be an integer, got {self.smc_offset}'

        if set(('all', 'control')).intersection(update_set):
            assert self.control, f'control must not be None, False or empty'
            # assert isinstance(self.control, str), f'control must be string, got {self.control}'

        if set(('all', 'treatment')).intersection(update_set):
            assert self.treatment, f'treatment must not be None, False or empty'
            # assert isinstance(self.treatment, str), f'treatment must be string, got {self.treatment}'

        if set(('all', 'efficacy')).intersection(update_set):
            assert self.efficacy in ['Cumulative', 'Weekly', 'both'], f'efficacy must be Cumulative, Weekly or both, got {self.efficacy}'


class Data:
    """ For any data we wish to plot in the graphs """
    def __init__(self, case_def='primary', seasonality='perennial', datafile=None):
        assert seasonality in ['perennial', 'seasonal'], f'seasonality must be perennial or seasonal, got {seasonality}'
        assert datafile is None or isinstance(datafile, str), 'datafile must be string or None'
        self.case_def = case_def
        self.seasonality = seasonality
        self.datafile = datafile
        if datafile:
            # read in vaccination data
            self.dataframe = read_csv(datafile)


class Variables:
    def __init__(self, display_vars=[]):
        assert isinstance(display_vars, (tuple, list)), 'display_vars must be a list or tuple'
        for var in display_vars:
            assert isinstance(var, str), f'all variables must be strings, got {type(var)}'
            assert var in variables, f'variables must be from {variables}, got {var}'
        self.display_vars = display_vars


class Measures:
    def __init__(self, display_measures=[]):
        assert isinstance(display_measures, (tuple, list)), 'display_measures must be a list or tuple'
        for measure in display_measures:
            assert isinstance(measure, str), f'all measures must be strings, got {type(measure)}'
            assert measure in measures, f'measures must be from {measures}, got {measure}'
        self.display_measures = display_measures


class Cohort:
    def __init__(self, duration, pars, t_record_first=None, max_bite_bins=15, time_warning=False):
        """ initial chort of children across a range of age classes
            children are vaccinated ages min_vac_age to minvac_age+vac_age_range-1
        """
        assert isinstance(duration, int) and duration > 0, f'duration must be positive integer, got {duration}'
        assert isinstance(max_bite_bins, int) and max_bite_bins > 0, f'max_bite_bins must be positive integer, got {max_bite_bins}'
        assert t_record_first is None or (isinstance(t_record_first, int) and t_record_first >= 0), f't_record_first must be non-negative integer or None, got {t_record_first}'
        assert isinstance(time_warning, int), f'time_warning must be a non-negative integer got {time_warning}'

        self.minage = 0
        self.size = pars.cohort_size_at_birth # cohort size at birth (adjusted to maintain "popsize" fully vaccinated children, eg 100,000)
        self.min_vac_age = pars.min_vac_age
        self.max_vac_age = pars.max_vac_age
        self.age_classes = pars.vac_age_range
        self.max_bite_bins = max_bite_bins
        self.β = pars.β
        self.bite_rates, self.p_bite_rates = self.bite_rate_pmf(pars.β, max_bite_bins)
        self.nbr = len(self.p_bite_rates)
        self.age_liver_vac = None
        self.age_blood_vac = None
        self.age_smc = None

        # an array of bite rates by age class which modifes the probability of infection each week
        # can initialise this here because it doesn't change with time
        self.Lmod = self.bite_rates[:, np.newaxis] * (1 - pars.elp[np.newaxis, :])

        # min and max cumulative sum of infection rates used for calculating range of exposures to update
        self.sum_L_min = 0
        self.sum_L_max = 0

        # the maximum number of possible infections (ie at most one infection per child per week)
        max_infections = duration

        # the number of children by, axis 0: exposures, axis 1: bite rate, axis 2: age class
        # distribute uninfected children (column 0) according to their bite rate and equally across age classes
        x1 = np.array([self.p_bite_rates] * self.age_classes).T * self.size
        x2 = np.zeros_like(x1)
        self.num_children = np.array([x1] + [x2]*max_infections)

        # number of new infections clinical episodes, severe episodes and deaths in a timesteo
        # axis 0: time, axis 1: exposures, axis 2: age class
        self.new_infections = np.zeros_like(self.num_children)
        self.clinical = np.zeros_like(self.num_children)
        self.severe = np.zeros_like(self.num_children)
        self.direct_deaths = np.zeros_like(self.num_children)

        # a record of the total number of blood-stage infections, clinical and severe episodes and deaths on each time step
        self.all_infections = np.zeros(duration)
        self.all_clinical = np.zeros(duration)
        self.all_severe = np.zeros(duration)
        self.all_direct_deaths = np.zeros(duration)

        if t_record_first:
            # a record of the number of blood-stage infections since time t_record_first
            self.first_infections = np.zeros(duration)
            # a record of the number of first clinical cases since time t_record_first
            self.first_clinical = np.zeros(duration)
            # the number of children not infected since time t_record_first (earlier cases are ignored for recording purposes)
            self.non_infected = np.zeros(self.nbr * self.age_classes).reshape(self.nbr, self.age_classes)
            # the number of children not meeting clinical case definition since time t_record_first (earlier cases are ignored for recording purposes)
            self.non_clinical = np.zeros(self.nbr * self.age_classes).reshape(self.nbr, self.age_classes)

        if time_warning:
            self.calculate_simulation_duration(duration, pars)

    def bite_rate_pmf(self, β, nbins):
        assert 0 <= β <= 1, f'β must be 0 to 1, got {β}'
        assert isinstance(nbins, int) and nbins > 0, 'nbins must be positive integer'
        if β == 0:
            bite_rates = np.ones(2)
            p_bite_rates = np.ones(1)
        else:
            bite_rates = np.linspace(1-β, 1+β, nbins)
            d = bite_rates[1]-bite_rates[0]
            # calculate risk of bite rate for each bin
            # use a triangular distribution that has a well-defined upper limit. this is to
            # prevent very large infection rates which destabilise the model solution
            # These issues can occur with the gamma distribution as suggested by Smith Nature (2005)
            # Data from Rodriguez-Barraquer et al. (2013) suggests that a triangular distribution is a reasonable approximation
            x = triang.pdf(bite_rates, 0.5, loc=1-β, scale=2*β)
            p_bite_rates = 0.5*d*(x[:-1] + x[1:])

            # scale p_bites rates so that weighted sum of bite rates equals 1
            # drop the first bite_rate because p_bite_rates is shorter by 1
            p_bite_rates /= (bite_rates[1:]*p_bite_rates).sum()
        return bite_rates[1:], p_bite_rates

    def liver_stage_vaccinate(self, offset=0):
        """ cohort age at liver-stage vaccination """
        assert isinstance(offset, int), 'offset must be a non-negative integer'
        self.age_liver_vac = max(0, self.min_vac_age + offset)

    def blood_stage_vaccinate(self, offset=0):
        assert isinstance(offset, int), 'offset must be a non-negative integer'
        """ cohort age at blood-stage vaccination """
        self.age_blood_vac = max(0, self.min_vac_age + offset)

    def SMC(self, offset=0):
        assert isinstance(offset, int), 'offset must be a non-negative integer'
        """ cohort age at first dose of SMC """
        self.age_smc = max(0, self.min_vac_age + offset)

    def calculate_simulation_duration(self, duration, pars):
        """ calculate an approximate simulation time if time_warning is positive"""
        L_t = self.Lmod[-1, self.age_classes:self.age_classes+duration]
        n = len(L_t)
        t = np.arange(n)
        L_m = pars.Λ * self.Lmod[0, :n] * t
        L_p = pars.Λ * L_t * t
        N1 = (L_m - 3*np.sqrt(L_m)).astype(int)
        N2 = (L_p + 4*np.sqrt(L_p)).astype(int)
        N1 = np.where(N1 < 0, 0, N1)
        N2 = np.where(N2 < 2, 2, N2)
        N2 = np.where(t < N2, t, N2) + 1
        s1 = (N2-N1).sum() * self.nbr * duration * self.age_classes
        s2 = (duration+1)**2//2 * self.nbr * duration * self.age_classes
        sim_time = s1/2048162916800*22
        if sim_time > pars.time_warning:
            print(f'approx. simulation time={sim_time:.0f} sec, weeks={duration}, age classes={self.age_classes}, bite rates={self.nbr}')


def timestep(cohort, t, t_record_first, pars):
    """
        this timestep
        1. calculate number of new infections (new_infectionss) by bite rate and age class
        2. calculate clinical and severe episodes and deaths by bite rate and age class
        3. update number of children with N infections (N=0,..., t) by bite rate and age class
        4. sum over bite rates and age classes to get total number of infections, clinical episodes, severe episodes and deaths
        4. if recording first episodes, save number of first infections and first clinical episodes (currently not implemented)
    """

    # The simulation can be very slow if the number of years is high and bite rate/susceptibility is heterogeneous
    # To speed up simulations we only have to update the number of children within a range of exposures N1->N2
    # as the number of children with exposures outside this range is negligible.
    # Given a cumulative infection rate λ, the number of infections in a population is Poisson distributed
    # with parameter λ. Therefore we need to keep track of the minimum cumulative infection rate λmin and
    # the maximum cumulative infection rate λmax. We then set the range as:
    # N1 is the minimum cumulative infection rate minus 3 times its standard deviation (with lower bound zero)
    # N2 is 1 plus the maximum cumulative infection rate plus 4 times its standard deviation (with upper bound t)
    N1 = int(max(0, cohort.sum_L_min - 3*np.sqrt(cohort.sum_L_min)))
    N2 = 1+int(min(t, max(2, cohort.sum_L_max + 4*np.sqrt(cohort.sum_L_max))))
    exposuresA = slice(N1, N2)
    exposuresB = slice(N1+1, N2+1)
    exposuresC = slice(N1+1, N2)
    exposuresD = slice(N1+2, N2+1)

    # the range of ages of the children in the cohort on this time step
    ages = slice(cohort.minage, cohort.minage + cohort.age_classes)

    # natural deaths
    if pars.μ is not None:
        cohort.num_children *= 1-pars.μ[ages]

    # commence recording new infections if required
    # if t_record_first and t == t_record_first:
    #     np.copyto(cohort.non_infected, cohort.num_children.sum(axis=0))
    #     np.copyto(cohort.non_clinical, cohort.num_children.sum(axis=0))

    #################### update number of children with infections if in malaria season
    if pars.malaria_season[t % 52] > 0:
        # Λ is the force of infection averaged over one year (units of blood infections per child per week)
        # Λ is modified by seasonality, liver-stage vaccination, blood-stage vaccination,
        # early-life protection and bite rate to give L which is the infection risk for a given age and bite rate this week
        L = pars.Λ

        if pars.season_width != 1:
            # Modify L by seasonality.
            # The peak of the season is at week "season_peak" relative to last primary vaccination
            L *= pars.malaria_season[(t-cohort.max_vac_age) % 52]

        # Modify L by liver-stage vaccination.
        # The level of protection of liver-stage vaccination depends only on time since vaccination.
        if cohort.age_liver_vac is not None and cohort.minage >= cohort.age_liver_vac:
            L *= 1 - pars.liver_vac[cohort.minage - cohort.age_liver_vac]

        # Modify L by blood-stage vaccination.
        # blood-stage vaccines can
        # 1. reduce the risk of a liver infection becoming a blood infection (calculated here)
        # 2. reduce the risk of a blood infection becoming clinical (calculated below using v_clinical)
        # 3. reduce the risk of a clinical episode becoming severe (calculated below using v_severe)
        v_clinical = 1
        v_severe = 1
        if cohort.age_blood_vac is not None and cohort.minage >= cohort.age_blood_vac:
            blood_vac_protection = pars.blood_vac[cohort.minage - cohort.age_blood_vac]
            if pars.ω_infection > 0:
                L *= 1 - pars.α_infection * pars.ω_infection * blood_vac_protection
            if pars.ω_clinical > 0:
                v_clinical -= pars.α_clinical * pars.ω_clinical * blood_vac_protection
            if pars.ω_severe > 0:
                v_severe -= pars.α_severe * pars.ω_severe * blood_vac_protection

        # Modify L by seasonal malaria chemoprevention
        if cohort.age_smc is not None and cohort.minage >= cohort.age_smc:
            L *= 1 - pars.smc_coverage * pars.smc_protection[cohort.minage - cohort.age_smc]

        # Modify L by early life protection and bite rate
        # L is transformed from a scalar to an array (axis 0: bite rate, axis 1: age class)
        # Lmod is initialied in Cohort.__init__()
        L *= cohort.Lmod[:, ages]

        # cumulative min and max infection rates
        cohort.sum_L_min += L[0, 0] # smallest bite rate in the youngest age class
        cohort.sum_L_max += L[-1, -1] # largest bite rate in the oldest age class

        ###################### new blood-stage infections
        # copy new infections to temporary storage
        # axis 0: exposure, axis 1: bite rate, axis 2: age class
        np.copyto(cohort.new_infections[exposuresB], cohort.num_children[exposuresA] * L)

        ###################### calculate clinical and severe episodes and deaths by bite rate and age class
        cohort.clinical[exposuresB] = v_clinical * pars.clinical[exposuresA, np.newaxis, np.newaxis] * cohort.new_infections[exposuresB]
        cohort.severe[exposuresB] = v_severe * pars.severe_risk[exposuresA, np.newaxis, ages] * cohort.clinical[exposuresB]
        cohort.direct_deaths[exposuresB] = pars.direct_deaths[ages] * cohort.severe[exposuresB]

        ###################### total number of infections, clinical episodes, severe episodes and deaths
        cohort.all_direct_deaths[t] = cohort.direct_deaths[exposuresB].sum()
        cohort.all_severe[t] = cohort.severe[exposuresB].sum()
        # do not count severe as clinical and adjust for clinical case definition
        cohort.all_clinical[t] = max(0, cohort.clinical[exposuresB].sum() * pars.case_def_clinical)
        cohort.all_infections[t] = cohort.new_infections[exposuresB].sum()

        # if t_record_first and t >= t_record_first:
        #     ###################### new first infections by bite rate and age
        #     cohort.non_infected *= 1-pars.μ[cohort.minage: cohort.minage + cohort.age_classes]
        #     f_new = cohort.non_infected * L
        #     cohort.non_infected -= f_new
        #     cohort.first_infections[t] = f_new.sum()

        #     ###################### new first clinical cases meeting case definition by bite rate and age
        #     # First clinical cases is only approximate because we do not track clinical immunity through prior infections
        #     # nor do we track infections in clinical cases that do not meet case definition
        #     # So this should only be used for short periods after vaccination when most first clinical cases occur
        #     cohort.non_clinical *= 1-pars.μ[cohort.minage: cohort.minage + cohort.age_classes]
        #     f_new = cohort.non_clinical * L * v_clinical * pars.case_def_clinical * pars.ρ_clini   cohort.non_clinical -= f_new
        #     cohort.first_clinical[t] = f_new.sum()


        #################### update num_children
        # remove new infections and add to the next exposure class minus those that die from this infection
        cohort.num_children[N1] += -cohort.new_infections[N1+1]
        cohort.num_children[exposuresC] += -cohort.new_infections[exposuresD] + cohort.new_infections[exposuresC] - cohort.direct_deaths[exposuresC]
        cohort.num_children[N2] = cohort.new_infections[N2] - cohort.direct_deaths[N2] # the new maximum exposure

    else:
        # out of the malaria season
        cohort.all_infections[t] = 0
        cohort.all_clinical[t] = 0
        cohort.all_severe[t] = 0
        cohort.all_direct_deaths[t] = 0
        if t_record_first:
            cohort.first_infections[t] = 0
            cohort.first_clinical[t] = 0

    # increase cohort's minimum age by 1 week
    cohort.minage += 1


def init_unvaccinated_cohort(cohort, t_record_first, pars, logfile=''):
    """
        Start with a single, unvaccincated cohort at birth and follow until the last primary does
        Record init_cohort infections from min_vac_age to max_vac_age
        These will form the basis of the vaccinated cohort
    """
    if logfile:
        log_simulation(logfile, pars)

    init_cohort = Cohort(cohort.max_vac_age, pars, t_record_first=t_record_first, max_bite_bins=cohort.max_bite_bins)

    for t in np.arange(1, cohort.max_vac_age):
        timestep(init_cohort, t, t_record_first, pars)

        cohort.all_infections[t] = init_cohort.all_infections[t]
        cohort.all_clinical[t] = init_cohort.all_clinical[t]
        cohort.all_severe[t] = init_cohort.all_severe[t]
        cohort.all_direct_deaths[t] = init_cohort.all_direct_deaths[t]
        if t_record_first:
            cohort.first_infections[t] = init_cohort.first_infections[t]
            cohort.first_clinical[t] = init_cohort.first_clinical[t]

        # once init_cohort reaches min_vac_age, start copying init_cohort into the vaccinated cohort
        A = t-cohort.min_vac_age
        if A >= 0:
            # set the minimum cumulative infection rate that the cohort will have experienced
            if A == 0:
                cohort.sum_L_min = init_cohort.sum_L_min

            N = t # maximum possible infections of a cohort of age t
            np.copyto(cohort.num_children[:N+1, :, A], init_cohort.num_children[:N+1, :, 0])
            if t_record_first:
                np.copyto(cohort.non_infected[:t, A], init_cohort.non_infected[:t, 0])
                np.copyto(cohort.non_clinical[:t, A], init_cohort.non_clinical[:t, 0])

    # set the maximum cumulative infection rate that the cohort will have experienced
    cohort.sum_L_max = init_cohort.sum_L_max

    # set the minimum age of vaccinated cohort to min_vac_age
    # on exit all vacinees have had their last primary dose and
    # t is the age in weeks of the the oldest vaccinee (max_vac_age=min_vac_age+vac_age_range)
    cohort.minage = cohort.min_vac_age


def sim_one_cohort_through_time(cohort, t_start, t_end, t_record_first, pars):
    """
        Simulate a cohort of children of initial ages min_vac_age to max_vac_age
        from the end of their last primary dose for t_end minus t_start weeks
        If t_record_first is True then also record first infections and first clinical cases
    """
    for t in np.arange(t_start, t_end):
        timestep(cohort, t, t_record_first, pars)


def divide(a, b):
    """ return a/b or 1 when b=0 """
    return np.divide(a, b, out=np.ones_like(a), where=b!=0)


def get_results(cohorts, t_record_first, pars, display_vars, display_measures=[], mean_ages=False):
    def offset_cumsum(data, mask):
        """ set masked elements to zero and return cumsum """
        y = ma.array(data=data, mask=mask)
        z = ma.filled(y, 0)
        return z.cumsum()

    def sum_cohorts(c):
        if pars.children == 'cohort':
            # count cases in the cohort of children born at the start of the study period
            return c
        elif pars.children == 'population':
            # count cases in all children born during the study period
            return np.array([i*x for i, x in zip(range(len(c), 0, -1), c)])
        else:
            raise NotImplementedError("children type not implemented, 'cohort' or 'population' only")

    def extract_results(x, cohort, t_record_first, pars, display_vars, display_measures):
        """ return all cases over time and cumulative cases from t_start_counting """
        # cases per week for each variable (measure=cases)
        if 'All infection' in display_vars:
            x['All infection cases'] = sum_cohorts(cohort.all_infections)
        if 'All clinical' in display_vars:
            x['All clinical cases'] = sum_cohorts(cohort.all_clinical)
        if 'Severe malaria' in display_vars:
            x['Severe malaria cases'] = sum_cohorts(cohort.all_severe)
        if 'Direct deaths' in display_vars:
            x['Direct deaths cases'] = sum_cohorts(cohort.all_direct_deaths)
        # if 'Indirect deaths' in display_vars:
        #     x['Indirect deaths cases'] = sum_cohorts(cohort.indirect_deaths)
        # if 'All deaths' in display_vars:
        #     x['All deaths cases'] = sum_cohorts(cohort.direct_deaths + cohort.indirect_deaths)
        if t_record_first:
            if 'First infection' in display_vars:
                x['First infection cases'] = cohort.first_infections
            if 'First clinical' in display_vars:
                x['First clinical cases'] = cohort.first_clinical

        if 'Kaplan-Meier' in display_measures:
            # kaplan-meier survival function
            for t in ['All', 'Direct', 'Indirect']:
                if f'{t} deaths' in display_vars:
                    s = 1 - np.concatenate((np.zeros(pars.t_start_counting), x[f'{t} deaths cases'][pars.t_start_counting:])) / pars.popsize
                    x[f'{t} deaths Kaplan-Meier'] = s.cumprod()

        # Calculate the cumulative distribution function (CDF) for each variable in `display_vars`.
        # This provides cumulative cases over time, starting from the first recording time (`pars.t_start_counting`).
        mask = np.zeros_like(cohort.all_infections, dtype=bool)
        mask[:pars.t_start_counting] = True
        for variable in display_vars:
            x[f'{variable} cdf'] = offset_cumsum(x[f'{variable} cases'], mask)

        # count from t_start_counting onwards for each variable
        r = {k:v[pars.t_start_counting:] for k, v in x.items()}

        # mean age of severe malaria in weeks
        r['severe_mean_age'] = 0
        if mean_ages:
            Z = cohort.all_severe.sum()
            if Z > 0:
                r['severe_mean_age'] = (cohort.all_severe * np.arange(len(cohort.all_severe))).sum() / Z

        return r

    return {i:extract_results({}, j, t_record_first, pars, display_vars, display_measures) for i, j in cohorts.items()}


def log_simulation(logfile, pars):
    """ log simulation time and parameters to logfile """

    with open(logfile, 'a') as f:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f'Simulation started at {now}\n')

        # Loop through all attributes of pars and write to logfile
        for attr in dir(pars):
            # Skip private/magic attributes and methods
            if not attr.startswith('_') and not callable(getattr(pars, attr)):
                value = getattr(pars, attr)
                f.write(f'{attr} = {value}\n')

