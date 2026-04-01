import numpy as np
from scipy.stats import weibull_min

weeks_per_month = 52/12

def itn(a, args):
    """
        step function for ITN coverage, used to model:
        Mortality in a seven-and-a-half-year follow-up of a trial of insecticide-treated mosquito nets in Ghana
        F. N. Binka, , A. Hodgson, M. Adjuik and T. Smith
        Bednets used for 2 years, then no coverage
        could gradually decrease coverage after 2 years rather than step function
    """
    ν, τ, B, nboosters = args

    return np.piecewise(a,
                        [(a >= 0) & (a < 104), a >= 104],
                        [lambda a: ν, 0]
    )


def exponential_vaccination(a, args):
    """
        A generic exponentially decaying vaccine profile

        ν: maximum vaccine efficacy
        τ: half-life of vaccine efficacy (months)
        period: period between boosters in years, default 1
        nboosters: number of boosters after primary vaccination
        B: booster B months after primary vaccination, default 12
    """
    ν, τ, period, nboosters = args

    # weeks between last primary vaccination and first booster
    B = 52
    # convert period to weeks
    period *= 52

    V_primary = ν
    V_boost = ν
    if τ is None:
        δ = 0
    else:
        δ = np.log(2) / (τ*weeks_per_month)

    def V(t, V_max, δ):
        return V_max * np.exp(-δ*t)

    if nboosters == 0:
        return V(a, V_primary, δ)
    else:
        return np.piecewise(a,
                            [
                                (a >= 0) & (a < B),
                                (a >= B) & (a < B+period*nboosters),
                                (a >= B+period*nboosters)
                            ],
                            [
                                lambda a: V(a, V_primary, δ),
                                lambda a: V((a-B) % period, V_boost, δ),
                                lambda a: V(a-B-(nboosters-1)*period, V_boost, δ)
                            ]
                            )


def OpenMalaria_vaccination(a, args):
    """
        Values taken from Penny et al 2016

        ν: modification of vaccine efficacy, default 1
        τ: modification of half-life, default 1
        period: period between boosters in years, default 1
        B: booster B months after primary vaccination, default 12
        nboosters: number of boosters after primary vaccination
    """
    ν, τ, period, nboosters = args

    # weeks between last primary vaccination and first booster
    B = 52
    # convert period to weeks
    period *= 52

    c = 0.69
    V_primary = 0.91*ν
    V_boost = 0.49*ν
    δ = np.log(2) / (τ * 7.32*weeks_per_month)

    def V(t, V_max, δ):
        return V_max * (1-weibull_min.cdf(δ*t, c))

    if nboosters == 0:
        return V(a, V_primary, δ)
    else:
        return np.piecewise(a,
                            [
                                (a >= 0) & (a < B),
                                (a >= B) & (a < B+period*nboosters),
                                (a >= B+period*nboosters)
                            ],
                            [
                                lambda a: V(a, V_primary, δ),
                                lambda a: V((a-B) % period, V_boost, δ),
                                lambda a: V(a-B-(nboosters-1)*period, V_boost, δ)
                            ]
                            )


def Imperial_RTSS_vaccination(a, args):
    """
        Based on analysis in White et al. 2015 Lancet Inf. Dis.

        ν: modification of vaccine efficacy, default 1
        τ: modification of half-life of long-lived compnent, default 1
        period: period between boosters in years, default 1
        B: booster B months after primary vaccination, default 12
        nboosters: number of boosters after primary vaccination
    """
    ν, τ, period, nboosters = args

    # weeks between last primary vaccination and first booster
    B = 52
    # convert period to weeks
    period *= 52

    CS_peak = 583*ν # mean from Table 1 in White et al. 2015
    ρ_peak = 0.88
    CS_boost = 289*ν # mean from Table 1 in White et al. 2015
    ρ_boost = 0.70
    # convert to weeks
    r_s = np.log(2) / (45 / 7)
    r_l = np.log(2) / (τ * 591 / 7)
    alpha = 0.74
    beta = 99.2
    V_max = 0.93

    def V(t, CS_max, ρ):
        CS = CS_max*(ρ*np.exp(-r_s*t) + (1-ρ)*np.exp(-r_l*t))
        return V_max*(1 - 1/(1 + (CS / beta)**alpha))

    if nboosters == 0:
        return V(a, CS_peak, ρ_peak)
    else:
        return np.piecewise(a,
                            [
                                (a >= 0) & (a < B),
                                (a >= B) & (a < B+period*nboosters),
                                (a >= B+period*nboosters)
                            ],
                            [
                                lambda a: V(a, CS_peak, ρ_peak),
                                lambda a: V((a-B) % period, CS_boost, ρ_boost),
                                lambda a: V(a-B-(nboosters-1)*period, CS_boost, ρ_boost)
                            ]
                            )


def Imperial_R21_vaccination(a, args):
    """
        Based on analysis in Schmit et al 2024 Lancet

        ν: modification of vaccine efficacy, default 1
        τ: modification of half-life of long-lived compnent, default 1
        period: period between boosters in years, default 1
        B: booster B months after primary vaccination, default 12
        nboosters: number of boosters after primary vaccination
    """
    ν, τ, period, nboosters = args
    # weeks between last primary vaccination and first booster
    B = 52
    # convert period to weeks
    period *= 52

    CS_peak = 10000 # from Fig. 1A Schmit et al 2024 Lancet
    ρ_peak = 0.69
    CS_boost = 10000 # from Fig. 1A Schmit et al 2024 Lancet
    ρ_boost = 0.52
    # convert to weeks
    r_s = np.log(2) / (44.6 / 7)
    r_l = np.log(2) / (τ * 533 / 7)
    alpha = 0.91
    beta = 471
    V_max = 0.87 # Imperial estimate from Schmit et al 2024 Lancet
    V_max *= 0.95 # 5% reduction due to Imperial's over-estimate
    V_max *= ν # scale by ν (default 1)

    def V(t, CS_max, ρ):
        CS = CS_max*(ρ*np.exp(-r_s*t) + (1-ρ)*np.exp(-r_l*t))
        V = V_max*(1 - 1/(1 + (CS / beta)**alpha))
        if ν != 1:
            return np.where(V <= 1, V, 1)
        else:
            return V

    if nboosters == 0:
        return V(a, CS_peak, ρ_peak)
    else:
        return np.piecewise(a,
                            [
                                (a >= 0) & (a < B),
                                (a >= B) & (a < B+period*nboosters),
                                (a >= B+period*nboosters)
                            ],
                            [
                                lambda a: V(a, CS_peak, ρ_peak),
                                lambda a: V((a-B) % period, CS_boost, ρ_boost),
                                lambda a: V(a-B-(nboosters-1)*period, CS_boost, ρ_boost)
                            ]
                            )


def RH5_Delayed_vaccination(a, args):
    """
        Based on analysis in Silk et al. 2024
    """
    _, τ, period, nboosters = args

    # weeks between last primary vaccination and first booster
    B = 52
    # convert period to weeks
    period *= 52

    def V(a):
        x = τ * np.array([0.0, 2.0, 9.89, 13.81, 42.05, 94.05])
        y = np.array([755.32, 648.56, 287.01, 195.9, 76.47, 76.47])
        # if a is greater than the last value in x, return the last value in y
        return np.interp(a, x, y)

    if nboosters == 0:
        return V(a)
    else:
        return np.piecewise(a,
                            [
                                (a >= 0) & (a < B),
                                (a >= B) & (a < B+period*nboosters),
                                (a >= B+period*nboosters)
                            ],
                            [
                                lambda a: V(a),
                                lambda a: V((a-B) % period),
                                lambda a: V(a-B-(nboosters-1)*period)
                            ]
                            )


def RH5_Monthly_vaccination(a, args):
    """
        Based on analysis in Silk et al. 2024
    """
    _, τ, period, nboosters = args

    # weeks between last primary vaccination and first booster
    B = 52
    # convert period to weeks
    period *= 52

    def V(a):
        x = τ * np.array([0.0, 1.99, 9.91, 13.88, 42.07, 94.07])
        y = [336.12, 311.75, 87.99, 61.62, 27.35, 17.72]
        return np.interp(a, x, y)

    if nboosters == 0:
        return V(a)
    else:
        return np.piecewise(a,
                            [
                                (a >= 0) & (a < B),
                                (a >= B) & (a < B+period*nboosters),
                                (a >= B+period*nboosters)
                            ],
                            [
                                lambda a: V(a),
                                lambda a: V((a-B) % period),
                                lambda a: V(a-B-(nboosters-1)*period)
                            ]
                            )


def smc_protection(a, args):
    ramp, rounds, repeats, coverage = args

    n = int(rounds*weeks_per_month)
    smc_protection_year = np.zeros(52)
    smc_protection_year[:n] = 1 - np.exp(-ramp * np.arange(n))

    # repeat SMC for "repeats+1" years
    smc_protection = np.zeros_like(a, dtype=float)
    for age in range(repeats+1):
        l = smc_protection[age*52:(age+1)*52].shape[0]
        smc_protection[age*52:(age+1)*52] = smc_protection_year[:l]
    return coverage * smc_protection
