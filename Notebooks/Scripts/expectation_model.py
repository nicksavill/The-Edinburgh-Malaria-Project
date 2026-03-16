from numpy import exp, log, linspace, trapz
from collections import namedtuple

Pars_severe = namedtuple('Parameters', 'λ δs ρs ρe τe δc ρc γc')

def infections(a, p):
    b = log(2) / p.τe
    e = p.ρe * exp(-b * a)
    l = p.λ * (1 - e)        # rate of blood infection at age a
    n = p.λ * (a - (p.ρe - e) / b)  # cumulative number of blood infections by age a
    return l, n

def clinical_model(n, p):
    # risk of clinical episode on nth infection
    return p.ρc * (1 + exp(-p.δc*p.γc)) / (1 + exp(p.δc*(n - p.γc)))

def severe_model(n, p):
    # risk of severe episode given clinical on nth infection
    return p.ρs * exp(-p.δs*n)

def clinical_cases(a, p):
    l, n = infections(a, p)
    return l * clinical_model(n, p)  # rate of clinical episode at age a (episodes per unit time)

def severe_cases(a, p):
    l, n = infections(a, p)
    return l * clinical_model(n, p) * severe_model(n, p) # rate of clinical episode at age a (episodes per unit time)

def cumulative_clinical_cases(a1, a2, p):
    # number of clinical episodes between ages a1 and a2
    b = log(2) / p.τe
    f = lambda a: log(1 + exp(-p.δc * (p.γc - a * p.λ + p.λ * p.ρe / b * (1 - exp(-b * a)))))
    return p.ρc*p.λ*(1 + exp(-p.δc*p.γc))*(a2 - a1 + p.ρe / b*(exp(-b*a2) - exp(-b*a1)) + (f(a1) - f(a2)) / (p.λ*p.δc))

def cumulative_severe_cases(a1, a2, p):
    a = linspace(a1, a2, 100)
    return trapz(severe_cases(a, p), a)

