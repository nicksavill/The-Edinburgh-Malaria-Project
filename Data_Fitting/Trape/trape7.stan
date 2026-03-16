// Population-level sigmoidal model of clinical malaria risk heterogeneous immunity parameters

functions {
    real f(real a, array[] real theta) {
        real lambda = theta[1] / 52.; // convert to weeks
        real rho_elp = theta[2];
        real tau_elp = theta[3] * 52; // convert to weeks
        real delta_clinical = theta[4];
        real gamma_clinical = theta[6];
        real b = log(2) / tau_elp;

        return 2 * exp(-delta_clinical * gamma_clinical) - 1 -
            exp(-delta_clinical * (gamma_clinical - lambda * (a - rho_elp / b * (1 - exp(-b * a)))));
    }

    real integrate_rate(real a1, real a2, array[] real theta) {
        // the number of clinical episodes between weeks a1 and a2
        real lambda = theta[1] / 52.; // convert to weeks
        real rho_elp = theta[2];
        real tau_elp = theta[3] * 52; // convert to weeks
        real delta_clinical = theta[4];
        real rho_clinical = theta[5];
        real gamma_clinical = theta[6];
        real b = log(2) / tau_elp;

        return rho_clinical / delta_clinical * (1 - exp(-delta_clinical * gamma_clinical)) /
            (1 - 2 * exp(-delta_clinical * gamma_clinical)) *
            (delta_clinical * lambda * rho_elp / b *
                (exp(-b * a2) - exp(-b * a1)) - (a1 - a2) * delta_clinical * lambda - log(f(a2, theta) / f(a1, theta)));
    }
}

data {
    int<lower=0> M; // number of children observed in Dielmo
    int<lower=0> N_dielmo;
    int<lower=0> N_ndiop;
    array[N_dielmo] int<lower=0> dielmo_cases;
    array[N_dielmo] int<lower=0> dielmo_person_days;
    array[N_dielmo + 1] int<lower=0> dielmo_periods;
    array[N_ndiop] real<lower=0> ndiop_rate;
    array[N_ndiop + 1] int<lower=0> ndiop_periods;
}

transformed data {
    vector<lower=0>[N_dielmo] dielmo_person_years;
    for (i in 1 : N_dielmo) {
        dielmo_person_years[i] = dielmo_person_days[i] / 365.;
    }
    // we only have yearly rates for Ndiop, convert to counts assuming the same number of children as Dielmo
    array[N_ndiop] int<lower=0> ndiop;
    for (i in 1 : N_ndiop) {
        ndiop[i] = to_int(M * ndiop_rate[i] * (ndiop_periods[i + 1] - ndiop_periods[i]));
    }
}

parameters {
    vector<lower=0>[2] tau_elp; // half-life in years of early life protection against blood infection
    vector<lower=0>[2] lambda_raw; // rate of infection per child per year divided by 10
    vector<lower=0>[2] delta_clinical_raw; // decay rate of risk of clinical malaria per 100 infections
    vector<lower=0>[2] gamma_clinical_raw; // half-maximum number of infections divided by 100
    vector<lower=0, upper=1>[2] rho_elp; // protection from blood infection in newborns
    vector<lower=0, upper=1>[2] rho_clinical; // maximum proportion of infections that cause fever
}

transformed parameters {
    vector[2] lambda = lambda_raw * 10.; // scale to infections per year
    vector[2] delta_clinical = delta_clinical_raw / 100.; // scale to per infection
    vector[2] gamma_clinical = gamma_clinical_raw * 100.; // scale to infections
}

model {
    // priors
    rho_elp ~ beta(5, 1);
    tau_elp ~ exponential(2);
    rho_clinical ~ normal(0.789, 0.014) T[0, 1];
    lambda_raw[1] ~ normal(1.5, 0.2) T[0, ];
    lambda_raw[2] ~ normal(0.5, 0.2) T[0, ];
    delta_clinical_raw ~ normal(1, 1) T[0, ];
    gamma_clinical_raw ~ normal(1, 0.5) T[0, ];

    // likelihood
    for (i in 1 : N_dielmo) {
        real r = integrate_rate(dielmo_periods[i] * 52, dielmo_periods[i + 1] * 52, {lambda[1], rho_elp[1], tau_elp[1],
            delta_clinical[1], rho_clinical[1], gamma_clinical[1]});
        dielmo_cases[i] ~ poisson(dielmo_person_years[i] * r);
    }
    for (i in 1 : N_ndiop) {
        real r = integrate_rate(ndiop_periods[i] * 52, ndiop_periods[i + 1] * 52, {lambda[2], rho_elp[2], tau_elp[2],
            delta_clinical[2], rho_clinical[2], gamma_clinical[2]});
        ndiop[i] ~ poisson(M * r);
    }
}

generated quantities {
    vector[N_dielmo + N_ndiop] log_lik;
    array[N_dielmo] int dielmo_tilde;
    array[N_ndiop] int ndiop_tilde;

    // log likelihoods for model comparison with LOO-CV
    for (i in 1 : N_dielmo) {
        real r = integrate_rate(dielmo_periods[i] * 52, dielmo_periods[i + 1] * 52, {lambda[1], rho_elp[1], tau_elp[1],
            delta_clinical[1], rho_clinical[1], gamma_clinical[1]});
        log_lik[i] = poisson_lpmf(dielmo_cases[i] | dielmo_person_years[i] * r);
        dielmo_tilde[i] = poisson_rng(M * r);
    }
    for (i in 1 : N_ndiop) {
        real r = integrate_rate(ndiop_periods[i] * 52, ndiop_periods[i + 1] * 52, {lambda[2], rho_elp[2], tau_elp[2],
            delta_clinical[2], rho_clinical[2], gamma_clinical[2]});
        log_lik[i + N_dielmo] = poisson_lpmf(ndiop[i] | M * r);
        ndiop_tilde[i] = poisson_rng(M * r);
    }
}
