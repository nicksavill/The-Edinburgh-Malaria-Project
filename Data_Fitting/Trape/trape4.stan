// Individual-level sigmoidal hierarchical model of clinical malaria risk

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
    int<lower=0> N;
    int<lower=0> T;
    array[N, 6] int cases;
    array[N, 6] int person_days;
}

transformed data {
    // integral time periods in weeks corresponding to data
    array[7] real periods = {0, 52, 4 * 52, 7 * 52, 10 * 52, 15 * 52, 26 * 52};

    int K = 0;
    for (i in 1 : T)
        for (p in 1 : 6)
            if (cases[i, p] >= 0)
                K += 1;
}

parameters {
    real<lower=0> lambda_tau_elp;
    real<lower=0> mu_delta_clinical_raw;
    real<lower=0> sigma_delta_clinical_raw;
    vector<lower=0>[N] tau_elp; // half-life in years of early life protection against blood infection
    vector<lower=0>[N] lambda_raw; // rate of infection per child per year divided by 10
    vector<lower=0>[N] delta_clinical_raw; // decay rate of risk of clinical malaria per 100 infections
    vector<lower=0>[N] gamma_clinical_raw; // half-maximum number of infections divided by 100
    vector<lower=0, upper=1>[N] rho_clinical; // maximum proportion of infections that cause fever
}

transformed parameters {
    vector[N] lambda = lambda_raw * 10.; // scale to infections per year
    vector[N] delta_clinical = delta_clinical_raw / 100.; // scale to per infection
    vector[N] gamma_clinical = gamma_clinical_raw * 100.; // scale to infections
}

model {
    // hyper_priors
    lambda_tau_elp ~ normal(1, 1) T[0, ];
    mu_delta_clinical_raw ~ normal(1, 1) T[0, ];
    sigma_delta_clinical_raw ~ normal(1, 1) T[0, ];
    // priors
    tau_elp ~ exponential(lambda_tau_elp);
    rho_clinical ~ uniform(0, 1);
    lambda_raw ~ normal(1.5, 0.2) T[0, ];
    delta_clinical_raw ~ normal(mu_delta_clinical_raw, sigma_delta_clinical_raw) T[0, ];
    gamma_clinical_raw ~ normal(1, 0.5) T[0, ];

    // likelihood
    for (i in 1 : T) {
        for (p in 1 : 6) {
            if (cases[i, p] >= 0) {
                real r = integrate_rate(periods[p], periods[p + 1], {lambda[i], 1.0, tau_elp[i], delta_clinical[i], rho_clinical[i], gamma_clinical[i]});
                cases[i, p] ~ poisson(r * person_days[i, p] / (periods[p + 1] - periods[p]) / 7.);
            }
        }
    }
}

generated quantities {
    int j = 1;
    matrix[T, 6] r;
    vector[K] log_lik;
    for (i in 1 : T) {
        for (p in 1 : 6) {
            r[i, p] = integrate_rate(periods[p], periods[p + 1], {lambda[i], 1.0, tau_elp[i], delta_clinical[i], rho_clinical[i], gamma_clinical[i]});
            if (cases[i, p] >= 0) {
                log_lik[j] = poisson_lpmf(cases[i, p] | r[i, p] * person_days[i, p] / (periods[p + 1] - periods[p]) / 7.);
                j += 1;
            }
        }
    }
}
