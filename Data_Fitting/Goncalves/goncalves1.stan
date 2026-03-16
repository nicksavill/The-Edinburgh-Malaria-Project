functions {
    real raised_cosine_lpdf(real y, real mu, real s) {
        real z = (y - mu) / s;
        return -log(2) - log(s) + log1p(cos(z * pi()));
    }

    real raised_cosine_lccdf(real y, real mu, real s) {
        if (y <= mu - s || y >= mu + s) {
            return negative_infinity();
        }
        real z = (y - mu) / s;
        return -log(2) + log1p(-(z + sin(z * pi()) / pi()));
    }

    real g_x(real x, real xc, array[] real theta, array[] real xr, array[] int xi) {
        // p(x|mu,sigma) / (1 + exp(-aS * (x - cS))) / (1 + exp(-aF * (x - cF)))
        real mu = theta[2];
        real s = theta[3];
        real aF = theta[4];
        real cF = theta[5];
        real aS = theta[6];
        real cS = theta[7];

        real ln_p_x = raised_cosine_lpdf(x | mu, s);
        real ln_P_fever_given_x = -log1p(exp(-aF * (x - cF)));
        real ln_g_x = -log1p(exp(-aS * (x - cS)));
        return exp(ln_P_fever_given_x + ln_g_x + ln_p_x);
    }

    real p_SM_x(real x, real xc, array[] real theta, array[] real xr, array[] int xi) {
        // probability density of x joint with SM
        real lambda = theta[1];
        real mu = theta[2];
        real s = theta[3];
        real aF = theta[4];
        real cF = theta[5];
        real aS = theta[6];
        real cS = theta[7];
        real f = theta[8];

        real ln_p_x = raised_cosine_lpdf(x | mu, s);
        real ln_P_fever_given_x = -log1p(exp(-aF * (x - cF)));
        real ln_P_SM_given_fever_x = -log1p(exp(-aS * (x - cS))) + log(f);
        return exp(ln_P_SM_given_fever_x + ln_P_fever_given_x + ln_p_x);
    }

    real p_fever_x(real x, real xc, array[] real theta, array[] real xr, array[] int xi) {
        // probability density of x joint with fever
        real mu = theta[2];
        real s = theta[3];
        real aF = theta[4];
        real cF = theta[5];

        real ln_p_x = raised_cosine_lpdf(x | mu, s);
        real ln_P_fever_given_x = -log1p(exp(-aF * (x - cF)));
        return exp(ln_P_fever_given_x + ln_p_x);
    }

    real p_nofever_x(real x, real xc, array[] real theta, array[] real xr, array[] int xi) {
        // probability density of x joint with no fever (asymptomatic)
        real mu = theta[2];
        real s = theta[3];
        real aF = theta[4];
        real cF = theta[5];

        real ln_p_x = raised_cosine_lpdf(x | mu, s);
        real ln_P_fever_given_x = -log1p(exp(-aF * (x - cF)));
        real ln_P_nofever_given_x = log1p(-1 / (1 + exp(-aF * (x - cF))));
        return exp(ln_P_nofever_given_x + ln_p_x);
    }

    // we need to check that the limits of integration lie inside the domain of the raised cosine distribution
    // otherwise the integration fails
    real g_x_integrate(real a, real b, array[] real theta, data array[] real xr, array[] int xi) {
        real mu = theta[2];
        real s = theta[3];
        real lower_limit, upper_limit;

        if (b <= mu - s || a >= mu + s) {
            return 0; // both limits are outside of domain of p(x)
        }
        if (a <= mu - s) {
            lower_limit = mu - s; // lower limit is outside of domain of p(x)
        } else {
            lower_limit = a;
        }
        if (b >= mu + s) {
            upper_limit = mu + s; // upper limit is outside of domain of p(x)
        } else {
            upper_limit = b;
        }
        return integrate_1d(g_x, lower_limit, upper_limit, theta, xr, xi);
    }

    real p_fever_x_integrate(real a, real b, array[] real theta, data array[] real xr, array[] int xi) {
        real mu = theta[2];
        real s = theta[3];
        real lower_limit, upper_limit;

        if (b <= mu - s || a >= mu + s) {
            return 0; // both limits are outside of domain of p(x)
        }
        if (a <= mu - s) {
            lower_limit = mu - s; // lower limit is outside of domain of p(x)
        } else {
            lower_limit = a;
        }
        if (b >= mu + s) {
            upper_limit = mu + s; // upper limit is outside of domain of p(x)
        } else {
            upper_limit = b;
        }
        return integrate_1d(p_fever_x, lower_limit, upper_limit, theta, xr, xi);
    }

    real p_nofever_x_integrate(real a, real b, array[] real theta, data array[] real xr, array[] int xi) {
        real mu = theta[2];
        real s = theta[3];
        real lower_limit, upper_limit;

        if (b <= mu - s || a >= mu + s) {
            return 0; // both limits are outside of domain of p(x)
        }
        if (a <= mu - s) {
            lower_limit = mu - s; // lower limit is outside of domain of p(x)
        } else {
            lower_limit = a;
        }
        if (b >= mu + s) {
            upper_limit = mu + s; // upper limit is outside of domain of p(x)
        } else {
            upper_limit = b;
        }
        return integrate_1d(p_nofever_x, lower_limit, upper_limit, theta, xr, xi);
    }

    real p_SM_x_integrate(real a, real b, array[] real theta, data array[] real xr, array[] int xi) {
        real mu = theta[2];
        real s = theta[3];
        real lower_limit, upper_limit;

        if (b <= mu - s || a >= mu + s) {
            return 0; // both limits are outside of domain of p(x)
        }
        if (a <= mu - s) {
            lower_limit = mu - s; // lower limit is outside of domain of p(x)
        } else {
            lower_limit = a;
        }
        if (b >= mu + s) {
            upper_limit = mu + s; // upper limit is outside of domain of p(x)
        } else {
            upper_limit = b;
        }
        return integrate_1d(p_SM_x, lower_limit, upper_limit, theta, xr, xi);
    }

    vector lower_truncated_raised_cosine_mean(vector mu, real s, data real mup, data real a) {
        vector[1] fmu;
        if (a <= mu[1] - s) {
            fmu = mu;
        } else if (a >= mu[1] + s) {
            fmu = [a]';
        } else {
            fmu = ((-a ^ 2 + mu ^ 2) * pi() ^ 2 + 2 * mu * pi() ^ 2 * s + (-2 + pi() ^ 2) * s ^ 2 - 2 * s * (s * cos(((a - mu) * pi()) / s) + a * pi() * sin(((a - mu) * pi()) / s))) / (4 * pi() ^ 2 * s);
        }
        // Solve the algebraic system y = f(mu) - mu' for y = 0, ie find mu to make this true
        return fmu - mup;
    }

    real F(int C_n, data array[,] real p_n, real dS, real lambda) {
        int row;
        real f = 0;
        // determine which row of p_n to use based on lambda. we have to use this lookup table because
        // there is no closed form solution of p_n (the PMF of the nth infection).
        // fortunately the estimate of lambda is quite precise (2.35±0.04) so this should not introduce much error
        if (lambda < 2.1)
            row = 1;
        else if (lambda < 2.3)
            row = 2;
        else if (lambda < 2.5)
            row = 3;
        else if (lambda < 2.7)
            row = 4;
        else if (lambda < 2.9)
            row = 5;
        else
            row = 6;
        for (n in 1 : C_n) {
            f += p_n[row, n] * exp(-dS * (n - 1));
        }
        return f;
    }
}

data {
    real mup;
    real<lower=0> child_years;
    int<lower=0> children;
    int<lower=0> cases;
    int<lower=0> N_SM;
    array[N_SM] real SM_logp;
    array[N_SM] int<lower=0> SM;
    int<lower=0> N_mild;
    array[N_mild] real mild_logp;
    array[N_mild] int<lower=0> mild;
    int<lower=0> N_asym;
    array[N_asym] real asym_logp;
    array[N_asym] int<lower=0> asym;
    int<lower=0> N_inf;
    array[2, N_inf] int<lower=0> SM_risk;
    int<lower=0> R_n;
    int<lower=0> C_n;
    array[R_n, C_n] real<lower=0> p_n;
}

transformed data {
    array[0] real xr; // dummy for integral_1d
    array[0] int xi; // dummy for integral_1d

    int y_cases = cases; // total number of cases

    // number of cases between consecutive logp values for SM
    array[N_SM - 1] int<lower=0> y_SM;
    for (i in 1 : N_SM - 1) {
        y_SM[i] = SM[i] - SM[i + 1];
    }
    // number of cases between consecutive logp values for mild
    array[N_mild - 1] int<lower=0> y_mild;
    for (i in 1 : N_mild - 1) {
        y_mild[i] = mild[i] - mild[i + 1];
    }
    // number of cases between consecutive logp values for asymptomatic
    array[N_asym - 1] int<lower=0> y_asym;
    for (i in 1 : N_asym - 1) {
        y_asym[i] = asym[i] - asym[i + 1];
    }
}

parameters {
    real<lower=0> lambda;
    real<lower=0> s;
    real<lower=0> aF;
    real cF;
    real<lower=0> aS;
    real cS;
    real<lower=0, upper=1> bS;
    real<lower=0> dS;
}

model {
    // priors
    lambda ~ gamma(4.0, 2.0);
    s ~ gamma(16.0, 8.0);
    aF ~ normal(1.0, 1.0) T[0, ];
    cF ~ normal(-1.0, 0.5);
    aS ~ normal(1.0, 1.0) T[0, ];
    cS ~ normal(3.0, 0.5);
    bS ~ normal(0.1, 0.1) T[0, ];
    dS ~ normal(0.12, 0.04) T[0, ];

    real P_SM; // probability case is SM
    real P_detect; // probability of detectable parasites
    vector[N_SM - 1] P_Dx_given_SM; // probability case is SM between consecutive parasitaemias
    vector[N_mild - 1] P_mild_Dx; // probability case is mild between consecutive parasitaemias
    vector[N_asym - 1] P_asym_Dx; // probability case is asymptomatic between consecutive parasitaemias
    vector[N_inf] P_SM_given_n_detect; // probability of SM with cumulative exposure

    real mu = solve_newton(lower_truncated_raised_cosine_mean, [mup]', s, mup, 0.0)[1];
    array[8] real theta = {lambda, mu, s, aF, cF, aS, cS, bS * F(C_n, p_n, dS, lambda)};
    real I = g_x_integrate(0, positive_infinity(), theta, xr, xi);

    P_detect = exp(raised_cosine_lccdf(0 | mu, s));
    P_SM = p_SM_x_integrate(negative_infinity(), positive_infinity(), theta, xr, xi);
    for (i in 1 : N_mild - 1) {
        P_mild_Dx[i] = p_fever_x_integrate(mild_logp[i], mild_logp[i + 1], theta, xr, xi) - p_SM_x_integrate(mild_logp[i], mild_logp[i + 1], theta, xr, xi);
    }
    for (i in 1 : N_asym - 1) {
        P_asym_Dx[i] = p_nofever_x_integrate(asym_logp[i], asym_logp[i + 1], theta, xr, xi);
    }
    for (i in 1 : N_SM - 1) {
        P_Dx_given_SM[i] = p_SM_x_integrate(SM_logp[i], SM_logp[i + 1], theta, xr, xi) / P_SM;
    }
    for (n in 1 : N_inf) {
        P_SM_given_n_detect[n] = I * bS * exp(-dS * (n - 1)) / P_detect;
    }

    // likelihood
    y_mild ~ poisson(lambda * child_years * P_mild_Dx);
    y_asym ~ poisson(lambda * child_years * P_asym_Dx);
    y_cases ~ poisson(lambda * child_years * P_detect);
    y_SM ~ multinomial(P_Dx_given_SM);
    SM_risk[1] ~ binomial(SM_risk[2], P_SM_given_n_detect);
}

generated quantities {
    real P_SM; // probability case is SM
    real P_detect; // probability of detectable parasites
    real P_fever; // probability of fever
    real P_fever_and_detect; // probability of fever and >= 1 parasite per 200 WBC
    real P_fever_and_5000; // probability of fever and >= 5000 parasite per μL
    vector[N_SM - 1] P_Dx_given_SM; // probability case is SM between consecutive parasitaemias
    vector[N_mild - 1] P_mild_Dx; // probability case is mild between consecutive parasitaemias
    vector[N_asym - 1] P_asym_Dx; // probability case is asymptomatic between consecutive parasitaemias
    vector[N_inf] P_SM_given_n_detect; // probability of SM with cumulative exposure

    real mu = solve_newton(lower_truncated_raised_cosine_mean, [mup]', s, mup, 0.0)[1];
    array[8] real theta = {lambda, mu, s, aF, cF, aS, cS, bS * F(C_n, p_n, dS, lambda)};
    real I = g_x_integrate(0, positive_infinity(), theta, xr, xi);

    P_fever = p_fever_x_integrate(negative_infinity(), positive_infinity(), theta, xr, xi);
    P_detect = exp(raised_cosine_lccdf(0 | mu, s));
    P_fever_and_detect = p_fever_x_integrate(0, positive_infinity(), theta, xr, xi);
    P_fever_and_5000 = p_fever_x_integrate(log10(5000. / 50.), positive_infinity(), theta, xr, xi);
    P_SM = p_SM_x_integrate(negative_infinity(), positive_infinity(), theta, xr, xi);

    int j = 1;
    vector[N_mild-1 + N_asym-1 + 2 + N_inf] log_lik;

    for (i in 1 : N_mild - 1) {
        P_mild_Dx[i] = p_fever_x_integrate(mild_logp[i], mild_logp[i + 1], theta, xr, xi) - p_SM_x_integrate(mild_logp[i], mild_logp[i + 1], theta, xr, xi);
        log_lik[j] = poisson_lpmf(y_mild[i] | lambda * child_years * P_mild_Dx[i]);
        j += 1;
    }

    for (i in 1 : N_asym - 1) {
        P_asym_Dx[i] = p_nofever_x_integrate(asym_logp[i], asym_logp[i + 1], theta, xr, xi);
        log_lik[j] = poisson_lpmf(y_asym[i] | lambda * child_years * P_asym_Dx[i]);
        j += 1;
    }

    log_lik[j] = poisson_lpmf(y_cases | lambda * child_years * P_detect);
    j += 1;

    for (i in 1 : N_SM - 1) {
        P_Dx_given_SM[i] = p_SM_x_integrate(SM_logp[i], SM_logp[i + 1], theta, xr, xi) / P_SM;
    }
    log_lik[j] = multinomial_lpmf(y_SM | P_Dx_given_SM);
    j += 1;

    for (n in 1 : N_inf) {
        P_SM_given_n_detect[n] = I * bS * exp(-dS * (n - 1)) / P_detect;
        log_lik[j] = binomial_lpmf(SM_risk[1, n] | SM_risk[2, n], P_SM_given_n_detect[n]);
        j += 1;
    }
}
