// cc -fPIC -O3 -shared -o timestep.so timestep.c

/*
 * C translation of Parameters, Cohort and sim_one_cohort_through_time()
 *
 * ARRAY LAYOUT NOTES
 * ------------------
 * All multi-dimensional NumPy arrays are flattened to 1-D double* in row-major
 * (C) order.  Macros are provided for every array so the indexing intent is
 * explicit and matches the Python axis comments.
 *
 *   pars.clinical        [n]              1-D, length = max_weeks
 *   pars.severe_risk     [n][age]         2-D, shape (max_weeks, max_weeks)
 *   pars.direct_deaths   [age]            1-D, length = max_weeks
 *   pars.survival        [age]            1-D, length = max_weeks
 *   pars.malaria_season  [week % 52]      1-D, length = 52
 *   pars.liver_vac       [weeks_post_vac] 1-D, length = max_weeks  (NULL if lsv==false)
 *   pars.blood_vac       [weeks_post_vac] 1-D, length = max_weeks  (NULL if bsv==false)
 *   pars.smc_protection  [weeks_post_smc] 1-D, length = max_weeks  (NULL if smc==false)
 *
 *   cohort.Lmod          [br][age]        2-D, shape (nbr, max_weeks)
 *   cohort.num_children  [n][br][ac]      3-D, shape (max_weeks+1, nbr, age_classes)
 *   cohort.new_infections[n][br][ac]      3-D, same shape
 *   cohort.clinical      [n][br][ac]      3-D, same shape
 *   cohort.severe        [n][br][ac]      3-D, same shape
 *   cohort.direct_deaths [n][br][ac]      3-D, same shape
 *
 * "None" optional fields use a sentinel value:
 *   age_liver_vac / age_blood_vac / age_smc == -1  means "not set" (Python None)
 *
 * OWNERSHIP
 * ---------
 * All pointer fields must be allocated by the caller before passing to any
 * function.  No allocation or free() is performed inside this file.
 */

#include <math.h>   /* sqrt, fmax, fmin */
#include <stddef.h> /* NULL */
#include <stdio.h>  /* printf */

/* =========================================================================
 * Parameters struct
 * Mirrors the computed/derived fields of the Python Parameters class that
 * are actually consumed by sim_one_cohort_through_time().
 * ========================================================================= */
typedef struct {
    int    max_weeks;
    int    max_vac_age;         /* = min_vac_age + vac_age_range */
    double season_width;        /* duration measure of malaria season (0,1] */
    double Lambda;              /* Λ = λ/52: weekly force of infection */
    double case_def_clinical;

    /* blood-stage vaccine scaling parameters (set by vac_profile_blood choice) */
    double alpha_infection;     /* α_infection */
    double alpha_clinical;      /* α_clinical */
    double alpha_severe;        /* α_severe */
    double omega_infection;     /* ω_infection */
    double omega_clinical;      /* ω_clinical */
    double omega_severe;        /* ω_severe */

    /* ---- derived arrays (caller-allocated) ---- */
    double *survival;           /* 1-μ[age]: natural mortality per week, length max_weeks */
    double *malaria_season;     /* seasonal multiplier, length 52 */
    double *clinical;
    double *severe_risk;
    double *direct_deaths;

    /* vaccination waning curves — NULL when the corresponding intervention is off */
    double *liver_vac;          /* liver-stage protection by weeks post-vac */
    double *blood_vac;          /* blood-stage protection by weeks post-vac  */
    double *smc_protection;     /* SMC protection by weeks post-first-dose   */
} Parameters;

/* =========================================================================
 * Cohort struct
 * Mirrors all fields of the Python Cohort class used during simulation.
 * ========================================================================= */
typedef struct {
    /* ---- scalars ---- */
    int    max_weeks;
    int    max_vac_age;     /* weeks */
    int    age_classes;     /* = vac_age_range: number of age-class slots */
    int    nbr;             /* number of bite-rate bins (= len(p_bite_rates)) */

    /*
     * Vaccination / SMC onset ages.
     * -1 means "not set" (Python None), i.e. the intervention is absent.
     * Set by liver_stage_vaccinate(), blood_stage_vaccinate(), SMC().
     */
    int    age_liver_vac;
    int    age_blood_vac;
    int    age_smc;

    double *Lmod;

    /*
    * 3-D simulation arrays, all shape (max_weeks+1, nbr, age_classes).
    * Flat index: [n][br][ac] = n*(nbr*age_classes) + br*age_classes + ac
    */
    double *num_children;
    double *new_infections;
    double *clinical;
    double *severe;
    double *direct_deaths;

    /* ---- 1-D output arrays, length = max_weeks (= duration) ---- */
    double *all_infections;
    double *all_clinical;
    double *all_severe;
    double *all_direct_deaths;

    /*
     * First-infection / first-clinical tracking.
     * All four pointers are NULL when t_record_first was not requested.
     */
    // double *first_infections;   /* length = max_weeks */
    // double *first_clinical;     /* length = max_weeks */
    // double *non_infected;       /* 2-D [nbr][age_classes], flattened */
    // double *non_clinical;       /* 2-D [nbr][age_classes], flattened */
} Cohort;

/* =========================================================================
 * Index macros
 * ========================================================================= */

 /* Index helper: severe_risk[n][age_abs] */
#define SEVERE_RISK(pars, n, age_abs) \
    ((pars)->severe_risk[(size_t)(n) * (pars)->max_weeks \
    + (age_abs)])

/* 3-D cohort arrays: [n][br][ac] */
#define IDX3(cohort, n, br, ac) \
    ((size_t)(n) * ((cohort)->nbr * (cohort)->age_classes) \
     + (size_t)(br) * (cohort)->age_classes \
     + (ac))

/* Lmod[br][age_abs], shape (nbr, max_weeks) */
#define LMOD(cohort, pars, br, age_abs) \
    ((cohort)->Lmod[(size_t)(br) * (cohort)->max_weeks \
     + (age_abs)])



/* =========================================================================
 * Simulate cohort one timestep.
 * Pass t_record_first != 0 to record first infections / first clinical cases
 * (the actual accumulation code is currently commented out in the Python
 *  original; only the zero-fill in the off-season branch is active here).
 * ========================================================================= */
double* timestep(
    int         t,
    Cohort     *cohort,
    Parameters *pars,
    int        *minage,
    double     *sum_L_min,
    double     *sum_L_max
)
{
    const int BR = cohort->nbr;
    const int AC = cohort->age_classes;

    /* ------------------------------------------------------------------
    * Exposure window [N1, N2]
    * ------------------------------------------------------------------ */
    int N1 = (int)fmax(0.0, *sum_L_min - 3.0 * sqrt(*sum_L_min));
    int N2 = 1 + (int)fmin((double)t, fmax(2.0, *sum_L_max + 4.0 * sqrt(*sum_L_max)));

    int age_start = *minage;      /* absolute age, inclusive */

    /* ------------------------------------------------------------------
    * Natural deaths
    * ------------------------------------------------------------------ */
    for (int n = N1; n <= N2; n++)
        for (int br = 0; br < BR; br++)
            for (int ac = 0; ac < AC; ac++)
                cohort->num_children[IDX3(cohort, n, br, ac)] *= pars->survival[age_start + ac];

    /* ------------------------------------------------------------------
    * Malaria season check
    * ------------------------------------------------------------------ */
    if (pars->malaria_season[t % 52] > 0.0) {
        /*
        * Build L[br][ac]: infection risk this week for each
        * (bite-rate, age-class) pair.
        *
        * Start from scalar Λ, then apply each modifier in the same
        * order as the Python code.
        *
        * C99 VLA — replace with a heap-allocated scratch buffer if BR
        * or AC can be large in your application.
        */
        double L[BR][AC];
        double Lambda = pars->Lambda;

        /* Seasonality (skipped when season_width == 1, i.e. no season) */
        if (pars->season_width != 1.0) {
            /* Match Python: (t - cohort.max_vac_age) % 52, always positive */
            int season_idx = ((t - cohort->max_vac_age) % 52 + 52) % 52;
            Lambda *= pars->malaria_season[season_idx];
        }

        /* Liver-stage vaccine (-1 means not vaccinated) */
        if (cohort->age_liver_vac >= 0 && *minage >= cohort->age_liver_vac) {
            int wk = *minage - cohort->age_liver_vac;
            Lambda *= 1.0 - pars->liver_vac[wk];
        }

        /* Blood-stage vaccine */
        double v_clinical = 1.0;
        double v_severe   = 1.0;
        if (cohort->age_blood_vac >= 0 && *minage >= cohort->age_blood_vac) {
            int    wk  = *minage - cohort->age_blood_vac;
            double bvp = pars->blood_vac[wk];

            if (pars->omega_infection > 0.0)
                Lambda *= 1.0 - pars->alpha_infection * pars->omega_infection * bvp;
            if (pars->omega_clinical > 0.0)
                v_clinical -= pars->alpha_clinical * pars->omega_clinical * bvp;
            if (pars->omega_severe > 0.0)
                v_severe   -= pars->alpha_severe   * pars->omega_severe   * bvp;
        }

        /* SMC (-1 means not in use) */
        if (cohort->age_smc >= 0 && *minage >= cohort->age_smc) {
            int wk = *minage - cohort->age_smc;
            Lambda *= 1.0 - pars->smc_protection[wk];
        }

        /*
        * Multiply by Lmod[br][age_abs]
        * = bite_rates[br] * (1 - elp[age_abs])
        * This converts L from a scalar to a (bite_rate × age_class) array,
        */
        for (int br = 0; br < BR; br++)
            for (int ac = 0; ac < AC; ac++)
                L[br][ac] = Lambda * LMOD(cohort, pars, br, age_start + ac);

        /*
        * Update cumulative min/max infection rates:
        *   L[0][0]       = smallest bite rate × youngest age class
        *   L[BR-1][AC-1] = largest bite rate  × oldest  age class
        */
        *sum_L_min += L[0][0];
        *sum_L_max += L[BR-1][AC-1];

        /* ----------------------------------------------------------------
        * New blood-stage infections
        *   new_infections[n+1][br][ac] = num_children[n][br][ac] * L[br][ac]
        * for n in [N1, N2)
        * -------------------------------------------------------------- */
        for (int n = N1; n < N2; n++)
            for (int br = 0; br < BR; br++)
                for (int ac = 0; ac < AC; ac++)
                    cohort->new_infections[IDX3(cohort, n+1, br, ac)] = cohort->num_children[IDX3(cohort, n, br, ac)] * L[br][ac];

        /* ----------------------------------------------------------------
        * Clinical episodes, severe episodes, direct deaths
        *
        *   clinical[n+1]      = v_clinical
        *                      * pars.clinical[n]           (exposure n)
        *                      * new_infections[n+1]
        *
        *   severe[n+1]        = v_severe
        *                      * pars.severe_risk[n][age]   (exposure n, age)
        *                      * clinical[n+1]
        *
        *   direct_deaths[n+1] = pars.direct_deaths[age]    (age only)
        *                      * severe[n+1]
        *
        * pars.clinical   varies with exposure n only  (not br, not ac).
        * pars.severe_risk varies with exposure n AND absolute age        (not br).
        * pars.direct_deaths varies with absolute age only               (not br, not n).
        * This exactly mirrors the NumPy broadcasting in the Python source.
        * -------------------------------------------------------------- */
        double sum_infections = 0.0;
        double sum_clinical   = 0.0;
        double sum_severe     = 0.0;
        double sum_deaths     = 0.0;

        for (int n = N1; n < N2; n++) {
            int nb = n + 1;   /* index into exposuresB */
            for (int br = 0; br < BR; br++) {
                for (int ac = 0; ac < AC; ac++) {
                    int    age_abs = age_start + ac;
                    double ni   = cohort->new_infections[IDX3(cohort, nb, br, ac)];
                    double clin = v_clinical * pars->clinical[n] * ni;
                    double sev  = v_severe   * SEVERE_RISK(pars, n, age_abs) * clin;
                    double dd   = pars->direct_deaths[age_abs] * sev;

                    cohort->clinical     [IDX3(cohort, nb, br, ac)] = clin;
                    cohort->severe       [IDX3(cohort, nb, br, ac)] = sev;
                    cohort->direct_deaths[IDX3(cohort, nb, br, ac)] = dd;

                    sum_infections += ni;
                    sum_clinical   += clin;
                    sum_severe     += sev;
                    sum_deaths     += dd;
                }
            }
        }

        cohort->all_direct_deaths[t] = sum_deaths;
        cohort->all_severe       [t] = sum_severe;
        /* do not count severe episodes as clinical; apply case definition */
        cohort->all_clinical     [t] = fmax(0.0, sum_clinical * pars->case_def_clinical - sum_severe);
        cohort->all_infections   [t] = sum_infections;

        /* ----------------------------------------------------------------
        * Update num_children
        *
        * Python:
        *   num_children[N1]         += -new_infections[N1+1]
        *   num_children[N1+1..N2-1] += -new_infections[N1+2..N2]
        *                             +  new_infections[N1+1..N2-1]
        *                             -  direct_deaths [N1+1..N2-1]
        *   num_children[N2]          =  new_infections[N2]
        *                             -  direct_deaths [N2]
        *
        * (exposuresC = slice(N1+1, N2), exposuresD = slice(N1+2, N2+1))
        * -------------------------------------------------------------- */

        /* Row N1 */
        for (int br = 0; br < BR; br++)
            for (int ac = 0; ac < AC; ac++)
                cohort->num_children[IDX3(cohort, N1, br, ac)]
                    -= cohort->new_infections[IDX3(cohort, N1+1, br, ac)];

        /* Rows N1+1 .. N2-1 */
        for (int n = N1+1; n < N2; n++)
            for (int br = 0; br < BR; br++)
                for (int ac = 0; ac < AC; ac++)
                    cohort->num_children[IDX3(cohort, n, br, ac)]
                        += -cohort->new_infections [IDX3(cohort, n+1, br, ac)]
                            + cohort->new_infections [IDX3(cohort, n,   br, ac)]
                            - cohort->direct_deaths  [IDX3(cohort, n,   br, ac)];

        /* Row N2 */
        for (int br = 0; br < BR; br++)
            for (int ac = 0; ac < AC; ac++)
                cohort->num_children[IDX3(cohort, N2, br, ac)] =
                        cohort->new_infections [IDX3(cohort, N2, br, ac)]
                    - cohort->direct_deaths  [IDX3(cohort, N2, br, ac)];

    } else {
        /* Out of malaria season */
        cohort->all_infections   [t] = 0.0;
        cohort->all_clinical     [t] = 0.0;
        cohort->all_severe       [t] = 0.0;
        cohort->all_direct_deaths[t] = 0.0;

        // if (t_record_first) {
        //     if (cohort->first_infections) cohort->first_infections[t] = 0.0;
        //     if (cohort->first_clinical)   cohort->first_clinical  [t] = 0.0;
        // }
    }

    /* Advance cohort minimum age by one week */
    *minage += 1;
}
