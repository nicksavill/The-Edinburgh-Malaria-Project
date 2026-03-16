import graphviz
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pygam import LinearGAM, s

from Scripts import goncalves

rh5 = pd.read_excel('Unpublished_data/nicksavill_rh51_allmalaria_6m.xlsx')
rh5 = rh5.rename(columns={'Parasitedensitytrophozoites': 'Parasites per μl'})
rh5['pd'] = np.log10(rh5['Parasites per μl'])
rh5['arm'] = rh5['arm'].replace({'1.Rabies-delayed':'rabies', '2.Rh5.1-delayed':'delayed', '3.Rabies-monthly':'rabies', '4.Rh5.1-monthly':'monthly'})
rh5['fever'] = rh5['fever'].replace({'1. Yes': 'Yes', '0. No':'No'})
rh5['positive'] = rh5['malaria'] == '1. Malaria positive'
rh5['detection'] = rh5['Atwhatvisitwasmalariasuspec'].apply(lambda x: 'passive' if x == 'UNSCHEDULEDVISIT' else 'active')
rh5['t'] = (rh5['date'] - rh5['datelastvac']).dt.days
rh5 = rh5.sort_values('t')
infection = rh5[rh5['positive']].copy()


def clinical_time_series(monthly=True):
    def plot(group, arm, ax):
        # Cumulative count by t
        df = group.groupby('t').size().cumsum().to_frame()
        # not all days are in df. Add all days into the index
        df = df.merge(pd.Series(0, index=range(df.index.min(), maxt+1), name='tmp'), how='outer', left_index=True, right_index=True).drop('tmp', axis=1)
        # forward fill values into the new empty days
        df = df.ffill()
        df.columns = [arm]
        ax.plot(df.index, df.values, label=arm)

    # Filter for fever episodes
    fever_df = rh5[rh5['fever'] == 'Yes'].copy()
    maxt = fever_df['t'].max()
    if monthly == False:
        fever_df = fever_df.query('arm != "monthly"')

    fig, axs = plt.subplots(1, 2, figsize=(7, 3))
    fig.subplots_adjust(wspace=0.4)

    ax = axs[0]
    for arm, group in fever_df.groupby('arm'):
        plot(group, arm, ax)
    ax.set_ylabel('Cumulative all-cause fevers')

    ax = axs[1]
    for arm, group in fever_df.query('positive == True').groupby('arm'):
        plot(group, arm, ax)
    ax.set_ylabel('Cumulative malaria fevers\nwith detectable parasites')

    for ax, t in zip(axs, 'AB'):
        ax.legend(title='Arm')
        ax.set_xlabel('Days post last vaccination')
        ax.set_title(t, loc='left')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(visible=True, which='both', axis='both', color='lightgrey', linestyle='--', linewidth=0.5)

    return fig, axs

def last_day_of_season():
    def cumulative(group, arm):
        # Cumulative count by t
        df = group.groupby('t').size().cumsum().to_frame()
        # not all days are in df. Add all days into the index
        df = df.merge(pd.Series(0, index=range(df.index.min(), df.index.max()+1), name='tmp'), how='outer', left_index=True, right_index=True).drop('tmp', axis=1)
        # forward fill values into the new empty days
        df = df.ffill()
        df.columns = [arm]
        return df

    # Filter for fever episodes
    fever_df = rh5[rh5['fever'] == 'Yes'].copy()

    all_arms = fever_df.groupby('arm')
    malaria_arms = fever_df.query('positive == True').groupby('arm')

    last_day = malaria_arms['t'].max()['rabies']

    malaria = {}
    for arm, group in malaria_arms:
        df = cumulative(group, arm)
        malaria[arm] = int(df.iloc[-1].values[0])
        # malaria[arm] = int(df.loc[last_day].values[0])

    all_cause = {}
    for arm, group in all_arms:
        df = cumulative(group, arm)
        # all_cause[arm] = int(df.iloc[-1].values[0])
        all_cause[arm] = int(df.loc[last_day].values[0])

    return last_day, all_cause, malaria

def fold_change():
    # fold change in parasitaemia in each arm
    x = infection.groupby('arm')['pd'].mean()
    return 10**(x['rabies'] - x)

def clinical_rates():
    natama = pd.read_csv('Data/rh5_detected_cases.csv')
    natama['week'] = natama['t'] / 7
    cohorts = 'rabies', 'monthly', 'delayed'

    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    ax = axs[0]
    ax = sns.scatterplot(data=natama, x='week', y='count', hue='arm', hue_order=cohorts, markers=('o', 's', '^'), lw=0, alpha=0.25, style='arm', ax=ax)
    ax.set_ylabel('Cumulative clinical episodes')

    pdep = {}
    days = np.linspace(0, 180, 201)
    for cohort in cohorts:
        d = natama.query('arm == @cohort')
        x = d['t'].values
        y = d['count'].values
        gam = LinearGAM(s(0, constraints='monotonic_inc', n_splines=10), fit_intercept=False).gridsearch(x.reshape(-1, 1), y, progress=False)
        pdep[cohort] = gam.partial_dependence(term=0, X=days)
        ax.plot(days/7, pdep[cohort])

    ax = axs[1]
    ax.plot(days/7, np.gradient(pdep['rabies'], days), label='rabies', lw=2, ls=':', color='C0')
    ax.plot(days/7, np.gradient(pdep['monthly'], days), label='monthly', lw=2, ls='--', color='C1')
    ax.plot(days/7, np.gradient(pdep['delayed'], days), label='delayed', lw=2, ls='-', color='C2')
    ax.set_ylabel('Clinical episodes per day');
    ax.legend(title='arm')

    ax.set_title('B', loc='left')
    for ax, title in zip(axs, 'AB'):
        ax.grid(visible=True, which='both', axis='both', color='lightgrey', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Weeks post last primary vaccination')
        ax.set_title(title, loc='left')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    return fig, axs, days, pdep

def parasitaemia_distributions():
    def annotation(a, **kwargs):
        plt.axvline(**kwargs)
        plt.text(0.2, 0.9, f"n = {len(a.dropna())}", transform=plt.gca().transAxes)

    g = sns.displot(
        data=infection,
        x="Parasites per μl",
        bins=20,
        col="arm",
        col_order=("rabies", "monthly", "delayed"),
        log_scale=(True, False),
        height=3

    ).set_axis_labels('Parasitaemia (parasites/μl)');

    g.map(annotation, 'Parasites per μl', x=min(infection['Parasites per μl'].dropna()), color='k', ls='--', lw=1)
    return g

def severe_risk(filename):
    def p_severe_first_given_fever(p, x):
        # white blood cell to cell/μl conversion
        conversion = np.log10(50)
        return p.bS / (1 + np.exp(-p.aS*(x-(p.cS+conversion))))

    p = goncalves.get_estimates(filename)
    arms = infection['arm'].unique()

    severe_risk = {}
    for arm in arms:
        log_parasitaemias = infection.query('arm == @arm')['pd'].values.reshape(-1, 1)
        severe_risk[arm] = p_severe_first_given_fever(p, log_parasitaemias).mean(axis=0)

    severe_risk_reduction = {}
    for arm in arms:
        severe_risk_reduction[arm] = (severe_risk["rabies"] - severe_risk[arm])/severe_risk["rabies"]

    return severe_risk, severe_risk_reduction

def rh5_antibody_profiles():
    silk_delayed = pd.read_csv('Data/silk_2024_figS4_delayed.txt', skiprows=1, header=None)
    silk_monthly = pd.read_csv('Data/silk_2024_figS4_monthly.txt', skiprows=1, header=None)

    x_delayed = (silk_delayed.loc[:, 0] - silk_delayed.loc[0, 0]) / (365/52)
    y_delayed = silk_delayed.loc[:, 1]
    x_monthly = (silk_monthly.loc[:, 0] - silk_monthly.loc[0, 0]) / (365/52)
    y_monthly = silk_monthly.loc[:, 1]

    fig, ax = plt.subplots(figsize=(3.5, 3))
    ax.plot(x_delayed, y_delayed, color='C2', label='Delayed')
    ax.plot(x_monthly, y_monthly, color='C1', label='Monthly')
    ax.set_xlabel('Weeks post last primary vaccination')
    ax.set_ylabel('Anti-RH5.1 IgG (μg/μl)')
    ax.legend(title='Arm')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(visible=True, which='both', axis='both', color='lightgrey', linestyle='--', linewidth=0.5)
    return fig, ax

def find_alpha_waning_immunity(days, pdep):
    def p(a, x, y):
        return np.interp(a, x, y)

    silk_delayed = pd.read_csv('Data/silk_2024_figS4_delayed.txt', skiprows=1, header=None)
    silk_monthly = pd.read_csv('Data/silk_2024_figS4_monthly.txt', skiprows=1, header=None)

    x_delayed = (silk_delayed.loc[:, 0] - silk_delayed.loc[0, 0]) / (365/52)
    y_delayed = silk_delayed.loc[:, 1]
    x_monthly = (silk_monthly.loc[:, 0] - silk_monthly.loc[0, 0]) / (365/52)
    y_monthly = silk_monthly.loc[:, 1]

    control_cases_per_day = np.gradient(pdep['rabies'], days)
    control_total_cases = pdep['rabies'][-1]

    alpha_clinical_monthly = control_total_cases / np.trapz(p(days, x_monthly*7, y_monthly) * control_cases_per_day, days)
    alpha_clinical_delayed = control_total_cases / np.trapz(p(days, x_delayed*7, y_delayed) * control_cases_per_day, days)

    alpha_severe_monthly = 1/y_monthly.max()
    alpha_severe_delayed = 1/y_delayed.max()
    return alpha_clinical_delayed, alpha_severe_delayed, alpha_clinical_monthly, alpha_severe_monthly

def clinical_model():
    g = graphviz.Digraph(filename='model', strict=True, directory='../Figures', format='svg')

    with g.subgraph(name='cluster_Rabies') as c:
        c.attr(label='Rabies vaccine control')
        c.attr(labeljust='left')
        c.attr(style='filled', color='lightgrey')
        c.node('rabies_Non-malaria', label='Non-malaria', style='filled', color='DarkRed', fillcolor='DarkRed', fontname='Helvetica', fontcolor='White')
        c.node('rabies_Asymptomatic', label='Asymptomatic', style='filled', color='DarkRed', fillcolor='DarkRed', fontname='Helvetica', fontcolor='White')
        c.node('rabies_Detected', label='Detected', style='filled', color='DarkRed', fillcolor='DarkRed', fontname='Helvetica', fontcolor='White')
        c.node('rabies_Undetected', label='Undetected', style='filled', color='DarkRed', fillcolor='DarkRed', fontname='Helvetica', fontcolor='White')
    with g.subgraph(name='cluster_delayed') as c:
        c.attr(label='Delayed RH5.1 vaccine regimen')
        c.attr(labelloc='bottom')
        c.attr(labeljust='left')
        c.attr(style='filled', color='lightgrey')
        c.node('delayed_Non-malaria', label='Non-malaria', style='filled', color='DarkRed', fillcolor='DarkRed', fontname='Helvetica', fontcolor='White')
        c.node('delayed_Detected', label='Detected', style='filled', color='DarkRed', fillcolor='DarkRed', fontname='Helvetica', fontcolor='White')
        c.node('delayed_Undetected', label='Undetected', style='filled', color='DarkRed', fillcolor='DarkRed', fontname='Helvetica', fontcolor='White')
        c.node('delayed_Asymptomatic', label='Asymptomatic', style='filled', color='DarkRed', fillcolor='DarkRed', fontname='Helvetica', fontcolor='White')
    g.edge('rabies_Undetected', 'delayed_Asymptomatic', label='<ρ<sub>U</sub>>')
    g.edge('rabies_Undetected', 'delayed_Undetected', weight='2', label='<1-ρ<sub>U</sub>>')
    g.edge('rabies_Detected', 'delayed_Undetected', label='<ρ<sub>D</sub>>')
    g.edge('rabies_Detected', 'delayed_Detected', weight='2', label='<1-ρ<sub>D</sub>>')
    return g

def blood_model():
    g = graphviz.Digraph(filename='model', strict=True, directory='../Figures', format='svg')

    with g.subgraph(name='cluster_Rabies') as c:
        c.attr(label='Rabies vaccine control')
        c.attr(labeljust='left')
        c.attr(style='filled', color='lightgrey')
        c.node('rabies_Non-malaria', label='Non-malaria', style='filled', color='DarkRed', fillcolor='DarkRed', fontname='Helvetica', fontcolor='White')
        c.node('rabies_Detected', label='Detected', style='filled', color='DarkRed', fillcolor='DarkRed', fontname='Helvetica', fontcolor='White')
        c.node('rabies_Undetected', label='Undetected', style='filled', color='DarkRed', fillcolor='DarkRed', fontname='Helvetica', fontcolor='White')
        c.node('rabies_Asymptomatic', label='Asymptomatic', style='filled', color='DarkRed', fillcolor='DarkRed', fontname='Helvetica', fontcolor='White')
    with g.subgraph(name='cluster_delayed') as c:
        c.attr(label='Delayed RH5.1 vaccine regimen')
        c.attr(labelloc='bottom')
        c.attr(labeljust='left')
        c.attr(style='filled', color='lightgrey')
        c.node('delayed_Non-malaria', label='Non-malaria', style='filled', color='DarkRed', fillcolor='DarkRed', fontname='Helvetica', fontcolor='White')
        c.node('delayed_Detected', label='Detected', style='filled', color='DarkRed', fillcolor='DarkRed', fontname='Helvetica', fontcolor='White')
        c.node('delayed_Undetected', label='Undetected', style='filled', color='DarkRed', fillcolor='DarkRed', fontname='Helvetica', fontcolor='White')
        c.node('delayed_Asymptomatic', label='Asymptomatic', style='filled', color='DarkRed', fillcolor='DarkRed', fontname='Helvetica', fontcolor='White')
    g.edge('rabies_Undetected', 'delayed_Undetected', weight='2', label='1-ρ')
    g.edge('rabies_Detected', 'delayed_Detected', weight='2', label='1-ρ')
    g.edge('rabies_Asymptomatic', 'delayed_Asymptomatic', weight='2', label='1-ρ')
    return g

def omega_clinical(dTd, dTm, D_r):
    Urd = np.arange(dTd, 130)
    Urm = np.arange(dTm, 130)
    ω_clinical_delayed = lambda Urd: dTd/(Urd+D_r)
    ω_clinical_monthly = lambda Urm: dTm/(Urm+D_r)

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.plot(Urd, ω_clinical_delayed(Urd), color='C2', label='Delayed')
    ax.plot(Urm, ω_clinical_monthly(Urm), color='C1', label='Monthly')
    ax.scatter(x=dTd, y=0, color='w')
    ax.scatter(x=dTm, y=0, color='w')
    ax.set_xlabel('Malaria fevers with undetected\nparasites in the rabies arm')
    ax.set_ylabel('RH5.1 protection against\nclinical malaria')
    ax.legend(title='Arm')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(visible=True, which='both', axis='both', color='lightgrey', linestyle='--', linewidth=0.5)
    return fig, ax

def Ur_Ud(T_r, T_d, D_r, D_d):
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))

    # prediction for when RH5 does not reduce risk of blood infectins
    U_r_clinical = np.arange(59, 131)
    U_d_clinical = lambda U_r: D_r - D_d - (T_r - T_d) + U_r
    ax.plot(U_r_clinical, U_d_clinical(U_r_clinical), label='not reduced')

    # prediction for when RH5 does reduce risk of blood infectins
    rho = 1 - D_d/D_r
    U_r_blood = (T_r - T_d - (D_r - D_d)) / rho
    U_d_blood = (1-rho) * U_r_blood
    ax.scatter(x=U_r_blood, y=U_d_blood, color='C0', label='reduced')

    ax.legend(title='Risk of blood infection')
    ax.set_xlabel('Malaria fevers with undetected\nparasites in the rabies arm')
    ax.set_ylabel('Malaria fevers with undetected\nparasites in the delayed arm')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(visible=True, which='both', axis='both', color='lightgrey', linestyle='--', linewidth=0.5)

    return fig, ax

