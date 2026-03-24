# The Edinburgh Malaria Project

The Edinburgh Malaria Project develops and applies a mathematical model to evaluate the long-term impact of malaria interventions, particularly vaccines such as RTS,S, R21, and RH5.1. The project aims to predict clinical malaria cases, severe malaria episodes, and malaria-related deaths averted by these interventions over the lifetime of vaccinated cohorts, accounting for factors like waning vaccine efficacy, rebound effects due to delayed development of naturally acquired immunity, age-specific mortality, and varying transmission intensities.

Key research questions include:
- What are the risks of rebound morbidity and mortality after vaccination cessation?
- How do transmission intensity and intervention duration affect long-term outcomes?

Our goal is to help inform vaccine development, clinical trial design and implementation of vaccine programmes.

## Simulation Model

The core model is a deterministic, population-based simulation of *Plasmodium falciparum* blood-stage infections in a single, multi-age cohort followed weekly from birth until last death. It incorporates:

- **Transmission Dynamics**: Per-capita, year-averaged blood infection rates modified by seasonality, early-life protection, and interventions.
- **Disease Progression**: Infections can lead to clinical malaria (defined as fever) or severe malaria, with only severe cases potentially fatal.
- **Immunity**: Exposure-dependent naturally acquired immunities to clinical and severe malaria.
- **Interventions**: Vaccine protection profiles (e.g., RTS,S, R21 and RH5.1).
- **Mortality**: Age-specific severe malaria-associated death rates.

The model is designed to be parsimonious, is based on empirical data, and is validated against field trials of clinical cases, deaths, and vaccine efficacies. The code is designed to be fast to allow responsive [visualisations](https://temp.bio.ed.ac.uk/Vaccines). Model parameters are estimated from empirical data using Bayesian inference implemented in Stan.

## Installation

### Prerequisites

- Python 3.11 or higher
- Quarto for rendering Jupyter notebooks
- Stan for Bayesian inference

### Clone the Repository

```bash
git clone https://github.com/nicksavill/The-Edinburgh-Malaria-Project
```

### Install Python Dependencies

Install the required Python packages:

- arviz: 0.23.4
- bokeh: 3.6.2
- scipy: 1.15.3
- pandas: 2.1.1
- graphviz: 0.21
- numpy: 1.26.0
- pygam: 0.12.0
- seaborn: 0.13.2
- jupyterlab: 4.3.4
- matplotlib: 3.8.0
- cmdstanpy: 1.2.5
- statsmodels: 0.14.1

### Install Quarto

Install Quarto 1.8.27 to render the Jupyter notebooks to pdf.

### Install CmdStan for Bayesian inference with Stan

- Install CmdStan 2.38 following the [official instructions](https://mc-stan.org/docs/cmdstan-guide/installation.html).
- Ensure `cmdstan` is in your PATH.

## Project Structure

- `Edinburgh_Model/`: Core model Python code
- `Data_Fitting/`: Stan models and data for parameter estimation
    - `Goncalves/`: Fit to data from Goncalves et al. (2014)
    - `Trape/`: Fit to data from Trape and Rogier (1996) and Trape et al. (2024).
- `Notebooks/`: Jupyter notebooks for model development, paper figures, and example code
    - `Data/`: Input and output data files
    - `Figures/`: Notebook and paper figures
    - `Scripts/`: Python scripts for the Jupyter notebooks
    - `PDFS/`: Pdfs of the supplementary notebooks and paper figures


## Usage

### 1. Bayesian model testing and parameter estimation with Stan

Data from Goncalves et al. (2014) are first fit to estimate the relationship between parasitaemia and the risk of clinical malaria and between parasitaemia, exposure and the risk of severe malaria. The code is in directory `Data_fitting/Goncalves`. The Notebook `goncalves1.ipynb` calls CmdStan to fit the model in `goncalves1.stan` using the data in `goncalves.data.json`. See [Notebook S4](Notebooks/S4_goncalves.ipynb) for a discussion of model development and testing, prior definitions, fits and parameter estimates.

Next, clinical attack data from Trape and Rogier (1996) and Trape et al. (2024) are fitted to estimate early life protection parameters and the risk of clinical malaria with exposure. The code is in directory `Data_fitting/Trape`. See [Notebook S5](Notebooks/S5_clinical_immunity.ipynb) for a discussion of model development and testing, prior definitions, fits and parameter estimates.

### 2. Model development

Then we further develop and discuss the model in a series of Jupyter notebooks in the directory `Notebooks`. This includes seasonality, risk of severe malaria with cumulative number of infections, age-specific death rates, seasonal malaria chemoprevention and RH5.1 efficacy The notebooks use python scripts in the directory `Notebooks/Scripts` and datasets in `Notebooks/Data`. The notebooks are rendered as pdfs and published as supplementary material with the paper.

### 3. Core model

The core model code is in directory `Edinburgh_Model/`. Default parameters, derived in steps 1 and 2, are hardcoded in `model.py` for reference. They can be amended by passing values when initialising a simulation. Vaccine protective profiles are hard-coded in `vaccination_types.py` and can be amended.

### 4. Example simulation code

Versions of the paper figures are produced in `Notebooks/paper_figs.ipynb` which uses the python script `Notebooks/Scripts/paper.py`. These are plotted using matplotlib and then were refined in Adobe Illustrator for publication.

The notebook `example.ipynb` provides some example code for plotting static matplotlib figures of a simulation of R21 and a comparison of R21 and RTS,S.

The notebook `interactive.ipynb` provides example code to create an interactive app which can be run in a Jupyter notebook. It uses Bokeh to render the plots and provides sliders and buttons to examine the sensitivity of the model outputs to parameter changes.


# License

This project is licensed under the terms specified in `LICENSE`.

# Contributing

Contributions are welcome. Please open issues or pull requests on GitHub.

# Citation

If you use this code, please cite the associated paper (details to be added).