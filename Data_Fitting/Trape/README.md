Stan code for fitting to data in Trape and Rogier (1996) and Trape et al. (2024).

See [Notebook S5](../../Notebooks/S5_clinical_immunity.ipynb) for the mathematical model development, prior definitions, fits and parameter estimates

File | Description
:-- | :--
trape_populations.json | Population-level data for Ndiop and Dielmo
trape_individuals.json | Individual-level data for Dielmo
*.stan | Stan model code
*.ipynb | Notebooks fitting model to data with markov chain diagnostics
*_draws.csv | random draws from the posterior
*_fit.pickle | pickled Stan fit

#### Individual-level models

Model No. | Shape | Hierarchical parameters | Notes
:-- | :-- | :-- | :--
1 | Exponential | None |
2 | Sigmoidal | None |
3 | Sigmoidal | None | ρ~elp~ = 1
4 | Sigmoidal | τ~elp~, δ~clinical~ | ρ~elp~ = 1
5 | Exponential | τ~elp~, δ~clinical~ | ρ~elp~ = 1

#### Population-level models

Model No. | Shape | Parameters | Notes
:-- | :-- | :-- | :--
6 | Sigmoidal | Heterogeneous |
7 | Sigmoidal | Heterogeneous | Informative prior on ρ~clinical~
8 | Sigmoidal | Homogeneous | Informative prior on ρ~clinical~

