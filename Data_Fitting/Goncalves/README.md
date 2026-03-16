Stan code for fitting to data in Goncalves et al. (2014).

See Notebooks/S3_goncalves.ipynb for the mathematical model development, prior definitions, fits and parameter estimates

File | Description
:-- | :--
goncalves.data.json | The data to fit from Goncalves et al. (2014)
goncalves.init.json | Initial parameter guesses
goncalves1.stan | Stan model code
goncalves1.ipynb | Notebook fitting model to data with markov chain diagnostics
goncalves1_draws.csv | random draws from the posterior
goncalves1_fit.pickle | pickled Stan fit