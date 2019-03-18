# Instructions
`virtualenv -p python2.7 venv`
`source ./venv/bin/activate`
`pip -r requirements.txt`

[WORK IN PROCESS]

# Changes in this fork
* Implements (BFGS optimization) MAP tuning of likelihood lower bound
* Fixed MH acceptance probability for brightness variables
* Correct implementation of `pseudo_log_lik_cache` to cache pseudo log
  likelihood elements rather than whole subvectors
* Oveflow protection throughout code

# Firefly Monte Carlo : Exact MCMC with Subsets of Data

This package implements the Firefly Monte Carlo algorithm
described [here](https://hips.seas.harvard.edu/files/maclaurin-firefly-uai-2014.pdf).
To get started, check out the [toy data example](examples/toy_dataset.py).
