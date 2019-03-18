# Instructions
```bash
virtualenv -p python2.7 venv
source ./venv/bin/activate
pip install -r requirements.txt
cd experiment
python logistic.py
cd ..
R --no-save < ess.R
```

See outputs in `experiment/result`.


# Changes in this fork
* Only remained logistic regression experiment
* Slight code refactoring

# Changes in [upstream fork](https://github.com/feynmanliang/firefly-monte-carlo)
* Implements (BFGS optimization) MAP tuning of likelihood lower bound
* Fixed MH acceptance probability for brightness variables
* Correct implementation of `pseudo_log_lik_cache` to cache pseudo log
  likelihood elements rather than whole subvectors
* Oveflow protection throughout code

# [Firefly Monte Carlo : Exact MCMC with Subsets of Data](https://github.com/HIPS/firefly-monte-carlo)

This package implements the Firefly Monte Carlo algorithm
described [here](https://hips.seas.harvard.edu/files/maclaurin-firefly-uai-2014.pdf).
To get started, check out the [toy data example](examples/toy_dataset.py).

