import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

# Initialize random number generator
np.random.seed(123)

# True parameter values
alpha, sigma = 1, 1
beta = [1, 2.5]

# Size of dataset
size = 100

# Predictor variable
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2

# Simulate outcome variable
Y = alpha + beta[0]*X1 + beta[1]*X2 + np.random.randn(size)*sigma

import pymc3 as pm
print('Running on PyMC3 v{}'.format(pm.__version__))

basic_model = pm.Model()

with basic_model:

    # Priors for unknown model parameters
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10, shape=2)
    sigma = pm.HalfNormal('sigma', sigma=1)

    # Expected value of outcome
    mu = alpha + beta[0]*X1 + beta[1]*X2

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=Y)

basic_model = Model()

with basic_model:
    alpha = Normal('alpha', mu=0, sigma=10)
    beta = Normal('beta', mu=0, sigma=10, shape=2)
    sigma = HalfNormal('sigma', sigma=1)
    mu = alpha + beta[0]*X1 + beta[1]*X2
    Y_obs = Normal('Y_obs', mu=mu, sigma=sigma, observed=Y)

map_estimate = pm.find_MAP(model=basic_model)

print(map_estimate)

with basic_model:
    # draw 500 posterior samples
    trace = pm.sample(500)

pm.summary(trace).round(2)

print(pm.summary(trace).round(2))
