import warnings

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

plt.style.use("default")
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["figure.dpi"] = 200
plt.rcParams["font.size"] = 16
plt.rcParams["font.family"] = "serif"

import astropy.coordinates as coord
import astropy.table as at
import astropy.units as u
import numpy as np
import corner
import pymc as pm
import arviz as az
import thejoker as tj
import pickle

# set up a random number generator to ensure reproducibility
rnd = np.random.default_rng(seed=8675309)

data_tbl = at.QTable.read("data.ecsv")
sub_tbl = data_tbl[rnd.choice(len(data_tbl), size=18, replace=False)]  # downsample data
data = tj.RVData.guess_from_table(sub_tbl, t_ref=data_tbl.meta["t_ref"])

prior = tj.JokerPrior.default(
    P_min=2 * u.day,
    P_max=1e3 * u.day,
    sigma_K0=30 * u.km / u.s,
    sigma_v=100 * u.km / u.s,
)

prior_samples = prior.sample(size=250_000, rng=rnd)

joker = tj.TheJoker(prior, rng=rnd)
joker_samples = joker.rejection_sample(data, prior_samples, max_posterior_samples=256)

with prior.model:
    mcmc_init = joker.setup_mcmc(data, joker_samples)

    trace = pm.sample(tune=500, draws=500, start=mcmc_init, cores=1, chains=2)

az.summary(trace, var_names=prior.par_names)

mcmc_samples = tj.JokerSamples.from_inference_data(prior, trace, data)
mcmc_samples.wrap_K()


with open("true-orbit.pkl", "rb") as f:
    truth = pickle.load(f)

# make sure the angles are wrapped the same way
if np.median(mcmc_samples["omega"]) < 0:
    truth["omega"] = coord.Angle(truth["omega"]).wrap_at(np.pi * u.radian)

if np.median(mcmc_samples["M0"]) < 0:
    truth["M0"] = coord.Angle(truth["M0"]).wrap_at(np.pi * u.radian)
