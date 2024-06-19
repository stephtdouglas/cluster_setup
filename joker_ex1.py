import warnings

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

plt.style.use("default")
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["figure.dpi"] = 200
plt.rcParams["font.size"] = 16
plt.rcParams["font.family"] = "serif"

import astropy.table as at
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import thejoker as tj
from astropy.time import Time
from astropy.visualization.units import quantity_support

# set up a random generator to ensure reproducibility
rnd = np.random.default_rng(seed=42)

data_tbl = at.QTable.read("data.ecsv")
sub_tbl = data_tbl[rnd.choice(len(data_tbl), size=4, replace=False)]

t = Time(sub_tbl["bjd"], format="jd", scale="tcb")
data = tj.RVData(t=t, rv=sub_tbl["rv"], rv_err=sub_tbl["rv_err"])

data = tj.RVData.guess_from_table(sub_tbl)

prior = tj.JokerPrior.default(
    P_min=2 * u.day,
    P_max=1e3 * u.day,
    sigma_K0=30 * u.km / u.s,
    sigma_v=100 * u.km / u.s,
)

prior_samples = prior.sample(size=250_000, rng=rnd)

joker = tj.TheJoker(prior, rng=rnd)
joker_samples = joker.rejection_sample(data, prior_samples, max_posterior_samples=256)

joker_samples = joker.rejection_sample(
    data, "prior_samples.hdf5", max_posterior_samples=256
)

