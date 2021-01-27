
# coding: utf-8

# In[1]:

import os, sys

import numpy as np
import schwimmbad
import h5py
import astropy.io.ascii as at
import astropy.units as u
import astropy.units.cds as cds
from astropy.table import Table
from astropy.time import Time

import h5py
import pymc3 as pm
import exoplanet as xo

from thejoker.data import RVData
from thejoker import JokerPrior, JokerSamples
from thejoker import TheJoker

import theano
theano.config.gcc.cxxflags = '-march=core2'

# In[2]:

rnd = np.random.RandomState(seed=42)

TWOFACE_CACHE_PATH = os.getenv("TWOFACE_CACHE_PATH")
cluster = "Ruprecht"
month = "Jan21"
treslist = at.read("Ruprecht/ruprecht_targets.csv",delimiter=",")
data_dir = os.path.expanduser("/observing/TRES/Ruprecht/")


def get_data(name,to_delete=None):
    print(name)
    if cluster=="Praesepe":
        ccf_sum_file = os.path.join(data_dir,"{0}/{0}.ccfSum.txt".format(name))
        vzero_file = os.path.join(data_dir,"{0}/{0}.vzero.txt".format(name))
    elif (cluster=="Ruprecht") or (cluster=="ComaBer"):
        ccf_sum_file = os.path.join(data_dir,"{0}.ccfSum.txt".format(name))
        vzero_file = os.path.join(data_dir,"{0}.vzero.txt".format(name))
    #ccf_sum_file = os.path.join(data_dir,"{0}/{0}.ccfSum.txt".format(name))
    #vzero_file = os.path.join(data_dir,"{0}/{0}.vzero.txt".format(name))
    print(data_dir)
    print(vzero_file)

    if os.path.exists(vzero_file):
        print("using Vzero corrected file")
        vzeros = at.read(vzero_file)
        t_raw = vzeros["col1"]*cds.JD
        rv = vzeros["col2"]/1000*u.km/u.s
        rve = vzeros["col3"]/1000*u.km/u.s
    elif os.path.exists(ccf_sum_file):
        print("Using relative RVs")
        ccfs = at.read(ccf_sum_file)
        if (type(ccfs["BJD_UTC"][0])!=np.float64) and (";" in ccfs["BJD_UTC"][0]):
            ccfs = ccfs[1:]
        t_raw = np.asarray(ccfs["BJD_UTC"],dtype=np.float64)*cds.JD
        rv = np.asarray(ccfs["rv"],dtype=np.float64)/1000*u.km/u.s
        rve = np.asarray(ccfs["rv_err"],dtype=np.float64)/1000*u.km/u.s
    else:
        return None

    # Set up the RV Data object
    t_day = t_raw.to(u.day)
    t = t_day.value
    data = RVData(t=t, rv=rv, rv_err=rve)
    return data

# Set up the prior
prior = JokerPrior.default(P_min=0.5*u.day,P_max=24000*u.day,
                           sigma_K0=30*u.km/u.s,
                           sigma_v=100*u.km/u.s)

joker = TheJoker(prior,random_state=rnd)

# Read in the initial Joker results
results_filename = os.path.join(TWOFACE_CACHE_PATH, "test{0}_2019{1}.hdf5".format(cluster,month))
if not os.path.exists(results_filename):
    print("file not found")


print(results_filename)

name = "732m0862104"

data = get_data(name)

with h5py.File(results_filename, 'r') as f:
    print(results_filename)
    samples0 = JokerSamples.read(f[name])

#with prior.model:
mcmc_init = joker.setup_mcmc(data, samples0)
trace = pm.sample(tune=1000, draws=1000,
              start=mcmc_init,
              step = xo.get_dense_nuts_step(target_accept=0.95))
print(pm.summary(trace, var_names=prior.par_names))
