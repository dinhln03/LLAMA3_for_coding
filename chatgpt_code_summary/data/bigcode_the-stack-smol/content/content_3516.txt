"""
This script:
- Train Sobolev Alignment.
- Save the two networks.
"""

import os, sys, getopt
import pandas as pd
import numpy as np
import re
from anndata import AnnData
import torch
from pickle import dump, load
from copy import deepcopy
import gc

from sobolev_alignment import SobolevAlignment

# Import params
from model_III_synthetic_params import *
from read_data import read_data


# Import parameters
n_artificial_samples = None
tmp_file = None
opts, args = getopt.getopt(sys.argv[1:],'o:d:n:t:j:p:',['output=', 'data=', 'artifsamples=', 'temp=', 'job=', 'perm='])
for opt, arg in opts:
    if opt in ("-o", "--output"):
        output_folder = str(arg)
    elif opt in ("-d", "--data"):
        data_subfolder = str(arg)
    elif opt in ('-n', '--artifsamples'):
        n_artificial_samples = int(arg)
    elif opt in ('-t', '--temp'):
        tmp_file = str(arg)
    elif opt in ('-j', '--job'):
        n_jobs = int(arg)
n_artificial_samples = n_artificial_samples if n_artificial_samples is not None else 10**6
n_artificial_samples = int(n_artificial_samples)
tmp_file = tmp_file if tmp_file is not None else '/tmp/SM/'

###
# IMPORT DATA
###

X_source, X_target = read_data(data_folder, data_subfolder)
gc.collect()

###
# Sobolev Alignment start
###

# Read best parameters
cell_line_scvi_params, tumor_scvi_params = read_scvi_params(output_folder)

sobolev_alignment_clf = SobolevAlignment(
    source_scvi_params=cell_line_scvi_params,
    target_scvi_params=tumor_scvi_params,
    source_krr_params=default_krr_params,
    target_krr_params=default_krr_params,
    n_jobs=n_jobs
)

###
# Training Sobolev Alignment if not already saved.
###

if 'sobolev_alignment_model' not in os.listdir(output_folder): 
    pass
else:
    sys.exit("VAE ALREADY TRAINED")

sobolev_alignment_clf.n_jobs = n_jobs
sobolev_alignment_clf.fit(
    X_source=X_source,
    X_target=X_target,
    source_batch_name=batch_name,
    target_batch_name=batch_name,
    continuous_covariate_names=continuous_covariate_names,
    n_artificial_samples=100,
    fit_vae=True,
    sample_artificial=False,
    krr_approx=False,
    n_samples_per_sample_batch=10**6,
    frac_save_artificial=1.,
    save_mmap=tmp_file,
    log_input=log_input,
    no_posterior_collapse=no_posterior_collapse,
    frob_norm_source=frob_norm_source
)

if 'sobolev_alignment_model' not in os.listdir(output_folder):
    sobolev_alignment_clf.save('%s/sobolev_alignment_model/'%(output_folder), with_krr=False)
    gc.collect()

    # Save embedding
    for x in sobolev_alignment_clf.scvi_models:
        np.savetxt(
            '%s/scvi_embedding_%s.csv'%(output_folder, x),
            sobolev_alignment_clf.scvi_models[x].get_latent_representation()
        )

torch.cuda.empty_cache()
gc.collect()
sys.exit("FINISH VAE TRAINING")