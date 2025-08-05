#!/usr/bin/env python

#SBATCH --job-name=unc_generate-0
#SBATCH --error=logs/carma%j.err
#SBATCH --output=logs/slurm%j.out
#SBATCH --account=pi-abbot
#SBATCH --nodes=1
#SBATCH --exclusive

# %%

def k_bootstrap_Mp_Rp(
    X,
    y_Mp,
    y_Rp,
    sample_sizes,
    n_bootstrap,
    mu_mp,
    mu_rp,
):
    from scipy.stats import linregress
    import numpy as np

    from cosmic_shoreline import CosmicShoreline

    cs = CosmicShoreline(data_path='./data-interpolation/')

    """
    Evaluate how linear regression parameters vary with sample size using bootstrap.
    
    Parameters
    ----------
    X : array-like of shape (N,) or (N, d)
        Features from the original sample of size N=10000.
    y : array-like of shape (N,)
        Target/response from the original sample of size N=10000.
    sample_sizes : list of int
        List of sample sizes for which we want to test the linear model.
    n_bootstrap : int
        Number of bootstrap iterations for each sample size.
    
    Returns
    -------
    results : dict
        Dictionary keyed by sample_size, containing statistics (mean, std) for slope, intercept,
        and optionally other metrics.
    """
    
    # y_Mp_rel_err = 10**np.random.normal(np.log10(mu_mp), std_mp, len(X))
    # y_Rp_rel_err = 10**np.random.normal(np.log10(mu_rp), std_rp, len(X))

    # y_Mp_obs = np.zeros(len(X))
    # y_Rp_obs = np.zeros(len(X))
    # for i in range(len(X)):
    #     y_Mp_obs[i] = y_Mp[i] * (1 + y_Mp_rel_err[i]*np.random.normal(0, 1))
    #     y_Rp_obs[i] = y_Rp[i] * (1 + y_Rp_rel_err[i]*np.random.normal(0, 1))

    y_Mp_rel_err = mu_mp*np.random.normal(0, 1, len(X))
    y_Rp_rel_err = mu_rp*np.random.normal(0, 1, len(X))


    y_Mp_obs = y_Mp * (1 + y_Mp_rel_err)
    y_Rp_obs = y_Rp * (1 + y_Rp_rel_err)


    y = cs.M_R_fit(y_Mp_obs, x_M_or_R='M', type='Earth')**3/y_Rp_obs**3
    
    X = np.array(X)
    y = np.array(y)


    slope_mean_arr = np.zeros(len(sample_sizes))
    slope_conf_arr = np.zeros((len(sample_sizes),2))

    for i, size in enumerate(sample_sizes):
        slopes = []
        intercepts = []
        
        # Bootstrap loop: sample with replacement and fit the line
        for _ in range(n_bootstrap):
            idx = np.random.choice(range(len(X)), size=size)
            X_resample = X[idx]
            y_resample = y[idx]
            
            slope_b, intercept_b, _, _, _ = linregress(X_resample, y_resample)
            slopes.append(slope_b)
            intercepts.append(intercept_b)
        
        boot_slopes = np.array(slopes)

        # Get the mean and confidence intervals (90%)
        slope_mean = np.mean(boot_slopes)
        slope_conf = np.percentile(boot_slopes, [10, 90])
        
        slope_mean_arr[i] = slope_mean
        slope_conf_arr[i] = slope_conf
        
    return slope_mean_arr, slope_conf_arr


if __name__ == '__main__':
    import numpy as np
    import scipy.interpolate
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    import os
    import sys
    import multiprocessing


    # %% [markdown]
    # # 3. Density Test

    # %%
    # Define the function to calculate remaining mass

    prefix = '_insol'
    earth_mass = 5.972e24  # in kg
    earth_radius = 6.371e6  # in m

    from cosmic_shoreline import CosmicShoreline

    cs = CosmicShoreline(data_path='./data-interpolation/')

    data_no_comp = pd.read_csv('./data-montecarlo/rho_data_no_comp%s.csv'% prefix)
    data_1sigma = pd.read_csv('./data-montecarlo/rho_data_1sigma%s.csv'% prefix)
    data_3sigma = pd.read_csv('./data-montecarlo/rho_data_3sigma%s.csv'% prefix)
    data_uni = pd.read_csv('./data-montecarlo/rho_data_uni%s.csv'% prefix)
    data_uni_rho = pd.read_csv('./data-montecarlo/rho_data_uni_rho%s.csv'% prefix)


    # create pool of ncpus workers
    # get number of cpus available to job
    try:
        num_processes = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
    except KeyError:
        num_processes = multiprocessing.cpu_count()
    p = multiprocessing.Pool(num_processes)

    print("Number of processes: ", num_processes)



    # %%
    from scipy.stats import linregress
    # # %%


    # report the time taken
    import time
    start_time = time.time()

    k_teq, k_conf_teq = k_bootstrap_Mp_Rp(data_no_comp['pl_teq'], data_no_comp['pl_masse'], data_no_comp['pl_rade_transit_upper'], 
                                np.logspace(1, 3.5, 20).astype(int), 1000, 0, 0)
    N_T = 10**scipy.interpolate.interp1d(k_conf_teq[:,0],np.log10(np.logspace(1, 3.5, 20).astype(int)),bounds_error=False,fill_value='extrapolate')(0)
    print("N_T for no_comp: ", N_T)


    sample_size = np.logspace(1, 3.5, 10).astype(int)
    mass_err_mu = np.logspace(-2, -0.5, 50)
    rad_err_mu = np.logspace(-2.5, -1, 50)
    N_T_max = np.zeros([len(mass_err_mu),len(rad_err_mu)])

    input_items= []

    start_time = time.time()

    for i, mass_err in enumerate(mass_err_mu):
        for j, rad_err in enumerate(rad_err_mu):
            input_items.append((data_no_comp['pl_teq'], data_no_comp['pl_masse'], data_no_comp['pl_rade_transit_upper'], 
                                sample_size, 1000, mass_err, rad_err))
            

    results = p.starmap(k_bootstrap_Mp_Rp, input_items)
    



    for i, (k_teq, k_conf_teq) in enumerate(results):
        N_T = 10**scipy.interpolate.interp1d(k_conf_teq[:,0],np.log10(sample_size),bounds_error=False,fill_value='extrapolate')(0)
        N_T_max[i//len(rad_err_mu), i%len(rad_err_mu)] = N_T

    np.save('./data-montecarlo/N_T_max_no_comp%s.npy'%prefix,N_T_max)

    
    # print the time taken
    print("Time taken for task1: %s seconds" % (time.time() - start_time))

    #================================================================

    # start_time = time.time()

    # sample_size = np.logspace(3, 4, 20).astype(int)
    # mass_err_mu = np.logspace(-2, -0.5, 50)
    # rad_err_mu = np.logspace(-2.5, -1, 50)
    # N_T_max = np.zeros([len(mass_err_mu),len(rad_err_mu)])

    # input_items= []

    # for i, mass_err in enumerate(mass_err_mu):
    #     for j, rad_err in enumerate(rad_err_mu):
    #         input_items.append((data_1sigma['pl_teq'], data_1sigma['pl_masse'], data_1sigma['pl_rade_transit_upper'], 
    #                             sample_size, 1000, mass_err, rad_err))
    # results = p.starmap(k_bootstrap_Mp_Rp, input_items)
    # for i, (k_teq, k_conf_teq) in enumerate(results):
    #     N_T = 10**scipy.interpolate.interp1d(k_conf_teq[:,0],np.log10(sample_size),bounds_error=False,fill_value='extrapolate')(0)
    #     N_T_max[i//len(rad_err_mu), i%len(rad_err_mu)] = N_T

    # np.save('./data-montecarlo/N_T_max_1sigma%s.npy'%prefix,N_T_max)

    # print("Time taken for task2: %s seconds" % (time.time() - start_time))
    # #================================================================

    start_time = time.time()
    sample_size = np.logspace(1, 4, 10).astype(int)
    mass_err_mu = np.logspace(-2, -0.5, 50)
    rad_err_mu = np.logspace(-2.5, -1, 50)
    N_T_max = np.zeros([len(mass_err_mu),len(rad_err_mu)])

    input_items= []
    for i, mass_err in enumerate(mass_err_mu):
        for j, rad_err in enumerate(rad_err_mu):
            input_items.append((data_3sigma['pl_teq'], data_3sigma['pl_masse'], data_3sigma['pl_rade_transit_upper'], 
                                sample_size, 1000, mass_err, rad_err))
    results = p.starmap(k_bootstrap_Mp_Rp, input_items)
    for i, (k_teq, k_conf_teq) in enumerate(results):
        N_T = 10**scipy.interpolate.interp1d(k_conf_teq[:,0],np.log10(sample_size),bounds_error=False,fill_value='extrapolate')(0)
        N_T_max[i//len(rad_err_mu), i%len(rad_err_mu)] = N_T

    np.save('./data-montecarlo/N_T_max_3sigma%s.npy'%prefix,N_T_max)
    print("Time taken for task3: %s seconds" % (time.time() - start_time))
    #================================================================
