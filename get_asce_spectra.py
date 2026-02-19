#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 18:07:46 2024

@author: nbijelic

Script computes the ASCE design/mce spectra given the input values.

"""


#%% Import modules

import numpy as np
import pandas as pd
import scipy
import copy
from pathlib import Path

### plotting
import matplotlib.pyplot as plt
# change font globally
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams.update({'font.size': 14})

import seaborn as sns

#%% Function to compute the ASCE design spectrum

def get_asce_spectrum(s_ds, s_d1, T):
    """ Function compute the two point spectra according to ASCE 7-10, 7-16 
    
    Parameters
    ----------
    s_ds : float
    s_d1 : float
    T : array, periods at which to compute the spectra
    
    
    
    Returns
    -------
    array of floats corresponding to the ASCE design spectrum
    
    """
    T_L = 8 # long-period transition period
    T_s = s_d1/s_ds
    T_0 = 0.2*T_s

    # compute the sa at specified period values
    sa = []
    for per in T:
        if( per <= T_0 ):
            sa.append( s_ds*( 0.4 + 0.6*per/T_0 ) )
        elif( per <= T_s ):
            sa.append( s_ds )
        elif( per <= T_L ):
            sa.append( s_d1 / per )
        else:
            sa.append( s_d1*T_L / per**2 )
    
    return np.array(sa)


T_plot = np.geomspace(0.001, 9, 1000)
s_ds_plot = 1.47
s_d1_plot = 0.76

sa_asce_plot = get_asce_spectrum(s_ds_plot, s_d1_plot, T_plot)

plt.figure(dpi = 200)
plt.plot(T_plot, sa_asce_plot, '-k')
plt.xlim(0, 9)
plt.ylim(0, 1.5)

plt.figure(dpi = 200)
plt.plot(T_plot, 1.5*sa_asce_plot, '-k')
plt.xlim(0, 9)
plt.ylim(0, 2.5)

#%% ASCE design spectrum -- LA site, selection for Nikos

T_plot = np.geomspace(0.001, 10, 1000)
s_ds_plot = 1.00
s_d1_plot = 0.60

sa_asce_plot = get_asce_spectrum(s_ds_plot, s_d1_plot, T_plot)

plt.figure(dpi = 200)
plt.plot(T_plot, sa_asce_plot, '-k', label = 'DBE')
plt.plot(T_plot, 1.5*sa_asce_plot, '--k', label = 'MCEr')


plt.xlabel('T (s)')
plt.ylabel('Sa(T) (g)')
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.01, 10)
plt.ylim(0.01, 5.)
plt.legend()
plt.show()

#%% Compute sa_avg value based on the DBE and MCE spectra

## 4-story EBF
# T_start = 0.2
# T_end = 3.3
# delta_T = 0.1

## 8-story EBF
T_start = 0.5
T_end = 7.5
delta_T = 0.1

##
s_ds = 1.00
s_d1 = 0.60

T_sa_avg = np.arange(T_start, T_end + 0.00001, delta_T)

sa_T_sa_avg = get_asce_spectrum(s_ds, s_d1, T_sa_avg)

plt.figure(dpi = 200)
plt.plot(T_plot, sa_asce_plot, '-k', label = 'DBE')
plt.plot(T_plot, 1.5*sa_asce_plot, '--k', label = 'MCEr')
# add points where averaging is performed
plt.plot(T_sa_avg, sa_T_sa_avg, 'xr')
# add boundaries for the averaging range
plt.plot([T_start, T_start], [0.001, 5], ':k', alpha = 0.7)
plt.plot([T_end, T_end], [0.001, 5], ':k', alpha = 0.7)

plt.xlabel('T (s)')
plt.ylabel('Sa(T) (g)')
# plt.xlim(0., 10)
# plt.ylim(0., 2.)
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.01, 10)
plt.ylim(0.01, 5.)
plt.legend()
plt.show()

#%%

from scipy.stats import gmean
print(f'SaAvg (DBE) = {gmean(sa_T_sa_avg)}')
print(f'SaAvg (MCE) = {1.5*gmean(sa_T_sa_avg)}')


#%% Compute the average from the multi-period response spectra

# computation for Hiro, Seattle site, MCEr spectrum
t_spectra = np.array([0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 
                      0.3, 0.4, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 7.5, 10])
sa_spectra = np.array([0.82, 0.82, 0.84, 0.86, 0.95, 1.14, 1.33, 1.59, 1.82, 
                       1.9, 1.93, 1.87, 1.69, 1.52, 1.39, 0.97, 0.75, 0.48, 
                       0.35, 0.27, 0.17, 0.12 ])

t_interp = np.linspace(start = 0.3, stop = 5, num = 48)
sa_interp = np.exp(np.interp(x = np.log(t_interp), 
                      xp = np.log(t_spectra), fp = np.log(sa_spectra) ))


fig, ax = plt.subplots(dpi = 200)
ax.plot(t_spectra, sa_spectra, '-k')
ax.plot(t_interp, sa_interp, 'or')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([0.1, 10])
ax.set_ylim([0.01, 5])

print(f'SaAvg (MCEr) = {gmean(sa_interp)}')








