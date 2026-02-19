#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 13:44:52 2024

@author: nbijelic
"""

# function to convert Vs30 (m/s) to Z1p0 (km) using the empirical equations 
# California

import numpy as np
import pandas as pd


### plotting
import matplotlib.pyplot as plt
# change font globally
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams.update({'font.size': 14})
# plotly
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
import seaborn as sns


#%%

def vs30_to_z1p0(vs30):
    """ Function relating Vs30(m/s) to Z1p0 (km) following ASK14 GMPE for 
    California and non-Japan regions.
    
    Parameters
    ----------
    vs30: float, vs30 value in m/s
    
    
    Returns
    -------
    float, Z1p0 value in km
    
    """
    
    z1_ref = 1/1000*np.exp( -7.67/4 * np.log( (vs30**4 + 610**4) / ( 1360**4 + 412**4 ) )  )
    return z1_ref


    
    
    
    
    
    
