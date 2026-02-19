#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 11:48:47 2023

@author: nbijelic
"""

""" 
Script implementing functions for conditional spectra (CS) computations, 
including sa_avg as the conditioning IM. The CS target computation allows for 
combining the contributions of multiple ruptures (i.e., magnitude and distance
pairs).

In addition, the GMPEs are implemented here as well and the computation 
examples are also provided for reference (see note below for examples).

Note: the previous example usage is saved in the main_hazard_utils_backup.py 
script. Moving forward, I will load the functions from this file in order to do
the CS target computations. All other development of the hazard utils is done 
in this file.

"""

# TODO
#     - make sure that lambda (the rake angle), Rup and Site are consistent 
#       between GMPEs so that I can use the same Rup and Site objects !!!
#       - this is especially for the as_2016 duration model and the 
#         Sa GMPEs I am using (Campbel & Bozorgnia 2014)





# TODO for improvements:
# (1) Add sample cs realization function to the ConditionalSpectra object
#  - this can be used to generate the realizations of spectra from multivariate
#    gaussian defined by the CS-target



# rup_dict :: dict with rupture properties, needs to have all that GMPE needs
# seis_dict :: dict with seismological properties, all info needed by the GMPE

# -----------------------------------------------------------------------------
# Import modules, function definitions
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import scipy
import copy
from pathlib import Path
from os.path import join
from scipy.interpolate import interpn

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

from abrahamson_gulerce_2020_gmpe import AbrahamsonGulerce2020
from bahrampouri_2021_duration_interface_gmpe import BahrampouriEtAlSInter2020_duration
from bahrampouri_2021_duration_interface_gmpe import BahrampouriEtAlSSlab2020_duration


from abrahamson_2015_usgs_basin import AbrahamsonEtAl2015SInter


# -----------------------------------------------------------------------------
# function parsing the openQuake hazard curve computation xml file
# -----------------------------------------------------------------------------

import re # regular expressions
import xml.etree.ElementTree as ET # xml parser

def parse_oq_hazard_xml(xml_file_in): 
    # TODO :
    # Document inputs and outputs
        
    mytree = ET.parse(xml_file_in)
    ### Parse IMLs
    IML_str = []
    for x in mytree.iter('{http://openquake.org/xmlns/nrml/0.5}IMLs'):
        IML_str.append(x.text)
    s = IML_str[0]
    p = re.compile(r'\d+\.\d+[E][+-]\d+')  # Compile a pattern to capture float values
    floats = [float(i) for i in p.findall(s)]  # Convert strings to float
    # floats = [i for i in p.findall(s)]  # parsed strings -- just for control
    # print(floats) # print for control
    
    imls = np.array(floats) # these are the parsed IMLs, should correspond to all 
        # hazard curves in the file
        
    ### Parse poEs (probabilities of exceedance for each site -- output from openQuake)
    POE_str = []
    for x in mytree.iter('{http://openquake.org/xmlns/nrml/0.5}poEs'):
        POE_str.append(x.text)
    
    poe_list = []
    
    for s in POE_str:
        p = re.compile(r'\d+\.\d+[E][+-]\d+')  # Compile a pattern to capture float values
        floats = [float(i) for i in p.findall(s)]  # Convert strings to float
        # floats = [i for i in p.findall(s)]  # parsed strings -- just for control
        # print(floats) # print for control
        poe_list.append(np.array(floats)) # these are the parsed exceedances
    
    ### Parse lon/lat for all sites
    lonlat_str = []
    for x in mytree.iter('{http://www.opengis.net/gml}pos'):
        lonlat_str.append(x.text)
    
    lonlat_list = []
    
    for s in lonlat_str:
        p = re.compile(r'\d+\.\d+')  # Compile a pattern to capture float values
        floats = [float(i) for i in p.findall(s)]  # Convert strings to float
        # floats = [i for i in p.findall(s)]  # parsed strings -- just for control
        # print(floats) # print for control
        lonlat_list.append(np.array(floats)) # these are the parsed exceedances
    
    ### convert poEs to rates
    
    # parse the investigation time from the attributes of the hazard curve lists
    # later in the function it is assumed I only have one investigation time
    T_years_lst = []
    for x in mytree.iter('{http://openquake.org/xmlns/nrml/0.5}hazardCurves'):
        T_years_lst.append( float(x.attrib['investigationTime']) )
    
    T_years = T_years_lst[0]
    rate_list = []
    for poe in poe_list:
        rate_list.append( -np.log(1 - poe)/T_years )
        
    # store output to a dict and return    
    dict_out = {'T_years' : T_years,
                'IMLs' : imls,
                'lonLat_lst' :lonlat_list,
                'poe_lst' : poe_list,
                'rate_lst' : rate_list
                }
    
    return dict_out

# -----------------------------------------------------------------------------
# baker_jayaram correlation function
# -----------------------------------------------------------------------------

def sa_corr_baker(T_i, T_j):
    """correlation coefficient for SA at different T_i, T_j, Baker and Jayaram (2008)"""
    
    # Convert to scalar if array-like (handles both scalars and arrays)
    T_i_scalar = float(np.atleast_1d(T_i).flatten()[0])
    T_j_scalar = float(np.atleast_1d(T_j).flatten()[0])
    
    T_min = np.min([T_i_scalar, T_j_scalar])
    T_max = np.max([T_i_scalar, T_j_scalar])
        
    C_1 = 1-np.cos( np.pi/2 - 0.366*np.log( T_max / ( np.max( [T_min, 0.109] ) ) ) )
    
    if(T_max < 0.2):
        C_2 = 1 - 0.105*( 1 - 1 / (1 + np.exp(100*T_max - 5) )  )*( (T_max - T_min) / (T_max - 0.0099) )
    else:
        C_2 = 0
    
    if(T_max < 0.109):
        C_3 = C_2
    else:
        C_3 = C_1
        
    C_4 = C_1 + 0.5*( np.sqrt(C_3) - C_3 )*(1 + np.cos( np.pi*T_min / 0.109 ))
    
    # compute correlation coefficient: rho(T_i, T_j)
    if(T_max < 0.109):
        rho = C_2
    elif(T_min > 0.109):
        rho = C_1
    elif(T_max < 0.2):
        rho = np.min([C_2, C_4])
    else:
        rho = C_4
        
    return rho

# -----------------------------------------------------------------------------
# da5_75_sa_corr_bradley_2011(T_j) 
# -----------------------------------------------------------------------------

def da5_75_sa_corr_bradley_2011(T):
    """ Correlation between significant duration (Da5_75) and Sa(T) intensity
    measures. Based on Equation (12) in:
    Bradley B. (2011): "Correlation of Significant Duration with Amplitude and
         Cumulative Intensity Measures and Its Use in Ground Motion Selection",
         Journal of Earthquake Engineering, 15:809–832, 2011
        
    Parameters
    ----------
    T : float, period of the Sa(T)
    
    Returns
    -------
    float, correlation coefficient
         
    """
    
    # get coefficients for the correlation equation based on value of T
    b_n_all = np.array([0.01, 0.09, 0.3, 1.4, 6.5, 10.001])
    a_n_all = np.array([-0.45, -0.39, -0.39, -0.06, 0.16, 0.00])
    # get indices to use by comparing
    bn_low = np.array([np.max( b_n_all[b_n_all <= T] )])
    bn_hi = np.array([np.min( b_n_all[b_n_all > T] )])
    
    # idx_low = b_n_all.index(bn_low)
    # idx_hi = b_n_all.index(bn_hi)
    idx_low = np.where(b_n_all == bn_low)[0][0]
    # print(idx_low)
    idx_hi = np.where(b_n_all == bn_hi)[0][0]
    # print(idx_hi)
    
    
    # coefficients for the correlation function
    a_n_1 = a_n_all[idx_low]
    a_n = a_n_all[idx_hi]
    b_n_1 = b_n_all[idx_low]
    b_n = b_n_all[idx_hi]
    # compute the correlation coefficient
    rho = a_n_1 + ( np.log(T/b_n_1) / np.log(b_n/b_n_1) ) * (a_n - a_n_1)
    return rho

# -----------------------------------------------------------------------------
# Epsilon correlation, Al Atik (2011)
# -----------------------------------------------------------------------------

def sa_corr_alatik(T_i, T_j):
    """ Epsilon correlation coefficient based on the paper:
    
    Al Atik, L. (2011). "Correlation of spectral acceleration values for 
    subduction and crustal models, COSMOS Technical Session. Emeryville, 
    California, 4 November 2011.
    
    Parameters
    ----------
    T_i, T_j : floats, periods for which to compute the correlation coefficient
    
    Returns
    -------
    
    float, correlation coefficient
    """
    
    
    T_min = np.min([T_i, T_j])
    T_max = np.max([T_i, T_j])
    
    # return NaN if outside of the range of application
    if(T_min < 0.01 or T_max > 5.0):
        return np.nan
        # here I can plug in the Baker Jayaram correlation function just to see
        # the comparison in the spectra without breaking the code
    
    # Period points
    periods_interp = np.array([0.01, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 
                               0.4, 0.5, 0.6, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5])
    
    rho_coefs = np.array([
        [1, 0.9594, 0.9376, 0.9303, 0.93, 0.9102, 0.8866, 0.8579, 0.8169, 0.7289, 0.6482, 0.5801, 0.4641, 0.3913, 0.309, 0.2744, 0.2795, 0.2231, 0.2307],
        [0.9594, 1, 0.971, 0.9415, 0.8996, 0.8548, 0.8198, 0.783, 0.7166, 0.6342, 0.5478, 0.4768, 0.4085, 0.2995, 0.218, 0.2028, 0.1865, 0.1487, 0.1679],
        [0.9376, 0.971, 1, 0.9677, 0.9089, 0.8487, 0.7984, 0.7543, 0.6795, 0.5842, 0.4975, 0.427, 0.3532, 0.2471, 0.1751, 0.1618, 0.1558, 0.1208, 0.1347],
        [0.9303, 0.9415, 0.9677, 1, 0.9346, 0.8737, 0.8058, 0.7534, 0.6988, 0.5761, 0.4869, 0.4159, 0.2836, 0.2295, 0.1669, 0.1462, 0.1622, 0.1123, 0.1313],
        [0.93, 0.8996, 0.9089, 0.9346, 1, 0.9338, 0.8702, 0.8124, 0.7246, 0.624, 0.5358, 0.4643, 0.3842, 0.2693, 0.1903, 0.1746, 0.1709, 0.1404, 0.1468],
        [0.9102, 0.8548, 0.8487, 0.8737, 0.9338, 1, 0.9411, 0.8847, 0.8101, 0.6998, 0.6138, 0.5354, 0.4188, 0.3256, 0.278, 0.2282, 0.2452, 0.181, 0.184],
        [0.8866, 0.8198, 0.7984, 0.8058, 0.8702, 0.9411, 1, 0.9516, 0.8601, 0.7659, 0.6785, 0.5977, 0.5028, 0.383, 0.301, 0.2763, 0.256, 0.2248, 0.22],
        [0.8579, 0.783, 0.7543, 0.7534, 0.8124, 0.8847, 0.9516, 1, 0.9174, 0.8256, 0.7427, 0.6643, 0.5636, 0.4369, 0.3542, 0.3278, 0.3007, 0.2694, 0.2549],
        [0.8169, 0.7166, 0.6795, 0.6988, 0.7246, 0.8101, 0.8601, 0.9174, 1, 0.9213, 0.8425, 0.7619, 0.6272, 0.5306, 0.4328, 0.4118, 0.3713, 0.3315, 0.3061],
        [0.7289, 0.6342, 0.5842, 0.5761, 0.624, 0.6998, 0.7659, 0.8256, 0.9213, 1, 0.9392, 0.8637, 0.7561, 0.6156, 0.5217, 0.4832, 0.4361, 0.3919, 0.3637],
        [0.6482, 0.5478, 0.4975, 0.4869, 0.5358, 0.6138, 0.6785, 0.7427, 0.8425, 0.9392, 1, 0.9307, 0.8176, 0.6817, 0.5811, 0.5316, 0.4812, 0.4357, 0.4],
        [0.5801, 0.4768, 0.427, 0.4159, 0.4643, 0.5354, 0.5977, 0.6643, 0.7619, 0.8637, 0.9307, 1, 0.9043, 0.772, 0.6765, 0.6241, 0.5707, 0.5264, 0.4983],
        [0.4641, 0.4085, 0.3532, 0.2836, 0.3842, 0.4188, 0.5028, 0.5636, 0.6272, 0.7561, 0.8176, 0.9043, 1, 0.8793, 0.7771, 0.7287, 0.6714, 0.6288, 0.5863],
        [0.3913, 0.2995, 0.2471, 0.2295, 0.2693, 0.3256, 0.383, 0.4369, 0.5306, 0.6156, 0.6817, 0.772, 0.8793, 1, 0.9243, 0.8663, 0.8156, 0.754, 0.7093],
        [0.309, 0.218, 0.1751, 0.1669, 0.1903, 0.278, 0.301, 0.3542, 0.4328, 0.5217, 0.5811, 0.6765, 0.7771, 0.9243, 1, 0.9474, 0.901, 0.8455, 0.7984],
        [0.2744, 0.2028, 0.1618, 0.1462, 0.1746, 0.2282, 0.2763, 0.3278, 0.4118, 0.4832, 0.5316, 0.6241, 0.7287, 0.8663, 0.9474, 1, 0.9653, 0.901, 0.8609],
        [0.2795, 0.1865, 0.1558, 0.1622, 0.1709, 0.2452, 0.256, 0.3007, 0.3713, 0.4361, 0.4812, 0.5707, 0.6714, 0.8156, 0.901, 0.9653, 1, 0.9463, 0.9062],
        [0.2231, 0.1487, 0.1208, 0.1123, 0.1404, 0.181, 0.2248, 0.2694, 0.3315, 0.3919, 0.4357, 0.5264, 0.6288, 0.754, 0.8455, 0.901, 0.9463, 1, 0.9611],
        [0.2307, 0.1679, 0.1347, 0.1313, 0.1468, 0.184, 0.22, 0.2549, 0.3061, 0.3637, 0.4, 0.4983, 0.5863, 0.7093, 0.7984, 0.8609, 0.9062, 0.9611, 1]])
    
    points = (periods_interp, periods_interp)
    values = rho_coefs
    rho = interpn( points, values, (T_i, T_j) )

    
    return rho[0]





# -----------------------------------------------------------------------------
# IMPLEMENTING the GMMs
# Note: follow and implement the approach from: https://github.com/bakerjw/GMMs
# -----------------------------------------------------------------------------
    
class Rup(object):
    """Class of rupture object to be used as inputs to GMMs; 
    defines rupture class and properties"""

    ## Class attributes that need to be specified:
    #     M : Magnitude
    #     R : Generic distance
    
    ## Class attributes that can be defined with kwargs (allowed keywords):
    #     Rrup : Closest distance to fault rupture
    #     Rjb : Joyner-Boore distance
    #     Rhyp : Hypocentral distance
    #     Rx : Perpendicular distance to projection of rupture edge
    #     Ry0 : Parallel distance off end of rupture
    #     HW : Hanging wall indicator = 1 for hanging wall, 0 otherwise
    #     AS : Aftershock indicator   = 1 for aftershock, 0 otherwise
    #     Ztor : Depth to top of rupture
    #     Zhyp : Hypocentral depth
    #     h_eff : Effective height
    #     W : Down-dip width
    #     delta : Average dip angle
    #     lam : Rake angle (used to determine fault types)
    #
    #     hypo_depth : hypocentral depth (km), used in Kotha et al. (2020)
        
    def __init__(self, M, R, **kwargs):
        self.M = M 
        self.R = R 
        # set class attributes from kwargs, see: https://stackoverflow.com/questions/8187082/how-can-you-set-class-attributes-from-variable-arguments-kwargs-in-python
        # self.__dict__.update(kwargs)
        
        # set class attributes from kwargs, specifying allowed keys: https://stackoverflow.com/questions/8187082/how-can-you-set-class-attributes-from-variable-arguments-kwargs-in-python
        allowed_keys = { 'Rrup', 'Rjb', 'Rhyp', 'Rx', 'Ry0', 'HW', 'AS', 'Ztor',
                        'Zhyp', 'h_eff', 'W', 'delta', 'lam', 'hypo_depth', 
                        'Zbot', 'delta', 'Fhw', 'trt'}
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)

    
class Site(object):
    """Defines site class and properties"""
    
    ## Class attribute that needs to be specified:
    #     Vs30 : Average shear wave velocity over top 30 m
    
    ## Class attributes that can be defined with kwargs (allowed keywords):
    #     is_soil : Flag for soil conditions
    # %               = 0 for soil
    # %               = 1 for soft rock
    # %               = 2 for hard rock
    #         
    #     fvs30 : Flag for shear wave velocity
    # %               = 0 for Vs30 inferred from geology
    # %               = 1 for measured Vs30
    #     Z25 : Depth to Vs = 2.5km/s
    #     Z10 : Depth to Vs = 1.0km/s
    #     Zbot : Depth to bottom of the seismogenic crust
    #     region : Country
    # %               = 0 for global
    # %               = 1 for California
    # %               = 2 for Japan
    # %               = 3 for China
    # %               = 4 for Italy
    # %               = 5 for Turkey
    # %               = 6 for Taiwan
    #
    #     hypo_depth : hypocentral depth (km), for Kotha et al. (2020)
        
    def __init__(self, Vs30, **kwargs):
        self.Vs30 = Vs30
        # set class attributes from kwargs, specifying allowed keys: https://stackoverflow.com/questions/8187082/how-can-you-set-class-attributes-from-variable-arguments-kwargs-in-python
        allowed_keys = { 'is_soil', 'fvs30', 'Z25', 'Z10', 'Zbot', 'region', 
                        'hypo_depth', 'A1100', 'backarc'}
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)

# -----------------------------------------------------------------------------
# Campbell & Bozorgnia 2014 GMPE 
# -----------------------------------------------------------------------------

class CB_2014_active(object):
    """Ground motion model (GMM) class for the Campbell&Bozorgnia 2014 model"""
    ###########################################################################
    # Based on the matlab code by:
    # % Created by Yue Hua,  4/22/14, yuehua@stanford.edu
    # % Modified 4/5/2015 by Jack Baker to fix a few small bugs
    # % Modified 12/9/2019 by Jack Baker to update a few c6 coefficients that 
    # %   changed from the PEER report to the Earthquake Spectra paper (thank you
    # %   Yenan Cao for noting this)
    # % Updated by Emily Mongold, 11/27/20
    # % Modified on 5/3/22 by Emily Mongold to prevent estimating Ztor values
    # %   within width and Zhyp calculations when user provided (thanks to Tom
    # %   Son for noting this)
    # %
    # % Source Model:
    # % Campbell, K. W., and Bozorgnia, Y. (2014). "NGA-West2 Ground Motion Model 
    # % for the Average Horizontal Components of PGA, PGV, and 5% Damped Linear 
    # % Acceleration Response Spectra." Earthquake Spectra, 30(3), 1087-1115.
    # % 
    # % Provides ground-motion prediction equations for computing medians and
    # % standard deviations of average horizontal components of PGA, PGV and 5%
    # % damped linear pseudo-absolute aceeleration response spectra
    # 
    # % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # % INPUT
    # %   T             = Period (sec);
    # %                   Use 1000 for output the array of Sa with period
    # %   Rup           = rupture object input containing the following
    # %                   variables:
    # %       M             = Magnitude
    # %       Rrup          = Closest distance coseismic rupture (km)
    # %       Rjb           = Joyner-Boore distance (km)
    # %       Rx            = Closest distance to the surface projection of the
    # %                       coseismic fault rupture plane
    # %       W             = down-dip width of the fault rupture plane
    # %       Ztor          = Depth to the top of coseismic rupture (km)
    # %       Zbot          = Depth to the bottom of the seismogenic crust
    # %                       needed only when W is unknown;
    # %       Zhyp          = Hypocentral depth of the earthquake measured from sea level
    # %       delta         = average dip of the rupture place (degree)
    # %       lam           = rake angle (degree) - average angle of slip measured in
    # %                       the plane of rupture
    # %       Fhw           = hanging wall effect
    # %                     = 1 for including
    # %                     = 0 for excluding
    # %   Site          = site object input containing the following
    # %                   variables:
    # %       Vs30          = shear wave velocity averaged over top 30 m (m/s)
    # %       Z25           = Depth to the 2.5 km/s shear-wave velocity horizon (km)
    # %                       set to 'unknown' if not known
    # %       region        = 0 for global
    # %                     = 1 for California
    # %                     = 2 for Japan
    # %                     = 3 for China
    # %                     = 4 for Italy
    # %                     = 5 for Turkey (locally = 3)
    # %                     = 6 for Taiwan (locally = 0)
    # % OUTPUT
    # %   median        = Median spectral acceleration prediction
    # %   sigma         = logarithmic standard deviation of spectral acceleration
    # %                   prediction
    # %   period1       = periods for which the median and sigma values are
    # %                   provided. If T = 1000, then period1 = the full set of
    # %                   available periods. Else period1 = T
    # % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def __init__(self, T, Rup, Site, is_scale_to_rotd100 = True):
        Rup.Frv, Rup.Fnm = self.get_FaultType(Rup) # compute fault type based on the rake angle
        Rup.Ztor = self.get_Ztor(Rup)
        Rup.W, Rup.W_empty_flag = self.get_W(Rup)
        Rup.Zhyp = self.get_Zhyp(Rup)
        Site.A1100 = 999 # Set initial A1100 value to 999. 
            # A1100: median estimated value of PGA on rock with Vs30 = 1100m/s
        self.is_scale_to_rotd100 = is_scale_to_rotd100
        
        self.period1 = self.get_periods(T) # define the IMT, i.e., get periods 
            # at which to compute the Sa values 
        self.median, self.sigma = self.get_median_and_sigma(self.period1, Rup, Site)    


    def fn_rotd50_2_rotd100(self, T):
        """ Function to compute scaling factor which converts RotD50 to RotD100 
        following the approach of Boore and Kishida (2017), specifically using
        equation (2) and coefficients from Table 2. 
        
        Reference
            Boore and Kishida (2017) Relations between some horizontal-component 
            ground-motion intensity measures used in practice    
        
        Parameters
        ----------
        T: float, period for which to compute the scaling factor
        
        Returns
        -------
        float
        
        """
        
        # # Regression coefficients for RotD100/RotD50
        R_i = [0, 1.188, 1.225, 1.241, 1.287, 1.287]
        T_i = [0, 0.12,  0.41,  3.14,  10.0,  0  ]
        
        # Regression coefficients for Larger/RotD50
        # R_i = [0, 1.107, 1.133, 1.149, 1.178, 1.178]
        # T_i = [0, 0.10,  0.45,  4.36,  8.78,  0  ]
        
        # Handle PGA case (T=0) - return the first coefficient
        if T == 0 or T < 1e-6:
            return R_i[1]
            
        # Compute the scaling factors    
        a = R_i[1] + ( ( R_i[2] - R_i[1] ) / np.log(T_i[2]/T_i[1]) )*np.log(T / T_i[1]) 
        b = R_i[2] + ( ( R_i[3] - R_i[2] ) / np.log(T_i[3]/T_i[2]) )*np.log(T / T_i[2]) 
        c = np.min([ a, b ])
        d_1 = R_i[3] + ( ( R_i[4] - R_i[3] ) / np.log(T_i[4]/T_i[3]) )*np.log(T / T_i[3])
        d = np.min([ d_1, R_i[5] ])
        e = np.max([ c, d ])
        ratio = np.max( [R_i[1], e] )
          
        return ratio


    def get_FaultType(self, Rup):
        """Get the style of faulting based on the rake angle lam. """
        lam = Rup.lam
        Frv = ( lam > 30 and lam < 150 )      
        Fnm = ( lam > -150 and lam < -30 )
        return Frv, Fnm


    def get_Ztor(self, Rup):
        """ Compute Ztor if not specified. """
        if not hasattr(Rup, 'Ztor'):
            if(Rup.Frv):
                Ztor = np.max([ 2.704 - 1.226*np.max( [Rup.M-5.849, 0] ), 0] )**2
            else:
                Ztor = np.max( [2.673 - 1.136*np.max( [Rup.M-4.970, 0] ), 0] )**2
        else:
            Ztor = Rup.Ztor            
        return Ztor

    
    def get_W(self, Rup):
        W_empty_flag = 0 # To set rup.W = [] at the end
        if(not hasattr(Rup, 'W')):
            W_empty_flag = 1
            W = np.min( [np.sqrt( 10**( (Rup.M-4.07) / 0.98) ), (Rup.Zbot - Rup.Ztor)/np.sin(np.pi/180*Rup.delta)] )     
        else:
            W = Rup.W
        return W, W_empty_flag
    
    
    def get_Zhyp(self, Rup):
        if( ( not hasattr(Rup, 'Zhyp') and hasattr(Rup, 'W') ) ):
            if(Rup.M < 6.75):
                fdZM = -4.317 + 0.984*Rup.M
            else:
                fdZM = 2.325
                
            if(Rup.delta <= 40):
                fdZD = 0.0445*(Rup.delta - 40)
            else:
                fdZD = 0
                
            Zbor = Rup.Ztor + Rup.W*np.sin(np.pi/180*Rup.delta) # The depth to the bottom of the rupture plane
            d_Z = np.exp( np.min( [fdZM+fdZD, np.log( 0.9*(Zbor-Rup.Ztor) )] ) )
            Zhyp = d_Z + Rup.Ztor
        else:
            Zhyp = Rup.Zhyp
        
        return Zhyp

    def get_periods(self, T):
        """Define computation periods"""
        if( np.any(T == 1000) ):
            # use original periods from the GMM
            return np.array([0.010, 0.020, 0.030, 0.050, 0.075, 0.10, 0.15, 
                             0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0, 
                             1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 0, -1])
        else: return T
     
    def CB_2014_nga_sub(self, Rup, Site_obj, ip):
        Frv = Rup.Frv
        Fnm = Rup.Fnm
        Ztor = Rup.Ztor 
        Zhyp = Rup.Zhyp
        A1100 = Site_obj.A1100
        
        M = Rup.M
        Rrup = Rup.Rrup
        Rjb = Rup.Rjb
        Rx = Rup.Rx
        W = Rup.W
        delta = Rup.delta
        Fhw = Rup.Fhw
        Vs30 = Site_obj.Vs30 
        Z25in = Site_obj.Z25
        
        # set the local variable for region
        if(Site_obj.region == 5):
            region = 3
        elif(Site_obj.region == 6):
            region = 0
        else:
            region = Site_obj.region
        
        # coefficients from  the GMPE     
        c0 = [-4.365, -4.348, -4.024, -3.479, -3.293, -3.666, -4.866, -5.411, -5.962, -6.403, -7.566, -8.379, -9.841, -11.011, -12.469, -12.969, -13.306, -14.02, -14.558, -15.509, -15.975, -4.416, -2.895]
        c1 = [0.977, 0.976, 0.931, 0.887, 0.902, 0.993, 1.267, 1.366, 1.458, 1.528, 1.739, 1.872, 2.021, 2.180, 2.270, 2.271, 2.150, 2.132, 2.116, 2.223, 2.132, 0.984, 1.510]
        c2 = [0.533, 0.549, 0.628, 0.674, 0.726, 0.698, 0.510, 0.447, 0.274, 0.193, -0.020, -0.121, -0.042, -0.069, 0.047, 0.149, 0.368, 0.726, 1.027, 0.169, 0.367, 0.537, 0.270]
        c3 = [-1.485, -1.488, -1.494, -1.388, -1.469, -1.572, -1.669, -1.750, -1.711, -1.770, -1.594, -1.577, -1.757, -1.707, -1.621, -1.512, -1.315, -1.506, -1.721, -0.756, -0.800, -1.499, -1.299]
        c4 = [-0.499, -0.501, -0.517, -0.615, -0.596, -0.536, -0.490, -0.451, -0.404, -0.321, -0.426, -0.440, -0.443, -0.527, -0.630, -0.768, -0.890, -0.885, -0.878, -1.077, -1.282, -0.496, -0.453]
        c5 = [-2.773, -2.772, -2.782, -2.791, -2.745, -2.633, -2.458, -2.421, -2.392, -2.376, -2.303, -2.296, -2.232, -2.158, -2.063, -2.104, -2.051, -1.986, -2.021, -2.179, -2.244, -2.773, -2.466]
        #c6, = [0.248, 0.247, 0.246, 0.240, 0.227, 0.210, 0.183, 0.182, 0.189, 0.195, 0.185, 0.186, 0.186, 0.169, 0.158, 0.158, 0.148, 0.135, 0.140, 0.178, 0.194, 0.248, 0.204], #, these, coefficients, are, in, the, PEER, report
        c6 = [0.248, 0.247, 0.246, 0.240, 0.227, 0.210, 0.183, 0.182, 0.189, 0.195, 0.185, 0.186, 0.186, 0.169, 0.158, 0.158, 0.148, 0.135, 0.135, 0.165, 0.180, 0.248, 0.204] #, these, coefficients, are, in, the, Earthquake, Spectra, paper
        c7 = [6.753, 6.502, 6.291, 6.317, 6.861, 7.294, 8.031, 8.385, 7.534, 6.990, 7.012, 6.902, 5.522, 5.650, 5.795, 6.632, 6.759, 7.978, 8.538, 8.468, 6.564, 6.768, 5.837]
        c8 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        c9 = [-0.214, -0.208, -0.213, -0.244, -0.266, -0.229, -0.211, -0.163, -0.150, -0.131, -0.159, -0.153, -0.090, -0.105, -0.058, -0.028, 0, 0, 0, 0, 0, -0.212, -0.168]
        c10 = [0.720, 0.730, 0.759, 0.826, 0.815, 0.831, 0.749, 0.764, 0.716, 0.737, 0.738, 0.718, 0.795, 0.556, 0.480, 0.401, 0.206, 0.105, 0, 0, 0, 0.720, 0.305]
        c11 = [1.094, 1.149, 1.290, 1.449, 1.535, 1.615, 1.877, 2.069, 2.205, 2.306, 2.398, 2.355, 1.995, 1.447, 0.330, -0.514, -0.848, -0.793, -0.748, -0.664, -0.576, 1.090, 1.713]
        c12 = [2.191, 2.189, 2.164, 2.138, 2.446, 2.969, 3.544, 3.707, 3.343, 3.334, 3.544, 3.016, 2.616, 2.470, 2.108, 1.327, 0.601, 0.568, 0.356, 0.075, -0.027, 2.186, 2.602]
        c13 = [1.416, 1.453, 1.476, 1.549, 1.772, 1.916, 2.161, 2.465, 2.766, 3.011, 3.203, 3.333, 3.054, 2.562, 1.453, 0.657, 0.367, 0.306, 0.268, 0.374, 0.297, 1.420, 2.457]
        c14 = [-0.0070, -0.0167, -0.0422, -0.0663, -0.0794, -0.0294, 0.0642, 0.0968, 0.1441, 0.1597, 0.1410, 0.1474, 0.1764, 0.2593, 0.2881, 0.3112, 0.3478, 0.3747, 0.3382, 0.3754, 0.3506, -0.0064, 0.1060]
        c15 = [-0.207, -0.199, -0.202, -0.339, -0.404, -0.416, -0.407, -0.311, -0.172, -0.084, 0.085, 0.233, 0.411, 0.479, 0.566, 0.562, 0.534, 0.522, 0.477, 0.321, 0.174, -0.202, 0.332]
        c16 = [0.390, 0.387, 0.378, 0.295, 0.322, 0.384, 0.417, 0.404, 0.466, 0.528, 0.540, 0.638, 0.776, 0.771, 0.748, 0.763, 0.686, 0.691, 0.670, 0.757, 0.621, 0.393, 0.585]
        c17 = [0.0981, 0.1009, 0.1095, 0.1226, 0.1165, 0.0998, 0.0760, 0.0571, 0.0437, 0.0323, 0.0209, 0.0092, -0.0082, -0.0131, -0.0187, -0.0258, -0.0311, -0.0413, -0.0281, -0.0205, 0.0009, 0.0977, 0.0517]
        c18 = [0.0334, 0.0327, 0.0331, 0.0270, 0.0288, 0.0325, 0.0388, 0.0437, 0.0463, 0.0508, 0.0432, 0.0405, 0.0420, 0.0426, 0.0380, 0.0252, 0.0236, 0.0102, 0.0034, 0.0050, 0.0099, 0.0333, 0.0327]
        c19 = [0.00755, 0.00759, 0.00790, 0.00803, 0.00811, 0.00744, 0.00716, 0.00688, 0.00556, 0.00458, 0.00401, 0.00388, 0.00420, 0.00409, 0.00424, 0.00448, 0.00345, 0.00603, 0.00805, 0.00280, 0.00458, 0.00757, 0.00613]
        c20 = [-0.0055, -0.0055, -0.0057, -0.0063, -0.0070, -0.0073, -0.0069, -0.0060, -0.0055, -0.0049, -0.0037, -0.0027, -0.0016, -0.0006, 0, 0, 0, 0, 0, 0, 0, -0.0055, -0.0017]
        Dc20 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        Dc20_JI = [-0.0035, -0.0035, -0.0034, -0.0037, -0.0037, -0.0034, -0.0030, -0.0031, -0.0033, -0.0035, -0.0034, -0.0034, -0.0032, -0.0030, -0.0019, -0.0005, 0, 0, 0, 0, 0, -0.0035, -0.0006]
        Dc20_CH = [0.0036, 0.0036, 0.0037, 0.0040, 0.0039, 0.0042, 0.0042, 0.0041, 0.0036, 0.0031, 0.0028, 0.0025, 0.0016, 0.0006, 0, 0, 0, 0, 0, 0, 0, 0.0036, 0.0017]
        a2 = [0.168, 0.166, 0.167, 0.173, 0.198, 0.174, 0.198, 0.204, 0.185, 0.164, 0.160, 0.184, 0.216, 0.596, 0.596, 0.596, 0.596, 0.596, 0.596, 0.596, 0.596, 0.167, 0.596]
        h1 = [0.242, 0.244, 0.246, 0.251, 0.260, 0.259, 0.254, 0.237, 0.206, 0.210, 0.226, 0.217, 0.154, 0.117, 0.117, 0.117, 0.117, 0.117, 0.117, 0.117, 0.117, 0.241, 0.117]
        h2 = [1.471, 1.467, 1.467, 1.449, 1.435, 1.449, 1.461, 1.484, 1.581, 1.586, 1.544, 1.554, 1.626, 1.616, 1.616, 1.616, 1.616, 1.616, 1.616, 1.616, 1.616, 1.474, 1.616]
        h3 = [-0.714, -0.711, -0.713, -0.701, -0.695, -0.708, -0.715, -0.721, -0.787, -0.795, -0.770, -0.770, -0.780, -0.733, -0.733, -0.733, -0.733, -0.733, -0.733, -0.733, -0.733, -0.715, -0.733]
        h4 = [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000]
        h5 = [-0.336, -0.339, -0.338, -0.338, -0.347, -0.391, -0.449, -0.393, -0.339, -0.447, -0.525, -0.407, -0.371, -0.128, -0.128, -0.128, -0.128, -0.128, -0.128, -0.128, -0.128, -0.337, -0.128]
        h6 = [-0.270, -0.263, -0.259, -0.263, -0.219, -0.201, -0.099, -0.198, -0.210, -0.121, -0.086, -0.281, -0.285, -0.756, -0.756, -0.756, -0.756, -0.756, -0.756, -0.756, -0.756, -0.270, -0.756]
        k1 = [865, 865, 908, 1054, 1086, 1032, 878, 748, 654, 587, 503, 457, 410, 400, 400, 400, 400, 400, 400, 400, 400, 865, 400]
        k2 = [-1.186, -1.219, -1.273, -1.346, -1.471, -1.624, -1.931, -2.188, -2.381, -2.518, -2.657, -2.669, -2.401, -1.955, -1.025, -0.299, 0.000, 0.000, 0.000, 0.000, 0.000, -1.186, -1.955]
        k3 = [1.839, 1.840, 1.841, 1.843, 1.845, 1.847, 1.852, 1.856, 1.861, 1.865, 1.874, 1.883, 1.906, 1.929, 1.974, 2.019, 2.110, 2.200, 2.291, 2.517, 2.744, 1.839, 1.929]
        c = 1.88
        n = 1.18
        f1 = [0.734, 0.738, 0.747, 0.777, 0.782, 0.769, 0.769, 0.761, 0.744, 0.727, 0.690, 0.663, 0.606, 0.579, 0.541, 0.529, 0.527, 0.521, 0.502, 0.457, 0.441, 0.734, 0.655]
        f2 = [0.492, 0.496, 0.503, 0.520, 0.535, 0.543, 0.543, 0.552, 0.545, 0.568, 0.593, 0.611, 0.633, 0.628, 0.603, 0.588, 0.578, 0.559, 0.551, 0.546, 0.543, 0.492, 0.494]
        t1 = [0.404, 0.417, 0.446, 0.508, 0.504, 0.445, 0.382, 0.339, 0.340, 0.340, 0.356, 0.379, 0.430, 0.470, 0.497, 0.499, 0.500, 0.543, 0.534, 0.523, 0.466, 0.409, 0.317]
        t2 = [0.325, 0.326, 0.344, 0.377, 0.418, 0.426, 0.387, 0.338, 0.316, 0.300, 0.264, 0.263, 0.326, 0.353, 0.399, 0.400, 0.417, 0.393, 0.421, 0.438, 0.438, 0.322, 0.297]
        flnAF = [0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300]
        rlnPGA_lnY = [1.000, 0.998, 0.986, 0.938, 0.887, 0.870, 0.876, 0.870, 0.850, 0.819, 0.743, 0.684, 0.562, 0.467, 0.364, 0.298, 0.234, 0.202, 0.184, 0.176, 0.154, 1.000, 0.684]    
                
        # ---------------------------------------------------------------------
        # Adjustment factor based on region
        # ---------------------------------------------------------------------
        
        if( (region == 2) or (region == 4) ):
            Dc20 = Dc20_JI
        elif(region == 3):
            Dc20 = Dc20_CH

        # if region is in Japan...
        Sj = region == 2 
        
        # if Z2.5 is unknown...
        if(Z25in == 'unknown'):
            if(not region == 2):  # if in California or other locations
                Z25 = np.exp( 7.089 - 1.144 * np.log( Vs30 ) )
                Z25A = np.exp( 7.089 - 1.144 * np.log( 1100 ) )
            elif(region == 2):  # if in Japan
                Z25 = np.exp( 5.359 - 1.102 * np.log( Vs30 ) )
                Z25A = np.exp( 5.359 - 1.102 * np.log( 1100 ) )
            
        else:
        # Assign Z2.5 from user input into Z25 and calc Z25A for Vs30=1100m/s
            if(not region == 2):  # if in California or other locations
                Z25 = Z25in;
                Z25A = np.exp( 7.089 - 1.144 * np.log( 1100 ) )
            elif(region == 2):  # if in Japan
                Z25 = Z25in
                Z25A = np.exp( 5.359 - 1.102 * np.log( 1100 ) )
        
        # ---------------------------------------------------------------------
        # Magnitude dependence
        # ---------------------------------------------------------------------
        if(M <= 4.5):
            fmag = c0[ip] + c1[ip] * M
        elif(M <= 5.5):
            fmag = c0[ip] + c1[ip] * M + c2[ip] * (M - 4.5)
        elif(M <= 6.5):
            fmag = c0[ip] + c1[ip] * M + c2[ip] * (M - 4.5) + c3[ip] * (M - 5.5)
        else:
            fmag = c0[ip] + c1[ip] * M + c2[ip] * (M - 4.5) + c3[ip] * (M - 5.5) + c4[ip] * (M-6.5);

        # ---------------------------------------------------------------------
        # Geometric attenuation term
        # ---------------------------------------------------------------------
        fdis = ( c5[ip] + c6[ip] * M) * np.log(np.sqrt(Rrup**2 + c7[ip]**2))
        
        # ---------------------------------------------------------------------
        # Style of faulting
        # ---------------------------------------------------------------------
        
        if(M <= 4.5):
            F_fltm = 0
        elif(M <= 5.5):
            F_fltm = M - 4.5
        else:
            F_fltm = 1
        
        fflt = ( ( c8[ip] * Frv ) + ( c9[ip] * Fnm ) )*F_fltm
        
        # ---------------------------------------------------------------------
        # Hanging-wall effects
        # ---------------------------------------------------------------------
    
        R1 = W * np.cos(np.pi/180*delta) # W - downdip width
        R2 = 62 * M - 350
        
        f1_Rx = h1[ip] + h2[ip]*(  Rx/R1 ) + h3[ip]*(Rx/R1)**2
        f2_Rx = h4[ip] + h5[ip]*( (Rx-R1)/(R2-R1) ) + h6[ip]*( (Rx-R1) / (R2-R1) )**2
        
        if(Fhw == 0):
            f_hngRx = 0
        elif( Rx < R1 and Fhw == 1 ):
            f_hngRx = f1_Rx
        elif( Rx >= R1 and Fhw == 1):
            f_hngRx = np.max(f2_Rx, 0)
        
        if(Rrup == 0):
            f_hngRup = 1
        else:
            f_hngRup = (Rrup-Rjb)/Rrup

        
        if(M <= 5.5):
            f_hngM = 0
        elif(M <= 6.5):
            f_hngM = (M-5.5)*(1+a2[ip]*(M-6.5))
        else:
            f_hngM = 1+a2[ip]*(M-6.5)
        
        if(Ztor <= 16.66):
            f_hngZ = 1-0.06*Ztor
        else:
            f_hngZ = 0
        
        f_hngdelta = (90-delta)/45
        
        fhng = c10[ip] * f_hngRx * f_hngRup * f_hngM * f_hngZ * f_hngdelta;
        
        # ---------------------------------------------------------------------
        # Site conditions
        # ---------------------------------------------------------------------
        if( Vs30 <= k1[ip] ):
            if(A1100 == 999):
               # compute A1100:  median estimated value of PGA on rock with Vs30 = 1100m/s 
               Site1100 = Site(Vs30 = 1100, fvs30 = 0, Z25 = Z25A, region = region)
               gmpe_curr = CB_2014_active( np.array([0]), Rup, Site1100 )
               A1100, _ = gmpe_curr.get_median_and_sigma(T = np.array([0]), Rup = Rup, Site = Site1100) 
               
               # keep matlab code for debugging
               #site1100 = site(sitevar.is_soil,1100,0,Z25A,sitevar.Z10,sitevar.Zbot,sitevar.region);
               #A1100 = cb_2014_active(0, Rup, site1100);
            f_siteG = c11[ip] * np.log( Vs30 / k1[ip] ) + k2[ip] * ( np.log( A1100 + c * ( Vs30 / k1[ip] )**n) - np.log(A1100 + c) )
            
        elif( Vs30 > k1[ip]):
            f_siteG = ( c11[ip] + k2[ip] * n ) * np.log( Vs30 / k1[ip] )



        if(Vs30 <= 200):
            f_siteJ = ( c12[ip] + k2[ip]*n ) * ( np.log(Vs30/k1[ip] ) - np.log( 200 / k1[ip] ) )*Sj
        else:
            f_siteJ = ( c13[ip] + k2[ip]*n ) * np.log(Vs30/k1[ip])*Sj

        
        fsite = f_siteG + f_siteJ
        
        # ---------------------------------------------------------------------
        # Basin Response Term - Sediment effects
        # ---------------------------------------------------------------------
        
        if(Z25 <= 1):
            fsed = (c14[ip]+c15[ip]*Sj) * (Z25 - 1);
        elif(Z25 <= 3):
            fsed = 0
        elif(Z25 > 3):
            fsed = c16[ip] * k3[ip] * np.exp(-0.75) * (1 - np.exp(-0.25 * (Z25 - 3)))
        
        
        # ---------------------------------------------------------------------
        # Hypocenteral Depth term
        # ---------------------------------------------------------------------
        
        if(Zhyp <= 7):
            f_hypH = 0
        elif(Zhyp <= 20):
            f_hypH = Zhyp - 7
        else:
            f_hypH = 13
        
        if(M <= 5.5):
            f_hypM = c17[ip]
        elif(M <= 6.5):
            f_hypM = c17[ip] + ( c18[ip]-c17[ip] )*(M-5.5);
        else:
            f_hypM = c18[ip]
        
        fhyp = f_hypH * f_hypM
        
        # ---------------------------------------------------------------------
        # Fault Dip term
        # ---------------------------------------------------------------------
        if(M <= 4.5):
            f_dip = c19[ip] * delta
        elif(M <= 5.5):
            f_dip = c19[ip] *(5.5-M)* delta
        else:
            f_dip = 0
        
        # ---------------------------------------------------------------------
        # Anelastic Attenuation Term
        # ---------------------------------------------------------------------
        if(Rrup > 80):
            f_atn = ( c20[ip] + Dc20[ip] )*(Rrup-80)
        else:
            f_atn = 0
        
        # ---------------------------------------------------------------------
        # Median value
        # ---------------------------------------------------------------------
        median = np.exp(fmag + fdis + fflt + fhng + fsite + fsed + fhyp + f_dip + f_atn)
                
        # ---------------------------------------------------------------------
        # Standard deviation computations
        # ---------------------------------------------------------------------
        
        ip_pga = -2
        if(M <= 4.5):
            tau_lny = t1[ip]
            tau_lnPGA = t1[ip_pga]   # ip = PGA, 22 in matlab code
            phi_lny = f1[ip]
            phi_lnPGA = f1[ip_pga]
        elif(M < 5.5):
            tau_lny = t2[ip]+( t1[ip]-t2[ip] )*(5.5-M)
            tau_lnPGA = t2[ip_pga]+(t1[ip_pga]-t2[ip_pga])*(5.5-M)    #ip = PGA
            phi_lny = f2[ip]+( f1[ip]-f2[ip] )*(5.5-M)
            phi_lnPGA = f2[ip_pga]+(f1[ip_pga]-f2[ip_pga])*(5.5-M)
        else:
            tau_lny = t2[ip]
            tau_lnPGA = t2[ip_pga]
            phi_lny = f2[ip]
            phi_lnPGA = f2[ip_pga]

        
        tau_lnyB = tau_lny
        tau_lnPGAB = tau_lnPGA
        phi_lnyB = np.sqrt(phi_lny**2-flnAF[ip]**2)
        phi_lnPGAB = np.sqrt(phi_lnPGA**2-flnAF[ip]**2);
        
        if(Vs30 < k1[ip] ):
            alpha = k2[ip] * A1100 * ((A1100 + c*(Vs30/ k1[ip] )**n)**(-1) - (A1100+c)**(-1))
        else:
            alpha = 0

        tau = np.sqrt(tau_lnyB**2 + alpha**2*tau_lnPGAB**2 + 2*alpha*rlnPGA_lnY[ip]*tau_lnyB*tau_lnPGAB)
        phi = np.sqrt(phi_lnyB**2 + flnAF[ip]**2 + alpha**2*phi_lnPGAB**2 + 2*alpha*rlnPGA_lnY[ip]*phi_lnyB*phi_lnPGAB)
        sigma = np.sqrt(tau**2+phi**2)       
    
        return median, sigma
    
    # Note: PGV not implemented, only periods and PGA   
    def get_median_and_sigma(self, T, Rup, Site):
        
        ## compute the medians and sigmas for the input periods 
        # preallocate median and sigma arrays
        nT = T.shape[0]  
        median = np.zeros( nT )
        sigma = np.zeros( nT )
        
        # periods for which the GMPE is defined
        periods_gmm = np.array([0.010, 0.020, 0.030, 0.050, 0.075, 0.10, 0.15, 
                         0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0, 
                         1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 0])
        
        for it in range(0, nT):
            # Teach = self.period1[it]
            Teach = T[it]
            # interpolate between periods, compute the median and sigma
            if(  ~np.any( (np.abs( periods_gmm - Teach ) < 0.0001) )  ):
                # interpolations periods
                T_low = np.array([np.max( periods_gmm[periods_gmm < Teach] )])
                T_hi = np.array([np.min( periods_gmm[periods_gmm > Teach] )])
                
                # this part is recursive
                sa_low, sigma_low = self.get_median_and_sigma(T_low, Rup, Site);
                sa_hi, sigma_hi = self.get_median_and_sigma(T_hi, Rup, Site);
                
                # interpolation -- linear in the log scale
                x = np.array([ np.log(T_low[0]), np.log(T_hi[0]) ])
                Y_sa = np.array([ np.log(sa_low[0]), np.log(sa_hi[0]) ])                
                Y_sigma = np.array([ sigma_low[0], sigma_hi[0] ])                
                f = scipy.interpolate.interp1d(x, Y_sa)
                median[it] = np.exp( f( np.log(Teach) )  )
                # conversion to RotD100
                if(self.is_scale_to_rotd100):
                    median[it] = median[it] * self.fn_rotd50_2_rotd100(Teach)

                f = scipy.interpolate.interp1d(x, Y_sigma)
                sigma[it] = f( np.log(Teach) )
            else:
                # identify which period -- use the correct index
                idx = np.argwhere(np.abs( periods_gmm - Teach ) < 0.0001)
                ip_curr = idx[0][0]
                # compute the IM 
                median[it], sigma[it] = self.CB_2014_nga_sub(Rup, Site, ip_curr)
                # conversion to RotD100
                if(self.is_scale_to_rotd100):
                    median[it] = median[it] * self.fn_rotd50_2_rotd100(Teach)

        
        return median, sigma


# -----------------------------------------------------------------------------
# Boore & Atkinson 2008 GMPE 
# -----------------------------------------------------------------------------  

class BA_2008_active(object):
    """Ground motion model (GMM) class for the Boore&Atkinson 2008 model"""
    
    ###########################################################################
    # Based on the matlab code by:
    #  Yoshifumi Yamamoto, 11/10/08, yama4423@stanford.edu
    #  Updated by Emily Mongold, 11/27/20
    #  URL: https://github.com/bakerjw/GMMs/blob/master/gmms/ba_2008_active.m    
    #  Source Model: 
    #  Boore D. M. and Atkinson, G. M.(2008). "Ground-Motion Prediction
    #  Equations for the Average Horizontal Component of PGA, PGV, and 5%-Damped
    #  PSA at Spectral Periods between 0.01s and 10.0s." Earthquake Spectra,
    #  24(1), 99-138. 
    #  
    #  This script has been modified to correct an error based on the website
    #  <http://www.daveboore.com/pubs_online.php>.
    #  Table 6 should read "Distance-scaling coefficients (Mref=4.5 and Rref=1.0
    #  km for all periods)"
    ###########################################################################
    
    # INPUTS:
    #   T : array of periods (sec); use -1 for PGV computation; use 1000 for 
    #       computation of Sa at periods used in the GMM 
    #   Rup  : rupture object with following attributes:        
    #        Rjb         = Joyner-Boore distance (km)
    #        lamb        = rake angle, used to set FaultType:
    #                        = 1 for unspecified fault 
    #                        = 2 for strike-slip fault
    #                        = 3 for normal fault
    #                        = 4 for reverse fault    
    #   Site : site object with following attributes:
    #        Vs30        = shear wave velocity averaged over top 30 m in m/s   
    
    # Class attributes: 
    #    median      = Median spectral acceleration prediction
    #    sigma       = logarithmic standard deviation of spectral acceleration
    #                  prediction
    #    period1     = periods for which the median and sigma values are
    #                  provided. If T = 1000, then period1 = the full set of
    #                  available periods. Else period1 = T
    
    # Class methods:
    #   TODO :: document   
    
    
    def __init__(self, T, Rup, Site):
        self.FaultType = self.get_FaultType(Rup) # compute fault type based on 
            # the rake angle
        self.period1 = self.get_periods(T) # define the IMT, i.e., get periods 
            # at which to compute the Sa values 
        self.median, self.sigma = self.get_median_and_sigma(self.period1, Rup, Site)
        
        
    def get_FaultType(self, Rup):
        """Setting the fault type based on the rake angle"""
        lam = Rup.lam
        if( ( lam <= 30 and lam >= -30 ) or ( lam <= 180 and lam >= 150 ) or ( lam <= -150 and lam >= -180 ) ):
            return 2 # Strike-slip fault
        elif( lam <= -60 and lam >= -120 ):
            return 3 # Normal fault
        elif( lam <= 120 and lam >= 60 ):
            return 4 # Reverse fault
        else:
            return 1 # Other / unspecified

    def get_periods(self, T):
        """Define computation periods"""
        if( np.any(T == 1000) ):
            # use original periods from the GMM
            return np.array([0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15,	
                             0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1, 1.5, 2, 	
                             3, 4, 5, 7.5, 10])
        else: return T
            
    def BA_2008_nga_sub(self, M, ip, Rjb, U, S, N, R, Vs30):
        """Method computes median and sigma using the Boore and Atkinson 2008 GMM"""

        # Inputs
        #   ip : index corresponding to the period being computed

        # % Coefficients from the Boore and Atkinson 2008 paper
        e01 = np.array([5.00121, -0.53804, -0.52883	, -0.52192, -0.45285, -0.28476,
                        0.00767, 0.20109, 0.46128, 0.5718, 0.51884, 0.43825, 0.3922, 
                        0.18957, -0.21338, -0.46896, -0.86271, -1.22652, -1.82979,
                        -2.24656, -1.28408, -1.43145, -2.15446])
        e02 = np.array([5.04727, -0.5035, -0.49429, -0.48508, -0.41831, -0.25022, 
                        0.04912, 0.23102, 0.48661, 0.59253, 0.53496, 0.44516, 0.40602, 
                        0.19878, -0.19496, -0.43443, -0.79593, -1.15514, -1.7469, 
                        -2.15906, -1.2127, -1.31632, -2.16137])
        e03 = np.array([4.63188, -0.75472, -0.74551, -0.73906, -0.66722, -0.48462,
                        -0.20578, 0.03058, 0.30185, 0.4086, 0.3388, 0.25356, 0.21398,
                        0.00967, -0.49176, -0.78465, -1.20902, -1.57697, -2.22584, 
                        -2.58228, -1.50904, -1.81022, -2.53323])
        e04 = np.array([5.0821, -0.5097, -0.49966, -0.48895, -0.42229, -0.26092,
                        0.02706, 0.22193, 0.49328, 0.61472, 0.57747, 0.5199, 
                        0.4608, 0.26337, -0.10813, -0.3933, -0.88085, -1.27669, 
                        -1.91814, -2.38168, -1.41093, -1.59217, -2.14635])
        e05 = np.array([0.18322, 0.28805, 0.28897, 0.25144, 0.17976, 0.06369, 0.0117,
                        0.04697, 0.1799, 0.52729, 0.6088, 0.64472, 0.7861, 0.76837, 
                        0.75179, 0.6788, 0.70689, 0.77989, 0.77966, 1.24961, 0.14271,
                        0.52407, 0.40387])
        e06 = np.array([-0.12736, -0.10164, -0.10019, -0.11006, -0.12858, -0.15752, 
                        -0.17051, -0.15948, -0.14539, -0.12964, -0.13843, -0.15694,
                        -0.07843, -0.09054, -0.14053, -0.18257, -0.2595, -0.29657,
                        -0.45384, -0.35874, -0.39006, -0.37578, -0.48492])
        e07 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00102, 0.08607, 0.10601, 0.02262,
                        0, 0.10302, 0.05393, 0.19082, 0.29888, 0.67466, 0.79508, 0, 0, 0])
        mh = np.array([8.5, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 
                       6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 8.5, 8.5, 8.5])
        c01 = np.array([-0.8737, -0.6605, -0.6622, -0.666, -0.6901, -0.717, -0.7205, 
                        -0.7081, -0.6961, -0.583, -0.5726, -0.5543, -0.6443, -0.6914, 
                        -0.7408, -0.8183, -0.8303, -0.8285, -0.7844, -0.6854, -0.5096, 
                        -0.3724, -0.09824])
        c02 = np.array([0.1006, 0.1197, 0.12, 0.1228, 0.1283, 0.1317, 0.1237, 0.1117, 
                        0.09884, 0.04273, 0.02977, 0.01955, 0.04394, 0.0608, 0.07518, 
                        0.1027, 0.09793, 0.09432, 0.07282, 0.03758, -0.02391, -0.06568,
                        -0.138])
        c03 = np.array([-0.00334, -0.01151, -0.01151, -0.01151, -0.01151, -0.01151, 
                        -0.01151, -0.01151, -0.01113, -0.00952, -0.00837, -0.0075, 
                        -0.00626, -0.0054, -0.00409, -0.00334, -0.00255, -0.00217, 
                        -0.00191, -0.00191, -0.00191, -0.00191, -0.00191])
        h = np.array([2.54, 1.35, 1.35, 1.35, 1.35, 1.35, 1.55, 1.68, 1.86, 1.98, 
                      2.07, 2.14, 2.24, 2.32, 2.46, 2.54, 2.66, 2.73, 2.83, 2.89, 
                      2.93, 3, 3.04])
        blin = np.array([-0.6, -0.36, -0.36, -0.34, -0.33, -0.29, -0.23, -0.25, -0.28, 
                         -0.31, -0.39, -0.44, -0.5, -0.6, -0.69, -0.7, -0.72, -0.73, 
                         -0.74, -0.75, -0.75, -0.692, -0.65])
        b1 = np.array([-0.5, -0.64, -0.64, -0.63, -0.62, -0.64, -0.64, -0.6, -0.53, 
                       -0.52, -0.52, -0.52, -0.51, -0.5, -0.47, -0.44, -0.4, -0.38,
                       -0.34, -0.31, -0.291, -0.247, -0.215])
        b2 = np.array([-0.06, -0.14, -0.14, -0.12, -0.11, -0.11, -0.11, -0.13, -0.18, 
                       -0.19, -0.16, -0.14, -0.1, -0.06, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        sig1 = np.array([0.5, 0.502, 0.502, 0.502, 0.507, 0.516, 0.513, 0.52, 0.518, 
                         0.523, 0.527, 0.546, 0.541, 0.555, 0.571, 0.573, 0.566, 0.58, 
                         0.566, 0.583, 0.601, 0.626, 0.645])
        sig2u = np.array([0.286, 0.265, 0.267, 0.267, 0.276, 0.286, 0.322, 0.313, 
                          0.288, 0.283, 0.267, 0.272, 0.267, 0.265, 0.311, 0.318,
                          0.382, 0.398, 0.41, 0.394, 0.414, 0.465, 0.355])
        sigtu = np.array([0.576, 0.566, 0.569, 0.569, 0.578, 0.589, 0.606, 0.608, 
                          0.592, 0.596, 0.592, 0.608, 0.603, 0.615, 0.649, 0.654, 
                          0.684, 0.702, 0.7, 0.702, 0.73, 0.781, 0.735])
        sig2m = np.array([0.256, 0.26, 0.262, 0.262, 0.274, 0.286, 0.32, 0.318, 0.29,
                          0.288, 0.267, 0.269, 0.267, 0.265, 0.299, 0.302, 0.373, 0.389,
                          0.401, 0.385, 0.437, 0.477, 0.477])
        sigtm = np.array([0.56, 0.564, 0.566, 0.566, 0.576, 0.589, 0.606, 0.608, 0.594, 
                          0.596, 0.592, 0.608, 0.603, 0.615, 0.645, 0.647, 0.679, 0.7, 
                          0.695, 0.698, 0.744, 0.787, 0.801])
        a1 = 0.03
        pga_low = 0.06
        a2 = 0.09
        v1 = 180
        v2 = 300
        vref = 760
        mref=4.5
        rref=1
        
        # Note:
        # - the coefficients have shape 23, need to be careful to include period
        #   indices associated with "periods" -1 and 0 properly
        # - MATLAB is indexed from 1, so I changed indices in some of the lines 
        #   below, see comments
        
        ## Magnitude Scaling
        if(M <= mh[ip]):
            Fm = e01[ip] * U + e02[ip] * S + e03[ip] * N + e04[ip] * R + e05[ip] * (M - mh[ip]) + e06[ip] * (M - mh[ip])**2
        else:
            Fm = e01[ip] * U + e02[ip] * S + e03[ip] * N + e04[ip] * R + e07[ip] * (M - mh[ip])
        
        ## Distance Scaling
        r = np.sqrt(Rjb**2 + h[ip]**2)
        Fd = ( c01[ip] + c02[ip] * (M - mref) ) * np.log(r / rref) + c03[ip] * (r - rref)
        
        if( Vs30 !=vref or ip != 1): # original code ip ~= 2, indexing from 1 in Matlab, so I changed to 1
            pga4nl, _ = self.BA_2008_nga_sub(M, 1, Rjb, U, S, N, R, vref) # original code : BA_2008_nga_sub(M, 2, Rjb, U, S, N, R, vref); I changed period index to 1
        else:
            # % Compute median and sigma
            lny = Fm + Fd
            median = np.exp(lny) 
            sigma = (U == 1) * sigtu[1] + (U != 1) * sigtm[1] # original code: (U==1) * sigtu(2) + (U != 1) * sigtm(2); I changed period index to 1
            return median, sigma
        
        ## Site Amplification
        # Linear term
        Flin = blin[ip] * np.log(Vs30 / vref)
        
        # Nonlinear term
        # Computation of nonlinear factor
        if(Vs30 <= v1):
            bnl = b1[ip]
        elif(Vs30 <= v2):
            bnl = b2[ip] + ( b1[ip] - b2[ip] ) * np.log(Vs30 / v2) / np.log(v1 / v2)
        elif(Vs30 <= vref):
            bnl = b2[ip] * np.log(Vs30 / vref) / np.log(v2 / vref)
        else:
            bnl = 0.0

        deltax = np.log(a2/a1)
        deltay = bnl * np.log(a2/pga_low)
        c = (3 * deltay - bnl * deltax) / (deltax**2)
        d = - (2 * deltay - bnl * deltax) / (deltax**3)
        
        if(pga4nl <= a1):
            Fnl = bnl * np.log(pga_low / 0.1)
        elif(pga4nl <= a2):
            Fnl = bnl * np.log(pga_low / 0.1) + c * ( np.log(pga4nl / a1) )**2 + d * ( np.log(pga4nl / a1) )**3
        else:
            Fnl = bnl * np.log(pga4nl / 0.1)
        
        Fs = Flin + Fnl
        
        # Compute median and sigma
        lny = Fm + Fd + Fs
        median = np.exp(lny) 
        sigma = ( U==1 ) * sigtu[ip] + (U !=1 ) * sigtm[ip]
        
        return median, sigma
                   
    def get_median_and_sigma(self, T, Rup, Site):
        """Compute median and sigma at specified periods using the GMM relation"""        
        
        # define U, S, N, R based on the self.FaultType
        U = (self.FaultType == 1)
        S = (self.FaultType == 2)
        N = (self.FaultType == 3)
        R = (self.FaultType == 4)
        
        # find periods for which the interpolation is necessary
        periods_gmm = np.array([-1, 0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15,	
                         0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1, 1.5, 2, 	
                         3, 4, 5, 7.5, 10])    
        
        # periods_notInterp = self.period1[ np.isin( self.period1, periods_gmm ) ]
        # periods_interp = self.period1[ ~np.isin( self.period1, periods_gmm ) ]
        
        periods_notInterp = T[ np.isin( T, periods_gmm ) ]
        periods_interp = T[ ~np.isin( T, periods_gmm ) ]
        
        ## compute the medians and sigmas for the input periods 
        # preallocate median and sigma arrays
        # nT = self.period1.shape[0]
        # print(T)
        # print('Here')
        nT = T.shape[0]  
        median = np.zeros( nT )
        sigma = np.zeros( nT )
        
        for it in range(0, nT):
            # Teach = self.period1[it]
            Teach = T[it]
            # interpolate between periods, compute the median and sigma
            if(  ~np.any( (np.abs( periods_gmm - Teach ) < 0.0001) )  ):
                # interpolations periods
                T_low = np.array([np.max( periods_gmm[periods_gmm < Teach] )])
                T_hi = np.array([np.min( periods_gmm[periods_gmm > Teach] )])
                # this part is recursive
                sa_low, sigma_low = self.get_median_and_sigma(T_low, Rup, Site);
                sa_hi, sigma_hi = self.get_median_and_sigma(T_hi, Rup, Site);
                # interpolation -- linear in the log scale
                x = np.array([ np.log(T_low[0]), np.log(T_hi[0]) ])
                Y_sa = np.array([ np.log(sa_low[0]), np.log(sa_hi[0]) ])                
                Y_sigma = np.array([ sigma_low[0], sigma_hi[0] ])                
                f = scipy.interpolate.interp1d(x, Y_sa)
                median[it] = np.exp( f( np.log(Teach) )  )
                f = scipy.interpolate.interp1d(x, Y_sigma)
                sigma[it] = f( np.log(Teach) )
            else:
                # identify which period -- use the correct index
                idx = np.argwhere(np.abs( periods_gmm - Teach ) < 0.0001)
                i = idx[0][0]
                # compute the IM 
                median[it], sigma[it] = self.BA_2008_nga_sub(Rup.M, i, Rup.Rjb, U, S, N, R, Site.Vs30)

        return median, sigma

# -----------------------------------------------------------------------------    
# Kotha et al. 2020 GMPE 
# -----------------------------------------------------------------------------    
# pseudocode:
#   - get the coefficients from the table into a dataframe X
#   - get the constants for the GMPE X
#   - compute geometric spreading X
#   - compute anelastic attenuation X
#   - compute magnitude scaling X
#   - compute site-response component (have a stump to compute fixed effects first) X
#   - merge all effects together to compute the mean ground motion, convert to units of g (use a bool here) X
#   - have a function to compute the standard deviations at all periods X
#
#   - make a final function that gets mean and dispersions including interpolation if the regired periods are not amongst the default periods
#       - for periods that are not part of the GMPE do the interpolation between the closest periods   
 
# Inputs:
    # Mw, R_JB, vs30 for the site-response, T (period where to do the computation)
    # 

import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO

class KothaEtAl_2020(object):
    """Ground motion model (GMM) class for the Kotha et al. (2020) model;
    Based on the following reference:
        Kotha, Sreeram Reddy, Graeme Weatherill, Dino Bindi, and Fabrice Cotton. 
        “A Regionally-Adaptable Ground-Motion Model for Shallow Crustal 
        Earthquakes in Europe.” Bulletin of Earthquake Engineering 18, no. 9 
        (July 1, 2020): 4091–4125. https://doi.org/10.1007/s10518-020-00869-1.
    """
    ###########################################################################
    # INPUTS:
    #   T : np.array of periods (sec); use -1 for PGV computation; use None for 
    #       computation of Sa at periods used in the GMM
    #       NOTE: currently the script is not made to support PGV if converting
    #             to units of g
    #   Rup  : rupture object with following attributes:        
    #        M           = Magnitude        
    #        Rjb         = Joyner-Boore distance (km)
    #        hypo_depth  = hypocentral-depth (km); used to determine the 
    #                      h-parameter constant
    #   Site : site object with following attributes:
    #        Vs30        = shear wave velocity averaged over top 30 m in m/s   
    
    # Class attributes: 
    #    median      = Median spectral acceleration prediction (g)
    #    sigma       = logarithmic standard deviation of spectral acceleration
    #                  prediction
    #    period1     = periods for which the median and sigma values are
    #                  provided. If T = 1000, then period1 = the full set of
    #                  available periods. Else period1 = T
    #   regression_coefficients = pandas dataframe with fixed effects regression
    #                   coefficients from Kotha et al. 2020
    #   constants   = dict with constants used in Kotha et al. 2020
    
    # Class methods:
    #   TODO :: document   

    def __init__(self, T, Rup, Site):
        
        self.regression_coefficients = self.get_regression_coefficients()
        self.period1 = self.get_periods(T) # define the IMT, i.e., get periods 
            # at which to compute the Sa values 
        self.get_constants()
        self.median, self.sigma = self.get_median_and_sigma(self.period1, Rup, Site)


    def get_regression_coefficients(self):
        """Fixed-effects regression coefficients from Kotha et al. (2020); 
          note: coefficients taken from the openQuake implementation of the GMPE"""
          
        coeffdata = StringIO(
            """imt                   e1                 b1                  b2                  b3                  c1                   c2                   c3              tau_c3              phis2s         tau_event_0             tau_l2l               phi_0              g0_vs30              g1_vs30               g2_vs30         phi_s2s_vs30              g0_slope              g1_slope               g2_slope       phi_s2s_slope
            -1     1.11912161648479   2.55771078860152   0.353267224391297   0.879839839344054   -1.41931258132547   0.2706807258213520   -0.304426142175370   0.178233997535235   0.560627759977840   0.422935885699239   0.258560350227890   0.446525247049620   -0.232891265610189   -0.492356618589364    0.0247963168536102    0.366726744441574   -0.0550827970556740   -0.1469535974165200   -0.00893120461876375   0.434256033254051
            0       3.93782347219377   2.06573167101440   0.304988012209292   0.444773874960317   -1.49787542346412   0.2812414746313380   -0.609876182476899   0.253818777234181   0.606771946180224   0.441761487685862   0.355279206886721   0.467151252053241   -0.222196028066344   -0.558848724731566   -0.1330148640403130    0.389712940326169   -0.0267105106085816   -0.1098813702713090   -0.01742373265620930   0.506725958082485
            0.010   3.94038760011295   2.06441772899445   0.305294151898347   0.444352974827805   -1.50006146971318   0.2816120431678390   -0.608869451197394   0.253797652143759   0.607030265833062   0.441635449735044   0.356047209347534   0.467206938011971   -0.221989239810027   -0.558181442039516   -0.1330144520414310    0.391254585814764   -0.0266572723455345   -0.1097145490975510   -0.01741863169765470   0.506706245975056
            0.025   3.97499686979384   2.04519749120013   0.308841647142436   0.439374383710060   -1.54376149680542   0.2830031280602480   -0.573207556417252   0.252734624432000   0.610030865927204   0.437676505154608   0.368398604288111   0.468698397037258   -0.218745638720123   -0.546810177342948   -0.1315295091425130    0.395303566681041   -0.0254040142855204   -0.1072422064249640   -0.01765069385301560   0.506705856554187
            0.040   4.08702279605872   1.99149766561616   0.319673428428720   0.418531185104657   -1.63671359040283   0.2984823762486280   -0.535139204130152   0.244894143623498   0.626413180170373   0.429637401735540   0.412921240156940   0.473730661220076   -0.206923687805771   -0.525141264234585   -0.1368798835282360    0.415116874033842   -0.0222919270649348   -0.1024278275345350   -0.01847074311083690   0.515812197849121
            0.050   4.18397570399970   1.96912968528742   0.328982074841989   0.389853296189063   -1.66358950776148   0.3121928913488560   -0.555191107011420   0.260330694464557   0.638967955474841   0.433639923327438   0.444324049044753   0.479898166019243   -0.205629239209508   -0.514739138349666   -0.1368385040078350    0.422549340781658   -0.0209153599570857   -0.0989203779863760   -0.01851248498790100   0.526875631632610
            0.070   4.38176649786342   1.92450788134500   0.321182873495225   0.379581373255289   -1.64352914575492   0.3138101953091510   -0.641089475725666   0.286976037026550   0.661064599433347   0.444338223383705   0.470938801038256   0.487060899687138   -0.209348356311787   -0.506896476331228   -0.1456117952510990    0.443318525820235   -0.0188838682625869   -0.0951010574545904   -0.01880576764531640   0.553542604942032
            0.100   4.60722959404894   1.90125096928647   0.298805051330753   0.393002352641809   -1.54339428982169   0.2849395739776680   -0.744270750619733   0.321927482439715   0.663309669119995   0.458382304191096   0.478737965504940   0.496152397155402   -0.193509476649993   -0.521463491048192   -0.1824674441457950    0.437214022468042   -0.0165212272103937   -0.0871969707343552   -0.01674749313351450   0.537128822815826
            0.150   4.78583314367062   1.92620172077838   0.249893333649662   0.435396192976506   -1.38136438628699   0.2254113422224680   -0.815688997995934   0.322145126407981   0.655406109737959   0.459702777214781   0.414046169030935   0.497805936702476   -0.215418461095753   -0.579757224642522   -0.2016525247813580    0.457311836251173   -0.0153013615272199   -0.0898557092287409   -0.01820533201066010   0.548306674706135
            0.200   4.81847463780069   1.97006598187863   0.218722883323200   0.469713318293785   -1.30697558633587   0.1826533194804230   -0.773372802995208   0.301795870071949   0.643585009231006   0.464006126996261   0.321975745683642   0.494075956910651   -0.232802520913539   -0.646162914187111   -0.2102452066359760    0.449595599604904   -0.0185432743074803   -0.1091715402153590   -0.02203326475372750   0.542391858770537
            0.250   4.75134747347049   2.01097445156370   0.195062831156806   0.532210412551561   -1.26259484078950   0.1551575007473110   -0.722012122448262   0.274998157533509   0.623240061418664   0.457687642192569   0.293329526713994   0.488950837091220   -0.238646255489286   -0.649028548718928   -0.1965317433344580    0.449701754122993   -0.0268512786854638   -0.1177223461809770   -0.01990310375762760   0.514759188358396
            0.300   4.65252285968525   2.09278551802016   0.194929941231544   0.557034893811231   -1.24071282395616   0.1370008066985060   -0.660466290850886   0.260774631679394   0.609748615552919   0.457514283978959   0.266836791529257   0.482157450259502   -0.246093988657936   -0.645741652187205   -0.1720972685448300    0.429850112026890   -0.0356644839782008   -0.1265719157414280   -0.01728437065375890   0.490014753971745
            0.350   4.53350897671045   2.14179725762371   0.189511462582876   0.609892595327716   -1.21514531872583   0.1247122464559250   -0.618593385936676   0.254261888951322   0.609506191611413   0.450960093750492   0.231614185359720   0.480254056040507   -0.254026518879524   -0.648402249765170   -0.1446513637358710    0.397602725132059   -0.0423519589829896   -0.1401638874897640   -0.01672203482354180   0.483807852643816
            0.400   4.44193244811952   2.22862498827440   0.200305171692326   0.614767001033243   -1.18897228839914   0.1156387616270450   -0.591574546068960   0.243643375298288   0.615477199296824   0.441122908694716   0.240825814626397   0.475193646646757   -0.263328502132230   -0.653476851717702   -0.1186474533289450    0.439991306965322   -0.0452239204802930   -0.1514100096093150   -0.01778303668068960   0.500388492016146
            0.450   4.33697728548038   2.29103572171716   0.209573442606565   0.634252522127606   -1.18013993982454   0.1100834686500940   -0.555234498707119   0.245883260391068   0.619384591074073   0.436294164198843   0.249245758570064   0.469672671050266   -0.264631841951527   -0.638852650094042   -0.0836039291412020    0.424224393510765   -0.0543649832422398   -0.1588148016645050   -0.01500762961938830   0.492980996451707
            0.500   4.23507897753587   2.35399193121686   0.218088423514177   0.658541873692286   -1.17726165949601   0.1026978146186720   -0.519413341065942   0.238559829231160   0.624993564560933   0.428500398327627   0.243778652813106   0.463165027132890   -0.269124654561252   -0.626175743644433   -0.0537720540773490    0.423230860170143   -0.0610661425543540   -0.1647334612739770   -0.01304441434577370   0.495138633047097
            0.600   4.02306439391925   2.42753387249929   0.218787915039312   0.754615594874153   -1.16678688970027   0.0940582863096094   -0.454043559543982   0.216855298090451   0.635090711921061   0.426296731581312   0.246117069779268   0.451206692163190   -0.269626118151597   -0.582682427052082    0.0203225530214242    0.475220856944347   -0.0680919086636438   -0.1730542985615550   -0.00960057312582767   0.510149252547482
            0.700   3.83201580121827   2.51268432884949   0.225024841305000   0.765438564882833   -1.16236278470164   0.0865917976706938   -0.397781532595396   0.215716276719833   0.633635835573626   0.425379430268476   0.246750734502549   0.446704739768374   -0.272441022824943   -0.558163103244591    0.0652728074463838    0.446489639181972   -0.0742129950461250   -0.1739452472381870   -0.00549504377749866   0.502939558871623
            0.750   3.74614211993052   2.55840246083607   0.231604957273506   0.793480645885641   -1.15333203234665   0.0824927940948198   -0.376630503031279   0.209593410875067   0.637877956868669   0.428563811859323   0.245166749142241   0.444311331912854   -0.268471953245116   -0.546146873703377    0.0840210504832594    0.451727019248850   -0.0742883211225450   -0.1757280229442730   -0.00571924409424620   0.513908669690317
            0.800   3.65168809980226   2.59467404437385   0.237334498546207   0.828241777740572   -1.14645090256437   0.0837439530041729   -0.363246464853852   0.192106714053294   0.638753820813416   0.433880652259324   0.240072953116796   0.439300059540554   -0.268043587730749   -0.528310722806634    0.1053131905955920    0.476641301777151   -0.0733362133528447   -0.1769632805164950   -0.00623439334393725   0.516534123477592
            0.900   3.51228638217709   2.68810225072750   0.251716558693382   0.845561170244942   -1.13599614124436   0.0834018259445213   -0.333908265367165   0.177456610405390   0.640328521929993   0.438913972406961   0.247662698012904   0.433043490235851   -0.270747888599204   -0.498749188701101    0.1514549282913290    0.492678009609922   -0.0705690120386147   -0.1842212802961380   -0.00948523310240806   0.508758129697782
            1.000   3.36982044793917   2.74249776483975   0.256784133033388   0.896648260528882   -1.12443352348542   0.0854384622609198   -0.317465939881623   0.171997778367260   0.638429444564638   0.444086895369946   0.238111905941701   0.426703815544157   -0.268682366673877   -0.472355589159814    0.1912725393732170    0.486349823748500   -0.0730202296385978   -0.1861995093276410   -0.00833302021378029   0.499129039268700
            1.200   3.10224418952824   2.82683484364226   0.262683442221073   0.982921357727718   -1.12116148624672   0.0973231293288241   -0.275616235541070   0.160445653296358   0.640086303643832   0.446121165446841   0.226825215617356   0.416539877732589   -0.263517582328224   -0.465411813875967    0.2014565230611100    0.460802894674431   -0.0761329216007339   -0.1923688484322410   -0.00790676960410267   0.494333782654409
            1.400   2.84933745949861   2.89911332547612   0.272065572034688   1.040000637056720   -1.12848926976065   0.1002887249133400   -0.234977212668109   0.150949141990859   0.649359928046388   0.457011583377380   0.231922092201736   0.409641113489270   -0.253077954003716   -0.450716220871832    0.1900019177957120    0.520330220947425   -0.0777847149574368   -0.1977821544457880   -0.00694977055552574   0.521824672837616
            1.600   2.63503429015231   2.98365736561984   0.289670716036571   1.073002118658300   -1.14064711059980   0.1100788214866130   -0.198050139347725   0.148738498099927   0.650540540696659   0.462781403376806   0.223897549097876   0.404985162254916   -0.246009048662975   -0.427498542497053    0.2013164560891230    0.498576704112864   -0.0808481108779988   -0.1956817304755080   -0.00420478503206788   0.520676267977361
            1.800   2.43032254290751   3.06358840071518   0.316828766785138   1.109809835991900   -1.15419967841818   0.1131278831612640   -0.167123738873435   0.156141593013035   0.656949311785981   0.468432106332010   0.205207971335941   0.399057812399511   -0.259365145858505   -0.436165813138372    0.2103523943478280    0.494419960120798   -0.0866501788741884   -0.1968633287340960    0.00084917955133917   0.521315249011902
            2.000   2.24716354703519   3.11067747935049   0.326774527695550   1.132479221218060   -1.16620971948721   0.1162990300931710   -0.140731664063789   0.155054491423268   0.647763389017009   0.476577198889343   0.196850466599025   0.396502973620567   -0.255846430844076   -0.425096032934296    0.2073318834508050    0.484354097558551   -0.0881098607385541   -0.1980665849538590    0.00178776027496752   0.509385313956226
            2.500   1.83108464781202   3.23289020747997   0.374214285707986   1.226390493979360   -1.17531326311999   0.1395412164588280   -0.120745041347963   0.176744551716694   0.629481669044830   0.479859874942997   0.190867925368865   0.393288023064441   -0.257425360830402   -0.394240493031487    0.2135940556445740    0.460612029226665   -0.0842255772225518   -0.1909303606402940    0.00128428761198652   0.505686965707424
            3.000   1.58259215964414   3.44640772476285   0.454951810817816   1.313954219909490   -1.15664484431459   0.1494902905791280   -0.149050671035371   0.174876785480317   0.616446588503561   0.488309107285476   0.220914253465451   0.390859427279163   -0.251876760182310   -0.364653376508969    0.2122004191615380    0.407986805228384   -0.0784780440908414   -0.1844510105227600   -0.00047381737627311   0.485603444879608
            3.500   1.32153652077149   3.56445182133655   0.518610571029448   1.394984393379380   -1.16368470057735   0.1543445278711660   -0.142873831246493   0.193619214137258   0.600202108018105   0.479187019962682   0.237281350236338   0.388102875218375   -0.242628051593659   -0.322323015714785    0.2138248326399060    0.396737062193148   -0.0787732613082041   -0.1718918693565610    0.00223831455352896   0.479608514060425
            4.000   1.10607064193676   3.64336885536264   0.555331865800278   1.418144933323620   -1.17757508691221   0.1730832048262120   -0.142053716741244   0.193571789393738   0.593046407283143   0.482524831704549   0.233827536969510   0.386956009422453   -0.239634395956042   -0.294311486158724    0.2268951652965890    0.396113359026388   -0.0764209712209348   -0.1648847320168560    0.00295327439998048   0.475041314185757
            4.500   1.05987610378773   3.82152567982841   0.666476453600402   1.430548279466630   -1.17323633891422   0.1936210609543320   -0.156076448842833   0.152553585766189   0.581331910387036   0.456765160173852   0.196697785051230   0.372827866334900   -0.246998133746262   -0.241579092689847    0.2474533712720740    0.397717123177902   -0.0668746312766319   -0.1735273164380950   -0.00530669973001712   0.473200567096548
            5.000   0.82373381739570   3.84747968562771   0.684665144355361   1.496536314224210   -1.20969230916539   0.2213041109459350   -0.126052481240424   0.137919529808920   0.558954997903623   0.464229101930025   0.195572800413952   0.377458812369736   -0.234334071379258   -0.208962718979667    0.2332755435126690    0.338344656676906   -0.0617201190392144   -0.1636990315777190   -0.00649134386415973   0.450949884766277
            6.000   0.50685354955206   3.80040950285788   0.700805222359295   1.625591116375650   -1.22440411739130   0.2292764533844400   -0.113766839623945   0.141669390606605   0.538973145096788   0.439059204276786   0.190680023411634   0.384862538848542   -0.205342867591920   -0.166350345553781    0.2189842473229210    0.338688052762081   -0.0568786587375636   -0.1519590377762100   -0.00580039515645921   0.439827391985479
            7.000   0.19675504234642   3.78431011962409   0.716569352050671   1.696310364814470   -1.28517895409644   0.2596896867469380   -0.070585399916418   0.146488759166368   0.523331606096182   0.434396029381517   0.208231539543981   0.385850838707000   -0.204046508080049   -0.155173106999605    0.2164856914333770    0.339211265835413   -0.0541313319257671   -0.1393109833551150   -0.00443019667996698   0.432359150492787
            8.000  -0.08979569600589   3.74815514351616   0.726493405776986   1.695347146909250   -1.32882937608962   0.2849197966362740   -0.051296439369391   0.150981191615944   0.508537123776905   0.429104860654150   0.216201318346277   0.387633769846605   -0.193908824182191   -0.148759113452472    0.2094261301289650    0.337650861518699   -0.0507933301386227   -0.1365792860813190   -0.00532310915144333   0.411101516213337""")
            
        # load the coefficients into pandas dataframe
        df_reg_coeffs = pd.read_csv(coeffdata, sep = '\s+', index_col = 0)
        return df_reg_coeffs 
    
    def get_constants(self):
        """Constants used in the functional form of the GMPE"""
        
        CONSTANTS = {"Mref": 4.5, "Rref": 30., "Mh": 5.7, # openQuake had Mh = 5.7 while the paper uses Mh = 6.2; which one should be used and why?
              "h_D10": 4.0, "h_10D20": 8.0, "h_D20": 12.0}
        # CONSTANTS = {"Mref": 4.5, "Rref": 30., "Mh": 6.2,
        #      "h_D10": 4.0, "h_10D20": 8.0, "h_D20": 12.0}
        
        self.constants = CONSTANTS
    
    def get_h_D(self, Rup):
        """Return the h-parameter term h_D based on hypocentral depth"""
        h_depth = Rup.hypo_depth
        if(h_depth < 10):
            return self.constants['h_D10']
        elif(h_depth >= 20):
            return self.constants['h_D20']
        else:
            return self.constants['h_10D20']
        
    def get_periods(self, T):
        """Define periods at which to compute the GMPE"""
        # if( np.any(T == 1000) ):
        if T is None:
            # use original periods from the GMM
            return np.array([0, 0.010, 0.025, 0.040 ,0.050, 0.070, 0.100, 0.150,
                            0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500, 
                            0.600, 0.700, 0.750, 0.800, 0.900, 1.000, 1.200,
                            1.400, 1.600, 1.800, 2.000, 2.500, 3.000, 3.500,
                            4.000, 4.500, 5.000, 6.000, 7.000, 8.000])
        else: return T
            
    def get_geometric_term(self, T, Rup):
        """Compute the geometric spreading term, Eq (3) with added e1 term"""
        d_M = Rup.M - self.constants['Mref']
        h_D = self.get_h_D(Rup)
        d_Rjb = np.sqrt(Rup.Rjb**2 + h_D**2)
        d_Rref = np.sqrt(self.constants['Rref']**2 + h_D**2)
        
        # get e1, c1, and c2 for the computation periods
        e1 = self.regression_coefficients['e1'].loc[T]
        c1 = self.regression_coefficients['c1'].loc[T]
        c2 = self.regression_coefficients['c2'].loc[T]
        
        f_r_g = e1 + (c1 + c2*d_M) * ( np.log(d_Rjb) - np.log(d_Rref) )
        # dict_out = {'d_M': d_M, 'h_D': h_D, 'd_Rjb': d_Rjb, 'd_Rref': d_Rref,
        #             'c1': c1, 'c2': c2, 'f_r_g': f_r_g}
        
        return f_r_g#.to_numpy() #dict_out
    
    def get_anelastic_term(self, T, Rup):
        """Compute the anelastic attenuation term, Eq (4)"""
        h_D = self.get_h_D(Rup)
        d_Rjb = np.sqrt(Rup.Rjb**2 + h_D**2)
        d_Rref = np.sqrt(self.constants['Rref']**2 + h_D**2)
        # get c3 for the computation periods
        c3 = self.regression_coefficients['c3'].loc[T]
        
        f_r_a = c3/100 * ( d_Rjb - d_Rref )
        return f_r_a#.to_numpy()
    
    def get_magnitude_scaling_term(self, T, Rup):
        """Compute the magnitude scaling term, Eq (5)"""
        d_M = Rup.M - self.constants['Mh']
        
        # get coefficients b1, b2, and b3
        b1 = self.regression_coefficients['b1'].loc[T]
        b2 = self.regression_coefficients['b2'].loc[T]
        b3 = self.regression_coefficients['b3'].loc[T]
        
        # compute the magnitude scaling term
        if(Rup.M <= self.constants['Mh']):
            f_M = b1*d_M + b2*d_M**2
        else:
            f_M = b3*d_M
        
        return f_M#.to_numpy()
    
    def get_site_response_vs30(self, T, Site, is_fixed_fx_only = False):
        """Compute the site-amplification factor based on the Vs30 proxy, Eq (6)"""
        # reference Vs30 value
        ln_vs30_ref = np.log(Site.Vs30 / 800)
        
        # get coefficients g0_vs30, g1_vs30, and g2_vs30
        g0 = self.regression_coefficients['g0_vs30'].loc[T]
        g1 = self.regression_coefficients['g1_vs30'].loc[T]
        g2 = self.regression_coefficients['g2_vs30'].loc[T]
        
        # compute the site response
        sr_vs30 = g0 + g1*ln_vs30_ref + g2*(ln_vs30_ref**2)
        if(is_fixed_fx_only):
            return 0.0
        else:
            return sr_vs30#.to_numpy()
    
    def get_mean_ground_motion(self, T, Rup, Site, is_convert_to_g = True):
        """Compute the mean ground motion, Eq(2) plus the Vs30 proxy site 
        response from Eq (6)"""
        f_r_g = self.get_geometric_term(T, Rup)
        f_r_a = self.get_anelastic_term(T, Rup)
        f_M = self.get_magnitude_scaling_term(T, Rup)
        sr_vs30 = self.get_site_response_vs30(T, Site)
        
        ln_mu = f_r_g + f_r_a + f_M + sr_vs30
        
        if(is_convert_to_g):
            # GMPE is in gal (cm / s**2) so convert to units of g
            ln_mu -= np.log(981)
        return ln_mu
    
    def get_sigma(self, T, Rup, Site):
        """Compute standard deviations (sigma) at each of the periods; this 
        uses the formula for the ergodic variability but adjusted for the 
        site amplification using Vs30 as the proxy"""
        # get coefficients tau_event_0, tau_l2l, phi_0, and phi_s2s_vs30
        tau_event_0 = self.regression_coefficients['tau_event_0'].loc[T]
        tau_l2l = self.regression_coefficients['tau_l2l'].loc[T]
        phi_0 = self.regression_coefficients['phi_0'].loc[T]
        phi_s2s_vs30 = self.regression_coefficients['phi_s2s_vs30'].loc[T]
        
        # compute the total standard deviation
        sigma = np.sqrt( tau_event_0**2 + tau_l2l**2 + 
                        phi_0**2 + phi_s2s_vs30**2 )
        
        return sigma
    
    def get_median_and_sigma(self, T, Rup, Site):
        """Return the median ground motion (units of g, not in log space) and 
        the standard deviation at each of the periods"""
        
        
        # find periods which need interpolation, find neighbouring periods
        # make predictions / interpolations
        # get the final result
        
        periods_gmm = self.get_periods(T = None) # get periods for which the 
            # GMPE is defined
        # find periods for which the interpolation is necessary    
        periods_notInterp = T[ np.isin( T, periods_gmm ) ]
        periods_interp = T[ ~np.isin( T, periods_gmm ) ]
        
        ## compute the medians and sigmas for the input periods 
        # preallocate median and sigma arrays
        nT = T.shape[0]  
        median = np.zeros( nT )
        sigma = np.zeros( nT )
        
        for it in range(0, nT):
            # Teach = self.period1[it]
            Teach = T[it]
            # interpolate between periods, compute the median and sigma
            if(  ~np.any( (np.abs( periods_gmm - Teach ) < 0.0001) )  ):
                # interpolations periods
                T_low = np.array([np.max( periods_gmm[periods_gmm < Teach] )])
                T_hi = np.array([np.min( periods_gmm[periods_gmm > Teach] )])
                # this part is recursive
                sa_low, sigma_low = self.get_median_and_sigma(T_low, Rup, Site);
                sa_hi, sigma_hi = self.get_median_and_sigma(T_hi, Rup, Site);
                # interpolation -- linear in the log scale
                x = np.array([ np.log(T_low[0]), np.log(T_hi[0]) ])
                Y_sa = np.array([ np.log(sa_low[0]), np.log(sa_hi[0]) ])                
                Y_sigma = np.array([ sigma_low[0], sigma_hi[0] ])                
                f = scipy.interpolate.interp1d(x, Y_sa)
                median[it] = np.exp( f( np.log(Teach) )  )
                f = scipy.interpolate.interp1d(x, Y_sigma)
                sigma[it] = f( np.log(Teach) )
            else:
                median[it] = np.exp( self.get_mean_ground_motion(np.array([Teach]), Rup, Site, is_convert_to_g = True) )
                sigma[it] = self.get_sigma(np.array([Teach]), Rup, Site)
                
        # dict_out = {'periods_gmm' : periods_gmm,
        #             'periods_interp' : periods_interp,
        #             'periods_notInterp' : periods_notInterp,
        #             'median' : median,
        #             'sigma' : sigma
        #             }
        
        return median, sigma #dict_out #None

# -----------------------------------------------------------------------------
# Afshari & Stewart 2016, GMPE for significant duration
# -----------------------------------------------------------------------------

class AS_2016_duration(object):
    """Ground motion model (GMM) class for the Afshari&Stewart 2016 model"""
    
    # % Created by Jack Baker, August 9, 2016
    # % Based on code from Kioumars Afshari
    # % Updated 12/8/2017 to correct errors in AS16 equations 11 and 12
    # % Updated by Emily Mongold, 4/12/2021
    # %
    # % Source Model:
    # % Afshari, K., and Stewart, J. P. (2016). "Physically Parameterized
    # % Prediction Equations for Significant Duration in Active Crustal Regions."
    # % Earthquake Spectra, Vol. 32, No. 4, pp. 2057-2081.  
    # % doi: http://dx.doi.org/10.1193/063015EQS106M 
    # %
    # % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # % INPUT
    # %   rup           = rupture object input containing the following
    # %                   variables:
    # %       M             = earthquake magnitude
    # %       Rrup          = closest distance to rupture (km) (R, locally)
    # %       lambda        = rake angle, used to set Mech 
    # %                   Locally, Mech  = Rupture mechanism (0=unknown, 1=Normal, 2=Reverse, 3=Strike-slip)
    # %   site          = site object input containing the following
    # %                   variable:
    # %       Z10           = Basin depth (km); depth from the groundsurface to the
    # %                       1km/s shear-wave horizon.
    # %       Vs30          = shear wave velocity averaged over top 30 m in m/s
    # %       region        = 0 for global
    # %                     = 1 for California
    # %                     = 2 for Japan
    # %                     = 3 for China 
    # %                     = 4 for Italy 
    # %                     = 5 for Turkey
    # %                     = 6 for Taiwan
    # %                   Used to set local CJ flag for California (0), Japan
    # %                   (1), or other (-999)
    # %   dur_type      = 1 for 5-75% horizontal significant duration
    # %                 = 2 for 5-75% vertical significant duration
    # %                 = 3 for 5-95% horizontal significant duration
    # %                 = 4 for 5-95% vertical significant duration
    # %                 = 5 for 20-80% significant duration
    # %                 Locally, Def   = 1 for Ds5-75, =2 for DS5-95, = 3 for DS20-80
    # % OUTPUT
    # %   median          = median predicted duration
    # %   sigma           = log standard deviation of predicted duration
    # %   tau             = within-event log standard deviation
    # %   phi             = between-event log standard deviation
    # % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    # Class methods:
    #   TODO :: document   
    
    
    def __init__(self, Rup, Site, dur_type):
        self.Mech = self.get_Mech(Rup) # get fault type based on the rake angle
        self.Def = self.get_Def(dur_type) # duration definition to use
        self.median, self.sigma = self.get_median_and_sigma(Rup, Site)
        
    def get_Mech(self, Rup):
        """ Use Rup.lam (rake angle) to set Mech (rupture mechanism). """
        lam = Rup.lam
        if( ( lam <= 30 and lam >= -30 ) or ( lam <= 180 and lam >= 150 ) or ( lam <= -150 and lam >= -180 ) ):
            return 3 # Strike-slip fault
        elif( lam <= -30 ):
            return 1 # Normal fault
        elif( lam >= 30 ):
            return 2 # Reverse fault
        else:
            return 0 # Other / unspecified
        
    def get_Def(self, dur_type):
        """ Get which duration definition to use. """
        if( dur_type == 1 or dur_type == 2 ):
            return 1
        elif( dur_type == 3 or dur_type == 4 ):
            return 2
        elif( dur_type == 5 ):
            return 3
        else:
            print('Error--invalid value for input paramter ''Def'' \n')
            return

    def get_coeffs(self, Def, Mech):
        """Function to get duration definition-specific coefficients."""
        
        # ordering of all coefficients below is [5-75, 5-95, 20-80]
        Def = Def - 1 # change to zero-based indexing (MATLAB is 1-based)
        
        # ---------------------------------------------------------------------
        # source coefficients
        # ---------------------------------------------------------------------
        M1 = [5.35, 5.2, 5.2]
        M2 = [7.15, 7.4, 7.4]
        b00 = [1.28, 2.182, 0.8822]
        b01 = [1.555, 2.541, 1.409]
        b02 = [0.7806, 1.612, 0.7729]
        b03 = [1.279, 2.302, 0.8804]
        b10 = [5.576, 3.628, 6.182]
        b11 = [4.992, 3.17, 4.778]
        b12 = [7.061, 4.536, 6.579]
        b13 = [5.578, 3.467, 6.188]
        b2 = [0.9011, 0.9443, 0.7414]
        b3 = [-1.684, -3.911, -3.164]
        Mstar = [6, 6, 6]

        # ---------------------------------------------------------------------
        # path coefficients
        # ---------------------------------------------------------------------
        c1 = [0.1159, 0.3165, 0.0646]
        RR1 = [10, 10, 10]
        RR2 = [50, 50, 50]
        c2 = [0.1065, 0.2539, 0.0865]
        c3 = [0.0682, 0.0932, 0.0373]
        
        # ---------------------------------------------------------------------
        # site coefficients
        # ---------------------------------------------------------------------
        c4 = [-0.2246, -0.3183, -0.4237]
        Vref = [368.2, 369.9, 369.6]
        V1 = [600, 600, 600]
        c5 = [0.0006, 0.0006, 0.0005]
        dz1ref = [200, 200, 200]

        # ---------------------------------------------------------------------
        # compute definition-specific coefficients
        # ---------------------------------------------------------------------
        M1		= M1[Def]
        M2		= M2[Def]
        b00		= b00[Def]
        b01		= b01[Def]
        b02		= b02[Def]
        b03		= b03[Def]
        b10		= b10[Def]
        b11		= b11[Def]
        b12		= b12[Def]
        b13		= b13[Def]
        b2		= b2[Def]
        b3		= b3[Def]
        Mstar	= Mstar[Def]
        c1		= c1[Def]
        RR1		= RR1[Def]
        RR2		= RR2[Def]
        c2		= c2[Def]
        c3		= c3[Def]
        c4		= c4[Def]
        Vref	= Vref[Def]
        V1		= V1[Def]
        c5		= c5[Def]
        dz1ref	= dz1ref[Def]

        # ---------------------------------------------------------------------
        # mechanism-based coefficients
        # ---------------------------------------------------------------------
        if(Mech==0):
            b1=b10
            b0=b00
        elif(Mech==1):
            b1=b11
            b0=b01
        elif(Mech==2):
            b1=b12
            b0=b02
        elif(Mech==3):
            b1=b13
            b0=b03
            
        return M1, M2, b0, b1, b2, b3, Mstar, c1, RR1, RR2, c2, c3, c4, Vref, V1, c5, dz1ref
    
    def get_standard_dev(self, Def, M):
        
        Def = Def - 1 # change to zero-based indexing (MATLAB is 1-based)
        
        # ---------------------------------------------------------------------
        # standard deviation coefficients
        # ---------------------------------------------------------------------
        phi1	= [	0.54	, 0.43, 0.56]
        phi2	= [	0.41	, 0.35, 0.45]
        tau1	= [	0.28	, 0.25, 0.30]
        tau2	= [	0.25, 0.19, 0.19]
        # ---------------------------------------------------------------------
        # compute phi (eq 15)
        # ---------------------------------------------------------------------
        if(M<5.5):
            phi = phi1[Def]
        elif(M<5.75):
            phi = phi1[Def] + ( phi2[Def] - phi1[Def] ) * (M-5.5)/(5.75-5.5)
        else:
            phi = phi2[Def]

        # ---------------------------------------------------------------------
        # compute tau (eq 14)
        # ---------------------------------------------------------------------
        if(M<6.5):
            tau = tau1[Def]
        elif(M<7):
            tau = tau1[Def] + ( tau2[Def] - tau1[Def] ) * (M-6.5)/(7-6.5);
        else:
            tau = tau2[Def]
        
        return phi, tau
        
    def get_median_and_sigma(self, Rup, Site):
        """Compute median and sigma using the GMM relation"""        
        
        # ---------------------------------------------------------------------
        # Convert Site.region to local variable CJ
        # ---------------------------------------------------------------------
        if(Site.region == 1):
            CJ = 0
        elif(Site.region == 2):
            CJ = 1
        else:
            CJ = -999
            
        # ---------------------------------------------------------------------            
        # estimate median basin depth from Vs30
        # ---------------------------------------------------------------------
        if(CJ == 0): # California (eq 11)
            mu_z1=np.exp( -7.15 / 4 * np.log( (Site.Vs30**4 + 570.94**4)/(1360**4 + 570.94**4)) )
        else: # other regions (eq 12)
            mu_z1=np.exp( -5.23 / 4 * np.log( (Site.Vs30**2 + 412.39**2)/(1360**2 + 412.39**2)) )
        
        # ---------------------------------------------------------------------            
        # differential basin depth (eq 10)
        # ---------------------------------------------------------------------
        if( not hasattr(Site, 'Z10') or CJ == -999 ): # if z1p0 basin depth not defined
            dz1 = 0
        else:
            dz1 = Site.Z10 - mu_z1;


        # ---------------------------------------------------------------------
        # get coefficients
        # ---------------------------------------------------------------------
        M1, M2, b0, b1, b2, b3, Mstar, c1, RR1, RR2, c2, c3, c4, Vref, V1, c5, dz1ref = self.get_coeffs(self.Def, self.Mech)
        
        # ---------------------------------------------------------------------
        # Source term (eq 3)
        # ---------------------------------------------------------------------
        if(Rup.M<M1):
            F_E = b0 # constant duration at small M
        else:
            # Stress index parameter (eq 6)
            if(Rup.M<M2):
                deltaSigma = np.exp( b1 + b2*(Rup.M - Mstar) )
            else:
                deltaSigma = np.exp( b1 + b2*(M2-Mstar) + b3*(Rup.M-M2) )

            M_0 = 10**(1.5*Rup.M+16.05) # seismic moment (eq 5)
            f_0 = 4.9E6 * 3.2 * (deltaSigma / M_0)**(1/3) # corner frequency (eq 4)
            F_E = 1/f_0

        # ---------------------------------------------------------------------
        # Path term (eq 7)
        # ---------------------------------------------------------------------
        if(Rup.Rrup<RR1):
            F_P=c1*Rup.Rrup
        elif(Rup.Rrup<RR2):
            F_P=c1*RR1+c2*(Rup.Rrup-RR1)
        else:
            F_P=c1*RR1+c2*(RR2-RR1)+c3*(Rup.Rrup-RR2)

        # ---------------------------------------------------------------------
        # F_deltaz term (eq 9)
        # ---------------------------------------------------------------------
        if(dz1 <= dz1ref):
            F_deltaz = c5*dz1
        else:
            F_deltaz = c5*dz1ref

        # ---------------------------------------------------------------------
        # Site term (eq 8)
        # ---------------------------------------------------------------------
        if(Site.Vs30<V1):
            F_S=c4*np.log(Site.Vs30/Vref) + F_deltaz
        else:
            F_S=c4*np.log(V1/Vref) + F_deltaz
            
        # ---------------------------------------------------------------------
        # store different contributions in the object (for plotting)
        # ---------------------------------------------------------------------
        self.F_E = F_E # source term (s)
        self.F_P = F_P # path term (s)
        self.F_S = F_S # site term (take log to get seconds)
        
        # ---------------------------------------------------------------------
        # median duration (eq 2)
        # ---------------------------------------------------------------------
        ln_dur = np.log(F_E + F_P) + F_S
        median = np.exp(ln_dur)

        # ---------------------------------------------------------------------
        # standard deviation terms
        # ---------------------------------------------------------------------
        phi, tau = self.get_standard_dev(self.Def, Rup.M)
        self.phi = phi
        self.tau = tau
        sigma = np.sqrt(phi**2 + tau**2) # total standard deviation (eq 13)
        
        return median, sigma

# -----------------------------------------------------------------------------    
# IMPLEMENT the sa_avg class; 
# - this enables computation of sa_avg given the averaging periods, GMM object, 
#   and a correlation object (or function) 
# -----------------------------------------------------------------------------    

class SaAverage(object):
    """Class to compute Sa_avg intensity measure given the averaging periods,
    a GMM to use, correlation to use, and Rup and Site objects"""
    
    # Computation is based on Equations (11) and (13) in Baker&Jayaram 2008:
    # Correlation of spectral acceleration values from the NGA ground motion 
    # models
    
    # Class attributes (consistent with GMM object):
    #   median : median sa_avg in units of g
    #   sigma : logarithmic standard deviation of sa_avg 
    #   period1 : periods over which the averaging was performed      
    
    def __init__(self, T_avg, GMM_name, corr_fn, Rup, Site):
        
        """Compute sa_avg using the specified GMM, Rup, and Site objects;
        the corr_fn computes the correlation between periods, format needs to
        be corr_fn(T_i, T_j)"""
        
        # T_avg : array of averaging periods
        # GMM_name : string specifying which GMM object to use
        # Rup, Site : rupture and site objects for the specified GMM
        # corr_fn : function (handle) which computes the correlation of Sa 
        #   between different periods; input assumed as: rho = corr_fn(T_i, T_j)   
        
        if(GMM_name == 'BA_2008_active'):
            # instantiate the GMM
            gmm = BA_2008_active(T_avg, Rup, Site)  
        elif(GMM_name == 'KothaEtAl_2020'):
            gmm = KothaEtAl_2020(T_avg, Rup, Site)
        elif(GMM_name == 'CB_2014_active'):
            gmm = CB_2014_active(T_avg, Rup, Site)
        # TODO: implement handle for AG_2020    
        elif(GMM_name == 'AG_2020'):
            gmm = AbrahamsonGulerce2020(T_avg, Rup, Site, ergodic=False, apply_usa_adjustment=True)
        # TODO: implement handle for Abrahamson2015SInter
        elif(GMM_name == 'AbrahamsonEtAl2015SInter'): 
            gmm = AbrahamsonEtAl2015SInter(T_avg, Rup, Site)
        else:
            print('GMM {} not supported'.format(GMM_name))
            return None
        
        # compute the mean (Eq. 11, Baker and Jayaram 2008), convert to median
        median = np.exp( np.average( np.log( gmm.median ) ) )
        # compute sigma (Eq. 13, Baker and Jayaram 2008)
        periods = gmm.period1
        nT = periods.shape[0]
        sigma = 0
        for i in range(0, nT): # if made like a matrix that would likely be much faster
            for j in range(0, nT):
                sigma += corr_fn( periods[i], periods[j] ) * gmm.sigma[i] * gmm.sigma[j]
        sigma = np.sqrt( sigma / (nT**2) )
        
        # populate the class attributes
        self.median = np.array([median])
        self.sigma = np.array([sigma])
        self.period1 = periods
        # also store sigmas at all periods -- needed in correlation computation
        # for conditional spectra
        sigma_all_periods = gmm.sigma
        self.sigma_all_periods = sigma_all_periods





# -----------------------------------------------------------------------------    
# Conditional spectra computation
# -----------------------------------------------------------------------------

class IntensityMeasureType(object):
    """Class representing the type of intensity measure"""
    
    # Class attributes:
    #   im_name : string indicating type of IM, currently 'Sa', 'SaAvg', 'da5_75'
    #   sa_period : numpy array indicating the period of Sa (if im_name = 'Sa')
    #       or numpy array containing the averaging periods (if im_name = 'SaAvg')    
    
    def __init__(self, im_name, **kwargs):
        self.im_name = im_name # string representing the IM name;
        
        allowed_keys = {'sa_period'}
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)

class IntensityMeasure(object):
    """Class for intensity measure"""
    # Class attributes:
    #   iml : float, represents the value of the IM
    #   imt : object of IntensityMeasureType
    #   GMM_name : string, indicates the name of the GMM that can be used to predict
    #       this IM    
    #
    # stores type, value, corresponding GMM_name that is used to compute this IM
    def __init__(self, iml, imt, GMM_name):
        self.iml = iml # value of the intensity measure
        self.imt = imt # class representing the intensity measure type and 
            # additional info needed (e.g, period for Sa or averaging periods 
            # for SaAvg)
        self.GMM_name = GMM_name
        
    # implement a function to get median and sigma given Site and rupture
    # June 16, 2024
    def get_median_sigma(self, Rup, Site):
        """ Function to compute the median and sigma for this IM given Rup and
        Site objects. The function returns the median and sigma but does not
        store them as attributes of the object.
        
        Note: currently only implemented for the da5_75 horizontal IM.
        """
        
        if(self.GMM_name == 'AS_2016_duration'):
            # instantiate the GMM
            
            # flag to determine the duration to use
            if(self.imt.im_name == 'da5_75'):
                dur_type_use = 1
            elif(self.imt.im_name == 'da5_95'):
                dur_type_use = 3
            else:
                print('Duration type not supported for AS_2016_duration GMPE.')
            
            gmm = AS_2016_duration(Rup, Site, dur_type = dur_type_use) #1)
        elif(self.GMM_name == 'BahrampouriEtAlSInter2020_duration'):
            # instantiate the GMM
            # CHANGE THIS
            gmm = BahrampouriEtAlSInter2020_duration(Rup, Site, self.imt.im_name ) #dur_type = 'da5_75') #
            # gmm = BahrampouriEtAlSInter2020_duration(Rup, Site, dur_type = 'da5_95')
        elif(self.GMM_name == 'BahrampouriEtAlSSlab2020_duration'):
            # instantiate the GMM
            gmm = BahrampouriEtAlSSlab2020_duration(Rup, Site, self.imt.im_name )
        else:
            print('GMM {} not supported'.format(self.GMM_name))
            return None
        
        median = gmm.median
        sigma = gmm.sigma
        return median, sigma

    
def get_im_correlation(IM_star, IM_cond):
    """Function that computes the correlation coefficient given the conditioning 
    and conditional IM"""
    
    # Inputs: 
    #   IM_star : IntensityMeasure object, conditioning IM
    #   IM_cond : IntensityMeasure object, IM conditioned on IM_star    
    #
    # Output:    
    #   rho : float, correlation coefficient
    
    # Note: 
    #   IMs supported for IM_star : {'Sa', 'SaAvg'}
    #   IMs supported for IM_cond : {'Sa', 'da5_75'} TODO: add duration da5_95
    
    if(IM_star.imt.im_name == 'Sa' and IM_cond.imt.im_name == 'Sa'):
        # compute correlation using Baker and Jayaram 2008
        return float(sa_corr_baker(IM_star.imt.sa_period, IM_cond.imt.sa_period))
    
    if( (IM_star.imt.im_name == 'SaAvg' and IM_cond.imt.im_name == 'Sa') or
        (IM_star.imt.im_name == 'Sa' and IM_cond.imt.im_name == 'SaAvg') ):
        
        if(IM_star.imt.im_name == 'Sa' and IM_cond.imt.im_name == 'SaAvg'):
            # flip IM_star and IM_cond so that I can use the same code
            a_curr = copy.deepcopy(IM_star)
            b_curr = copy.deepcopy(IM_cond)
            IM_star = b_curr
            IM_cond = a_curr
        
        # compute correlation using Baker and Jayaram 2008 and corrected 
        # Equation 7 from Khorangi et al. 2017 (sqrt in the denominator)
        
        ## numerator
        num = 0
        T_i = IM_cond.imt.sa_period
        for j in range(0, IM_star.imt.sa_period.shape[0]):
            T_j = IM_star.imt.sa_period[j]
            sigma_j = IM_star.sigma_all_periods[j]
            num += sa_corr_baker(T_i, T_j) * sigma_j
        # ## denominator
        denom = 0
        nT = IM_star.imt.sa_period.shape[0]
        # print(nT)
        for i in range(0, nT):
            for j in range(0, nT):
                T_i = float(IM_star.imt.sa_period[i])
                T_j = float(IM_star.imt.sa_period[j])
                sigma_i = float(IM_star.sigma_all_periods[i])
                sigma_j = float(IM_star.sigma_all_periods[j])
                denom += sa_corr_baker(T_i, T_j) * sigma_i * sigma_j
        denom = np.sqrt(denom)
        return float(num / denom)
    
    # when using significant duration as IM_cond
    if( (IM_star.imt.im_name == 'SaAvg' and IM_cond.imt.im_name == 'da5_75') or
        (IM_star.imt.im_name == 'da5_75' and IM_cond.imt.im_name == 'SaAvg') ):
        # compute correlation using Baker&Jayaram 2008 and Bradley 2011 from
        # the corrected Eq. 7 in Khorangi et al. 2017 (sqrt in the denominator)
        
        if(IM_star.imt.im_name == 'da5_75' and IM_cond.imt.im_name == 'SaAvg'):
            # flip IM_star and IM_cond so that I can use the same code
            a_curr = copy.deepcopy(IM_star)
            b_curr = copy.deepcopy(IM_cond)
            IM_star = b_curr
            IM_cond = a_curr
        
        ## numerator
        num = 0
        # T_i = IM_cond.imt.sa_period
        for j in range(0, IM_star.imt.sa_period.shape[0]):
            T_j = IM_star.imt.sa_period[j]
            sigma_j = IM_star.sigma_all_periods[j]
            num += da5_75_sa_corr_bradley_2011(T_j) * sigma_j
        ## denominator
        denom = 0
        nT = IM_star.imt.sa_period.shape[0]
        # print(nT)
        for i in range(0, nT):
            for j in range(0, nT):
                T_i = float(IM_star.imt.sa_period[i])
                T_j = float(IM_star.imt.sa_period[j])
                sigma_i = float(IM_star.sigma_all_periods[i])
                sigma_j = float(IM_star.sigma_all_periods[j])
                denom += sa_corr_baker(T_i, T_j) * sigma_i * sigma_j
        denom = np.sqrt(denom)
        return float(num / denom)
    
    if( (IM_star.imt.im_name == 'Sa' and IM_cond.imt.im_name == 'da5_75') or
        (IM_star.imt.im_name == 'da5_75' and IM_cond.imt.im_name == 'Sa') ):
        
        if(IM_star.imt.im_name == 'da5_75' and IM_cond.imt.im_name == 'Sa'):
            # flip IM_star and IM_cond so that I can use the same code
            a_curr = copy.deepcopy(IM_star)
            b_curr = copy.deepcopy(IM_cond)
            IM_star = b_curr
            IM_cond = a_curr
        
        return float(da5_75_sa_corr_bradley_2011(IM_star.imt.sa_period))
    
    # I need this case when computing the covariance matrix in get_CS_target()
    if( IM_star.imt.im_name == 'da5_75' and IM_cond.imt.im_name == 'da5_75' ):
        return float(1.0)
    if( IM_star.imt.im_name == 'SaAvg' and IM_cond.imt.im_name == 'SaAvg' ):
        return float(1.0)
    
    # if none above retured a value, the print this 
    return 'Correlation not defined for this combination of IMs'
    
class ConditionalSpectra(object):
    """Class for Conditional Spectra given a single rupture, site, conditioning
    IM, IMs for which to compute the conditional distributions, correlation
    between IMs"""
    
    # Class attributes:
    #   Rup : rupture object
    #   Site : site object
    #   IM_star : IntensityMeasure object corresponding to the conditioning IM
    #       Additional attributes:
    #           median : computed from the specified GMM for the Rup and Site
    #           sigma : computed from the specified GMM for the Rup and Site  
    #           epsilon : back-calculated epsilon
    #   im_cond_list : list of IntensityMeasure objects corresponding to IMs
    #       for which the conditional distributions are to be computed    
    #   corr_fn : function computing correlation coefficients if IM_star is 
    #       SaAvg
    #    
    # Pseudocode:
    # - back calculate epsilon for IM_star :: DONE
    # - compute correlations IM_star and IM_cond :: DONE
    #   -- need to add an option to support significant duration as IM_cond
    # - compute the conditional medians and sigmas using Eqs. 5 and 6 :: DONE

    def __init__(self, Rup, Site, IM_star, im_cond_list, corr_fn):
           
        Rup = copy.deepcopy(Rup) 
        Site = copy.deepcopy(Site) 
        IM_star = copy.deepcopy(IM_star)
        im_cond_list = copy.deepcopy(im_cond_list)
        
        ## Compute median and sigma for the specified IMs for the Rup/Site
        #  Compute rho between IMs

        # Computation for IM_star -- need epsilon
        if( IM_star.imt.im_name in {'Sa', 'SaAvg'} ): 
            sa_avg_star = SaAverage(IM_star.imt.sa_period, IM_star.GMM_name, 
                                      corr_fn, Rup, Site)
            IM_star.median = sa_avg_star.median
            IM_star.sigma = sa_avg_star.sigma
            self.IM_star = IM_star
            #  back-calculate epsilon for IM_star
            IM_star.epsilon = self.get_epsilon(IM_star.iml, sa_avg_star.median, sa_avg_star.sigma)
            # compute correlation coefficient with the conditioning IM -- here equal to 1
            IM_star.rho = 1.0
            # store sigma values from all periods
            IM_star.sigma_all_periods = sa_avg_star.sigma_all_periods
        else:
            print('Conditional imt (IM_star) not supported')
        # Computation for all IMs conditioned on IM_star    
        self.im_cond_list = im_cond_list
        for IM_cond in im_cond_list:
            # Sa-based IMs
            if( IM_cond.imt.im_name in {'Sa', 'SaAvg'} ):  
                sa_avg_cond = SaAverage(IM_cond.imt.sa_period, IM_cond.GMM_name, 
                                          corr_fn, Rup, Site)
                IM_cond.median = sa_avg_cond.median
                IM_cond.sigma = sa_avg_cond.sigma
                # compute the correlation with the IM_star, save as attribute
                IM_cond.rho = get_im_correlation(IM_star, IM_cond)
            # significant duration, da5_75 IM case
            elif( IM_cond.imt.im_name in {'da5_75'} ):
                # compute median and sigma given Rup, Site
                IM_cond.median, IM_cond.sigma = IM_cond.get_median_sigma(Rup, Site)
                # compute correlation with IM_star
                IM_cond.rho = get_im_correlation(IM_star, IM_cond)
            else:
                print('imt not supported to be conditioned on IM_star')
        # the objects in the im_cond_list have these values stored as additional
        # attributes
        
        ## Compute conditional distributions of IMs : EQs. 5 and 6 in 
        #  Kohrangi et al. 2017
        for IM_cond in im_cond_list:
            # conditional median
            IM_cond.median_conditional = np.exp( np.log( IM_cond.median ) + 
                        IM_cond.rho * IM_cond.sigma * IM_star.epsilon)
            # conditional sigma
            IM_cond.sigma_conditional = IM_cond.sigma * np.sqrt( 1 - IM_cond.rho**2 )
    
        # compute the CS target and add to the atributes
        self.get_CS_target()
    
    def get_CS_target(self):
        """ Function called in init, computes the conditional covariance matrix
            ln_cov_mat_cond by reconstructing the matrix from the correlation 
            matrix and the 
            
            Parameters
            ----------
            
            Returns
            -------
            self.CS_target : class attribute, dict with following keys:
                ln_mu_cond : np array, mean vector of conditional IM 
                    distributions (in log)
                mu_cond : np array, np.exp(mu_cond)
                ln_std_cond : np array, lognormal standard deviations of 
                              conditional distributions
                ln_cov_mat_cond : n x n np.array, covariance matrix of 
                    conditional IM distributions (in log)
                im_obj_lst : list, either periods (floats) of Sa IMs or a string
                    representing duration (TODO: appended last), the elements
                    in the list are in the same order as in ln_mu_cond, mu_cond,
                    and ln_cov_mat_cond --- currently this is a list of 
                    IntensityMeasure objects (sorted based on period, where 
                    non-Sa IMs will be at the end), this should be used to get
                    CS distributions (plotting and targets) when dealing with 
                    all IMs :: TODO
                period_arr_cs : np array, floats representing periods of Sa IMs
            
            Notes: if IM_star (conditioning IM) is 'Sa' that IM is added to the 
                conditional spectra target; in case that IM_star is 'SaAvg' 
                then IM_star is not included in the IMs from the conditional 
                spectra target
            
        """
        # TODO: implement duration, stored duration and other non-Sa IMs at
        #       the end of the arrays and list in self.CS_target
        
        im_cond_list_cs = [] # list to store IntensityMeasure objects for CS
        period_list_cs = [] # list to periods (or string name for non-Sa IMs) 
                            # associated with IMs in im_cond_list_cs
        ln_mu_cond_list_cs = [] # mean of the conditional distrubutuions (in log)
        ln_sigma_cond_list_cs = [] # log standard deviations of conditional distributions
        
        # Add IM_star to the list in case it is an Sa intensity measure
        if(self.IM_star.imt.im_name == 'Sa'):
            im_cond_list_cs.append(self.IM_star)
            period_list_cs.append(self.IM_star.imt.sa_period[0])
            ln_mu_cond_list_cs.append( np.log(self.IM_star.iml) )
            ln_sigma_cond_list_cs.append(0.0) # conditioning IM
            
        # if SaAvg only has a single period (so it is essentially 'Sa'), also
        #   add to the list of conditional IMs
        if(self.IM_star.imt.im_name == 'SaAvg' and self.IM_star.imt.sa_period.shape[0] == 1):
            im_cond_list_cs.append(self.IM_star)
            period_list_cs.append(self.IM_star.imt.sa_period[0])
            ln_mu_cond_list_cs.append( np.log(self.IM_star.iml) )
            ln_sigma_cond_list_cs.append(0.0) # conditioning IM
        
        # # Add other Sa intensity measures to the list -- original code
        # for IM_cond in self.im_cond_list:
        #     if(IM_cond.imt.im_name == 'Sa'):
        #         im_cond_list_cs.append(IM_cond)
        #         period_list_cs.append(IM_cond.imt.sa_period[0]) 
        #         ln_mu_cond_list_cs.append( np.log(IM_cond.median_conditional[0]) )
        #         ln_sigma_cond_list_cs.append(IM_cond.sigma_conditional[0]) 
        
        # Add other Sa intensity measures to the list but don't duplicate if the period already exists
        for IM_cond in self.im_cond_list:
            if( IM_cond.imt.im_name == 'Sa' and ( IM_cond.imt.sa_period[0] not in period_list_cs ) ):
                im_cond_list_cs.append(IM_cond)
                period_list_cs.append(IM_cond.imt.sa_period[0]) 
                ln_mu_cond_list_cs.append( np.log(IM_cond.median_conditional[0]) )
                ln_sigma_cond_list_cs.append(IM_cond.sigma_conditional[0])         
        
        
        # Sort Sa IMs based on the periods
        # convert to np array, sort periods in the ascending order
        period_arr_cs = np.array(period_list_cs)
        idx_period_sort = np.argsort(period_list_cs)
        period_arr_cs = period_arr_cs[idx_period_sort] # sorted array of periods
        ln_mu_cond_arr_cs = np.array(ln_mu_cond_list_cs)[idx_period_sort] # sorted
        ln_sigma_cond_arr_cs = np.array(ln_sigma_cond_list_cs)[idx_period_sort] # sorted
        im_cond_list_cs = [ im_cond_list_cs[x] for x in idx_period_sort] # sorted

        # Append other non-Sa IMs
        # TODO: when implementing duration and other non-Sa IMs, add them to 
        #       the list here after Sa IMs are already sorted, I should add this
        #       to the list of periods
        
        # Append da5_75 duration to the end
        for IM_cond in self.im_cond_list:
            if(IM_cond.imt.im_name == 'da5_75'):
                im_cond_list_cs.append(IM_cond)
                # put im_name in the list of periods
                period_list_cs.append(IM_cond.imt.im_name)  
                # append to numpy arrays
                # print( IM_cond.median_conditional[0] )
                # print( IM_cond.sigma_conditional[0] )
                ln_mu_cond_arr_cs = np.append( ln_mu_cond_arr_cs, np.log(IM_cond.median_conditional[0]) )
                # np.append( ln_sigma_cond_arr_cs, IM_cond.sigma_conditional[0] )
                ln_sigma_cond_arr_cs = np.append( ln_sigma_cond_arr_cs, IM_cond.sigma_conditional )

        # get the n x n correlation matrix for the IMs in the im_cond_list_cs
        ln_cov_mat_cond, rho_mat = self.get_covariance_matrix(im_cond_list_cs, 
                                                        ln_sigma_cond_arr_cs)
        
        # add the attributes to self
        dict_cs = {
            'ln_mu_cond' : ln_mu_cond_arr_cs, 
            'mu_cond' : np.exp(ln_mu_cond_arr_cs),
            'ln_std_cond' : ln_sigma_cond_arr_cs,
            'ln_cov_mat_cond' : ln_cov_mat_cond,
            'im_obj_lst' : im_cond_list_cs,
            'period_arr_cs' : period_arr_cs,
            }
        
        self.CS_target = dict_cs
        
    def get_covariance_matrix_reconst(self, im_list, sigma_arr):
        """ Note: this one is based on reconstructing the covariance matrix;
        I need to check if this is correct; below I am implementing the 
        approach which follows Jayaram, Lin, Baker script ."""
        """ Given the list of IntensityMeasure objects and the array of 
        standard deviations, reconstruct the covariance matrix.
        
        # Discussion on how the link between the correlation and covariance 
        # matrices:
        # https://math.stackexchange.com/questions/198257/intuition-for-the-product-of-vector-and-matrices-xtax/198280#198280    
        
        Parameters
        ----------
        im_list : list of IntensityMeasure objects
        sigma_arr : array of floats, standard deviations
        
        Returns
        -------
        cov_mat : array n x n, covariance matrix where n is the number of 
            elements in im_list
        rho_arr : array n x n, correlation matrix
        
        """
        
        # compute the correlation matrix
        rho_list = []
        for IM_star in im_list:
            for IM_cond in im_list:
                rho_list.append( get_im_correlation(IM_star, IM_cond) )
        
        # convert to np.array and reshape to square matrix
        rho_arr = np.array(rho_list)
        n_dim = len(im_list)
        rho_arr = np.reshape(rho_arr, (n_dim, n_dim))
        
        # compute the covariance matrix
        cov_mat = np.dot( np.dot( np.diag(sigma_arr), rho_arr), np.diag(sigma_arr) )
        
        return cov_mat, rho_arr
    
    def get_covariance_matrix(self, im_list, sigma_arr):
        """ Given the list of IntensityMeasure objects and the array of 
        standard deviations, reconstruct the covariance matrix.
        
        # Discussion on how the link between the correlation and covariance 
        # matrices:
        # https://math.stackexchange.com/questions/198257/intuition-for-the-product-of-vector-and-matrices-xtax/198280#198280    
        
        Parameters
        ----------
        im_list : list of IntensityMeasure objects
        sigma_arr : array of floats, standard deviations
        
        Returns
        -------
        cov_mat : array n x n, covariance matrix where n is the number of 
            elements in im_list
        rho_arr : array n x n, correlation matrix
        
        """
        
        # ind_i = 0
        # ind_j = 0
        nIMs = len(im_list)
        cov_mat = np.zeros( (nIMs, nIMs) )
        # compute the covariance matrix
        for ind_i, IM_star in enumerate(im_list):
            for ind_j, IM_cond in enumerate(im_list):
                
                # # variaces
                if( np.isscalar(self.IM_star.sigma) ):
                    varT = (self.IM_star.sigma)**2
                else:
                    varT = (self.IM_star.sigma[0])**2
                # print(varT)
                sigma22 = varT;
                
                if( np.isscalar(IM_star.sigma) ):
                    var1 = (IM_star.sigma)**2
                else:    
                    var1 = (IM_star.sigma[0])**2
                
                if( np.isscalar(IM_cond.sigma) ):
                    var2 = (IM_cond.sigma)**2
                else:
                    var2 = (IM_cond.sigma[0])**2

                
                # covariances
                sigmaCorr = get_im_correlation(IM_star, IM_cond)*np.sqrt(var1*var2) 
                sigma11 = np.array([ [var1, sigmaCorr], [sigmaCorr, var2] ])
                sigma12 = np.array([ np.sqrt(var1*varT)*get_im_correlation(IM_star, self.IM_star) , np.sqrt(varT*var2) * get_im_correlation(self.IM_star, IM_cond)  ])                
                sigma12 = np.reshape(sigma12, (-1, 2))
                # sigmaCond = sigma11 - np.dot( np.dot(sigma12, np.linalg.inv(sigma22)), sigma12.T) 
                
                # sigmaCond = sigma11 - np.dot( sigma12*(1/sigma22) , sigma12.T)
                sigmaCond = sigma11 - np.dot( sigma12.T*(1/sigma22) , sigma12)
                # print(sigmaCond[0,1])
                cov_mat[ind_i, ind_j] = sigmaCond[0,1] 
                # debuggig, for np.dot -- I need to have 2d arrays, otherwise it does not do what I think it would    
                # if(ind_i == ind_j):
                #     print(f'index = {ind_j}')
                #     print('sigma22')
                #     print(sigma22)
                #     print('sigmaCorr')
                #     print(sigmaCorr)
                #     print('sigma11')
                #     print(sigma11)
                #     print('sigma12')
                #     print(sigma12)
                #     print('sigmaCond')
                #     print(sigmaCond)
                #     print('------------------')
                
        
        # find covariance values of zero and set them to a small number so that
        # random number generation can be performed
        cov_mat[ np.abs(cov_mat) < 1e-10 ] = 1e-10
        
        # compute correlation matrix
        rho_arr = None
        
        return cov_mat, rho_arr
        
    def get_epsilon(self, iml_target, median, sigma):
        # computes epsilon for the iml, Rup, Site, and GMM_name
        return ( np.log( iml_target ) - np.log( median ) ) / sigma
        
    def sample_cs_realizations(self, n_sample):
        """ Functions to draw realizations from the multivariate gaussian 
        representing the CS distributions. These realizations are the intensity
        measures of ground motions.
        
        Parameters
        ----------
        n_sample : int, number of realizations (i.e., ground motions) to draw
        
        Returns
        -------
        think what should go here, perhaps a dataframe with realizations? I have
        the "period_arr_cs" in the conditional spectra target already (strings 
        for durations should go here as well when enabled)
        
        
        """
        # TODO
    
    def export_CS_target(self, folder_path):
        """ The function exports the computed CS target to csv files in the 
        specified folder.
        
        Parameters
        ----------
        folder_path : str, folder to which to save the CS target data.
        
        Returns
        -------
        
        The following *.csv files are saved to the specified folder:
            ln_mu.csv : vector of means of conditional normal distributions
            ln_std.csv : vector of standard deviations of conditional 
                         normal distributions
            ln_cov_mat.csv : conditional covariance matrix of multivariate
                        normal distributions
        Note: all these results are for log-transformed values.
        
        """
        
        # create folder if it does not exist
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        
        # save numpy arrays to csv files
        np.savetxt(join(folder_path, "ln_mu.csv"), 
                      self.CS_target['ln_mu_cond'], delimiter=",")
        np.savetxt(join(folder_path, "ln_std.csv"), 
                      self.CS_target['ln_std_cond'], delimiter=",")
        np.savetxt(join(folder_path, "ln_cov_mat.csv"), 
                      self.CS_target['ln_cov_mat_cond'], delimiter=",")
        np.savetxt(join(folder_path, "cs_periods.csv"), 
                      self.CS_target['period_arr_cs'], delimiter=",")         
        
    
    def plot_CS(self, legend_title = None, is_sample_CS = True, n_sample = 25,
                is_plot_cs = True, is_plot_da5_75 = True, fig_name = None,
                is_save_fig = False):
        """Function to plot the conditional spectrum after the computation is done"""
        
        # get all IMs, see which ones are Sa and their periods
        # sort by period
        # plot the median and dispersions around (2.5 and 97.5 percentiles)
        
        # TODO:
            # (1) CS-target computed during initialization so I do not have to 
            #     compute this again here; can modify later
            # (2) add option to plot realizations of spectra -- this needs the 
            #     CS-target computed when initializing self; 
            # (3) overlay for control the CS-target computed during init; 
            #     done, works all good  
            
         
        # create lists containing conditional median, sigma, and period for all 'Sa' IMs
        median_conditional_list = []
        sigma_conditional_list = []
        period_list = []
        # Add IM_star to the list in case it is an Sa intensity measure
        if(self.IM_star.imt.im_name == 'Sa'):
            median_conditional_list.append(self.IM_star.iml)
            sigma_conditional_list.append(0.0) # conditioning IM
            period_list.append(self.IM_star.imt.sa_period[0])
        # case if SaAvg only has a single period    
        if(self.IM_star.imt.im_name == 'SaAvg' and self.IM_star.imt.sa_period.shape[0] == 1):
            median_conditional_list.append(self.IM_star.iml)
            sigma_conditional_list.append(0.0) # conditioning IM
            period_list.append(self.IM_star.imt.sa_period[0])
            
            
        # Add other Sa intensity measures
        for IM_cond in self.im_cond_list:
            if(IM_cond.imt.im_name == 'Sa'):
                median_conditional_list.append(IM_cond.median_conditional[0])
                sigma_conditional_list.append(IM_cond.sigma_conditional[0]) 
                period_list.append(IM_cond.imt.sa_period[0])
        
        # convert to np array, sort so that the periods are increasing   
        median_conditional_list = np.array(median_conditional_list)
        sigma_conditional_list = np.array(sigma_conditional_list)
        period_list = np.array(period_list)
        idx_sort = np.argsort(period_list)

        # plot the figure
        if(is_plot_cs):
            plt.figure(dpi = 200)
            plt.plot(period_list[idx_sort], median_conditional_list[idx_sort], '-ok',
                     label = 'median')
            # plot the percentiles
            plt.plot(period_list[idx_sort], 
                     np.exp( np.log(median_conditional_list[idx_sort]) + 1.96*sigma_conditional_list[idx_sort]  ),
                     '--k', label = '2.5 and 97.5 percentiles')
            plt.plot(period_list[idx_sort], 
                     np.exp( np.log(median_conditional_list[idx_sort]) - 1.96*sigma_conditional_list[idx_sort]  ),
                     '--k')
            
            # for control add in blue color the CS-target computed during init
            if(1 == 0):
                plt.plot(self.CS_target['period_arr_cs'], self.CS_target['mu_cond'],
                         '-b') #,label = 'median')
                plt.plot(self.CS_target['period_arr_cs'], 
                         np.exp( np.log(self.CS_target['mu_cond']) + 1.96*self.CS_target['ln_std_cond']  ),
                         '--b') #, label = '2.5 and 97.5 percentiles')
                plt.plot(self.CS_target['period_arr_cs'], 
                         np.exp( np.log(self.CS_target['mu_cond']) - 1.96*self.CS_target['ln_std_cond'] ),
                          '--b')
            
            # draw realizations of samples from CS-target distribution and plot
            if(is_sample_CS):
                # Modification to account for duration IMs -- plot only the portion
                # or indices up to size of self.CS_target['period_arr_cs']
                stop_idx = self.CS_target['period_arr_cs'].shape[0]
                #
                ln_mu = self.CS_target['ln_mu_cond']
                ln_cov = self.CS_target['ln_cov_mat_cond']
                y_samp = np.random.multivariate_normal(ln_mu, ln_cov, n_sample).T
                
                # ln_mu = self.CS_target['ln_mu_cond'][:stop_idx]
                # ln_cov = self.CS_target['ln_cov_mat_cond'][:stop_idx, :stop_idx]
                
                # y_samp = np.random.multivariate_normal(ln_mu, ln_cov, n_sample).T
                # plt.plot(self.CS_target['period_arr_cs'], np.exp(y_samp), '-k', alpha = 0.2)
                # plt.plot(self.CS_target['period_arr_cs'], np.exp(y_samp[:,0]), '-k', 
                #          label = 'sample spectrum', alpha = 0.2)
                
                y_samp = np.random.multivariate_normal(ln_mu, ln_cov, n_sample).T
                plt.plot(self.CS_target['period_arr_cs'], np.exp(y_samp[:stop_idx]), '-k', alpha = 0.2)
                plt.plot(self.CS_target['period_arr_cs'], np.exp(y_samp[:stop_idx,0]), '-k', 
                         label = 'sample spectrum', alpha = 0.2)
                
            plt.grid()
            plt.xscale('log')
            plt.yscale('log')
            plt.xlim(0.1, 10)
            #plt.ylim(1e-3, 5)
            plt.ylim(1e-2, 5)
            if legend_title is None:
                plt.legend(loc = 'lower left')
            else:
                plt.legend(loc = 'lower left', title = legend_title)
            plt.xlabel('Period (s)')
            plt.ylabel(r'$S_a (g)$')
            if is_save_fig:
                plt.savefig(fig_name+'_cs.png', bbox_inches='tight')
            plt.show()
        
        # If there are non-Sa IMs in the im_obj_lst, then plot separate figure
        # for each of those IMs
        
        # number of Sa IMs, these are ordered and they are the first n_periods
        # elements in the arrays
        n_periods = self.CS_target['period_arr_cs'].shape[0]
        n_IMs = len(self.CS_target['im_obj_lst'])
        n_non_sa_im = n_IMs - n_periods
        
        if(is_plot_da5_75):
            if(n_non_sa_im > 0):
                for idx_back in range(1, n_non_sa_im + 1):
                    # get the IM from the array from the end
                    im_name_curr = self.CS_target['im_obj_lst'][-idx_back].imt.im_name
                    median_cond_curr = self.CS_target['im_obj_lst'][-idx_back].median_conditional[0]
                    sigma_cond_curr = self.CS_target['im_obj_lst'][-idx_back].sigma_conditional
                    # plot the lognormal distribution
                    dist = scipy.stats.norm( loc = np.log(median_cond_curr), 
                                             scale = sigma_cond_curr )
                    x_plt = np.geomspace(1, 500, 500)
                    y_plt = np.array([ dist.cdf( np.log(x) ) for x in x_plt ])
                    
                    
                    plt.figure(dpi = 200)
                    plt.plot(x_plt, y_plt, '-k', label = im_name_curr)
                    if(is_sample_CS):
                        # extracct the correspondig realizations
                        y_plt_sample = y_samp[-idx_back, :]
                        sns.ecdfplot( np.exp(y_plt_sample) )
                    
                    # plt.legend()
                    if legend_title is None:
                        plt.legend(loc = 'lower left')
                    else:
                        plt.legend(loc = 'lower left', title = legend_title)
                    plt.grid()
                    plt.xlabel('IM values')
                    plt.ylabel('CDF')
                    plt.ylim(0, 1)
                    plt.xlim(1, 500)
                    plt.xscale('log')
                    if is_save_fig:
                        plt.savefig(fig_name+'_da5-75.png', bbox_inches='tight')
                    plt.show()
                
#%% EXACT CS Computation from the deaggregation output

# -----------------------------------------------------------------------------
# deaggregation plotting function
# -----------------------------------------------------------------------------

## This version of the function has issues with the clipping of the bars in the
#   3D view; I addressed this in the function plot_deagg import deagg_plot

# def deagg_plot(deagg_file, prob_col_name, delta_M, delta_R, investigation_time,
#                deagg_name):
#     """Plots the deaggregation output from openQuake; M/R deagg supported currently,
#     openQuake output is converted to the traditional format P(M=m, R=r | X > x)"""
    
#     # load the deagg csv to a dataframe
#     df_deagg = pd.read_csv(deagg_file, header = 1)

#     # compute P(M=m, R=r | X>x) :: openQuake Hazard Science Manual, section 2.4.2
#     tot_prob_ex = 1 - np.prod( 1 - df_deagg[prob_col_name].to_numpy() ) # P(X > x | T), 
#         # should be the poe for which the deaggregation is made, but small differences
#         # may arrise for numeric reasons (sampling of the hazard curve, because iml 
#         # is interpolated)
#     nu = -np.log( 1 - tot_prob_ex ) / investigation_time# this rate should be the 
#         # inverse of the return period
#     nu_m = -np.log( 1 - df_deagg[prob_col_name].to_numpy() ) / investigation_time + 0.0 
#         # I was getting -0.0 in the array so I add +0.0 to get rid of that

#     df_deagg['P(m | X>x)'] = nu_m / nu # traditional deaggregation output, should sum to 1; 

#     # compute mean M and mean R -- report on the plot in the second line of the title
#     mu_M = np.sum(df_deagg['P(m | X>x)'].to_numpy() * df_deagg['mag'].to_numpy())
#     mu_R = np.sum(df_deagg['P(m | X>x)'].to_numpy() * df_deagg['dist'].to_numpy())
    
#     # plot the deaggregation

#     x = df_deagg['dist'].to_numpy() - delta_R/2
#     y = df_deagg['mag'].to_numpy() - delta_M/2
#     z = [0] * x.shape[0]

#     dx = [delta_R/2] * x.shape[0]
#     dy = [delta_M/2] * x.shape[0]
#     dz = df_deagg['P(m | X>x)'].to_numpy()

#     plt.rcParams.update({'font.size': 16})

#     # plt.figure(dpi = 200, figsize = (8, 12))
#     plt.figure(dpi = 200, figsize = (16, 24))
#     ax = plt.axes(projection = '3d')
#     ax.bar3d(x, y, z, dx, dy,  dz, alpha = 0.7)

#     # ax.set_xlim(0,500)
#     ax.set_xlim(0,100)
#     ax.set_ylim(4.0, 10.0)

#     ax.set_xlabel('R [km]')
#     ax.set_ylabel('M')

#     ax.set_zlabel('% contribution')
#     plt.title(deagg_name + '\n mean: M = {}, R = {} km'.format(np.round(mu_M, 2), np.round(mu_R, 2)))

#     plt.show()

# from plot_deagg import deagg_plot # this is the deaggregation plotting function
# # that does not have an issue with the clipping; also alows for weight to be 
# # applied


#% fixing the clipping issue in the 3d bar plot

def sph2cart(r, theta, phi):
    '''spherical to cartesian transformation.'''
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def sphview(ax):
    '''returns the camera position for 3D axes in spherical coordinates'''
    r = np.square(np.max([ax.get_xlim(), ax.get_ylim()], 1)).sum()
    theta, phi = np.radians((90-ax.elev, ax.azim))
    return r, theta, phi

def ravzip(*itr):
    '''flatten and zip arrays'''
    return zip(*map(np.ravel, itr))

def deagg_plot(deagg_file, prob_col_name, delta_M, delta_R, investigation_time,
               deagg_name, plot_lim_distance = None, cam_elev = 35, cam_azim = 315,
               deagg_weight = 1.0):
    """Plots the deaggregation output from openQuake; M/R deagg supported currently,
    openQuake output is converted to the traditional format P(M=m, R=r | X > x)"""
    
    """Fixed the issue with the clipping in the bar plots; see following link:
    https://stackoverflow.com/questions/18602660/matplotlib-bar3d-clipping-problems"""
    
    # load the deagg csv to a dataframe
    df_deagg = pd.read_csv(deagg_file, header = 1)

    # compute P(M=m, R=r | X>x) :: openQuake Hazard Science Manual, section 2.4.2
    tot_prob_ex = 1 - np.prod( 1 - df_deagg[prob_col_name].to_numpy() ) # P(X > x | T), 
        # should be the poe for which the deaggregation is made, but small differences
        # may arrise for numeric reasons (sampling of the hazard curve, because iml 
        # is interpolated)
    nu = -np.log( 1 - tot_prob_ex ) / investigation_time# this rate should be the 
        # inverse of the return period
    nu_m = -np.log( 1 - df_deagg[prob_col_name].to_numpy() ) / investigation_time + 0.0 
        # I was getting -0.0 in the array so I add +0.0 to get rid of that

    df_deagg['P(m | X>x)'] = nu_m / nu # traditional deaggregation output, should sum to 1; 

    # compute mean M and mean R -- report on the plot in the second line of the title
    mu_M = np.sum(df_deagg['P(m | X>x)'].to_numpy() * df_deagg['mag'].to_numpy())
    mu_R = np.sum(df_deagg['P(m | X>x)'].to_numpy() * df_deagg['dist'].to_numpy())
    
    # remove parts of the data where distance is larger than specified limit
    if(plot_lim_distance is not None):
        df_deagg = df_deagg[df_deagg['dist'] <= plot_lim_distance]
        
    # plot the deaggregation

    x = df_deagg['dist'].to_numpy() - delta_R/2
    y = df_deagg['mag'].to_numpy() - delta_M/2
    z = [0] * x.shape[0]


    dx = [delta_R/2] * x.shape[0]
    dy = [delta_M/2] * x.shape[0]
    dz = deagg_weight*100*df_deagg['P(m | X>x)'].to_numpy() # percentage contribution to hazard

    plt.rcParams.update({'font.size': 16})

    plt.figure(dpi = 200, figsize = (14, 24))
    ax = plt.axes(projection = '3d')

    # ax.view_init(elev=45., azim=330)
    ax.view_init(elev = cam_elev, azim = cam_azim)

    # get the camera position and distance of each individual bar    
    x_c, y_c, z_c = sph2cart(*sphview(ax))       # camera position in xyz

    # compute distances of each bar to the camera
    zo = []
    for xx, yy, zz in zip(x, y, z):
        zo.append( xx*x_c + yy*y_c + zz*z_c ) # "distance" of bars from camera
    zo = np.array(zo)
    idx_plot_order = zo.argsort() # plotting order is based on sorted distances
    
    # plot each bar
    for idx in idx_plot_order: 
        x_pl, y_pl, z_pl, dx_pl, dy_pl, dz_pl = x[idx], y[idx], z[idx], dx[idx], dy[idx], dz[idx]
        if(dz_pl < 0.05):
            color_curr = 'white'
            edge_col_curr = 'white'
        else:
            color_curr = 'blue'
            edge_col_curr = 'black'
        
        pl = ax.bar3d(x_pl, y_pl, z_pl, dx_pl, dy_pl, dz_pl, color = color_curr, alpha = 0.9, 
                  edgecolor = edge_col_curr,
                  shade = False)
        pl._sort_zpos = zo[idx]
        
    # Get rid of the panes -- white background                          
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))   
    
    # ax.set_xlim(0,500)
    # ax.set_xlim(0,100)
    ax.set_xlim(0,200)
    ax.set_ylim(4.0, 10.0)
    ax.set_zlim(0.0, 20.0)

    ax.set_xlabel('R [km]')
    ax.set_ylabel('M')

    ax.set_zlabel('% contribution')
    plt.title(deagg_name + '\n mean: M = {}, R = {} km'.format(np.round(mu_M, 2), np.round(mu_R, 2)))
    
    # plt.autoscale(enable=True, axis='both', tight=True)
    # plt.grid()
    # plt.show(block=False)
    
    ax.grid(True, linestyle='--')
    plt.show()
    
    # return df_deagg





# -----------------------------------------------------------------------------
# parse deaggregation function
# -----------------------------------------------------------------------------

def deagg_parse(deagg_file, prob_col_name, delta_M, delta_R, investigation_time):
    """Plots the deaggregation output from openQuake; M/R deagg supported currently,
    openQuake output is converted to the traditional format P(M=m, R=r | X > x)"""
    
    # load the deagg csv to a dataframe
    df_deagg = pd.read_csv(deagg_file, header = 1)

    # compute P(M=m, R=r | X>x) :: openQuake Hazard Science Manual, section 2.4.2
    tot_prob_ex = 1 - np.prod( 1 - df_deagg[prob_col_name].to_numpy() ) # P(X > x | T), 
        # should be the poe for which the deaggregation is made, but small differences
        # may arrise for numeric reasons (sampling of the hazard curve, because iml 
        # is interpolated)
    nu = -np.log( 1 - tot_prob_ex ) / investigation_time# this rate should be the 
        # inverse of the return period
    nu_m = -np.log( 1 - df_deagg[prob_col_name].to_numpy() ) / investigation_time + 0.0 
        # I was getting -0.0 in the array so I add +0.0 to get rid of that

    df_deagg['P(m | X>x)'] = nu_m / nu # traditional deaggregation output, should sum to 1; 

    # compute mean M and mean R -- report on the plot in the second line of the title
    mu_M = np.sum(df_deagg['P(m | X>x)'].to_numpy() * df_deagg['mag'].to_numpy())
    mu_R = np.sum(df_deagg['P(m | X>x)'].to_numpy() * df_deagg['dist'].to_numpy())
    
    # plot the deaggregation

    x = df_deagg['dist'].to_numpy() - delta_R/2
    y = df_deagg['mag'].to_numpy() - delta_M/2
    z = [0] * x.shape[0]

    dx = [delta_R/2] * x.shape[0]
    dy = [delta_M/2] * x.shape[0]
    dz = df_deagg['P(m | X>x)'].to_numpy()

    return df_deagg

#%% ConditionalSpectraExact object, plotting

class ConditionalSpectraSimple(ConditionalSpectra):
    """ Subclass of ConditionalSpectra that reimplemments covariance 
    computation using a dummy function."""
        
    def get_covariance_matrix(self, im_list, sigma_arr):
        """ This is a stub that is currently useful so that the covariance is 
        not computed for the ConditionalSpectra object when computing the 
        covariance for the list of ruptures. Makes the computation faster. """
        # print('computing dummy covariance')
        return None, None
        

class ConditionalSpectraExact(ConditionalSpectra):
    """Subclass of ConditionalSpectra used to compute the 'exact' CS 
    distributions following the Lin et al. 2013 BSSA paper. """
    
    
    def __init__(self, rup_lst, site_cse, p_jd_arr, im_star_cse, 
                 im_cond_list_cse, corr_fn ):
        """ Initializes the ConditionalSpectraExact object. 
        
        Parameters
        ----------
        rup_lst : list, contains Rup objects defined from deaggregation
        site_cse : object, Site object defined for the location of interest
        p_jd_arr : array, floats defining probabilities associated with each of the
            ruptures in the rup_lst
        im_star_cse : IntensityMeasure object, this is the conditioning IM
        im_cond_list_cse : list, IntensityMeasure objects conditioned on im_star_cse 
        corr_fn : function computing correlation coefficients if IM_star is SaAvg
        
        """    
        # get a list of ConditionalSpectraSimple objects for all ruptures
        cs_lst = [ ConditionalSpectraSimple(rup, site_cse, im_star_cse, im_cond_list_cse, 
                                      sa_corr_baker) for rup in rup_lst ]  
        # compute the exact conditional spectrum
        self.cs_lst = cs_lst
        self.IM_star = im_star_cse
        self.im_cond_list = im_cond_list_cse
        self.CS_target = self.get_cs_exact(self.cs_lst, p_jd_arr)
    
    def get_cs_exact(self, cs_lst, p_jd_arr):
        # get exact ln_mu_cond : Eq 14 from Lin et al. 2013
        ln_mu_cond_exact = np.zeros( cs_lst[0].CS_target['ln_mu_cond'].shape[0] )
        ln_std_cond_exact = np.zeros( cs_lst[0].CS_target['ln_mu_cond'].shape[0] )
    
        for cs, prob in zip(cs_lst, p_jd_arr):
            ln_mu_cond_exact += prob * cs.CS_target['ln_mu_cond']
    
        # get exact ln_std_cond : Eq 15 from Lin et al. 2013
        for cs, prob in zip(cs_lst, p_jd_arr):
            ln_std_cond_exact += prob * ( cs.CS_target['ln_std_cond']**2 + ( cs.CS_target['ln_mu_cond'] - ln_mu_cond_exact )**2 )
        ln_std_cond_exact = np.sqrt(ln_std_cond_exact)
    
        # here I need to call a new get_covariance method; this is the idea
        # from each CS in the list, I sample n realizations (e.g. 100 samples);
        # the covariance for the cs_exact is then obtained as the weighted 
        # covariance of the sample realizations; to ensure that the matrix is 
        # PSD, I find the "closest" PSD matrix and that is used for the
        # conditional covariance
        # Notes: for this to work I need: 
            # compute the covariance matrices for each CS in the list -- this is a slow computation currently
            # have a get_realizations function that can sample each CS target to get the realizations
            # TODO:
                
        dict_cse = {
            'ln_mu_cond' : ln_mu_cond_exact, 
            'mu_cond' : np.exp(ln_mu_cond_exact),
            'ln_std_cond' : ln_std_cond_exact,
            'ln_cov_mat_cond' : 'TODO',#ln_cov_mat_cond,
            'im_obj_lst' : 'TODO',#im_cond_list_cs,
            'period_arr_cs' : 'TODO'#period_arr_cs,
            }
        return dict_cse


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
#%% TODO :: action items to finish
  #(1) - extend the CS spectra to have  FIV3 as 
    #   conditional IMs (all conditioned on Sa or SaAvg)
