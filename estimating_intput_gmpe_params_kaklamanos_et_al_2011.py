#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 10:44:36 2024

@author: nbijelic


Set of procedures to estimate the unknown parameters required for GMPEs as per:

Kaklamanos, James, Laurie G. Baise, and David M. Boore. "Estimating unknown 
input parameters when implementing the NGA ground-motion prediction equations 
in engineering practice." Earthquake Spectra 27.4 (2011): 1219-1235."

Main function is get_src_dist_params(M, R_jb, fault_style, F_hw)


Note: Conversions of vs30 to z1p0 and z2p5 not implemented.

"""

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Input variables
# -----------------------------------------------------------------------------
# M : moment magnitude
# R_jb : Joyner-Boore distance (km)
# fault_style : string ['reverse', 'normal', 'strike slip']
# F_hw : bool indicating is the site is on the hanging-wall side
#
# -----------------------------------------------------------------------------
# Computed variables
# -----------------------------------------------------------------------------
# lambda : rake angle (degrees)
# delta : fault dip angle (degrees)
# W : down-dip rupture width (km)
# Z_hyp : hypocentral depth (km)
# Z_tor : depth-to-top of rupture (km)
# alpha : azimuth (degrees)
# R_x : slant distance to closest point on the rupture plane (km)
# R_rup : horizontal distance to the surface projection of the top edge of the
#         rupture measured perpendicular to the fault strike (site coordinate)

# -----------------------------------------------------------------------------
# Pseudocode
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
## Part 1: SOURCE CHARACTERISTICS
# -----------------------------------------------------------------------------
# Step 1: recommended rake angle (lambda) based on the style of faulting
#   reverse fault, lambda = 90 degrees
#   normal fault, lambda = -90 degrees
#   left-lateral strike slip, lambda = 0 degrees
#   right-lateral strike slip, lambda = 180 degrees
#
# Step 2: fault dip angle, delta
#   - recommended values based on the style of faulting from the rake angle (lambda)
#   strike slip fault (vertical), delta = 90 degrees
#   normal fault, delta = 50 degrees (recommended average value)
#   reverse fault, delta = 40 degrees (recommended average value)
#
# Step 3: down-dip rupture width, W
#   - estimate from M and style of faulting using Wells&Coppersmith 1994 (Eq. 2)
#
# Step 4: depth-to-top of rupture, Z_tor
#   - estimate Z_hyp using Scherbaum et al. (2004)
#   - compute Z_tor from Z_hyp, W, delta (Eq. 3)
# -----------------------------------------------------------------------------
## Part 2: DISTANCE PARAMETERS
# -----------------------------------------------------------------------------
# Step 1: if alpha (azimuth) is unknown, use the recommended alpha values
#   - alpha = 50 degrees if site is on the hanging-wall site (F_hw = 1)
#   - alpha = -50 degrees if site is on the foot-wall site (F_hw = 0)
#
# Step 2: Compute R_x
#   - if delta = 90 degrees (vertical fault) -- Eq. (13)
#   - if delta != 90 degrees -- Table 2
#       - use conditional statements and intermediate results
#       
# Step 3: Compute R_rup
#   - if delta = 90 degrees (vertical fault) -- Eq. (21)
#   - if delta != 90 degrees -- Eqs. (14 - 20)

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

def get_lambda(fault_style):
    """ Get the recommended rake angle (lambda) based on the style of faulting.
    This can be used when the style of faulting is known by the rake is not
    known.
    
    Parameters
    ----------
    fault_style : string, {'reverse', 'normal', 'strike slip'}
    
    Returns
    -------
    np.array float
    
    """
    
    if(fault_style == 'reverse'): 
        return np.array([90])
    elif(fault_style == 'normal'): 
        return np.array([-90])
    elif(fault_style == 'strike_slip'): 
        return np.array([180])
    else:
        return 'fault_style not supported for lambda.'


def get_delta(fault_style):
    """ Estimate the fault dip angle based on the rake angle or style of 
    faulting.
    
    Parameters
    ----------
    fault_style : string, {'reverse', 'normal', 'strike slip'}
    
    Returns
    -------
    np.array float
    """
    
    if(fault_style == 'strike_slip'):
        return np.array([90])
    elif(fault_style == 'normal'):
        return np.array([50])
    elif(fault_style == 'reverse'):
        return np.array([40])
    else:
        return 'fault_style not supported for delta.'


def get_W(M, fault_style):
    """Estimate the down-dip rupture width W based on magnitude and style of 
    faulting using the empirical equations of Wells and Coppersmith (1994). 
    This is Equation (2) from the paper.
    
    Parameters
    ----------
    M : magnitude, np.array float
    fault_style : string, {'reverse', 'normal', 'strike slip'}
    
    Returns
    -------
    np.array float
    """

    if(fault_style == 'strike_slip'):
        return 10**(-0.76+0.27*M)
    elif(fault_style == 'reverse'):
        return 10**(-1.61+0.41*M)
    elif(fault_style == 'normal'):
        return 10**(-1.14+0.35*M)
    else:
        return 'fault_style not supported for W.'


def get_Z_hyp(M, fault_style):
    """ Estimate Z_hyp using the linear relationship between ZHYP and M 
    published in:
        Scherbaum, F., Schmedes, J., and Cotton, F., 2004. On the conversion of
        source-to-site distance measures for extended earthquake source models,
        Bulletin of the Seismological Society of America 94, 1053–1069.
        
    Equation (4) from the paper.    
    
    Parameters
    ----------
    M : magnitude, np.array float
    fault_style : string, {'reverse', 'normal', 'strike slip', 'unspecified'}
    
    Returns
    -------
    np.array float
    """
    
    if(fault_style == 'strike_slip'):
        return 5.63 + 0.68*M
    elif(fault_style in ['reverse','normal']):
        return 11.24 - 0.2*M
    elif(fault_style == 'unspecified'):
        return 7.08 + 0.61*M
    else:
        return 'fault_style not supported for Z_hyp.'


def get_Z_tor(Z_hyp, W, delta):
    """ 
    Compute the depth to top-of-rupture, Z_tor (km).
    Equation (3) from the paper. 
    
    Parameters
    ----------
    Z_hyp : np.array float, hypocentral depth (km)
    W : : np.array float, down-dip rupture width (km)
    delta : np.array float, fault dip angle (degrees)
    
    Returns
    -------
    np.array float
    """
    
    # Use np.maximum for element-wise comparison between array and scalar
    # This avoids the "inhomogeneous shape" error in NumPy 1.24+
    return np.maximum(Z_hyp - 0.6 * W * np.sin(np.deg2rad(delta)), 0)


def get_alpha(F_hw):
    """ Return recommended values of azimuth alpha based on the hanging-wall 
    flag. This is for the case when alpha is unspecified.
    
    Parameters
    ----------
    F_hw, bool flag indicating if the site is on the hanging-wall side
        if F_hw = True then hanging-wall side, otherwise foot-wall side
    
    Returns
    -------
    np.array float azimuth in degrees
    """
    
    if(F_hw):
        return np.array([50])
    else:
        return np.array([-50])


def get_R_x(alpha, R_jb, W, delta):
    """ Compute the horizontal distance to the surface projection of the top 
    edge of the rupture measured perpendicular to the fault strike, R_x using 
    Table 3 and Eq. 13.
    
    Parameters
    ----------
    alpha: np.array float, azimuth (degrees)
    R_jb: np.array float, Joyner-Boore distance (km)
    
    W: np.array float, down-dip rupture width (km)
    delta: np.array float, fault dip angle (degrees)
    
    
    Returns
    -------
    np.array float, R_x (km); horizontal distance to the surface projection of 
        the top edge of the rupture measured perpendicular to the fault strike.
    """
    
    alpha_rad = np.deg2rad(alpha)
    delta_rad = np.deg2rad(delta)
    
    if(delta == 90):
        return R_jb * np.sin( alpha_rad ) # Eq. 13
    if( (0 <= alpha <= 180) and (alpha != 90) ): # Equations (7) and (8)
        if( R_jb * np.abs( np.tan(alpha_rad) ) <= W * np.cos( delta_rad )  ):
            return R_jb * np.abs( np.tan(alpha_rad) ) #Eq. 7
        else:
            a = R_jb * np.tan(alpha_rad)
            b = np.arcsin( W * np.cos(delta_rad ) * np.cos(alpha_rad ) / R_jb )
            return a * np.cos( alpha_rad - b) #Eq. 8
    if( alpha == 90 ):
        if(R_jb > 0): 
            return R_jb + W * np.cos(delta_rad) # Eq. 9
        if(R_jb == 0):
            return 'R_jb == 0 is not supported, see Eqs. (10) and (11) for guidance.'     
    if( -180 <= alpha < 0 ):
        return R_jb * np.sin( alpha_rad ) # Eq. 12
    

def get_R_rup(R_x, R_jb, Z_tor, alpha, delta, W):
    """ Compute the slant distance to the closest point on the rupture plane, 
    R_rup using Eqs. (14) to (21).
    
    Parameters
    ----------
    R_x: np.array float, horizontal distance to the surface projection of the 
        top edge of the rupture measured perpendicular to the fault strike (km)
    R_jb: np.array float, Joyner-Boore distance (km)
    Z_tor: np.array float, depth to top-of-rupture (km) 
    alpha: np.array float, azimuth (degrees)
    delta: np.array float, fault dip (degrees)
    W: np.array float, down-dip rupture width (km)

    Returns
    -------
    np.array float
    
    """
        
    R_rup_prime = get_R_rup_prime(R_x, Z_tor, delta, W)
    R_y = get_R_y(alpha, R_jb, R_x)
    
    if( delta != 90 ):
        return np.sqrt( R_rup_prime**2 + R_y**2 ) # Eq. 14
    else:
        return np.sqrt( R_jb**2 + Z_tor**2 ) # Eq. 21


def get_R_rup_prime(R_x, Z_tor, delta, W):
    """ Function used in computation of R_rup."""
    
    delta_rad = np.deg2rad(delta) 
    a = Z_tor * np.tan(delta_rad)
    b = W / np.cos( delta_rad )
    
    if(R_x < a):
        return (R_x**2 + Z_tor**2)**0.5
    elif( (R_x >= a) and (R_x <= a + b) ):
        return R_x * np.sin( delta_rad ) + Z_tor * np.cos( delta_rad )
    elif( R_x > a + b ):
        return np.sqrt( ( R_x - W * np.cos( delta_rad ) )**0.5 + ( Z_tor + W * np.sin( delta_rad ) )**0.5 )


def get_R_y(alpha, R_jb, R_x):
    """ Function used in computation of R_rup. """
    if( np.abs( alpha ) == 90 ):
        return 0
    elif( ( alpha == 0 ) or ( np.abs( alpha ) == 180 ) ):
        return R_jb
    else:
        return np.abs( R_x / np.tan ( np.deg2rad( alpha ) ) )

# -----------------------------------------------------------------------------    
# Wrapper function for computing source characteristics and distance metrics    
# -----------------------------------------------------------------------------


def get_source_param(M, R_jb, fault_style):
    """ Function to compute the source characteristics
    
    Parameters
    ----------
    M : np.array float, moment magnitude
    R_jb : np.array float, Joyner-Boore distance (km)
    fault_style: string, style of faulting ['reverse', 'normal', 'strike slip']
    
    
    Returns
    -------
    dict source parameters, np.arrays float
        'lambda' : rake angle (degrees)
        'delta' : fault dip (degrees)
        'W' : rupture width (km)
        'Z_hyp' : hypocentral depth (km)
        'Z_tor' : depth to top-of-rupture (km)
    """
    
    lambda_deg = get_lambda(fault_style)
    delta_deg = get_delta(fault_style)
    W = get_W(M, fault_style)
    Z_hyp = get_Z_hyp(M, fault_style)
    Z_tor = get_Z_tor(Z_hyp, W, delta_deg)
    
    return {'lambda' : lambda_deg,
            'delta' : delta_deg,
            'W' : W,
            'Z_hyp' : Z_hyp,
            'Z_tor' : Z_tor
            }
  
def get_distance_param(F_hw, R_jb, W, delta, Z_tor):
    """ Function to compute the distance parameters.
    
    Parameters
    ----------
    F_hw : bool, True is site on the hanging-wall side
    R_jb : np.array float, Joyner-Boore distance (km)
    W : np.array float, rupture width (km)
    delta: np.array float, fault dip angle (degrees)
    Z_tor: np.array float, depth to top-of-rupture (km)
    
    Returns
    -------
    dict distance parameters, np.arrays float
        'alpha': azimuth (degrees)
        'R_x': horizontal distance to surface projection of the top edge of the
            rupture measured perpendicular to the fault strike (km)
        'R_rup': slant distance to closest point on rupture plane (km)
    
    """
    alpha_deg = get_alpha(F_hw)
    R_x = get_R_x(alpha_deg, R_jb, W, delta)
    R_rup = get_R_rup(R_x, R_jb, Z_tor, alpha_deg, delta, W)
    return {'alpha' : alpha_deg,
            'R_x' : R_x,
            'R_rup' : R_rup
           }

# -----------------------------------------------------------------------------    
# Main function
# -----------------------------------------------------------------------------

def get_src_dist_params(M, R_jb, fault_style, F_hw):
    """ Main function to estimate the source and distance parameters as per:

    Kaklamanos, J., Baise, L.G., and Boore, D.M. "Estimating unknown input 
    parameters when implementing the NGA ground-motion prediction equations 
    in engineering practice." Earthquake Spectra 27.4 (2011): 1219-1235.".
    
    
    Parameters
    ----------
    M : np.array float, moment magnitude
    R_jb : np.array float, Joyner-Boore distance (km)
    fault_style: string, style of faulting ['reverse', 'normal', 'strike slip']
    F_hw : bool, True is site on the hanging-wall side
    
    see get_source_param() and get_distance_param()
    
    
    Returns
    -------
    tuple with dictionaries (input_params, source_params, distance_params)
    """
    
    src_param = get_source_param(M, R_jb, fault_style)
    dist_param = get_distance_param(F_hw, R_jb, W = src_param['W'], 
                                    delta = src_param['delta'], 
                                    Z_tor = src_param['Z_tor'])
    input_param = {'M': M,
                  'R_jb': R_jb,
                  'fault_style': fault_style,
                  'F_hw': F_hw
        }
    
    return (input_param, src_param, dist_param)
    
#%% Example computation using the functions   
  
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
  
M = np.array([7.2])
R_jb = np.array([7])
fault_style = 'strike_slip'
F_hw = True    

src_param = get_source_param(M, R_jb, fault_style)
    
#src_dist_param_dict = get_src_dist_params(M, R_jb, fault_style, F_hw)

# Example output:
    # ({'M': array([7.2]),
    #   'R_jb': array([7]),
    #   'fault_style': 'strike_slip',
    #   'F_hw': True},
    #  {'lambda': array([180]),
    #   'delta': array([90]),
    #   'W': array([15.27566058]),
    #   'Z_hyp': array([10.526]),
    #   'Z_tor': array([1.36060365])},
    #  {'alpha': array([50]),
    #   'R_x': array([5.3623111]),
    #   'R_rup': array([7.1310057])})

#%%




#%% COMPUTATIONS FOR HIRO, SEATTLE SITE
#%%
# -----------------------------------------------------------------------------
# Seattle, 10in50, TRT: ACTIVE SHALLOW CRUST
# -----------------------------------------------------------------------------

M = np.array([6.87])
R_jb = np.array([12.45])
fault_style = 'strike_slip'
F_hw = False #True    
    
src_dist_param_dict = get_src_dist_params(M, R_jb, fault_style, F_hw)

# # For F_hw = True
# ({'M': array([6.87]),
#   'R_jb': array([12.45]),
#   'fault_style': 'strike_slip',
#   'F_hw': True},
#  {'lambda': array([180]),
#   'delta': array([90]),
#   'W': array([12.44228085]),
#   'Z_hyp': array([10.3016]),
#   'Z_tor': array([2.83623149])},
#  {'alpha': array([50]),
#   'R_x': array([9.53725332]),
#   'R_rup': array([12.76897447])})

# # For F_hw = False
# ({'M': array([6.87]),
#   'R_jb': array([12.45]),
#   'fault_style': 'strike_slip',
#   'F_hw': False},
#  {'lambda': array([180]),
#   'delta': array([90]),
#   'W': array([12.44228085]),
#   'Z_hyp': array([10.3016]),
#   'Z_tor': array([2.83623149])},
#  {'alpha': array([-50]),
#   'R_x': array([-9.53725332]),
#   'R_rup': array([12.76897447])})

# Note: the only difference for F_hw True or False is in the sign of alpha and 
#   R_x; use F_hw = True

#%% Seattle, 10in50, TRT: SUBDUCTION IN-SLAB
# -----------------------------------------------------------------------------

M = np.array([7.1])
R_jb = np.array([66.46])
fault_style = 'normal' #'reverse'
F_hw = True    
    
src_dist_param_dict = get_src_dist_params(M, R_jb, fault_style, F_hw)

## 'reverse'
# ({'M': array([7.1]),
#   'R_jb': array([66.46]),
#   'fault_style': 'reverse',
#   'F_hw': True},
#  {'lambda': array([90]),
#   'delta': array([40]),
#   'W': array([19.9986187]),
#   'Z_hyp': array([9.82]),
#   'Z_tor': array([2.10708141])},
#  {'alpha': array([50]),
#   'R_x': array([59.33939332]),
#   'R_rup': array([49.89701895])})

## 'normal'
# ({'M': array([7.1]),
#   'R_jb': array([66.46]),
#   'fault_style': 'normal',
#   'F_hw': True},
#  {'lambda': array([-90]),
#   'delta': array([50]),
#   'W': array([22.1309471]),
#   'Z_hyp': array([9.82]),
#   'Z_tor': 0},
#  {'alpha': array([50]),
#   'R_x': array([58.77500249]),
#   'R_rup': array([49.4273738])})

## 'strike-slip'
# ({'M': array([7.1]),
#   'R_jb': array([66.46]),
#   'fault_style': 'strike_slip',
#   'F_hw': True},
#  {'lambda': array([180]),
#   'delta': array([90]),
#   'W': array([14.35489433]),
#   'Z_hyp': array([10.458]),
#   'Z_tor': array([1.8450634])},
#  {'alpha': array([50]),
#   'R_x': array([50.91131369]),
#   'R_rup': array([66.4856064])})

#%% Seattle, 10in50, TRT: SUBDUCTION INTERFACE
# -----------------------------------------------------------------------------

M = np.array([8.96])
R_jb = np.array([114.59])
fault_style = 'reverse' #'strike_slip' #'normal' 
F_hw = True    
    
src_dist_param_dict = get_src_dist_params(M, R_jb, fault_style, F_hw)

## 'reverse'
# ({'M': array([8.96]),
#   'R_jb': array([114.59]),
#   'fault_style': 'reverse',
#   'F_hw': True},
#  {'lambda': array([90]),
#   'delta': array([40]),
#   'W': array([115.77105741]),
#   'Z_hyp': array([9.448]),
#   'Z_tor': 0},
#  {'alpha': array([50]),
#   'R_x': array([128.19089061]),
#   'R_rup': array([135.49868715])})

## 'normal'
# ({'M': array([8.96]),
#   'R_jb': array([114.59]),
#   'fault_style': 'normal',
#   'F_hw': True},
#  {'lambda': array([-90]),
#   'delta': array([50]),
#   'W': array([99.08319449]),
#   'Z_hyp': array([9.448]),
#   'Z_tor': 0},
#  {'alpha': array([50]),
#   'R_x': array([119.36231738]),
#   'R_rup': array([135.61746056])})

## 'strike_slip'
# ({'M': array([8.96]),
#   'R_jb': array([114.59]),
#   'fault_style': 'strike_slip',
#   'F_hw': True},
#  {'lambda': array([180]),
#   'delta': array([90]),
#   'W': array([45.62469771]),
#   'Z_hyp': array([11.7228]),
#   'Z_tor': 0},
#  {'alpha': array([50]), 
#   'R_x': array([87.78103274]), 
#   'R_rup': array([114.59])})

#%%
# -----------------------------------------------------------------------------
# Seattle, 2in50, TRT: ACTIVE SHALLOW CRUST
# -----------------------------------------------------------------------------

M = np.array([6.92])
R_jb = np.array([5.57])
fault_style = 'strike_slip' # 'normal' # 'reverse' # 
F_hw = True    
    
src_dist_param_dict = get_src_dist_params(M, R_jb, fault_style, F_hw)

## 'strike_slip'
# ({'M': array([6.92]),
#   'R_jb': array([5.57]),
#   'fault_style': 'strike_slip',
#   'F_hw': True},
#  {'lambda': array([180]),
#   'delta': array([90]),
#   'W': array([12.83512197]),
#   'Z_hyp': array([10.3356]),
#   'Z_tor': array([2.63452682])},
#  {'alpha': array([50]),
#   'R_x': array([4.26686755]),
#   'R_rup': array([6.16162572])})
#
## 'reverse'
# ({'M': array([6.92]),
#   'R_jb': array([5.57]),
#   'fault_style': 'reverse',
#   'F_hw': True},
#  {'lambda': array([90]),
#   'delta': array([40]),
#   'W': array([16.87329891]),
#   'Z_hyp': array([9.856]),
#   'Z_tor': array([3.34843152])},
#  {'alpha': array([50]),
#   'R_x': array([6.63806751]),
#   'R_rup': array([8.81475815])})
#
## 'normal'
# ({'M': array([6.92]),
#   'R_jb': array([5.57]),
#   'fault_style': 'normal',
#   'F_hw': True},
#  {'lambda': array([-90]),
#   'delta': array([50]),
#   'W': array([19.14255925]),
#   'Z_hyp': array([9.856]),
#   'Z_tor': array([1.05756932])},
#  {'alpha': array([50]),
#   'R_x': array([6.63806751]),
#   'R_rup': array([8.01613143])})

#%%
# -----------------------------------------------------------------------------
# Seattle, 2in50, TRT: SUBDUCTION IN-SLAB
# -----------------------------------------------------------------------------

M = np.array([7.17])
R_jb = np.array([62.48])
fault_style = 'strike_slip' # 'normal' # 'reverse' # 
F_hw = True    
    
src_dist_param_dict = get_src_dist_params(M, R_jb, fault_style, F_hw)

## 'reverse'
# ({'M': array([7.17]),
#   'R_jb': array([62.48]),
#   'fault_style': 'reverse',
#   'F_hw': True},
#  {'lambda': array([90]),
#   'delta': array([40]),
#   'W': array([21.36485748]),
#   'Z_hyp': array([9.806]),
#   'Z_tor': array([1.5661606])},
#  {'alpha': array([50]),
#   'R_x': array([56.78333175]),
#   'R_rup': array([47.75451112])})
#
## 'normal'
# ({'M': array([7.17]),
#   'R_jb': array([62.48]),
#   'fault_style': 'normal',
#   'F_hw': True},
#  {'lambda': array([-90]),
#   'delta': array([50]),
#   'W': array([23.41531475]),
#   'Z_hyp': array([9.806]),
#   'Z_tor': 0},
#  {'alpha': array([50]),
#   'R_x': array([56.11751851]),
#   'R_rup': array([47.20107088])})
#
## 'strike_slip' 
# ({'M': array([7.17]),
#   'R_jb': array([62.48]),
#   'fault_style': 'strike_slip',
#   'F_hw': True},
#  {'lambda': array([180]),
#   'delta': array([90]),
#   'W': array([14.9933956]),
#   'Z_hyp': array([10.5056]),
#   'Z_tor': array([1.50956264])},
#  {'alpha': array([50]),
#   'R_x': array([47.86245681]),
#   'R_rup': array([62.49823341])})

#%%
# -----------------------------------------------------------------------------
# Seattle, 2in50, TRT: SUBDUCTION INTERFACE
# -----------------------------------------------------------------------------

M = np.array([9.03])
R_jb = np.array([105.8])
fault_style = 'normal' # 'strike_slip' # 'reverse' #  
F_hw = True    
    
src_dist_param_dict = get_src_dist_params(M, R_jb, fault_style, F_hw)

## 'reverse'
# ({'M': array([9.03]),
#   'R_jb': array([105.8]),
#   'fault_style': 'reverse',
#   'F_hw': True},
#  {'lambda': array([90]),
#   'delta': array([40]),
#   'W': array([123.68014906]),
#   'Z_hyp': array([9.434]),
#   'Z_tor': 0},
#  {'alpha': array([50]),
#   'R_x': array([121.87230177]),
#   'R_rup': array([128.81989361])})
#
## 'normal'
# ({'M': array([9.03]),
#   'R_jb': array([105.8]),
#   'fault_style': 'normal',
#   'F_hw': True},
#  {'lambda': array([-90]),
#   'delta': array([50]),
#   'W': array([104.83347936]),
#   'Z_hyp': array([9.434]),
#   'Z_tor': 0},
#  {'alpha': array([50]),
#   'R_x': array([113.48757197]),
#   'R_rup': array([128.94267348])})
#
## 'strike_slip'
# ({'M': array([9.03]),
#   'R_jb': array([105.8]),
#   'fault_style': 'strike_slip',
#   'F_hw': True},
#  {'lambda': array([180]),
#   'delta': array([90]),
#   'W': array([47.65407017]),
#   'Z_hyp': array([11.7704]),
#   'Z_tor': 0},
#  {'alpha': array([50]), 
#   'R_x': array([81.04750208]), 
#   'R_rup': array([105.8])})
