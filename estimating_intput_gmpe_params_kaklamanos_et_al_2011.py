#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
