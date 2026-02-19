#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri June 7 10:43:31 2024

@author: nbijelic

Implements the Bahrampouri et al. 2021 GMPE for significant durations to be 
used for the subduction interface events. Publication:
"Bahrampouri M, Rodriguez-Marek A, Green RA. Ground motion prediction equations 
for significant duration using the KiK-net database. Earthquake Spectra. 
2021;37(2):903-920. doi:10.1177/8755293020970971"

Code adapted from openquake engine 3.18: 
https://docs.openquake.org/oq-engine/3.18/reference/_modules/openquake/hazardlib/gsim/bahrampouri_2021_duration.html#BahrampouriEtAldm2021Asc

The ctx structure seems to have attributes from site and rupture


ctx.rake (but not needed for interface events, always reverse faulting style)
ctx.mag
ctx.rrup
ctx.vs30
ctx.z1pt0

# not needed
ctx.ztor

"""


import numpy as np
import pandas as pd
import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO


# Make a class for the CoeffsTable
class CoeffsTable(object):
    """ Class to mimicking the behavior of the openquake CoeffsTable. The class
    implements the call method so that the following works:
        
        C = CoeffsTable(coeff_table_string) # initialize the object
        C('da5_75') #returns the dictionary of coefficients for the 5-75% 
        significant duration.
        C('da5_75')['m1'] = 0.4733 # returns coefficient m1 for da5_75
        
        Note: openQuake uses call C['da5_75']['m1'], i.e., there the CoeffsTable
        object acts like a dictionary of dictionaries. This is the difference 
        here where, e.g., C('da5_75')['m1'] would be used. 
    
    Parameters
    ----------
    coeff_table_string : string of coefficients as used in openquake, e.g., for
        Bahrampouri et al. 2021 for interface events the following string is used:
            
     coeff_table_string = ""\
     imt       m1     m2      m3_RS  M1    M2    r1      r2       R1           s1      s2        s3      sig    tau    phi_s2s  phi_ss
     da5_95  0.2384  4.16    8.4117  5.5    8.0  0.08862  0.04194   200.0  -0.2875 0.001234  -0.03137  0.403  0.191   0.233    0.275
     da5_75  0.4733  6.1623  0.515   5.0    8.0  0.07505  0.0156    200.0  -0.1464 0.00075     0.357   0.490  0.218   0.238    0.369
     ""       
     
    """
    def __init__(self, coeff_table_string):
        coeff_table_string_oq = StringIO(coeff_table_string)
        self.df = pd.read_table(coeff_table_string_oq, delim_whitespace = True, 
                                index_col=0)
        # self.dur_type = self.dur_type
        # self.df.index = self.dur_type
        
    def __call__(self, dur_type):
        if( dur_type in ['da5_95', 'da5_75'] ):
            return self.df.loc[dur_type].to_dict()
        else:
            raise ValueError('Duration IM not supported.')

# Bahrampouri et al. 2021 for interface events - functions

def _get_source_term(trt, C, ctx):
    """
    Compute the source term described in Eq. 8:
    `` 10.^(m1*(M-m2))+m3``
    m3 = varies as per focal mechanism for ASC and Slab TRTs
    """
    if trt == 'subduction_interface':
        m3 = np.full_like(ctx.rake, C["m3_RS"])  # reverse
    else:
        # print('TRT not supported')
        # return None
        ss = C["m3_SS"]  # strike-slip
        m3 = np.full_like(ctx.rake, ss)
        m3[(ctx.rake <= -45.) & (ctx.rake >= -135.)] = C["m3_NS"]  # normal
        m3[(ctx.rake >= 45.) & (ctx.rake <= 135.)] = C["m3_RS"]  # reverse
    
    fsource = np.round(10 ** (C['m1'] * (ctx.mag - C['m2'])) + m3, 4)
    return fsource

def _get_path_term(C, ctx):
    """
    Implementing Eqs. 9, 10 and 11
    """
    slope = C['r1'] + C['r2'] * (ctx.mag - 4.0)
    mse = (ctx.mag - C['M1']) / (C['M2'] - C['M1'])
    mse[ctx.mag > C['M2']] = 1.
    mse[ctx.mag <= C['M1']] = 0.
    fpath = np.round(slope * ctx.rrup, 4)
    idx = ctx.rrup > C['R1']
    term = mse[idx] * (ctx.rrup[idx] - C['R1'])
    fpath[idx] = np.round(slope[idx] * (C['R1'] + term), 4)
    return fpath


def _get_site_term(C, ctx):
    """
    Implementing Eqs. 5, 6 and 12
    """
    mean_z1pt0 = (np.exp(((-5.23 / 2.) * np.log((ctx.vs30 ** 2. +
                  412.39 ** 2.) / (1360 ** 2. + 412.39 ** 2.)))-0.9))
    delta_z1pt0 = np.round(ctx.z1pt0 - mean_z1pt0, 4)
    fsite = []
    for i, value in enumerate(delta_z1pt0):
        s = (np.round(C['s1'] * np.log(min(ctx.vs30[i], 600.) / 600.) +
             C['s2']*min(delta_z1pt0[i], 250.0) + C['s3'], 4))
        fsite.append(s)
    return fsite


def _get_stddevs(C):
    """
    The authors have provided the values of
    sigma = np.sqrt(tau**2+phi_ss**2+phi_s2s**2)
    The within event term (phi) has been calculated by
    combining the phi_ss and phi_s2s
    """
    sig = C['sig']
    tau = C['tau']
    phi = np.sqrt(C['phi_ss']**2 + C['phi_s2s']**2)
    return sig, tau, phi

# Bahrampouri et al. 2021 for interface events - GMPE object

class BahrampouriEtAlSInter2020_duration(object):
    """
    Implements the significant duration GMPE for subduction interface events 
    based on:
        
    Implements the Bahrampouri et al. 2021 GMPE for significant durations to be 
    used for the subduction interface events. Publication:
    "Bahrampouri M, Rodriguez-Marek A, Green RA. Ground motion prediction equations 
    for significant duration using the KiK-net database. Earthquake Spectra. 
    2021;37(2):903-920. doi:10.1177/8755293020970971"
    
    Code adapted from openquake engine 3.18: 
    https://docs.openquake.org/oq-engine/3.18/reference/_modules/openquake/hazardlib/gsim/bahrampouri_2021_duration.html#BahrampouriEtAldm2021Asc

    
    Parameters
    ----------
    dur_type : string, 'da5_75' or 'da5_95'

    Rup : rupture object with following atributes:
        TODO ::: for the GMPE
        TODO ::: trt attribute goes here : 'subduction_interface'
        
    Site : site object with following attributes:
        TODO ::: for the GMPE


    Returns
    -------

    """
    def __init__(self, Rup, Site, dur_type, **kwargs):
        self.median, self.sigma = self.get_median_and_sigma(Rup, Site, dur_type)
    
    def get_median_and_sigma(self, Rup, Site, dur_type): #compute(self, ctx: np.recarray, imts, mean, sig, tau, phi):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.compute>`
        for spec of input and result values.
        """
        trt = Rup.trt #self.DEFINED_FOR_TECTONIC_REGION_TYPE
        
        # define ctx as an empty recarray, then set values like so:
        # x = np.recarray((1,), dtype=[('ztor', float), ('rrup', float), ('mag', float)]) 
        # x.rrup = 5.5

        ctx = np.recarray((1,), dtype=[('rake', float), ('rrup', float), 
                                       ('vs30', float), ('mag', float),
                                       ('z1pt0', float) ]) 
        ctx.rake = Rup.lam # rake angle
        ctx.rrup = Rup.Rrup
        ctx.vs30 = Site.Vs30
        ctx.mag = Rup.M
        ctx.z1pt0 = Site.Z10
        
        # create imts from dur_type; define other arrays as (M, 1)
        # at the end return median and sigma -- median is np.exp of the mean
        imts = [dur_type] # [ x for x in dur_type ] # list of M periods
        mean = np.empty(shape=( len(imts), 1))
        sig = np.empty(shape=( len(imts), 1))
        tau = np.empty(shape=( len(imts), 1))
        phi = np.empty(shape=( len(imts), 1))

        # computation happens here  
        

        for m, imt in enumerate(imts):
            C = self.COEFFS(imt)
            mean[m] = np.log(
                _get_source_term(trt, C, ctx) + _get_path_term(C, ctx)
            ) + _get_site_term(C, ctx)

            sig[m], tau[m], phi[m] = _get_stddevs(C)
    
        # modification so that the output works with the rest of the scripts
        mean = np.array([x[0] for x in mean])
        sig = np.array([x[0] for x in sig])
        median = np.exp(mean)
        sigma = sig        
        
        return median, sigma    
    
    # Coefficients taken from digital files supplied by Norm Abrahamson
    coeff_table_string = """\
    
    imt       m1     m2      m3_RS  M1    M2    r1      r2       R1           s1      s2        s3      sig    tau    phi_s2s  phi_ss
    da5_95  0.2384  4.16    8.4117  5.5    8.0  0.08862  0.04194   200.0  -0.2875 0.001234  -0.03137  0.403  0.191   0.233    0.275
    da5_75  0.4733  6.1623  0.515   5.0    8.0  0.07505  0.0156    200.0  -0.1464 0.00075     0.357   0.490  0.218   0.238    0.369
    """
    
    COEFFS = CoeffsTable(coeff_table_string)

# Bahrampouri et al. 2021 for in-slab events - GMPE object

class BahrampouriEtAlSSlab2020_duration(BahrampouriEtAlSInter2020_duration):
    
    # Coefficients taken from digital files supplied by Norm Abrahamson
    coeff_table_string = """\
    imt       m1     m2    m3_RS  m3_SS  m3_NS   M1  M2    r1      r2      R1     s1      s2        s3      sig    tau    phi_s2s  phi_ss
    da5_95  0.385  4.1604  5.828  4.231  5.496  5.5  8.0  0.09936  0.02495  200.0  -0.244  0.001409  -0.04109  0.458   0.194   0.245   0.335
    da5_75  0.4232 5.1674  0.975  0.3965 0.8712 5.0  8.0  0.057576  0.01316  200.0  -0.1431  0.001440  0.04534  0.593   0.261   0.288   0.449
    """

    COEFFS = CoeffsTable(coeff_table_string)
    
    
    
    
    
    