#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nbijelic

This script is used to create the generalized coniditional intensity measure 
targets, specifically the conditional spectra and significant durations, for 
the Seattle site used in the paper:
    
"Dissipative Embedded Column Base Connections for Mitigating Collapse and 
Demolition Risk of Steel Moment-Resisting Frame Buildings" by Hiroyuki Inamasu,
Nenad Bijelic, and Dimitrios G. Lignos

Running the script will generate the figures of CS and Da5-75% targets for the 
4-story and 8-story MRFs at the 2in50 intensities for different TRTs. In addition,
the figures comparing the mean CS targets and the Da5-75% distributions for all
TRTs will also be generated. Finally, the script save the comuted GCIM targets.

"""
#import seaborn as sns
#import plotly.io as pio
#import plotly.graph_objects as go

import numpy as np
import pandas as pd
import scipy
import copy
import main_hazard_utils as mhu
from estimating_intput_gmpe_params_kaklamanos_et_al_2011 import get_src_dist_params
from os.path import join


# plotting
import matplotlib.pyplot as plt
# change font globally
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams.update({'font.size': 14})
# plotly
#pio.renderers.default = "browser"

#%% CS computations and figures for the 4-story MRF, 2in50, Seattle site

#==============================================================================
# 4-story MRF
#==============================================================================

# %% SEATTLE Site, for Hiro, 2in50 -- ACTIVE SHALLOW CRUST

# -----------------------------------------------------------------------------
# GMPE INFORMATION
# -----------------------------------------------------------------------------

# Sa GMPE: CB_2014
# Da5-75 GMPE: AS_2016
# Sa correlations: sa_corr_baker (Baker & Jayaram, 2008)
# Sa-Da correlations: Bradley (2011)
# Currently by default the GMMs output RotD100

# -----------------------------------------------------------------------------
# Deaggregation file data :: 2in50
# -----------------------------------------------------------------------------

haz_data_root = join('.','hazard_data', '4-story_mrf')
deagg_file = join(haz_data_root, 'saAvg_crustal_trt_4_story_mrf_2in50', 'Mag_Dist-mean_for_plot.csv')

prob_col_name = 'mean'  # column name containing P(X>x | M, R)
delta_M = 0.2
delta_R = 5
investigation_time = 1
# title of the plot
deagg_name = 'Seattle, SaAvg(0.2 : 0.1: 3.0), RP = 2475 years (2% in 50 years), active shallow crust'
# weight to apply to deaggregation values (enter manually)
deagg_weight_use = 0.738

mhu.deagg_plot(deagg_file, prob_col_name, delta_M, delta_R, investigation_time,
               deagg_name, plot_lim_distance=200, deagg_weight=deagg_weight_use)

df_deagg = mhu.deagg_parse(deagg_file, prob_col_name, delta_M, delta_R,
                           investigation_time)

# keep rows where df_deagg['P(m | X>x)'] is not zero, i.e., bins which contribute
df_deagg = df_deagg.loc[df_deagg['P(m | X>x)'] != 0]

# store data to rup_dict, this is used in subsequent computations of CS targets
rup_dict = {
    'M': df_deagg['mag'].to_numpy(),
    'R': df_deagg['dist'].to_numpy(),
    'p_jd': df_deagg['P(m | X>x)'].to_numpy(),
    'M_mean': np.sum(df_deagg['P(m | X>x)'].to_numpy() * df_deagg['mag'].to_numpy()),
    'R_mean': np.sum(df_deagg['P(m | X>x)'].to_numpy() * df_deagg['dist'].to_numpy())
}

# -----------------------------------------------------------------------------
# Estimate the source and distance parameters using Kaklamanos et al. 2011
# -----------------------------------------------------------------------------

"""
Faulting style for the Seatle fault zone (SFZ) taken as 'reverse' based on: 

Samuel Y. Johnson, Shawn V. Dadisman, Jonathan R. Childs, William D. Stanley; 
Active tectonics of the Seattle fault and central Puget Sound, Washington — 
Implications for earthquake hazards. GSA Bulletin 1999;; 111 (7): 1042–1053.

The Seattle fault is a zone of thrust or reverse faults that strikes through 
downtown Seattle in the densely populated Puget Lowland of western Washington.

"""
src_dist_dict = get_src_dist_params(M=rup_dict['M_mean'],
                                    R_jb=rup_dict['R_mean'],
                                    fault_style='reverse',
                                    F_hw=False)

# Note: I selected the case for the site on the foot wall because the distances
#   are smaller in that case compared to assuming the site on the hanging wall.

# -----------------------------------------------------------------------------
# Specify additional site parameters
# -----------------------------------------------------------------------------

vs30_seattle = 260  # assuming soil class D
# basin depths, USGS NSHM2018 consolidated values for the site
z_2p5_seattle = 6.374  # units: km
z_1p0_seattle = 0.876  # units: km -- AS_2016 uses the value in km
# global region in the CB_2014 GMPE for active shallow crust
region_seattle_cb_2014 = 0

# -----------------------------------------------------------------------------
# Instantiate site and rupture objects
# -----------------------------------------------------------------------------

site_cse = mhu.Site(Vs30=vs30_seattle,
                    region=region_seattle_cb_2014,
                    Z25=z_2p5_seattle,  # CB_2014 needs this value in km
                    Z10=z_1p0_seattle)  # AS_2016 needs this value in km

rup_mean = mhu.Rup(M=src_dist_dict[0]['M'],
                   R=None, # not needed
                   Rjb=src_dist_dict[0]['R_jb'],
                   Rrup=src_dist_dict[2]['R_rup'],
                   Rx=src_dist_dict[2]['R_x'],
                   Fhw=0,
                   W=src_dist_dict[1]['W'],
                   delta=src_dist_dict[1]['delta'],
                   lam=src_dist_dict[1]['lambda'],  
                   hypo_depth=src_dist_dict[1]['Z_hyp'],  
                   Ztor=src_dist_dict[1]['Z_tor'],
                   trt=None) # Not used in the CB_2014 GMPE

# -----------------------------------------------------------------------------
# Define IM_star and the list of IM_cond for CS computations
# -----------------------------------------------------------------------------

# IM_star & IM_cond -- Sa, SaAvg
GMPE_to_use = 'CB_2014_active'
iml_im_star = 0.766  # conditioning value of the IM, RotD100

# define the imt for the conditioning IM -- 4-story MRF
imt_star_cse = mhu.IntensityMeasureType('SaAvg', sa_period=np.array([0.2, 0.3,
                    0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
                    1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
                    2.8, 2.9, 3.0]))

# define the im object (intensity measure) for IM_star -- specified above
im_star_cse = mhu.IntensityMeasure(iml=iml_im_star, imt=imt_star_cse,
                                   GMM_name=GMPE_to_use)

# Target periods for computation of CS targets; here the same periods as used 
# the ground motion selection script by Baker et al. are used
period_tgt_list = [0.100000000000000, 0.117210229753348, 0.137382379588326,
                   0.161026202756094, 0.188739182213510, 0.221221629107045,
                   0.259294379740467, 0.303919538231320, 0.356224789026244,
                   0.417531893656040, 0.489390091847749, 0.573615251044868,
                   0.672335753649934, 0.788046281566991, 0.923670857187386,
                   1.08263673387405,  1.26896100316792,  1.48735210729351,
                   1.74332882219999,  2.04335971785694,  2.39502661998749,
                   2.80721620394118,  3.29034456231267,  3.85662042116347,
                   4.52035365636024,  5.29831690628371,  6.21016941891562,
                   7.27895384398315,  8.53167852417281,  10.0]


# Define the list of conditional IM, these are the IMs conditioned on the
# conditioning IM im_star
im_cond_list_cse = [mhu.IntensityMeasure(iml=None,
                    imt=mhu.IntensityMeasureType(
                        'Sa', sa_period=np.array([x])),
                    GMM_name=GMPE_to_use)
                    for x in period_tgt_list]

# IM_cond DURATION -- append duration IM
duration_GMPE_to_use = 'AS_2016_duration'
im_cond_list_cse.append(mhu.IntensityMeasure(iml=None,
                            imt=mhu.IntensityMeasureType('da5_75'),
                            GMM_name=duration_GMPE_to_use))

# -----------------------------------------------------------------------------
# Compute conditional spectra given a single rupture (i.e., M and R definition)
# -----------------------------------------------------------------------------
cs_mean = mhu.ConditionalSpectra(rup_mean, site_cse,
                                 im_star_cse, im_cond_list_cse,
                                 mhu.sa_corr_baker)

# plot the CS target based on the mean M and R
cs_mean.plot_CS(legend_title=None, is_sample_CS=False, n_sample=25,
                is_plot_cs=True, is_plot_da5_75=True,
                fig_name = "4_story_mrf_seattle_crustal_2in50", 
                is_save_fig = True)

# deepcopy the computed CS target for subsequent plotting
cs_mean_active = copy.deepcopy(cs_mean)

# %% SEATTLE Site, for Hiro, 2in50 -- SUBDUCTION IN-SLAB

# -----------------------------------------------------------------------------
# GMPE INFORMATION
# -----------------------------------------------------------------------------

# Sa GMPE: AG_2020 (Abrahamson & Gulerce 2020 GMPE for subduction zone events)
# Da5-75 GMPE: bahrampouri_et_al_2021 (duration GMPE for subduction zone events); compare to AS_2016
# Sa correlations: sa_corr_baker (Baker & Jayaram, 2008)
# Sa-Da correlations: Bradley (2011)

# -----------------------------------------------------------------------------
# Deaggregation file data :: 2in50
# -----------------------------------------------------------------------------

haz_data_root = join('.','hazard_data', '4-story_mrf')
deagg_file = join(haz_data_root, 'saAvg_inslab_trt_4_story_mrf_2in50', 'Mag_Dist-mean_for_plot.csv')


prob_col_name = 'mean'  # column name containing P(X>x | M, R)
delta_M = 0.2
delta_R = 5
investigation_time = 1
# title of the plot
deagg_name = 'Seattle, SaAvg(0.2 : 0.1: 3.0), RP = 2475 years (2% in 50 years), subduction in-slab'
# weight to apply to deaggregation values (enter manually)
deagg_weight_use = 0.0683

mhu.deagg_plot(deagg_file, prob_col_name, delta_M, delta_R, investigation_time,
               deagg_name, plot_lim_distance=200, deagg_weight=deagg_weight_use)

df_deagg = mhu.deagg_parse(deagg_file, prob_col_name, delta_M, delta_R,
                           investigation_time)

# keep rows where df_deagg['P(m | X>x)'] is not zero, i.e., bins which contribute
df_deagg = df_deagg.loc[df_deagg['P(m | X>x)'] != 0]

# store data to rup_dict, this is used in subsequent computations of CS targets
rup_dict = {
    'M': df_deagg['mag'].to_numpy(),
    'R': df_deagg['dist'].to_numpy(),
    'p_jd': df_deagg['P(m | X>x)'].to_numpy(),
    'M_mean': np.sum(df_deagg['P(m | X>x)'].to_numpy() * df_deagg['mag'].to_numpy()),
    'R_mean': np.sum(df_deagg['P(m | X>x)'].to_numpy() * df_deagg['dist'].to_numpy())
}

# -----------------------------------------------------------------------------
# Estimate the source and distance parameters using Kaklamanos et al. 2011
# -----------------------------------------------------------------------------

"""
Faulting style for the subduction zone taken as 'reverse'. This is not needed 
in the AG_2020 GMPE nor the bahrampouri_et_al_2021 duration GMPE. But it is 
used to estimate the unknown source and distance parameters. Additionally, this
would be needed for the AS_2016 duration GMPE (this one is the active shallow
crust events).

Note: 
    In this case I used the Z_tor and Z_hyp values as from the 2001 M6.8 
    Nisqually earthquake (in-slab event), all other parameters estimated using 
    the Kaklamanos et al. 2011 recommendations. The reason I did this was that
    when using the Kaklamanos et al. 2011 recommendations for estimation of all
    parameters, the Z_tor and Z_hyp values were smaller which resulted with 
    lower Sa intensities at small periods. This looked a bit odd and also would 
    yield a less good of a match during selection of the motions. I have stored
    both figures just for reference.

"""
src_dist_dict = get_src_dist_params(M=rup_dict['M_mean'],
                                    R_jb=rup_dict['R_mean'],
                                    fault_style='normal',
                                    F_hw=True)

# -----------------------------------------------------------------------------
# Specify additional site parameters
# -----------------------------------------------------------------------------

vs30_seattle = 260  # assuming soil class D
# basin depth terms, USGS NSHM2018 consolidated values for the site
z_2p5_seattle = 6.374  # units: km
z_1p0_seattle = 0.876  # units: km 
# global region in the CB_2014 GMPE for active shallow crust
region_seattle_cb_2014 = 0
region_seattle_AG_2020 = "CAS"  # cascadia subduction zone region

z_1p0_seattle_as_2016 = z_1p0_seattle  # AS_2016 needs this value in km
z_1p0_seattle_bahrampouri_2021 = z_1p0_seattle*1000 # # bahrampouri_et_al_2021 needs this value in (m)

# -----------------------------------------------------------------------------
# Instantiate site and rupture objects
# -----------------------------------------------------------------------------

site_cse = mhu.Site(Vs30=vs30_seattle,
                    region=region_seattle_AG_2020,
                    Z25=z_2p5_seattle,  # AG_2020 needs this value in km
                    Z10=z_1p0_seattle_bahrampouri_2021)  # AS_2016 needs this value in km;
# bahrampouri_et_al_2021 needs this value in (m)

rup_mean = mhu.Rup(M=src_dist_dict[0]['M'],
                   R=None,  # not needed
                   Rjb=src_dist_dict[0]['R_jb'],
                   Rrup=np.array([86.58347057]),
                   Rx=np.array([56.13215502]),
                   Fhw=0,
                   W=src_dist_dict[1]['W'],
                   delta=src_dist_dict[1]['delta'],
                   lam=src_dist_dict[1]['lambda'],
                   hypo_depth=np.array([53.17]),
                   Ztor=np.array([46.13]),  
                   trt='subduction_inslab')

# -----------------------------------------------------------------------------
# Define IM_star and the list of IM_cond for CS computations
# -----------------------------------------------------------------------------

# NOTE: the function always uses the below booleans by default, currently this
#       cannot be changed
# is_ergodic = False
# is_apply_usa_adjustment = True


# IM_star & IM_cond -- Sa, SaAvg
GMPE_to_use = 'AG_2020'
iml_im_star = 0.766  # conditioning value of the IM, RotD100

# define the imt for the conditioning IM -- 4-story MRF
imt_star_cse = mhu.IntensityMeasureType('SaAvg', sa_period=np.array([0.2, 0.3,
                    0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
                    1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
                    2.8, 2.9, 3.0]))


# define the im object (intensity measure) for IM_star -- specified above
im_star_cse = mhu.IntensityMeasure(iml=iml_im_star, imt=imt_star_cse,
                                   GMM_name=GMPE_to_use)

# Target periods for CS computation
period_tgt_list = [0.100000000000000, 0.117210229753348, 0.137382379588326,
                   0.161026202756094, 0.188739182213510, 0.221221629107045,
                   0.259294379740467, 0.303919538231320, 0.356224789026244,
                   0.417531893656040, 0.489390091847749, 0.573615251044868,
                   0.672335753649934, 0.788046281566991, 0.923670857187386,
                   1.08263673387405,  1.26896100316792,  1.48735210729351,
                   1.74332882219999,  2.04335971785694,  2.39502661998749,
                   2.80721620394118,  3.29034456231267,  3.85662042116347,
                   4.52035365636024,  5.29831690628371,  6.21016941891562,
                   7.27895384398315,  8.53167852417281,  10.0]


# Define the list of conditional IM, these are the IMs conditioned on the
# conditioning IM im_star
im_cond_list_cse = [mhu.IntensityMeasure(iml=None,
                    imt=mhu.IntensityMeasureType(
                        'Sa', sa_period=np.array([x])),
                    GMM_name=GMPE_to_use)
                    for x in period_tgt_list]

# IM_cond DURATION -- append duration IM
# duration_GMPE_to_use = AS_2016_duration'
duration_GMPE_to_use = 'BahrampouriEtAlSSlab2020_duration'
im_cond_list_cse.append(mhu.IntensityMeasure(iml=None,
                        imt=mhu.IntensityMeasureType('da5_75'),
                        GMM_name=duration_GMPE_to_use))

# -----------------------------------------------------------------------------
# Compute conditional spectra given a single rupture (i.e., M and R definition)
# -----------------------------------------------------------------------------
cs_mean = mhu.ConditionalSpectra(rup_mean, site_cse,
                                 im_star_cse, im_cond_list_cse,
                                 mhu.sa_corr_baker)

# plot the CS target based on the mean M and R
cs_mean.plot_CS(legend_title=None, is_sample_CS=False, n_sample=25,
                is_plot_cs=True, is_plot_da5_75=True,
                fig_name = "4_story_mrf_seattle_inslab_2in50", 
                is_save_fig = True)

# deepcopy the computed CS target for subsequent plotting
cs_mean_inslab = copy.deepcopy(cs_mean)

# %% SEATTLE Site, for Hiro, 2in50 -- SUBDUCTION INTERFACE

# -----------------------------------------------------------------------------
# GMPE INFORMATION
# -----------------------------------------------------------------------------

# Sa GMPE: AG_2020 (Abrahamson & Gulerce 2020 GMPE for subduction zone events)
# Da5-75 GMPE: bahrampouri_et_al_2021 (duration GMPE for subduction zone events); compare to AS_2016
# Sa correlations: sa_corr_baker (Baker & Jayaram, 2008)
# Sa-Da correlations: Bradley (2011)

# -----------------------------------------------------------------------------
# Deaggregation file data :: 2in50
# -----------------------------------------------------------------------------

haz_data_root = join('.','hazard_data', '4-story_mrf')
deagg_file = join(haz_data_root, 'saAvg_interface_trt_4_story_mrf_2in50', 'Mag_Dist-mean_for_plot.csv')



prob_col_name = 'mean'  # column name containing P(X>x | M, R); 
delta_M = 0.2
delta_R = 5
investigation_time = 1
# title of the plot
deagg_name = 'Seattle, SaAvg(0.2 : 0.1: 3.0), RP = 2475 years (2% in 50 years), subduction interface'
# weight to apply to deaggregation values (enter manually)
deagg_weight_use = 0.207

mhu.deagg_plot(deagg_file, prob_col_name, delta_M, delta_R, investigation_time,
               deagg_name, plot_lim_distance=200, deagg_weight=deagg_weight_use)

df_deagg = mhu.deagg_parse(deagg_file, prob_col_name, delta_M, delta_R,
                           investigation_time)

# keep rows where df_deagg['P(m | X>x)'] is not zero, i.e., bins which contribute
df_deagg = df_deagg.loc[df_deagg['P(m | X>x)'] != 0]

# store data to rup_dict, this is used in subsequent computations of CS targets
rup_dict = {
    'M': df_deagg['mag'].to_numpy(),
    'R': df_deagg['dist'].to_numpy(),
    'p_jd': df_deagg['P(m | X>x)'].to_numpy(),
    'M_mean': np.sum(df_deagg['P(m | X>x)'].to_numpy() * df_deagg['mag'].to_numpy()),
    'R_mean': np.sum(df_deagg['P(m | X>x)'].to_numpy() * df_deagg['dist'].to_numpy())
}

# -----------------------------------------------------------------------------
# Estimate the source and distance parameters using Kaklamanos et al. 2011
# -----------------------------------------------------------------------------

"""
Faulting style for the subduction zone taken as 'reverse'. This is not needed 
in the AG_2020 GMPE nor the bahrampouri_et_al_2021 duration GMPE. But it is 
used to estimate the unknown source and distance parameters. Additionally, this
would be needed for the AS_2016 duration GMPE (this one is the active shallow
crust events).

"""
src_dist_dict = get_src_dist_params(M=rup_dict['M_mean'],
                                    R_jb=rup_dict['R_mean'],
                                    fault_style='reverse',
                                    F_hw=False)

# -----------------------------------------------------------------------------
# Specify additional site parameters
# -----------------------------------------------------------------------------

vs30_seattle = 260  # assuming soil class D
# basin depth terms, USGS NSHM2018 consolidated values for the site
z_2p5_seattle = 6.374 # units: km
z_1p0_seattle = 0.876 # units: km
# global region in the CB_2014 GMPE for active shallow crust
region_seattle_cb_2014 = 0
region_seattle_AG_2020 = "CAS"  # cascadia subduction zone region

z_1p0_seattle_as_2016 = z_1p0_seattle  # AS_2016 needs this value in km
z_1p0_seattle_bahrampouri_2021 = z_1p0_seattle*1000 # bahrampouri_et_al_2021 needs this value in (m)

# -----------------------------------------------------------------------------
# Instantiate site and rupture objects
# -----------------------------------------------------------------------------

site_cse = mhu.Site(Vs30=vs30_seattle,
                    region=region_seattle_AG_2020,
                    Z25=z_2p5_seattle,  # AG_2020 needs this value in km
                    Z10=z_1p0_seattle_bahrampouri_2021, 
                    backarc = False)  

rup_mean = mhu.Rup(M=src_dist_dict[0]['M'],
                   R=None,  # not needed
                   Rjb=src_dist_dict[0]['R_jb'],
                   Rrup=src_dist_dict[2]['R_rup'],
                   # value is negative, there is no difference here when positive value is used
                   Rx=src_dist_dict[2]['R_x'],
                   Fhw=0,
                   W=src_dist_dict[1]['W'],
                   delta=src_dist_dict[1]['delta'],
                   lam=src_dist_dict[1]['lambda'],  
                   hypo_depth=src_dist_dict[1]['Z_hyp'],  
                   Ztor=src_dist_dict[1]['Z_tor'],
                   trt='subduction_interface')  


# -----------------------------------------------------------------------------
# Define IM_star and the list of IM_cond for CS computations
# -----------------------------------------------------------------------------

# NOTE: the function always uses the below booleans by default, currently this
#       cannot be changed
# is_ergodic = False
# is_apply_usa_adjustment = True


# IM_star & IM_cond -- Sa, SaAvg
GMPE_to_use ='AbrahamsonEtAl2015SInter' 

iml_im_star = 0.766  # conditioning value of the IM, RotD100


# define the imt for the conditioning IM -- 4-story MRF
imt_star_cse = mhu.IntensityMeasureType('SaAvg', sa_period=np.array([0.2, 0.3,
                    0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
                    1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
                    2.8, 2.9, 3.0]))

# define the im object (intensity measure) for IM_star -- specified above
im_star_cse = mhu.IntensityMeasure(iml=iml_im_star, imt=imt_star_cse,
                                   GMM_name=GMPE_to_use)

# Target periods for CS target computation
period_tgt_list = [0.100000000000000, 0.117210229753348, 0.137382379588326,
                   0.161026202756094, 0.188739182213510, 0.221221629107045,
                   0.259294379740467, 0.303919538231320, 0.356224789026244,
                   0.417531893656040, 0.489390091847749, 0.573615251044868,
                   0.672335753649934, 0.788046281566991, 0.923670857187386,
                   1.08263673387405,  1.26896100316792,  1.48735210729351,
                   1.74332882219999,  2.04335971785694,  2.39502661998749,
                   2.80721620394118,  3.29034456231267,  3.85662042116347,
                   4.52035365636024,  5.29831690628371,  6.21016941891562,
                   7.27895384398315,  8.53167852417281,  10.0]


# Define the list of conditional IM, these are the IMs conditioned on the
# conditioning IM im_star
im_cond_list_cse = [mhu.IntensityMeasure(iml=None,
                    imt=mhu.IntensityMeasureType(
                        'Sa', sa_period=np.array([x])),
                    GMM_name=GMPE_to_use)
                    for x in period_tgt_list]

# IM_cond DURATION -- append duration IM
duration_GMPE_to_use = 'BahrampouriEtAlSInter2020_duration'
im_cond_list_cse.append(mhu.IntensityMeasure(iml=None,
                        imt=mhu.IntensityMeasureType('da5_75'),
                        GMM_name=duration_GMPE_to_use))

# -----------------------------------------------------------------------------
# Compute conditional spectra given a single rupture (i.e., M and R definition)
# -----------------------------------------------------------------------------
cs_mean = mhu.ConditionalSpectra(rup_mean, site_cse,
                                 im_star_cse, im_cond_list_cse,
                                 mhu.sa_corr_baker)

# plot the CS target based on the mean M and R
cs_mean.plot_CS(legend_title=None, is_sample_CS=False, n_sample=25,
                is_plot_cs=True, is_plot_da5_75=True,
                fig_name = "4_story_mrf_seattle_interface_2in50", 
                is_save_fig = True)

# deepcopy the computed CS target for subsequent plotting
cs_mean_interface = copy.deepcopy(cs_mean)

# %% SEATTLE Site, for Hiro, 2in50 -- comparison of CS targets for different TRTs

color_crustal = 'k'
linestyle_crustal = '--'

color_inslab = 'goldenrod'
linestyle_inslab = ':'

color_interface = 'darkgreen'
linestyle_interface = (0, (6, 2, 3, 2))

sa_median_active = cs_mean_active.CS_target['mu_cond'][:-1]
sa_median_inslab = cs_mean_inslab.CS_target['mu_cond'][:-1]
sa_median_interface = cs_mean_interface.CS_target['mu_cond'][:-1]
periods_cs = cs_mean_active.CS_target['period_arr_cs']


plt.figure(dpi=200)
plt.plot(periods_cs, sa_median_active, linewidth=2.5,
         linestyle=linestyle_crustal, 
         color=color_crustal,
         label='Crustal (73.5%)')

plt.plot(periods_cs, sa_median_inslab, linewidth=2.5,
         linestyle=linestyle_inslab, 
         color=color_inslab,
         label='In-slab (6.5%)')

plt.plot(periods_cs, sa_median_interface, linewidth=2.5,
         linestyle=linestyle_interface,
         color=color_interface,
         label='Interface (20.0%)')

plt.xlabel('Period (s)')
plt.ylabel('Sa(T) (g)')
plt.xlim(0.1, 10)
plt.ylim(0.01, 5)
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig('4_story_mrf_seattle_cs_2in50.png', bbox_inches='tight')
plt.show()


da_median_active = cs_mean_active.CS_target['mu_cond'][-1]
da_median_inslab = cs_mean_inslab.CS_target['mu_cond'][-1]
da_median_interface = cs_mean_interface.CS_target['mu_cond'][-1]

da_sigma_active = cs_mean_active.CS_target['ln_std_cond'][-1]
da_sigma_inslab = cs_mean_inslab.CS_target['ln_std_cond'][-1]
da_sigma_interface = cs_mean_interface.CS_target['ln_std_cond'][-1]

# compute the lognormal distributions
x_plt = np.geomspace(1, 500, 500)

# active
median_cond_curr = da_median_active
sigma_cond_curr = da_sigma_active
dist = scipy.stats.norm(loc=np.log(median_cond_curr),
                        scale=sigma_cond_curr)
y_plt = np.array([dist.cdf(np.log(x)) for x in x_plt])
y_plt_active = y_plt

# inslab
median_cond_curr = da_median_inslab
sigma_cond_curr = da_sigma_inslab
dist = scipy.stats.norm(loc=np.log(median_cond_curr),
                        scale=sigma_cond_curr)
y_plt = np.array([dist.cdf(np.log(x)) for x in x_plt])
y_plt_inslab = y_plt

# interface
median_cond_curr = da_median_interface
sigma_cond_curr = da_sigma_interface
dist = scipy.stats.norm(loc=np.log(median_cond_curr),
                        scale=sigma_cond_curr)
y_plt = np.array([dist.cdf(np.log(x)) for x in x_plt])
y_plt_interface = y_plt


plt.figure(dpi=200)
plt.plot(x_plt, y_plt_active,
         linestyle=linestyle_crustal, 
         color=color_crustal,
         label='Crustal', 
         linewidth=2.5)

plt.plot(x_plt, y_plt_inslab,
         linestyle=linestyle_inslab, 
         color=color_inslab,
         label='In-slab', 
         linewidth=2.5)

plt.plot(x_plt, y_plt_interface,
         linestyle=linestyle_interface,
         color=color_interface,
         label='Interface', 
         linewidth=2.5)

# plt.legend(loc = 'best')
plt.grid()
plt.xlabel('Da 5-75% (s)')
plt.ylabel('CDF')
plt.ylim(0, 1)
plt.xlim(1, 500)
plt.xscale('log')
plt.savefig('4_story_mrf_seattle_da5-75_2in50.png', bbox_inches='tight')
plt.show()

# %% SEATTLE Site, for Hiro, 2in50 - save the CS targets

cs_save_dir = join('.','CS_target', 'seattle_4_story_mrf_2in50_rotD100')

folder_save = join(cs_save_dir, 'active_shallow_crust')
cs_mean_active.export_CS_target(folder_save)

folder_save = join(cs_save_dir,'interface')
cs_mean_interface.export_CS_target(folder_save)

folder_save = join(cs_save_dir,'inslab')
cs_mean_inslab.export_CS_target(folder_save)

#%% CS computations and figures for the 8-story MRF, 2in50, Seattle site




# -----------------------------------------------------------------------------
# 8-story MRF :: 2in50 intensity
# -----------------------------------------------------------------------------





# %% SEATTLE Site, for Hiro, 2in50 -- ACTIVE SHALLOW CRUST

# -----------------------------------------------------------------------------
# GMPE INFORMATION
# -----------------------------------------------------------------------------

# Sa GMPE: CB_2014
# Da5-75 GMPE: AS_2016
# Sa correlations: sa_corr_baker (Baker & Jayaram, 2008)
# Sa-Da correlations: Bradley (2011)

# Currently by default the GMMs output RotD100

# -----------------------------------------------------------------------------
# Deaggregation file data :: 2in50
# -----------------------------------------------------------------------------

haz_data_root = join('.','hazard_data', '8-story_mrf')
deagg_file = join(haz_data_root, 'saAvg_crustal_trt_8_story_mrf_2in50', 'Mag_Dist-mean_for_plot.csv')

prob_col_name = 'mean'  # column name containing P(X>x | M, R);
delta_M = 0.2
delta_R = 5
investigation_time = 1
# title of the plot
deagg_name = 'Seattle, SaAvg(0.3 : 0.1: 5.0), RP = 2475 years (2% in 50 years), active shallow crust'
# weight to apply to deaggregation values (enter manually)
deagg_weight_use = 0.764851485

mhu.deagg_plot(deagg_file, prob_col_name, delta_M, delta_R, investigation_time,
               deagg_name, plot_lim_distance=200, deagg_weight=deagg_weight_use)

df_deagg = mhu.deagg_parse(deagg_file, prob_col_name, delta_M, delta_R,
                           investigation_time)

# keep rows where df_deagg['P(m | X>x)'] is not zero, i.e., bins which contribute
df_deagg = df_deagg.loc[df_deagg['P(m | X>x)'] != 0]

# store data to rup_dict, this is used in subsequent computations of CS targets
rup_dict = {
    'M': df_deagg['mag'].to_numpy(),
    'R': df_deagg['dist'].to_numpy(),
    'p_jd': df_deagg['P(m | X>x)'].to_numpy(),
    'M_mean': np.sum(df_deagg['P(m | X>x)'].to_numpy() * df_deagg['mag'].to_numpy()),
    'R_mean': np.sum(df_deagg['P(m | X>x)'].to_numpy() * df_deagg['dist'].to_numpy())
}

# -----------------------------------------------------------------------------
# Estimate the source and distance parameters using Kaklamanos et al. 2011
# -----------------------------------------------------------------------------

"""
Faulting style for the Seatle fault zone (SFZ) taken as 'reverse' based on: 

Samuel Y. Johnson, Shawn V. Dadisman, Jonathan R. Childs, William D. Stanley; 
Active tectonics of the Seattle fault and central Puget Sound, Washington — 
Implications for earthquake hazards. GSA Bulletin 1999;; 111 (7): 1042–1053.

The Seattle fault is a zone of thrust or reverse faults that strikes through 
downtown Seattle in the densely populated Puget Lowland of western Washington.

"""
src_dist_dict = get_src_dist_params(M=rup_dict['M_mean'],
                                    R_jb=rup_dict['R_mean'],
                                    fault_style='reverse',
                                    F_hw=False)


# -----------------------------------------------------------------------------
# Specify additional site parameters
# -----------------------------------------------------------------------------

vs30_seattle = 260  # assuming soil class D
# basin depth terms, USGS NSHM2018 consolidated values for the site
z_2p5_seattle = 6.374 # units: km
z_1p0_seattle = 0.876 # units: km 
# global region in the CB_2014 GMPE for active shallow crust
region_seattle_cb_2014 = 0

# -----------------------------------------------------------------------------
# Instantiate site and rupture objects
# -----------------------------------------------------------------------------

site_cse = mhu.Site(Vs30=vs30_seattle,
                    region=region_seattle_cb_2014,
                    Z25=z_2p5_seattle,  # CB_2014 needs this value in km
                    Z10=z_1p0_seattle)  # AS_2016 needs this value in km

rup_mean = mhu.Rup(M=src_dist_dict[0]['M'],
                   R=None,  # not needed
                   Rjb=src_dist_dict[0]['R_jb'],
                   Rrup=src_dist_dict[2]['R_rup'],
                   Rx=src_dist_dict[2]['R_x'],
                   Fhw=0,
                   W=src_dist_dict[1]['W'],
                   delta=src_dist_dict[1]['delta'],
                   lam=src_dist_dict[1]['lambda'], 
                   hypo_depth=src_dist_dict[1]['Z_hyp'],
                   Ztor=src_dist_dict[1]['Z_tor'],
                   trt=None)  # Not used in the CB_2014 GMPE

# -----------------------------------------------------------------------------
# Define IM_star and the list of IM_cond for CS computations
# -----------------------------------------------------------------------------

# IM_star & IM_cond -- Sa, SaAvg
GMPE_to_use = 'CB_2014_active'
iml_im_star = 0.488  # conditioning value of the IM, RotD100

# define the imt for the conditioning IM -- 8-story MRF
imt_star_cse = mhu.IntensityMeasureType('SaAvg', sa_period=np.array([0.3,
                                                                     0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
                                                                     1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
                                                                     2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9,
                                                                     4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0]))


# define the im object (intensity measure) for IM_star -- specified above
im_star_cse = mhu.IntensityMeasure(iml=iml_im_star, imt=imt_star_cse,
                                   GMM_name=GMPE_to_use)

# Target periods for CS computation
period_tgt_list = [0.100000000000000, 0.117210229753348, 0.137382379588326,
                   0.161026202756094, 0.188739182213510, 0.221221629107045,
                   0.259294379740467, 0.303919538231320, 0.356224789026244,
                   0.417531893656040, 0.489390091847749, 0.573615251044868,
                   0.672335753649934, 0.788046281566991, 0.923670857187386,
                   1.08263673387405,  1.26896100316792,  1.48735210729351,
                   1.74332882219999,  2.04335971785694,  2.39502661998749,
                   2.80721620394118,  3.29034456231267,  3.85662042116347,
                   4.52035365636024,  5.29831690628371,  6.21016941891562,
                   7.27895384398315,  8.53167852417281,  10.0]

# Define the list of conditional IM, these are the IMs conditioned on the
# conditioning IM im_star
im_cond_list_cse = [mhu.IntensityMeasure(iml=None,
                    imt=mhu.IntensityMeasureType(
                        'Sa', sa_period=np.array([x])),
                    GMM_name=GMPE_to_use)
                    for x in period_tgt_list]

# IM_cond DURATION -- append duration IM
duration_GMPE_to_use = 'AS_2016_duration'
im_cond_list_cse.append(mhu.IntensityMeasure(iml=None,
                        imt=mhu.IntensityMeasureType('da5_75'),
                        GMM_name=duration_GMPE_to_use))

# -----------------------------------------------------------------------------
# Compute conditional spectra given a single rupture (i.e., M and R definition)
# -----------------------------------------------------------------------------
cs_mean = mhu.ConditionalSpectra(rup_mean, site_cse,
                                 im_star_cse, im_cond_list_cse,
                                 mhu.sa_corr_baker)

# plot the CS target based on the mean M and R
cs_mean.plot_CS(legend_title=None, is_sample_CS=False, n_sample=25,
                is_plot_cs=True, is_plot_da5_75=True,
                fig_name = "8_story_mrf_seattle_crustal_2in50", 
                is_save_fig = True)

# deepcopy the computed CS target for subsequent plotting
cs_mean_active = copy.deepcopy(cs_mean)

# %% SEATTLE Site, for Hiro, 2in50 -- SUBDUCTION IN-SLAB

# -----------------------------------------------------------------------------
# GMPE INFORMATION
# -----------------------------------------------------------------------------

# Sa GMPE: AG_2020 (Abrahamson & Gulerce 2020 GMPE for subduction zone events)
# Da5-75 GMPE: bahrampouri_et_al_2021 (duration GMPE for subduction zone events); compare to AS_2016
# Sa correlations: sa_corr_baker (Baker & Jayaram, 2008)
# Sa-Da correlations: Bradley (2011)

# -----------------------------------------------------------------------------
# Deaggregation file data :: 2in50
# -----------------------------------------------------------------------------

haz_data_root = join('.','hazard_data', '8-story_mrf')
deagg_file = join(haz_data_root, 'saAvg_inslab_trt_8_story_mrf_2in50', 'Mag_Dist-mean_for_plot.csv')

prob_col_name = 'mean'  # column name containing P(X>x | M, R)
delta_M = 0.2
delta_R = 5
investigation_time = 1
# title of the plot
deagg_name = 'Seattle, SaAvg(0.3 : 0.1: 5.0), RP = 2475 years (2% in 50 years), subduction in-slab'
# weight to apply to deaggregation values (enter manually)
deagg_weight_use = 0.027475248

mhu.deagg_plot(deagg_file, prob_col_name, delta_M, delta_R, investigation_time,
               deagg_name, plot_lim_distance=200, deagg_weight=deagg_weight_use)

df_deagg = mhu.deagg_parse(deagg_file, prob_col_name, delta_M, delta_R,
                           investigation_time)

# keep rows where df_deagg['P(m | X>x)'] is not zero, i.e., bins which contribute
df_deagg = df_deagg.loc[df_deagg['P(m | X>x)'] != 0]

# store data to rup_dict, this is used in subsequent computations of CS targets
rup_dict = {
    'M': df_deagg['mag'].to_numpy(),
    'R': df_deagg['dist'].to_numpy(),
    'p_jd': df_deagg['P(m | X>x)'].to_numpy(),
    'M_mean': np.sum(df_deagg['P(m | X>x)'].to_numpy() * df_deagg['mag'].to_numpy()),
    'R_mean': np.sum(df_deagg['P(m | X>x)'].to_numpy() * df_deagg['dist'].to_numpy())
}

# -----------------------------------------------------------------------------
# Estimate the source and distance parameters using Kaklamanos et al. 2011
# -----------------------------------------------------------------------------

"""
Faulting style for the subduction zone taken as 'reverse'. This is not needed 
in the AG_2020 GMPE nor the bahrampouri_et_al_2021 duration GMPE. But it is 
used to estimate the unknown source and distance parameters. Additionally, this
would be needed for the AS_2016 duration GMPE (this one is the active shallow
crust events).

Note: 
    In this case I used the Z_tor and Z_hyp values as from the 2001 M6.8 
    Nisqually earthquake (in-slab event), all other parameters estimated using 
    the Kaklamanos et al. 2011 recommendations. The reason I did this was that
    when using the Kaklamanos et al. 2011 recommendations for estimation of all
    parameters, the Z_tor and Z_hyp values were smaller which resulted with 
    lower Sa intensities at small periods. This looked a bit odd and also would 
    yield a less good of a match during selection of the motions. I have stored
    both figures just for reference.

"""
src_dist_dict = get_src_dist_params(M=rup_dict['M_mean'],
                                    R_jb=rup_dict['R_mean'],
                                    fault_style='normal',
                                    F_hw=True)

# -----------------------------------------------------------------------------
# Specify additional site parameters
# -----------------------------------------------------------------------------

vs30_seattle = 260  # assuming soil class D
# basin depth terms, USGS NSHM2018 consolidated values for the site
z_2p5_seattle = 6.374 # units: km
z_1p0_seattle = 0.876 # units: km 
# global region in the CB_2014 GMPE for active shallow crust
region_seattle_cb_2014 = 0
region_seattle_AG_2020 = "CAS"  # cascadia subduction zone region

z_1p0_seattle_as_2016 = z_1p0_seattle  # AS_2016 needs this value in km
z_1p0_seattle_bahrampouri_2021 = z_1p0_seattle*1000 # bahrampouri_et_al_2021 needs this value in (m)

# -----------------------------------------------------------------------------
# Instantiate site and rupture objects
# -----------------------------------------------------------------------------

site_cse = mhu.Site(Vs30=vs30_seattle,
                    region=region_seattle_AG_2020,
                    Z25=z_2p5_seattle,  
                    Z10=z_1p0_seattle_bahrampouri_2021)  


rup_mean = mhu.Rup(M=src_dist_dict[0]['M'],
                   R=None,  # not needed
                   Rjb=src_dist_dict[0]['R_jb'],
                   Rrup=np.array([86.58347057]),
                   Rx=np.array([56.13215502]),
                   Fhw=0,
                   W=src_dist_dict[1]['W'],
                   delta=src_dist_dict[1]['delta'],
                   lam=src_dist_dict[1]['lambda'],
                   hypo_depth=np.array([53.17]),
                   Ztor=np.array([46.13]),  
                   trt='subduction_inslab')

# -----------------------------------------------------------------------------
# Define IM_star and the list of IM_cond for CS computations
# -----------------------------------------------------------------------------


# IM_star & IM_cond -- Sa, SaAvg
GMPE_to_use = 'AG_2020'
iml_im_star = 0.488  # conditioning value of the IM, RotD100

# define the imt for the conditioning IM -- 8-story MRF
imt_star_cse = mhu.IntensityMeasureType('SaAvg', sa_period=np.array([0.3,
                                                                     0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
                                                                     1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
                                                                     2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9,
                                                                     4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0]))


# define the im object (intensity measure) for IM_star -- specified above
im_star_cse = mhu.IntensityMeasure(iml=iml_im_star, imt=imt_star_cse,
                                   GMM_name=GMPE_to_use)

# Target periods used for CS computation
period_tgt_list = [0.100000000000000, 0.117210229753348, 0.137382379588326,
                   0.161026202756094, 0.188739182213510, 0.221221629107045,
                   0.259294379740467, 0.303919538231320, 0.356224789026244,
                   0.417531893656040, 0.489390091847749, 0.573615251044868,
                   0.672335753649934, 0.788046281566991, 0.923670857187386,
                   1.08263673387405,  1.26896100316792,  1.48735210729351,
                   1.74332882219999,  2.04335971785694,  2.39502661998749,
                   2.80721620394118,  3.29034456231267,  3.85662042116347,
                   4.52035365636024,  5.29831690628371,  6.21016941891562,
                   7.27895384398315,  8.53167852417281,  10.0]

# Define the list of conditional IM, these are the IMs conditioned on the
# conditioning IM im_star
im_cond_list_cse = [mhu.IntensityMeasure(iml=None,
                    imt=mhu.IntensityMeasureType(
                        'Sa', sa_period=np.array([x])),
                    GMM_name=GMPE_to_use)
                    for x in period_tgt_list]

# IM_cond DURATION -- append duration IM

# duration_GMPE_to_use = AS_2016_duration'
duration_GMPE_to_use = 'BahrampouriEtAlSSlab2020_duration'
im_cond_list_cse.append(mhu.IntensityMeasure(iml=None,
                        imt=mhu.IntensityMeasureType('da5_75'),
                        GMM_name=duration_GMPE_to_use))

# -----------------------------------------------------------------------------
# Compute conditional spectra given a single rupture (i.e., M and R definition)
# -----------------------------------------------------------------------------
cs_mean = mhu.ConditionalSpectra(rup_mean, site_cse,
                                 im_star_cse, im_cond_list_cse,
                                 mhu.sa_corr_baker)

# plot the CS target based on the mean M and R
cs_mean.plot_CS(legend_title=None, is_sample_CS=False, n_sample=25,
                is_plot_cs=True, is_plot_da5_75=True,
                fig_name = "8_story_mrf_seattle_inslab_2in50", 
                is_save_fig = True)

# deepcopy the computed CS target for subsequent plotting
cs_mean_inslab = copy.deepcopy(cs_mean)

# %% SEATTLE Site, for Hiro, 2in50 -- SUBDUCTION INTERFACE

# -----------------------------------------------------------------------------
# GMPE INFORMATION
# -----------------------------------------------------------------------------

# Sa GMPE: AG_2020 (Abrahamson & Gulerce 2020 GMPE for subduction zone events)
# Da5-75 GMPE: bahrampouri_et_al_2021 (duration GMPE for subduction zone events); compare to AS_2016
# Sa correlations: sa_corr_baker (Baker & Jayaram, 2008)
# Sa-Da correlations: Bradley (2011)

# -----------------------------------------------------------------------------
# Deaggregation file data :: 2in50
# -----------------------------------------------------------------------------

#deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_191_saAvg_interface_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/Mag_Dist-mean-0_191_for_plotting.csv'

haz_data_root = join('.','hazard_data', '8-story_mrf')
deagg_file = join(haz_data_root, 'saAvg_interface_trt_8_story_mrf_2in50', 'Mag_Dist-mean_for_plot.csv')


prob_col_name = 'mean'  # column name containing P(X>x | M, R)
delta_M = 0.2
delta_R = 5
investigation_time = 1
# title of the plot
deagg_name = 'Seattle, SaAvg(0.3 : 0.1: 5.0), RP = 2475 years (2% in 50 years), subduction interface'
# weight to apply to deaggregation values (enter manually)
deagg_weight_use = 0.208168317

mhu.deagg_plot(deagg_file, prob_col_name, delta_M, delta_R, investigation_time,
               deagg_name, plot_lim_distance=200, deagg_weight=deagg_weight_use)

df_deagg = mhu.deagg_parse(deagg_file, prob_col_name, delta_M, delta_R,
                           investigation_time)

# keep rows where df_deagg['P(m | X>x)'] is not zero, i.e., bins which contribute
df_deagg = df_deagg.loc[df_deagg['P(m | X>x)'] != 0]

# store data to rup_dict, this is used in subsequent computations of CS targets
rup_dict = {
    'M': df_deagg['mag'].to_numpy(),
    'R': df_deagg['dist'].to_numpy(),
    'p_jd': df_deagg['P(m | X>x)'].to_numpy(),
    'M_mean': np.sum(df_deagg['P(m | X>x)'].to_numpy() * df_deagg['mag'].to_numpy()),
    'R_mean': np.sum(df_deagg['P(m | X>x)'].to_numpy() * df_deagg['dist'].to_numpy())
}

# -----------------------------------------------------------------------------
# Estimate the source and distance parameters using Kaklamanos et al. 2011
# -----------------------------------------------------------------------------

"""
Faulting style for the subduction zone taken as 'reverse'. This is not needed 
in the AG_2020 GMPE nor the bahrampouri_et_al_2021 duration GMPE. But it is 
used to estimate the unknown source and distance parameters. Additionally, this
would be needed for the AS_2016 duration GMPE (this one is the active shallow
crust events).

"""
src_dist_dict = get_src_dist_params(M=rup_dict['M_mean'],
                                    R_jb=rup_dict['R_mean'],
                                    fault_style='reverse',
                                    F_hw=False)

# -----------------------------------------------------------------------------
# Specify additional site parameters
# -----------------------------------------------------------------------------

vs30_seattle = 260  # assuming soil class D
# basin depth terms, USGS NSHM2018 consolidated values for the site
z_2p5_seattle = 6.374 # units: km
z_1p0_seattle = 0.876 # units: km
# global region in the CB_2014 GMPE for active shallow crust
region_seattle_cb_2014 = 0
region_seattle_AG_2020 = "CAS"  # cascadia subduction zone region

z_1p0_seattle_as_2016 = z_1p0_seattle  # AS_2016 needs this value in km
z_1p0_seattle_bahrampouri_2021 = z_1p0_seattle*1000 # bahrampouri_et_al_2021 needs this value in (m)

# -----------------------------------------------------------------------------
# Instantiate site and rupture objects
# -----------------------------------------------------------------------------

site_cse = mhu.Site(Vs30=vs30_seattle,
                    region=region_seattle_AG_2020,
                    Z25=z_2p5_seattle,  
                    Z10=z_1p0_seattle_bahrampouri_2021, 
                    backarc = False)  

rup_mean = mhu.Rup(M=src_dist_dict[0]['M'],
                   R=None,  # not needed
                   Rjb=src_dist_dict[0]['R_jb'],
                   Rrup=src_dist_dict[2]['R_rup'],
                   Rx=src_dist_dict[2]['R_x'],
                   Fhw=0,
                   W=src_dist_dict[1]['W'],
                   delta=src_dist_dict[1]['delta'],
                   lam=src_dist_dict[1]['lambda'],  # 1000,
                   hypo_depth=src_dist_dict[1]['Z_hyp'],  # 5,
                   Ztor=src_dist_dict[1]['Z_tor'],
                   trt='subduction_interface')  # Not used in the CB_2014 GMPE

# -----------------------------------------------------------------------------
# Define IM_star and the list of IM_cond for CS computations
# -----------------------------------------------------------------------------

# IM_star & IM_cond -- Sa, SaAvg
GMPE_to_use ='AbrahamsonEtAl2015SInter' # check the effect of using a different GMM

iml_im_star = 0.488  # conditioning value of the IM, RotD100

# define the imt for the conditioning IM -- 8-story MRF
imt_star_cse = mhu.IntensityMeasureType('SaAvg', sa_period=np.array([0.3,
                                                                     0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
                                                                     1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
                                                                     2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9,
                                                                     4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0]))


# define the im object (intensity measure) for IM_star -- specified above
im_star_cse = mhu.IntensityMeasure(iml=iml_im_star, imt=imt_star_cse,
                                   GMM_name=GMPE_to_use)

# Target periods used for CS target computation
period_tgt_list = [0.100000000000000, 0.117210229753348, 0.137382379588326,
                   0.161026202756094, 0.188739182213510, 0.221221629107045,
                   0.259294379740467, 0.303919538231320, 0.356224789026244,
                   0.417531893656040, 0.489390091847749, 0.573615251044868,
                   0.672335753649934, 0.788046281566991, 0.923670857187386,
                   1.08263673387405,  1.26896100316792,  1.48735210729351,
                   1.74332882219999,  2.04335971785694,  2.39502661998749,
                   2.80721620394118,  3.29034456231267,  3.85662042116347,
                   4.52035365636024,  5.29831690628371,  6.21016941891562,
                   7.27895384398315,  8.53167852417281,  10.0]


# Define the list of conditional IM, these are the IMs conditioned on the
# conditioning IM im_star
im_cond_list_cse = [mhu.IntensityMeasure(iml=None,
                    imt=mhu.IntensityMeasureType(
                        'Sa', sa_period=np.array([x])),
                    GMM_name=GMPE_to_use)
                    for x in period_tgt_list]

# IM_cond DURATION -- append duration IM
duration_GMPE_to_use = 'BahrampouriEtAlSInter2020_duration'
im_cond_list_cse.append(mhu.IntensityMeasure(iml=None,
                        imt=mhu.IntensityMeasureType('da5_75'),
                        GMM_name=duration_GMPE_to_use))

# -----------------------------------------------------------------------------
# Compute conditional spectra given a single rupture (i.e., M and R definition)
# -----------------------------------------------------------------------------
cs_mean = mhu.ConditionalSpectra(rup_mean, site_cse,
                                 im_star_cse, im_cond_list_cse,
                                 mhu.sa_corr_baker)

# plot the CS target based on the mean M and R
cs_mean.plot_CS(legend_title=None, is_sample_CS=False, n_sample=25,
                is_plot_cs=True, is_plot_da5_75=True,
                fig_name = "8_story_mrf_seattle_interface_2in50", 
                is_save_fig = True)

# deepcopy the computed CS target for subsequent plotting
cs_mean_interface = copy.deepcopy(cs_mean)

# %% SEATTLE Site, for Hiro, 2in50 -- comparison of CS targets for different TRTs

color_crustal = 'k'
linestyle_crustal = '--'

color_inslab = 'goldenrod'
linestyle_inslab = ':'

color_interface = 'darkgreen'
linestyle_interface = (0, (6, 2, 3, 2))

sa_median_active = cs_mean_active.CS_target['mu_cond'][:-1]
sa_median_inslab = cs_mean_inslab.CS_target['mu_cond'][:-1]
sa_median_interface = cs_mean_interface.CS_target['mu_cond'][:-1]
periods_cs = cs_mean_active.CS_target['period_arr_cs']


plt.figure(dpi=200)
plt.plot(periods_cs, sa_median_active, linewidth=2.5,
         linestyle=linestyle_crustal, 
         color=color_crustal,
         label='Crustal (76.5%)')

plt.plot(periods_cs, sa_median_inslab, linewidth=2.5,
         linestyle=linestyle_inslab, 
         color=color_inslab,
         label='In-slab (2.7%)')

plt.plot(periods_cs, sa_median_interface, linewidth=2.5,
         linestyle=linestyle_interface,
         color=color_interface,
         label='Interface (20.8%)')

plt.xlabel('Period (s)')
plt.ylabel('Sa(T) (g)')
plt.xlim(0.1, 10)
plt.ylim(0.01, 5)
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig('8_story_mrf_seattle_cs_2in50.png', bbox_inches='tight')
plt.show()


da_median_active = cs_mean_active.CS_target['mu_cond'][-1]
da_median_inslab = cs_mean_inslab.CS_target['mu_cond'][-1]
da_median_interface = cs_mean_interface.CS_target['mu_cond'][-1]

da_sigma_active = cs_mean_active.CS_target['ln_std_cond'][-1]
da_sigma_inslab = cs_mean_inslab.CS_target['ln_std_cond'][-1]
da_sigma_interface = cs_mean_interface.CS_target['ln_std_cond'][-1]

# compute the lognormal distributions
x_plt = np.geomspace(1, 500, 500)

# active
median_cond_curr = da_median_active
sigma_cond_curr = da_sigma_active
dist = scipy.stats.norm(loc=np.log(median_cond_curr),
                        scale=sigma_cond_curr)
y_plt = np.array([dist.cdf(np.log(x)) for x in x_plt])
y_plt_active = y_plt

# inslab
median_cond_curr = da_median_inslab
sigma_cond_curr = da_sigma_inslab
dist = scipy.stats.norm(loc=np.log(median_cond_curr),
                        scale=sigma_cond_curr)
y_plt = np.array([dist.cdf(np.log(x)) for x in x_plt])
y_plt_inslab = y_plt

# interface
median_cond_curr = da_median_interface
sigma_cond_curr = da_sigma_interface
dist = scipy.stats.norm(loc=np.log(median_cond_curr),
                        scale=sigma_cond_curr)
y_plt = np.array([dist.cdf(np.log(x)) for x in x_plt])
y_plt_interface = y_plt


plt.figure(dpi=200)
plt.plot(x_plt, y_plt_active,
         linestyle=linestyle_crustal, 
         color=color_crustal,
         label='Crustal', 
         linewidth=2.5)

plt.plot(x_plt, y_plt_inslab,
         linestyle=linestyle_inslab, 
         color=color_inslab,
         label='In-slab', 
         linewidth=2.5)

plt.plot(x_plt, y_plt_interface,
         linestyle=linestyle_interface,
         color=color_interface,
         label='Interface', 
         linewidth=2.5)

# plt.legend(loc = 'best')
plt.grid()
plt.xlabel('Da 5-75% (s)')
plt.ylabel('CDF')
plt.ylim(0, 1)
plt.xlim(1, 500)
plt.xscale('log')
plt.savefig('8_story_mrf_seattle_da5-75_2in50.png', bbox_inches='tight')
plt.show()


# %% SEATTLE Site, for Hiro, 2in50 - save the CS targets

cs_save_dir = join('.','CS_target', 'seattle_8_story_mrf_2in50_rotD100')

folder_save = join(cs_save_dir, 'active_shallow_crust')
cs_mean_active.export_CS_target(folder_save)

folder_save = join(cs_save_dir,'interface')
cs_mean_interface.export_CS_target(folder_save)

folder_save = join(cs_save_dir,'inslab')
cs_mean_inslab.export_CS_target(folder_save)