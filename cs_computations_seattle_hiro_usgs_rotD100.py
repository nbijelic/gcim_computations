#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 12:55:32 2024

@author: nbijelic

Scrip to compute CS targets used for selection of ground motions. Core 
functions are in the main_hazard_utils.py; additional GMPEs implemented in 
separate files.

Previous examples of usage stored in 'main_hazard_utils_backup.py'

"""
import seaborn as sns
import plotly.io as pio
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import scipy
import copy
import main_hazard_utils_FIXED as mhu
from estimating_intput_gmpe_params_kaklamanos_et_al_2011_FIXED import get_src_dist_params

# plotting
import matplotlib.pyplot as plt
# change font globally
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams.update({'font.size': 14})
# plotly
pio.renderers.default = "browser"

#%%

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#
# 4-story MRF
#
# -----------------------------------------------------------------------------
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

deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/4-story_mrf/calc_id_200_saAvg_crustal_trt_4_story_mrf_2in50/Mag_Dist-mean-0_200_for_plot.csv'

prob_col_name = 'mean'  # column name containing P(X>x | M, R); here I assume
# only a single realization is used
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
#
# -----------------------------------------------------------------------------
# Enter other parameters and estimate the unknown source and distance params
# -----------------------------------------------------------------------------

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
#
# -----------------------------------------------------------------------------
# Specify additional site parameters
# -----------------------------------------------------------------------------

vs30_seattle = 260  # assuming soil class D
# in (km), USGS NSHM2018 consolidated values for the site
z_2p5_seattle = 6.374
z_1p0_seattle = 0.876  # in (km) -- AS_2016 uses the value in km
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
                   # value is negative, there is no difference here when positive value is used
                   Rx=src_dist_dict[2]['R_x'],
                   Fhw=0,
                   # Zbot = None, # for CB_2014 needed only when W is unknown, so I leave this undefined (code checks if attribute exists)
                   W=src_dist_dict[1]['W'],
                   delta=src_dist_dict[1]['delta'],
                   lam=src_dist_dict[1]['lambda'],  # 1000,
                   hypo_depth=src_dist_dict[1]['Z_hyp'],  # 5,
                   Ztor=src_dist_dict[1]['Z_tor'],
                   trt=None)  # Not used in the CB_2014 GMPE

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


# # define the imt for the conditioning IM -- 8-story MRF
# imt_star_cse = mhu.IntensityMeasureType( 'SaAvg', sa_period = np.array([1.0]) )


# define the im object (intensity measure) for IM_star -- specified above
im_star_cse = mhu.IntensityMeasure(iml=iml_im_star, imt=imt_star_cse,
                                   GMM_name=GMPE_to_use)

# Target periods used in the ground motion selections script by Prof. Baker;
# I manually copied the 30 periods computed in:
# selectionParams.TgtPer = logspace(log10(selectionParams.Tmin),log10(selectionParams.Tmax),30)
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
# shorter list so that my computation is faster in develpment
# period_tgt_list = [0.1, 0.5, 1, 3, 5, 10]


# Define the list of conditional IM, these are the IMs conditioned on the
# conditioning IM im_star
im_cond_list_cse = [mhu.IntensityMeasure(iml=None,
                    imt=mhu.IntensityMeasureType(
                        'Sa', sa_period=np.array([x])),
                    GMM_name=GMPE_to_use)
                    for x in period_tgt_list]

# IM_cond DURATION -- append duration IM
duration_GMPE_to_use = 'AS_2016_duration'  # ''
im_cond_list_cse.append(mhu.IntensityMeasure(iml=None,
                                             imt=mhu.IntensityMeasureType(
                                                 'da5_75'),
                                             GMM_name=duration_GMPE_to_use))

# -----------------------------------------------------------------------------
# Compute conditional spectra given a single rupture (i.e., M and R definition)
# -----------------------------------------------------------------------------
cs_mean = mhu.ConditionalSpectra(rup_mean, site_cse,
                                 im_star_cse, im_cond_list_cse,
                                 mhu.sa_corr_baker)

# plot the CS target based on the mean M and R
cs_mean.plot_CS(legend_title=None, is_sample_CS=False, n_sample=25,
                is_plot_cs=True, is_plot_da5_75=True)

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

deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/4-story_mrf/calc_id_202_saAvg_inslab_trt_4_story_mrf_2in50/Mag_Dist-mean-0_202_for_plotting.csv'

prob_col_name = 'mean'  # column name containing P(X>x | M, R); here I assume
# only a single realization is used
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
# 
# -----------------------------------------------------------------------------
# Enter other parameters and estimate the unknown source and distance params
# -----------------------------------------------------------------------------

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
                                    fault_style='normal',  # 'reverse', #'normal',#'strike_slip', #
                                    F_hw=True)
#
# 
# -----------------------------------------------------------------------------
# Specify additional site parameters
# -----------------------------------------------------------------------------

vs30_seattle = 260  # assuming soil class D
# in (km), USGS NSHM2018 consolidated values for the site
z_2p5_seattle = 6.374
z_1p0_seattle = 0.876  # in (km) -- AS_2016 uses the value in km
# global region in the CB_2014 GMPE for active shallow crust
region_seattle_cb_2014 = 0
region_seattle_AG_2020 = "CAS"  # cascadia subduction zone region

z_1p0_seattle_as_2016 = z_1p0_seattle  # AS_2016 needs this value in km
# bahrampouri_et_al_2021 needs this value in (m)
z_1p0_seattle_bahrampouri_2021 = z_1p0_seattle*1000

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
                   # Zbot = None, # for CB_2014 needed only when W is unknown, so I leave this undefined (code checks if attribute exists)
                   W=src_dist_dict[1]['W'],
                   delta=src_dist_dict[1]['delta'],
                   lam=src_dist_dict[1]['lambda'],
                   hypo_depth=np.array([53.17]),
                   Ztor=np.array([46.13]),  
                   trt='subduction_inslab')

# rup_mean = mhu.Rup(M=src_dist_dict[0]['M'],
#                    R=None,  # not needed
#                    Rjb=src_dist_dict[0]['R_jb'],
#                    Rrup=src_dist_dict[2]['R_rup'],
#                    # value is negative, there is no difference here when positive value is used
#                    Rx=src_dist_dict[2]['R_x'],
#                    Fhw=0,
#                    # Zbot = None, # for CB_2014 needed only when W is unknown, so I leave this undefined (code checks if attribute exists)
#                    W=src_dist_dict[1]['W'],
#                    delta=src_dist_dict[1]['delta'],
#                    lam=src_dist_dict[1]['lambda'],  # 1000,
#                    hypo_depth=src_dist_dict[1]['Z_hyp'],  # 5,
#                    Ztor=src_dist_dict[1]['Z_tor'],
#                    trt='subduction_inslab')  # Not used in the CB_2014 GMPE

# rup_mean = mhu.Rup(M=np.array([7.3]),
#                    R=None,  # not needed
#                    Rjb=np.array([62.3]),
#                    Rrup=np.array([86.58347057]), 
#                    Rx=np.array([56.13215502]),
#                    Fhw=1,
#                    W=np.array([26.01]),
#                    delta=np.array([50.]),
#                    lam=np.array([-90.]),
#                    hypo_depth=np.array([53.17]),
#                    Ztor=np.array([46.13]),  
#                    trt='subduction_inslab')

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

# # define the imt for the conditioning IM -- 8-story MRF
# imt_star_cse = mhu.IntensityMeasureType( 'SaAvg', sa_period = np.array([1.0]) )


# define the im object (intensity measure) for IM_star -- specified above
im_star_cse = mhu.IntensityMeasure(iml=iml_im_star, imt=imt_star_cse,
                                   GMM_name=GMPE_to_use)

# Target periods used in the ground motion selections script by Prof. Baker;
# I manually copied the 30 periods computed in:
# selectionParams.TgtPer = logspace(log10(selectionParams.Tmin),log10(selectionParams.Tmax),30)
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
# shorter list so that my computation is faster in develpment
# period_tgt_list = [0.1, 0.5, 1, 3, 5, 10]


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
                                             imt=mhu.IntensityMeasureType(
                                                 'da5_75'),
                                             GMM_name=duration_GMPE_to_use))

# -----------------------------------------------------------------------------
# Compute conditional spectra given a single rupture (i.e., M and R definition)
# -----------------------------------------------------------------------------
cs_mean = mhu.ConditionalSpectra(rup_mean, site_cse,
                                 im_star_cse, im_cond_list_cse,
                                 mhu.sa_corr_baker)

# plot the CS target based on the mean M and R
cs_mean.plot_CS(legend_title=None, is_sample_CS=False, n_sample=25,
                is_plot_cs=True, is_plot_da5_75=True)

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

deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/4-story_mrf/calc_id_201_saAvg_interface_trt_4_story_mrf_2in50/Mag_Dist-mean-0_201_for_plotting.csv'

prob_col_name = 'mean'  # column name containing P(X>x | M, R); here I assume
# only a single realization is used
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
#
# -----------------------------------------------------------------------------
# Enter other parameters and estimate the unknown source and distance params
# -----------------------------------------------------------------------------

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


# Note: I selected the case for the site on the foot wall because the distances
#   are smaller in that case compared to assuming the site on the hanging wall.
#   In this case the Sa values will be larger, but the durations will be
#   smaller.

#
# -----------------------------------------------------------------------------
# Specify additional site parameters
# -----------------------------------------------------------------------------

vs30_seattle = 260  # assuming soil class D
# in (km), USGS NSHM2018 consolidated values for the site
z_2p5_seattle = 6.374
z_1p0_seattle = 0.876  # in (km) -- AS_2016 uses the value in km
# global region in the CB_2014 GMPE for active shallow crust
region_seattle_cb_2014 = 0
region_seattle_AG_2020 = "CAS"  # cascadia subduction zone region

z_1p0_seattle_as_2016 = z_1p0_seattle  # AS_2016 needs this value in km
# bahrampouri_et_al_2021 needs this value in (m)
z_1p0_seattle_bahrampouri_2021 = z_1p0_seattle*1000

# -----------------------------------------------------------------------------
# Instantiate site and rupture objects
# -----------------------------------------------------------------------------

site_cse = mhu.Site(Vs30=vs30_seattle,
                    region=region_seattle_AG_2020,
                    Z25=z_2p5_seattle,  # AG_2020 needs this value in km
                    # Z10 = z_1p0_seattle_as_2016)
                    Z10=z_1p0_seattle_bahrampouri_2021, # AS_2016 needs this value in km;
                    backarc = False)  
# bahrampouri_et_al_2021 needs this value in (m)

rup_mean = mhu.Rup(M=src_dist_dict[0]['M'],
                   R=None,  # not needed
                   Rjb=src_dist_dict[0]['R_jb'],
                   Rrup=src_dist_dict[2]['R_rup'],
                   # value is negative, there is no difference here when positive value is used
                   Rx=src_dist_dict[2]['R_x'],
                   Fhw=0,
                   # Zbot = None, # for CB_2014 needed only when W is unknown, so I leave this undefined (code checks if attribute exists)
                   W=src_dist_dict[1]['W'],
                   delta=src_dist_dict[1]['delta'],
                   lam=src_dist_dict[1]['lambda'],  # 1000,
                   hypo_depth=src_dist_dict[1]['Z_hyp'],  # 5,
                   Ztor=src_dist_dict[1]['Z_tor'],
                   trt='subduction_interface')  # Not used in the CB_2014 GMPE


# -----------------------------------------------------------------------------
# Define IM_star and the list of IM_cond for CS computations
# -----------------------------------------------------------------------------


# NOTE: the function always uses the below booleans by default, currently this
#       cannot be changed
# is_ergodic = False
# is_apply_usa_adjustment = True


# IM_star & IM_cond -- Sa, SaAvg
# GMPE_to_use = 'AG_2020'
# GMPE_to_use = 'CB_2014_active' # check the effect of using a different GMM
GMPE_to_use ='AbrahamsonEtAl2015SInter' # check the effect of using a different GMM


iml_im_star = 0.766  # conditioning value of the IM, RotD100


# define the imt for the conditioning IM -- 4-story MRF
imt_star_cse = mhu.IntensityMeasureType('SaAvg', sa_period=np.array([0.2, 0.3,
                    0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
                    1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
                    2.8, 2.9, 3.0]))

# # define the imt for the conditioning IM -- 8-story MRF
# imt_star_cse = mhu.IntensityMeasureType( 'SaAvg', sa_period = np.array([1.0]) )


# define the im object (intensity measure) for IM_star -- specified above
im_star_cse = mhu.IntensityMeasure(iml=iml_im_star, imt=imt_star_cse,
                                   GMM_name=GMPE_to_use)

# Target periods used in the ground motion selections script by Prof. Baker;
# I manually copied the 30 periods computed in:
# selectionParams.TgtPer = logspace(log10(selectionParams.Tmin),log10(selectionParams.Tmax),30)
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
# shorter list so that my computation is faster in develpment
# period_tgt_list = [0.1, 0.5, 1, 3, 5, 10]


# Define the list of conditional IM, these are the IMs conditioned on the
# conditioning IM im_star
im_cond_list_cse = [mhu.IntensityMeasure(iml=None,
                    imt=mhu.IntensityMeasureType(
                        'Sa', sa_period=np.array([x])),
                    GMM_name=GMPE_to_use)
                    for x in period_tgt_list]

# IM_cond DURATION -- append duration IM

# duration_GMPE_to_use = 'AS_2016_duration'
duration_GMPE_to_use = 'BahrampouriEtAlSInter2020_duration'
im_cond_list_cse.append(mhu.IntensityMeasure(iml=None,
                                             imt=mhu.IntensityMeasureType(
                                                 'da5_75'),
                                             GMM_name=duration_GMPE_to_use))

# -----------------------------------------------------------------------------
# Compute conditional spectra given a single rupture (i.e., M and R definition)
# -----------------------------------------------------------------------------
cs_mean = mhu.ConditionalSpectra(rup_mean, site_cse,
                                 im_star_cse, im_cond_list_cse,
                                 mhu.sa_corr_baker)

# plot the CS target based on the mean M and R
cs_mean.plot_CS(legend_title=None, is_sample_CS=False, n_sample=25,
                is_plot_cs=True, is_plot_da5_75=True)

# deepcopy the computed CS target for subsequent plotting
cs_mean_interface = copy.deepcopy(cs_mean)


# %% SEATTLE Site, for Hiro, 2in50 -- comparison of CS targets for different TRTs

sa_median_active = cs_mean_active.CS_target['mu_cond'][:-1]
sa_median_inslab = cs_mean_inslab.CS_target['mu_cond'][:-1]
sa_median_interface = cs_mean_interface.CS_target['mu_cond'][:-1]
periods_cs = cs_mean_active.CS_target['period_arr_cs']


plt.figure(dpi=200)
plt.plot(periods_cs, sa_median_active, '-k', linewidth=1.5,
         label='Crustal (73.5%)')

plt.plot(periods_cs, sa_median_inslab, '-.k', linewidth=1.5,
         label='In-slab (6.5%)')

plt.plot(periods_cs, sa_median_interface, '--k', linewidth=1.5,
         label='Interface (20.0%)')

plt.xlabel('Period (s)')
plt.ylabel('Sa(T) (g)')
plt.xlim(0.1, 10)
plt.ylim(0.01, 5)
plt.xscale('log')
plt.yscale('log')
plt.legend()
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
plt.plot(x_plt, y_plt_active, '-k', label='Crustal', linewidth=1.5)
plt.plot(x_plt, y_plt_inslab, '-.k', label='In-slab', linewidth=1.5)
plt.plot(x_plt, y_plt_interface, '--k', label='Interface', linewidth=1.5)

# plt.legend(loc = 'best')
plt.grid()
plt.xlabel('Da 5-75% (s)')
plt.ylabel('CDF')
plt.ylim(0, 1)
plt.xlim(1, 500)
plt.xscale('log')
plt.show()


#%% compute SaAvg from the mean spectra -- should correspond to the value of 
# the conditioning IM

from scipy.stats import gmean

# compute SaAvg from mean spectra, Active Crustal TRT
x_interp = np.log(periods_cs)
y_interp = np.log(sa_median_active)
x_hat = np.log(np.linspace(0.2, 3.0, 29))
y_hat = np.interp(x = x_hat, xp = x_interp, fp = y_interp)

sa_avg_active = gmean(np.exp(y_hat))

# compute SaAvg from mean spectra, Inslab TRT
x_interp = np.log(periods_cs)
y_interp = np.log(sa_median_inslab)
x_hat = np.log(np.linspace(0.2, 3.0, 29))
y_hat = np.interp(x = x_hat, xp = x_interp, fp = y_interp)

sa_avg_inslab = gmean(np.exp(y_hat))

# compute SaAvg from mean spectra, Interface TRT
x_interp = np.log(periods_cs)
y_interp = np.log(sa_median_interface)
x_hat = np.log(np.linspace(0.2, 3.0, 29))
y_hat = np.interp(x = x_hat, xp = x_interp, fp = y_interp)

sa_avg_interface = gmean(np.exp(y_hat))


# %% SEATTLE Site, for Hiro, 2in50 - save the CS targets

folder_save = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/CS_targets/seattle_4_story_mrf_2in50_rotD100/active_shallow_crust/'
cs_mean_active.export_CS_target(folder_save)

folder_save = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/CS_targets/seattle_4_story_mrf_2in50_rotD100/interface/'
cs_mean_interface.export_CS_target(folder_save)

folder_save = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/CS_targets/seattle_4_story_mrf_2in50_rotD100/inslab/'
cs_mean_inslab.export_CS_target(folder_save)











#%%

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#
# 8-story MRF
#
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

#%%

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#
# 8-story MRF :: 10in50 intensity
#
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# %% SEATTLE Site, for Hiro, 10in50 -- ACTIVE SHALLOW CRUST

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

deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_206_saAvg_crustal_trt_8_story_mrf_10in50/Mag_Dist-mean-0_206_for_plotting.csv'

prob_col_name = 'mean'  # column name containing P(X>x | M, R); here I assume
# only a single realization is used
delta_M = 0.2
delta_R = 5
investigation_time = 1
# title of the plot
deagg_name = 'Seattle, SaAvg(0.3 : 0.1: 5.0), RP = 475 years (10% in 50 years), active shallow crust'
# weight to apply to deaggregation values (enter manually)
deagg_weight_use = 0.542

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
#
# -----------------------------------------------------------------------------
# Enter other parameters and estimate the unknown source and distance params
# -----------------------------------------------------------------------------

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
#
# -----------------------------------------------------------------------------
# Specify additional site parameters
# -----------------------------------------------------------------------------

vs30_seattle = 260  # assuming soil class D
# in (km), USGS NSHM2018 consolidated values for the site
z_2p5_seattle = 6.374
z_1p0_seattle = 0.876  # in (km) -- AS_2016 uses the value in km
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
                   # value is negative, there is no difference here when positive value is used
                   Rx=src_dist_dict[2]['R_x'],
                   Fhw=0,
                   # Zbot = None, # for CB_2014 needed only when W is unknown, so I leave this undefined (code checks if attribute exists)
                   W=src_dist_dict[1]['W'],
                   delta=src_dist_dict[1]['delta'],
                   lam=src_dist_dict[1]['lambda'],  # 1000,
                   hypo_depth=src_dist_dict[1]['Z_hyp'],  # 5,
                   Ztor=src_dist_dict[1]['Z_tor'],
                   trt=None)  # Not used in the CB_2014 GMPE

# -----------------------------------------------------------------------------
# Define IM_star and the list of IM_cond for CS computations
# -----------------------------------------------------------------------------

# IM_star & IM_cond -- Sa, SaAvg
GMPE_to_use = 'CB_2014_active'
# iml_im_star = 0.400 # conditioning value of the IM, RotD50
# iml_im_star = 0.488  # conditioning value of the IM, RotD100
iml_im_star = 0.200  # conditioning value of the IM, RotD100

# define the imt for the conditioning IM -- 8-story MRF
imt_star_cse = mhu.IntensityMeasureType('SaAvg', sa_period=np.array([0.3,
                                                                     0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
                                                                     1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
                                                                     2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9,
                                                                     4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0]))


# # define the imt for the conditioning IM -- 8-story MRF
# imt_star_cse = mhu.IntensityMeasureType( 'SaAvg', sa_period = np.array([1.0]) )


# define the im object (intensity measure) for IM_star -- specified above
im_star_cse = mhu.IntensityMeasure(iml=iml_im_star, imt=imt_star_cse,
                                   GMM_name=GMPE_to_use)

# Target periods used in the ground motion selections script by Prof. Baker;
# I manually copied the 30 periods computed in:
# selectionParams.TgtPer = logspace(log10(selectionParams.Tmin),log10(selectionParams.Tmax),30)
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
# shorter list so that my computation is faster in develpment
# period_tgt_list = [0.1, 0.5, 1, 3, 5, 10]


# Define the list of conditional IM, these are the IMs conditioned on the
# conditioning IM im_star
im_cond_list_cse = [mhu.IntensityMeasure(iml=None,
                    imt=mhu.IntensityMeasureType(
                        'Sa', sa_period=np.array([x])),
                    GMM_name=GMPE_to_use)
                    for x in period_tgt_list]

# IM_cond DURATION -- append duration IM
duration_GMPE_to_use = 'AS_2016_duration'  # ''
im_cond_list_cse.append(mhu.IntensityMeasure(iml=None,
                                             imt=mhu.IntensityMeasureType(
                                                 'da5_75'),
                                             GMM_name=duration_GMPE_to_use))

# -----------------------------------------------------------------------------
# Compute conditional spectra given a single rupture (i.e., M and R definition)
# -----------------------------------------------------------------------------
cs_mean = mhu.ConditionalSpectra(rup_mean, site_cse,
                                 im_star_cse, im_cond_list_cse,
                                 mhu.sa_corr_baker)

# plot the CS target based on the mean M and R
cs_mean.plot_CS(legend_title=None, is_sample_CS=False, n_sample=25,
                is_plot_cs=True, is_plot_da5_75=True)

# deepcopy the computed CS target for subsequent plotting
cs_mean_active = copy.deepcopy(cs_mean)


# %% SEATTLE Site, for Hiro, 10in50 -- SUBDUCTION IN-SLAB

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

deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_208_saAvg_inslab_trt_8_story_mrf_10in50/Mag_Dist-mean-0_208_for_plotting.csv'

prob_col_name = 'mean'  # column name containing P(X>x | M, R); here I assume
# only a single realization is used
delta_M = 0.2
delta_R = 5
investigation_time = 1
# title of the plot
deagg_name = 'Seattle, SaAvg(0.3 : 0.1: 5.0), RP = 475 years (10% in 50 years), subduction in-slab'
# weight to apply to deaggregation values (enter manually)
deagg_weight_use = 0.131

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
# 
# -----------------------------------------------------------------------------
# Enter other parameters and estimate the unknown source and distance params
# -----------------------------------------------------------------------------

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
                                    fault_style='normal',  # 'reverse', #'normal',#'strike_slip', #
                                    F_hw=True)
#
# 
# -----------------------------------------------------------------------------
# Specify additional site parameters
# -----------------------------------------------------------------------------

vs30_seattle = 260  # assuming soil class D
# in (km), USGS NSHM2018 consolidated values for the site
z_2p5_seattle = 6.374
z_1p0_seattle = 0.876  # in (km) -- AS_2016 uses the value in km
# global region in the CB_2014 GMPE for active shallow crust
region_seattle_cb_2014 = 0
region_seattle_AG_2020 = "CAS"  # cascadia subduction zone region

z_1p0_seattle_as_2016 = z_1p0_seattle  # AS_2016 needs this value in km
# bahrampouri_et_al_2021 needs this value in (m)
z_1p0_seattle_bahrampouri_2021 = z_1p0_seattle*1000

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
                   # Zbot = None, # for CB_2014 needed only when W is unknown, so I leave this undefined (code checks if attribute exists)
                   W=src_dist_dict[1]['W'],
                   delta=src_dist_dict[1]['delta'],
                   lam=src_dist_dict[1]['lambda'],
                   hypo_depth=np.array([53.17]),
                   Ztor=np.array([46.13]),  
                   trt='subduction_inslab')

# rup_mean = mhu.Rup(M=src_dist_dict[0]['M'],
#                    R=None,  # not needed
#                    Rjb=src_dist_dict[0]['R_jb'],
#                    Rrup=src_dist_dict[2]['R_rup'],
#                    # value is negative, there is no difference here when positive value is used
#                    Rx=src_dist_dict[2]['R_x'],
#                    Fhw=0,
#                    # Zbot = None, # for CB_2014 needed only when W is unknown, so I leave this undefined (code checks if attribute exists)
#                    W=src_dist_dict[1]['W'],
#                    delta=src_dist_dict[1]['delta'],
#                    lam=src_dist_dict[1]['lambda'],  # 1000,
#                    hypo_depth=src_dist_dict[1]['Z_hyp'],  # 5,
#                    Ztor=src_dist_dict[1]['Z_tor'],
#                    trt='subduction_inslab')  # Not used in the CB_2014 GMPE

# rup_mean = mhu.Rup(M=np.array([7.3]),
#                    R=None,  # not needed
#                    Rjb=np.array([62.3]),
#                    Rrup=np.array([86.58347057]), 
#                    Rx=np.array([56.13215502]),
#                    Fhw=1,
#                    W=np.array([26.01]),
#                    delta=np.array([50.]),
#                    lam=np.array([-90.]),
#                    hypo_depth=np.array([53.17]),
#                    Ztor=np.array([46.13]),  
#                    trt='subduction_inslab')

# -----------------------------------------------------------------------------
# Define IM_star and the list of IM_cond for CS computations
# -----------------------------------------------------------------------------

# NOTE: the function always uses the below booleans by default, currently this
#       cannot be changed
# is_ergodic = False
# is_apply_usa_adjustment = True


# IM_star & IM_cond -- Sa, SaAvg
GMPE_to_use = 'AG_2020'
# iml_im_star = 0.400 # conditioning value of the IM, RotD50
# iml_im_star = 0.488  # conditioning value of the IM, RotD100
iml_im_star = 0.200  # conditioning value of the IM, RotD100

# define the imt for the conditioning IM -- 8-story MRF
imt_star_cse = mhu.IntensityMeasureType('SaAvg', sa_period=np.array([0.3,
                                                                     0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
                                                                     1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
                                                                     2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9,
                                                                     4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0]))


# # define the imt for the conditioning IM -- 8-story MRF
# imt_star_cse = mhu.IntensityMeasureType( 'SaAvg', sa_period = np.array([1.0]) )


# define the im object (intensity measure) for IM_star -- specified above
im_star_cse = mhu.IntensityMeasure(iml=iml_im_star, imt=imt_star_cse,
                                   GMM_name=GMPE_to_use)

# Target periods used in the ground motion selections script by Prof. Baker;
# I manually copied the 30 periods computed in:
# selectionParams.TgtPer = logspace(log10(selectionParams.Tmin),log10(selectionParams.Tmax),30)
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
# shorter list so that my computation is faster in develpment
# period_tgt_list = [0.1, 0.5, 1, 3, 5, 10]


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
                                             imt=mhu.IntensityMeasureType(
                                                 'da5_75'),
                                             GMM_name=duration_GMPE_to_use))

# -----------------------------------------------------------------------------
# Compute conditional spectra given a single rupture (i.e., M and R definition)
# -----------------------------------------------------------------------------
cs_mean = mhu.ConditionalSpectra(rup_mean, site_cse,
                                 im_star_cse, im_cond_list_cse,
                                 mhu.sa_corr_baker)

# plot the CS target based on the mean M and R
cs_mean.plot_CS(legend_title=None, is_sample_CS=False, n_sample=25,
                is_plot_cs=True, is_plot_da5_75=True)

# deepcopy the computed CS target for subsequent plotting
cs_mean_inslab = copy.deepcopy(cs_mean)


# %% SEATTLE Site, for Hiro, 10in50 -- SUBDUCTION INTERFACE

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

deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_207_saAvg_interface_trt_8_story_mrf_10in50/Mag_Dist-mean-0_207_for_plotting.csv'

prob_col_name = 'mean'  # column name containing P(X>x | M, R); here I assume
# only a single realization is used
delta_M = 0.2
delta_R = 5
investigation_time = 1
# title of the plot
deagg_name = 'Seattle, SaAvg(0.3 : 0.1: 5.0), RP = 475 years (10% in 50 years), subduction interface'
# weight to apply to deaggregation values (enter manually)
deagg_weight_use = 0.327

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
#
# -----------------------------------------------------------------------------
# Enter other parameters and estimate the unknown source and distance params
# -----------------------------------------------------------------------------

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


# Note: I selected the case for the site on the foot wall because the distances
#   are smaller in that case compared to assuming the site on the hanging wall.
#   In this case the Sa values will be larger, but the durations will be
#   smaller.

#
# -----------------------------------------------------------------------------
# Specify additional site parameters
# -----------------------------------------------------------------------------

vs30_seattle = 260  # assuming soil class D
# in (km), USGS NSHM2018 consolidated values for the site
z_2p5_seattle = 6.374
z_1p0_seattle = 0.876  # in (km) -- AS_2016 uses the value in km
# global region in the CB_2014 GMPE for active shallow crust
region_seattle_cb_2014 = 0
region_seattle_AG_2020 = "CAS"  # cascadia subduction zone region

z_1p0_seattle_as_2016 = z_1p0_seattle  # AS_2016 needs this value in km
# bahrampouri_et_al_2021 needs this value in (m)
z_1p0_seattle_bahrampouri_2021 = z_1p0_seattle*1000

# -----------------------------------------------------------------------------
# Instantiate site and rupture objects
# -----------------------------------------------------------------------------

site_cse = mhu.Site(Vs30=vs30_seattle,
                    region=region_seattle_AG_2020,
                    Z25=z_2p5_seattle,  # AG_2020 needs this value in km
                    # Z10 = z_1p0_seattle_as_2016)
                    Z10=z_1p0_seattle_bahrampouri_2021, # AS_2016 needs this value in km;
                    backarc = False)  
# bahrampouri_et_al_2021 needs this value in (m)

rup_mean = mhu.Rup(M=src_dist_dict[0]['M'],
                   R=None,  # not needed
                   Rjb=src_dist_dict[0]['R_jb'],
                   Rrup=src_dist_dict[2]['R_rup'],
                   # value is negative, there is no difference here when positive value is used
                   Rx=src_dist_dict[2]['R_x'],
                   Fhw=0,
                   # Zbot = None, # for CB_2014 needed only when W is unknown, so I leave this undefined (code checks if attribute exists)
                   W=src_dist_dict[1]['W'],
                   delta=src_dist_dict[1]['delta'],
                   lam=src_dist_dict[1]['lambda'],  # 1000,
                   hypo_depth=src_dist_dict[1]['Z_hyp'],  # 5,
                   Ztor=src_dist_dict[1]['Z_tor'],
                   trt='subduction_interface')  # Not used in the CB_2014 GMPE


# -----------------------------------------------------------------------------
# Define IM_star and the list of IM_cond for CS computations
# -----------------------------------------------------------------------------


# NOTE: the function always uses the below booleans by default, currently this
#       cannot be changed
# is_ergodic = False
# is_apply_usa_adjustment = True


# IM_star & IM_cond -- Sa, SaAvg
# GMPE_to_use = 'AG_2020'
# GMPE_to_use = 'CB_2014_active' # check the effect of using a different GMM
GMPE_to_use ='AbrahamsonEtAl2015SInter' # check the effect of using a different GMM

# iml_im_star = 0.400 # conditioning value of the IM, RotD50
# iml_im_star = 0.488  # conditioning value of the IM, RotD100
iml_im_star = 0.200  # conditioning value of the IM, RotD100

# define the imt for the conditioning IM -- 8-story MRF
imt_star_cse = mhu.IntensityMeasureType('SaAvg', sa_period=np.array([0.3,
                                                                     0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
                                                                     1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
                                                                     2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9,
                                                                     4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0]))


# # define the imt for the conditioning IM -- 8-story MRF
# imt_star_cse = mhu.IntensityMeasureType( 'SaAvg', sa_period = np.array([1.0]) )


# define the im object (intensity measure) for IM_star -- specified above
im_star_cse = mhu.IntensityMeasure(iml=iml_im_star, imt=imt_star_cse,
                                   GMM_name=GMPE_to_use)

# Target periods used in the ground motion selections script by Prof. Baker;
# I manually copied the 30 periods computed in:
# selectionParams.TgtPer = logspace(log10(selectionParams.Tmin),log10(selectionParams.Tmax),30)
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
# shorter list so that my computation is faster in develpment
# period_tgt_list = [0.1, 0.5, 1, 3, 5, 10]


# Define the list of conditional IM, these are the IMs conditioned on the
# conditioning IM im_star
im_cond_list_cse = [mhu.IntensityMeasure(iml=None,
                    imt=mhu.IntensityMeasureType(
                        'Sa', sa_period=np.array([x])),
                    GMM_name=GMPE_to_use)
                    for x in period_tgt_list]

# IM_cond DURATION -- append duration IM

# duration_GMPE_to_use = 'AS_2016_duration'
duration_GMPE_to_use = 'BahrampouriEtAlSInter2020_duration'
im_cond_list_cse.append(mhu.IntensityMeasure(iml=None,
                                             imt=mhu.IntensityMeasureType(
                                                 'da5_75'),
                                             GMM_name=duration_GMPE_to_use))

# im_cond_list_cse.append(mhu.IntensityMeasure(iml=None,
#                                              imt=mhu.IntensityMeasureType(
#                                                  'da5_95'),
#                                              GMM_name=duration_GMPE_to_use))

# -----------------------------------------------------------------------------
# Compute conditional spectra given a single rupture (i.e., M and R definition)
# -----------------------------------------------------------------------------
cs_mean = mhu.ConditionalSpectra(rup_mean, site_cse,
                                 im_star_cse, im_cond_list_cse,
                                 mhu.sa_corr_baker)

# plot the CS target based on the mean M and R
cs_mean.plot_CS(legend_title=None, is_sample_CS=False, n_sample=25,
                is_plot_cs=True, is_plot_da5_75=True)

# deepcopy the computed CS target for subsequent plotting
cs_mean_interface = copy.deepcopy(cs_mean)


# %% SEATTLE Site, for Hiro, 10in50 -- comparison of CS targets for different TRTs

sa_median_active = cs_mean_active.CS_target['mu_cond'][:-1]
sa_median_inslab = cs_mean_inslab.CS_target['mu_cond'][:-1]
sa_median_interface = cs_mean_interface.CS_target['mu_cond'][:-1]
periods_cs = cs_mean_active.CS_target['period_arr_cs']


plt.figure(dpi=200)
plt.plot(periods_cs, sa_median_active, '-k', linewidth=1.5,
         label='Crustal (54.2%)')

plt.plot(periods_cs, sa_median_inslab, '-.k', linewidth=1.5,
         label='In-slab (13.1%)')

plt.plot(periods_cs, sa_median_interface, '--k', linewidth=1.5,
         label='Interface (32.7%)')

plt.xlabel('Period (s)')
plt.ylabel('Sa(T) (g)')
plt.xlim(0.1, 10)
plt.ylim(0.01, 5)
plt.xscale('log')
plt.yscale('log')
plt.legend()
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
plt.plot(x_plt, y_plt_active, '-k', label='Crustal', linewidth=1.5)
plt.plot(x_plt, y_plt_inslab, '-.k', label='In-slab', linewidth=1.5)
plt.plot(x_plt, y_plt_interface, '--k', label='Interface', linewidth=1.5)

# plt.legend(loc = 'best')
plt.grid()
plt.xlabel('Da 5-75% (s)')
plt.ylabel('CDF')
plt.ylim(0, 1)
plt.xlim(1, 500)
plt.xscale('log')
plt.show()


#%% compute SaAvg from the mean spectra -- should correspond to the value of 
# the conditioning IM

from scipy.stats import gmean

# compute SaAvg from mean spectra, Active Crustal TRT
x_interp = np.log(periods_cs)
y_interp = np.log(sa_median_active)
x_hat = np.log(np.linspace(0.3, 5.0, 48))
y_hat = np.interp(x = x_hat, xp = x_interp, fp = y_interp)

sa_avg_active = gmean(np.exp(y_hat))

# compute SaAvg from mean spectra, Inslab TRT
x_interp = np.log(periods_cs)
y_interp = np.log(sa_median_inslab)
x_hat = np.log(np.linspace(0.3, 5.0, 48))
y_hat = np.interp(x = x_hat, xp = x_interp, fp = y_interp)

sa_avg_inslab = gmean(np.exp(y_hat))

# compute SaAvg from mean spectra, Interface TRT
x_interp = np.log(periods_cs)
y_interp = np.log(sa_median_interface)
x_hat = np.log(np.linspace(0.3, 5.0, 48))
y_hat = np.interp(x = x_hat, xp = x_interp, fp = y_interp)

sa_avg_interface = gmean(np.exp(y_hat))


# %% SEATTLE Site, for Hiro, 10in50 - save the CS targets

# folder_save = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/CS_targets/seattle_8_story_mrf_10in50_rotD100/active_shallow_crust/'
# cs_mean_active.export_CS_target(folder_save)

# folder_save = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/CS_targets/seattle_8_story_mrf_10in50_rotD100/interface/'
# cs_mean_interface.export_CS_target(folder_save)

# folder_save = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/CS_targets/seattle_8_story_mrf_10in50_rotD100/inslab/'
# cs_mean_inslab.export_CS_target(folder_save)

#%%




# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#
# 8-story MRF :: DBE (2/3MCER) intensity
#
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


# %% SEATTLE Site, for Hiro, DBE -- ACTIVE SHALLOW CRUST

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

deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_210_saAvg_crustal_trt_8_story_mrf_dbe/Mag_Dist-mean-0_210_plot.csv'

prob_col_name = 'mean'  # column name containing P(X>x | M, R); here I assume
# only a single realization is used
delta_M = 0.2
delta_R = 5
investigation_time = 1
# title of the plot
deagg_name = 'Seattle, SaAvg(0.3 : 0.1: 5.0), DBE (2/3MCER), active shallow crust'
# weight to apply to deaggregation values (enter manually)
deagg_weight_use = 0.718

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
#
# -----------------------------------------------------------------------------
# Enter other parameters and estimate the unknown source and distance params
# -----------------------------------------------------------------------------

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
#
# -----------------------------------------------------------------------------
# Specify additional site parameters
# -----------------------------------------------------------------------------

vs30_seattle = 260  # assuming soil class D
# in (km), USGS NSHM2018 consolidated values for the site
z_2p5_seattle = 6.374
z_1p0_seattle = 0.876  # in (km) -- AS_2016 uses the value in km
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
                   # value is negative, there is no difference here when positive value is used
                   Rx=src_dist_dict[2]['R_x'],
                   Fhw=0,
                   # Zbot = None, # for CB_2014 needed only when W is unknown, so I leave this undefined (code checks if attribute exists)
                   W=src_dist_dict[1]['W'],
                   delta=src_dist_dict[1]['delta'],
                   lam=src_dist_dict[1]['lambda'],  # 1000,
                   hypo_depth=src_dist_dict[1]['Z_hyp'],  # 5,
                   Ztor=src_dist_dict[1]['Z_tor'],
                   trt=None)  # Not used in the CB_2014 GMPE

# -----------------------------------------------------------------------------
# Define IM_star and the list of IM_cond for CS computations
# -----------------------------------------------------------------------------

# IM_star & IM_cond -- Sa, SaAvg
GMPE_to_use = 'CB_2014_active'
# iml_im_star = 0.400 # conditioning value of the IM, RotD50
# iml_im_star = 0.488  # conditioning value of the IM, RotD100
# iml_im_star = 0.200  # conditioning value of the IM, RotD100
iml_im_star = 0.411  # conditioning value of the IM, RotD100

# define the imt for the conditioning IM -- 8-story MRF
imt_star_cse = mhu.IntensityMeasureType('SaAvg', sa_period=np.array([0.3,
                                                                     0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
                                                                     1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
                                                                     2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9,
                                                                     4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0]))


# # define the imt for the conditioning IM -- 8-story MRF
# imt_star_cse = mhu.IntensityMeasureType( 'SaAvg', sa_period = np.array([1.0]) )


# define the im object (intensity measure) for IM_star -- specified above
im_star_cse = mhu.IntensityMeasure(iml=iml_im_star, imt=imt_star_cse,
                                   GMM_name=GMPE_to_use)

# Target periods used in the ground motion selections script by Prof. Baker;
# I manually copied the 30 periods computed in:
# selectionParams.TgtPer = logspace(log10(selectionParams.Tmin),log10(selectionParams.Tmax),30)
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
# shorter list so that my computation is faster in develpment
# period_tgt_list = [0.1, 0.5, 1, 3, 5, 10]


# Define the list of conditional IM, these are the IMs conditioned on the
# conditioning IM im_star
im_cond_list_cse = [mhu.IntensityMeasure(iml=None,
                    imt=mhu.IntensityMeasureType(
                        'Sa', sa_period=np.array([x])),
                    GMM_name=GMPE_to_use)
                    for x in period_tgt_list]

# IM_cond DURATION -- append duration IM
duration_GMPE_to_use = 'AS_2016_duration'  # ''
im_cond_list_cse.append(mhu.IntensityMeasure(iml=None,
                                             imt=mhu.IntensityMeasureType(
                                                 'da5_75'),
                                             GMM_name=duration_GMPE_to_use))

# -----------------------------------------------------------------------------
# Compute conditional spectra given a single rupture (i.e., M and R definition)
# -----------------------------------------------------------------------------
cs_mean = mhu.ConditionalSpectra(rup_mean, site_cse,
                                 im_star_cse, im_cond_list_cse,
                                 mhu.sa_corr_baker)

# plot the CS target based on the mean M and R
cs_mean.plot_CS(legend_title=None, is_sample_CS=False, n_sample=25,
                is_plot_cs=True, is_plot_da5_75=True)

# deepcopy the computed CS target for subsequent plotting
cs_mean_active = copy.deepcopy(cs_mean)


# %% SEATTLE Site, for Hiro, DBE (2/3MCER) -- SUBDUCTION IN-SLAB

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

deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_212_saAvg_inslab_trt_8_story_mrf_dbe/Mag_Dist-mean-0_212_plot.csv'

prob_col_name = 'mean'  # column name containing P(X>x | M, R); here I assume
# only a single realization is used
delta_M = 0.2
delta_R = 5
investigation_time = 1
# title of the plot
deagg_name = 'Seattle, SaAvg(0.3 : 0.1: 5.0), DBE (2/3MCER), subduction in-slab'
# weight to apply to deaggregation values (enter manually)
deagg_weight_use = 0.04

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
# 
# -----------------------------------------------------------------------------
# Enter other parameters and estimate the unknown source and distance params
# -----------------------------------------------------------------------------

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
                                    fault_style='normal',  # 'reverse', #'normal',#'strike_slip', #
                                    F_hw=True)
#
# 
# -----------------------------------------------------------------------------
# Specify additional site parameters
# -----------------------------------------------------------------------------

vs30_seattle = 260  # assuming soil class D
# in (km), USGS NSHM2018 consolidated values for the site
z_2p5_seattle = 6.374
z_1p0_seattle = 0.876  # in (km) -- AS_2016 uses the value in km
# global region in the CB_2014 GMPE for active shallow crust
region_seattle_cb_2014 = 0
region_seattle_AG_2020 = "CAS"  # cascadia subduction zone region

z_1p0_seattle_as_2016 = z_1p0_seattle  # AS_2016 needs this value in km
# bahrampouri_et_al_2021 needs this value in (m)
z_1p0_seattle_bahrampouri_2021 = z_1p0_seattle*1000

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
                   # Zbot = None, # for CB_2014 needed only when W is unknown, so I leave this undefined (code checks if attribute exists)
                   W=src_dist_dict[1]['W'],
                   delta=src_dist_dict[1]['delta'],
                   lam=src_dist_dict[1]['lambda'],
                   hypo_depth=np.array([53.17]),
                   Ztor=np.array([46.13]),  
                   trt='subduction_inslab')

# rup_mean = mhu.Rup(M=src_dist_dict[0]['M'],
#                    R=None,  # not needed
#                    Rjb=src_dist_dict[0]['R_jb'],
#                    Rrup=src_dist_dict[2]['R_rup'],
#                    # value is negative, there is no difference here when positive value is used
#                    Rx=src_dist_dict[2]['R_x'],
#                    Fhw=0,
#                    # Zbot = None, # for CB_2014 needed only when W is unknown, so I leave this undefined (code checks if attribute exists)
#                    W=src_dist_dict[1]['W'],
#                    delta=src_dist_dict[1]['delta'],
#                    lam=src_dist_dict[1]['lambda'],  # 1000,
#                    hypo_depth=src_dist_dict[1]['Z_hyp'],  # 5,
#                    Ztor=src_dist_dict[1]['Z_tor'],
#                    trt='subduction_inslab')  # Not used in the CB_2014 GMPE

# rup_mean = mhu.Rup(M=np.array([7.3]),
#                    R=None,  # not needed
#                    Rjb=np.array([62.3]),
#                    Rrup=np.array([86.58347057]), 
#                    Rx=np.array([56.13215502]),
#                    Fhw=1,
#                    W=np.array([26.01]),
#                    delta=np.array([50.]),
#                    lam=np.array([-90.]),
#                    hypo_depth=np.array([53.17]),
#                    Ztor=np.array([46.13]),  
#                    trt='subduction_inslab')

# -----------------------------------------------------------------------------
# Define IM_star and the list of IM_cond for CS computations
# -----------------------------------------------------------------------------

# NOTE: the function always uses the below booleans by default, currently this
#       cannot be changed
# is_ergodic = False
# is_apply_usa_adjustment = True


# IM_star & IM_cond -- Sa, SaAvg
GMPE_to_use = 'AG_2020'
# iml_im_star = 0.400 # conditioning value of the IM, RotD50
# iml_im_star = 0.488  # conditioning value of the IM, RotD100
# iml_im_star = 0.200  # conditioning value of the IM, RotD100
iml_im_star = 0.411  # conditioning value of the IM, RotD100

# define the imt for the conditioning IM -- 8-story MRF
imt_star_cse = mhu.IntensityMeasureType('SaAvg', sa_period=np.array([0.3,
                                                                     0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
                                                                     1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
                                                                     2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9,
                                                                     4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0]))


# # define the imt for the conditioning IM -- 8-story MRF
# imt_star_cse = mhu.IntensityMeasureType( 'SaAvg', sa_period = np.array([1.0]) )


# define the im object (intensity measure) for IM_star -- specified above
im_star_cse = mhu.IntensityMeasure(iml=iml_im_star, imt=imt_star_cse,
                                   GMM_name=GMPE_to_use)

# Target periods used in the ground motion selections script by Prof. Baker;
# I manually copied the 30 periods computed in:
# selectionParams.TgtPer = logspace(log10(selectionParams.Tmin),log10(selectionParams.Tmax),30)
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
# shorter list so that my computation is faster in develpment
# period_tgt_list = [0.1, 0.5, 1, 3, 5, 10]


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
                                             imt=mhu.IntensityMeasureType(
                                                 'da5_75'),
                                             GMM_name=duration_GMPE_to_use))

# -----------------------------------------------------------------------------
# Compute conditional spectra given a single rupture (i.e., M and R definition)
# -----------------------------------------------------------------------------
cs_mean = mhu.ConditionalSpectra(rup_mean, site_cse,
                                 im_star_cse, im_cond_list_cse,
                                 mhu.sa_corr_baker)

# plot the CS target based on the mean M and R
cs_mean.plot_CS(legend_title=None, is_sample_CS=False, n_sample=25,
                is_plot_cs=True, is_plot_da5_75=True)

# deepcopy the computed CS target for subsequent plotting
cs_mean_inslab = copy.deepcopy(cs_mean)


# %% SEATTLE Site, for Hiro, DBE (2/3MCER) -- SUBDUCTION INTERFACE

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

deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_211_saAvg_interface_trt_8_story_mrf_dbe/Mag_Dist-mean-0_211_plot.csv'

prob_col_name = 'mean'  # column name containing P(X>x | M, R); here I assume
# only a single realization is used
delta_M = 0.2
delta_R = 5
investigation_time = 1
# title of the plot
deagg_name = 'Seattle, SaAvg(0.3 : 0.1: 5.0), DBE (2/3MCER), subduction interface'
# weight to apply to deaggregation values (enter manually)
deagg_weight_use = 0.242

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
#
# -----------------------------------------------------------------------------
# Enter other parameters and estimate the unknown source and distance params
# -----------------------------------------------------------------------------

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


# Note: I selected the case for the site on the foot wall because the distances
#   are smaller in that case compared to assuming the site on the hanging wall.
#   In this case the Sa values will be larger, but the durations will be
#   smaller.

#
# -----------------------------------------------------------------------------
# Specify additional site parameters
# -----------------------------------------------------------------------------

vs30_seattle = 260  # assuming soil class D
# in (km), USGS NSHM2018 consolidated values for the site
z_2p5_seattle = 6.374
z_1p0_seattle = 0.876  # in (km) -- AS_2016 uses the value in km
# global region in the CB_2014 GMPE for active shallow crust
region_seattle_cb_2014 = 0
region_seattle_AG_2020 = "CAS"  # cascadia subduction zone region

z_1p0_seattle_as_2016 = z_1p0_seattle  # AS_2016 needs this value in km
# bahrampouri_et_al_2021 needs this value in (m)
z_1p0_seattle_bahrampouri_2021 = z_1p0_seattle*1000

# -----------------------------------------------------------------------------
# Instantiate site and rupture objects
# -----------------------------------------------------------------------------

site_cse = mhu.Site(Vs30=vs30_seattle,
                    region=region_seattle_AG_2020,
                    Z25=z_2p5_seattle,  # AG_2020 needs this value in km
                    # Z10 = z_1p0_seattle_as_2016)
                    Z10=z_1p0_seattle_bahrampouri_2021, # AS_2016 needs this value in km;
                    backarc = False)  
# bahrampouri_et_al_2021 needs this value in (m)

rup_mean = mhu.Rup(M=src_dist_dict[0]['M'],
                   R=None,  # not needed
                   Rjb=src_dist_dict[0]['R_jb'],
                   Rrup=src_dist_dict[2]['R_rup'],
                   # value is negative, there is no difference here when positive value is used
                   Rx=src_dist_dict[2]['R_x'],
                   Fhw=0,
                   # Zbot = None, # for CB_2014 needed only when W is unknown, so I leave this undefined (code checks if attribute exists)
                   W=src_dist_dict[1]['W'],
                   delta=src_dist_dict[1]['delta'],
                   lam=src_dist_dict[1]['lambda'],  # 1000,
                   hypo_depth=src_dist_dict[1]['Z_hyp'],  # 5,
                   Ztor=src_dist_dict[1]['Z_tor'],
                   trt='subduction_interface')  # Not used in the CB_2014 GMPE


# -----------------------------------------------------------------------------
# Define IM_star and the list of IM_cond for CS computations
# -----------------------------------------------------------------------------


# NOTE: the function always uses the below booleans by default, currently this
#       cannot be changed
# is_ergodic = False
# is_apply_usa_adjustment = True


# IM_star & IM_cond -- Sa, SaAvg
# GMPE_to_use = 'AG_2020'
# GMPE_to_use = 'CB_2014_active' # check the effect of using a different GMM
GMPE_to_use ='AbrahamsonEtAl2015SInter' # check the effect of using a different GMM

# iml_im_star = 0.400 # conditioning value of the IM, RotD50
# iml_im_star = 0.488  # conditioning value of the IM, RotD100
# iml_im_star = 0.200  # conditioning value of the IM, RotD100
iml_im_star = 0.411  # conditioning value of the IM, RotD100

# define the imt for the conditioning IM -- 8-story MRF
imt_star_cse = mhu.IntensityMeasureType('SaAvg', sa_period=np.array([0.3,
                                                                     0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
                                                                     1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
                                                                     2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9,
                                                                     4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0]))


# # define the imt for the conditioning IM -- 8-story MRF
# imt_star_cse = mhu.IntensityMeasureType( 'SaAvg', sa_period = np.array([1.0]) )


# define the im object (intensity measure) for IM_star -- specified above
im_star_cse = mhu.IntensityMeasure(iml=iml_im_star, imt=imt_star_cse,
                                   GMM_name=GMPE_to_use)

# Target periods used in the ground motion selections script by Prof. Baker;
# I manually copied the 30 periods computed in:
# selectionParams.TgtPer = logspace(log10(selectionParams.Tmin),log10(selectionParams.Tmax),30)
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
# shorter list so that my computation is faster in develpment
# period_tgt_list = [0.1, 0.5, 1, 3, 5, 10]


# Define the list of conditional IM, these are the IMs conditioned on the
# conditioning IM im_star
im_cond_list_cse = [mhu.IntensityMeasure(iml=None,
                    imt=mhu.IntensityMeasureType(
                        'Sa', sa_period=np.array([x])),
                    GMM_name=GMPE_to_use)
                    for x in period_tgt_list]

# IM_cond DURATION -- append duration IM

# duration_GMPE_to_use = 'AS_2016_duration'
duration_GMPE_to_use = 'BahrampouriEtAlSInter2020_duration'
im_cond_list_cse.append(mhu.IntensityMeasure(iml=None,
                                             imt=mhu.IntensityMeasureType(
                                                 'da5_75'),
                                             GMM_name=duration_GMPE_to_use))

# im_cond_list_cse.append(mhu.IntensityMeasure(iml=None,
#                                              imt=mhu.IntensityMeasureType(
#                                                  'da5_95'),
#                                              GMM_name=duration_GMPE_to_use))

# -----------------------------------------------------------------------------
# Compute conditional spectra given a single rupture (i.e., M and R definition)
# -----------------------------------------------------------------------------
cs_mean = mhu.ConditionalSpectra(rup_mean, site_cse,
                                 im_star_cse, im_cond_list_cse,
                                 mhu.sa_corr_baker)

# plot the CS target based on the mean M and R
cs_mean.plot_CS(legend_title=None, is_sample_CS=False, n_sample=25,
                is_plot_cs=True, is_plot_da5_75=True)

# deepcopy the computed CS target for subsequent plotting
cs_mean_interface = copy.deepcopy(cs_mean)


# %% SEATTLE Site, for Hiro, DBE (2/3MCER) -- comparison of CS targets for different TRTs

sa_median_active = cs_mean_active.CS_target['mu_cond'][:-1]
sa_median_inslab = cs_mean_inslab.CS_target['mu_cond'][:-1]
sa_median_interface = cs_mean_interface.CS_target['mu_cond'][:-1]
periods_cs = cs_mean_active.CS_target['period_arr_cs']


plt.figure(dpi=200)
plt.plot(periods_cs, sa_median_active, '-k', linewidth=1.5,
         label='Crustal (71.8%)')

plt.plot(periods_cs, sa_median_inslab, '-.k', linewidth=1.5,
         label='In-slab (4.0%)')

plt.plot(periods_cs, sa_median_interface, '--k', linewidth=1.5,
         label='Interface (24.2%)')

plt.xlabel('Period (s)')
plt.ylabel('Sa(T) (g)')
plt.xlim(0.1, 10)
plt.ylim(0.01, 5)
plt.xscale('log')
plt.yscale('log')
plt.legend()
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
plt.plot(x_plt, y_plt_active, '-k', label='Crustal', linewidth=1.5)
plt.plot(x_plt, y_plt_inslab, '-.k', label='In-slab', linewidth=1.5)
plt.plot(x_plt, y_plt_interface, '--k', label='Interface', linewidth=1.5)

# plt.legend(loc = 'best')
plt.grid()
plt.xlabel('Da 5-75% (s)')
plt.ylabel('CDF')
plt.ylim(0, 1)
plt.xlim(1, 500)
plt.xscale('log')
plt.show()


#%% compute SaAvg from the mean spectra -- should correspond to the value of 
# the conditioning IM

from scipy.stats import gmean

# compute SaAvg from mean spectra, Active Crustal TRT
x_interp = np.log(periods_cs)
y_interp = np.log(sa_median_active)
x_hat = np.log(np.linspace(0.3, 5.0, 48))
y_hat = np.interp(x = x_hat, xp = x_interp, fp = y_interp)

sa_avg_active = gmean(np.exp(y_hat))

# compute SaAvg from mean spectra, Inslab TRT
x_interp = np.log(periods_cs)
y_interp = np.log(sa_median_inslab)
x_hat = np.log(np.linspace(0.3, 5.0, 48))
y_hat = np.interp(x = x_hat, xp = x_interp, fp = y_interp)

sa_avg_inslab = gmean(np.exp(y_hat))

# compute SaAvg from mean spectra, Interface TRT
x_interp = np.log(periods_cs)
y_interp = np.log(sa_median_interface)
x_hat = np.log(np.linspace(0.3, 5.0, 48))
y_hat = np.interp(x = x_hat, xp = x_interp, fp = y_interp)

sa_avg_interface = gmean(np.exp(y_hat))


# %% SEATTLE Site, for Hiro, 10in50 - save the CS targets

folder_save = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/CS_targets/seattle_8_story_mrf_dbe_rotD100/active_shallow_crust/'
cs_mean_active.export_CS_target(folder_save)

folder_save = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/CS_targets/seattle_8_story_mrf_dbe_rotD100/interface/'
cs_mean_interface.export_CS_target(folder_save)

folder_save = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/CS_targets/seattle_8_story_mrf_dbe_rotD100/inslab/'
cs_mean_inslab.export_CS_target(folder_save)



#%%

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#
# 8-story MRF :: 2in50 intensity
#
# -----------------------------------------------------------------------------
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

deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_190_saAvg_crustal_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/Mag_Dist-mean-0_190_for_plotting.csv'

prob_col_name = 'mean'  # column name containing P(X>x | M, R); here I assume
# only a single realization is used
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
#
# -----------------------------------------------------------------------------
# Enter other parameters and estimate the unknown source and distance params
# -----------------------------------------------------------------------------

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
#
# -----------------------------------------------------------------------------
# Specify additional site parameters
# -----------------------------------------------------------------------------

vs30_seattle = 260  # assuming soil class D
# in (km), USGS NSHM2018 consolidated values for the site
z_2p5_seattle = 6.374
z_1p0_seattle = 0.876  # in (km) -- AS_2016 uses the value in km
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
                   # value is negative, there is no difference here when positive value is used
                   Rx=src_dist_dict[2]['R_x'],
                   Fhw=0,
                   # Zbot = None, # for CB_2014 needed only when W is unknown, so I leave this undefined (code checks if attribute exists)
                   W=src_dist_dict[1]['W'],
                   delta=src_dist_dict[1]['delta'],
                   lam=src_dist_dict[1]['lambda'],  # 1000,
                   hypo_depth=src_dist_dict[1]['Z_hyp'],  # 5,
                   Ztor=src_dist_dict[1]['Z_tor'],
                   trt=None)  # Not used in the CB_2014 GMPE

# -----------------------------------------------------------------------------
# Define IM_star and the list of IM_cond for CS computations
# -----------------------------------------------------------------------------

# IM_star & IM_cond -- Sa, SaAvg
GMPE_to_use = 'CB_2014_active'
# iml_im_star = 0.400 # conditioning value of the IM, RotD50
iml_im_star = 0.488  # conditioning value of the IM, RotD100

# define the imt for the conditioning IM -- 8-story MRF
imt_star_cse = mhu.IntensityMeasureType('SaAvg', sa_period=np.array([0.3,
                                                                     0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
                                                                     1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
                                                                     2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9,
                                                                     4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0]))


# # define the imt for the conditioning IM -- 8-story MRF
# imt_star_cse = mhu.IntensityMeasureType( 'SaAvg', sa_period = np.array([1.0]) )


# define the im object (intensity measure) for IM_star -- specified above
im_star_cse = mhu.IntensityMeasure(iml=iml_im_star, imt=imt_star_cse,
                                   GMM_name=GMPE_to_use)

# Target periods used in the ground motion selections script by Prof. Baker;
# I manually copied the 30 periods computed in:
# selectionParams.TgtPer = logspace(log10(selectionParams.Tmin),log10(selectionParams.Tmax),30)
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
# shorter list so that my computation is faster in develpment
# period_tgt_list = [0.1, 0.5, 1, 3, 5, 10]


# Define the list of conditional IM, these are the IMs conditioned on the
# conditioning IM im_star
im_cond_list_cse = [mhu.IntensityMeasure(iml=None,
                    imt=mhu.IntensityMeasureType(
                        'Sa', sa_period=np.array([x])),
                    GMM_name=GMPE_to_use)
                    for x in period_tgt_list]

# IM_cond DURATION -- append duration IM
duration_GMPE_to_use = 'AS_2016_duration'  # ''
im_cond_list_cse.append(mhu.IntensityMeasure(iml=None,
                                             imt=mhu.IntensityMeasureType(
                                                 'da5_75'),
                                             GMM_name=duration_GMPE_to_use))

# -----------------------------------------------------------------------------
# Compute conditional spectra given a single rupture (i.e., M and R definition)
# -----------------------------------------------------------------------------
cs_mean = mhu.ConditionalSpectra(rup_mean, site_cse,
                                 im_star_cse, im_cond_list_cse,
                                 mhu.sa_corr_baker)

# plot the CS target based on the mean M and R
cs_mean.plot_CS(legend_title=None, is_sample_CS=False, n_sample=25,
                is_plot_cs=True, is_plot_da5_75=True)

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

deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_192_saAvg_inslab_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/Mag_Dist-mean-0_192_for_plotting.csv'

prob_col_name = 'mean'  # column name containing P(X>x | M, R); here I assume
# only a single realization is used
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
# 
# -----------------------------------------------------------------------------
# Enter other parameters and estimate the unknown source and distance params
# -----------------------------------------------------------------------------

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
                                    fault_style='normal',  # 'reverse', #'normal',#'strike_slip', #
                                    F_hw=True)
#
# 
# -----------------------------------------------------------------------------
# Specify additional site parameters
# -----------------------------------------------------------------------------

vs30_seattle = 260  # assuming soil class D
# in (km), USGS NSHM2018 consolidated values for the site
z_2p5_seattle = 6.374
z_1p0_seattle = 0.876  # in (km) -- AS_2016 uses the value in km
# global region in the CB_2014 GMPE for active shallow crust
region_seattle_cb_2014 = 0
region_seattle_AG_2020 = "CAS"  # cascadia subduction zone region

z_1p0_seattle_as_2016 = z_1p0_seattle  # AS_2016 needs this value in km
# bahrampouri_et_al_2021 needs this value in (m)
z_1p0_seattle_bahrampouri_2021 = z_1p0_seattle*1000

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
                   # Zbot = None, # for CB_2014 needed only when W is unknown, so I leave this undefined (code checks if attribute exists)
                   W=src_dist_dict[1]['W'],
                   delta=src_dist_dict[1]['delta'],
                   lam=src_dist_dict[1]['lambda'],
                   hypo_depth=np.array([53.17]),
                   Ztor=np.array([46.13]),  
                   trt='subduction_inslab')

# rup_mean = mhu.Rup(M=src_dist_dict[0]['M'],
#                    R=None,  # not needed
#                    Rjb=src_dist_dict[0]['R_jb'],
#                    Rrup=src_dist_dict[2]['R_rup'],
#                    # value is negative, there is no difference here when positive value is used
#                    Rx=src_dist_dict[2]['R_x'],
#                    Fhw=0,
#                    # Zbot = None, # for CB_2014 needed only when W is unknown, so I leave this undefined (code checks if attribute exists)
#                    W=src_dist_dict[1]['W'],
#                    delta=src_dist_dict[1]['delta'],
#                    lam=src_dist_dict[1]['lambda'],  # 1000,
#                    hypo_depth=src_dist_dict[1]['Z_hyp'],  # 5,
#                    Ztor=src_dist_dict[1]['Z_tor'],
#                    trt='subduction_inslab')  # Not used in the CB_2014 GMPE

# rup_mean = mhu.Rup(M=np.array([7.3]),
#                    R=None,  # not needed
#                    Rjb=np.array([62.3]),
#                    Rrup=np.array([86.58347057]), 
#                    Rx=np.array([56.13215502]),
#                    Fhw=1,
#                    W=np.array([26.01]),
#                    delta=np.array([50.]),
#                    lam=np.array([-90.]),
#                    hypo_depth=np.array([53.17]),
#                    Ztor=np.array([46.13]),  
#                    trt='subduction_inslab')

# -----------------------------------------------------------------------------
# Define IM_star and the list of IM_cond for CS computations
# -----------------------------------------------------------------------------

# NOTE: the function always uses the below booleans by default, currently this
#       cannot be changed
# is_ergodic = False
# is_apply_usa_adjustment = True


# IM_star & IM_cond -- Sa, SaAvg
GMPE_to_use = 'AG_2020'
# iml_im_star = 0.400 # conditioning value of the IM, RotD50
iml_im_star = 0.488  # conditioning value of the IM, RotD100

# define the imt for the conditioning IM -- 8-story MRF
imt_star_cse = mhu.IntensityMeasureType('SaAvg', sa_period=np.array([0.3,
                                                                     0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
                                                                     1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
                                                                     2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9,
                                                                     4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0]))


# # define the imt for the conditioning IM -- 8-story MRF
# imt_star_cse = mhu.IntensityMeasureType( 'SaAvg', sa_period = np.array([1.0]) )


# define the im object (intensity measure) for IM_star -- specified above
im_star_cse = mhu.IntensityMeasure(iml=iml_im_star, imt=imt_star_cse,
                                   GMM_name=GMPE_to_use)

# Target periods used in the ground motion selections script by Prof. Baker;
# I manually copied the 30 periods computed in:
# selectionParams.TgtPer = logspace(log10(selectionParams.Tmin),log10(selectionParams.Tmax),30)
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
# shorter list so that my computation is faster in develpment
# period_tgt_list = [0.1, 0.5, 1, 3, 5, 10]


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
                                             imt=mhu.IntensityMeasureType(
                                                 'da5_75'),
                                             GMM_name=duration_GMPE_to_use))

# -----------------------------------------------------------------------------
# Compute conditional spectra given a single rupture (i.e., M and R definition)
# -----------------------------------------------------------------------------
cs_mean = mhu.ConditionalSpectra(rup_mean, site_cse,
                                 im_star_cse, im_cond_list_cse,
                                 mhu.sa_corr_baker)

# plot the CS target based on the mean M and R
cs_mean.plot_CS(legend_title=None, is_sample_CS=False, n_sample=25,
                is_plot_cs=True, is_plot_da5_75=True)

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

deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_191_saAvg_interface_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/Mag_Dist-mean-0_191_for_plotting.csv'

prob_col_name = 'mean'  # column name containing P(X>x | M, R); here I assume
# only a single realization is used
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
#
# -----------------------------------------------------------------------------
# Enter other parameters and estimate the unknown source and distance params
# -----------------------------------------------------------------------------

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


# Note: I selected the case for the site on the foot wall because the distances
#   are smaller in that case compared to assuming the site on the hanging wall.
#   In this case the Sa values will be larger, but the durations will be
#   smaller.

#
# -----------------------------------------------------------------------------
# Specify additional site parameters
# -----------------------------------------------------------------------------

vs30_seattle = 260  # assuming soil class D
# in (km), USGS NSHM2018 consolidated values for the site
z_2p5_seattle = 6.374
z_1p0_seattle = 0.876  # in (km) -- AS_2016 uses the value in km
# global region in the CB_2014 GMPE for active shallow crust
region_seattle_cb_2014 = 0
region_seattle_AG_2020 = "CAS"  # cascadia subduction zone region

z_1p0_seattle_as_2016 = z_1p0_seattle  # AS_2016 needs this value in km
# bahrampouri_et_al_2021 needs this value in (m)
z_1p0_seattle_bahrampouri_2021 = z_1p0_seattle*1000

# -----------------------------------------------------------------------------
# Instantiate site and rupture objects
# -----------------------------------------------------------------------------

site_cse = mhu.Site(Vs30=vs30_seattle,
                    region=region_seattle_AG_2020,
                    Z25=z_2p5_seattle,  # AG_2020 needs this value in km
                    # Z10 = z_1p0_seattle_as_2016)
                    Z10=z_1p0_seattle_bahrampouri_2021, # AS_2016 needs this value in km;
                    backarc = False)  
# bahrampouri_et_al_2021 needs this value in (m)

rup_mean = mhu.Rup(M=src_dist_dict[0]['M'],
                   R=None,  # not needed
                   Rjb=src_dist_dict[0]['R_jb'],
                   Rrup=src_dist_dict[2]['R_rup'],
                   # value is negative, there is no difference here when positive value is used
                   Rx=src_dist_dict[2]['R_x'],
                   Fhw=0,
                   # Zbot = None, # for CB_2014 needed only when W is unknown, so I leave this undefined (code checks if attribute exists)
                   W=src_dist_dict[1]['W'],
                   delta=src_dist_dict[1]['delta'],
                   lam=src_dist_dict[1]['lambda'],  # 1000,
                   hypo_depth=src_dist_dict[1]['Z_hyp'],  # 5,
                   Ztor=src_dist_dict[1]['Z_tor'],
                   trt='subduction_interface')  # Not used in the CB_2014 GMPE


# -----------------------------------------------------------------------------
# Define IM_star and the list of IM_cond for CS computations
# -----------------------------------------------------------------------------


# NOTE: the function always uses the below booleans by default, currently this
#       cannot be changed
# is_ergodic = False
# is_apply_usa_adjustment = True


# IM_star & IM_cond -- Sa, SaAvg
# GMPE_to_use = 'AG_2020'
# GMPE_to_use = 'CB_2014_active' # check the effect of using a different GMM
GMPE_to_use ='AbrahamsonEtAl2015SInter' # check the effect of using a different GMM

# iml_im_star = 0.400 # conditioning value of the IM, RotD50
iml_im_star = 0.488  # conditioning value of the IM, RotD100


# define the imt for the conditioning IM -- 8-story MRF
imt_star_cse = mhu.IntensityMeasureType('SaAvg', sa_period=np.array([0.3,
                                                                     0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
                                                                     1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
                                                                     2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9,
                                                                     4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0]))


# # define the imt for the conditioning IM -- 8-story MRF
# imt_star_cse = mhu.IntensityMeasureType( 'SaAvg', sa_period = np.array([1.0]) )


# define the im object (intensity measure) for IM_star -- specified above
im_star_cse = mhu.IntensityMeasure(iml=iml_im_star, imt=imt_star_cse,
                                   GMM_name=GMPE_to_use)

# Target periods used in the ground motion selections script by Prof. Baker;
# I manually copied the 30 periods computed in:
# selectionParams.TgtPer = logspace(log10(selectionParams.Tmin),log10(selectionParams.Tmax),30)
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
# shorter list so that my computation is faster in develpment
# period_tgt_list = [0.1, 0.5, 1, 3, 5, 10]


# Define the list of conditional IM, these are the IMs conditioned on the
# conditioning IM im_star
im_cond_list_cse = [mhu.IntensityMeasure(iml=None,
                    imt=mhu.IntensityMeasureType(
                        'Sa', sa_period=np.array([x])),
                    GMM_name=GMPE_to_use)
                    for x in period_tgt_list]

# IM_cond DURATION -- append duration IM

# duration_GMPE_to_use = 'AS_2016_duration'
duration_GMPE_to_use = 'BahrampouriEtAlSInter2020_duration'
im_cond_list_cse.append(mhu.IntensityMeasure(iml=None,
                                             imt=mhu.IntensityMeasureType(
                                                 'da5_75'),
                                             GMM_name=duration_GMPE_to_use))

# im_cond_list_cse.append(mhu.IntensityMeasure(iml=None,
#                                              imt=mhu.IntensityMeasureType(
#                                                  'da5_95'),
#                                              GMM_name=duration_GMPE_to_use))

# -----------------------------------------------------------------------------
# Compute conditional spectra given a single rupture (i.e., M and R definition)
# -----------------------------------------------------------------------------
cs_mean = mhu.ConditionalSpectra(rup_mean, site_cse,
                                 im_star_cse, im_cond_list_cse,
                                 mhu.sa_corr_baker)

# plot the CS target based on the mean M and R
cs_mean.plot_CS(legend_title=None, is_sample_CS=False, n_sample=25,
                is_plot_cs=True, is_plot_da5_75=True)

# deepcopy the computed CS target for subsequent plotting
cs_mean_interface = copy.deepcopy(cs_mean)

# %% SEATTLE Site, for Hiro, 2in50 -- comparison of CS targets for different TRTs

sa_median_active = cs_mean_active.CS_target['mu_cond'][:-1]
sa_median_inslab = cs_mean_inslab.CS_target['mu_cond'][:-1]
sa_median_interface = cs_mean_interface.CS_target['mu_cond'][:-1]
periods_cs = cs_mean_active.CS_target['period_arr_cs']


plt.figure(dpi=200)
plt.plot(periods_cs, sa_median_active, '-k', linewidth=1.5,
         label='Crustal (76.5%)')

plt.plot(periods_cs, sa_median_inslab, '-.k', linewidth=1.5,
         label='In-slab (2.7%)')

plt.plot(periods_cs, sa_median_interface, '--k', linewidth=1.5,
         label='Interface (20.8%)')

plt.xlabel('Period (s)')
plt.ylabel('Sa(T) (g)')
plt.xlim(0.1, 10)
plt.ylim(0.01, 5)
plt.xscale('log')
plt.yscale('log')
plt.legend()
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
plt.plot(x_plt, y_plt_active, '-k', label='Crustal', linewidth=1.5)
plt.plot(x_plt, y_plt_inslab, '-.k', label='In-slab', linewidth=1.5)
plt.plot(x_plt, y_plt_interface, '--k', label='Interface', linewidth=1.5)

# plt.legend(loc = 'best')
plt.grid()
plt.xlabel('Da 5-75% (s)')
plt.ylabel('CDF')
plt.ylim(0, 1)
plt.xlim(1, 500)
plt.xscale('log')
plt.show()


#%% compute SaAvg from the mean spectra -- should correspond to the value of 
# the conditioning IM

from scipy.stats import gmean

# compute SaAvg from mean spectra, Active Crustal TRT
x_interp = np.log(periods_cs)
y_interp = np.log(sa_median_active)
x_hat = np.log(np.linspace(0.3, 5.0, 48))
y_hat = np.interp(x = x_hat, xp = x_interp, fp = y_interp)

sa_avg_active = gmean(np.exp(y_hat))

# compute SaAvg from mean spectra, Inslab TRT
x_interp = np.log(periods_cs)
y_interp = np.log(sa_median_inslab)
x_hat = np.log(np.linspace(0.3, 5.0, 48))
y_hat = np.interp(x = x_hat, xp = x_interp, fp = y_interp)

sa_avg_inslab = gmean(np.exp(y_hat))

# compute SaAvg from mean spectra, Interface TRT
x_interp = np.log(periods_cs)
y_interp = np.log(sa_median_interface)
x_hat = np.log(np.linspace(0.3, 5.0, 48))
y_hat = np.interp(x = x_hat, xp = x_interp, fp = y_interp)

sa_avg_interface = gmean(np.exp(y_hat))


# %% SEATTLE Site, for Hiro, 2in50 - save the CS targets

# folder_save = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/CS_targets/seattle_8_story_mrf_2in50_rotD100/active_shallow_crust/'
# cs_mean_active.export_CS_target(folder_save)

# folder_save = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/CS_targets/seattle_8_story_mrf_2in50_rotD100/interface/'
# cs_mean_interface.export_CS_target(folder_save)

# folder_save = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/CS_targets/seattle_8_story_mrf_2in50_rotD100/inslab/'
# cs_mean_inslab.export_CS_target(folder_save)

# folder_save = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/CS_targets/seattle_8_story_mrf_2in50_rotD100/interface_cb14/'
# cs_mean_interface.export_CS_target(folder_save)









































# %% CODE BELOW IS A TEMPLATE FROM BEFORE -- HERE I DO THE RotD100 version,
# update in April 2025

# %% SEATTLE Site, for Hiro, CS target computations

# -----------------------------------------------------------------------------
# Intensity level: 10% in 50 years
# -----------------------------------------------------------------------------

# %% SEATTLE Site, for Hiro, 10in50 -- ACTIVE SHALLOW CRUST

# -----------------------------------------------------------------------------
# GMPE INFORMATION
# -----------------------------------------------------------------------------

# Sa GMPE: CB_2014
# Da5-75 GMPE: AS_2016
# Sa correlations: sa_corr_baker (Baker & Jayaram, 2008)
# Sa-Da correlations: Bradley (2011)

# -----------------------------------------------------------------------------
# Deaggregation file data :: 10in50
# -----------------------------------------------------------------------------

deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/10in50/calc_id_117_10in50_saAVG_active_TRT_vs30_260_AbrahamsonGulerce2020_cas_adjust_true_z1p0_876_z2p5_6p374_inslab_and_interface/Mag_Dist-mean-0_117_for_plot.csv'

prob_col_name = 'mean'  # column name containing P(X>x | M, R); here I assume
# only a single realization is used
delta_M = 0.2
delta_R = 5
investigation_time = 1
# title of the plot
deagg_name = 'Seattle, SaAvg(0.3 : 0.1: 5.0), RP = 475 years (10% in 50 years), active shallow crust'
# weight to apply to deaggregation values (enter manually)
deagg_weight_use = 0.3545

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
# Enter other parameters and estimate the unknown source and distance params
# -----------------------------------------------------------------------------

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

# F_hw = True; site on the hanging wall
# ({'M': 6.867259580640582,
#   'R_jb': 12.451128276230822,
#   'fault_style': 'reverse',
#   'F_hw': True},
#  {'lambda': array([90]),
#   'delta': array([40]),
#   'W': 16.053747504876775,
#   'Z_hyp': 9.866548083871884,
#   'Z_tor': array([3.67505809])},
#  {'alpha': array([50]),
#   'R_x': array([14.58597513]),
#   'R_rup': array([17.2746722])})

# F_hw = False; site on the foot wall
# ({'M': 6.867259580640582,
#   'R_jb': 12.451128276230822,
#   'fault_style': 'reverse',
#   'F_hw': False},
#  {'lambda': array([90]),
#   'delta': array([40]),
#   'W': 16.053747504876775,
#   'Z_hyp': 9.866548083871884,
#   'Z_tor': array([3.67505809])},
#  {'alpha': array([-50]),
#   'R_x': array([-9.53811763]),
#   'R_rup': array([12.98216651])})

# Note: I selected the case for the site on the foot wall because the distances
#   are smaller in that case compared to assuming the site on the hanging wall.

# -----------------------------------------------------------------------------
# Specify additional site parameters
# -----------------------------------------------------------------------------

vs30_seattle = 260  # assuming soil class D
# in (km), USGS NSHM2018 consolidated values for the site
z_2p5_seattle = 6.374
z_1p0_seattle = 0.876  # in (km) -- AS_2016 uses the value in km
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
                   # value is negative, there is no difference here when positive value is used
                   Rx=src_dist_dict[2]['R_x'],
                   Fhw=0,
                   # Zbot = None, # for CB_2014 needed only when W is unknown, so I leave this undefined (code checks if attribute exists)
                   W=src_dist_dict[1]['W'],
                   delta=src_dist_dict[1]['delta'],
                   lam=src_dist_dict[1]['lambda'],  # 1000,
                   hypo_depth=src_dist_dict[1]['Z_hyp'],  # 5,
                   Ztor=src_dist_dict[1]['Z_tor'],
                   trt=None)  # Not used in the CB_2014 GMPE

# -----------------------------------------------------------------------------
# Define IM_star and the list of IM_cond for CS computations
# -----------------------------------------------------------------------------

# IM_star & IM_cond -- Sa, SaAvg
GMPE_to_use = 'CB_2014_active'
iml_im_star = 0.188  # conditioning value of the IM

# define the imt for the conditioning IM -- 8-story MRF
imt_star_cse = mhu.IntensityMeasureType('SaAvg', sa_period=np.array([0.3,
                                                                     0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
                                                                     1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
                                                                     2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9,
                                                                     4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0]))


# # define the imt for the conditioning IM -- 8-story MRF
# imt_star_cse = mhu.IntensityMeasureType( 'SaAvg', sa_period = np.array([1.0]) )


# define the im object (intensity measure) for IM_star -- specified above
im_star_cse = mhu.IntensityMeasure(iml=iml_im_star, imt=imt_star_cse,
                                   GMM_name=GMPE_to_use)

# Target periods used in the ground motion selections script by Prof. Baker;
# I manually copied the 30 periods computed in:
# selectionParams.TgtPer = logspace(log10(selectionParams.Tmin),log10(selectionParams.Tmax),30)
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
# shorter list so that my computation is faster in develpment
# period_tgt_list = [0.1, 0.5, 1, 3, 5, 10]


# Define the list of conditional IM, these are the IMs conditioned on the
# conditioning IM im_star
im_cond_list_cse = [mhu.IntensityMeasure(iml=None,
                    imt=mhu.IntensityMeasureType(
                        'Sa', sa_period=np.array([x])),
                    GMM_name=GMPE_to_use)
                    for x in period_tgt_list]

# IM_cond DURATION -- append duration IM
duration_GMPE_to_use = 'AS_2016_duration'  # ''
im_cond_list_cse.append(mhu.IntensityMeasure(iml=None,
                                             imt=mhu.IntensityMeasureType(
                                                 'da5_75'),
                                             GMM_name=duration_GMPE_to_use))

# -----------------------------------------------------------------------------
# Compute conditional spectra given a single rupture (i.e., M and R definition)
# -----------------------------------------------------------------------------
cs_mean = mhu.ConditionalSpectra(rup_mean, site_cse,
                                 im_star_cse, im_cond_list_cse,
                                 mhu.sa_corr_baker)

# plot the CS target based on the mean M and R
cs_mean.plot_CS(legend_title=None, is_sample_CS=False, n_sample=25,
                is_plot_cs=True, is_plot_da5_75=True)

# deepcopy the computed CS target for subsequent plotting
cs_mean_active = copy.deepcopy(cs_mean)

# %% SEATTLE Site, for Hiro, 10in50 -- SUBDUCTION INTERFACE

# -----------------------------------------------------------------------------
# GMPE INFORMATION
# -----------------------------------------------------------------------------

# Sa GMPE: AG_2020 (Abrahamson & Gulerce 2020 GMPE for subduction zone events)
# Da5-75 GMPE: bahrampouri_et_al_2021 (duration GMPE for subduction zone events); compare to AS_2016
# Sa correlations: sa_corr_baker (Baker & Jayaram, 2008)
# Sa-Da correlations: Bradley (2011)

# -----------------------------------------------------------------------------
# Deaggregation file data :: 10in50
# -----------------------------------------------------------------------------

deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/10in50/calc_id_119_10in50_saAVG_interface_TRT_vs30_260_AbrahamsonGulerce2020_cas_adjust_true_z1p0_876_z2p5_6p374_inslab_and_interface/Mag_Dist-mean-0_119_for_plot.csv'

prob_col_name = 'mean'  # column name containing P(X>x | M, R); here I assume
# only a single realization is used
delta_M = 0.2
delta_R = 5
investigation_time = 1
# title of the plot
deagg_name = 'Seattle, SaAvg(0.3 : 0.1: 5.0), RP = 475 years (10% in 50 years), subduction interface'
# weight to apply to deaggregation values (enter manually)
deagg_weight_use = 0.3649

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
# Enter other parameters and estimate the unknown source and distance params
# -----------------------------------------------------------------------------

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

# F_hw = True; site on the hanging wall
# ({'M': 8.962866629007227,
#   'R_jb': 114.59478883289601,
#   'fault_style': 'reverse',
#   'F_hw': True},
#  {'lambda': array([90]),
#   'delta': array([40]),
#   'W': 116.08478941873268,
#   'Z_hyp': 9.447426674198555,
#   'Z_tor': 0},
#  {'alpha': array([50]),
#   'R_x': array([128.26816805]),
#   'R_rup': array([135.58036996])})

# F_hw = False; site on the foot wall
# ({'M': 8.962866629007227,
#   'R_jb': 114.59478883289601,
#   'fault_style': 'reverse',
#   'F_hw': False},
#  {'lambda': array([90]),
#   'delta': array([40]),
#   'W': 116.08478941873268,
#   'Z_hyp': 9.447426674198555,
#   'Z_tor': 0},
#  {'alpha': array([-50]),
#   'R_x': array([-87.7847012]),
#   'R_rup': array([114.59478883])})

# Note: I selected the case for the site on the foot wall because the distances
#   are smaller in that case compared to assuming the site on the hanging wall.
#   In this case the Sa values will be larger, but the durations will be
#   smaller.


# -----------------------------------------------------------------------------
# Specify additional site parameters
# -----------------------------------------------------------------------------

vs30_seattle = 260  # assuming soil class D
# in (km), USGS NSHM2018 consolidated values for the site
z_2p5_seattle = 6.374
z_1p0_seattle = 0.876  # in (km) -- AS_2016 uses the value in km
# global region in the CB_2014 GMPE for active shallow crust
region_seattle_cb_2014 = 0
region_seattle_AG_2020 = "CAS"  # cascadia subduction zone region

z_1p0_seattle_as_2016 = z_1p0_seattle  # AS_2016 needs this value in km
# bahrampouri_et_al_2021 needs this value in (m)
z_1p0_seattle_bahrampouri_2021 = z_1p0_seattle*1000

# -----------------------------------------------------------------------------
# Instantiate site and rupture objects
# -----------------------------------------------------------------------------

site_cse = mhu.Site(Vs30=vs30_seattle,
                    region=region_seattle_AG_2020,
                    Z25=z_2p5_seattle,  # AG_2020 needs this value in km
                    # Z10 = z_1p0_seattle_as_2016)
                    Z10=z_1p0_seattle_bahrampouri_2021)  # AS_2016 needs this value in km;
# bahrampouri_et_al_2021 needs this value in (m)

rup_mean = mhu.Rup(M=src_dist_dict[0]['M'],
                   R=None,  # not needed
                   Rjb=src_dist_dict[0]['R_jb'],
                   Rrup=src_dist_dict[2]['R_rup'],
                   # value is negative, there is no difference here when positive value is used
                   Rx=src_dist_dict[2]['R_x'],
                   Fhw=0,
                   # Zbot = None, # for CB_2014 needed only when W is unknown, so I leave this undefined (code checks if attribute exists)
                   W=src_dist_dict[1]['W'],
                   delta=src_dist_dict[1]['delta'],
                   lam=src_dist_dict[1]['lambda'],  # 1000,
                   hypo_depth=src_dist_dict[1]['Z_hyp'],  # 5,
                   Ztor=src_dist_dict[1]['Z_tor'],
                   trt='subduction_interface')  # Not used in the CB_2014 GMPE

# -----------------------------------------------------------------------------
# Define IM_star and the list of IM_cond for CS computations
# -----------------------------------------------------------------------------


# NOTE: the function always uses the below booleans by default, currently this
#       cannot be changed
# is_ergodic = False
# is_apply_usa_adjustment = True


# IM_star & IM_cond -- Sa, SaAvg
GMPE_to_use = 'AG_2020'
iml_im_star = 0.188  # conditioning value of the IM

# define the imt for the conditioning IM -- 8-story MRF
imt_star_cse = mhu.IntensityMeasureType('SaAvg', sa_period=np.array([0.3,
                                                                     0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
                                                                     1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
                                                                     2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9,
                                                                     4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0]))


# # define the imt for the conditioning IM -- 8-story MRF
# imt_star_cse = mhu.IntensityMeasureType( 'SaAvg', sa_period = np.array([1.0]) )


# define the im object (intensity measure) for IM_star -- specified above
im_star_cse = mhu.IntensityMeasure(iml=iml_im_star, imt=imt_star_cse,
                                   GMM_name=GMPE_to_use)

# Target periods used in the ground motion selections script by Prof. Baker;
# I manually copied the 30 periods computed in:
# selectionParams.TgtPer = logspace(log10(selectionParams.Tmin),log10(selectionParams.Tmax),30)
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
# shorter list so that my computation is faster in develpment
# period_tgt_list = [0.1, 0.5, 1, 3, 5, 10]


# Define the list of conditional IM, these are the IMs conditioned on the
# conditioning IM im_star
im_cond_list_cse = [mhu.IntensityMeasure(iml=None,
                    imt=mhu.IntensityMeasureType(
                        'Sa', sa_period=np.array([x])),
                    GMM_name=GMPE_to_use)
                    for x in period_tgt_list]

# IM_cond DURATION -- append duration IM

# duration_GMPE_to_use = 'AS_2016_duration'
duration_GMPE_to_use = 'BahrampouriEtAlSInter2020_duration'
im_cond_list_cse.append(mhu.IntensityMeasure(iml=None,
                                             imt=mhu.IntensityMeasureType(
                                                 'da5_75'),
                                             GMM_name=duration_GMPE_to_use))

# -----------------------------------------------------------------------------
# Compute conditional spectra given a single rupture (i.e., M and R definition)
# -----------------------------------------------------------------------------
cs_mean = mhu.ConditionalSpectra(rup_mean, site_cse,
                                 im_star_cse, im_cond_list_cse,
                                 mhu.sa_corr_baker)

# plot the CS target based on the mean M and R
cs_mean.plot_CS(legend_title=None, is_sample_CS=False, n_sample=25,
                is_plot_cs=True, is_plot_da5_75=True)

# deepcopy the computed CS target for subsequent plotting
cs_mean_interface = copy.deepcopy(cs_mean)

# %% SEATTLE Site, for Hiro, 10in50 -- SUBDUCTION IN-SLAB

# -----------------------------------------------------------------------------
# GMPE INFORMATION
# -----------------------------------------------------------------------------

# Sa GMPE: AG_2020 (Abrahamson & Gulerce 2020 GMPE for subduction zone events)
# Da5-75 GMPE: bahrampouri_et_al_2021 (duration GMPE for subduction zone events); compare to AS_2016
# Sa correlations: sa_corr_baker (Baker & Jayaram, 2008)
# Sa-Da correlations: Bradley (2011)

# -----------------------------------------------------------------------------
# Deaggregation file data :: 10in50
# -----------------------------------------------------------------------------

deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/10in50/calc_id_120_10in50_saAVG_inslab_TRT_vs30_260_AbrahamsonGulerce2020_cas_adjust_true_z1p0_876_z2p5_6p374_inslab_and_interface/Mag_Dist-mean-0_120_for_plot.csv'

prob_col_name = 'mean'  # column name containing P(X>x | M, R); here I assume
# only a single realization is used
delta_M = 0.2
delta_R = 5
investigation_time = 1
# title of the plot
deagg_name = 'Seattle, SaAvg(0.3 : 0.1: 5.0), RP = 475 years (10% in 50 years), subduction in-slab'
# weight to apply to deaggregation values (enter manually)
deagg_weight_use = 0.2806

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
# Enter other parameters and estimate the unknown source and distance params
# -----------------------------------------------------------------------------

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
                                    fault_style='normal',  # 'reverse', #'normal',#'strike_slip', #
                                    F_hw=True)


# F_hw = True; site on the hanging wall, 'reverse'
# ({'M': 7.101391270500994,
#   'R_jb': 66.45838770293972,
#   'fault_style': 'reverse',
#   'F_hw': True},
#  {'lambda': array([90]),
#   'delta': array([40]),
#   'W': 20.024902993214635,
#   'Z_hyp': 9.819721745899802,
#   'Z_tor': array([2.09666603])},
#  {'alpha': array([50]),
#   'R_x': array([59.34847364]),
#   'R_rup': array([49.90462221])})

# F_hw = False; site on the foot wall, 'reverse'
# ({'M': 7.101391270500994,
#   'R_jb': 66.45838770293972,
#   'fault_style': 'reverse',
#   'F_hw': False},
#  {'lambda': array([90]),
#   'delta': array([40]),
#   'W': 20.024902993214635,
#   'Z_hyp': 9.819721745899802,
#   'Z_tor': array([2.09666603])},
#  {'alpha': array([-50]),
#   'R_x': array([-50.9100786]),
#   'R_rup': array([66.49145287])})

# -----------------------------------------------------------------------------
# Specify additional site parameters
# -----------------------------------------------------------------------------

vs30_seattle = 260  # assuming soil class D
# in (km), USGS NSHM2018 consolidated values for the site
z_2p5_seattle = 6.374
z_1p0_seattle = 0.876  # in (km) -- AS_2016 uses the value in km
# global region in the CB_2014 GMPE for active shallow crust
region_seattle_cb_2014 = 0
region_seattle_AG_2020 = "CAS"  # cascadia subduction zone region

z_1p0_seattle_as_2016 = z_1p0_seattle  # AS_2016 needs this value in km
# bahrampouri_et_al_2021 needs this value in (m)
z_1p0_seattle_bahrampouri_2021 = z_1p0_seattle*1000

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
                   Rrup=np.array([88.73893426]),  # src_dist_dict[2]['R_rup'],
                   # src_dist_dict[2]['R_x'], # value is negative, there is no difference here when positive value is used
                   Rx=np.array([59.33612014]),
                   Fhw=0,
                   # Zbot = None, # for CB_2014 needed only when W is unknown, so I leave this undefined (code checks if attribute exists)
                   W=src_dist_dict[1]['W'],
                   delta=src_dist_dict[1]['delta'],
                   lam=src_dist_dict[1]['lambda'],  # 1000,
                   # src_dist_dict[1]['Z_hyp'], #5,
                   hypo_depth=np.array([53.17]),
                   Ztor=np.array([46.13]),  # src_dist_dict[1]['Z_tor'], # 100
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
iml_im_star = 0.188  # conditioning value of the IM

# define the imt for the conditioning IM -- 8-story MRF
imt_star_cse = mhu.IntensityMeasureType('SaAvg', sa_period=np.array([0.3,
                                                                     0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
                                                                     1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
                                                                     2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9,
                                                                     4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0]))


# # define the imt for the conditioning IM -- 8-story MRF
# imt_star_cse = mhu.IntensityMeasureType( 'SaAvg', sa_period = np.array([1.0]) )


# define the im object (intensity measure) for IM_star -- specified above
im_star_cse = mhu.IntensityMeasure(iml=iml_im_star, imt=imt_star_cse,
                                   GMM_name=GMPE_to_use)

# Target periods used in the ground motion selections script by Prof. Baker;
# I manually copied the 30 periods computed in:
# selectionParams.TgtPer = logspace(log10(selectionParams.Tmin),log10(selectionParams.Tmax),30)
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
# shorter list so that my computation is faster in develpment
# period_tgt_list = [0.1, 0.5, 1, 3, 5, 10]


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
                                             imt=mhu.IntensityMeasureType(
                                                 'da5_75'),
                                             GMM_name=duration_GMPE_to_use))

# -----------------------------------------------------------------------------
# Compute conditional spectra given a single rupture (i.e., M and R definition)
# -----------------------------------------------------------------------------
cs_mean = mhu.ConditionalSpectra(rup_mean, site_cse,
                                 im_star_cse, im_cond_list_cse,
                                 mhu.sa_corr_baker)

# plot the CS target based on the mean M and R
cs_mean.plot_CS(legend_title=None, is_sample_CS=False, n_sample=25,
                is_plot_cs=True, is_plot_da5_75=True)

# deepcopy the computed CS target for subsequent plotting
cs_mean_inslab = copy.deepcopy(cs_mean)

# %% SEATTLE Site, for Hiro, 10in50 -- comparison of CS targets for different TRTs

sa_median_active = cs_mean_active.CS_target['mu_cond'][:-1]
sa_median_inslab = cs_mean_inslab.CS_target['mu_cond'][:-1]
sa_median_interface = cs_mean_interface.CS_target['mu_cond'][:-1]
periods_cs = cs_mean_active.CS_target['period_arr_cs']


plt.figure(dpi=200)
plt.plot(periods_cs, sa_median_active, '-k', linewidth=1.5,
         label='Crustal (35%)')

plt.plot(periods_cs, sa_median_inslab, '-.k', linewidth=1.5,
         label='In-slab (28%)')

plt.plot(periods_cs, sa_median_interface, '--k', linewidth=1.5,
         label='Interface (37%)')

plt.xlabel('Period (s)')
plt.ylabel('Sa(T) (g)')
plt.xlim(0.1, 10)
plt.ylim(0.01, 5)
plt.xscale('log')
plt.yscale('log')
plt.legend()
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
plt.plot(x_plt, y_plt_active, '-k', label='Crustal', linewidth=1.5)
plt.plot(x_plt, y_plt_inslab, '-.k', label='In-slab', linewidth=1.5)
plt.plot(x_plt, y_plt_interface, '--k', label='Interface', linewidth=1.5)

# plt.legend(loc = 'best')
plt.grid()
plt.xlabel('Da 5-75% (s)')
plt.ylabel('CDF')
plt.ylim(0, 1)
plt.xlim(1, 500)
plt.xscale('log')
plt.show()

# %% SEATTLE Site, for Hiro, 10in50 - save the CS targets

# folder_save = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/CS_targets/seattle_8_story_mrf_10in50/active_shallow_crust/'
# cs_mean_active.export_CS_target(folder_save)

# folder_save = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/CS_targets/seattle_8_story_mrf_10in50/interface/'
# cs_mean_interface.export_CS_target(folder_save)

# folder_save = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/CS_targets/seattle_8_story_mrf_10in50/inslab/'
# cs_mean_inslab.export_CS_target(folder_save)

# %% ---------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Intensity level: 2% in 50 years
# -----------------------------------------------------------------------------

# %% SEATTLE Site, for Hiro, 2in50 -- ACTIVE SHALLOW CRUST

# -----------------------------------------------------------------------------
# GMPE INFORMATION
# -----------------------------------------------------------------------------

# Sa GMPE: CB_2014
# Da5-75 GMPE: AS_2016
# Sa correlations: sa_corr_baker (Baker & Jayaram, 2008)
# Sa-Da correlations: Bradley (2011)

# -----------------------------------------------------------------------------
# Deaggregation file data :: 2in50
# -----------------------------------------------------------------------------

deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/2in50/calc_id_111_saAVG_active_TRT_vs30_260_AbrahamsonGulerce2020_cas_adjust_true_z1p0_876_z2p5_6p374_inslab_and_interface/Mag_Dist-mean-0_111_for_plot.csv'

prob_col_name = 'mean'  # column name containing P(X>x | M, R); here I assume
# only a single realization is used
delta_M = 0.2
delta_R = 5
investigation_time = 1
# title of the plot
deagg_name = 'Seattle, SaAvg(0.3 : 0.1: 5.0), RP = 2475 years (2% in 50 years), active shallow crust'
# weight to apply to deaggregation values (enter manually)
deagg_weight_use = 0.5522

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
#
# -----------------------------------------------------------------------------
# Enter other parameters and estimate the unknown source and distance params
# -----------------------------------------------------------------------------

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

# F_hw = True; site on the hanging wall
# ({'M': 6.920743084082292,
#   'R_jb': 5.566509565480057,
#   'fault_style': 'reverse',
#   'F_hw': True},
#  {'lambda': array([90]),
#   'delta': array([40]),
#   'W': 16.885139946167616,
#   'Z_hyp': 9.855851383183541,
#   'Z_tor': array([3.34371614])},
#  {'alpha': array([50]),
#   'R_x': array([6.63390777]),
#   'R_rup': array([8.80768065])})

# F_hw = False; site on the foot wall
# ({'M': 6.920743084082292,
#   'R_jb': 5.566509565480057,
#   'fault_style': 'reverse',
#   'F_hw': False},
#  {'lambda': array([90]),
#   'delta': array([40]),
#   'W': 16.885139946167616,
#   'Z_hyp': 9.855851383183541,
#   'Z_tor': array([3.34371614])},
#  {'alpha': array([-50]),
#   'R_x': array([-4.26419372]),
#   'R_rup': array([6.49357115])})

# Note: I selected the case for the site on the foot wall because the distances
#   are smaller in that case compared to assuming the site on the hanging wall.
# %%
# -----------------------------------------------------------------------------
# Specify additional site parameters
# -----------------------------------------------------------------------------

vs30_seattle = 260  # assuming soil class D
# in (km), USGS NSHM2018 consolidated values for the site
z_2p5_seattle = 6.374
z_1p0_seattle = 0.876  # in (km) -- AS_2016 uses the value in km
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
                   # value is negative, there is no difference here when positive value is used
                   Rx=src_dist_dict[2]['R_x'],
                   Fhw=0,
                   # Zbot = None, # for CB_2014 needed only when W is unknown, so I leave this undefined (code checks if attribute exists)
                   W=src_dist_dict[1]['W'],
                   delta=src_dist_dict[1]['delta'],
                   lam=src_dist_dict[1]['lambda'],  # 1000,
                   hypo_depth=src_dist_dict[1]['Z_hyp'],  # 5,
                   Ztor=src_dist_dict[1]['Z_tor'],
                   trt=None)  # Not used in the CB_2014 GMPE

# -----------------------------------------------------------------------------
# Define IM_star and the list of IM_cond for CS computations
# -----------------------------------------------------------------------------

# IM_star & IM_cond -- Sa, SaAvg
GMPE_to_use = 'CB_2014_active'
# iml_im_star = 0.400 # conditioning value of the IM, RotD50
iml_im_star = 0.477  # conditioning value of the IM, RotD100

# define the imt for the conditioning IM -- 8-story MRF
imt_star_cse = mhu.IntensityMeasureType('SaAvg', sa_period=np.array([0.3,
                                                                     0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
                                                                     1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
                                                                     2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9,
                                                                     4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0]))


# # define the imt for the conditioning IM -- 8-story MRF
# imt_star_cse = mhu.IntensityMeasureType( 'SaAvg', sa_period = np.array([1.0]) )


# define the im object (intensity measure) for IM_star -- specified above
im_star_cse = mhu.IntensityMeasure(iml=iml_im_star, imt=imt_star_cse,
                                   GMM_name=GMPE_to_use)

# Target periods used in the ground motion selections script by Prof. Baker;
# I manually copied the 30 periods computed in:
# selectionParams.TgtPer = logspace(log10(selectionParams.Tmin),log10(selectionParams.Tmax),30)
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
# shorter list so that my computation is faster in develpment
# period_tgt_list = [0.1, 0.5, 1, 3, 5, 10]


# Define the list of conditional IM, these are the IMs conditioned on the
# conditioning IM im_star
im_cond_list_cse = [mhu.IntensityMeasure(iml=None,
                    imt=mhu.IntensityMeasureType(
                        'Sa', sa_period=np.array([x])),
                    GMM_name=GMPE_to_use)
                    for x in period_tgt_list]

# IM_cond DURATION -- append duration IM
duration_GMPE_to_use = 'AS_2016_duration'  # ''
im_cond_list_cse.append(mhu.IntensityMeasure(iml=None,
                                             imt=mhu.IntensityMeasureType(
                                                 'da5_75'),
                                             GMM_name=duration_GMPE_to_use))

# -----------------------------------------------------------------------------
# Compute conditional spectra given a single rupture (i.e., M and R definition)
# -----------------------------------------------------------------------------
cs_mean = mhu.ConditionalSpectra(rup_mean, site_cse,
                                 im_star_cse, im_cond_list_cse,
                                 mhu.sa_corr_baker)

# plot the CS target based on the mean M and R
cs_mean.plot_CS(legend_title=None, is_sample_CS=False, n_sample=25,
                is_plot_cs=True, is_plot_da5_75=True)

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

deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/2in50/calc_id_114_saAVG_inslab_TRT_vs30_260_AbrahamsonGulerce2020_cas_adjust_true_z1p0_876_z2p5_6p374_inslab_and_interface/Mag_Dist-mean-0_114_for_plot.csv'

prob_col_name = 'mean'  # column name containing P(X>x | M, R); here I assume
# only a single realization is used
delta_M = 0.2
delta_R = 5
investigation_time = 1
# title of the plot
deagg_name = 'Seattle, SaAvg(0.3 : 0.1: 5.0), RP = 2475 years (2% in 50 years), subduction in-slab'
# weight to apply to deaggregation values (enter manually)
deagg_weight_use = 0.1170

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
# %%
# -----------------------------------------------------------------------------
# Enter other parameters and estimate the unknown source and distance params
# -----------------------------------------------------------------------------

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
                                    fault_style='normal',  # 'reverse', #'normal',#'strike_slip', #
                                    F_hw=True)

# F_hw = True; site on the hanging wall, 'normal'
# ({'M': 7.172487267935882,
#   'R_jb': 62.481741095404686,
#   'fault_style': 'normal',
#   'F_hw': True},
#  {'lambda': array([-90]),
#   'delta': array([50]),
#   'W': 23.46229784955982,
#   'Z_hyp': 9.805502546412823,
#   'Z_tor': 0},
#  {'alpha': array([50]),
#   'R_x': array([56.13425766]),
#   'R_rup': array([47.21511696])})

# F_hw = True; site on the hanging wall, 'reverse'
# ??

# F_hw = False; site on the foot wall, 'reverse'
# ??
# %%
# -----------------------------------------------------------------------------
# Specify additional site parameters
# -----------------------------------------------------------------------------

vs30_seattle = 260  # assuming soil class D
# in (km), USGS NSHM2018 consolidated values for the site
z_2p5_seattle = 6.374
z_1p0_seattle = 0.876  # in (km) -- AS_2016 uses the value in km
# global region in the CB_2014 GMPE for active shallow crust
region_seattle_cb_2014 = 0
region_seattle_AG_2020 = "CAS"  # cascadia subduction zone region

z_1p0_seattle_as_2016 = z_1p0_seattle  # AS_2016 needs this value in km
# bahrampouri_et_al_2021 needs this value in (m)
z_1p0_seattle_bahrampouri_2021 = z_1p0_seattle*1000

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
                   Rrup=np.array([86.58347057]),  # src_dist_dict[2]['R_rup'],
                   # src_dist_dict[2]['R_x'], # value is negative, there is no difference here when positive value is used
                   Rx=np.array([56.13215502]),
                   Fhw=0,
                   # Zbot = None, # for CB_2014 needed only when W is unknown, so I leave this undefined (code checks if attribute exists)
                   W=src_dist_dict[1]['W'],
                   delta=src_dist_dict[1]['delta'],
                   lam=src_dist_dict[1]['lambda'],  # 1000,
                   # src_dist_dict[1]['Z_hyp'], #5,
                   hypo_depth=np.array([53.17]),
                   Ztor=np.array([46.13]),  # src_dist_dict[1]['Z_tor'], # 100
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
# iml_im_star = 0.400 # conditioning value of the IM, RotD50
iml_im_star = 0.477  # conditioning value of the IM, RotD100

# define the imt for the conditioning IM -- 8-story MRF
imt_star_cse = mhu.IntensityMeasureType('SaAvg', sa_period=np.array([0.3,
                                                                     0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
                                                                     1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
                                                                     2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9,
                                                                     4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0]))


# # define the imt for the conditioning IM -- 8-story MRF
# imt_star_cse = mhu.IntensityMeasureType( 'SaAvg', sa_period = np.array([1.0]) )


# define the im object (intensity measure) for IM_star -- specified above
im_star_cse = mhu.IntensityMeasure(iml=iml_im_star, imt=imt_star_cse,
                                   GMM_name=GMPE_to_use)

# Target periods used in the ground motion selections script by Prof. Baker;
# I manually copied the 30 periods computed in:
# selectionParams.TgtPer = logspace(log10(selectionParams.Tmin),log10(selectionParams.Tmax),30)
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
# shorter list so that my computation is faster in develpment
# period_tgt_list = [0.1, 0.5, 1, 3, 5, 10]


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
                                             imt=mhu.IntensityMeasureType(
                                                 'da5_75'),
                                             GMM_name=duration_GMPE_to_use))

# -----------------------------------------------------------------------------
# Compute conditional spectra given a single rupture (i.e., M and R definition)
# -----------------------------------------------------------------------------
cs_mean = mhu.ConditionalSpectra(rup_mean, site_cse,
                                 im_star_cse, im_cond_list_cse,
                                 mhu.sa_corr_baker)

# plot the CS target based on the mean M and R
cs_mean.plot_CS(legend_title=None, is_sample_CS=False, n_sample=25,
                is_plot_cs=True, is_plot_da5_75=True)

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

deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/2in50/calc_id_113_saAVG_interface_TRT_vs30_260_AbrahamsonGulerce2020_cas_adjust_true_z1p0_876_z2p5_6p374_inslab_and_interface/Mag_Dist-mean-0_113_for_plot.csv'

prob_col_name = 'mean'  # column name containing P(X>x | M, R); here I assume
# only a single realization is used
delta_M = 0.2
delta_R = 5
investigation_time = 1
# title of the plot
deagg_name = 'Seattle, SaAvg(0.3 : 0.1: 5.0), RP = 2475 years (2% in 50 years), subduction interface'
# weight to apply to deaggregation values (enter manually)
deagg_weight_use = 0.3308

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
# Enter other parameters and estimate the unknown source and distance params
# -----------------------------------------------------------------------------

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

# F_hw = True; site on the hanging wall
# ({'M': 9.034859925226439,
#   'R_jb': 105.79929938826292,
#   'fault_style': 'reverse',
#   'F_hw': True},
#  {'lambda': array([90]),
#   'delta': array([40]),
#   'W': 124.24890481069198,
#   'Z_hyp': 9.433028014954711,
#   'Z_tor': 0},
#  {'alpha': array([50]),
#   'R_x': array([121.97577834]),
#   'R_rup': array([128.92926908])})

# F_hw = False; site on the foot wall
# ({'M': 9.034859925226439,
#   'R_jb': 105.79929938826292,
#   'fault_style': 'reverse',
#   'F_hw': False},
#  {'lambda': array([90]),
#   'delta': array([40]),
#   'W': 124.24890481069198,
#   'Z_hyp': 9.433028014954711,
#   'Z_tor': 0},
#  {'alpha': array([-50]),
#   'R_x': array([-81.04696538]),
#   'R_rup': array([105.79929939])})

# Note: I selected the case for the site on the foot wall because the distances
#   are smaller in that case compared to assuming the site on the hanging wall.
#   In this case the Sa values will be larger, but the durations will be
#   smaller.


# -----------------------------------------------------------------------------
# Specify additional site parameters
# -----------------------------------------------------------------------------

vs30_seattle = 260  # assuming soil class D
# in (km), USGS NSHM2018 consolidated values for the site
z_2p5_seattle = 6.374
z_1p0_seattle = 0.876  # in (km) -- AS_2016 uses the value in km
# global region in the CB_2014 GMPE for active shallow crust
region_seattle_cb_2014 = 0
region_seattle_AG_2020 = "CAS"  # cascadia subduction zone region

z_1p0_seattle_as_2016 = z_1p0_seattle  # AS_2016 needs this value in km
# bahrampouri_et_al_2021 needs this value in (m)
z_1p0_seattle_bahrampouri_2021 = z_1p0_seattle*1000

# -----------------------------------------------------------------------------
# Instantiate site and rupture objects
# -----------------------------------------------------------------------------

site_cse = mhu.Site(Vs30=vs30_seattle,
                    region=region_seattle_AG_2020,
                    Z25=z_2p5_seattle,  # AG_2020 needs this value in km
                    # Z10 = z_1p0_seattle_as_2016)
                    Z10=z_1p0_seattle_bahrampouri_2021)  # AS_2016 needs this value in km;
# bahrampouri_et_al_2021 needs this value in (m)

rup_mean = mhu.Rup(M=src_dist_dict[0]['M'],
                   R=None,  # not needed
                   Rjb=src_dist_dict[0]['R_jb'],
                   Rrup=src_dist_dict[2]['R_rup'],
                   # value is negative, there is no difference here when positive value is used
                   Rx=src_dist_dict[2]['R_x'],
                   Fhw=0,
                   # Zbot = None, # for CB_2014 needed only when W is unknown, so I leave this undefined (code checks if attribute exists)
                   W=src_dist_dict[1]['W'],
                   delta=src_dist_dict[1]['delta'],
                   lam=src_dist_dict[1]['lambda'],  # 1000,
                   hypo_depth=src_dist_dict[1]['Z_hyp'],  # 5,
                   Ztor=src_dist_dict[1]['Z_tor'],
                   trt='subduction_interface')  # Not used in the CB_2014 GMPE

# -----------------------------------------------------------------------------
# Define IM_star and the list of IM_cond for CS computations
# -----------------------------------------------------------------------------


# NOTE: the function always uses the below booleans by default, currently this
#       cannot be changed
# is_ergodic = False
# is_apply_usa_adjustment = True


# IM_star & IM_cond -- Sa, SaAvg
GMPE_to_use = 'AG_2020'
# iml_im_star = 0.400 # conditioning value of the IM, RotD50
iml_im_star = 0.477  # conditioning value of the IM, RotD100

# define the imt for the conditioning IM -- 8-story MRF
imt_star_cse = mhu.IntensityMeasureType('SaAvg', sa_period=np.array([0.3,
                                                                     0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
                                                                     1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
                                                                     2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9,
                                                                     4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0]))


# # define the imt for the conditioning IM -- 8-story MRF
# imt_star_cse = mhu.IntensityMeasureType( 'SaAvg', sa_period = np.array([1.0]) )


# define the im object (intensity measure) for IM_star -- specified above
im_star_cse = mhu.IntensityMeasure(iml=iml_im_star, imt=imt_star_cse,
                                   GMM_name=GMPE_to_use)

# Target periods used in the ground motion selections script by Prof. Baker;
# I manually copied the 30 periods computed in:
# selectionParams.TgtPer = logspace(log10(selectionParams.Tmin),log10(selectionParams.Tmax),30)
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
# shorter list so that my computation is faster in develpment
# period_tgt_list = [0.1, 0.5, 1, 3, 5, 10]


# Define the list of conditional IM, these are the IMs conditioned on the
# conditioning IM im_star
im_cond_list_cse = [mhu.IntensityMeasure(iml=None,
                    imt=mhu.IntensityMeasureType(
                        'Sa', sa_period=np.array([x])),
                    GMM_name=GMPE_to_use)
                    for x in period_tgt_list]

# IM_cond DURATION -- append duration IM

# duration_GMPE_to_use = 'AS_2016_duration'
duration_GMPE_to_use = 'BahrampouriEtAlSInter2020_duration'
im_cond_list_cse.append(mhu.IntensityMeasure(iml=None,
                                             imt=mhu.IntensityMeasureType(
                                                 'da5_75'),
                                             GMM_name=duration_GMPE_to_use))

# -----------------------------------------------------------------------------
# Compute conditional spectra given a single rupture (i.e., M and R definition)
# -----------------------------------------------------------------------------
cs_mean = mhu.ConditionalSpectra(rup_mean, site_cse,
                                 im_star_cse, im_cond_list_cse,
                                 mhu.sa_corr_baker)

# plot the CS target based on the mean M and R
cs_mean.plot_CS(legend_title=None, is_sample_CS=False, n_sample=25,
                is_plot_cs=True, is_plot_da5_75=True)

# deepcopy the computed CS target for subsequent plotting
cs_mean_interface = copy.deepcopy(cs_mean)

# %% SEATTLE Site, for Hiro, 2in50 -- comparison of CS targets for different TRTs

sa_median_active = cs_mean_active.CS_target['mu_cond'][:-1]
sa_median_inslab = cs_mean_inslab.CS_target['mu_cond'][:-1]
sa_median_interface = cs_mean_interface.CS_target['mu_cond'][:-1]
periods_cs = cs_mean_active.CS_target['period_arr_cs']


plt.figure(dpi=200)
plt.plot(periods_cs, sa_median_active, '-k', linewidth=1.5,
         label='Crustal (55%)')

plt.plot(periods_cs, sa_median_inslab, '-.k', linewidth=1.5,
         label='In-slab (12%)')

plt.plot(periods_cs, sa_median_interface, '--k', linewidth=1.5,
         label='Interface (33%)')

plt.xlabel('Period (s)')
plt.ylabel('Sa(T) (g)')
plt.xlim(0.1, 10)
plt.ylim(0.01, 5)
plt.xscale('log')
plt.yscale('log')
plt.legend()
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
plt.plot(x_plt, y_plt_active, '-k', label='Crustal', linewidth=1.5)
plt.plot(x_plt, y_plt_inslab, '-.k', label='In-slab', linewidth=1.5)
plt.plot(x_plt, y_plt_interface, '--k', label='Interface', linewidth=1.5)

# plt.legend(loc = 'best')
plt.grid()
plt.xlabel('Da 5-75% (s)')
plt.ylabel('CDF')
plt.ylim(0, 1)
plt.xlim(1, 500)
plt.xscale('log')
plt.show()

# %% SEATTLE Site, for Hiro, 2in50 - save the CS targets

# folder_save = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/CS_targets/seattle_8_story_mrf_2in50/active_shallow_crust/'
# cs_mean_active.export_CS_target(folder_save)

# folder_save = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/CS_targets/seattle_8_story_mrf_2in50/interface/'
# cs_mean_interface.export_CS_target(folder_save)

# folder_save = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/CS_targets/seattle_8_story_mrf_2in50/inslab/'
# cs_mean_inslab.export_CS_target(folder_save)
