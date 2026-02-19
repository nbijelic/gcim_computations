#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 15:32:18 2023

@author: nbijelic
"""

#%% Plots for project with Hiro

import numpy as np
import pandas as pd

### plotting
import matplotlib.pyplot as plt
# change font globally
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams.update({'font.size': 12})
# plotly
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
import seaborn as sns

# Function to parse the hazard curves
def parse_haz_curve_csv(file):
    df = pd.read_csv(file, header= None, skiprows=1)
    df = df.iloc[:,3:]
    iml_haz = df.iloc[0,:]
    iml_haz = np.array([ float(x[4:]) for x in iml_haz ])
    poe_haz = df.iloc[1,:].to_numpy()
    poe_haz = np.array([ float(x) for x in poe_haz ])
    return {'iml_haz': iml_haz, 'poe_haz': poe_haz} 


#%% Plot hazard curves:
# Study for Hiro
# Site: LA
# Lat/Lon: 34.0000 / -118.1500
# 4-story frame, sa_avg [0.19s, 2.85s]    

# Here I manually enter the poe values form the hazard curve csv file
# test below with the xml parser

im_name = 'SaAvg(0.19s : 0.1s : 2.85s) (g)'

iml = np.array([5.000000000E-03, 7.108274886E-03, 1.010551437E-02, 1.436655480E-02,
                2.042428414E-02, 2.903628520E-02, 4.127957938E-02, 5.868531948E-02,
                8.343027653E-02, 1.186090679E-01, 1.686211717E-01, 2.397211280E-01,
                3.408007348E-01, 4.845010608E-01, 6.887933446E-01, 9.792264866E-01,
                1.392122208E+00, 1.979117467E+00, 2.813622197E+00, 4.000000000E+00])

# from calc_id_4 - hazard computation, active shallow crust only, max_depth = 200
# poe_site_1 = np.array([
#     3.503886E-01, 2.874884E-01, 2.286774E-01, 1.764676E-01, 
#     1.320331E-01, 9.535030E-02, 6.596622E-02, 4.355079E-02, 
#     2.759255E-02, 1.693591E-02, 1.006441E-02, 5.712211E-03, 
#     3.040552E-03, 1.483396E-03, 6.400049E-04, 2.329350E-04, 
#     6.830692E-05, 1.551211E-05, 2.652407E-06, 3.576279E-07
#     ])

# # from calc_id_5 - deagg hazard computation, active shallow crust only, max_depth = 200
poe_site_1 = np.array([
    3.503886E-01, 2.874884E-01, 2.286774E-01, 1.764676E-01, 
    1.320331E-01, 9.535030E-02, 6.596622E-02, 4.355079E-02, 
    2.759255E-02, 1.693591E-02, 1.006441E-02, 5.712211E-03, 
    3.040552E-03, 1.483396E-03, 6.400049E-04, 2.329350E-04, 
    6.830692E-05, 1.551211E-05, 2.652407E-06, 3.576279E-07
    ])

# from calc_id_6 - deagg hazard computation, 
# active shallow crust only, max_depth = 300
poe_site_2 = np.array([
    4.164610E-01, 3.358026E-01, 2.609208E-01, 1.957966E-01,
    1.422057E-01, 9.990500E-02, 6.760797E-02, 4.394704E-02, 
    2.757284E-02, 1.683588E-02, 9.988397E-03, 5.673483E-03, 
    3.025755E-03, 1.479089E-03, 6.389916E-04, 2.327710E-04, 
    6.824732E-05, 1.551211E-05, 2.652407E-06, 3.576279E-07        
    ])

name_site_1 = 'LA, active shallow crust, max dist = 200km'
name_site_2 = 'LA, active shallow crust, max dist = 300km'


T_years = 1
lam_rate_site_1 = -np.log(1 - poe_site_1)/T_years
lam_rate_site_2 = -np.log(1 - poe_site_2)/T_years


plt.figure(dpi = 200)
plt.plot(iml, lam_rate_site_1, '-or', label = name_site_1)
plt.plot(iml, lam_rate_site_2, ':xm', label = name_site_2)

plt.plot([0.005, 4], [1/2475, 1/2475], '--k')

# plt.plot([0.857220137962931], [1/2475], 'xb')
# plt.plot([8.08394E-01], [1/2475], 'xb')
# plt.plot([0.808458126355189], [1/2475], 'xb')


plt.ylim([1E-6, 1])
# plt.xlim([0.01, 4])
plt.xlim([0.005, 4])
plt.grid()
plt.legend()
plt.ylabel(r'$\lambda (X>x)$')
plt.xlabel(im_name)
plt.xscale('log')
plt.yscale('log')
plt.show()

## This is how to obtain the iml at a desired exceedance rate
# np.interp(1/475, np.flip(lam_rate_site_1), np.flip(iml)) :: for 10in50
# The one above is not good, interpolate in log-log
# np.exp(np.interp( np.log(1/2475), np.log(np.flip(lam_rate_site_1)), np.log(np.flip(iml))))
# sa_avg conditional value interpolated from the hazard curve:
# - calc_id_5 (max_dist = 200km): 0.808458126355189 g  
# - calc_id_6 (max_dist = 300km): 0.808124754912721 g

#%% Comparison of SA(T) hazard curves: active shallow crust TRT only vs ALL TRTs

def parse_haz_curve_csv(file):
    df = pd.read_csv(file, header= None, skiprows=1)
    df = df.iloc[:,3:]
    iml_haz = df.iloc[0,:]
    iml_haz = np.array([ float(x[4:]) for x in iml_haz ])
    poe_haz = df.iloc[1,:].to_numpy()
    poe_haz = np.array([ float(x) for x in poe_haz ])
    return {'iml_haz': iml_haz, 'poe_haz': poe_haz} 


# # -- Sa(0.2)
# im_name = 'Sa(0.2) (g)'
# file_active_trt = '/Users/nbijelic/Research/EPFL/projects/Hiro/oq_hazard_results/LA_site_4s_frame/calc_id_11_sa_active_trt/hazard_curve-mean-SA(0.2)_11.csv'
# file_all_trt = '/Users/nbijelic/Research/EPFL/projects/Hiro/oq_hazard_results/LA_site_4s_frame/calc_id_12_all_trt/hazard_curve-mean-SA(0.2)_12.csv'

# # -- Sa(0.3)
# im_name = 'Sa(0.3) (g)'
# file_active_trt = '/Users/nbijelic/Research/EPFL/projects/Hiro/oq_hazard_results/LA_site_8s_frame/calc_id_11_sa_active_trt/hazard_curve-mean-SA(0.3)_11.csv'
# file_all_trt = '/Users/nbijelic/Research/EPFL/projects/Hiro/oq_hazard_results/LA_site_4s_frame/calc_id_12_all_trt/hazard_curve-mean-SA(0.3)_12.csv'

# # -- Sa(0.6)
# im_name = 'Sa(0.6) (g)'
# file_active_trt = '/Users/nbijelic/Research/EPFL/projects/Hiro/oq_hazard_results/LA_site_8s_frame/calc_id_11_sa_active_trt/hazard_curve-mean-SA(0.6)_11.csv'
# file_all_trt = '/Users/nbijelic/Research/EPFL/projects/Hiro/oq_hazard_results/LA_site_4s_frame/calc_id_12_all_trt/hazard_curve-mean-SA(0.6)_12.csv'

# # -- Sa(1.0)
# im_name = 'Sa(1.0) (g)'
# file_active_trt = '/Users/nbijelic/Research/EPFL/projects/Hiro/oq_hazard_results/LA_site_8s_frame/calc_id_11_sa_active_trt/hazard_curve-mean-SA(1.0)_11.csv'
# file_all_trt = '/Users/nbijelic/Research/EPFL/projects/Hiro/oq_hazard_results/LA_site_4s_frame/calc_id_12_all_trt/hazard_curve-mean-SA(1.0)_12.csv'


# # -- Sa(2.0)
# im_name = 'Sa(2.0) (g)'
# file_active_trt = '/Users/nbijelic/Research/EPFL/projects/Hiro/oq_hazard_results/LA_site_8s_frame/calc_id_11_sa_active_trt/hazard_curve-mean-SA(2.0)_11.csv'
# file_all_trt = '/Users/nbijelic/Research/EPFL/projects/Hiro/oq_hazard_results/LA_site_4s_frame/calc_id_12_all_trt/hazard_curve-mean-SA(2.0)_12.csv'


# # -- Sa(2.7)
# im_name = 'Sa(2.7) (g)'
# file_active_trt = '/Users/nbijelic/Research/EPFL/projects/Hiro/oq_hazard_results/LA_site_4s_frame/calc_id_14_sa_active_trt/hazard_curve-mean-SA(2.7)_14.csv'
# file_all_trt = '/Users/nbijelic/Research/EPFL/projects/Hiro/oq_hazard_results/LA_site_4s_frame/calc_id_15_sa_all_trt/hazard_curve-mean-SA(2.7)_15.csv'


# # -- Sa(3.0)
# im_name = 'Sa(3.0) (g)'
# file_active_trt = '/Users/nbijelic/Research/EPFL/projects/Hiro/oq_hazard_results/LA_site_8s_frame/calc_id_14_sa_active_trt/hazard_curve-mean-SA(3.0)_14.csv'
# file_all_trt = '/Users/nbijelic/Research/EPFL/projects/Hiro/oq_hazard_results/LA_site_4s_frame/calc_id_15_sa_all_trt/hazard_curve-mean-SA(3.0)_15.csv'


# # -- Sa(4.0)
# im_name = 'Sa(4.0) (g)'
# file_active_trt = '/Users/nbijelic/Research/EPFL/projects/Hiro/oq_hazard_results/LA_site_8s_frame/calc_id_14_sa_active_trt/hazard_curve-mean-SA(4.0)_14.csv'
# file_all_trt = '/Users/nbijelic/Research/EPFL/projects/Hiro/oq_hazard_results/LA_site_4s_frame/calc_id_15_sa_all_trt/hazard_curve-mean-SA(4.0)_15.csv'


# # -- Sa(5.0)
# im_name = 'Sa(5.0) (g)'
# file_active_trt = '/Users/nbijelic/Research/EPFL/projects/Hiro/oq_hazard_results/LA_site_8s_frame/calc_id_14_sa_active_trt/hazard_curve-mean-SA(5.0)_14.csv'
# file_all_trt = '/Users/nbijelic/Research/EPFL/projects/Hiro/oq_hazard_results/LA_site_4s_frame/calc_id_15_sa_all_trt/hazard_curve-mean-SA(5.0)_15.csv'

# label_1 = 'LA, active shallow crust TRT'
# label_2 = 'LA, all TRTs'

label_1 = 'LA'


# # FINAL HAZARD CURVE FOR HIRO, LA site, 8-story MRF
# # -- 'SaAvg(0.3s : 0.1s : 5.0s) (g)'
im_name = 'SaAvg(0.3s : 0.1s : 5.0s) (g)'
file_active_trt = '/Users/nbijelic/Research/EPFL/projects/Hiro/oq_hazard_results/LA_site_8s_frame/calc_id_13_FINAL_sa_avg_hiro_LA_8_story_mrf/hazard_curve-mean-AvgSA_13.csv'
# file_all_trt = '/Users/nbijelic/Research/EPFL/projects/Hiro/oq_hazard_results/LA_site_4s_frame/calc_id_16/hazard_curve-mean-AvgSA_16.csv'
# label_1 = 'LA, pointsource_distance = 100 km'
# label_2 = 'LA, pointsource_distance = 0 km'


# -----------------------------------------------------------------------------
# Hazard curves - NIKOS
# -----------------------------------------------------------------------------
# # 4-story EBF 
# im_name = 'SaAvg(0.2s : 0.1s : 3.3s) (g)'
# file_active_trt = '/Users/nbijelic/Research/EPFL/projects/Nikos/oq_hazard_results/calc_id_21_nikos_4s_ebf_la/hazard_curve-mean-AvgSA_21.csv'

# # 8-story EBF 
# im_name = 'SaAvg(0.5s : 0.1s : 7.5s) (g)'
# file_active_trt = '/Users/nbijelic/Research/EPFL/projects/Nikos/oq_hazard_results/calc_id_24_nikos_8s_ebf_la_deagg_PSHA_imls/hazard_curve-mean-AvgSA_24.csv'




# parse hazard curves - iml and poe values
haz_crv_1 = parse_haz_curve_csv(file_active_trt) # haz_crv_active_trt
# haz_crv_2 = parse_haz_curve_csv(file_all_trt) # haz_crv_all_trt
    
# convert to exceedance rates    
T_years = 1
lam_rate_1 = -np.log(1 - haz_crv_1['poe_haz'])/T_years
# lam_rate_2 = -np.log(1 - haz_crv_2['poe_haz'])/T_years

# compute iml at specified return periods
iml_2in50 = np.exp(np.interp( np.log(1/2475), np.log(np.flip(lam_rate_1)), np.log(np.flip(haz_crv_1['iml_haz']))))
iml_10in50 = np.exp(np.interp( np.log(1/475), np.log(np.flip(lam_rate_1)), np.log(np.flip(haz_crv_1['iml_haz']))))


# change font globally
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams.update({'font.size': 16})


plt.figure(dpi = 400)
plt.plot(haz_crv_1['iml_haz'], lam_rate_1, '-or', label = label_1)
# plt.plot(haz_crv_2['iml_haz'], lam_rate_2, ':xm', label = label_2)
# add MCE hazard level
plt.plot([0.005, 8], [1/2475, 1/2475], '--k')
plt.plot([iml_2in50], [1/2475], 'xk')
# plt.plot([0.005, haz_crv_1['iml_haz'].max()], [1/2475, 1/2475], 'xk')


# for sa_avg @ 2in50 - HIRO
# plt.plot([0.4662412816357129], [1/2475], 'xb')

# # for sa_avg from MCEr - HIRO
plt.plot([0.005, 8], [1/9475, 1/9475], ':k')
plt.plot([0.7168748200072746], [1/9475], 'xk')

# # Hazard levels -- NIKOS
# plt.plot([0.005, 8], [1/2475, 1/2475], '--k')
# plt.plot([iml_2in50], [1/2475], 'xk')

# plt.plot([0.005, 8], [1/475, 1/475], '-.k')
# plt.plot([iml_10in50], [1/475], 'xk')

# plt.ylim([1E-5, 5])

plt.ylim([1E-5, 1])
plt.xlim([0.005, 8])

# # plt.xlim([0.01, 4])
# plt.xlim([0.005, 8])
# # plt.xlim([0.005, haz_crv_1['iml_haz'].max()])


plt.grid()
plt.legend()
plt.ylabel(r'$\lambda (X>x)$')
plt.xlabel(im_name)
plt.xscale('log')
plt.yscale('log')
plt.show()

## This is how to obtain the iml at a desired exceedance rate
# np.interp(1/475, np.flip(lam_rate_site_1), np.flip(iml)) :: for 10in50
# The one above is not good, interpolate in log-log
# np.exp(np.interp( np.log(1/2475), np.log(np.flip(lam_rate_site_1)), np.log(np.flip(iml))))

# 10in50 Sa(T) values for different periods
# Periods: 5, 4, 3, 2, 1, 0.6, 0.3
# Sa: 0.10278052086900909, 0.13947986898459977, 0.20241521324682352, 0.32669402578259005, 0.6592708671423725, 0.9551589706051304, 1.1781609947817522

#%% SEATTLE Site, HIRO

def parse_haz_curve_csv(file):
    df = pd.read_csv(file, header= None, skiprows=1)
    df = df.iloc[:,3:]
    iml_haz = df.iloc[0,:]
    iml_haz = np.array([ float(x[4:]) for x in iml_haz ])
    poe_haz = df.iloc[1,:].to_numpy()
    poe_haz = np.array([ float(x) for x in poe_haz ])
    return {'iml_haz': iml_haz, 'poe_haz': poe_haz} 


# # -- Sa(0.2)
# im_name = 'Sa(0.2) (g)'
# file_active_trt = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/calc_id_40_all_TRTs_finer_M-R_grid/hazard_curve-mean-SA(0.2)_40.csv'
# file_all_trt = '/Users/nbijelic/Research/EPFL/projects/Hiro/oq_hazard_results/LA_site_4s_frame/calc_id_12_all_trt/hazard_curve-mean-SA(0.2)_12.csv'

# # -- Sa(0.3)
# im_name = 'Sa(0.3) (g)'
# file_active_trt = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/calc_id_72_all_TRTs_Sa/hazard_curve-mean-SA(0.3)_72.csv'
# file_active_trt = '/Users/nbijelic/Research/EPFL/projects/Hiro/oq_hazard_results/LA_site_8s_frame/calc_id_11_sa_active_trt/hazard_curve-mean-SA(0.3)_11.csv'
# file_all_trt = '/Users/nbijelic/Research/EPFL/projects/Hiro/oq_hazard_results/LA_site_4s_frame/calc_id_12_all_trt/hazard_curve-mean-SA(0.3)_12.csv'


# # # -- Sa(1.0)
# im_name = 'Sa(1.0) (g)'
# file_active_trt = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/calc_id_72_all_TRTs_Sa/hazard_curve-mean-SA(1.0)_72.csv'
# file_active_trt = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/calc_id_40_all_TRTs_finer_M-R_grid/hazard_curve-mean-SA(1.0)_40.csv'
# # file_all_trt = '/Users/nbijelic/Research/EPFL/projects/Hiro/oq_hazard_results/LA_site_4s_frame/calc_id_12_all_trt/hazard_curve-mean-SA(1.0)_12.csv'


# # -- Sa(2.0)
# im_name = 'Sa(2.0) (g)'
# file_active_trt = '/Users/nbijelic/Research/EPFL/projects/Hiro/oq_hazard_results/LA_site_8s_frame/calc_id_11_sa_active_trt/hazard_curve-mean-SA(2.0)_11.csv'
# file_all_trt = '/Users/nbijelic/Research/EPFL/projects/Hiro/oq_hazard_results/LA_site_4s_frame/calc_id_12_all_trt/hazard_curve-mean-SA(2.0)_12.csv'


# # -- Sa(3.0)
# im_name = 'Sa(3.0) (g)'
# file_active_trt = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/calc_id_72_all_TRTs_Sa/hazard_curve-mean-SA(3.0)_72.csv'
# file_active_trt = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/calc_id_40_all_TRTs_finer_M-R_grid/hazard_curve-mean-SA(3.0)_40.csv'
# file_active_trt = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/calc_id_85_Sa_only_no_deagg_ps_grid_spacing_0p0/hazard_curve-mean-SA(3.0)_85.csv'
# file_all_trt = '/Users/nbijelic/Research/EPFL/projects/Hiro/oq_hazard_results/LA_site_4s_frame/calc_id_15_sa_all_trt/hazard_curve-mean-SA(3.0)_15.csv'

# # # -- SaAvg(0.3: 0.1: 5.0)
im_name = 'SaAvg(0.3s : 0.1s : 5.0s) (g)'
# # file_active_trt = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/SaAVG_0p3-0p1-5p0/calc_id_53_saAvg_all_trt_deagg_at_2in50/hazard_curve-mean-AvgSA_53.csv'
# # file_all_trt = '/Users/nbijelic/Research/EPFL/projects/Hiro/oq_hazard_results/LA_site_4s_frame/calc_id_12_all_trt/hazard_curve-mean-SA(1.0)_12.csv'
file_all_trt = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_187_saAvg_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-AvgSA_187.csv'

# # # -- SaAvg(0.2: 0.1: 3.0)
# im_name = 'SaAvg(0.2s : 0.1s : 3.0s) (g)'
# file_all_trt = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/4-story_mrf/calc_id_198_saAvg_all_trt_4_story_mrf_2in50/hazard_curve-mean-AvgSA_198.csv'


# label_1 = 'LA, active shallow crust TRT'
# label_2 = 'LA, all TRTs'

label_2 = 'all TRTs'
legend_title = 'Seattle'

# parse hazard curves - iml and poe values
# haz_crv_1 = parse_haz_curve_csv(file_active_trt) # haz_crv_active_trt
haz_crv_2 = parse_haz_curve_csv(file_all_trt) # haz_crv_all_trt
    
# convert to exceedance rates    
T_years = 1
# lam_rate_1 = -np.log(1 - haz_crv_1['poe_haz'])/T_years
lam_rate_2 = -np.log(1 - haz_crv_2['poe_haz'])/T_years

# compute iml at specified return periods
# iml_2in50 = np.exp(np.interp( np.log(1/2475), np.log(np.flip(lam_rate_1)), np.log(np.flip(haz_crv_1['iml_haz']))))
# iml_10in50 = np.exp(np.interp( np.log(1/475), np.log(np.flip(lam_rate_1)), np.log(np.flip(haz_crv_1['iml_haz']))))
iml_2in50 = np.exp(np.interp( np.log(1/2475), np.log(np.flip(lam_rate_2)), np.log(np.flip(haz_crv_2['iml_haz']))))
iml_10in50 = np.exp(np.interp( np.log(1/475), np.log(np.flip(lam_rate_2)), np.log(np.flip(haz_crv_2['iml_haz']))))

# change font globally
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams.update({'font.size': 16})


plt.figure(dpi = 400)
# plt.plot(haz_crv_1['iml_haz'], lam_rate_1, '-or', label = label_1)
# plt.plot(haz_crv_2['iml_haz'], lam_rate_2, ':xm', label = label_2)
plt.plot(haz_crv_2['iml_haz'], lam_rate_2, '-k', label = label_2)

# add MCE hazard level
plt.plot([0.005, 10], [1/2475, 1/2475], '--k')
plt.plot([iml_2in50], [1/2475], 'xk')
# plt.plot([0.005, haz_crv_1['iml_haz'].max()], [1/2475, 1/2475], 'xk')

# add 10in50 level
plt.plot([0.005, 10], [1/475, 1/475], ':k')
plt.plot([iml_10in50], [1/475], 'xk')

# plt.ylim([1E-5, 5])

plt.ylim([1E-5, 1])
plt.xlim([0.005, 10])

# # plt.xlim([0.01, 4])
# plt.xlim([0.005, 8])
# # plt.xlim([0.005, haz_crv_1['iml_haz'].max()])

plt.grid()
plt.legend(title = legend_title)
plt.ylabel(r'$\lambda (X>x)$')
plt.xlabel(im_name)
plt.xscale('log')
plt.yscale('log')
plt.show()

#%%
iml_10in50 = np.exp(np.interp( np.log(1/1750), np.log(np.flip(lam_rate_2)), np.log(np.flip(haz_crv_2['iml_haz']))))

#%% SEATTLE Site, HIRO -- all curves on a single plot

# List of im_names and files to use

im_name_lst = ['SA(0.2)', 'SA(0.3)', 'SA(0.5)', 'SA(1.0)', 'SA(2.0)', 'SA(3.0)' ]
# calc_id = '108'
# path_to_calc = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/calc_id_108_sa_all_TRT_sa_vs30_260_AbrahamsonGulerce2020_cas_adjust_true_z1p0_1100_z2p5_7p000_inslab_and_interface/'

# calc_id = '85'
# path_to_calc = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/calc_id_85_Sa_only_no_deagg_ps_grid_spacing_0p0/'

# calc_id = '94'
# path_to_calc = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/calc_id_94_all_TRT_sa_vs30_260_AbrahamsonGulerce2020_cas_adjust_true_z1p0_876_z2p5_6p374/'

calc_id = '109'
path_to_calc = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/calc_id_109_sa_all_TRT_sa_vs30_260_AbrahamsonGulerce2020_cas_adjust_true_z1p0_876_z2p5_6p374_inslab_and_interface/'


T_years = 1 # period for which the hazard curves were computed for

# loop over files and parse hazard curves to dict
haz_crv_dict={}
for im_name in im_name_lst:
    file_curr = f'hazard_curve-mean-{im_name}_{calc_id}.csv'
    haz_crv_dict[im_name] = parse_haz_curve_csv(path_to_calc+file_curr)
    # compute exceedance rates and add to the dict
    haz_crv_dict[im_name]['lambda_haz'] =  -np.log(1 - haz_crv_dict[im_name]['poe_haz'])/T_years



# plot the hazard curves -- exceedance rates
site_name = 'Seattle'

# change font globally
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams.update({'font.size': 12})

plt.figure(dpi = 200)
for im_name in haz_crv_dict.keys():
    plt.plot(haz_crv_dict[im_name]['iml_haz'], haz_crv_dict[im_name]['lambda_haz'],
             '--', label = im_name, linewidth = 1.0)
# add the 2in50 hazard level
plt.plot([0.005, 10], [1/2475, 1/2475], '--k')

plt.ylim([1E-5, 1])
plt.xlim([0.005, 10])


plt.grid()
plt.legend(title = site_name)
plt.ylabel(r'$\lambda (X>x)$')
plt.xlabel('Sa(T) (g)')
plt.xscale('log')
plt.yscale('log')
plt.show()

#%% SEATTLE Site, HIRO, SaAvg final -- all curves on a single plot

# List of im_names and files to use

# im_name_lst = ['all TRTs', 'active shallow crust', 'in-slab', 'interface'  ]

# haz_crv_file_list = [
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/finalized/calc_id_110_saAVG_all_TRT_vs30_260_AbrahamsonGulerce2020_cas_adjust_true_z1p0_876_z2p5_6p374_inslab_and_interface/hazard_curve-mean-AvgSA_110.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/finalized/calc_id_111_saAVG_active_TRT_vs30_260_AbrahamsonGulerce2020_cas_adjust_true_z1p0_876_z2p5_6p374_inslab_and_interface/hazard_curve-mean-AvgSA_111.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/finalized/calc_id_114_saAVG_inslab_TRT_vs30_260_AbrahamsonGulerce2020_cas_adjust_true_z1p0_876_z2p5_6p374_inslab_and_interface/hazard_curve-mean-AvgSA_114.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/finalized/calc_id_113_saAVG_interface_TRT_vs30_260_AbrahamsonGulerce2020_cas_adjust_true_z1p0_876_z2p5_6p374_inslab_and_interface/hazard_curve-mean-AvgSA_113.csv'
#     ]

im_name_lst = ['all TRTs']

haz_crv_file_list = [
    # '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/finalized/calc_id_110_saAVG_all_TRT_vs30_260_AbrahamsonGulerce2020_cas_adjust_true_z1p0_876_z2p5_6p374_inslab_and_interface/hazard_curve-mean-AvgSA_110.csv',
    '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/10in50/calc_id_115_10in50_saAVG_all_TRT_vs30_260_AbrahamsonGulerce2020_cas_adjust_true_z1p0_876_z2p5_6p374_inslab_and_interface/hazard_curve-mean-AvgSA_115.csv'
    ]

color_lst = ['k']
linestyle_lst = ['-']

T_years = 1 # period for which the hazard curves were computed for

# loop over files and parse hazard curves to dict
haz_crv_dict={}
for im_name, file_haz in zip(im_name_lst, haz_crv_file_list):
    file_curr = file_haz
    haz_crv_dict[im_name] = parse_haz_curve_csv(file_curr)
    # compute exceedance rates and add to the dict
    haz_crv_dict[im_name]['lambda_haz'] =  -np.log(1 - haz_crv_dict[im_name]['poe_haz'])/T_years



# plot the hazard curves -- exceedance rates
site_name = 'Seattle'

# change font globally
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams.update({'font.size': 12})

plt.figure(dpi = 200)
for im_name, color_plt, linestyle_plt in zip(haz_crv_dict.keys(), color_lst, linestyle_lst):
    plt.plot(haz_crv_dict[im_name]['iml_haz'], haz_crv_dict[im_name]['lambda_haz'],
             linestyle = linestyle_plt, label = im_name, linewidth = 1.0, color = color_plt)
# # add the 2in50 hazard level
# plt.plot([0.005, 10], [1/2475, 1/2475], '--k')

# add the 10in50 hazard level
plt.plot([0.005, 10], [1/475, 1/475], '--k')

plt.ylim([1E-5, 1])
plt.xlim([0.005, 10])


plt.grid()
plt.legend(title = site_name)
plt.ylabel(r'$\lambda (X>x)$')
plt.xlabel('SaAvg(0.3s : 0.1s : 5.0s) (g)')
plt.xscale('log')
plt.yscale('log')
plt.show()

# iml_2in50 = np.exp(np.interp( np.log(1/2475), np.log(np.flip(haz_crv_dict[im_name]['lambda_haz'])), np.log(np.flip(haz_crv_dict[im_name]['iml_haz']))))

#%% SEATTLE Site, HIRO, SaAvg final :: tests with the scaling factors

# List of im_names and files to use

# im_name_lst = ['all TRTs', 'active shallow crust', 'in-slab', 'interface'  ]

# haz_crv_file_list = [
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/finalized/calc_id_110_saAVG_all_TRT_vs30_260_AbrahamsonGulerce2020_cas_adjust_true_z1p0_876_z2p5_6p374_inslab_and_interface/hazard_curve-mean-AvgSA_110.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/finalized/calc_id_111_saAVG_active_TRT_vs30_260_AbrahamsonGulerce2020_cas_adjust_true_z1p0_876_z2p5_6p374_inslab_and_interface/hazard_curve-mean-AvgSA_111.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/finalized/calc_id_114_saAVG_inslab_TRT_vs30_260_AbrahamsonGulerce2020_cas_adjust_true_z1p0_876_z2p5_6p374_inslab_and_interface/hazard_curve-mean-AvgSA_114.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/finalized/calc_id_113_saAVG_interface_TRT_vs30_260_AbrahamsonGulerce2020_cas_adjust_true_z1p0_876_z2p5_6p374_inslab_and_interface/hazard_curve-mean-AvgSA_113.csv'
#     ]

im_name_lst = ['all TRTs']

haz_crv_file_list = [
    # '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/finalized/calc_id_110_saAVG_all_TRT_vs30_260_AbrahamsonGulerce2020_cas_adjust_true_z1p0_876_z2p5_6p374_inslab_and_interface/hazard_curve-mean-AvgSA_110.csv',
    # '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/10in50/calc_id_115_10in50_saAVG_all_TRT_vs30_260_AbrahamsonGulerce2020_cas_adjust_true_z1p0_876_z2p5_6p374_inslab_and_interface/hazard_curve-mean-AvgSA_115.csv',
    # '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/test_scaling_factors/calc_id_160_saAvg_all_trt_RotD100/hazard_curve-mean-AvgSA_160.csv'
    '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/test_scaling_factors/calc_id_163_sa_all_trt_z1p0_876_z2p5_6p374_RotD100/hazard_curve-mean-SA(0.1)_163.csv'
    ]

color_lst = ['k']
linestyle_lst = ['-']

T_years = 1 # period for which the hazard curves were computed for

# loop over files and parse hazard curves to dict
haz_crv_dict={}
for im_name, file_haz in zip(im_name_lst, haz_crv_file_list):
    file_curr = file_haz
    haz_crv_dict[im_name] = parse_haz_curve_csv(file_curr)
    # compute exceedance rates and add to the dict
    haz_crv_dict[im_name]['lambda_haz'] =  -np.log(1 - haz_crv_dict[im_name]['poe_haz'])/T_years



# plot the hazard curves -- exceedance rates
site_name = 'Seattle'

# change font globally
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams.update({'font.size': 12})

plt.figure(dpi = 200)
for im_name, color_plt, linestyle_plt in zip(haz_crv_dict.keys(), color_lst, linestyle_lst):
    plt.plot(haz_crv_dict[im_name]['iml_haz'], haz_crv_dict[im_name]['lambda_haz'],
             linestyle = linestyle_plt, label = im_name, linewidth = 1.0, color = color_plt)
# # add the 2in50 hazard level
# plt.plot([0.005, 10], [1/2475, 1/2475], '--k')

# add the 10in50 hazard level
plt.plot([0.005, 10], [1/475, 1/475], '--k')

plt.ylim([1E-5, 1])
plt.xlim([0.005, 10])


plt.grid()
plt.legend(title = site_name)
plt.ylabel(r'$\lambda (X>x)$')
plt.xlabel('SaAvg(0.3s : 0.1s : 5.0s) (g)')
plt.xscale('log')
plt.yscale('log')
plt.show()

# iml_2in50 = np.exp(np.interp( np.log(1/2475), np.log(np.flip(haz_crv_dict[im_name]['lambda_haz'])), np.log(np.flip(haz_crv_dict[im_name]['iml_haz']))))

#%% MCEr computation from the hazard curves
#0.005 , 0.0071083, 0.0101055, 0.0143666, 0.0204243, 0.0290363, 0.0412796, 0.0586853, 0.0834303, 0.1186091, 0.1686212, 0.2397211, 0.3408007, 0.4845011, 0.6887933, 0.9792265, 1.3921222, 1.9791175, 2.8136222, 4.
# 0.113047541,0.0893633443,0.0692816668,0.0526682814,0.0392057389,0.0285238135,0.0202472287,0.0139818825,0.00932071825,0.00590590860,0.00348236741,0.00187209327,0.000904994884,0.000389996339,0.000147765717,0.0000478781761,0.0000128150621,0.00000262260744,0.000000447034900,0.0000000298023204

#%% UHS comparison for 10in50 and 2in50 

# per_10in50 = np.array([5, 4, 3, 2, 1, 0.6, 0.3])
# sa_10in50 = np.array([ 0.10278052086900909, 0.13947986898459977, 0.20241521324682352, 0.32669402578259005, 0.6592708671423725, 0.9551589706051304, 1.1781609947817522 ])


# plt.figure(dpi = 200)
# plt.plot( per_10in50, sa_10in50, '-or')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim([0.1, 10])
# plt.ylim([0.01, 5])
# plt.show()

# parse hazard curves, get values of Sa at specified return period, plot --- UHS

haz_crv_files_lst = [
'/Users/nbijelic/Research/EPFL/projects/Hiro/oq_hazard_results/LA_site_8s_frame/calc_id_11_sa_active_trt/hazard_curve-mean-SA(0.3)_11.csv',
'/Users/nbijelic/Research/EPFL/projects/Hiro/oq_hazard_results/LA_site_8s_frame/calc_id_11_sa_active_trt/hazard_curve-mean-SA(0.6)_11.csv',
'/Users/nbijelic/Research/EPFL/projects/Hiro/oq_hazard_results/LA_site_8s_frame/calc_id_11_sa_active_trt/hazard_curve-mean-SA(1.0)_11.csv',
'/Users/nbijelic/Research/EPFL/projects/Hiro/oq_hazard_results/LA_site_8s_frame/calc_id_11_sa_active_trt/hazard_curve-mean-SA(2.0)_11.csv',
'/Users/nbijelic/Research/EPFL/projects/Hiro/oq_hazard_results/LA_site_8s_frame/calc_id_14_sa_active_trt/hazard_curve-mean-SA(2.7)_14.csv',
'/Users/nbijelic/Research/EPFL/projects/Hiro/oq_hazard_results/LA_site_8s_frame/calc_id_14_sa_active_trt/hazard_curve-mean-SA(3.0)_14.csv',
'/Users/nbijelic/Research/EPFL/projects/Hiro/oq_hazard_results/LA_site_8s_frame/calc_id_14_sa_active_trt/hazard_curve-mean-SA(4.0)_14.csv',
'/Users/nbijelic/Research/EPFL/projects/Hiro/oq_hazard_results/LA_site_8s_frame/calc_id_14_sa_active_trt/hazard_curve-mean-SA(5.0)_14.csv'
]

per_lst = [0.3, 0.6, 1., 2., 2.7, 3., 4., 5.,]

RP_extract_all = [475, 2475] # return period of interest
sa_uhs_all = []

for rp in RP_extract_all:
    sa_uhs_curr = []
    for file_curr in haz_crv_files_lst:
        haz_crv_curr = parse_haz_curve_csv(file_curr)    
        # convert to exceedance rates    
        T_years = 1
        lam_rate_curr = -np.log(1 - haz_crv_curr['poe_haz'])/T_years
        iml_haz_curr = haz_crv_curr['iml_haz']
        sa_uhs_curr_val = np.exp(np.interp( np.log(1/rp), np.log(np.flip(lam_rate_curr)), np.log(np.flip(iml_haz_curr))))
        sa_uhs_curr.append(sa_uhs_curr_val)
    sa_uhs_all.append(sa_uhs_curr)


linestyle_all = ['-', '--']

plt.figure(dpi = 200)
for rp, uhs, linestyle_curr in zip(RP_extract_all, sa_uhs_all, linestyle_all):
    plt.plot(per_lst, uhs, label = f'RP = {rp}', color = 'red', 
             linestyle = linestyle_curr, marker = 'x')

# plt.plot(per_lst, 1.5*np.array(sa_uhs_all[0]), label = '1.5 * DBE', color = 'blue', 
#          linestyle = linestyle_curr, marker = 'x')

# plt.plot(per_lst, 2/3*np.array(sa_uhs_all[1]), label = '2/3 * MCE', color = 'green', 
#          linestyle = linestyle_curr, marker = 'x')
    
# plt.xscale('log')
# plt.yscale('log')
plt.legend()
# plt.xlim([0.1, 10])
# plt.xlim([0.1, 6])
# plt.ylim([0.01, 5])

plt.xlim([0, 9])
# plt.ylim([0., 2.5])
plt.ylim([0., 1.6])
plt.xlabel('T (s)')
plt.ylabel('Sa (g)')
plt.show()

#%% DEAGGREGATION cases

TODO: 
    - parse the deagg file
    - compute mean M and R
    - plot the deaggregation
     - use scripts I have from before




#%% Plot hazard curves -- initial test for Hiro at the LA site;

# Here I manually enter the poe values form the hazard curve csv file
# test below with the xml parser

im_name = 'SaAvg(0.1 : 0.1 : 0.3) (g)'

iml = np.array([5.000000000E-03, 7.108274886E-03, 1.010551437E-02, 1.436655480E-02,
                2.042428414E-02, 2.903628520E-02, 4.127957938E-02, 5.868531948E-02,
                8.343027653E-02, 1.186090679E-01, 1.686211717E-01, 2.397211280E-01,
                3.408007348E-01, 4.845010608E-01, 6.887933446E-01, 9.792264866E-01,
                1.392122208E+00, 1.979117467E+00, 2.813622197E+00, 4.000000000E+00])

poe_site_1 = np.array([1.418811E-02, 1.418746E-02, 1.418447E-02, 1.417255E-02,
                       1.413345E-02	, 1.402497E-02, 1.377213E-02, 1.327074E-02,
                       1.241446E-02, 1.113892E-02,	9.469688E-03, 7.545829E-03,
                       5.589128E-03, 3.816783E-03, 2.368093E-03, 1.301050E-03, 
                       6.105900E-04, 2.351999E-04, 7.164478E-05, 1.662970E-05])

name_site_1 = 'LA'


T_years = 1
lam_rate_site_1 = -np.log(1 - poe_site_1)/T_years

plt.figure(dpi = 200)
plt.plot(iml, lam_rate_site_1, '-or', label = name_site_1)
plt.plot([0.005, 4], [1/2475, 1/2475], '--k')
plt.ylim([1E-6, 1])
# plt.xlim([0.01, 4])
plt.xlim([0.005, 4])
plt.grid()
plt.legend()
plt.ylabel(r'$\lambda (X>x)$')
plt.xlabel(im_name)
plt.xscale('log')
plt.yscale('log')
plt.show()

## This is how to obtain the iml at a desired exceedance rate
# np.interp(1/475, np.flip(lam_rate_site_1), np.flip(iml)) :: for 10in50

#%% TEMPLATE CODE TO USE with XML outputs --- check this works


#%% function to parse the hazard curve XML files output by openQuake
import re
import xml.etree.ElementTree as ET

xml_file_in = '/Users/nbijelic/Research/EPFL/students/Mohamad/hazard/oq_tests/Sion_test/ESHM2020/output/moderate/hazard_curve-mean_115-AvgSA.xml'


mytree = ET.parse(xml_file_in)
# myroot = mytree.getroot()

# print(list(mytree.iter()))

#%% print the contents of the tree
for x in mytree.iter():
    print(x)

#%%

# print the IMLs, this is the same for all hazard curves
for x in mytree.iter('{http://openquake.org/xmlns/nrml/0.5}IMLs'):
    print(x.text)

# Print the exceedances for all hazard curves in the file -- will the order always be as in the file?
for x in mytree.iter('{http://openquake.org/xmlns/nrml/0.5}poEs'):
    print(x.text)

#
# Print the longitude / latitude for all the points
for x in mytree.iter('{http://www.opengis.net/gml}pos'):
    print(x.text)


#%% Parse IMLs, poEs, lon/lat



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
    
### Parse poEs
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

### Parse lon/lat
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

#%% convert poEs to rates


# parse the investigation time from the attributes of the hazard curve lists
# later in the function it is assumed I only have one investigation time
T_years_lst = []
for x in mytree.iter('{http://openquake.org/xmlns/nrml/0.5}hazardCurves'):
    T_years_lst.append( float(x.attrib['investigationTime']) )

T_years = T_years_lst[0]

rate_list = []

for poe in poe_list:
    rate_list.append( -np.log(1 - poe)/T_years )


#%% function parsing the openQuake hazard curve computation xml file

import re
import xml.etree.ElementTree as ET

def parse_oq_hazard_xml(xml_file_in): 
    """Function that parses the xml file output by the openQuake engine"""
    
    ## Input: 
    #   xml_file_in : hazard curve output file from openQuake
    #
    ## Output:
    #   dict_out : dictionary with following entries  
    #              'T_years' : investigationTime used in openQuake calculation
    #              'IMLs' : np.array with iml values for which the exceedances 
    #                   are computed
    #              'lonLat_lst' : list containing np.arrays with [lon, lat] for 
    #                   each site
    #              'poe_lst' : list containing np.arrays with probabilities of
    #                   exceedance computed by openQuake for each site
    #              'rate_list' : 
    
    
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


#%% print the hazard curves from the parsed data
# provide site names, linetypes, line colors

site_name_lst = ['Sion', 'Tulcea', 'Katerini']
linetype_lst = ['-','--', '-.']
linecolor_lst = ['red', 'red', 'red']
im_name = 'SaAvg(0.4 : 0.1 : 4.4) (g)'

plt.figure(dpi = 200)

for i in range(0, len(site_name_lst)):    
    plt.plot(imls, rate_list[i], linestyle = linetype_lst[i], 
             color = linecolor_lst[i], label = site_name_lst[i])

plt.ylim([1E-6, 1])
plt.xlim([0.01, 2])
plt.grid()
plt.legend()
plt.ylabel(r'$\lambda (X>x)$')
plt.xlabel(im_name)
plt.xscale('log')
plt.yscale('log')
plt.show()
 

#%% Plots using data parsed with the function


### Specify hazard curve info

## Moderate seismicity case
# GMPEs: all ESHM20 GMPEs
# 50 year 
# xml_file_in = '/Users/nbijelic/Research/EPFL/students/Mohamad/hazard/oq_tests/Sion_test/ESHM2020/output/moderate/hazard_curve-mean_115-AvgSA.xml'
# 1 year
xml_file_in = '/Users/nbijelic/Research/EPFL/students/Mohamad/hazard/oq_tests/Sion_test/ESHM2020/output/moderate/hazard_curve-mean_116-AvgSA.xml'
# GMPEs: BA2008 GMPE
# xml_file_in = '/Users/nbijelic/Research/EPFL/students/Mohamad/hazard/oq_tests/Sion_test/ESHM2020/output/moderate/all_TRT/hazard_curve-mean_110-AvgSA.xml'

site_name_lst = ['Sion', 'Tulcea', 'Katerini']
linetype_lst = ['-','--', '-.']
linecolor_lst = ['red', 'red', 'red']
im_name = 'SaAvg(0.4 : 0.1 : 4.4) (g)'


## High seismicity case
# GMPEs: BA2008 GMPE
# xml_file_in = '/Users/nbijelic/Research/EPFL/students/Mohamad/hazard/oq_tests/Sion_test/ESHM2020/output/high/all_TRT/hazard_curve-mean_111-AvgSA.xml'

# site_name_lst = ['Loutraki', 'Trevi', 'Ploiesti']
# linetype_lst = ['-','--', '-.']
# linecolor_lst = ['red', 'red', 'red']
# im_name = 'SaAvg(0.4 : 0.1 : 4.4) (g)'






### Parse and plot the hazard curves
# parse openQuake output using a function
hazard_out = parse_oq_hazard_xml(xml_file_in)
# plot
plt.figure(dpi = 800)
for i in range(0, len(site_name_lst)):    
    plt.plot(imls, hazard_out['rate_lst'][i], linestyle = linetype_lst[i], 
             color = linecolor_lst[i], label = site_name_lst[i])
plt.ylim([1E-6, 1])
plt.xlim([0.01, 2])
plt.grid()
plt.legend()
plt.ylabel(r'$\lambda (X>x)$')
plt.xlabel(im_name)
plt.xscale('log')
plt.yscale('log')
plt.show()


# STOPPED HERE: 
# - openQuake running the case for the moderate seismicity with T_years = 1; 
#   compare to the one based on the 50 years (plot), 
# - read the hazard document to see if european model includes time dependent 
#   hazard or not


# TODO: 
#      - run openQuake for the sites needed --- include corresponding info from the ESHM20 csv file (region and other stuff)
#      - check if there is a difference between 1 year and 50 year computation when converted to rates (essentially are there time dependent models in the data)

    
    
    













































































