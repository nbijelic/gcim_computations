#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 08:49:20 2023

@author: nbijelic
"""

# script to develop parsing and plotting functions for deaggregation output
# from openQuake

# deaggregation type supported: 
    # magnitude and distance
    # M/R/epsilon -- add if needed; need stacked bar plots


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

# -----------------------------------------------------------------------------
# deaggregation plotting function
# -----------------------------------------------------------------------------

# this one has issue with clipping when plotting the bars
def deagg_plot_clip_issue(deagg_file, prob_col_name, delta_M, delta_R, investigation_time,
               deagg_name, plot_lim_distance = None):
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
    
    # remove parts of the data where distance is larger than specified limit
    if(plot_lim_distance is not None):
        df_deagg = df_deagg[df_deagg['dist'] <= plot_lim_distance]
    
    # sort descending by distance
    df_deagg.sort_values(by=['dist'], ascending = False)
    
    # plot the deaggregation

    x = df_deagg['dist'].to_numpy() - delta_R/2
    y = df_deagg['mag'].to_numpy() - delta_M/2
    z = [0] * x.shape[0]

    dx = [delta_R/2] * x.shape[0]
    dy = [delta_M/2] * x.shape[0]
    dz = 100*df_deagg['P(m | X>x)'].to_numpy() # percentage contribution to hazard

    plt.rcParams.update({'font.size': 16})

    # plt.figure(dpi = 200, figsize = (8, 12))
    plt.figure(dpi = 200, figsize = (14, 24))
    ax = plt.axes(projection = '3d')
    
    # ax.bar3d(x, y, z, dx, dy, dz, alpha = 0.7)

    for x, y, z, dx, dy, dz in zip(x, y, z, dx, dy, dz):
        if(dz < 0.05):
            color_curr = 'white'
            edge_col_curr = 'white'
        else:
            color_curr = 'blue'
            edge_col_curr = 'black'
        ax.bar3d(x, y, z, dx, dy, dz, color = color_curr, alpha = 0.9, 
                 edgecolor = edge_col_curr,
                 shade = False)
    
    # ax.view_init(elev=45., azim=330)
    ax.view_init(elev=35., azim=315)

    # ax.set_xlim(0,500)
    # ax.set_xlim(0,100)
    ax.set_xlim(0,200)
    ax.set_ylim(4.0, 10.0)

    ax.set_xlabel('R [km]')
    ax.set_ylabel('M')

    ax.set_zlabel('% contribution')
    plt.title(deagg_name + '\n mean: M = {}, R = {} km'.format(np.round(mu_M, 2), np.round(mu_R, 2)))
    
    plt.autoscale(enable=True, axis='both', tight=True)
    plt.grid()
    plt.show(block=False)
    
    # plt.show()
    
    return df_deagg

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



#%% HIRO, 8-story MRF, SaAvg(0.3s:0.1s:5.0s) -- final hazard curve


deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/oq_hazard_results/LA_site_8s_frame/calc_id_13_FINAL_sa_avg_hiro_LA_8_story_mrf/Mag_Dist-mean-0_13_for_plot.csv'
prob_col_name = 'mean' # column name containing P(X>x | M, R); here I assume 
                       # only a single realization is used
delta_M = 0.5
delta_R = 20 #10
investigation_time = 1
deagg_name = 'LA, 8-story MRF, SaAvg, 2% in 50 years' # for plotting title

deagg_plot(deagg_file, prob_col_name, delta_M, delta_R, investigation_time,
               deagg_name)

#%% HIRO, 8-story MRF, SaAvg(0.3s:0.1s:5.0s); intensity based on sa_avg from MCEr


deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/oq_hazard_results/LA_site_8s_frame/calc_id_20_mcer_sa_avg/Mag_Dist-mean-0_20_for_plot.csv'
prob_col_name = 'mean' # column name containing P(X>x | M, R); here I assume 
                       # only a single realization is used
delta_M = 0.5
delta_R = 20 #10
investigation_time = 1
deagg_name = 'LA, 8-story MRF, SaAvg from MCEr (~1% in 100 years)' # for plotting title

deagg_plot(deagg_file, prob_col_name, delta_M, delta_R, investigation_time,
               deagg_name)

#%% Seattle site, for Hiro, Sa(T = 1.0s)


deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/calc_id_33_all_TRTs/Mag_Dist-mean-0_33_for_plot.csv'
prob_col_name = 'mean' # column name containing P(X>x | M, R); here I assume 
                       # only a single realization is used
delta_M = 0.5
delta_R = 20 #10
investigation_time = 1
deagg_name = 'Seattle, Sa(T = 1.0s), RP = 2475 years (2% in 50 years)' # for plotting title

deagg_plot(deagg_file, prob_col_name, delta_M, delta_R, investigation_time,
               deagg_name)

#%% Seattle site, for Hiro, Sa(T = 1.0s), finer M R grid

deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/calc_id_40_all_TRTs_finer_M-R_grid/Mag_Dist-mean-0_40_for_plot.csv'
prob_col_name = 'mean' # column name containing P(X>x | M, R); here I assume 
                       # only a single realization is used
delta_M = 0.2
delta_R = 5
investigation_time = 1
deagg_name = 'Seattle, Sa(T = 1.0s), RP = 2475 years (2% in 50 years)' # for plotting title

deagg_plot(deagg_file, prob_col_name, delta_M, delta_R, investigation_time,
               deagg_name)

#%% Seattle site, for Hiro, SaAvg(0.3 : 0.1 : 5.0), finer M R grid
# TRT: all

# deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/SaAVG_0p3-0p1-5p0/calc_id_53_saAvg_all_trt_deagg_at_2in50/Mag_Dist-mean-0_53_for_plot.csv'

# # 10in50
# # deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/10in50/calc_id_115_10in50_saAVG_all_TRT_vs30_260_AbrahamsonGulerce2020_cas_adjust_true_z1p0_876_z2p5_6p374_inslab_and_interface/Mag_Dist-mean-0_115_for_plot.csv'
# deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_205_saAvg_all_trt_8_story_mrf_10in50/Mag_Dist-mean-0_205_for_plotting.csv'
# deagg_name = 'Seattle, SaAvg(0.3 : 0.1: 5.0), RP = 475 years (10% in 50 years)' # for plotting title

# DBE
deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_209_saAvg_all_trt_8_story_mrf_dbe/Mag_Dist-mean-0_209_plot.csv'
deagg_name = 'Seattle, SaAvg(0.3 : 0.1: 5.0), DBE (2/3MCER)' # for plotting title

# 2in50
#deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/finalized/calc_id_110_saAVG_all_TRT_vs30_260_AbrahamsonGulerce2020_cas_adjust_true_z1p0_876_z2p5_6p374_inslab_and_interface/Mag_Dist-mean-0_110_for_plot.csv'
# deagg_name = 'Seattle, SaAvg(0.3 : 0.1: 5.0), RP = 2475 years (2% in 50 years)' # for plotting title

prob_col_name = 'mean' # column name containing P(X>x | M, R); here I assume 
                       # only a single realization is used
delta_M = 0.2
delta_R = 5
investigation_time = 1
# deagg_name = 'Seattle, SaAvg(0.3 : 0.1: 5.0), RP = 2475 years (2% in 50 years)' # for plotting title

deagg_plot(deagg_file, prob_col_name, delta_M, delta_R, investigation_time,
               deagg_name, plot_lim_distance = 200)

#%% Seattle site, for Hiro, SaAvg(0.3 : 0.1 : 5.0), finer M R grid
# TRT: active shallow crust

# deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/SaAVG_0p3-0p1-5p0/calc_id_55_saAvg_active-shallow-crust_trt_iml_deagg_from_2in50_in_calc_id_53/Mag_Dist-mean-0_55_for_plot.csv'

# 10in50
# deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/10in50/calc_id_117_10in50_saAVG_active_TRT_vs30_260_AbrahamsonGulerce2020_cas_adjust_true_z1p0_876_z2p5_6p374_inslab_and_interface/Mag_Dist-mean-0_117_for_plot.csv'
# deagg_name = 'Seattle, SaAvg(0.3 : 0.1: 5.0), RP = 475 years (10% in 50 years), active shallow crust' # for plotting title
# deagg_weight_use = 0.3545

# deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_206_saAvg_crustal_trt_8_story_mrf_10in50/Mag_Dist-mean-0_206_for_plotting.csv'
# deagg_name = 'Seattle, SaAvg(0.3 : 0.1: 5.0), RP = 475 years (10% in 50 years), active shallow crust' # for plotting title
# deagg_weight_use = 0.542

# DBE
deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_210_saAvg_crustal_trt_8_story_mrf_dbe/Mag_Dist-mean-0_210_plot.csv'
deagg_name = 'Seattle, SaAvg(0.3 : 0.1: 5.0), DBE (2/3MCER), active shallow crust' # for plotting title
deagg_weight_use = 0.718


# # 2in50
# deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/finalized/calc_id_111_saAVG_active_TRT_vs30_260_AbrahamsonGulerce2020_cas_adjust_true_z1p0_876_z2p5_6p374_inslab_and_interface/Mag_Dist-mean-0_111_for_plot.csv'
# deagg_name = 'Seattle, SaAvg(0.3 : 0.1: 5.0), RP = 2475 years (2% in 50 years), active shallow crust' # for plotting title
# deagg_weight_use = 0.5610

prob_col_name = 'mean' # column name containing P(X>x | M, R); here I assume 
                       # only a single realization is used
delta_M = 0.2
delta_R = 5
investigation_time = 1



deagg_plot(deagg_file, prob_col_name, delta_M, delta_R, investigation_time,
               deagg_name, plot_lim_distance = 200, deagg_weight = deagg_weight_use)

#%% Seattle site, for Hiro, SaAvg(0.3 : 0.1 : 5.0), finer M R grid
# TRT: subduction inslab

# 10in50
# deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/10in50/calc_id_120_10in50_saAVG_inslab_TRT_vs30_260_AbrahamsonGulerce2020_cas_adjust_true_z1p0_876_z2p5_6p374_inslab_and_interface/Mag_Dist-mean-0_120_for_plot.csv'
# deagg_name = 'Seattle, SaAvg(0.3 : 0.1: 5.0), RP = 2475 years (10% in 50 years), subduction inslab' # for plotting title
# deagg_weight_use = 0.2806

# deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_208_saAvg_inslab_trt_8_story_mrf_10in50/Mag_Dist-mean-0_208_for_plotting.csv'
# deagg_name = 'Seattle, SaAvg(0.3 : 0.1: 5.0), RP = 475 years (10% in 50 years), subduction inslab' # for plotting title
# deagg_weight_use = 0.131

# DBE
deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_212_saAvg_inslab_trt_8_story_mrf_dbe/Mag_Dist-mean-0_212_plot.csv'
deagg_name = 'Seattle, SaAvg(0.3 : 0.1: 5.0), DBE (2/3MCER), subduction inslab' # for plotting title
deagg_weight_use = 0.04

# # 2in50
# # deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/SaAVG_0p3-0p1-5p0/calc_id_58_saAvg_subduction-inslab_trt_iml_deagg_from_2in50_in_calc_id_53/Mag_Dist-mean-0_58_for_plot.csv'
# deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/finalized/calc_id_114_saAVG_inslab_TRT_vs30_260_AbrahamsonGulerce2020_cas_adjust_true_z1p0_876_z2p5_6p374_inslab_and_interface/Mag_Dist-mean-0_114_for_plot.csv'
# deagg_name = 'Seattle, SaAvg(0.3 : 0.1: 5.0), RP = 2475 years (2% in 50 years), subduction inslab' # for plotting title
# deagg_weight_use = 0.1189


prob_col_name = 'mean' # column name containing P(X>x | M, R); here I assume 
                       # only a single realization is used
delta_M = 0.2
delta_R = 5
investigation_time = 1


deagg_plot(deagg_file, prob_col_name, delta_M, delta_R, investigation_time,
               deagg_name, plot_lim_distance = 200, deagg_weight = deagg_weight_use)

#%% Seattle site, for Hiro, SaAvg(0.3 : 0.1 : 5.0), finer M R grid
# TRT: subduction interface


# # 10in50
# deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/10in50/calc_id_119_10in50_saAVG_interface_TRT_vs30_260_AbrahamsonGulerce2020_cas_adjust_true_z1p0_876_z2p5_6p374_inslab_and_interface/Mag_Dist-mean-0_119_for_plot.csv'
# deagg_name = 'Seattle, SaAvg(0.3 : 0.1: 5.0), RP = 2475 years (10% in 50 years), subduction interface' # for plotting title
# deagg_weight_use = 0.3649

# deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_207_saAvg_interface_trt_8_story_mrf_10in50/Mag_Dist-mean-0_207_for_plotting.csv'
# deagg_name = 'Seattle, SaAvg(0.3 : 0.1: 5.0), RP = 475 years (10% in 50 years), subduction interface' # for plotting title
# deagg_weight_use = 0.327

# DBE
deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_211_saAvg_interface_trt_8_story_mrf_dbe/Mag_Dist-mean-0_211_plot.csv'
deagg_name = 'Seattle, SaAvg(0.3 : 0.1: 5.0), DBE (2/3MCER), subduction interface' # for plotting title
deagg_weight_use = 0.242

# # 2in50
# # deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/SaAVG_0p3-0p1-5p0/calc_id_57_saAvg_subduction-interface_trt_iml_deagg_from_2in50_in_calc_id_53/Mag_Dist-mean-0_57_for_plot.csv'
# deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/finalized/calc_id_113_saAVG_interface_TRT_vs30_260_AbrahamsonGulerce2020_cas_adjust_true_z1p0_876_z2p5_6p374_inslab_and_interface/Mag_Dist-mean-0_113_for_plot.csv'
# deagg_name = 'Seattle, SaAvg(0.3 : 0.1: 5.0), RP = 2475 years (2% in 50 years), subduction interface' # for plotting title
# deagg_weight_use = 0.3359


prob_col_name = 'mean' # column name containing P(X>x | M, R); here I assume 
                       # only a single realization is used
delta_M = 0.2
delta_R = 5
investigation_time = 1


deagg_plot(deagg_file, prob_col_name, delta_M, delta_R, investigation_time,
               deagg_name, plot_lim_distance = 200, deagg_weight = deagg_weight_use)

#%% HIRO, Seattle site, RotD100, SaAvg(0.3 : 0.1 : 5.0) 
# finer M R grid

# # ---- TRT: all
# # # 2in50
# deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_187_saAvg_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/Mag_Dist-mean-0_187_for_plot.csv'
# deagg_name = 'Seattle, SaAvg(0.3 : 0.1: 5.0), RP = 2475 years (2% in 50 years)' # for plotting title
# deagg_weight_use = 1.0


# prob_col_name = 'mean' # column name containing P(X>x | M, R); here I assume 
#                        # only a single realization is used
# delta_M = 0.2
# delta_R = 5
# investigation_time = 1


# deagg_plot(deagg_file, prob_col_name, delta_M, delta_R, investigation_time,
#                deagg_name, plot_lim_distance = 200, deagg_weight = deagg_weight_use)
# # -----------------------------------------------------------------------------

# # ---- TRT: active
# # # 2in50
# deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_190_saAvg_crustal_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/Mag_Dist-mean-0_190_for_plotting.csv'
# deagg_name = 'Seattle, SaAvg(0.3 : 0.1: 5.0), RP = 2475 years (2% in 50 years), active shallow crust' # for plotting title
# deagg_weight_use = 0.764851485


# prob_col_name = 'mean' # column name containing P(X>x | M, R); here I assume 
#                        # only a single realization is used
# delta_M = 0.2
# delta_R = 5
# investigation_time = 1


# deagg_plot(deagg_file, prob_col_name, delta_M, delta_R, investigation_time,
#                deagg_name, plot_lim_distance = 200, deagg_weight = deagg_weight_use)
# # -----------------------------------------------------------------------------

# # ---- TRT: inslab
# # # 2in50
# deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_192_saAvg_inslab_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/Mag_Dist-mean-0_192_for_plotting.csv'
# deagg_name = 'Seattle, SaAvg(0.3 : 0.1: 5.0), RP = 2475 years (2% in 50 years), subduction inslab' # for plotting title
# deagg_weight_use = 0.027475248


# prob_col_name = 'mean' # column name containing P(X>x | M, R); here I assume 
#                        # only a single realization is used
# delta_M = 0.2
# delta_R = 5
# investigation_time = 1


# deagg_plot(deagg_file, prob_col_name, delta_M, delta_R, investigation_time,
#                deagg_name, plot_lim_distance = 200, deagg_weight = deagg_weight_use)
# # -----------------------------------------------------------------------------

# # ---- TRT: interface
# # # 2in50
# deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_191_saAvg_interface_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/Mag_Dist-mean-0_191_for_plotting.csv'
# deagg_name = 'Seattle, SaAvg(0.3 : 0.1: 5.0), RP = 2475 years (2% in 50 years), subduction interface' # for plotting title
# deagg_weight_use = 0.208168317


# prob_col_name = 'mean' # column name containing P(X>x | M, R); here I assume 
#                        # only a single realization is used
# delta_M = 0.2
# delta_R = 5
# investigation_time = 1


# deagg_plot(deagg_file, prob_col_name, delta_M, delta_R, investigation_time,
#                deagg_name, plot_lim_distance = 200, deagg_weight = deagg_weight_use)
# # -----------------------------------------------------------------------------

# # ---- TRT: all
# # # MCEr intensity = 0.616 g
# deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_193_saAvg_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term_mcer_intensity/Mag_Dist-mean-0_193_for_plot.csv'
# deagg_name = 'Seattle, SaAvg(0.3 : 0.1: 5.0), MCEr intensity' # for plotting title
# deagg_weight_use = 1.0

# prob_col_name = 'mean' # column name containing P(X>x | M, R); here I assume 
#                        # only a single realization is used
# delta_M = 0.2
# delta_R = 5
# investigation_time = 1


# deagg_plot(deagg_file, prob_col_name, delta_M, delta_R, investigation_time,
#                deagg_name, plot_lim_distance = 200, deagg_weight = deagg_weight_use)
# # -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 4-story MRF, Hiro
# -----------------------------------------------------------------------------

# # ---- TRT: all
# # # 2in50 intensity = 0.766 g
# deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/4-story_mrf/calc_id_198_saAvg_all_trt_4_story_mrf_2in50/Mag_Dist-mean-0_198_for_plot.csv'
# deagg_name = 'Seattle, SaAvg(0.2 : 0.1: 3.0), RP = 2475 years (2% in 50 years)' # for plotting title
# deagg_weight_use = 1.0

# prob_col_name = 'mean' # column name containing P(X>x | M, R); here I assume 
#                        # only a single realization is used
# delta_M = 0.2
# delta_R = 5
# investigation_time = 1


# deagg_plot(deagg_file, prob_col_name, delta_M, delta_R, investigation_time,
#                deagg_name, plot_lim_distance = 200, deagg_weight = deagg_weight_use)
# # -----------------------------------------------------------------------------

# # ---- TRT: active
# # # 2in50 intensity = 0.766 g
# deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/4-story_mrf/calc_id_200_saAvg_crustal_trt_4_story_mrf_2in50/Mag_Dist-mean-0_200_for_plot.csv'
# deagg_name = 'Seattle, SaAvg(0.2 : 0.1: 3.0), RP = 2475 years (2% in 50 years), active shallow crust' # for plotting title
# deagg_weight_use = 0.738

# prob_col_name = 'mean' # column name containing P(X>x | M, R); here I assume 
#                        # only a single realization is used
# delta_M = 0.2
# delta_R = 5
# investigation_time = 1


# deagg_plot(deagg_file, prob_col_name, delta_M, delta_R, investigation_time,
#                deagg_name, plot_lim_distance = 200, deagg_weight = deagg_weight_use)
# # -----------------------------------------------------------------------------

# # ---- TRT: inslab
# # # 2in50 intensity = 0.766 g
# deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/4-story_mrf/calc_id_202_saAvg_inslab_trt_4_story_mrf_2in50/Mag_Dist-mean-0_202_for_plotting.csv'
# deagg_name = 'Seattle, SaAvg(0.2 : 0.1: 3.0), RP = 2475 years (2% in 50 years), subduction inslab' # for plotting title
# deagg_weight_use = 0.0683

# prob_col_name = 'mean' # column name containing P(X>x | M, R); here I assume 
#                        # only a single realization is used
# delta_M = 0.2
# delta_R = 5
# investigation_time = 1


# deagg_plot(deagg_file, prob_col_name, delta_M, delta_R, investigation_time,
#                deagg_name, plot_lim_distance = 200, deagg_weight = deagg_weight_use)
# # -----------------------------------------------------------------------------

# ---- TRT: interface
# # 2in50 intensity = 0.766 g
deagg_file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/4-story_mrf/calc_id_201_saAvg_interface_trt_4_story_mrf_2in50/Mag_Dist-mean-0_201_for_plotting.csv'
deagg_name = 'Seattle, SaAvg(0.2 : 0.1: 3.0), RP = 2475 years (2% in 50 years), subduction interface' # for plotting title
deagg_weight_use = 0.207

prob_col_name = 'mean' # column name containing P(X>x | M, R); here I assume 
                       # only a single realization is used
delta_M = 0.2
delta_R = 5
investigation_time = 1


deagg_plot(deagg_file, prob_col_name, delta_M, delta_R, investigation_time,
               deagg_name, plot_lim_distance = 200, deagg_weight = deagg_weight_use)
# -----------------------------------------------------------------------------



#%% TEMPLATE CODE FROM BEFORE

#%% load csv deagg output, convert to traditional deaggregation format

# define file, binning widths, investigation time, column name with probabilities
deagg_file = '/Users/nbijelic/Research/EPFL/students/Mohamad/hazard/oq_tests/Sion_test/ESHM2013/deagg_outputs/Mag_Dist-0_104.csv'
prob_col_name = 'rlz0' # column name containing P(X>x | M, R); here I assume 
                       # only a single realization is used
delta_M = 0.5
delta_R = 20
investigation_time = 50 # used for converting to rates with Poisson assumption
deagg_name = 'Sion, SaAvg, 2% in 50 years' # for plotting title
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

#%% plot the deaggregation

x = df_deagg['dist'].to_numpy() - delta_R/2
y = df_deagg['mag'].to_numpy() - delta_M/2
z = [0] * x.shape[0]

dx = [delta_R/2] * x.shape[0]
dy = [delta_M/2] * x.shape[0]
dz = df_deagg['P(m | X>x)'].to_numpy()

plt.rcParams.update({'font.size': 16})

# plt.figure(dpi = 200, figsize = (8, 12))
plt.figure(dpi = 200, figsize = (16, 24))
ax = plt.axes(projection = '3d')
ax.bar3d(x, y, z, dx, dy,  dz, alpha = 0.7)

ax.set_xlim(0,200)
ax.set_ylim(4.5,9)

ax.set_xlabel('R [km]')
ax.set_ylabel('M')

ax.set_zlabel('% contribution')
plt.title(deagg_name + '\n mean: M = {}, R = {} km'.format(np.round(mu_M, 2), np.round(mu_R, 2)))

plt.show()

#%% Function works

deagg_file = '/Users/nbijelic/Research/EPFL/students/Mohamad/hazard/oq_tests/Sion_test/ESHM2020/output/moderate/Mag_Dist-0_113.csv'
prob_col_name = 'rlz12' # column name containing P(X>x | M, R); here I assume 
                       # only a single realization is used
delta_M = 0.5
delta_R = 20 #10
investigation_time = 50
deagg_name = 'Sion, SaAvg, 10% in 50 years' # for plotting title

deagg_plot(deagg_file, prob_col_name, delta_M, delta_R, investigation_time,
               deagg_name)

