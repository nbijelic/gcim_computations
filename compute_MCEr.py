#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 15:07:14 2025

@author: nbijelic

Script to compute the MCEr values using the USGS python tool.

https://code.usgs.gov/ghsc/erp/erp-rtgm-calculator

"""

from rtgmpy import GroundMotionHazard, BuildingCodeRTGMCalc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#%% Defining hazard data from a csv file - must follow the same format as the example
# csv_path = 'rtgmpy//example-haz-inputs//example-hazcurve-input.csv'
# haz_data = GroundMotionHazard.from_csv(csv_path)

# Can also provide a path to a json file
# json_path = 'rtgmpy//example-haz-inputs//all-IMs//SeattleWA-example.json'
# haz_data = GroundMotionHazard.from_json(json_path)

# Or directly from a python dictionary (needs to have the correct structure)
# haz_dict = {'site':{'name':'Portland OR','lat':45.5,'lon':-122.6,'Vs30':760}}
# haz_dict['hazCurves'] = {'PGA':{
#                                 'iml':[0.0023,0.0035,0.0052,0.0079,0.0118,0.0177,0.0265,0.0398,
#                                        0.0597,0.0896,0.134,0.202,0.302,0.454,0.68,1.02,1.53],
#                                 'afe':[0.764,0.609,0.462,0.329,0.22,0.137,0.0808,0.0452,0.0247,0.0135,
#                                        0.00753,0.00426,0.00244,0.00129,0.000574,0.000195,4.68e-05]
#                                 }
#                         }            
# haz_data = GroundMotionHazard.from_dict(haz_dict)

#%% Parse the hazard curves and store to a dict

def parse_haz_curve_csv(file, T_years):
    """ Function to parse the hazard curve file as output by OpenQuake and 
    stored to a dictionary format required by USGS python tool.
    
    Parameters
    ----------
    file : hazard curve file from OpenQuake
    
    Outputs
    -------
    dict
    
    """
    df = pd.read_csv(file, header= None, skiprows=1)
    df = df.iloc[:,3:]
    iml_haz = df.iloc[0,:]
    iml_haz = np.array([ float(x[4:]) for x in iml_haz ])
    poe_haz = df.iloc[1,:].to_numpy()
    poe_haz = np.array([ float(x) for x in poe_haz ])
    lambda_haz =  -np.log(1 - poe_haz)/T_years
    return {'iml': iml_haz, 'afe': lambda_haz} 

#%% Computation of MCEr

# define the site information -- for plotting purposes
haz_dict = {'site':{'name':'Seattle','lat':47.6,'lon':-122.34,'Vs30':260}}

# define hazard curves to consider - names of IMs and corresponding files
IM_names = ['SA0P1', 'SA0P15', 'SA0P2', 'SA0P25', 'SA0P3', 'SA0P4', 'SA0P5',
            'SA0P75', 'SA1P0', 'SA1P5', 'SA2P0', 'SA3P0', 'SA4P0', 'SA5P0']

# calc_id 163, Sa curves, all trt, RotD100, Z1p0 = 876m, Z2p5 = 6.374 km
# haz_crv_file_list = [
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/test_scaling_factors/calc_id_163_sa_all_trt_z1p0_876_z2p5_6p374_RotD100/hazard_curve-mean-SA(0.1)_163.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/test_scaling_factors/calc_id_163_sa_all_trt_z1p0_876_z2p5_6p374_RotD100/hazard_curve-mean-SA(0.15)_163.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/test_scaling_factors/calc_id_163_sa_all_trt_z1p0_876_z2p5_6p374_RotD100/hazard_curve-mean-SA(0.2)_163.csv',    
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/test_scaling_factors/calc_id_163_sa_all_trt_z1p0_876_z2p5_6p374_RotD100/hazard_curve-mean-SA(0.25)_163.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/test_scaling_factors/calc_id_163_sa_all_trt_z1p0_876_z2p5_6p374_RotD100/hazard_curve-mean-SA(0.3)_163.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/test_scaling_factors/calc_id_163_sa_all_trt_z1p0_876_z2p5_6p374_RotD100/hazard_curve-mean-SA(0.4)_163.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/test_scaling_factors/calc_id_163_sa_all_trt_z1p0_876_z2p5_6p374_RotD100/hazard_curve-mean-SA(0.5)_163.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/test_scaling_factors/calc_id_163_sa_all_trt_z1p0_876_z2p5_6p374_RotD100/hazard_curve-mean-SA(0.75)_163.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/test_scaling_factors/calc_id_163_sa_all_trt_z1p0_876_z2p5_6p374_RotD100/hazard_curve-mean-SA(1.0)_163.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/test_scaling_factors/calc_id_163_sa_all_trt_z1p0_876_z2p5_6p374_RotD100/hazard_curve-mean-SA(1.5)_163.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/test_scaling_factors/calc_id_163_sa_all_trt_z1p0_876_z2p5_6p374_RotD100/hazard_curve-mean-SA(2.0)_163.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/test_scaling_factors/calc_id_163_sa_all_trt_z1p0_876_z2p5_6p374_RotD100/hazard_curve-mean-SA(3.0)_163.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/test_scaling_factors/calc_id_163_sa_all_trt_z1p0_876_z2p5_6p374_RotD100/hazard_curve-mean-SA(4.0)_163.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/test_scaling_factors/calc_id_163_sa_all_trt_z1p0_876_z2p5_6p374_RotD100/hazard_curve-mean-SA(5.0)_163.csv',
#     ]
# max_dir_sf = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1. ]

# # calc_id 164, Sa curves, all trt, RotD100, Z1p0 = 913m, Z2p5 = 6.726 km
# haz_crv_file_list = [
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/test_scaling_factors/calc_id_164_sa_all_trt_z1p0_913_z2p5_6p726_RotD100/hazard_curve-mean-SA(0.1)_164.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/test_scaling_factors/calc_id_164_sa_all_trt_z1p0_913_z2p5_6p726_RotD100/hazard_curve-mean-SA(0.15)_164.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/test_scaling_factors/calc_id_164_sa_all_trt_z1p0_913_z2p5_6p726_RotD100/hazard_curve-mean-SA(0.2)_164.csv',    
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/test_scaling_factors/calc_id_164_sa_all_trt_z1p0_913_z2p5_6p726_RotD100/hazard_curve-mean-SA(0.25)_164.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/test_scaling_factors/calc_id_164_sa_all_trt_z1p0_913_z2p5_6p726_RotD100/hazard_curve-mean-SA(0.3)_164.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/test_scaling_factors/calc_id_164_sa_all_trt_z1p0_913_z2p5_6p726_RotD100/hazard_curve-mean-SA(0.4)_164.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/test_scaling_factors/calc_id_164_sa_all_trt_z1p0_913_z2p5_6p726_RotD100/hazard_curve-mean-SA(0.5)_164.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/test_scaling_factors/calc_id_164_sa_all_trt_z1p0_913_z2p5_6p726_RotD100/hazard_curve-mean-SA(0.75)_164.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/test_scaling_factors/calc_id_164_sa_all_trt_z1p0_913_z2p5_6p726_RotD100/hazard_curve-mean-SA(1.0)_164.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/test_scaling_factors/calc_id_164_sa_all_trt_z1p0_913_z2p5_6p726_RotD100/hazard_curve-mean-SA(1.5)_164.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/test_scaling_factors/calc_id_164_sa_all_trt_z1p0_913_z2p5_6p726_RotD100/hazard_curve-mean-SA(2.0)_164.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/test_scaling_factors/calc_id_164_sa_all_trt_z1p0_913_z2p5_6p726_RotD100/hazard_curve-mean-SA(3.0)_164.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/test_scaling_factors/calc_id_164_sa_all_trt_z1p0_913_z2p5_6p726_RotD100/hazard_curve-mean-SA(4.0)_164.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/test_scaling_factors/calc_id_164_sa_all_trt_z1p0_913_z2p5_6p726_RotD100/hazard_curve-mean-SA(5.0)_164.csv',
#     ]
# max_dir_sf = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1. ]

# # calc_id 167, Sa curves, all trt, RotD100, Z1p0 = 913m, Z2p5 = 6.726 km, usgs basin scaling term using cb14
# haz_crv_file_list = [
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_167_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(0.1)_167.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_167_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(0.15)_167.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_167_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(0.2)_167.csv',    
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_167_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(0.25)_167.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_167_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(0.3)_167.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_167_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(0.4)_167.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_167_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(0.5)_167.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_167_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(0.75)_167.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_167_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(1.0)_167.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_167_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(1.5)_167.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_167_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(2.0)_167.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_167_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(3.0)_167.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_167_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(4.0)_167.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_167_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(5.0)_167.csv',
#     ]
# max_dir_sf = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1. ]

# # calc_id 169, Sa curves, all trt, RotD100, Z1p0 = 913m, Z2p5 = 6.726 km, usgs basin scaling term using cb14, did not work properly
# haz_crv_file_list = [
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_169_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(0.1)_169.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_169_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(0.15)_169.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_169_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(0.2)_169.csv',    
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_169_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(0.25)_169.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_169_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(0.3)_169.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_169_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(0.4)_169.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_169_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(0.5)_169.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_169_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(0.75)_169.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_169_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(1.0)_169.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_169_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(1.5)_169.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_169_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(2.0)_169.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_169_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(3.0)_169.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_169_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(4.0)_169.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_169_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(5.0)_169.csv',
#     ]
# max_dir_sf = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1. ]

# # calc_id 174, Sa curves, all trt, RotD100, Z1p0 = 913m, Z2p5 = 6.726 km, usgs basin scaling term using cb14
# haz_crv_file_list = [
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_174_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(0.1)_174.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_174_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(0.15)_174.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_174_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(0.2)_174.csv',    
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_174_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(0.25)_174.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_174_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(0.3)_174.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_174_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(0.4)_174.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_174_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(0.5)_174.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_174_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(0.75)_174.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_174_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(1.0)_174.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_174_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(1.5)_174.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_174_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(2.0)_174.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_174_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(3.0)_174.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_174_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(4.0)_174.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_174_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(5.0)_174.csv',
#     ]
# max_dir_sf = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1. ]

# # calc_id 178, Sa curves, all trt, RotD100, Z1p0 = 913m, Z2p5 = 6.726 km, usgs basin scaling term using cb14
# haz_crv_file_list = [
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_178_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(0.1)_178.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_178_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(0.15)_178.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_178_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(0.2)_178.csv',    
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_178_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(0.25)_178.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_178_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(0.3)_178.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_178_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(0.4)_178.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_178_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(0.5)_178.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_178_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(0.75)_178.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_178_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(1.0)_178.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_178_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(1.5)_178.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_178_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(2.0)_178.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_178_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(3.0)_178.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_178_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(4.0)_178.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_178_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(5.0)_178.csv',
#     ]
# max_dir_sf = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1. ]

# # calc_id 179, Sa curves, all trt, RotD50, Z1p0 = 913m, Z2p5 = 6.726 km, usgs basin scaling term using cb14
# haz_crv_file_list = [
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_179_sa_all_trt_z1p0_913_z2p5_6p726_RotD50_usgs_basin_cb14_term/hazard_curve-mean-SA(0.1)_179.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_179_sa_all_trt_z1p0_913_z2p5_6p726_RotD50_usgs_basin_cb14_term/hazard_curve-mean-SA(0.15)_179.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_179_sa_all_trt_z1p0_913_z2p5_6p726_RotD50_usgs_basin_cb14_term/hazard_curve-mean-SA(0.2)_179.csv',    
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_179_sa_all_trt_z1p0_913_z2p5_6p726_RotD50_usgs_basin_cb14_term/hazard_curve-mean-SA(0.25)_179.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_179_sa_all_trt_z1p0_913_z2p5_6p726_RotD50_usgs_basin_cb14_term/hazard_curve-mean-SA(0.3)_179.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_179_sa_all_trt_z1p0_913_z2p5_6p726_RotD50_usgs_basin_cb14_term/hazard_curve-mean-SA(0.4)_179.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_179_sa_all_trt_z1p0_913_z2p5_6p726_RotD50_usgs_basin_cb14_term/hazard_curve-mean-SA(0.5)_179.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_179_sa_all_trt_z1p0_913_z2p5_6p726_RotD50_usgs_basin_cb14_term/hazard_curve-mean-SA(0.75)_179.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_179_sa_all_trt_z1p0_913_z2p5_6p726_RotD50_usgs_basin_cb14_term/hazard_curve-mean-SA(1.0)_179.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_179_sa_all_trt_z1p0_913_z2p5_6p726_RotD50_usgs_basin_cb14_term/hazard_curve-mean-SA(1.5)_179.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_179_sa_all_trt_z1p0_913_z2p5_6p726_RotD50_usgs_basin_cb14_term/hazard_curve-mean-SA(2.0)_179.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_179_sa_all_trt_z1p0_913_z2p5_6p726_RotD50_usgs_basin_cb14_term/hazard_curve-mean-SA(3.0)_179.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_179_sa_all_trt_z1p0_913_z2p5_6p726_RotD50_usgs_basin_cb14_term/hazard_curve-mean-SA(4.0)_179.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_179_sa_all_trt_z1p0_913_z2p5_6p726_RotD50_usgs_basin_cb14_term/hazard_curve-mean-SA(5.0)_179.csv',
#     ]
# max_dir_sf = [1.2, 1.2, 1.2, 1.203, 1.206, 1.213, 1.219, 1.234, 1.25, 1.253, 1.256, 1.261, 1.267, 1.272 ]

# # calc_id 183, Sa curves, all trt, RotD50, Z1p0 = 913m, Z2p5 = 6.726 km, usgs basin scaling term using cb14, basin term above 1.0s
# haz_crv_file_list = [
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_183_sa_all_trt_z1p0_913_z2p5_6p726_RotD50_usgs_basin_cb14_term/hazard_curve-mean-SA(0.1)_183.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_183_sa_all_trt_z1p0_913_z2p5_6p726_RotD50_usgs_basin_cb14_term/hazard_curve-mean-SA(0.15)_183.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_183_sa_all_trt_z1p0_913_z2p5_6p726_RotD50_usgs_basin_cb14_term/hazard_curve-mean-SA(0.2)_183.csv',    
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_183_sa_all_trt_z1p0_913_z2p5_6p726_RotD50_usgs_basin_cb14_term/hazard_curve-mean-SA(0.25)_183.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_183_sa_all_trt_z1p0_913_z2p5_6p726_RotD50_usgs_basin_cb14_term/hazard_curve-mean-SA(0.3)_183.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_183_sa_all_trt_z1p0_913_z2p5_6p726_RotD50_usgs_basin_cb14_term/hazard_curve-mean-SA(0.4)_183.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_183_sa_all_trt_z1p0_913_z2p5_6p726_RotD50_usgs_basin_cb14_term/hazard_curve-mean-SA(0.5)_183.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_183_sa_all_trt_z1p0_913_z2p5_6p726_RotD50_usgs_basin_cb14_term/hazard_curve-mean-SA(0.75)_183.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_183_sa_all_trt_z1p0_913_z2p5_6p726_RotD50_usgs_basin_cb14_term/hazard_curve-mean-SA(1.0)_183.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_183_sa_all_trt_z1p0_913_z2p5_6p726_RotD50_usgs_basin_cb14_term/hazard_curve-mean-SA(1.5)_183.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_183_sa_all_trt_z1p0_913_z2p5_6p726_RotD50_usgs_basin_cb14_term/hazard_curve-mean-SA(2.0)_183.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_183_sa_all_trt_z1p0_913_z2p5_6p726_RotD50_usgs_basin_cb14_term/hazard_curve-mean-SA(3.0)_183.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_183_sa_all_trt_z1p0_913_z2p5_6p726_RotD50_usgs_basin_cb14_term/hazard_curve-mean-SA(4.0)_183.csv',
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_183_sa_all_trt_z1p0_913_z2p5_6p726_RotD50_usgs_basin_cb14_term/hazard_curve-mean-SA(5.0)_183.csv',
#     ]
# max_dir_sf = [1.2, 1.2, 1.2, 1.203, 1.206, 1.213, 1.219, 1.234, 1.25, 1.253, 1.256, 1.261, 1.267, 1.272 ]

# calc_id 184, Sa curves, all trt, RotD100, Z1p0 = 913m, Z2p5 = 6.726 km, usgs basin scaling term using cb14, basin term above 1.0s
haz_crv_file_list = [
    '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_184_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(0.1)_184.csv',
    '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_184_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(0.15)_184.csv',
    '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_184_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(0.2)_184.csv',    
    '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_184_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(0.25)_184.csv',
    '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_184_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(0.3)_184.csv',
    '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_184_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(0.4)_184.csv',
    '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_184_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(0.5)_184.csv',
    '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_184_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(0.75)_184.csv',
    '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_184_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(1.0)_184.csv',
    '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_184_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(1.5)_184.csv',
    '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_184_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(2.0)_184.csv',
    '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_184_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(3.0)_184.csv',
    '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_184_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(4.0)_184.csv',
    '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_184_sa_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-SA(5.0)_184.csv',
    ]
max_dir_sf = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1. ]

# # calc_id 187, SaAvg hazard curve, all trt, RotD100, Z1p0 = 913m, Z2p5 = 6.726 km, usgs basin scaling term using cb14, basin term above 1.0s
# haz_crv_file_list = [
#     '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/oq_hazard_data/final_computations/usgs_basin_term_cb14/calc_id_187_saAvg_all_trt_z1p0_913_z2p5_6p726_RotD100_usgs_basin_cb14_term/hazard_curve-mean-AvgSA_187.csv',
#     ]
# max_dir_sf = [1.]


# parse the specified hazard curves
haz_dict_temp = {}
for im, haz_file, mdsf in zip(IM_names, haz_crv_file_list, max_dir_sf):
    haz_dict_temp[im] = parse_haz_curve_csv(file = haz_file, T_years = 1)
    # scale for max direction
    haz_dict_temp[im]['iml'] = mdsf*haz_dict_temp[im]['iml']

haz_dict['hazCurves'] = haz_dict_temp

#%% Example risk-targeted ground motion calculation using ASCE 7 guidelines
haz_data = GroundMotionHazard.from_dict(haz_dict)
rtgm_data = BuildingCodeRTGMCalc.calc_rtgm(haz_data,bldg_code='ASCE7')

#%% Dictionary structure of output data
rtgm_data['summary']

#%% Summary plot of results
fig,axs = rtgm_data.plotFullRiskCalc('SA0P1');

#%%  Extract the MCEr data for plotting

# MCEr data from Hiro
periods_hiro = np.array([0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 
                      0.3, 0.4, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 7.5, 10])
MCEr_data_hiro = np.array([0.82, 0.82, 0.84, 0.86, 0.95, 1.14, 1.33, 1.59, 1.82, 
                       1.9, 1.93, 1.87, 1.69, 1.52, 1.39, 0.97, 0.75, 0.48, 
                       0.35, 0.27, 0.17, 0.12 ])

# get periods from SA IM names, extract corresponding MCEr values
periods = []
MCEr_data = []
UHS_data = []
for im in IM_names:
    periods.append( float(im[2]) + float(im[4:])/10**len(im[4:]) )
    MCEr_data.append( rtgm_data['summary'][im]['rtgm'] )
    UHS_data.append( rtgm_data['summary'][im]['uhgm'] )
    

fig, ax = plt.subplots(dpi = 200)

# plot the MCEr data
# ax.plot(periods, MCEr_data, 'ok', label = 'MCEr OpenQuake')
# plot UHS data (2in50)
ax.plot(periods, UHS_data, '*k', label = 'UHS OpenQuake')

# plot the data from hiro
ax.plot(periods_hiro, MCEr_data_hiro, 'xr', label = 'MCEr Hiro')
ax.legend()
# ax.set_xlim([0, 10])
# ax.set_xlim([0, ax.get_xlim()[1]])
# ax.set_ylim([0, ax.get_ylim()[1]])
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([0.1, 10])
ax.set_ylim([0.01, 5])
ax.set_xlabel('Period (s)')
ax.set_ylabel('Sa (g)')

#%% Plot the hazard curves


fig, ax = plt.subplots(dpi = 200)

im_to_plot = ['SA0P3', 'SA0P5', 'SA1P0', 'SA2P0', 'SA3P0']#, 'SA5P0']

for im in im_to_plot:#haz_data['hazCurves'].keys():
    haz_crv = haz_data['hazCurves'][im]
    ax.plot( haz_crv['iml'], haz_crv['afe'], label = im )

ax.legend()
ax.set_xlim([0.005, 10])
ax.set_ylim([1e-5, 1])
ax.set_xlabel('Sa (g)')
ax.set_ylabel('Annual frequency of exceedance')
ax.set_xscale('log')
ax.set_yscale('log')


#%% USGS hazard curves

file = '/Users/nbijelic/Research/EPFL/projects/Hiro/journal_paper/subduction_zone_data/basin_depth_info/static-hazard-curves-CONUS_2018-(-122.32,47.61).csv'

df = pd.read_csv(file, header = None)

haz_data_usgs = {
    'SA0P3' : { 'iml' : df.iloc[0].to_numpy(), 'afe' : df.iloc[1].to_numpy() },
    'SA0P5' : { 'iml' : df.iloc[2].to_numpy(), 'afe' : df.iloc[3].to_numpy() },    
    'SA1P0' : { 'iml' : df.iloc[4].to_numpy(), 'afe' : df.iloc[5].to_numpy() },
    'SA2P0' : { 'iml' : df.iloc[6].to_numpy(), 'afe' : df.iloc[7].to_numpy() },
    'SA3P0' : { 'iml' : df.iloc[8].to_numpy(), 'afe' : df.iloc[9].to_numpy() },    
    }
usgs_max_dir_sf = [1.206, 1.219, 1.25, 1.256, 1.261]

# compare USGS and OQ hazard curves

color_plot = ['k', 'r', 'b', 'm', 'c']

fig, ax = plt.subplots(dpi = 200)

im_to_plot = ['SA0P3', 'SA0P5', 'SA1P0', 'SA2P0', 'SA3P0']#, 'SA5P0']

for im, col in zip(im_to_plot, color_plot):#haz_data['hazCurves'].keys():
    haz_crv = haz_data['hazCurves'][im]
    ax.plot( haz_crv['iml'], haz_crv['afe'], label = im+'_oq', color = col )

# plot USGS hazard curves
for im, sf, col in zip(haz_data_usgs.keys(), usgs_max_dir_sf, color_plot):
    haz_crv = haz_data_usgs[im]
    ax.plot( sf*haz_crv['iml'], haz_crv['afe'], label = im+'_usgs',
            linestyle = '--', color = col)

# add 2475 year return period line
ax.plot( [0.005, 10], [1/2475, 1/2475], ':k', alpha = 1.0 )


ax.legend(loc = 'lower left', prop={'size': 9})
ax.set_xlim([0.005, 10])
ax.set_ylim([1e-5, 1])
ax.set_xlabel('Sa (g)')
ax.set_ylabel('Annual frequency of exceedance')
ax.set_xscale('log')
ax.set_yscale('log')







