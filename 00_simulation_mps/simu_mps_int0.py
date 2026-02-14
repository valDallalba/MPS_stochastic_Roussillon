import numpy as np
import pandas as pd
import os
import pickle
import joblib
from geone import img
from geone import imgplot as imgplt
import geone.deesseinterface as dsi
exec(open('functions_simu_mps_roussilon.py').read())


###
#Check that hd varname = ['X', 'Y', 'Z', 'facies']
###

###
#define some input parameters
###

interval   = 0   #to change for each script
nb_simu    = 1  #to change after test
simu_type  = 6   #either 3 or 6 corresponding to the facies description
seed       = 98769876 + interval * nb_simu + simu_type 

###
#import data
###
#path
path_data  = 'data/hd/data_int{}_6f.pickle'.format(interval)
path_param = 'data/param_mps_6f_2021.json'

#load
data = joblib.load(path_data)
grid, trend_1, trend_2, rotation, hd = data
param_mps = read_json(path_param)

###
#run simulation
###
simulation    = simu_mps_2021_run(hd, trend_1, trend_2, rotation, grid, nb_simu, param_mps, seed, interval, simu_type)
#proba_map    = simu_mps_2021_proba(simulation)
#entropy_map  = simu_mps_2021_entro(proba_map)

###
#save output
###
#path
#path_simu = 'simu_out/simu_int{0}.pickle'.format(interval)
#path_prob = 'simu_out/simu_prob_{}.pickle'.format(interval)
#path_entr = 'simu_out/simu_entropy_{}.pickle'.format(interval)

#save
#joblib.dump(simulation,   path_simu,compress=3, protocol=pickle.HIGHEST_PROTOCOL)
#joblib.dump(proba_map,    path_prob,compress=3, protocol=pickle.HIGHEST_PROTOCOL)
#joblib.dump(entropy_map,  path_entr,compress=3, protocol=pickle.HIGHEST_PROTOCOL)
#2oblib.dump(most_frq_map, path_most,compress=3, protocol=pickle.HIGHEST_PROTOCOL)
