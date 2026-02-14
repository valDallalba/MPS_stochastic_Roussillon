import numpy  as np
import pandas as pd
import os
import joblib
import pyvista as pv
import joblib

import geone
import geone.covModel as gcm
import geone.geosclassicinterface as gci

from numba import jit
from multiprocessing import Pool
import pickle5 as pickle
import argparse

#####
#####

parser = argparse.ArgumentParser()
parser.add_argument("-sim", help="increase output verbosity")
args = parser.parse_args()

if args.sim:
        simu_id = int(args.sim)
else:
        simu_id = 0
#####
#####

#load functions
path_code = 'functions_upscalling.py'
exec(open(path_code).read())

#import the different requiered grids
path_permea = '/cluster/raid/data/valentin/new_grid_2022/01_permea_out_test/'
path_alpha  = 'data/alpha_list_2023.pickle'
path_index  = 'data/list_index_2023.pickle'
path_save   = './upscalled_out_test/'

#load simulation    
simu_permea = joblib.load(os.path.join(path_permea,'simu_permea_00_0_{}.pickle'.format(simu_id)))

#load
with open(path_index, 'rb') as file:
    list_index, dim, dim_upsc = pickle.load(file)

with open(path_alpha, 'rb') as file:
    alpha_list                = pickle.load(file)

#dimension of the cell to upscalled
shape_upsc = [int(d/d_ups) for d, d_ups in zip(dim, dim_upsc)]

#upscalling of the permea field (we iterate through the simulations)
permea_upd  = upd_permea(simu_permea, dim)
permea_upsc = upsc_permea_map(permea_upd, list_index, dim_upsc, shape_upsc, alpha_list, 'aniso',worker_pool_number=6)    

with open(path_save+'simu_permea_upsc_00_0_{}.pickle'.format(simu_id), 'wb') as file:
    pickle.dump(permea_upsc, file)
