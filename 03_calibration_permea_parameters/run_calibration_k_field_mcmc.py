import numpy  as np
import pandas as pd
import pickle
import os
import joblib
import pyvista as pv
import joblib

import geone
import geone.covModel as gcm
import geone.geosclassicinterface as gci

from numba import jit
from multiprocessing import Pool
import scipy
import argparse


#load functions
path_code = '../02_upscalling_permea/functions_upscalling_2022.py'
exec(open(path_code).read())
path_code = '../01_assign_permeability/functions_assign_permea_2022.py'
exec(open(path_code).read())
path_code = '../03_calibration_k_field/function_calibration.py'
exec(open(path_code).read())


#get input argument = number of cpu available
parser = argparse.ArgumentParser()
parser.add_argument("-cpu_nb", help="increase output verbosity")
args = parser.parse_args()

if args.cpu_nb:
        cpu_nb = int(args.cpu_nb)
else:
        cpu_nb = -1

from flopy.utils.gridintersect import GridIntersect
from flopy.utils import Raster


#load modflow model
model_dir  = './calibration_k_files_brut/'
model_name = '3D_mf6'
exe_name   = '/home/dallalba-arnauv/2022/flopy/exe_linux/mf6'


#load simulation and grid
simulation = fp.mf6.MFSimulation.load(sim_name=model_name, exe_name=exe_name, sim_ws=model_dir, verbosity_level=0)
gwf        = simulation.get_model()
idomain    = np.copy(gwf.dis.idomain.array)
grid       = fp.discretization.StructuredGrid(gwf.dis.delc.array,gwf.dis.delr.array, xoff=gwf.dis.xorigin.array, yoff=gwf.dis.yorigin.array) 
print(idomain.shape)

# retrieve some info
top    = gwf.dis.top.array
top    = top[np.newaxis,:,:]
botom  = gwf.dis.botm.array
layers = np.concatenate([top,botom])

#load data piezo
path_piezo  = '../03_calibration_k_field/pz_hydriad_modif.csv'
piezo       = pd.read_csv(path_piezo, delimiter=';')
piezo['id'] = piezo.index
piezo['cell_x'] = [grid.intersect(vx,vy)[1] for vx,vy in zip(piezo.x.values,piezo.y.values)]
piezo['cell_y'] = [grid.intersect(vx,vy)[0] for vx,vy in zip(piezo.x.values,piezo.y.values)]


#load one mps simulation
path_mps = '../00_simulation_mps/simu_out_old/'
nb_simu  = 1
nb_inter = 4
simu_mps = []

#load simulations
for inte in range(nb_inter):
    simu_int = []
    
    for simu in range(nb_simu):
        simu_int.append(joblib.load(os.path.join(path_mps,'simu_{}_{}.pickle'.format(nb_inter-inte-1,simu))))
    
    simu_int = np.concatenate(simu_int, axis=0)
    simu_int = np.flipud(simu_int)
    simu_mps.append(simu_int)

simu_mps = np.concatenate(simu_mps, axis=1)

#define variogramme model for GRF simulation (one covariance model can be assign to each facies)
cm_prop_0  = gcm.CovModel3D(elem = [("spherical", {"w":0.25,"r":[50,50,5]})], alpha=0)   #TO MODIFY
cm_prop_1  = gcm.CovModel3D(elem = [("spherical", {"w":0.25,"r":[50,50,5]})], alpha=0)   #TO MODIFY
cm_prop_2  = gcm.CovModel3D(elem = [("spherical", {"w":0.25,"r":[10,10,5]})], alpha=80)  #TO MODIFY
cm_prop_3  = gcm.CovModel3D(elem = [("spherical", {"w":0.25,"r":[100,5,5]})], alpha=0)   #TO MODIFY
cm_prop_4  = gcm.CovModel3D(elem = [("spherical", {"w":0.25,"r":[50,5,5]})], alpha=-20)  #TO MODIFY
cm_prop_5  = gcm.CovModel3D(elem = [("spherical", {"w":0.25,"r":[10,10,5]})], alpha=0)   #TO MODIFY

#define dictionnary
dic_cm_0      = {0:cm_prop_0, 1:cm_prop_1, 2:cm_prop_2, 3:cm_prop_3, 4:cm_prop_4, 5:cm_prop_5}

#mean_values_0 = {0: -6, 1:-6, 2:-4.5, 3:-4, 4:-4, 5:-5} 
values_0      = [-6, -6, -4.5, -4, -4, -5]
bounds        = [(-7,-5.5), (-7,-5.5), (-5.5,-4), (-4.5,-3.5), (-4.5,-3.5), (-5.5,-4.5)]

#path
path_alpha  = '../02_upscalling_permea/data/alpha_list.pickle'
path_index  = '../02_upscalling_permea/data/list_index.pickle'

#load
list_index, dim, dim_upsc = joblib.load(path_index)
alpha_list                = joblib.load(path_alpha)

#optimisation
np.random.seed(19932013)
nb_simu     = 35
score_ini   = 10
proba_get   = 0.05
values_list = np.array([np.random.uniform(b[0],b[1],size=1) for b in bounds])[:,0]
step        = np.random.normal(scale=0.5,size=len(values_list))*0.1
values      = values_list+step

print(simu_mps[0].shape)
    
for i in range(nb_simu):
    score_t = optim_k_field_brut(values,  [piezo, dic_cm_0, simu_mps[0], list_index, dim, dim_upsc, alpha_list, cpu_nb])
    
    if score_t<score_ini:        
        with open('results_optim_k_brut_best.pickle','wb') as file:
            pickle.dump([score_t, values], file)

        score_ini = score_t
        step      = np.random.normal(scale=0.5,size=len(values_list))*0.1
        values    = values + step 
    

    else:
        values_list = np.array([np.random.uniform(b[0],b[1],size=1) for b in bounds])[:,0]
        step        = np.random.normal(scale=0.5,size=len(values_list))*0.1
        new_values  = values_list+step
        values      = [new_values, values+step][np.random.choice([0,1], p=[1-0.05, 0.05])]

    with open('results_files/results_optim_k_brut_{:02}.pickle'.format(i),'wb') as file:
            pickle.dump([score_t,values], file)
    
