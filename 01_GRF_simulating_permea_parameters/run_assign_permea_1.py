import numpy as np
import pandas as pd
import os
import pickle
import joblib
from geone import img
from geone import imgplot as imgplt
import geone.deesseinterface as dsi
exec(open('functions_assign_permea.py').read())

import numpy


path_facies = '/cluster/raid/data/valentin/new_grid_2022/simu_out_new_grid_mps/'
path_perma  = 'permea_out/'

#nunber of simulations to load
simu_start   = 0
nb_simu      = 20
nb_inter     = 4
nb_real_simu = 2
simu_all     = []

np.random.seed(seed=1245*nb_simu*nb_inter+20+simu_start)

nb_simu = range(simu_start, simu_start+nb_simu)


#define variogramme model for GRF simulation (one covariance model can be assign to each facies)
cm_prop_0  = gcm.CovModel3D(elem = [("spherical", {"w":0.25,"r":[50,50,5]})], alpha=0)   #TO MODIFY
cm_prop_1  = gcm.CovModel3D(elem = [("spherical", {"w":0.25,"r":[50,50,5]})], alpha=0)   #TO MODIFY
cm_prop_2  = gcm.CovModel3D(elem = [("spherical", {"w":0.25,"r":[10,10,5]})], alpha=80)    #TO MODIFY
cm_prop_3  = gcm.CovModel3D(elem = [("spherical", {"w":0.25,"r":[100,5,5]})], alpha=0)    #TO MODIFY
cm_prop_4  = gcm.CovModel3D(elem = [("spherical", {"w":0.25,"r":[50,5,5]})], alpha=-20)  #TO MODIFY
cm_prop_5  = gcm.CovModel3D(elem = [("spherical", {"w":0.25,"r":[10,10,5]})], alpha=0)     #TO MODIFY

#define dictionnary
dic_cm = {0:cm_prop_0, 1:cm_prop_1, 2:cm_prop_2, 3:cm_prop_3, 4:cm_prop_4, 5:cm_prop_5}

#define the log10 mean values for each facies 
#0 : Plaine innondation, 1: palustre, 2: crevasse, 3:barre accretion, 4:chenaux, 5:cone alluviaux
mean_values = {0: -6, 1:-5, 2:-4, 3:-3, 4:-3, 5:-4} 
mean_values = {0: -5.75, 1:-6, 2:-4.35, 3:-3.75, 4:-4, 5:-5} 


#load simulations
for inte in range(nb_inter):
    simu_int = []
    
    for simu in nb_simu:
        simu_int.append(joblib.load(os.path.join(path_facies,'simu_{}_{:02}.pickle'.format(nb_inter-inte-1,simu))))
    
    simu_int = np.concatenate(simu_int, axis=0)
    simu_int = np.flipud(simu_int)
    simu_all.append(simu_int)

simu_permea = np.concatenate(simu_all,axis=1)

for i,permea_f in enumerate(simu_permea):
    permea_k    = assign_permea_k(permea_f, mean_values, dic_cm, nb_cpu=31, nb_real_simu=nb_real_simu)

    for s in range(nb_real_simu):
        joblib.dump(permea_k[s], 'permea_out/simu_permea_{:02}_{}.pickle'.format(nb_simu[i], s), compress=3)

    print('Simu {} has is permeability assigned !'.format(nb_simu[i]))

print('All simu are done !')
