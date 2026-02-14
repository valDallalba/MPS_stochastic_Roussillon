import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import pickle
import geone

path_simu_mps = '/cluster/raid/data/valentin/new_grid_2022/00_simu_out_new_grid_mps/'

nb_simulation = 50
nb_inter      = 4
simu_mps      = []

#load simulations
for inte in range(nb_inter):
    simu_int = []
    
    for simu in range(nb_simulation):
        simu_int.append(joblib.load(os.path.join(path_simu_mps,'simu_{}_{:02}.pickle'.format(nb_inter-inte-1,simu))))

    simu_int = np.concatenate(simu_int, axis=0)
    simu_int = np.flipud(simu_int)
    simu_mps.append(simu_int)

simu_mps = np.concatenate(simu_mps, axis=1)

print(simu_mps.shape)
print('All simulaion are loaded !')


#simulation to image
simu_mps = geone.img.Img(simu_mps.shape[-1], simu_mps.shape[-2], simu_mps.shape[-3], nv=nb_simulation, 
              varname='mps_simu',val=simu_mps )

#calculate post stats
all_sim_stats = geone.img.imageCategProp(simu_mps, [3, 4])
all_stats = geone.img.imageCategProp(simu_mps,[0,1,2,3,4,5,6])
all_entro = geone.img.imageEntropy(all_stats)
all_entro_bis = geone.img.imageEntropy(all_stats, [3,4])

print('All the stats are calculated !')


#save output files
with open('output/mps_ensemble_stats.pickle','wb') as file:
    pickle.dump(all_sim_stats, file, pickle.HIGHEST_PROTOCOL)
    
with open('output/mps_ensemble_entropy.pickle','wb') as file:
    pickle.dump(all_entro, file, pickle.HIGHEST_PROTOCOL)

with open('output/mps_ensemble_entropy_riv.pickle','wb') as file:
    pickle.dump(all_entro_bis, file, pickle.HIGHEST_PROTOCOL)

print('All the files are saved!')
