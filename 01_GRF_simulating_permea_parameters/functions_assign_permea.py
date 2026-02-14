import numpy  as np
import pandas as pd
import pickle
import os
import pyvista as pv
import joblib
import geone
import geone.covModel as gcm
import geone.geosclassicinterface as gci
from numba import jit


###########
###########

def assign_permea_k(simu_arr, dict_means, dict_covmodels, spacing=(1,1,1), origin=(0,0,0), distri_type="Gaussian", nb_cpu=-1, nb_real_simu=1):
    
    """
    Function that takes facies simulation as inputs and populate them with property distributions. 
    simu_arr       : simulation array that are facies simulations [nlay, lny, nx]
    dict_means     : python dictionnary with facies ID as keys and means as values
    dict_covmodels : dictionary of geone covmodels for the GRF simulations of the facies
    spacing        : spacing of the simulation grid, to match with the variogramm info (x,y,z)
    origin         : origin of the model
    distri_type    : string, only Gaussian for now
    """
    
    #get info on the simulation.
    dim   = (simu_arr.shape[-1], simu_arr.shape[-2], 1)
    spa   = spacing
    ori   = origin
    
    #new simu array where the permeability are assigned.
    permea_field = np.ones((nb_real_simu, simu_arr.shape[0], simu_arr.shape[1], simu_arr.shape[2]))*-9999

    #list the facies presents in the simu array, we iterate through it.
    facies = np.unique(simu_arr[~np.isnan(simu_arr)])

    for iv in facies:
        #test that item are presents in both dictionnary
        assert iv in dict_means.keys(), 'Key {} of facies is note in dic_mean'.format(iv)
        assert iv in dict_covmodels.keys(), "Key {} of dic_means not in dic_covmodels".format(iv)
        
        #get vario model and mean value
        model = dict_covmodels[iv]
        mean  = dict_means[iv]
        
        #create mask to simulate grf only for one facies
        mask_facies = (simu_arr == iv)        

        #simulatation for facies i through all the simulation in the set
        for layer in range(simu_arr.shape[0]):
            #mask for the simulate layerr of the x facies
            mask_layer = mask_facies[layer]
            mask_layer[mask_layer]    = 1
            mask_layer[mask_layer!=1] = 0

            if np.sum(mask_layer==0) != np.prod(mask_layer.shape):
                #simulation sgs
                res = gci.simulate3D(model, dim, spa, ori, verbose=0, mask=mask_layer, nreal=nb_real_simu, mean=mean, nthreads=nb_cpu) 

                #copy value only in the mask
                mask                         = np.zeros((simu_arr.shape[-2], simu_arr.shape[-1]))
                mask[mask_facies[layer]]     = 1
                for nb_s in range(nb_real_simu):
                    permea_field[nb_s][layer][mask==1] = res["image"].val[nb_s][0][mask==1]
    
    #assign nan value
    permea_field[permea_field == -9999] = np.nan
    return 10**permea_field
