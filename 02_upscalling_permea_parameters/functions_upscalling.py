import numpy  as np
import pandas as pd
import pickle
import os
import pyvista as pv
import joblib
import matplotlib.pyplot  as plt
from matplotlib import colors
import geone
import geone.covModel as gcm
import geone.geosclassicinterface as gci

from numba import jit
from multiprocessing import Pool


##########
##########

def plot_3d(array, sx=1, sy=1, sz=1, threshold=False, v1=0, v2=1, show_edge=False, show_grid=False, show_3d=False):
    array = np.swapaxes(array,0,2)
    
    # Create the spatial reference
    grid = pv.UniformGrid()

    # Set the grid dimensions: shape + 1 because we want to inject our values on the CELL data
    grid.dimensions = np.array(array.shape)+1
    grid.spacing    = (sx,sy,sz)

    # Add the data values to the cell data
    grid.cell_data["values"] = array.flatten(order="F")  # Flatten the array!

    p       = pv.Plotter() 
    outline = grid.outline()
    edge    = grid.extract_all_edges()
    
    if threshold:
        thresh  = grid.threshold([v1,v2])
        p.add_mesh(thresh)
    else:
        p.add_mesh(grid)
        
    if show_edge:
        p.add_mesh(edge,color='k')
    if show_grid:
        p.add_mesh(outline,color='k')   
    
    if show_3d:
        p.show(jupyter_backend='static')
   
    return grid


##########
##########

def plot_slice(grid, dic_slice, show_edge=True, show_grid=True):
    
    slices = grid.slice_orthogonal(x=dic_slice['x'], y=dic_slice['y'], z=dic_slice['z'])
    slices.plot(show_edges=show_edge, show_grid=show_grid, jupyter_backend='static')
    
    return 


##########
##########

def plot_multi_slice(grid, nb_slice=['None','None','None'], axis='x', threshold=False, v1=0, v2=1, show_edge=False, show_grid=False):
    
    outline = grid.outline()
    edge    = grid.extract_all_edges()
    
    p = pv.Plotter(lighting='none')    
    
    if threshold:
        thresh  = grid.threshold([v1,v2])
        if nb_slice[0]!='None':
            slices  = thresh.slice_along_axis(n=int(nb_slice[0]), axis='x')
            p.add_mesh(slices)
        if nb_slice[1]!='None':
            slices  = thresh.slice_along_axis(n=int(nb_slice[1]), axis='y')
            p.add_mesh(slices)
        if nb_slice[2]!='None':
            slices  = thresh.slice_along_axis(n=int(nb_slice[2]), axis='z')
            p.add_mesh(slices)

    else:    
        slices  = grid.slice_along_axis(n=6, axis=axis)
        p.add_mesh(slices)
    
    if show_edge:
        p.add_mesh(edge,color='k')
    if show_grid:
        p.add_mesh(outline,color='k')
    
    p.show(jupyter_backend='static')
    
    return 


##########
##########
@jit
def harmo_mean(values):
    if np.sum(np.isnan(values))==len(values):
        return np.nan 
    else:
        return np.sum(~np.isnan(values))/(np.sum(1/values[~np.isnan(values)]))
@jit
def arithm_mean(values):
    return np.nanmean(values)
@jit 
def geom_mean(values):
    a = np.log(values)
    return np.exp(np.nanmean(a))

##########
##########

@jit
def k1(grid_values):
    '''
    Cardwell Max
    '''
    k1 = np.apply_along_axis(arithm_mean, 1, grid_values)
    k1 = np.apply_along_axis(arithm_mean, 0 , k1)
    k1 = np.apply_along_axis(harmo_mean, 0, k1)
    return k1
@jit
def k4(grid_values):
    k2 = np.apply_along_axis(arithm_mean, 1, grid_values)
    k2 = np.apply_along_axis(harmo_mean, 1, k2)
    k2 = np.apply_along_axis(arithm_mean, 0, k2)
    return k2
@jit
def k3(grid_values):
    k3 = np.apply_along_axis(arithm_mean, 0, grid_values)
    k3 = np.apply_along_axis(harmo_mean, 1, k3)
    k3 = np.apply_along_axis(arithm_mean, 0, k3)
    return k3
@jit
def k1b(grid_values):
    k1b = np.apply_along_axis(arithm_mean, 0, grid_values)
    k1b = np.apply_along_axis(arithm_mean, 0, k1b)
    k1b = np.apply_along_axis(harmo_mean, 0, k1b)
    return k1b
@jit
def k2(grid_values):
    k2 = np.apply_along_axis(harmo_mean, 2, grid_values)
    k2 = np.apply_along_axis(arithm_mean, 0, k2)
    k2 = np.apply_along_axis(arithm_mean, 0, k2)
    return k2
@jit
def k2b(grid_values):
    '''Cardwell Min'''
    k2b = np.apply_along_axis(harmo_mean, 2, grid_values)
    k2b = np.apply_along_axis(arithm_mean, 1, k2b)
    k2b = np.apply_along_axis(arithm_mean, 0, k2b)
    return k2b


###########
###########

@jit
def get_k_equi(grid_values, mode='iso', alpha_sub_list=None):
    kv = np.zeros(6)
    
    kv[0] = k1(grid_values)
    kv[1] = k1b(grid_values)
    kv[2] = k2(grid_values)
    kv[3] = k2b(grid_values)
    kv[4] = k3(grid_values)
    kv[5] = k4(grid_values)
    
    if mode == 'iso':
        return kmean_kruel_iso(kv), kv
    
    if mode == 'aniso':
        return kmean_kruel_aniso(kv, alpha_sub_list), kv
    
    else:
        print('error in the selected mode')
        return None
    
        
###########
###########
@jit
def kmean_kruel_iso(list_k):
    '''
    Use the geometric mean propose by Kruel_Romeu for delta cells isotrope
    '''
    return geom_mean(list_k)    


###########
###########
@jit
def kmean_kruel_aniso(list_k, alpha_sub_list):
    '''
    Use the power decomposition propose by Kruel-Romeu controlling anisotropie of the cell
    '''
    k1 = list_k[0]
    k2 = list_k[2]
    k3 = list_k[4]
    k4 = list_k[5]
    
    alpha_y2, alpha_z2, alpha_y3, alpha_z3 = alpha_sub_list
    expo1 = alpha_y2*alpha_z3 + alpha_z2*alpha_y3
    expo2 = 1-alpha_y3-alpha_z3
    expo3 = (1-alpha_y2)*alpha_z3
    expo4 = (1-alpha_z2)*alpha_y3

    return (k1**expo1)*(k2**expo2)*(k3**expo3)*(k4**expo4)


###########
###########

@jit
def get_k_equi_3d(grid_values, mode='iso', alpha_list=None):
    '''
    Get the k equivalent for the 3 main directions using the Kruel Romeu approach.
    The grid to upscalled is rotated in two directions in order to get kxxb, kyyb and kzzb.
    '''
    
    kxxb = get_k_equi(grid_values, mode, alpha_list)[0]
    kyyb = get_k_equi(np.rot90(grid_values,k=1,axes=(1,2)), mode, alpha_list)[0]
    kzzb = get_k_equi(np.rot90(grid_values,k=1,axes=(2,0)), mode, alpha_list)[0]
    
    return kxxb, kyyb, kzzb



@jit
def get_k_equi_3d_map(grid_values):
    '''
    Get the k equivalent for the 3 main directions using the Kruel Romeu approach.
    The grid to upscalled is rotated in two directions in order to get kxxb, kyyb and kzzb.
    '''
    grid_values, mode, alpha_sub_list = grid_values
    kxxb = get_k_equi(grid_values, mode, alpha_sub_list[0])[0]
    kyyb = get_k_equi(np.rot90(grid_values,k=1,axes=(1,2)), mode, alpha_sub_list[1])[0]
    kzzb = get_k_equi(np.rot90(grid_values,k=1,axes=(2,0)), mode, alpha_sub_list[2])[0]
    
    return kxxb, kyyb, kzzb
    

###########
###########

def create_upscalled_list(arr_k, sx, sy, sx_up, sy_up, nlay, nlay_up, check_nan=True):
    '''
    Create a list of i, j and z index coupled to dx, dy, dz cell size in order to sample sub-set of permeability value from the sub-grid to the super-grid.
    We can pass a permeabilit field to the function in order to return a flag indice with the list to detect the empty sub-set.
    Array_k is a 3d array
    '''

    #array to update
    arr_to_upd = np.copy(arr_k[0]) #[ny,nx]
    
    #Define dimension and spacing.
    nx, ny       = arr_k.shape[-1], arr_k.shape[-2]
    nx_up, ny_up = nx*sx/sx_up, ny*sy/sy_up
    sx_up, sy_up = sx_up, sy_up
    
    #repetition vertical of the grouped 
    nb_z = int(nlay/nlay_up)

    #Add empty columns until the extend of the upscalled grid matches the extend of the original grid. 
    while nx_up/int(nx_up)!= 1.0:
        add_h      = np.full((arr_to_upd.shape[0],1), np.nan)
        arr_to_upd = np.hstack((arr_to_upd, add_h))
        nx         = arr_to_upd.shape[1]
        nx_up      = nx*sx/sx_up
        
    #Add empty lines until the extend of the upscalled grid matches the extend of the original grid. 
    while ny_up/int(ny_up)!= 1.0:
        add_v      = np.full((1,arr_to_upd.shape[1]), np.nan) 
        arr_to_upd = np.vstack((arr_to_upd,add_v)) 
        ny         = arr_to_upd.shape[0]
        ny_up      = ny*sy/sy_up

    #Update the permeability array
    arr_upd = upd_permea(arr_k, (nlay, ny, nx))

    #get the spacing
    dx      = int(nx/nx_up)
    dy      = int(ny/ny_up)
    dz      = int(nlay/nlay_up)

    #create list array, with spacing position, and check nan flag
    list_arr = []
    for z in range(int(nlay_up)):
        for j in range(int(ny_up)):
            for i in range(int(nx_up)):
                pz = [dz*z, dz*(z+1)]
                py = [dy*j, dy*(j+1)]
                px = [dx*i, dx*(i+1)]

                if check_nan is True:
                    values = arr_upd[pz[0]:pz[1],py[0]:py[1],px[0]:px[1]]
                    if np.sum(np.isnan(values))==np.prod(values.shape):
                        check = 0
                    else:
                        check = 1

                else:
                    check = 1

                list_arr.append([[pz,py,px],check])
    return list_arr, (nlay,ny,nx), (int(nlay_up),int(ny_up),int(nx_up))


###########
###########

def create_upscalled_index(arr_k, sx, sy, sx_up, sy_up, nlay, nlay_up):
    '''
    OLD
    Create a 3D map of the group filter of the sub-cells toward their super-cell group identification.
    The input is a 2D grid.
    '''

    #array to update
    arr_to_upd = np.copy(arr_k) #[ny,nx]
    
    #Define dimension and spacing.
    nx, ny       = arr_k.shape[-1], arr_k.shape[-2]
    nx_up, ny_up = nx*sx/sx_up, ny*sy/sy_up
    sx_up, sy_up = sx_up, sy_up
    
    #repetition vertical of the grouped 
    nb_z = int(nlay/nlay_up)

    #Add empty columns until the extend of the upscalled grid matches the extend of the original grid. 
    while nx_up/int(nx_up)!= 1.0:
        add_h      = np.full((arr_to_upd.shape[0],1), np.nan)
        arr_to_upd = np.hstack((arr_to_upd, add_h))
        nx         = arr_to_upd.shape[1]
        nx_up      = nx*sx/sx_up
        
    #Add empty lines until the extend of the upscalled grid matches the extend of the original grid. 
    while ny_up/int(ny_up)!= 1.0:
        add_v      = np.full((1,arr_to_upd.shape[1]), np.nan) 
        arr_to_upd = np.vstack((arr_to_upd,add_v)) 
        ny         = arr_to_upd.shape[0]
        ny_up      = ny*sy/sy_up

    #coordinnates of the original grid cells.
    coord_x = np.arange(sx/2, nx*sx+sx/2, sx)
    coord_y = np.arange(sy/2, ny*sy+sy/2, sy)
    
    #For each cells of the original grid we assign its supercell groupe number (flattend position).
    #The supercell number is based from the modulo of the location of the cell regaring the new spacing.
    p_x = [int(np.trunc(cx/sx_up)) for cx in coord_x]
    p_y = [int(np.trunc((cy/sy_up))*nx) for cy in coord_y]
    p   = np.array([[px+py] for px,py in zip(p_x,p_y)])

    #associate the coordonnates to create a grid index for super cell selection #### On peut faire cette étape en dehors pour n'avoir à la répéter qu'une seul fois
    zones = np.zeros(arr_to_upd.shape)
    for j in range(ny):
        for i in range(nx):
            zones[j,i] = p_x[i]+p_y[j]+1
    
    #clean the zones index (can be removed to accelerate the running time)
    count = 1
    for val in np.sort(np.unique(zones)):
        zones[zones==val] = count
        count += 1
        
    #create the 3D upscalled grid cell number groups
    zones_3d = np.ones((nlay, ny, nx))
    count    = 0
    for z in range(nlay_up):
        zones_3d[count:count+nb_z+1,:,:] = zones
        count += nb_z
        zones += np.nanmax(zones)
        
    return zones_3d, (nlay,ny,nx), (int(nlay_up),int(ny_up),int(nx_up))


###########
###########

@jit
def upd_permea(permea, dim_upd):
    '''
    Update the permeability field to match the requiered dimension for upscalling.
    '''
    
    nx, ny       = permea.shape[-1], permea.shape[-2]
    nx_up, ny_up = dim_upd[-1], dim_upd[-2]
    permea_out = np.copy(permea)

    #Add empty columns until the extend of the permea grid matches the requiered dimension
    while nx<nx_up:
        permea     = np.insert(permea, nx, np.nan, 2) 
        nx         = permea.shape[-1]
        permea_out = np.copy(permea)   

    #Add empty lines until the extend of the permea grid matches the requiered dimension
    while ny<ny_up:
        permea     = np.insert(permea, ny, np.nan, 1) 
        ny         = permea.shape[-2]
        permea_out = np.copy(permea)   

    return permea_out


###########
###########

def upsc_permea(permea, zone_upsc, unique_ind, dxdy, dxdz, dim_upsc, shape_upsc, mode='iso'):
    '''
    OLD
    mode = iso or aniso
    '''
    grid_upsc = np.full((3,dim_upsc[-3],dim_upsc[-2],dim_upsc[-1]), np.nan)
    count     = 0
    
    if mode=='iso':
        for z in range(dim_upsc[-3]):
            for y in range(dim_upsc[-2]):
                for x in range(dim_upsc[-1]):
                    data_to_upsc       = np.reshape(permea[zone_upsc==unique_ind[count]], shape_upsc)
                    
                    if np.sum(np.isnan(data_to_upsc))!=np.prod(shape_upsc):
                        upscalled_values   = get_k_equi_3d(data_to_upsc,mode)
                        
                    else:
                        upscalled_values = [np.nan, np.nan, np.nan]
                        
                    grid_upsc[:,z,y,x] = upscalled_values
                    count += 1
                    
    else :
        for z in range(dim_upsc[-3]):
            for y in range(dim_upsc[-2]):
                for x in range(dim_upsc[-1]):
                    data_to_upsc       = np.reshape(permea[zone_upsc==unique_ind[count]], shape_upsc)
                    
                    if np.sum(np.isnan(data_to_upsc))!=np.prod(shape_upsc):
                        alpha_list         = get_alpha_expo(zone_upsc,unique_ind,count,dxdy,dxdz,shape_upsc)
                        upscalled_values   = get_k_equi_3d(data_to_upsc,mode,alpha_list)                    
                        
                    else:
                        upscalled_values = [np.nan, np.nan, np.nan]
                        
                    grid_upsc[:,z,y,x] = upscalled_values
                    count += 1
                    
    return grid_upsc


###########
###########

def upsc_facies_map(facies_arr, list_index, dim_upsc, nb_worker=1):
    '''
    upscalled a facies array based on the most frequent values.
    works in parallele.
    '''
    worker_pool_number = nb_worker
        
    #create the list of value for upscalling using the map function
    unique_list_grp    = [(facies_arr[elt[0][0][0]:elt[0][0][1],elt[0][1][0]:elt[0][1][1],elt[0][2][0]:elt[0][2][1]]) for elt in list_index]
    index_to_upsc      = [False if elt[1] == 0 else True for elt in list_index]

    #select only the non nan fulled values of the subgroups
    unique_list_clean  = np.array(unique_list_grp)[index_to_upsc]

    #initialise the function for the different workers
    f = get_most_freq_f_mix(unique_list_clean[0])

    #distribute the job to the different worker using pool and map function
    with Pool(worker_pool_number) as p:
        upscalled_grp   = p.map(get_most_freq_f_mix, unique_list_clean)

    #create the final outputs reshaped        
    upscalled_grp_full = np.full((1,len(unique_list_grp)), np.nan) 
    facies_upsc          = np.zeros((1, dim_upsc[-3], dim_upsc[-2],dim_upsc[-1]))
    
    upscalled_grp_full[0][index_to_upsc] = np.array(upscalled_grp)  
    facies_upsc[0][:] = np.reshape(upscalled_grp_full[0], (dim_upsc[-3], dim_upsc[-2],dim_upsc[-1]))      
                    
    return facies_upsc


###########
###########
@jit
def get_most_freq_f(array_f):
    '''
    return most frequent facies of a input array
    '''
    unique, count = np.unique(array_f[~np.isnan(array_f)], return_counts=True)
    most_f        = np.int(unique[np.argmax(count)])
    
    return most_f

###########
###########
@jit
def get_most_freq_f_cheat(array_f, cheat_value=0.35):
    '''
    return most frequent facies of a input array
    '''
    unique, count = np.unique(array_f[~np.isnan(array_f)], return_counts=True)

    if (3 in unique):
        percent = count/np.sum(count)
        if percent[unique==3]>=cheat_value:
            most_f = 3

        else:
            most_f = np.int(unique[np.argmax(count)])

    elif (4 in unique):
        percent = count/np.sum(count)
        if percent[unique==4]>=cheat_value:
            most_f = 4
        else:
            most_f = np.int(unique[np.argmax(count)])
    else: 
        most_f = np.int(unique[np.argmax(count)])
        
    return most_f

###########
###########

def upsc_permea_map(permea, list_index, dim_upsc, shape_upsc, alpha_list=None, mode='iso',worker_pool_number=6):
    '''
    mode = iso or aniso
    '''
        
    if mode=='iso':       
        #create the list of value for upscalling using the map function
        unique_list_grp    = [(permea[elt[0][0][0]:elt[0][0][1],elt[0][1][0]:elt[0][1][1],elt[0][2][0]:elt[0][2][1]],mode, None) for elt in list_index]
        index_to_upsc      = [False if elt[1] == 0 else True for elt in list_index]

        #select only the non nan fulled values of the subgroups
        unique_list_clean  = np.array(unique_list_grp)[index_to_upsc]

        #initialise the function for the different workers
        f = get_k_equi_3d_map(unique_list_clean[0])

        #distribute the job to the different worker using pool and map function
        with Pool(worker_pool_number) as p:
            upscalled_grp   = p.map(get_k_equi_3d_map, unique_list_clean)

        #create the final outputs reshaped        
        upscalled_grp_full = np.full((3,len(unique_list_grp)), np.nan) 
        grid_upsc          = np.zeros((3, dim_upsc[-3], dim_upsc[-2],dim_upsc[-1]))

        for i in range(3):
            upscalled_grp_full[i][index_to_upsc] = np.array([values[i] for values in upscalled_grp])  
            grid_upsc[i][:] = np.reshape(upscalled_grp_full[i], (dim_upsc[-3], dim_upsc[-2],dim_upsc[-1]))    
    
    else :        
        #create the list of value for upscalling using the map function
        unique_list_grp    = [(permea[elt[0][0][0]:elt[0][0][1],elt[0][1][0]:elt[0][1][1],elt[0][2][0]:elt[0][2][1]],mode, alpha) for elt,alpha in zip(list_index, alpha_list)]
        index_to_upsc      = [False if elt[1] == 0 else True for elt in list_index]

        #select only the non nan fulled values of the subgroups
        unique_list_clean  = np.array(unique_list_grp)[index_to_upsc]

        #initialise the function for the different workers
        f = get_k_equi_3d_map(unique_list_clean[0])

        #distribute the job to the different worker using pool and map function
        with Pool(worker_pool_number) as p:
            upscalled_grp   = p.map(get_k_equi_3d_map, unique_list_clean)

        #create the final outputs reshaped        
        upscalled_grp_full = np.full((3,len(unique_list_grp)), np.nan) 
        grid_upsc          = np.zeros((3, dim_upsc[-3], dim_upsc[-2],dim_upsc[-1]))

        for i in range(3):
            upscalled_grp_full[i][index_to_upsc] = np.array([values[i] for values in upscalled_grp])  
            grid_upsc[i][:] = np.reshape(upscalled_grp_full[i], (dim_upsc[-3], dim_upsc[-2],dim_upsc[-1]))    
                    
    return grid_upsc



###########
###########

@jit
def get_alpha_expo(zone_upsc, unique_ind, count, dxdy, dxdz, shape_upsc):
    '''
    Calculate the exponents for the Kruel Romeu upscalling approach
    Old approach
    '''
    dx_dy = np.mean(dxdy[zone_upsc==unique_ind[count]])
    dx_dz = np.mean(dxdz[zone_upsc==unique_ind[count]])
    
    ay = (kyy/kxx) * (dx_dy)**2
    az = (kzz/kxx) * (dx_dz)**2

    alpha_y2 = np.arctan(np.sqrt(ay)) / (np.pi/2)
    alpha_z2 = np.arctan(np.sqrt(az)) / (np.pi/2)

    alpha_y3 = (alpha_y2*(1-alpha_z2)) / (1-alpha_y2*alpha_z2)
    alpha_z3 = (alpha_z2*(1-alpha_y2)) / (1-alpha_y2*alpha_z2)
    
    return [alpha_y2, alpha_z2, alpha_y3, alpha_z3]
                              
                              
###########
###########

@jit
def get_alpha_expo_map(index_pos, dxdy, dxdz, shape_upsc):
    '''
    Calculate the exponents for the Kruel Romeu upscalling approach
    '''
    kyy, kxx, kzz = 1,1,1

    pz, py, px  = index_pos

    dx_dy = np.nanmean(dxdy[pz[0]:pz[1],py[0]:py[1],px[0]:px[1]])
    dx_dz = np.nanmean(dxdz[pz[0]:pz[1],py[0]:py[1],px[0]:px[1]])
    
    ay = (kyy/kxx) * (dx_dy)**2
    az = (kzz/kxx) * (dx_dz)**2

    alpha_y2 = np.arctan(np.sqrt(ay)) / (np.pi/2)
    alpha_z2 = np.arctan(np.sqrt(az)) / (np.pi/2)

    alpha_y3 = (alpha_y2*(1-alpha_z2)) / (1-alpha_y2*alpha_z2)
    alpha_z3 = (alpha_z2*(1-alpha_y2)) / (1-alpha_y2*alpha_z2)
    
    return [alpha_y2, alpha_z2, alpha_y3, alpha_z3]


###########
###########

@jit
def test_percent(percent, unique, value, proba_dic):
    for p in proba_dic:
        
        if percent[unique==value][0]>=p:
            return proba_dic[p]
        
    return 99

###########
###########

@jit
def get_most_freq_f_mix(array_f):
    '''
    return most frequent facies of a input array
    '''

    che_h  = 0.4
    che_m  = 0.3
    che_l  = 0.2

    bar_h  = 0.4
    bar_m  = 0.3
    bar_l  = 0.2

    crevasse = 0.3

    new_to_old = {0:0, 1:1, 2:2, 3:5, 4:8, 5:9}
    new_v      = {'pi':0, 'palu':1, 'crevasse':2, 'bar_l':3, 'bar_m':4, 'bar_h':5, 'che_l':6, 'che_m':7, 'che_h':8, 'cone':9}

    p_chenaux  = {che_h:new_v['che_h'], che_m:new_v['che_m'], che_l:new_v['che_l']}
    p_bar      = {bar_h:new_v['bar_h'], bar_m:new_v['bar_m'], bar_l:new_v['bar_l']}
    p_crevasse = {crevasse:new_v['crevasse']}


    unique, count = np.unique(array_f[~np.isnan(array_f)], return_counts=True)
    percent = count/np.sum(count)

    #if chenaux is present
    if (4 in unique):  
        #test chenaux
        r = test_percent(percent, unique, 4, p_chenaux)
        
        #test bar
        if (r==99)&(3 in unique):
            r = test_percent(percent, unique, 3, p_bar)
            
            #test crevasse
            if (r==99)&(2 in unique):
                r = test_percent(percent, unique, 2, p_crevasse)
                
                #assign default
                if r==99:
                    most_f = new_to_old[np.int(unique[np.argmax(count)])]
                    
                #assign crevasse
                else:
                    most_f = r
            
            elif r==99:
                most_f = new_to_old[np.int(unique[np.argmax(count)])]

            else:
                most_f = r

        #test crevasse
        elif (r==99)&(2 in unique):
            r = test_percent(percent, unique, 2, p_crevasse)
            
            #assign default
            if r==99:
                most_f = new_to_old[np.int(unique[np.argmax(count)])]

            #assign crevasse
            else:
                most_f = r

        #assign default
        elif r==99:
            most_f = new_to_old[np.int(unique[np.argmax(count)])]
        
        #assign chenaux
        else:
            most_f = r

    #if bar is present but no chenaux
    elif (3 in unique):
        r = test_percent(percent, unique, 3, p_bar)
            
        #test crevasse
        if (r==99)&(2 in unique):
            r = test_percent(percent, unique, 2, p_crevasse)

            #assign default
            if r==99:
                most_f = new_to_old[np.int(unique[np.argmax(count)])]

            #assign crevasse
            else:
                most_f = r

        #assign default
        elif r==99:
            most_f = new_to_old[np.int(unique[np.argmax(count)])]
        
        #assign crevasse
        else:
            most_f = r

    #test crevasse
    elif (2 in unique):
        r = test_percent(percent, unique, 2, p_crevasse)

        #assign default
        if r==99:
            most_f = new_to_old[np.int(unique[np.argmax(count)])]

        #assign crevasse
        else:
            most_f = r
            
    else:
        most_f = new_to_old[np.int(unique[np.argmax(count)])]

    return most_f


    