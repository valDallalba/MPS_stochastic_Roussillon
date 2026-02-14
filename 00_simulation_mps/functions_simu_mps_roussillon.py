import numpy as np
import pandas as pd
import math
import os
import pickle
import joblib
import json


############
############

def read_pickle(path):
	'''
	function to read a pickle file
	'''
	
	with open(path, 'rb')as file:
		f_read = pickle.load(file)
	
	return f_read


############
############

def write_pickle(path, file):
	'''
	to write pickle file
	'''
	
	with open(path,'wb')as f:
		pickle.dump(file, f, pickle.HIGHEST_PROTOCOL)
	
	
############
############

def write_json(path, file):
	'''
	to write dictionnary as json file
	'''
	
	with open(path,'w')as f:
		json.dump(file, f)

		
############
############

def read_json(path):
	'''
	to write dictionnary as json file
	'''
	
	with open(path,'r')as f:
		file = json.load(f)
		
	return file


############
############

def read_joblib(path):
	'''
	function to read a joblib file without extension
	'''    
	with open(path,'rb') as file:
		f_read = joblib.load(path)
	return f_read
		
	
############
############

def write_joblib(path, file):
	'''
	to write picke file
	'''    
	with open(path,'wb')as f:
		joblib.dump(file, f, compress=5, protocol=pickle.HIGHEST_PROTOCOL)
	return


############
############

def pandas_to_pointSet(df):
	"""
	Convert pandas DataFrame into a PointSet object

	:param df: (pandas.DataFrame) data frame to be converted

	:return: PointSet:
					   A PointSet object
	"""
	import pandas as pd

	return img.PointSet(npt=len(df),
			nv=len(df.columns),
			val=df.values.transpose(),
			varname=[str(column) for column in df.columns])


############
############

def pointSet_to_pandas(point_set):
	"""
	Convert PointSet object into pandas DataFrame

	:param point_set: (PointSet) PointSet to be converted

	:return: pd.DataFrame:
					   A pandas DataFrame
	"""
	import pandas as pd
	# need to convert to int?
	return pd.DataFrame(point_set.val.transpose(), columns=point_set.varname)


############
############

def moving_average_2d(img_mat, mask=False, vox_size=2, no_val=np.nan):
    '''
    Moving average of a 2d array define by a mask and a search moving window.
    Coordinate can be specify in order to preserve hard data location.
    '''
    
    #copy the matrix for calculation
    img_up  = np.copy(img_mat)
    img_out = np.copy(img_mat)
    
    #create the bigger img to sample the moving average
    c_add  = np.full((img_up.shape[0],vox_size),no_val)
    img_up = np.c_[c_add,img_up,c_add]
    l_add  = np.full((vox_size, img_up.shape[1]),no_val)
    img_up = np.r_[l_add, img_up, l_add]

    #change the no_val to numpy nan
    if ~np.isnan(no_val):
        img_out[img_out==no_val] = np.isnan
        
    if mask is False:
    #compute the moving average
        for j in range(img_mat.shape[0]):
            for i in range(img_mat.shape[1]):
                if ~np.isnan(img_out[j,i]):
                    pos_j = j+vox_size
                    pos_i = i+vox_size
                    img_out[j,i] = np.nanmean(img_up[pos_j-vox_size:pos_j+vox_size+1, pos_i-vox_size : pos_i+vox_size+1])

    else:
        for j in range(img_mat.shape[0]):
            for i in range(img_mat.shape[1]):
                if (mask[j,i]==1):
                    pos_j = j+vox_size
                    pos_i = i+vox_size
                    img_out[j,i] = np.nanmean(img_up[pos_j-vox_size:pos_j+vox_size+1, pos_i-vox_size : pos_i+vox_size+1])
        img_out[mask==0]=np.nan
    return img_out


############
############

def get_transition_matrix_hd(df_hd, size_interval=666, xname='X', yname='Y', zname='Z'):
	
	'''
	Calculate the vertical transition probability from a pandas dataFrame set.
	The pandas dataFrame has to possess at least the 4 following columns :
	X Y Z facies
	0 0 -120 0
	0 0 -122 0
	0 0 -124 1
	...
	returns [0] a dictionnary of vertical facies transition probability.
	returns [1] a list of the threshold value for the depth proability calculation.
	The interval from size pass are calculated from the bottom to the top.
	The hd must be organised from less top to bottom z alti.
	The last value of the dictionnary correspond to the overall transition probability, ref [666].
	it has to be used when the simulated layer is above or beneath the min/max depth describe by the hard data.
	check : 23/12/2020
	'''

	#we store the information in list
	x      = list(df_hd[xname])
	y      = list(df_hd[yname])
	z      = list(df_hd[zname])
	facies = list(df_hd['facies'])
	
	#we get the number of facies and the number of wells (unique X and Y coordinate)
	nbFacies = len(df_hd['facies'].unique())
	nbWells  = len(df_hd.groupby([xname,yname]).size())
	positionList = {}
	for i,elt in enumerate(np.unique(facies)):
		positionList[elt]=i

	zMin = min(z)
	zMax = max(z)
	
	#we calculate the limits of the depth intervals and store them in a list
	#the last value of the listePasse list will store the overall probabilites of transition
	if size_interval != 666:
		listePasse = []
		while zMin<zMax:
			listePasse.append(zMin)
			zMin += size_interval
		listePasse.append(666)
	else:
		listePasse = [666]    

	#we create the empty dictionnary which will contained the probabilities of transition per interval
	intervalProb  = {}
	for passe in listePasse:
		intervalProb[passe] = {}
		for facia in np.unique(facies):
			intervalProb[passe][facia] = ([0]*nbFacies)

	#we calculate the numbers of occurences 
	for position in range(1,len(facies)):
		#check well position i-1 == well position i
		if x[-position] == x[-position-1] and y[-position] ==y[-position-1]:
			classePasse = get_interval(listePasse, z[-position])

			#increment the interval transition between 2 facies
			elt1 = facies[-position]
			elt2 = positionList[facies[-position-1]]
			intervalProb[classePasse][elt1][elt2]+=1
			if size_interval != 666:
				#increment the overval transition between 2 facies
				intervalProb[listePasse[-1]][facies[-position]][facies[-position-1]]+=1

	#to export only the occurence dictionnary
	#probaTest = copy.deepcopy(intervalProb)    

	#we calculate the probabilities of transition per intervals
	for passe in listePasse:
		for facia in np.unique(facies):
			sumFacies = np.sum(intervalProb[passe][facia])
			intervalProb[passe][facia] = [round(elt/sumFacies,3) if sumFacies!=0 else 1.0/nbFacies for elt in intervalProb[passe][facia]]
			
	return intervalProb, listePasse


########################
########################

def get_interval(listP,depth):
	'''
	Works with get_transition_matrix_...
	This function is used to define to which interval a certain depth is linked.
	Take as input the list of interval and the evaluated depth.
	check 23/12/2020
	'''

	for pos in range(len(listP)):
	
		if depth<listP[0] or depth>=listP[-1]:
			return listP[-1]
			break

		elif depth>=listP[pos] and depth<listP[pos+1]:
			return listP[pos]
			break
			
			
############
############

def proba_hd_norm(vertical_prob):
	'''
	Normalise the probability of transition.
	'''
	for key in vertical_prob:
		vertical_prob[key] =[p/np.sum(vertical_prob[key]) for p in vertical_prob[key]]
		
	return vertical_prob
			
############
############

def simu_mps_2021_run(hd_df, trend_1, trend_2, rotation, grid, nb_simu, param_mps, seed, interval, simu_type):
	'''
	Base of the simulation loops
	'''
	
	#empty output simulation matrix
	#simu_out = np.full((1, grid.shape[1], param_mps['ny'], param_mps['nx']), np.nan)
	
	#We start by calculating the vertical transition (ivs) probability between facies.
	vertical_proba = get_transition_matrix_hd(hd_df, param_mps['interval'])
	int_size       = vertical_proba[1][1]-vertical_proba[1][0]    
	
	#mask
	grid_mask = grid[0][0]
	grid_mask[np.isnan(grid_mask)] = 0

	#count layer for interval
	count_interval = 0 
	
	for simu in range(nb_simu):
		#empty output simulation matrix
		simu_out = np.full((1, grid.shape[1], param_mps['ny'], param_mps['nx']), np.nan)

		seed += simu*interval

		for layer in range(grid.shape[1]):
			#random ti selection
			rd_ti = np.random.randint(0,100)
			ti    = joblib.load('data/set_ti/int_{0}_ti/ti_{1}f_int{0}_{2:03d}.pickle'.format(interval, simu_type, rd_ti))

			#random second trend modification
			rd_trend_2 = img.copyImg(trend_2)
			rd_trend_2.val[0,0] = moving_average_2d(rd_trend_2.val[0,0], grid_mask, np.random.randint(10,60))
			rd_trend_2.val[0,0][rd_trend_2.val[0,0]<1] = 0

			#depth
			depth  = layer-0.5*param_mps['sz']

			#simu layer 0
			if layer == 0:  
				#simu
				hd_z   = pandas_to_pointSet(hd_df[hd_df.Z==layer])
				simu_z, simu_img = deesse_run_2d(hd_z, ti, trend_1, rd_trend_2, rotation, grid_mask, depth, param_mps, seed+layer)
				simu_out[0,layer] = simu_z
				
			elif count_interval == param_mps['interval']:
				hd_z   = pandas_to_pointSet(hd_df[hd_df.Z==layer])
				simu_z, simu_img = deesse_run_2d(hd_z, ti, trend_1, rd_trend_2, rotation, grid_mask, depth, param_mps, seed+layer)
				simu_out[0,layer] = simu_z
				count_interval = 0

			#other layer x
			else:
				#sampling hd
				#if we are located at a depth that is not comprise in the interval describe in the hd
				if depth<vertical_proba[1][-2]:
					inter      = int(depth//int_size)
					prob       = vertical_proba[0][vertical_proba[1][inter]]     
					ivs        = proba_hd_norm(prob)
				#if not we take the global probability
				else:
					prob       = vertical_proba[0][666]     
					ivs        = proba_hd_norm(prob)  
				#simu
				hd_z   = hd_df[hd_df.Z==layer]
				hd_s   = sampling_hd_from_simu(simu_img, ivs, depth, param_mps)
				hd_f   = pandas_to_pointSet(pd.concat([hd_z, hd_s]))
				simu_z, simu_img = deesse_run_2d(hd_f, ti, trend_1, rd_trend_2, rotation, grid_mask, depth, param_mps, seed+layer)
				simu_out[0,layer] = simu_z
				count_interval +=1

		print('simu {} is done !'.format(simu))	
		write_joblib('simu_out/simu_{}_{:02}.pickle'.format(interval,int(simu)), simu_out)
	return simu_out #array (nb_simu, nz, ny, nx)


##############
##############

def sampling_hd_from_simu(simulation, vertical_prob, depth, param_mps):
	'''
	Function that create a hard data set based on random sampling and vertical transition probabilities.
	Random points of the 2D grid are selects on the previous simulation grid.
	The value of the facies are inferred based on the vertical transition probabilites.
	Input : a 2D simulation, vertical prob list, sampling rate.
	Return a hd point set.
	(nbFacies = number of facies in the hard data set)

	Last check 29/03/2021
	'''
	rate             = param_mps['sampling_rate']
	facies_to_sample = param_mps['facies_to_sample']
	nb_facies        = param_mps['nb_facies_tot']
	
	point_set_simu  = img.imageToPointSet(simulation)                                #We transform the simulation to point set.
	data_frame_simu = pointSet_to_pandas(point_set_simu)                             #And then to a data frame in order to sample it.
	data_frame_simu = data_frame_simu.rename(columns={'facies_real00000':'facies'})  #We rename the facies columns.
	data_frame_simu = data_frame_simu.dropna()                                       #We deleate the non simulate values.
	
	all_facies = data_frame_simu['facies'].unique()

	#data_frame_simu = data_frame_simu.drop('trend_1_real00000',1)                   #For multivariable simulation we also delete the other variable.
	#data_frame_simu = data_frame_simu.drop('trend_2_real00000',1)                   #For multivariable simulation we also delete the other variable.
		
	#We transform the facies value based on the transition probability and the sampling rate
	df_all=[]
	for facia in facies_to_sample:
		df_sample = data_frame_simu[data_frame_simu['facies']==facia].sample(n = int(np.sum(data_frame_simu['facies']==facia)*rate))

		#if the df is empty we multiply the sampling rate
		if len(df_sample) == 0:
			df_sample = data_frame_simu[data_frame_simu['facies']==facia].sample(n = int(np.sum(data_frame_simu['facies']==facia)*rate*10))
		
		#transform the facies regarding the vertical prob
		df_sample['facies'] = np.random.choice(nb_facies,len(df_sample['facies']),p=vertical_prob[int(facia)])
		df_all.append(df_sample)

	#We concat the dataFrame of every sampled facies.
	data_frame_sample = pd.concat(df_all)                   

	#We double check that the list is not empty.
	if len(data_frame_sample)==0:
		data_frame_sample = None

	#If not we create the final point set    
	else:
		data_frame_sample['Z'] = depth     #We fixe the correct z value.
		#sampled_hd            = pandasToPointSet(data_frame_sample)  #We transform the dataFrame to a point set.

	return data_frame_sample


############
############

def deesse_run_2d(hd_z, ti, trend_1, trend_2, rotation, grid_mask, depth, param_mps, seed):
	'''
	DeeSse 2D simulation
	'''
	homo = 0.75
	snp = []
	for snp_v in param_mps['snp']:
		snp_i = dsi.SearchNeighborhoodParameters(radiusMode = "manual", rx=snp_v[0], ry=snp_v[1], rz=snp_v[2])
		snp.append(snp_i)
		
	pyrGenParams = dsi.PyramidGeneralParameters(
	npyramidLevel=2,                 # number of pyramid levels, additional to the simulation grid
	kx=[2, 2], ky=[2, 2], kz=[0, 0]  # reduction factors from one level to the next one
									 #    (kz=[0, 0]: do not apply reduction along z axis)
	)
	pyrParams_categ = dsi.PyramidParameters( 
		nlevel=2,                      # number of levels
		pyramidType='categorical_to_continuous' # type of pyramid (accordingly to categorical variable in this example)
	)
	pyrParams_cont  = dsi.PyramidParameters( 
		nlevel=2,                      # number of levels
		pyramidType='continuous' # type of pyramid (accordingly to categorical variable in this example)
	)
	pyrParams = [pyrParams_cont,pyrParams_cont,pyrParams_categ]


	deesse_input = dsi.DeesseInput(
		nx = int(param_mps['nx']),   ny = int(param_mps['ny']),   nz = 1,                   
		sx = float(param_mps['sx']), sy = float(param_mps['sy']), sz = float(param_mps['sz']),                           
		ox = float(param_mps['ox']), oy = float(param_mps['oy']), oz = float(depth),     
		nv = param_mps['nv'],
		varname = param_mps['varname'],           #nb et nom des variables (plus rapide de mettre le trend en 1er)
		nTI     = 1, 
		TI      = ti,                             #nb TI 
		outputVarFlag = param_mps['out_flag'],    #défini les variables simulées à sauvegarder
		mask          = grid_mask,                   #mask (values)
		dataImage     = [trend_1, trend_2],       #trend (img)
		dataPointSet  = hd_z,
		searchNeighborhoodParameters = snp,                 #searching parameters
		rotationUsage                = param_mps['use_rotation'],        #type de rotation = avec tolérance
		rotationAzimuthLocal         = param_mps['rotation_azy_loc'],    #type de rotation = local et non constant
		rotationAzimuth              = rotation,                         #rotation (values)
		distanceType                 = param_mps['dist_type'],           #type de distance (0=categ, 1=continu)
		nneighboringNode             = param_mps['nb_nodes'],            #nombre de node conditionnant par variable
		distanceThreshold            = param_mps['dist_t'],              #threshold pour la distance
		conditioningWeightFactor     = param_mps['weight_f'], 
		maxScanFraction              = param_mps['scan_f'] ,         
		homothetyUsage=1,
		homothetyXLocal=False,
		homothetyXRatio=homo,
		homothetyYLocal=False,
		homothetyYRatio=homo,
		homothetyZLocal=False,
		homothetyZRatio=None, 
		npostProcessingPathMax=1,    
		#pyramidGeneralParameters=pyrGenParams, # set pyramid general parameters
		#pyramidParameters=pyrParams,           # set pyramid parameters for each variable              
		seed                         = seed,
		nrealization                 = 1)

	deesse_output = dsi.deesseRun(deesse_input,verbose=0,nthreads=30)
	simu          = deesse_output['sim'][0]

	return simu.val[0][0], simu



############
############

def simu_mps_2021_proba(simulation):
	'''
	Calculate the probability map for each facies in the simulation.
	'''
	
	facies = np.unique(simulation)[~np.isnan(np.unique(simulation))]
	proba  = np.zeros((len(facies), simulation.shape[1], simulation.shape[2], simulation.shape[3]))
	
	for i,f in enumerate(facies):
		proba[i] = np.sum(simulation[:]==f,axis=0)/(simulation.shape[0])
		proba[i][np.isnan(simulation[0])]= np.nan
		
	return proba

############
############

def simu_mps_2021_entro(proba_map):
	'''
	Calculate the shannon entropy of the realisation.
	'''

	image_prob = img.Img(nx=proba_map.shape[3], ny=proba_map.shape[2], nz=proba_map.shape[1],
				sx=1, sy=1, sz=1,
				ox=0, oy=0, oz=0,
				nv=proba_map.shape[0], val=proba_map,
				name='proba')
	
	image_entro = img.imageEntropy(image_prob)
	
	return image_entro


############
############
