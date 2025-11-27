#!/usr/bin/env python
# version: 1.0 (D. E. Olave-Rojas 05/07/2021)
# LAGASU: LAbeller of GAlaxies within SUbstructures is a python scripts 
# that assign galaxies to different substructures in and around a galaxy 
# cluster based on their density
# LAGASU uses the Gaussian Mixture Module (GMM) and Density-Based Spatial 
# Clustering of Application with Noise (DBSCAN), both availables from 
# python, to assign galaxies to different substructures found in and 
# around galaxy clusters
#-------------------------------------------------------------------
# created by    : D. E. Olave-Rojas & D. A. Olave-Rojas
# email         : daniela.olave@utalca.cl
# version       : 0.1.5
# update        : November 27, 2025
# maintainer    : D. E. Olave-Rojas 
#-------------------------------------------------------------------

#-- Import preexisting python modules
import numpy as np
import time
from sklearn import mixture
from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import jaccard_score
from astropy.stats import biweight_location

from calsagos import utils

__author__ = 'D. E. Olave-Rojas & D. A. Olave-Rojas'
__email__ = 'daniela.olave@utalca.cl'
__version__ = '0.1.5'
__maintainer__ = "D. E. Olave-Rojas"

#####################################################################################################################################################################################
#####################################################################################################################################################################################

def lagasu(id_galaxy, ra_galaxy, dec_galaxy, redshift_galaxy, range_cuts, galaxy_separation, n_galaxies, metric_distance, method, ra_cluster, dec_cluster, redshift_cluster, r200, flag):

    """ LAGASU is a function that assigns galaxies to different 
    susbtructures in and around a galaxy cluster

    This function was developed by D. E. Olave-Rojas and 
    D. A. Olave-Rojas (07/05/2021) and was updated by 
    D. E. Olave-Rojas (06/17/2024)

    The input of LAGASU can be a sample of galaxies in a cluster 
    of a sample of galaxies previously selected as potential 
    members of a substructure in and around a single galaxy 
    cluster. The selection of potential members of substructures
    can be done by using the Dressler-Schectamn Test or DS-Test
    (Dressler & Schectman 1988) 

    lagasu(id_galaxy, ra_galaxy, dec_galaxy, redshift_galaxy, 
    range_cuts, galaxy_separation, n_galaxies, metric_distance, 
    method, ra_cluster, dec_cluster, r200, flag)

    :param id_galaxy: identifier of each galaxy in the sample
	:param ra_galaxy: is the Right Ascension of each galaxy in
        the sample. This parameter must be in degree units
    :param dec_galaxy: is the Declination of each galaxy in
        the sample. This parameter must be in degree units
    :param redshift_galaxy: redshift of each galaxy in the 
        cluster
    :param range_cuts: to perform the cuts in the redshift 
        space is neccesary to give a range that allows find the 
        best number of cuts in redshift. The parameter "range_cuts" 
        is the end of the range which is defined in LAGASU as 
        range(1, range_cuts). Therefore, range_cuts must be greater 
        than 1
    :param galaxy_separation: physical separation between 
        galaxies in a substructure the units must be the 
        same as the ra_galaxy and dec_galaxy
    :param n_galaxies: minimum number of galaxies to define a group
    :param metric_distance: metric used to calcule the distance 
        between instances in a feature array. Metric must be 
        'euclidean' or 'haversine'
    :param method: clustering algorithm used to grouping galaxies
        in substructures. Method must be 'dbscan' or 'hdbscan'
    :param ra_cluster: central Right Ascention (R.A.)
        of the cluster 
    :param dec_cluster: central Declination (Dec.)
        of the cluster 
    :param redshift_cluster: central redshift of the 
        cluster
    :param r200: is the typical radius of a sphere 
        with a mean density equal to 200 times the 
        critical density. This parameter must be
        in degrees
    :param flag:  parameter that allows the user to 
        choose between photometric of spectroscopic 
        sample. If flag == 'zphot' the input must be
        photometric sample. If flag == 'zspec" the 
        input must be spectroscopic sample 

    :type id_galaxy         : array
    :type ra_galaxy         : array
    :type dec_galaxy        : array
	:type redshift_galaxy   : array
    :type range_cuts        : int
    :type galaxy_separation : int, float 
    :type n_galaxies        : int, float
    :type metric_distance   : string  
    :type method            : string  
    :type ra_cluster        : float
    :type dec_cluster       : float
    :type redshift_cluster  : float
    :type r200              : float
    :type flag              : string

	:returns: label to each galaxy,
        which corresponds to identify
        each group
	:rtype: array
   
    .. note::
    
    LAGASU will give us three labels as output: 
    i) lagasu[4] that corresponds to the label 
    putting by GMM and varies between 0 to N, 
    ii) lagasu[5] that corresponds to the label 
    putting by DBSCAN after to run gmm and varies 
    between -1 to N, where -1 corresponds to noise
    and galaxies within a substructure have a label 
    between 0 to N, and iii) lagasu[6] that corresponds 
    to the corrected label considering galaxies in
    substructures and in the principal halo. Galaxies
    in substructures are a label between 0 to N. 
    Whereas, galaxies in the principal halo have
    a label equal to -1. For details about this
    correction see the help of the function
    "rename_substructures" in utils module.

	"""
    print("-- starting LAGASU --")
    print("--- input parameters ---")
    print("Number of members    :", n_galaxies)
    print("metric               :", metric_distance)
    print("method               :", method)

    #-- evaluating if there is a value equal to -99. of 99. or < 0. in the redshift galaxy parameter
    if ((redshift_galaxy > -99.) & (redshift_galaxy <= 0.)).any():
        print(" WARNING!! There is a redshift lower to 0 or equal to 0")
        print("... replacing bad values ...")
        boolean_mask = (redshift_galaxy > -99.) & (redshift_galaxy <= 0.)  # create a boolean mask. It is an array with True/False and True corresponds to the old value
        redshift_galaxy[boolean_mask] = redshift_cluster
    elif np.isin(-99., redshift_galaxy).any():
        print(" WARNING!! There is a redshift equal to -99.")
        print("... replacing bad values ...")
        boolean_mask = (redshift_galaxy == -99.)  # create a boolean mask. It is an array with True/False and True corresponds to the old value
        redshift_galaxy[boolean_mask] = redshift_cluster
    elif np.isin(99., redshift_galaxy).any():
        print(" WARNING!! There is a redshift equal to 99.")
        print("... replacing bad values ...")
        boolean_mask = (redshift_galaxy == 99.)  # create a boolean mask. It is an array with True/False and True corresponds to the old value
        redshift_galaxy[boolean_mask] = redshift_cluster
    else: 
        print("Redshift array is ok")

    #-- Gaussian Mixture Models (GMM) divides the sample in ranges of redshift in order to consider the volume of the cluster
    #-- The divissión is perform without an arbitrary number of cuts in the redshift distribution and the number of cuts are in a range
    #-- The divissión in the redshift space is perform without an arbitrary number of cuts by using the Bayesian Information Criterion (BIC), 
    # which selects the better number of cuts according to the data 
    #-- The number of cuts found by BIC are in a range given in the input as "range_cuts"
    
    #-- creating a transposed array with the redshift_galaxy to be used as input in the GMM implementation
    candidate_galaxies = np.array([redshift_galaxy]).T

    #-- defining the lowest bic parameter
    lowest_bic = np.infty

    #-- initializing the variable
    bic = []
    
    #-- defining the number of possible cuts in the redshift space
    n_components_range = range(1, range_cuts)

    #-- defining a list with the types of covariances used in the implementation of GMM-BIC
    cv_types = ['spherical', 'tied', 'diag', 'full']
    
    #--- START OF LOOP ---

    for cv_type in cv_types:
    
        for n_components in n_components_range:
    
    	    #-- Fit a mixture of Gaussians with Expectation Maximization (EM)
            gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type, random_state=4)

            #-- Fit the Gaussian over the data 
            gmm.fit(candidate_galaxies)

            bic.append(gmm.bic(candidate_galaxies))

            #-- selecting the best fit
            if bic[-1] < lowest_bic:

                lowest_bic = bic[-1]

                best_gmm = gmm

    # -- END OF LOOP --
    bic = np.array(bic)
    clf = best_gmm
    print("Best model GMM       :", best_gmm)
    
    #-- defining the parameters of the best fit
    n_cuts_bic = clf.n_components # number of cuts do by the algorithms 
    print("Number of GMM cuts  :", n_cuts_bic)
    #-- putting a label to each galaxy. This label allows us to separe and assign each galaxy to a redshift cut
    labels_bic = clf.predict(candidate_galaxies) # using this label we can separate galaxies into n groups that correspond to redshifts cuts
    
    #-- To assign galaxies to different substructures we use Density-Based Spatial Clustering of Applications with Noise (DBSCAN, Ester et al. 1996)
    #-- To identify groups using DBSCAN we must define a minimum number of neighbouring objects separated by a specific distance. 
    #-- DBSCAN does not assign all objects in the sample to one group (Ester et al. 1996) and we can remove the galaxies that are not spatially grouped with others

    #-- sorting the output labels given by GMM-BIC implementation 
    sorted_labels_bic = np.sort(labels_bic)

    #--- START OF LOOP ---
    for ii in range(0,n_cuts_bic): # n_cuts_bic are the number of cuts given by the GMM implementation

        #-- defining a single redshift cut in which DBSCAN will be apply
        n_redshift_groups = np.where(labels_bic == ii)[0] 

        #-- selecting galaxies that are part of a signle cut in redshift
        ra = ra_galaxy[n_redshift_groups]
        dec = dec_galaxy[n_redshift_groups]
        id = id_galaxy[n_redshift_groups]

        #-- generate sample data
        #-- creating a transposed array with the poition of galaxies to be used as input in the DBSCAN implementation
        #-- metric implementation was added an June 17, 2024

        if metric_distance == 'euclidean':
            X = np.array([ra, dec]).T
        if metric_distance == 'haversine': 
            ra_rad = np.radians(ra)
            dec_rad = np.radians(dec)
            X = np.array([ra_rad, dec_rad]).T

        #-- Performing the clustering algothims DBSCAN or HDBSCAN
        if method == 'dbscan':
            cluster = DBSCAN(eps=galaxy_separation, min_samples=n_galaxies, metric=metric_distance, algorithm='ball_tree').fit(X)
        elif method == 'hdbscan':
            cluster = HDBSCAN(min_samples=n_galaxies, metric=metric_distance, algorithm='ball_tree').fit(X)   
     
        #-- putting a label to each galaxy. This label allows us to assign each galaxy to a substructure
        label_cluster_bic = cluster.labels_ # the label_dbscan_bic is the label of each odentified substructure. This parameter is a numpy.ndarray

        #-- selecting the labels of the groups found by DBSCAN in each redshift cut
        groups = np.unique(label_cluster_bic) # number of groups in each redshift cut

        if ii != 0:
            p = np.append(p,groups, axis=0)
            tam_groups = np.append(tam_groups, len(groups))
            labels_dbscan_bic = np.append(labels_dbscan_bic,label_cluster_bic)
            first_element_groups = np.append(first_element_groups, groups[0])

        #-- process of the first iteration: defining variables
        else:                
            p = groups # array with the unique labels that could be assign to a single galaxy
            tam_groups = len(groups) # size of the each idenfied substructure
            labels_dbscan_bic = label_cluster_bic # label used to identified each substructure
            first_element_groups = groups[0] # first element in the array with the labels of the substructures
        
    #-- Finally, we need to label the substructures from 0 to n
    #-- Initialization of variables
    groups_pos = []
    p_pos = 0

    #-- Position of the first element in each group is searched here 
    #-- This implementation allows us to consider the case of a little sample in which all galaxies are assign to a single substructure 
    if type(tam_groups) == int:
        tam_groups = np.array([tam_groups])
    
    else:
        tam_groups = tam_groups

    #--- START OF LOOP ---
    for ii in range(0,len(tam_groups)):

        if (len(tam_groups)==1):

            groups_pos = 0

        else: 

            for e in range(0,tam_groups[ii]):

                if(p[p_pos+e]==first_element_groups[ii] and ii==0):
                    groups_pos = 0

                if(p[p_pos+e]==first_element_groups[ii] and ii!=0):
                    groups_pos = np.append(groups_pos,(p_pos+e))
        
            p_pos = p_pos + tam_groups[ii]
    # -- END OF LOOP --

    #-- Here the correlative is assembled eliminating -1
    p3 = [] # p3 is an array with the label of all idenitified substructures

    #--- START OF LOOP ---
    for j in range(0,len(p)):

        if p[j] != -1:
            p3 = np.append(p3, j)

    for l in range(0,len(p3)):

        p3[l] = l
    # -- END OF LOOP --

    #-- initializing the variables
    p_pos = 0
    correlative = 0

    #-- Here the array p2 is assembled, which is an array with the label of all identified substructures plus noise
    #--- START OF LOOP ---
    for ii in range(0,len(tam_groups)):

        for e in range(0,tam_groups[ii]):

            if(e==0 and ii==0):
                p2 = p[0]

                if(p[0] != -1):
                    correlative +=1

            elif(p[p_pos+e]== -1):
                p2 = np.append(p2,-1)

            else:
                p2 = np.append(p2,correlative)
                correlative +=1

        p_pos = p_pos + tam_groups[ii]
    # -- END OF LOOP --

    #-- This implementation allows us to consider the case of a little sample in which all galaxies are assign to a single substructure 
    if type(groups_pos) == int:
        groups_pos = np.array([groups_pos])
    
    else:
        groups_pos = groups_pos
#========================================================
    #-- This loop allows us to assign a label from 0 to n lo each substructures plus noise which is labelled with -1
    #--- START OF LOOP ---

    #-- selecting the labels of the groups found by DBSCAN 
    final_groups = np.unique(labels_dbscan_bic)

    if  len(final_groups) == 1: # in this case none of galaxies are part of a subhalo

        labels_dbscan_corr = labels_dbscan_bic 
    
    else: 
        for k in range(0,len(labels_dbscan_bic)):
            aux = 0
            
            if(k == 0):
                while( labels_dbscan_bic[k] != p[groups_pos[sorted_labels_bic[k]]+aux]):
                    aux +=1
                labels_dbscan_corr = p2[groups_pos[sorted_labels_bic[k]]+aux]

            else:
                while( labels_dbscan_bic[k] != p[groups_pos[sorted_labels_bic[k]]+aux]):
                    aux +=1
                labels_dbscan_corr = np.append(labels_dbscan_corr,p2[groups_pos[sorted_labels_bic[k]]+aux])
    # -- END OF LOOP --

    for ii in range(0,n_cuts_bic):

        n_redshift_groups = np.where(labels_bic == ii)[0]

        id_gal_out = id_galaxy[n_redshift_groups]
        ra_out = ra_galaxy[n_redshift_groups]
        dec_out = dec_galaxy[n_redshift_groups]
        redshift_gal_out = redshift_galaxy[n_redshift_groups]
        gmm_labels = labels_bic[n_redshift_groups]

        if ii != 0:
            id_substructures = np.append(id_substructures,id_gal_out)
            ra_substructures = np.append(ra_substructures,ra_out)
            dec_substructures = np.append(dec_substructures, dec_out)
            redshift_substructures = np.append(redshift_substructures, redshift_gal_out)
            gmm_substructures = np.append(gmm_substructures, gmm_labels)
            
        else:              #-- process of the first iteration: defining variables
            id_substructures = id_gal_out
            ra_substructures = ra_out
            dec_substructures = dec_out
            redshift_substructures = redshift_gal_out
            gmm_substructures = gmm_labels

    # -- renaming the substructures identified by using lagasu in order to identify the principal halo and separate it from the substructures
    # if label == -1 the galaxy is only part of the principal halo. If galaxy is != -1 the galaxy is in a substructure
    id_final = utils.rename_substructures(ra_substructures, dec_substructures, redshift_substructures, labels_dbscan_corr, ra_cluster, dec_cluster, redshift_cluster, r200, flag)

    #-- building matrix with output quantities
    lagasu_parameters = np.array([id_substructures, ra_substructures, dec_substructures, redshift_substructures, gmm_substructures, labels_dbscan_corr, id_final], dtype=object)

    print("-- ending LAGASU --")

    #-- returning output quantity
    return lagasu_parameters

#####################################################################################################################################################################################
#####################################################################################################################################################################################

def lagasu_stability(ra_galaxy, dec_galaxy, redshift_galaxy, range_cuts, nsim, galaxy_separation, n_galaxies, metric_distance, method):

    """ LAGASU_stability is a function that evaluate 
    the stability of the assigantion of galaxies to
    different groups

    This function was developed by D. E. Olave-Rojas (11/27/2025)
    based on LAGASU

    The input of LAGASU_stability must be the same sample as
    in LAGASU
    
    lagasu_stability(ra_galaxy, dec_galaxy, redshift_galaxy, 
    range_cuts, nsim, galaxy_separation, n_galaxies, metric_distance, 
    method)

	:param ra_galaxy: is the Right Ascension of each galaxy in
        the sample. This parameter must be in degree units
    :param dec_galaxy: is the Declination of each galaxy in
        the sample. This parameter must be in degree units
    :param redshift_galaxy: redshift of each galaxy in the 
        cluster
    :param nsim: number of simulation in boostrap resampling
    :param range_cuts: to perform the cuts in the redshift 
        space is neccesary to give a range that allows find the 
        best number of cuts in redshift. The parameter "range_cuts" 
        is the end of the range which is defined in LAGASU as 
        range(1, range_cuts). Therefore, range_cuts must be greater 
        than 1
    :param galaxy_separation: physical separation between galaxies 
        in a group. If metric_distance="euclidean" is selected, 
        the unit should be degrees. If metric_distance="haversine" is 
        selected, the unit should be radians.
    :param n_galaxies: minimum number of galaxies to define a group
    :param metric_distance: metric used to calcule the distance 
        between instances in a feature array. Metric must be 
        'euclidean' or 'haversine'
    :param method: clustering algorithm used to grouping galaxies
        in substructures. Method must be 'dbscan' or 'hdbscan'

    :type ra_galaxy         : array
    :type dec_galaxy        : array
	:type redshift_galaxy   : array
    :type range_cuts        : int
    :type nsim              : int  
    :type galaxy_separation : int, float 
    :type n_galaxies        : int, float
    :type metric_distance   : string  
    :type method            : string
  
    .. note::
    
	"""
    print("-- starting LAGASU stability --")
    print(" ")
    print("--- input parameters ---")
    print("Number of members    :", n_galaxies)
    print("metric               :", metric_distance)
    print("method               :", method)
    print("No. simulations      :", nsim)

    #-- Gaussian Mixture Models (GMM) divides the sample in ranges of redshift in order to consider the volume of the cluster
    #-- The divissión is perform without an arbitrary number of cuts in the redshift distribution and the number of cuts are in a range
    #-- The divissión in the redshift space is perform without an arbitrary number of cuts by using the Bayesian Information Criterion (BIC), 
    # which selects the better number of cuts according to the data 
    #-- The number of cuts found by BIC are in a range given in the input as "range_cuts"

    print(" ")

    redshift_cluster = biweight_location(redshift_galaxy)

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------

    #-- evaluating if there is a value equal to -99. of 99. or < 0. in the redshift galaxy parameter
    if ((redshift_galaxy > -99.) & (redshift_galaxy <= 0.)).any():
        print(" WARNING!! There is a redshift lower to 0 or equal to 0")
        print("... replacing bad values ...")
        boolean_mask = (redshift_galaxy > -99.) & (redshift_galaxy <= 0.)  # create a boolean mask. It is an array with True/False and True corresponds to the old value
        redshift_galaxy[boolean_mask] = redshift_cluster
    elif np.isin(-99., redshift_galaxy).any():
        print(" WARNING!! There is a redshift equal to -99.")
        print("... replacing bad values ...")
        boolean_mask = (redshift_galaxy == -99.)  # create a boolean mask. It is an array with True/False and True corresponds to the old value
        redshift_galaxy[boolean_mask] = redshift_cluster
    elif np.isin(99., redshift_galaxy).any():
        print(" WARNING!! There is a redshift equal to 99.")
        print("... replacing bad values ...")
        boolean_mask = (redshift_galaxy == 99.)  # create a boolean mask. It is an array with True/False and True corresponds to the old value
        redshift_galaxy[boolean_mask] = redshift_cluster
    else: 
        print("Redshift array is ok")

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ================================================================================ GMM ===============================================================================

    print("  ")
    print(f"... starting GMM ...")
    #-- creating a transposed array with the redshift_galaxy to be used as input in the GMM implementation
    candidate_galaxies = np.array([redshift_galaxy]).T

    #-- defining the lowest bic parameter
    lowest_bic = np.infty

    #-- initializing the variable
    bic = []
    
    #-- defining the number of possible cuts in the redshift space
    n_components_range = range(1, range_cuts)

    #-- defining a list with the types of covariances used in the implementation of GMM-BIC
    cv_types = ['spherical', 'tied', 'diag', 'full']
    
    #--- START OF LOOP ---

    for cv_type in cv_types:
    
        for n_components in n_components_range:
    
    	    #-- Fit a mixture of Gaussians with Expectation Maximization (EM)
            gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type, random_state=4)

            #-- Fit the Gaussian over the data 
            gmm.fit(candidate_galaxies)

            bic.append(gmm.bic(candidate_galaxies))

            #-- selecting the best fit
            if bic[-1] < lowest_bic:

                lowest_bic = bic[-1]

                best_gmm = gmm

    # -- END OF LOOP --
    bic = np.array(bic)
    clf = best_gmm
    print("Best model GMM       :", best_gmm)
#    bars = []
    
    #-- defining the parameters of the best fit
    n_cuts_bic = clf.n_components # number of cuts do by the algorithms 
    print("Number of GMM cuts  :", n_cuts_bic)
    print(" ")

    print("... starting DBSCAN ...")
    #-- putting a label to each galaxy. This label allows us to separe and assign each galaxy to a redshift cut
    labels_bic = clf.predict(candidate_galaxies) # using this label we can separate galaxies into n groups that correspond to redshifts cuts

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ========================================================================== DBSCAN/HDBSCAN ==========================================================================

    #-- To assign galaxies to different substructures we use Density-Based Spatial Clustering of Applications with Noise (DBSCAN, Ester et al. 1996)
    #-- To identify groups using DBSCAN we must define a minimum number of neighbouring objects separated by a specific distance. 
    #-- DBSCAN does not assign all objects in the sample to one group (Ester et al. 1996) and we can remove the galaxies that are not spatially grouped with others

    #--- START OF LOOP ---
    for ii in range(0,n_cuts_bic): # n_cuts_bic are the number of cuts given by the GMM implementation

        #-- defining a single redshift cut in which DBSCAN will be apply
        n_redshift_groups = np.where(labels_bic == ii)[0] 

        #-- selecting galaxies that are part of a signle cut in redshift
        ra = ra_galaxy[n_redshift_groups]
        dec = dec_galaxy[n_redshift_groups]

        #-- generate sample data
        #-- creating a transposed array with the poition of galaxies to be used as input in the DBSCAN implementation
        #-- metric implementation was added an June 17, 2024

        if metric_distance == 'euclidean':
            X = np.array([ra, dec]).T
        if metric_distance == 'haversine': 
            ra_rad = np.radians(ra)
            dec_rad = np.radians(dec)
            X = np.array([ra_rad, dec_rad]).T

        #-- Performing the clustering algothims DBSCAN or HDBSCAN
        if method == 'dbscan':
            cluster = DBSCAN(eps=galaxy_separation, min_samples=n_galaxies, metric=metric_distance, algorithm='ball_tree').fit(X)
        elif method == 'hdbscan':
            cluster = HDBSCAN(min_samples=n_galaxies, metric=metric_distance, algorithm='ball_tree').fit(X)   

        #-- putting a label to each galaxy. This label allows us to assign each galaxy to a substructure
        label_cluster_bic = cluster.labels_ # the label_dbscan_bic is the label of each odentified substructure. This parameter is a numpy.ndarray

        #-- counting the number of groups in the random sample
        n_clusters_original_bic = len(set(label_cluster_bic)) - (1 if -1 in label_cluster_bic else 0)
        print(f"Number of groups in original sample : {n_clusters_original_bic}")

        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ================================================================= ADD RANDOM POINTS TO THE SAMPLE ==================================================================

        #-- creating random samples 
        #-- stabillity adding points was added an November 26, 2025
        print(" ")
        print("... adding random points to the sample ...")
        np.random.seed() # initializing random seed to reproductibility
        number = int((len(X)*0.8)) # add a random sample equivalent to 80% of the original sample

        random_points = np.random.uniform(low=X.min(axis=0), 
                                          high=X.max(axis=0), 
                                          size=(number, 2))

        #-- combining samples
        X_random = np.vstack([X, random_points])

        #-- Performing the clustering algothims DBSCAN or HDBSCAN in random sample
        if method == 'dbscan':
            cluster_random = DBSCAN(eps=galaxy_separation, min_samples=n_galaxies, metric=metric_distance, algorithm='ball_tree').fit(X_random)
        elif method == 'hdbscan':
            cluster_random = HDBSCAN(min_samples=n_galaxies, metric=metric_distance, algorithm='ball_tree').fit(X_random)   

        #-- putting a label to each galaxy in the random sample. This label allows us to assign each galaxy to a substructure
        label_cluster_bic_random = cluster_random.labels_ # the label_dbscan_bic_random is the label of each odentified substructure in the new sample. This parameter is a numpy.ndarray

        #-- counting number of groups in the random sample
        n_clusters_random = len(set(label_cluster_bic_random)) - (1 if -1 in label_cluster_bic_random else 0)
        print(f"Number of groups in random sample   : {n_clusters_random} ... adding 80% of the original sample")
        
        #-- preparing the labels for comparison 
        #-- selection of the labels for the first N initial points of X_random
        labels_random_initial = label_cluster_bic_random[:len(X)]

        #-- filter out noisy points for a purer stability comparison
        #--- creating a mask to include only the points that were NOT noisy in ANY of the runs
        stable_mask = (label_cluster_bic != -1) & (labels_random_initial != -1)

        #-- appliying mask to the labels
        labels_initial_stable = label_cluster_bic[stable_mask]
        labels_random_stable = labels_random_initial[stable_mask]

        #-- estimating cluster stabillity 
        if len(labels_initial_stable) > 0:
            ari_score_random = adjusted_rand_score(labels_initial_stable, labels_random_stable)
            print(f"ARI score for random sample         : {ari_score_random:.4f}")
        else:
            print("ARI cannot be estimated")

        print(" ")

        # --- Assesing individual stability 
        print("... assessing individual stability ...")
        print("...")
    
        cluster_ids = [c for c in np.unique(label_cluster_bic) if c != -1]

        stability_scores_noise = {}

        for k in cluster_ids:
            # Set A_initial_add: initial membership
            A_initial_add = (label_cluster_bic == k) # 1 if point 'i' initially belongs to cluster k, 0 otherwise.
    
            # set B_new: new membership 
            B_new = (labels_random_initial == k) # 1 if point 'i' (original) belongs to cluster k in the new execution, 0 otherwise.

            #-- filter for comparison
            #-- It's best to compare ONLY the points that were assigned to a cluster (not noise) in both runs. This measures the stability of the cluster's dense core.
    
            #-- stability mask: points that are not noise in either execution
            stable_core_mask = (label_cluster_bic != -1) & (labels_random_initial != -1)
    
            A_stable = A_initial_add[stable_core_mask]
            B_stable = B_new[stable_core_mask]
    
            #-- estimating jaccard Score
            if len(A_stable) == 0:
                jaccard_add = 0.0 # If the initial cluster had NO surviving kernel points
            else:
                #-- using jaccard_score function
                jaccard_add = jaccard_score(A_stable, B_stable, pos_label=1)

            stability_scores_noise[f'Group {k}'] = jaccard_add
    
        for cluster, score in stability_scores_noise.items():
            print(f"{cluster}: {score:.4f}")

        print(" ")

        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # =============================================================== REMOVE RANDOM POINTS FROM THE SAMPLE ===============================================================

        print("... removing random points from the sample ...")
        #-- Remove points at random (Subsampling)
        #-- stabillity removing points was added an November 26, 2025

        sample_size = len(X)
        removal_percentage = 0.20 # -- removing 20% of the sample
        n_to_remove = int(sample_size * removal_percentage)

        #-- Generate random indices to be deleted
        np.random.seed() # Fix the seed to know which points are removed
        indices_to_keep = np.random.choice(sample_size, 
                                           sample_size - n_to_remove, 
                                           replace=False)

        #-- Create the new subsampled sample
        X_subsample = X[indices_to_keep]

        #-- Performing the clustering algothims DBSCAN or HDBSCAN in random subsample
        if method == 'dbscan':
            cluster_subsample = DBSCAN(eps=galaxy_separation, min_samples=n_galaxies, metric=metric_distance, algorithm='ball_tree').fit(X_subsample)
        elif method == 'hdbscan':
            cluster_subsample = HDBSCAN(min_samples=n_galaxies, metric=metric_distance, algorithm='ball_tree').fit(X_subsample)   

        #-- putting a label to each galaxy in the subsample. This label allows us to assign each galaxy to a substructure
        label_cluster_bic_subsample = cluster_subsample.labels_ # the label_dbscan_bic_random is the label of each odentified substructure in the new sample. This parameter is a numpy.ndarray

        #-- counting the number of groups in the subsample
        n_clusters_subsample = len(set(label_cluster_bic_subsample)) - (1 if -1 in label_cluster_bic_subsample else 0)
        print(f"Number of groups in random sample   : {n_clusters_subsample} ... removing 20% of the original sample")

        #-- project the subsample labels onto the initial sample
        #-- create an array of labels the size of X, initialized to -1 (removed/noise)
        labels_projected = np.full(len(X), -1, dtype=int)

        #-- assign the new labels to the points that were retained
        labels_projected[indices_to_keep] = label_cluster_bic_subsample # indices_to_keep stores the positions of the points that survived

        #-- filter for a pure stability comparison
        #-- filter to compare only the points that were assigned to a cluster (not noise)
        #-- in the initial run and that survived the subsampling (are not -1 in the projection).
        stable_sub_mask = (label_cluster_bic != -1) & (labels_projected != -1)

        #-- apply the mask to the labels
        labels_initial_subsample_stable = label_cluster_bic[stable_sub_mask]
        labels_projected_stable = labels_projected[stable_sub_mask]

        #-- estimating cluster stabillity 
        if len(labels_initial_subsample_stable) >= 2: # at least 2 points are required for the calculation
            ari_score_sub = adjusted_rand_score(labels_initial_subsample_stable, labels_projected_stable)
            print(f"ARI score for random sample         : {ari_score_sub:.4f}")
        else:
            print("ARI cannot be estimated")

        #-- starting bootstrap
        N = len(X) # size of sample
        bootstrap_labels = [] # list to store labels found by bootstrap
        n_clusters_list = []  # list to store cluster numbers 
        ari_scores = []       # list to store the ARI's score 

        print(" ")

        # --- Assesing individual stability 
        print("... assessing individual stability ...")
        print("...")
            
        cluster_ids = [c for c in np.unique(label_cluster_bic) if c != -1]

        stability_scores = {}

        for k in cluster_ids:
            #-- setting A_initial: initial membership
            # Binary array: 1 if point 'i' belongs to group k, 0 otherwise.
            A_initial = (label_cluster_bic == k)
    
            #-- setting B_projected: projected membership
            # Binary array: 1 if point 'i' belongs to group k after subsampling, 0 otherwise.
            B_projected = (labels_projected == k)

            #-- filtering for comparison
            #--- considering only the points that were not removed (i.e., those in the subsample)
            #--- this ensures that the union and intersection are only counted on the surviving points.

            #-- masking the points that survived the subsampling:
            survivors_mask = (labels_projected != -1) 
    
            A_survivors = A_initial[survivors_mask] # A_survivors contains the original classification of those same points
            B_survivors = B_projected[survivors_mask]  # B_survivors contains only the points that were classified 

            #-- avoid division by zero (if the group disappeared completely from the subsampling)
            if np.sum(A_survivors) == 0 and np.sum(B_survivors) == 0:
                jaccard_subsample = 1.0 # Perfect stability if both are empty (shouldn't happen with this filter)
            elif np.sum(A_survivors) == 0 or np.sum(B_survivors) == 0:
                jaccard_subsample = 0.0 # group disappeared or did not exist.
            else:
                #-- using sklearn's jaccard_score function for the binary version
                jaccard_subsample = jaccard_score(A_survivors, B_survivors, pos_label=1)  # the function expects two binary arrays (0 and 1)
            stability_scores[f'Group {k}'] = jaccard_subsample

        for cluster, score in stability_scores.items():
            print(f"{cluster}: {score:.4f}")

        print(" ")

        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ===================================================================== BOOTSTRAP IMPLEMENTATION =====================================================================

        print("... Performing ", nsim, " Bootstrap iterations ...")
        print("...")

        for i in range(nsim):
            #-- Sampling with replacement (Bootstrap)
            sample_indices = np.random.choice(N, size=N, replace=True)
            X_bootstrap = X[sample_indices]

            #-- Apply DBSCAN to the Bootstrap sample
            if method == 'dbscan':
                cluster_bootstrap = DBSCAN(eps=galaxy_separation, min_samples=n_galaxies, metric=metric_distance, algorithm='ball_tree').fit(X_bootstrap)
            elif method == 'hdbscan':
                cluster_bootstrap = HDBSCAN(min_samples=n_galaxies, metric=metric_distance, algorithm='ball_tree').fit(X_bootstrap)  

            labels_boot = cluster_bootstrap.labels_
    
            #-- estimating number of clusters
            n_clusters_boot = len(set(labels_boot)) - (1 if -1 in labels_boot else 0)
            n_clusters_list.append(n_clusters_boot)

            #-- mapping labels to compute ARI index
            labels_boot_mapped = np.full(N, -2, dtype=int)
            labels_boot_mapped[sample_indices] = labels_boot

            bootstrap_labels.append(labels_boot_mapped)
    
            #-- Meassuring ARI index
            #--- ARI index is only estimating for the galaxies that were sampled
            #--- First we compare the original labels of those galaxies that were sampled with the bootstrap labels found
    
            #-- selecting original labels for the sampled galaxies 
            labels_original_subset = label_cluster_bic[sample_indices]
    
            #--Bootstrap labels found for those galaxies that were sampled 
            labels_boot_subset = labels_boot
    
            #-- importing ARI index to compare the similarity between two cluster assignments
            ari_score = adjusted_rand_score(labels_original_subset, labels_boot_subset)
            ari_scores.append(ari_score)
        
        #-- estimating mean and standard deviation of the groups found in the original sample   
        avg_n_clusters = np.mean(n_clusters_list)
        std_n_clusters = np.std(n_clusters_list)

        print(f"Number of groups in Bootstrap               : {avg_n_clusters:.0f}")
        print(f"Standar deviation of the numeber of groups  : {round(std_n_clusters):.0f}")

        #-- estimating mean and standard deviation of the ARI index   
        avg_ari = np.mean(ari_scores)
        std_ari = np.std(ari_scores)

        print(f"mean ARI score for the Bootstrap            : {avg_ari:.4f}")
        print(f"Standard deviation of the ARI score         : {std_ari:.4f}")
        print(" ")


        # --- Assesing individual stability 
        print("... assessing individual stability ...")
        print("...")

        co_occurrence_matrix = np.zeros((N, N))

        for labels in bootstrap_labels:
            #-- only points that were sampled (label != -2) are considered
            sampled_indices = np.where(labels != -2)[0]
    
            #-- create a temporal co-classification matrix for this iteration
            temp_co_matrix = np.zeros((N, N))
    
            #-- iterates over all pairs of sampled points
            for i in sampled_indices:
                for j in sampled_indices:
                    if i <= j: # Only the upper half is calculated for symmetry
                    # If both points have the same cluster label (including noise -1)
                    # and both were sampled (label != -2), increment the counter.
                        if labels[i] == labels[j] and labels[i] != -2:
                            temp_co_matrix[i, j] = 1
                        if i != j:
                            temp_co_matrix[j, i] = 1

            co_occurrence_matrix += temp_co_matrix

        #-- normalization and Stability Visualization
        #-- divide by the number of iterations to obtain the co-occurrence probability
        co_occurrence_probability = co_occurrence_matrix / nsim

    labels_original = label_cluster_bic # label in the original iteration 
    unique_clusters = set(labels_original)
    stability_results = {}

    for cluster_label in unique_clusters:
        if cluster_label == -1:
            #-- exclude "noise" points
            continue

        #-- find the indices of the points that belong to this cluster
        cluster_indices = np.where(labels_original == cluster_label)[0]

        #-- if the cluster has fewer than 2 points, co-occurrence cannot be calculated
        if len(cluster_indices) < 2:
            stability_results[cluster_label] = 0.0
            continue

        #-- extract the co-occurrence submatrix for this cluster
        submatrix = co_occurrence_probability[np.ix_(cluster_indices, cluster_indices)]

        #-- estimate the average co-occurrence:
        #--- Sum all the elements of the submatrix
        #--- Subtract the diagonal (the co-occurrence of a point with itself is always 1)
        #--- Divide by the total number of pairs (N * (N - 1))
    
        N_k = len(cluster_indices)
        sum_co_occurrence = np.sum(submatrix)
    
        #-- subtract the diagonal (N_k)
        total_co_occurrence_pairs = sum_co_occurrence - N_k

        #-- total number of possible unique pairs within the cluster (excluding the diagonal)
        num_pairs = N_k * (N_k - 1)
    
        if num_pairs > 0:
            avg_co_occurrence = total_co_occurrence_pairs / num_pairs
        else:
            avg_co_occurrence = 0.0

        stability_results[cluster_label] = avg_co_occurrence

    #-- print the results
    for label, stability in stability_results.items():
        print(f"Group {label} : {stability:.4f}")

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------
	
	print(" ")
    print("-- ending LAGASU stability--")

    #-- returning output quantity

    return

#####################################################################################################################################################################################
#####################################################################################################################################################################################

def lagasu_supercluster(id_galaxy, ra_galaxy, dec_galaxy, redshift_galaxy, range_cuts, galaxy_separation, n_galaxies, metric_distance, method):

    """ LAGASU_supercluster is a function that assigns galaxies 
    to different groups in and around a galaxy cluster when 
    there is no information about r200 or when it is desired to 
    identify groups in large areas such as supercluster

    This function was developed by D. E. Olave-Rojas (27/11/2025)
    based on LAGASU

    The input of LAGASU_supercluster can be a sample of galaxies 
    in a cluster or supercluster.
    
    lagasu(id_galaxy, ra_galaxy, dec_galaxy, redshift_galaxy, 
    range_cuts, nsim, galaxy_separation, n_galaxies, metric_distance, 
    method)

    :param id_galaxy: id of each galaxy in the sample
	:param ra_galaxy: is the Right Ascension of each galaxy in
        the sample. This parameter must be in degree units
    :param dec_galaxy: is the Declination of each galaxy in
        the sample. This parameter must be in degree units
    :param redshift_galaxy: redshift of each galaxy in the 
        cluster
    :param range_cuts: to perform the cuts in the redshift 
        space is neccesary to give a range that allows find the 
        best number of cuts in redshift. The parameter "range_cuts" 
        is the end of the range which is defined in LAGASU_supercluster
        as range(1, range_cuts). Therefore, range_cuts must be greater 
        than 1
    :param galaxy_separation: physical separation between galaxies 
        in a group. If metric_distance="euclidean" is selected, 
        the unit should be degrees. If metric_distance="haversine" is 
        selected, the unit should be radians.
    :param n_galaxies: minimum number of galaxies to define a group
    :param metric_distance: metric used to calcule the distance 
        between instances in a feature array. Metric must be 
        'euclidean' or 'haversine'
    :param method: clustering algorithm used to grouping galaxies
        in substructures. Method must be 'dbscan' or 'hdbscan'

    :type id_galaxy         : array
    :type ra_galaxy         : array
    :type dec_galaxy        : array
	:type redshift_galaxy   : array
    :type range_cuts        : int
    :type galaxy_separation : int, float 
    :type n_galaxies        : int, float
    :type metric_distance   : string  
    :type method            : string  

	:returns: label to each galaxy,
        which corresponds to identify
        each group
	:rtype: array
   
    .. note::
    
    LAGASU_supercluster will give us three labels as output: 
    i) lagasu[4] that corresponds to the label 
    putting by GMM and varies between 0 to N, and 
    ii) lagasu[5] that corresponds to the label 
    putting by DBSCAN after to run gmm and varies 
    between -1 to N, where -1 corresponds to noise
    and galaxies within a substructure have a label 
    between 0 to N.

	"""
    print("-- starting LAGASU_supercluster --")
    print("--- input parameters ---")
    print("Number of members    :", n_galaxies)
    print("metric               :", metric_distance)
    print("method               :", method)

    print(" ")
    
    redshift_cluster = biweight_location(redshift_galaxy)

    #-- evaluating if there is a value equal to -99. of 99. or < 0. in the redshift galaxy parameter
    if ((redshift_galaxy > -99.) & (redshift_galaxy <= 0.)).any():
        print(" WARNING!! There is a redshift lower to 0 or equal to 0")
        print("... replacing bad values ...")
        boolean_mask = (redshift_galaxy > -99.) & (redshift_galaxy <= 0.)  # create a boolean mask. It is an array with True/False and True corresponds to the old value
        redshift_galaxy[boolean_mask] = redshift_cluster
    elif np.isin(-99., redshift_galaxy).any():
        print(" WARNING!! There is a redshift equal to -99.")
        print("... replacing bad values ...")
        boolean_mask = (redshift_galaxy == -99.)  # create a boolean mask. It is an array with True/False and True corresponds to the old value
        redshift_galaxy[boolean_mask] = redshift_cluster
    elif np.isin(99., redshift_galaxy).any():
        print(" WARNING!! There is a redshift equal to 99.")
        print("... replacing bad values ...")
        boolean_mask = (redshift_galaxy == 99.)  # create a boolean mask. It is an array with True/False and True corresponds to the old value
        redshift_galaxy[boolean_mask] = redshift_cluster
    else: 
        print("Redshift array is ok")

    print("  ")
    print(f"... starting GMM ...")

    #-- Gaussian Mixture Models (GMM) divides the sample in ranges of redshift in order to consider the volume of the cluster
    #-- The divissión is perform without an arbitrary number of cuts in the redshift distribution and the number of cuts are in a range
    #-- The divissión in the redshift space is perform without an arbitrary number of cuts by using the Bayesian Information Criterion (BIC), 
    # which selects the better number of cuts according to the data 
    #-- The number of cuts found by BIC are in a range given in the input as "range_cuts"
    
    #-- creating a transposed array with the redshift_galaxy to be used as input in the GMM implementation
    candidate_galaxies = np.array([redshift_galaxy]).T

    #-- defining the lowest bic parameter
    lowest_bic = np.infty

    #-- initializing the variable
    bic = []
    
    #-- defining the number of possible cuts in the redshift space
    n_components_range = range(1, range_cuts)

    #-- defining a list with the types of covariances used in the implementation of GMM-BIC
    cv_types = ['spherical', 'tied', 'diag', 'full']
    
    #--- START OF LOOP ---

    for cv_type in cv_types:
    
        for n_components in n_components_range:
    
    	    #-- Fit a mixture of Gaussians with Expectation Maximization (EM)
            gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type, random_state=4)

            #-- Fit the Gaussian over the data 
            gmm.fit(candidate_galaxies)

            bic.append(gmm.bic(candidate_galaxies))

            #-- selecting the best fit
            if bic[-1] < lowest_bic:

                lowest_bic = bic[-1]

                best_gmm = gmm

    # -- END OF LOOP --
    bic = np.array(bic)
    clf = best_gmm
    print("Best model GMM       :", best_gmm)
    
    #-- defining the parameters of the best fit
    n_cuts_bic = clf.n_components # number of cuts do by the algorithms 
    print("Number of GMM cuts  :", n_cuts_bic)
    print(" ")

    print("... starting DBSCAN ...")
    #-- putting a label to each galaxy. This label allows us to separe and assign each galaxy to a redshift cut
    labels_bic = clf.predict(candidate_galaxies) # using this label we can separate galaxies into n groups that correspond to redshifts cuts
    
    #-- To assign galaxies to different substructures we use Density-Based Spatial Clustering of Applications with Noise (DBSCAN, Ester et al. 1996)
    #-- To identify groups using DBSCAN we must define a minimum number of neighbouring objects separated by a specific distance. 
    #-- DBSCAN does not assign all objects in the sample to one group (Ester et al. 1996) and we can remove the galaxies that are not spatially grouped with others

    #-- sorting the output labels given by GMM-BIC implementation 
    sorted_labels_bic = np.sort(labels_bic)

    #--- START OF LOOP ---
    for ii in range(0,n_cuts_bic): # n_cuts_bic are the number of cuts given by the GMM implementation

        #-- defining a single redshift cut in which DBSCAN will be apply
        n_redshift_groups = np.where(labels_bic == ii)[0] 

        #-- selecting galaxies that are part of a signle cut in redshift
        ra = ra_galaxy[n_redshift_groups]
        dec = dec_galaxy[n_redshift_groups]

        #-- generate sample data
        #-- creating a transposed array with the poition of galaxies to be used as input in the DBSCAN implementation
        #-- metric implementation was added an June 17, 2024

        if metric_distance == 'euclidean':
            X = np.array([ra, dec]).T
        if metric_distance == 'haversine': 
            ra_rad = np.radians(ra)
            dec_rad = np.radians(dec)
            X = np.array([ra_rad, dec_rad]).T

        #-- Performing the clustering algothims DBSCAN or HDBSCAN
        if method == 'dbscan':
            cluster = DBSCAN(eps=galaxy_separation, min_samples=n_galaxies, metric=metric_distance, algorithm='ball_tree').fit(X)
        elif method == 'hdbscan':
            cluster = HDBSCAN(min_samples=n_galaxies, metric=metric_distance, algorithm='ball_tree').fit(X)   
    
        #-- putting a label to each galaxy. This label allows us to assign each galaxy to a substructure
        label_cluster_bic = cluster.labels_ # the label_dbscan_bic is the label of each odentified substructure. This parameter is a numpy.ndarray

        #-- selecting the labels of the groups found by DBSCAN in each redshift cut
        groups = np.unique(label_cluster_bic) # number of groups in each redshift cut

        if ii != 0:
            p = np.append(p,groups, axis=0)
            tam_groups = np.append(tam_groups, len(groups))
            labels_dbscan_bic = np.append(labels_dbscan_bic,label_cluster_bic)
            first_element_groups = np.append(first_element_groups, groups[0])

        #-- process of the first iteration: defining variables
        else:                
            p = groups # array with the unique labels that could be assign to a single galaxy
            tam_groups = len(groups) # size of the each idenfied substructure
            labels_dbscan_bic = label_cluster_bic # label used to identified each substructure
            first_element_groups = groups[0] # first element in the array with the labels of the substructures
        
    #-- Finally, we need to label the substructures from 0 to n
    #-- Initialization of variables
    groups_pos = []
    p_pos = 0

    #-- Position of the first element in each group is searched here 
    #-- This implementation allows us to consider the case of a little sample in which all galaxies are assign to a single substructure 
    if type(tam_groups) == int:
        tam_groups = np.array([tam_groups])
    
    else:
        tam_groups = tam_groups

    #--- START OF LOOP ---
    for ii in range(0,len(tam_groups)):

        if (len(tam_groups)==1):

            groups_pos = 0

        else: 

            for e in range(0,tam_groups[ii]):

                if(p[p_pos+e]==first_element_groups[ii] and ii==0):
                    groups_pos = 0

                if(p[p_pos+e]==first_element_groups[ii] and ii!=0):
                    groups_pos = np.append(groups_pos,(p_pos+e))
        
            p_pos = p_pos + tam_groups[ii]
    # -- END OF LOOP --

    #-- Here the correlative is assembled eliminating -1
    p3 = [] # p3 is an array with the label of all idenitified substructures

    #--- START OF LOOP ---
    for j in range(0,len(p)):

        if p[j] != -1:
            p3 = np.append(p3, j)

    for l in range(0,len(p3)):

        p3[l] = l
    # -- END OF LOOP --

    #-- printing the total number of substructures in the cluster

    #-- initializing the variables
    p_pos = 0
    correlative = 0

    #-- Here the array p2 is assembled, which is an array with the label of all identified substructures plus noise
    #--- START OF LOOP ---
    for ii in range(0,len(tam_groups)):

        for e in range(0,tam_groups[ii]):

            if(e==0 and ii==0):
                p2 = p[0]

                if(p[0] != -1):
                    correlative +=1

            elif(p[p_pos+e]== -1):
                p2 = np.append(p2,-1)

            else:
                p2 = np.append(p2,correlative)
                correlative +=1

        p_pos = p_pos + tam_groups[ii]
    # -- END OF LOOP --

    #-- This implementation allows us to consider the case of a little sample in which all galaxies are assign to a single substructure 
    if type(groups_pos) == int:
        groups_pos = np.array([groups_pos])
    
    else:
        groups_pos = groups_pos
#========================================================
    #-- This loop allows us to assign a label from 0 to n lo each substructures plus noise which is labelled with -1
    #--- START OF LOOP ---

    #-- selecting the labels of the groups found by DBSCAN 
    final_groups = np.unique(labels_dbscan_bic)

    if  len(final_groups) == 1: # in this case none of galaxies are part of a subhalo

        labels_dbscan_corr = labels_dbscan_bic 
    
    else: 
        for k in range(0,len(labels_dbscan_bic)):
            aux = 0
            
            if(k == 0):
                while( labels_dbscan_bic[k] != p[groups_pos[sorted_labels_bic[k]]+aux]):
                    aux +=1
                labels_dbscan_corr = p2[groups_pos[sorted_labels_bic[k]]+aux]

            else:
                while( labels_dbscan_bic[k] != p[groups_pos[sorted_labels_bic[k]]+aux]):
                    aux +=1
                labels_dbscan_corr = np.append(labels_dbscan_corr,p2[groups_pos[sorted_labels_bic[k]]+aux])
    # -- END OF LOOP --

    for ii in range(0,n_cuts_bic):

        n_redshift_groups = np.where(labels_bic == ii)[0]

        id_gal_out = id_galaxy[n_redshift_groups]
        ra_out = ra_galaxy[n_redshift_groups]
        dec_out = dec_galaxy[n_redshift_groups]
        redshift_gal_out = redshift_galaxy[n_redshift_groups]
        gmm_labels = labels_bic[n_redshift_groups]

        if ii != 0:
            id_substructures = np.append(id_substructures,id_gal_out)
            ra_substructures = np.append(ra_substructures,ra_out)
            dec_substructures = np.append(dec_substructures, dec_out)
            redshift_substructures = np.append(redshift_substructures, redshift_gal_out)
            gmm_substructures = np.append(gmm_substructures, gmm_labels)
            
        else:              #-- process of the first iteration: defining variables
            id_substructures = id_gal_out
            ra_substructures = ra_out
            dec_substructures = dec_out
            redshift_substructures = redshift_gal_out
            gmm_substructures = gmm_labels

    # -- renaming the substructures identified by using lagasu in order to identify the principal halo and separate it from the group
    # if label == -1 the galaxy is only part of the principal halo. If galaxy is != -1 the galaxy is in a group

    #-- building matrix with output quantities
    lagasu_parameters = np.array([id_substructures, ra_substructures, dec_substructures, redshift_substructures, gmm_substructures, labels_dbscan_corr], dtype=object)

    print("-- ending LAGASU_supercluster --")

    #-- returning output quantity
    return lagasu_parameters



#####################################################################################################################################################################################
#####################################################################################################################################################################################

def lagasu_position(id_galaxy, ra_galaxy, dec_galaxy, range_cuts, galaxy_separation, n_galaxies, metric_distance, method):

    """ lagasu_position is a function that assigns galaxies 
    to different clusters within superclusters or substructures 
    in and around galaxy cluster considering only the position
    of galaxies

    The input of lagasu_position can be a sample of galaxies 
    in a cluster previously selected as members o a sample of
    galaxies in a supercluster
    
    lagasu_position(id_galaxy, ra_galaxy, dec_galaxy, range_cuts, 
    galaxy_separation, n_galaxies, metric_distance, method)

    This function was developed by D. E. Olave-Rojas (07/09/2024)
    and was based on lagasu

	:param ra_galaxy: is the Right Ascension of each galaxy in
        the sample. This parameter must be in degree units
    :param dec_galaxy: is the Declination of each galaxy in
        the sample. This parameter must be in degree units
    :param range_cuts: to perform the cuts in the redshift 
        space is neccesary to give a range that allows find the 
        best number of cuts in redshift. The parameter "range_cuts" 
        is the end of the range which is defined in LAGASU as 
        range(1, range_cuts). Therefore, range_cuts must be greater 
        than 1
    :param galaxy_separation: physical separation between 
        galaxies in a substructure the units must be the 
        same as the ra_galaxy and dec_galaxy
    :param n_galaxies: minimum number of galaxies to define a group
    :param metric_distance: metric used to calcule the distance 
        between instances in a feature array. Metric must be 
        'euclidean' or 'haversine'
    :param method: clustering algorithm used to grouping galaxies
        in substructures. Method must be 'dbscan' or 'hdbscan'

    :type ra_galaxy         : array
    :type dec_galaxy        : array
    :type range_cuts        : int
    :type galaxy_separation : int, float 
    :type n_galaxies        : int, float
    :type metric_distance   : string  
    :type method            : string  

	:returns: label to each galaxy,
        which corresponds to identify
        each substructure
	:rtype: array
   
    .. note::
    
    LAGASU_POSITION will give us three labels as output: 
    i) lagasu_position[3] corresponds to the label putting 
    by GMM and varies between 0 to N, ii) lagasu_position[4] 
    corresponds to the label putting by DBSCAN/HDBSCAN
    after running gmm and varies between -1 to N. 
    iii) lagasu_position[5] corresponds to the corrected label 
    after running GMM+DBSCAN/HDBSCAN and varies between 
    -1 to N. 

    Label == -1 corresponds to galaxies that are not
    part of a structure and galaxies within a substructure 
    have a label between 0 to N

    For lagasu_position[4]

	"""
    print("-- starting lagasu_position --")
    print("--- input parameters ---")
    print("Number of members    :", n_galaxies)
    print("metric               :", metric_distance)
    print("method               :", method)

    #-- Gaussian Mixture Models (GMM) divides the sample in ranges of redshift in order to consider the volume of the cluster
    #-- The divissión is perform without an arbitrary number of cuts in the redshift distribution and the number of cuts are in a range
    #-- The divissión in the redshift space is perform without an arbitrary number of cuts by using the Bayesian Information Criterion (BIC), 
    # which selects the better number of cuts according to the data 
    #-- The number of cuts found by BIC are in a range given in the input as "range_cuts"
    
    #-- creating a transposed array with the redshift_galaxy to be used as input in the GMM implementation
    candidate_galaxies = np.array([ra_galaxy, dec_galaxy]).T

    #-- defining the lowest bic parameter
    lowest_bic = np.infty

    #-- initializing the variable
    bic = []
    
    #-- defining the number of possible cuts in the redshift space
    n_components_range = range(1, range_cuts)
    
    #-- defining a list with the types of covariances used in the implementation of GMM-BIC
    cv_types = ['spherical', 'tied', 'diag', 'full']
    
    #--- START OF LOOP ---
    t_start = time.localtime()
    start_time = time.strftime("%H:%M:%S", t_start)
    print("start time GMM       :", start_time)

    for cv_type in cv_types:
    
        for n_components in n_components_range:
    
    	    #-- Fit a mixture of Gaussians with Expectation Maximization (EM)
            gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type, random_state=4)

            #-- Fit the Gaussian over the data 
            gmm.fit(candidate_galaxies)

            bic.append(gmm.bic(candidate_galaxies))

            #-- selecting the best fit
            if bic[-1] < lowest_bic:

                lowest_bic = bic[-1]

                best_gmm = gmm

    # -- END OF LOOP --
    bic = np.array(bic)
    clf = best_gmm

    # -- END OF LOOP --
    bic = np.array(bic)
    clf = best_gmm
#    bars = []

    # -- printing time
    t_end = time.localtime()
    end_time = time.strftime("%H:%M:%S", t_end)
    print("end time GMM         :", end_time)

    print("Best model GMM       :", best_gmm)

    #-- defining the parameters of the best fit
    n_cuts_bic = clf.n_components # number of cuts do by the algorithms 

    #-- putting a label to each galaxy. This label allows us to separe and assign each galaxy to a redshift cut
    labels_bic = clf.predict(candidate_galaxies) # using this label we can separate galaxies into n groups that correspond to redshifts cuts

    #-- To assign galaxies to different substructures we use Density-Based Spatial Clustering of Applications with Noise (DBSCAN, Ester et al. 1996)
    #-- To identify groups using DBSCAN we must define a minimum number of neighbouring objects separated by a specific distance. 
    #-- DBSCAN does not assign all objects in the sample to one group (Ester et al. 1996) and we can remove the galaxies that are not spatially grouped with others

    #-- sorting the output labels given by GMM-BIC implementation 
    sorted_labels_bic = np.sort(labels_bic)

    #--- START OF LOOP ---
    for ii in range(0,n_cuts_bic): # n_cuts_bic are the number of cuts given by the GMM implementation

        #-- defining a single redshift cut in which DBSCAN will be apply
        n_gmm_groups = np.where(labels_bic == ii)[0] 

        #-- selecting galaxies that are part of a signle cut in redshift
        ra = ra_galaxy[n_gmm_groups]
        dec = dec_galaxy[n_gmm_groups]

        #-- generate sample data
        #-- creating a transposed array with the poition of galaxies to be used as input in the DBSCAN implementation
        #-- metric implementation was added an June 17, 2024

        if metric_distance == 'euclidean':
            X = np.array([ra, dec]).T
        if metric_distance == 'haversine': 
            ra_rad = np.radians(ra)
            dec_rad = np.radians(dec)
            X = np.array([ra_rad, dec_rad]).T

        #-- Performing the clustering algothims DBSCAN or HDBSCAN
        if method == 'dbscan':
            cluster = DBSCAN(eps=galaxy_separation, min_samples=n_galaxies, metric=metric_distance, algorithm='ball_tree').fit(X)
        elif method == 'hdbscan':
            cluster = HDBSCAN(min_samples=n_galaxies, metric=metric_distance, algorithm='ball_tree').fit(X)   
    
        #-- putting a label to each galaxy. This label allows us to assign each galaxy to a substructure
        label_cluster_bic = cluster.labels_ # the label_dbscan_bic is the label of each odentified substructure. This parameter is a numpy.ndarray

        #-- selecting the labels of the groups found by DBSCAN in each redshift cut
        groups = np.unique(label_cluster_bic) # number of groups in each redshift cut

        if ii != 0:
            p = np.append(p,groups, axis=0)
            tam_groups = np.append(tam_groups, len(groups))
            labels_dbscan_bic = np.append(labels_dbscan_bic,label_cluster_bic)
            first_element_groups = np.append(first_element_groups, groups[0])

        #-- process of the first iteration: defining variables
        else:                
            p = groups # array with the unique labels that could be assign to a single galaxy
            tam_groups = len(groups) # size of the each idenfied substructure
            labels_dbscan_bic = label_cluster_bic # label used to identified each substructure
            first_element_groups = groups[0] # first element in the array with the labels of the substructures
        
    #-- Finally, we need to label the substructures from 0 to n
    #-- Initialization of variables
    groups_pos = []
    p_pos = 0

    #-- Position of the first element in each group is searched here 
    #-- This implementation allows us to consider the case of a little sample in which all galaxies are assign to a single substructure 
    if type(tam_groups) == int:
        tam_groups = np.array([tam_groups])
    
    else:
        tam_groups = tam_groups

    #--- START OF LOOP ---
    for ii in range(0,len(tam_groups)):

        if (len(tam_groups)==1):

            groups_pos = 0

        else: 

            for e in range(0,tam_groups[ii]):

                if(p[p_pos+e]==first_element_groups[ii] and ii==0):
                    groups_pos = 0

                if(p[p_pos+e]==first_element_groups[ii] and ii!=0):
                    groups_pos = np.append(groups_pos,(p_pos+e))
        
            p_pos = p_pos + tam_groups[ii]
    # -- END OF LOOP --

    #-- Here the correlative is assembled eliminating -1
    p3 = [] # p3 is an array with the label of all idenitified substructures

    #--- START OF LOOP ---
    for j in range(0,len(p)):

        if p[j] != -1:
            p3 = np.append(p3, j)

    for l in range(0,len(p3)):

        p3[l] = l
    # -- END OF LOOP --

    #-- printing the total number of substructures in the cluster
#    print("number of substructures =", len(p3))

    #-- initializing the variables
    p_pos = 0
    correlative = 0

    #-- Here the array p2 is assembled, which is an array with the label of all identified substructures plus noise
    #--- START OF LOOP ---
    for ii in range(0,len(tam_groups)):

        for e in range(0,tam_groups[ii]):

            if(e==0 and ii==0):
                p2 = p[0]

                if(p[0] != -1):
                    correlative +=1

            elif(p[p_pos+e]== -1):
                p2 = np.append(p2,-1)

            else:
                p2 = np.append(p2,correlative)
                correlative +=1

        p_pos = p_pos + tam_groups[ii]
    # -- END OF LOOP --

    #-- This implementation allows us to consider the case of a little sample in which all galaxies are assign to a single substructure 
    if type(groups_pos) == int:
        groups_pos = np.array([groups_pos])
    
    else:
        groups_pos = groups_pos
#========================================================
    #-- This loop allows us to assign a label from 0 to n lo each substructures plus noise which is labelled with -1
    #--- START OF LOOP ---

    #-- selecting the labels of the groups found by DBSCAN 
    final_groups = np.unique(labels_dbscan_bic)

    if  len(final_groups) == 1: # in this case none of galaxies are part of a subhalo

        labels_dbscan_corr = labels_dbscan_bic 
    
    else: 
        for k in range(0,len(labels_dbscan_bic)):
            aux = 0
            
            if(k == 0):
                while( labels_dbscan_bic[k] != p[groups_pos[sorted_labels_bic[k]]+aux]):
                    aux +=1
                labels_dbscan_corr = p2[groups_pos[sorted_labels_bic[k]]+aux]

            else:
                while( labels_dbscan_bic[k] != p[groups_pos[sorted_labels_bic[k]]+aux]):
                    aux +=1
                labels_dbscan_corr = np.append(labels_dbscan_corr,p2[groups_pos[sorted_labels_bic[k]]+aux])
    # -- END OF LOOP --

    for ii in range(0,n_cuts_bic):

        n_gmm_bic_groups = np.where(labels_bic == ii)[0]

        id_gal_out = id_galaxy[n_gmm_bic_groups]
        ra_out = ra_galaxy[n_gmm_bic_groups]
        dec_out = dec_galaxy[n_gmm_bic_groups]
        gmm_labels = labels_bic[n_gmm_bic_groups]
        labels_gmm_dbscan = labels_dbscan_bic[n_gmm_bic_groups]

        if ii != 0:
            id_substructures = np.append(id_substructures,id_gal_out)
            ra_substructures = np.append(ra_substructures,ra_out)
            dec_substructures = np.append(dec_substructures, dec_out)
            gmm_substructures = np.append(gmm_substructures, gmm_labels)
            gmm_dbscan_substructures = np.append(gmm_dbscan_substructures, labels_gmm_dbscan)
            
        else:              #-- process of the first iteration: defining variables
            id_substructures = id_gal_out
            ra_substructures = ra_out
            dec_substructures = dec_out
            gmm_substructures = gmm_labels
            gmm_dbscan_substructures = labels_gmm_dbscan

    # -- renaming the substructures identified by using lagasu in order to identify the principal halo and separate it from the substructures
    # if label == -1 the galaxy is only part of the principal halo. If galaxy is != -1 the galaxy is in a substructure

    #-- building matrix with output quantities
    lagasu_parameters = np.array([id_substructures, ra_substructures, dec_substructures, gmm_substructures, gmm_dbscan_substructures, labels_dbscan_corr], dtype=object)

    print("-- ending lagasu_position --")

    #-- returning output quantity
    return lagasu_parameters

