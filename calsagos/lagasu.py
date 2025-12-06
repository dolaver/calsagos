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
# version       : 0.1.6
# update        : December 05, 2025
# maintainer    : D. E. Olave-Rojas 
#-------------------------------------------------------------------

#-- Import preexisting python modules
import numpy as np
import time
import pandas as pd
from sklearn import mixture
from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import jaccard_score
from astropy.stats import biweight_location

from calsagos import utils

__author__ = 'D. E. Olave-Rojas & D. A. Olave-Rojas'
__email__ = 'daniela.olave@utalca.cl'
__version__ = '0.1.6'
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

    #-- creating a boolean mask. It is an array with True/False and True corresponds to the old value
    anomalous_mask = (
        ((redshift_galaxy > -99.) & (redshift_galaxy <= 0.)) | 
        (redshift_galaxy == -99.) | 
        (redshift_galaxy == 99.)
    )

    if anomalous_mask.any():
        #-- warning message 
        print(" WARNING!! Redshift values -99, 99, or <= 0 were found and will be replaced.")
        print("... replacing anomalous values with the redshift cluster value ...")
        #-- appliyinh mask 
        redshift_galaxy[anomalous_mask] = redshift_cluster
    else:
        print("Redshift array is OK")

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
#    bars = []
    
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
        else:
            #-- CORRECTION: handling unrecognized metrics
            #-- this correction was done on December 05, 2025
            X = np.array([ra, dec]).T
            metric_distance = 'euclidean'
            raise ValueError("Metric distance must be 'euclidean' or 'haversine' -- 'euclidean' metrics is used")
        
        #-- Performing the clustering algothims DBSCAN or HDBSCAN
        if method == 'dbscan':
            cluster = DBSCAN(eps=galaxy_separation, min_samples=n_galaxies, metric=metric_distance, algorithm='ball_tree').fit(X)
        elif method == 'hdbscan':
            cluster = HDBSCAN(min_cluster_size=n_galaxies, metric=metric_distance, algorithm='ball_tree').fit(X)   
        else:
            cluster = DBSCAN(eps=galaxy_separation, min_samples=n_galaxies, metric=metric_distance, algorithm='ball_tree').fit(X)
            #-- handling unrecognized method
            #-- this correction was done on December 05, 2025
            raise ValueError("Clustering method must be 'dbscan' or 'hdbscan' -- 'dbscan' method is used")            
     
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
    
    # -- estimating central redshift of the distribution
    good_z = np.where((redshift_galaxy != 99) & (redshift_galaxy > 0))[0]
    redshift_cluster = biweight_location(redshift_galaxy[good_z])

    #-- creating a boolean mask. It is an array with True/False and True corresponds to the old value
    anomalous_mask = (
        ((redshift_galaxy > -99.) & (redshift_galaxy <= 0.)) | 
        (redshift_galaxy == -99.) | 
        (redshift_galaxy == 99.)
    )

    if anomalous_mask.any():
        #-- warning message 
        print(" WARNING!! Redshift values -99, 99, or <= 0 were found and will be replaced.")
        print("... replacing anomalous values with the redshift cluster value ...")
        #-- appliyinh mask 
        redshift_galaxy[anomalous_mask] = redshift_cluster
    else:
        print("Redshift array is OK")

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
        else:
            #-- CORRECTION: handling unrecognized metrics
            #-- this correction was done on December 05, 2025
            X = np.array([ra, dec]).T
            metric_distance = 'euclidean'
            raise ValueError("Metric distance must be 'euclidean' or 'haversine' -- 'euclidean' metrics is used")
        
        #-- Performing the clustering algothims DBSCAN or HDBSCAN
        if method == 'dbscan':
            cluster = DBSCAN(eps=galaxy_separation, min_samples=n_galaxies, metric=metric_distance, algorithm='ball_tree').fit(X)
        elif method == 'hdbscan':
            cluster = HDBSCAN(min_cluster_size=n_galaxies, metric=metric_distance, algorithm='ball_tree').fit(X)   
        else:
            cluster = DBSCAN(eps=galaxy_separation, min_samples=n_galaxies, metric=metric_distance, algorithm='ball_tree').fit(X)
            #-- handling unrecognized method
            #-- this correction was done on December 05, 2025
            raise ValueError("Clustering method must be 'dbscan' or 'hdbscan' -- 'dbscan' method is used")    
            
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

def dbscan_stability(ra_galaxy, dec_galaxy, galaxy_separation, n_galaxies, metric_distance, method, nsim, percent_add, percent_rem, save_path):

    """ dbscan_stability is a function that evaluate 
    the stability of the assigantion of galaxies to
    different groups

    This function was developed by D. E. Olave-Rojas (11/27/2025)
    
	:param ra_galaxy: is the Right Ascension of each galaxy in
        the sample. This parameter must be in degree units
    :param dec_galaxy: is the Declination of each galaxy in
        the sample. This parameter must be in degree units
    :param redshift_galaxy: redshift of each galaxy in the 
        cluster
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
    :param nsim: number of simulation in boostrap resampling
    :param percent_add: percentage that will be added to the sample.
        This parameter must be a number between 0 to 1. 
    :param percent_rem: percentage that will be removed from the 
        sample. This parameter must be a number between 0 to 1.
    :param save_path: path where the files will be saved

    :type ra_galaxy         : array
    :type dec_galaxy        : array
	:type redshift_galaxy   : array
    :type galaxy_separation : int, float 
    :type n_galaxies        : int, float
    :type metric_distance   : string  
    :type method            : string
    :type nsim              : int  
    :type percent_add       : float
    :type percent_rem       : float
    :type save_path         : string
    
    .. note:: This test only evaluates the stability
    of dbscan/hdbscan in the assignment to galaxies 
    in groups.

	"""
    print("-- starting DBSCAN stability --")
    print(" ")
    print("--- input parameters ---")
    print("Number of members        :", n_galaxies)
    print("metric                   :", metric_distance)
    print("method                   :", method)
    print("No. simulations          :", nsim)
    print("percentage to adding     :", round(percent_add*100))
    print("percentage to removing   :", round(percent_rem*100))

    print("  ")

    #-- selecting galaxies that are part of a signle cut in redshift
    ra = ra_galaxy
    dec = dec_galaxy

    #-- generate sample data
    #-- creating a transposed array with the poition of galaxies to be used as input in the DBSCAN implementation
    #-- metric implementation was added an June 17, 2024

    if metric_distance == 'euclidean':
        X = np.array([ra, dec]).T
    elif metric_distance == 'haversine': 
        ra_rad = np.radians(ra)
        dec_rad = np.radians(dec)
        X = np.array([ra_rad, dec_rad]).T
    else:
        #-- CORRECTION: handling unrecognized metrics
        #-- this correction was done on December 05, 2025
        X = np.array([ra, dec]).T
        metric_distance = 'euclidean'
        raise ValueError("Metric distance must be 'euclidean' or 'haversine' -- 'euclidean' metrics is used")
    
    #-- Performing the clustering algothims DBSCAN or HDBSCAN
    if method == 'dbscan':
        cluster = DBSCAN(eps=galaxy_separation, min_samples=n_galaxies, metric=metric_distance, algorithm='ball_tree').fit(X)
    elif method == 'hdbscan':
        cluster = HDBSCAN(min_cluster_size=n_galaxies, metric=metric_distance, algorithm='ball_tree').fit(X)   
    else:
        cluster = DBSCAN(eps=galaxy_separation, min_samples=n_galaxies, metric=metric_distance, algorithm='ball_tree').fit(X)
        #-- handling unrecognized method
        #-- this correction was done on December 05, 2025
        raise ValueError("Clustering method must be 'dbscan' or 'hdbscan' -- 'dbscan' method is used")    
    
    #-- putting a label to each galaxy. This label allows us to assign each galaxy to a substructure
    label_cluster_bic = cluster.labels_ # the label_dbscan_bic is the label of each odentified substructure. This parameter is a numpy.ndarray

    #-- counting the number of groups in the random sample
    n_clusters_original_bic = [c for c in np.unique(label_cluster_bic) if c != -1]
    print(f"Number of groups in original sample         : {len(n_clusters_original_bic)}")

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ================================================================= ADD RANDOM POINTS TO THE SAMPLE ==================================================================

    #-- creating random samples 
    #-- stabillity adding points was added an November 26, 2025
    print(" ")
    print("... adding random points to the sample ...")

    #-- 
    n_clusters_list_random = []  # list to store cluster numbers 
    ari_global_results = []
    jaccard_individual_results = {k: [] for k in n_clusters_original_bic}

    for i in range(nsim):
        #-- Generate random points (the seed must be different in each iteration)
        np.random.seed(i) # use the loop index as the seed to ensure randomness
        number = int((len(X)*percent_add)) # add a random sample equivalent to 80% of the original sample

        random_points = np.random.uniform(low=X.min(axis=0), 
                                         high=X.max(axis=0), 
                                         size=(number, 2))

        #-- combining samples
        X_random = np.vstack([X, random_points])

        #-- Performing the clustering algothims DBSCAN or HDBSCAN in random sample
        if method == 'dbscan':
            cluster_random = DBSCAN(eps=galaxy_separation, min_samples=n_galaxies, metric=metric_distance, algorithm='ball_tree').fit(X_random)
        elif method == 'hdbscan':
            cluster_random = HDBSCAN(min_cluster_size=n_galaxies, metric=metric_distance, algorithm='ball_tree').fit(X_random)   

        #-- putting a label to each galaxy in the random sample. This label allows us to assign each galaxy to a substructure
        label_cluster_bic_random = cluster_random.labels_ # the label_dbscan_bic_random is the label of each odentified substructure in the new sample. This parameter is a numpy.ndarray

        #-- estimating number of clusters
        n_clusters_random = len(set(label_cluster_bic_random)) - (1 if -1 in label_cluster_bic_random else 0)
        n_clusters_list_random.append(n_clusters_random)

        #-- preparing the labels for comparison 
        #-- selection of the labels for the first N initial points of X_random
        labels_random_initial = label_cluster_bic_random[:len(X)]

        #-- filter out noisy points for a purer stability comparison
        #  --- creating a mask to include only the points that were NOT noisy in ANY of the runs
        stable_mask = (label_cluster_bic != -1) & (labels_random_initial != -1)

        if np.sum(stable_mask) < 2:
                       
            # Caso 1: No hay suficientes puntos estables para un cálculo significativo.
            ari_global_results.append(np.nan)
            for k in n_clusters_original_bic:
                jaccard_individual_results[k].append(np.nan)
            continue # Ir a la siguiente iteración
    
            # Caso 2: Sí hay suficientes puntos estables (al menos 2).
    
        # A. Cálculo de la Estabilidad Global (ARI)
        ari = adjusted_rand_score(label_cluster_bic[stable_mask], 
                                  labels_random_initial[stable_mask])
        ari_global_results.append(ari)
    
        # B. Cálculo de la Estabilidad Individual (Jaccard)
        for k in n_clusters_original_bic:
            A_inicial = (label_cluster_bic == k)
            B_nueva = (labels_random_initial == k)
        
            A_stable_jaccard = A_inicial[stable_mask]
            B_stable_jaccard = B_nueva[stable_mask]

            # Lógica para evitar Jaccard con conjuntos vacíos, garantizando que siempre se añada un valor
            if np.sum(A_stable_jaccard) == 0 and np.sum(B_stable_jaccard) == 0:
                jaccard = 1.0 
            elif np.sum(A_stable_jaccard) == 0 or np.sum(B_stable_jaccard) == 0:
                jaccard = 0.0 
            else:
                jaccard = jaccard_score(A_stable_jaccard, B_stable_jaccard, pos_label=1)
            
            jaccard_individual_results[k].append(jaccard) # <-- ¡Esta línea siempre se ejecuta ahora!

    avg_n_clusters_random = np.mean(n_clusters_list_random)
    std_n_clusters_random = np.std(n_clusters_list_random)

    print(f"Number of groups adding {round(percent_add*100)}% of galaxies     : {avg_n_clusters_random:.0f}")
    print(f"Standar deviation of the numeber of groups  : {round(std_n_clusters_random):.0f}")
       
    if ari_global_results:
        mean_ari = np.mean(ari_global_results)
        std_ari = np.std(ari_global_results)
        print(f"mean ARI score adding {round(percent_add*100)}% of galaxies       : {mean_ari:.4f}")
        print(f"Standard deviation of the ARI score         : {std_ari:.4f}")
    print(" ")

    # --- Assesing individual stability 
    print("... assessing individual stability ...")
    print("        --- Jaccard index ---")
    results_df = pd.DataFrame(jaccard_individual_results)
    summary = results_df.agg(['mean', 'std']).T
    print(summary.to_string(float_format="%.4f"))
    summary.columns = ['mean_add', 'std_add']
    
    print(" ")

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # =============================================================== REMOVE RANDOM POINTS FROM THE SAMPLE ===============================================================

    print("... removing random points from the sample ...")
    #-- Remove points at random (Subsampling)
    #-- stabillity removing points was added an November 26, 2025

    sample_size = len(X)
    removal_percentage = percent_rem # -- removing 20% of the sample
    n_to_remove = int(sample_size * removal_percentage)

    n_clusters_list_subsample = []
    ari_global_results_subsample = []
    jaccard_individual_results_subsample = {k: [] for k in n_clusters_original_bic}

    for i in range(nsim):
        #-- Generate random points (the seed must be different in each iteration)
        np.random.seed(i) # use the loop index as the seed to ensure randomness
        indices_to_keep = np.random.choice(sample_size, 
                                         sample_size - n_to_remove, 
                                         replace=False)

        #-- Create the new subsampled sample
        X_subsample = X[indices_to_keep]

        #-- Performing the clustering algothims DBSCAN or HDBSCAN in random subsample
        if method == 'dbscan':
            cluster_subsample = DBSCAN(eps=galaxy_separation, min_samples=n_galaxies, metric=metric_distance, algorithm='ball_tree').fit(X_subsample)
        elif method == 'hdbscan':
            cluster_subsample = HDBSCAN(min_cluster_size=n_galaxies, metric=metric_distance, algorithm='ball_tree').fit(X_subsample)   

        #-- putting a label to each galaxy in the subsample. This label allows us to assign each galaxy to a substructure
        label_cluster_bic_subsample = cluster_subsample.labels_ # the label_dbscan_bic_subsample is the label of each odentified substructure in the new sample. This parameter is a numpy.ndarray

        #-- estimating number of clusters
        n_clusters_subsample = len(set(label_cluster_bic_subsample)) - (1 if -1 in label_cluster_bic_subsample else 0)
        n_clusters_list_subsample.append(n_clusters_subsample)

        #-- project the subsample labels onto the initial sample
        #-- create an array of labels the size of X, initialized to -1 (removed/noise)
        labels_projected = np.full(len(X), -1, dtype=int)

        #-- assign the new labels to the points that were retained
        labels_projected[indices_to_keep] = label_cluster_bic_subsample # indices_to_keep stores the positions of the points that survived

        #-- filter for a pure stability comparison
        #-- filter to compare only the points that were assigned to a cluster (not noise)
        #-- in the initial run and that survived the subsampling (are not -1 in the projection).
        stable_sub_mask = (label_cluster_bic != -1) & (labels_projected != -1)

        if np.sum(stable_sub_mask) < 2:
                       
        # Caso 1: No hay suficientes puntos estables para un cálculo significativo.
            ari_global_results_subsample.append(np.nan)
            for k in n_clusters_original_bic:
                jaccard_individual_results_subsample[k].append(np.nan)
            continue # Ir a la siguiente iteración
        
        # A. Cálculo de la Estabilidad Global (ARI)
        ari_subsample = adjusted_rand_score(labels_projected[stable_sub_mask], 
                                    labels_random_initial[stable_sub_mask])
        ari_global_results_subsample.append(ari_subsample)

        # B. Cálculo de la Estabilidad Individual (Jaccard)
        for k in n_clusters_original_bic:
            A_initial = (label_cluster_bic == k)
            B_projected = (labels_projected== k)
        
            A_stable_jaccard_sub = A_initial[stable_sub_mask]
            B_stable_jaccard_sub = B_projected[stable_sub_mask]

            # Lógica para evitar Jaccard con conjuntos vacíos, garantizando que siempre se añada un valor
            if np.sum(A_stable_jaccard_sub) == 0 and np.sum(B_stable_jaccard_sub) == 0:
                jaccard_subsample = 1.0 
            elif np.sum(A_stable_jaccard_sub) == 0 or np.sum(B_stable_jaccard_sub) == 0:
                jaccard_subsample = 0.0 
            else:
                jaccard_subsample = jaccard_score(A_stable_jaccard_sub, B_stable_jaccard_sub, pos_label=1)
            
            jaccard_individual_results_subsample[k].append(jaccard_subsample) # <-- ¡Esta línea siempre se ejecuta ahora!

    avg_n_clusters_subsample = np.mean(n_clusters_list_subsample)
    std_n_clusters_subsample = np.std(n_clusters_list_subsample)

    print(f"Number of groups removing {round(percent_rem*100)}% of galaxies   : {avg_n_clusters_subsample:.0f}")
    print(f"Standar deviation of the numeber of groups  : {round(std_n_clusters_subsample):.0f}")
        
    if ari_global_results_subsample:
        mean_ari_sub = np.mean(ari_global_results_subsample)
        std_ari_sub = np.std(ari_global_results_subsample)
        print(f"mean ARI score removing {round(percent_rem*100)}% of galaxies     : {mean_ari_sub:.4f}")
        print(f"Standard deviation of the ARI score         : {std_ari_sub:.4f}")
    print(" ")

    # --- Assesing individual stability 
    print("... assessing individual stability ...")
    print("        --- Jaccard index ---")
    results_df_sub = pd.DataFrame(jaccard_individual_results_subsample)
    summary_sub = results_df_sub.agg(['mean', 'std']).T
    print(summary_sub.to_string(float_format="%.4f"))
    summary_sub.columns = ['mean_sub', 'std_sub']

    print(" ")

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ===================================================================== BOOTSTRAP IMPLEMENTATION =====================================================================

    print("... Performing ", nsim, " Bootstrap iterations ...")
    print("...")

    N = len(X) # size of sample
    bootstrap_labels = [] # list to store labels found by bootstrap
    n_clusters_list = []  # list to store cluster numbers 
    ari_scores = []       # list to store the ARI's score 

    for i in range(nsim):
        #-- Sampling with replacement (Bootstrap)
        sample_indices = np.random.choice(N, size=N, replace=True)
        X_bootstrap = X[sample_indices]

        #-- Apply DBSCAN to the Bootstrap sample
        if method == 'dbscan':
            cluster_bootstrap = DBSCAN(eps=galaxy_separation, min_samples=n_galaxies, metric=metric_distance, algorithm='ball_tree').fit(X_bootstrap)
        elif method == 'hdbscan':
            cluster_bootstrap = HDBSCAN(min_cluster_size=n_galaxies, metric=metric_distance, algorithm='ball_tree').fit(X_bootstrap)  

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

    results_boot = pd.DataFrame({k: [v] for k, v in stability_results.items()})
    summary_boot = results_boot.agg(['mean']).T
    print(summary_boot.to_string(float_format="%.4f"))
    summary_boot.columns = ['mean_boot']


    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------

    print(" ")
    print("-- ending LAGASU stability--")

    join_summary = pd.concat([summary, summary_sub, summary_boot], axis=1)
    join_summary_final = join_summary.reset_index().rename(columns={'index': 'groups'})
    join_summary_final.to_csv(save_path, float_format="%.4f", index=False)

    print(" ")
    print(f"Stability results successfully saved in : {save_path}" )    #-- returning output quantity

    return

#####################################################################################################################################################################################
#####################################################################################################################################################################################

def lagasu_raw(id_galaxy, ra_galaxy, dec_galaxy, redshift_galaxy, range_cuts, galaxy_separation, n_galaxies, metric_distance, method):

    """ LAGASU_raw is a raw version of LAGASU developed
    to analyze the stability of LAGASU

    *LAGASU_raw does not change the group labels considering 
    the cluster parameters 
    
    The input of LAGASU_raw must be the same as LAGASU. However, 
    the user does not need the information about central parameters 
    of the cluster

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
    
    LAGASU_raw will give us two labels as output: 
    i) lagasu[4] that corresponds to the label 
    putting by GMM and varies between 0 to N, 
    ii) lagasu[5] that corresponds to the label 
    putting by DBSCAN after to run gmm and varies 
    between -1 to N, where -1 corresponds to noise
    and galaxies within a substructure have a label 
    between 0 to N
    
    """

    # -- estimating central redshift of the distribution
    good_z = np.where((redshift_galaxy != 99) & (redshift_galaxy > 0))[0]
    redshift_cluster = biweight_location(redshift_galaxy[good_z])
    
    #-- creating a boolean mask. It is an array with True/False and True corresponds to the old value
    anomalous_mask = (
        ((redshift_galaxy > -99.) & (redshift_galaxy <= 0.)) | 
        (redshift_galaxy == -99.) | 
        (redshift_galaxy == 99.)
    )

    if anomalous_mask.any():
        #-- appliyinh mask 
        redshift_galaxy[anomalous_mask] = redshift_cluster
    else:
        pass

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
        n_redshift_groups = np.where(labels_bic == ii)[0] 

        #-- selecting galaxies that are part of a signle cut in redshift
        ra = ra_galaxy[n_redshift_groups]
        dec = dec_galaxy[n_redshift_groups]

        #-- generate sample data
        #-- creating a transposed array with the poition of galaxies to be used as input in the DBSCAN implementation
        #-- metric implementation was added an June 17, 2024

        if metric_distance == 'euclidean':
            X = np.array([ra, dec]).T
        elif metric_distance == 'haversine': 
            ra_rad = np.radians(ra)
            dec_rad = np.radians(dec)
            X = np.array([ra_rad, dec_rad]).T
        else:
            #-- CORRECTION: handling unrecognized metrics
            #-- this correction was done on December 05, 2025
            X = np.array([ra, dec]).T
            metric_distance = 'euclidean'
            raise ValueError("Metric distance must be 'euclidean' or 'haversine' -- 'euclidean' metrics is used")
        
        #-- Performing the clustering algothims DBSCAN or HDBSCAN
        if method == 'dbscan':
            cluster = DBSCAN(eps=galaxy_separation, min_samples=n_galaxies, metric=metric_distance, algorithm='ball_tree').fit(X)
        elif method == 'hdbscan':
            cluster = HDBSCAN(min_cluster_size=n_galaxies, metric=metric_distance, algorithm='ball_tree').fit(X)   
        else:
            cluster = DBSCAN(eps=galaxy_separation, min_samples=n_galaxies, metric=metric_distance, algorithm='ball_tree').fit(X)
            #-- handling unrecognized method
            #-- this correction was done on December 05, 2025
            raise ValueError("Clustering method must be 'dbscan' or 'hdbscan' -- 'dbscan' method is used") 
        
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

    #-- building matrix with output quantities
    lagasu_parameters = np.array([id_substructures, ra_substructures, dec_substructures, redshift_substructures, gmm_substructures, labels_dbscan_corr], dtype=object)

    #-- returning output quantity
    return lagasu_parameters

#####################################################################################################################################################################################
#####################################################################################################################################################################################

def lagasu_bootstrap_stability_analysis(id_galaxy, ra_galaxy, dec_galaxy, redshift_galaxy, range_cuts, galaxy_separation, n_galaxies, metric_distance, method, n_bootstrap):
        
    """
    Function that performs N bootstrap iterations to 
    analyze the stability of each group found in a sample
    using lagasu_raw.

    This function combines global stability analysis and 
    individual stability analysis 

    This funcion was develop by D. E. Olave-Rojas
    and check by Gemini (12/03/2025)

    The input is the same as of LAGASU_raw plus n_bootstrap

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
    :param n_bootstrap : number of simulation in boostrap resampling

    :type id_galaxy         : array
    :type ra_galaxy         : array
    :type dec_galaxy        : array
	:type redshift_galaxy   : array
    :type range_cuts        : int
    :type galaxy_separation : int, float 
    :type n_galaxies        : int, float
    :type metric_distance   : string  
    :type method            : string 
    :type n_bootstrap       : int 

	:returns: global and individual index to check 
        the stability of groups
   
    .. note::
        ** Global stability ** 
        mean_ari = lagasu_bootstrap_stability_analysis['ari']['mean']
        std_ari = lagasu_bootstrap_stability_analysis['std']['mean']

        ** Individual stability ** 
        mean_jaccard = lagasu_bootstrap_stability_analysis['jaccard_mean']
        std_jaccard = lagasu_bootstrap_stability_analysis['jaccard_std']
        n_galaxies = lagasu_bootstrap_stability_analysis['n_galaxies_original']

    """

    N = len(id_galaxy)
    ari_scores = []
    all_jaccard_scores = {}

    #-- runing lagasu_raw in the original sample
    print("... Runing lagasu_raw in the original data ...")
    result_original = lagasu_raw(id_galaxy, ra_galaxy, dec_galaxy, redshift_galaxy, range_cuts, galaxy_separation, n_galaxies, metric_distance, method)

    # -- selecting labels for each galaxy
    labels_original = result_original[5]

    #-- identifiying original groups excluding galaxies out of them
    unique_groups_original = [g for g in np.unique(labels_original) if g != -1]
    
    if not unique_groups_original:
        print(" WARNING: No groups were identified in the original sample")
        return {'ari': {'mean': 0.0, 'std': 0.0, 'scores': []}, 'individual': {}}

    original_group_indices = {
        g: np.where(labels_original == g)[0] for g in unique_groups_original
    }
    for g in unique_groups_original:
        all_jaccard_scores[g] = []

    print(f"Groups identified in the original sample: {unique_groups_original}")
    print(f"\n... Starting {n_bootstrap} Bootstrap iterations ...")
    print("...")
    
    #-- starting n bootstrap iterations
    for i in range(n_bootstrap):

        #-- Sampling with replacement
        indices_bootstrap = np.random.choice(N, size=N, replace=True)

        # Applying the resampling the indices to the data
        id_bs = id_galaxy[indices_bootstrap]
        ra_bs = ra_galaxy[indices_bootstrap]
        dec_bs = dec_galaxy[indices_bootstrap]
        redshift_bs = redshift_galaxy[indices_bootstrap]

        #-- runing lagasu_raw in the bootsrap sample
        result_bs = lagasu_raw(id_bs, ra_bs, dec_bs, redshift_bs, range_cuts, galaxy_separation, n_galaxies, metric_distance, method)
    
        # -- selecting labels for each galaxy
        labels_bs = result_bs[5]
        
        #-- estimating the global stability using ARI index
        # The Adjusted Rand Index (ARI) is used as a metric to compare groups labels obtained from the original dataset 
        # with those obtained from each bootstrap sample. The ARI measures the similarity between two groupings, ignoring permutations
        # a value of 1.0 for perfect agreement and 0.0 (or negative) for random agreement.
        
        labels_original_subset = labels_original[indices_bootstrap]
        ari = adjusted_rand_score(labels_original_subset, labels_bs)
        ari_scores.append(ari)

        #-- estimating the individual estability using the Jaccard Index
        # The Jaccard Index is ideal for measuring the similarity between two datasets.
        # For a group G from the original sample and a group H from a bootstrap sample, 
        # the Jaccard indez is defined as the ratio between the number of galaxies in both groups
        # and the number of galaxies in at least a group

        unique_groups_bs = [g for g in np.unique(labels_bs) if g != -1]

        for g_original in unique_groups_original:
            indices_original = original_group_indices[g_original]
            
            #-- selecting galaxies from the original group that were sampled
            galaxies_in_original_group_and_bootstrap = id_galaxy[indices_original]
            
            max_jaccard = 0.0
            
            for g_bs in unique_groups_bs:
                #-- selecting the id of the bootstrap group galaxy
                indices_bs_group = np.where(labels_bs == g_bs)[0]
                galaxies_in_bootstrap_group = id_bs[indices_bs_group]
                
                #-- estimating the Jaccard Index
                jaccard = utils.calculate_jaccard_index(
                    galaxies_in_original_group_and_bootstrap, 
                    galaxies_in_bootstrap_group
                )

                if jaccard > max_jaccard:
                    max_jaccard = jaccard
            
            all_jaccard_scores[g_original].append(max_jaccard)

    print("...")
    print(f"\n... Ending {n_bootstrap} Bootstrap iterations ...")

    #-- consolidation of Results
    
    #-- global Results (ARI)
    ari_mean = np.mean(ari_scores)
    ari_std = np.std(ari_scores)

    #-- individual Results (Jaccard Index)
    individual_results = {}
    for g, scores in all_jaccard_scores.items():
        if scores:
            individual_results[g] = {
                'jaccard_mean': np.mean(scores),
                'jaccard_std': np.std(scores),
                'n_galaxies_original': len(original_group_indices[g])
            }
        else:
            individual_results[g] = {'jaccard_mean': 0.0, 'jaccard_std': 0.0, 'n_galaxies_original': len(original_group_indices[g])}

    return {
        'ari': {'mean': ari_mean, 'std': ari_std, 'scores': ari_scores},
        'individual': individual_results
    }

#####################################################################################################################################################################################
#####################################################################################################################################################################################

def lagasu_subsampling_raw(id_galaxy, ra_galaxy, dec_galaxy, redshift_galaxy, range_cuts, galaxy_separation, n_galaxies, metric_distance, method):

    """ LAGASU_subsampling_raw is a raw version of LAGASU 
    developed to analyze the stability of LAGASU in a subsample

    This version of lagasu corrects possible internal errors 
    when handling empty or very small GMM cuts.

    *LAGASU_subsampling_raw does not change the group labels 
    considering the cluster parameters 
    
    The input of LAGASU_subsamoling_raw must be the same as LAGASU. 
    However, the user does not need the information about central 
    parameters of the cluster

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
    
    LAGASU_subsampling_raw will give us two labels 
    as output: i) lagasu[4] that corresponds to the 
    label putting by GMM and varies between 0 to N, 
    ii) lagasu[5] that corresponds to the label 
    putting by DBSCAN after to run gmm and varies 
    between -1 to N, where -1 corresponds to noise
    and galaxies within a substructure have a label 
    between 0 to N
    
    """

    # -- estimating central redshift of the distribution
    good_z = np.where((redshift_galaxy != 99) & (redshift_galaxy > 0))[0]
    redshift_cluster = biweight_location(redshift_galaxy[good_z])
    
    #-- creating a boolean mask. It is an array with True/False and True corresponds to the old value
    anomalous_mask = (
        ((redshift_galaxy > -99.) & (redshift_galaxy <= 0.)) | 
        (redshift_galaxy == -99.) | 
        (redshift_galaxy == 99.)
    )

    if anomalous_mask.any():
        #-- appliying mask to replace bad values
        redshift_galaxy[anomalous_mask] = redshift_cluster
    else:
        pass

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

            #--- GMM CORRECTION: Avoid fitting if there are not enough points
            #-- this correction was added on december, 05, 2025
            if len(candidate_galaxies) < n_components:
                continue 

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

    #-- defining the parameters of the best fit
    n_cuts_bic = clf.n_components # number of cuts do by the algorithms 

    #-- putting a label to each galaxy. This label allows us to separe and assign each galaxy to a redshift cut
    labels_bic = clf.predict(candidate_galaxies) # using this label we can separate galaxies into n groups that correspond to redshifts cuts

    #-- To assign galaxies to different substructures we use Density-Based Spatial Clustering of Applications with Noise (DBSCAN, Ester et al. 1996)
    #-- To identify groups using DBSCAN we must define a minimum number of neighbouring objects separated by a specific distance. 
    #-- DBSCAN does not assign all objects in the sample to one group (Ester et al. 1996) and we can remove the galaxies that are not spatially grouped with others

    #-- sorting the output labels given by GMM-BIC implementation 
    sorted_labels_bic = np.sort(labels_bic)

    #-- initialization of accumulation variables before the loop to ensure they exist
    p = np.array([], dtype=int)
    tam_groups = np.array([], dtype=int)
    labels_dbscan_bic = np.array([], dtype=int)
    first_element_groups = np.array([], dtype=int)

    is_first_valid_cut = True # flag para manejar la inicialización en el primer corte válido

    #--- START OF LOOP ---
    for ii in range(0,n_cuts_bic): # n_cuts_bic are the number of cuts given by the GMM implementation

        #-- defining a single redshift cut in which DBSCAN will be apply
        n_redshift_groups = np.where(labels_bic == ii)[0] 

        #-- CORRECTION: Handling empty or very small cuts
        #-- this correction was done on December 05, 2025
        if len(n_redshift_groups) < n_galaxies:
            if len(n_redshift_groups) == 0:
                # If the redshift slice is empty or very small, a -1 is assigned to all galaxies and continue to the next slice.
                continue # Skip this iteration if there are no galaxies in the slice
        
            # If the number of galaxies is less than the DBSCAN/HDBSCAN minimum,
            # all galaxies in this cutoff will be considered noise.
            label_cluster_bic = np.full(len(n_redshift_groups), -1, dtype=int)
            groups = np.array([-1], dtype=int) #-- setting only the noise group

        else:
            #-- selecting galaxies that are part of a signle cut in redshift
            ra = ra_galaxy[n_redshift_groups]
            dec = dec_galaxy[n_redshift_groups]
            id = id_galaxy[n_redshift_groups]

            #-- generate sample data
            #-- creating a transposed array with the poition of galaxies to be used as input in the DBSCAN implementation
            #-- metric implementation was added an June 17, 2024

            if metric_distance == 'euclidean':
                X = np.array([ra, dec]).T
            elif metric_distance == 'haversine': 
                ra_rad = np.radians(ra)
                dec_rad = np.radians(dec)
                X = np.array([ra_rad, dec_rad]).T
            else:
                #-- CORRECTION: handling unrecognized metrics
                #-- this correction was done on December 05, 2025
                X = np.array([ra, dec]).T
                raise ValueError("Metric distance must be 'euclidean' or 'haversine' -- 'euclidean' metrics is used")
        
            #-- Performing the clustering algothims DBSCAN or HDBSCAN
            if method == 'dbscan':
                cluster = DBSCAN(eps=galaxy_separation, min_samples=n_galaxies, metric=metric_distance, algorithm='ball_tree').fit(X)
            elif method == 'hdbscan':
                cluster = HDBSCAN(min_cluster_size=n_galaxies, metric=metric_distance, algorithm='ball_tree').fit(X)   
            else:
                cluster = DBSCAN(eps=galaxy_separation, min_samples=n_galaxies, metric=metric_distance, algorithm='ball_tree').fit(X)
                #-- handling unrecognized method
                #-- this correction was done on December 05, 2025
                raise ValueError("Clustering method must be 'dbscan' or 'hdbscan' -- 'dbscan' method is used")
        
            #-- putting a label to each galaxy. This label allows us to assign each galaxy to a substructure
            label_cluster_bic = cluster.labels_ # the label_dbscan_bic is the label of each odentified substructure. This parameter is a numpy.ndarray

            #-- selecting the labels of the groups found by DBSCAN in each redshift cut
            groups = np.unique(label_cluster_bic) # number of groups in each redshift cut

    # -- END OF LOOP --

    #-- CORRECCTION: Unified initialization of accumulation variables
    #-- this correction was done on December 05, 2025
    if is_first_valid_cut:
        p = groups # array with the unique labels that could be assign to a single galaxy
        tam_groups = np.array([len(groups)]) # size of the each idenfied substructure
        labels_dbscan_bic = label_cluster_bic # label used to identified each substructure
        first_element_groups = groups[[0]] if len(groups) > 0 else np.array([], dtype=int) # first element in the array with the labels of the substructures
        is_first_valid_cut = False # It has already been initialized
    else:
        p = np.append(p, groups)
        tam_groups = np.append(tam_groups, len(groups))
        labels_dbscan_bic = np.append(labels_dbscan_bic, label_cluster_bic)
        if len(groups) > 0:
            first_element_groups = np.append(first_element_groups, groups[0])

    #-- CORRECTION: Handling cases without valid cuts
    #-- this correction was done on December 05, 2025
    if is_first_valid_cut:
        #-- if there were no cuts with galaxies or not enough galaxies, the original GMM/BIC array is returned with the labels but no substructures.
        labels_dbscan_corr = np.full(len(redshift_galaxy), -1, dtype=int)
        gmm_substructures = labels_bic # using GMM tags

        # Instead of intermediate results, we use the complete original arrays
        id_substructures = id_galaxy
        ra_substructures = ra_galaxy
        dec_substructures = dec_galaxy
        redshift_substructures = redshift_galaxy

    else:
        #-- the rest of the code only executes if there was at least one valid GMM cut
        #-- Finally, we need to label the substructures from 0 to n
        #-- Initialization of variables
        groups_pos = []
        p_pos = 0

        #-- Position of the first element in each group is searched here 
        #-- This implementation allows us to consider the case of a little sample in which all galaxies are assign to a single substructure 
        if tam_groups.ndim == 0:
            tam_groups = np.array([tam_groups])

        #--- START OF LOOP ---
        for ii in range(0,len(tam_groups)):

            if (len(tam_groups)==1):

                groups_pos = 0

            else: 

                for e in range(0,tam_groups[ii]):

                    #-- only if p[p_pos+e] is equal to the first element of the current group
                    if (len(first_element_groups) > ii and p[p_pos+e] == first_element_groups[ii]):
                        if ii == 0:
                            groups_pos = 0
                        else:
                            if groups_pos == []:
                                groups_pos = np.array(p_pos + e)
                            else:
                                groups_pos = np.append(groups_pos, (p_pos + e))
        
                p_pos = p_pos + tam_groups[ii]
            # -- END OF LOOP --

        #-- Here the correlative is assembled eliminating -1
        p3 = np.array([], dtype=int) # p3 is an array with the label of all idenitified substructures

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
        if isinstance(groups_pos, int):
            groups_pos = np.array([groups_pos])
    
        #========================================================
        #-- This loop allows us to assign a label from 0 to n lo each substructures plus noise which is labelled with -1
        #--- START OF LOOP ---

        #-- selecting the labels of the groups found by DBSCAN 
        final_groups = np.unique(labels_dbscan_bic)

        if  len(final_groups) == 1 and final_groups[0] == -1: # in this case none of galaxies are part of a subhalo

            labels_dbscan_corr = labels_dbscan_bic 
    
        else: 
            labels_dbscan_corr = np.array([], dtype=int) # -- explicit Initialization

            for k in range(0,len(labels_dbscan_bic)):
                aux = 0

                # We look up the index of the current group (labels_dbscan_bic[k]) within the list of groups
                # (p) for the GMM slice to which the galaxy belongs (sorted_labels_bic[k]).
                # The relabeling logic is complex. We ensure that the groups_pos index is valid.
                # groups_pos[sorted_labels_bic[k]] is the position of the first element of group ii in p.
                # We verify that the indices are valid

                gmm_label_index = sorted_labels_bic[k]
                if gmm_label_index >= len(groups_pos) or gmm_label_index >= len(first_element_groups):
                    #-- if the index is out of range, we assign -1 (noise)
                    labels_dbscan_corr = np.append(labels_dbscan_corr, -1)
                    continue

                start_index = groups_pos[gmm_label_index]
                #-- finding the position of label_dbscan_bic[k] within p, starting from start_index
                try:
                    # Buscamos la posición del label_cluster_bic dentro de 'p'
                    current_group_label = labels_dbscan_bic[k]
                    segment_start = start_index
                    segment_end = start_index + tam_groups[gmm_label_index]

                    #-- loking for look for the local index of current_group_label in p[segment_start:segment_end]
                    # The np.where function returns a tuple of indices arrays; we take the first element (index 0) and the first value [0] if found.
                    local_index = np.where(p[segment_start:segment_end] == current_group_label)[0]

                    if len(local_index) > 0:
                        aux = local_index[0]
                        final_index = start_index + aux
                        
                        #-- assigning the re-labeled tag from p2
                        labels_dbscan_corr = np.append(labels_dbscan_corr, p2[final_index])
                    else:
                        # If it's not found or something went wrong, -1 is assigned.
                        labels_dbscan_corr = np.append(labels_dbscan_corr, -1)
                        
                except Exception:
                    #-- in case of any error, assign -1 and continue
                    labels_dbscan_corr = np.append(labels_dbscan_corr, -1)

        # -- END OF LOOP --

    #-- CORRECTION: Refactorización de la construcción del output 
    #-- this correction was done on December 05, 2025

    if not is_first_valid_cut:
        # If there were valid cuts, we reordered the original data according to the order of the GMM/BIC cuts
        sort_indices = np.argsort(labels_bic)

        id_substructures = id_galaxy[sort_indices]
        ra_substructures = ra_galaxy[sort_indices]
        dec_substructures = dec_galaxy[sort_indices]
        redshift_substructures = redshift_galaxy[sort_indices]
        gmm_substructures = labels_bic[sort_indices]

    #-- building matrix with output quantities
    lagasu_parameters = np.array([id_substructures, ra_substructures, dec_substructures, redshift_substructures, gmm_substructures, labels_dbscan_corr], dtype=object)

    #-- returning output quantity
    return lagasu_parameters

#####################################################################################################################################################################################
#####################################################################################################################################################################################

def lagasu_subsampling_stability_analysis(id_galaxy, ra_galaxy, dec_galaxy, redshift_galaxy, range_cuts, galaxy_separation, n_galaxies, metric_distance, method, n_subsamples, subsample_fraction):

    """
    
    Function that performs subsampling stability analysis 
    for global (ARI) and individual (Jaccard Index) stability.

    This funcion was develop by D. E. Olave-Rojas
    and check by Gemini (12/03/2025)

    The input is the same as of LAGASU_subsampling_raw plus 
    n_subsamples and subsample_fractions
        
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
    :param n_subsamples: Number of randomly created subsamples in 
        the resampling
    :param subsample_fraction: Fraction of the original data to be 
        removed from the original sample (e.g., 0.1 for 10%).

    :type id_galaxy             : array
    :type ra_galaxy             : array
    :type dec_galaxy            : array
	:type redshift_galaxy       : array
    :type range_cuts            : int
    :type galaxy_separation     : int, float 
    :type n_galaxies            : int, float
    :type metric_distance       : string  
    :type method                : string 
    :type n_subsamples          : int 
    :type subsample_fraction    : float

	:returns: global and individual index to check 
        the stability of groups
   
    .. note::
        ** Global stability ** 
        mean_ari = lagasu_subsampling_stability_analysis['ari']['mean']
        std_ari = lagasu_subsampling_stability_analysis['std']['mean']

        ** Individual stability ** 
        mean_jaccard = lagasu_subsampling_stability_analysis['jaccard_mean']
        std_jaccard = lagasu_subsampling_stability_analysis['jaccard_std']
        n_galaxies = lagasu_subsampling_stability_analysis['n_galaxies_original']

    """

    N = len(id_galaxy)    
    ari_scores = []
    all_jaccard_scores = {}

    #-- runing lagasu_raw in the original sample
    print("... Runing lagasu_subsampling_raw in the original data ...")
    result_original = lagasu_subsampling_raw(id_galaxy, ra_galaxy, dec_galaxy, redshift_galaxy, range_cuts, galaxy_separation, n_galaxies, metric_distance, method)    
    
    # -- selecting labels for each galaxy
    labels_original = result_original[5]

    #-- selecting the sorted id_galaxy
    id_original_sorted = result_original[0] 

    #-- creating a mapping of galaxy ID to its original tag for quick reference
    id_to_label_original = dict(zip(id_original_sorted, labels_original))
    
    #-- identifiying original groups excluding galaxies out of them
    unique_groups_original = [g for g in np.unique(labels_original) if g != -1]

    if not unique_groups_original:
        print(" WARNING: No groups were identified in the original sample")
        return {'ari': {'mean': 0.0, 'std': 0.0, 'scores': []}, 'individual': {}}

    original_group_indices = {
        g: np.where(labels_original == g)[0] for g in unique_groups_original
    }
    for g in unique_groups_original:
        all_jaccard_scores[g] = []

    print(f"Groups identified in the original sample: {unique_groups_original}")

    #-- setting the subsample to rerain
    retain_fraction = 1 - subsample_fraction
    
    #-- subsampling without replacement and smaller size
    sample_size = int(N * retain_fraction)

    print(f"\n... Starting {n_subsamples} iterations ...")
    print("...")

    #-- starting n_subsamples iterations
    for i in range(n_subsamples):

        # Sampling without replacement
        indices_subsample = np.random.choice(N, size=sample_size, replace=False)

        #-- apply the indices to the data
        id_ss = id_galaxy[indices_subsample]
        ra_ss = ra_galaxy[indices_subsample]
        dec_ss = dec_galaxy[indices_subsample]
        redshift_ss = redshift_galaxy[indices_subsample]

        #-- running lagasu_subsampling_raw on the bootstrap sample
        result_ss = lagasu_subsampling_raw(id_ss, ra_ss, dec_ss, redshift_ss, range_cuts, galaxy_separation, n_galaxies, metric_distance, method) 

        labels_ss_raw = result_ss[5] # DBSCAN/HDBSCAN labels, already ordered by the GMM/BIC of the subsample      
        id_ss_sorted = result_ss[0] # IDs of the galaxies that have valid labels in labels_ss_raw

        
        #-- recreating the original labels (labels_original_subset) for only the subsample (id_ss) in the order of the subsample
        labels_original_subset_for_comparison = np.array([
            id_to_label_original.get(galaxy_id, -1) for galaxy_id in id_ss
        ])
        
        #-- recreating the subsample labels (labels_ss) in the order of the subsample (id_ss)
        #-- First, map ID to label_ss
        id_to_label_ss = dict(zip(id_ss_sorted, labels_ss_raw))
        
        #-- Next, create the subsample's label array, assigning -1 if the ID does not appear in the lagasu_raw results (i.e., it was excluded internally)
        labels_ss_for_comparison = np.array([
            id_to_label_ss.get(galaxy_id, -1) for galaxy_id in id_ss
        ])

        #-- Global Stability (ARI)
        labels_original_subset = labels_original_subset_for_comparison
        labels_ss = labels_ss_for_comparison
        
        ari = adjusted_rand_score(labels_original_subset, labels_ss) 
        ari_scores.append(ari)

        #-- Individual Stability (Jaccard Index) 
        #-- selecting labels_ss_raw (labels sorted by GMM/BIC)
        unique_groups_ss = [g for g in np.unique(labels_ss_raw) if g != -1] 
        
        for g_original in unique_groups_original:
            indices_original = original_group_indices[g_original]
            
            #-- selecting the IDs that are in the original group and in the subsample
            is_in_subsample = np.in1d(id_galaxy[indices_original], id_ss)
            galaxies_in_original_group_and_subsample = id_galaxy[indices_original][is_in_subsample]
            
            max_jaccard = 0.0
            
            for g_ss in unique_groups_ss:
                #-- selectting galaxy IDs of the subsample group
                indices_ss_group = np.where(labels_ss == g_ss)[0]
                galaxies_in_subsample_group = id_ss[indices_ss_group]
                
                #-- estimating the Jaccard Index
                jaccard = utils.calculate_jaccard_index(
                    galaxies_in_original_group_and_subsample, 
                    galaxies_in_subsample_group
                )

                if jaccard > max_jaccard:
                    max_jaccard = jaccard
            
            all_jaccard_scores[g_original].append(max_jaccard)

    print("...")
    print(f"\n... Ending {n_subsamples} iterations ...")

    #-- consolidating and formatting the results
    ari_mean = np.mean(ari_scores)
    ari_std = np.std(ari_scores)

    #-- setting individual results (Jaccard Index)
    individual_results = {}
    for g, scores in all_jaccard_scores.items():
        if scores:
            individual_results[g] = {
                'jaccard_mean': np.mean(scores),
                'jaccard_std': np.std(scores),
                'n_galaxies_original': len(original_group_indices[g])
            }
        else:
            individual_results[g] = {'jaccard_mean': 0.0, 'jaccard_std': 0.0, 'n_galaxies_original': len(original_group_indices[g])}

    return {
        'ari': {'mean': ari_mean, 'std': ari_std, 'scores': ari_scores},
        'individual': individual_results
    }

#####################################################################################################################################################################################
#####################################################################################################################################################################################

def lagasu_noise_stability_analysis(id_galaxy, ra_galaxy, dec_galaxy, redshift_galaxy, range_cuts, galaxy_separation, n_galaxies, metric_distance, method, n_iterations, noise_fraction):

    """
    
    Function that evaluates the stability of LAGASU_raw 
    by adding random noise points.

    This funcion was develop by D. E. Olave-Rojas
    and check by Gemini (12/05/2025)

    The input is the same as of LAGASU__raw plus n_iterations 
    and noise_fractions

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
    :param n_siterations: Number of iterations (noise tests)
    :param noise_fraction: Fraction of noise points to add 
        (e.g., 0.1 means adding 10% of No. original data)

    :type id_galaxy             : array
    :type ra_galaxy             : array
    :type dec_galaxy            : array
	:type redshift_galaxy       : array
    :type range_cuts            : int
    :type galaxy_separation     : int, float 
    :type n_galaxies            : int, float
    :type metric_distance       : string  
    :type method                : string 
    :type n_iterations          : int 
    :type noise_fraction        : float

	:returns: global and individual index to check 
        the stability of groups
   
    .. note::
        ** Global stability ** 
        mean_ari = lagasu_subsampling_stability_analysis['ari']['mean']
        std_ari = lagasu_subsampling_stability_analysis['std']['mean']

        ** Individual stability ** 
        mean_jaccard = lagasu_subsampling_stability_analysis['jaccard_mean']
        std_jaccard = lagasu_subsampling_stability_analysis['jaccard_std']
        n_galaxies = lagasu_subsampling_stability_analysis['n_galaxies_original']

    """

    N_original = len(id_galaxy)
    n_noise_points = int(N_original * noise_fraction)
    ari_scores = []
    all_jaccard_scores = {}

    #-- runing lagasu_raw in the original sample
    print("... Runing lagasu_subsampling_raw in the original data ...")
    result_original = lagasu_raw(id_galaxy, ra_galaxy, dec_galaxy, redshift_galaxy, range_cuts, galaxy_separation, n_galaxies, metric_distance, method)    

    labels_original = result_original[5]

    #-- identifiying original groups excluding galaxies out of them
    unique_groups_original = [g for g in np.unique(labels_original) if g != -1]

    if not unique_groups_original:
        print(" WARNING: No groups were identified in the original sample")
        return {'ari': {'mean': 0.0, 'std': 0.0, 'scores': []}, 'individual': {}}

    original_group_indices = {
        g: np.where(labels_original == g)[0] for g in unique_groups_original
    }
    for g in unique_groups_original:
        all_jaccard_scores[g] = []

    print(f"Groups identified in the original sample: {unique_groups_original}")
    print(f"\n... Starting {n_iterations} iterations ...")
    print("...")
    #-- starting the noise stability loop
    for i in range(n_iterations):
        #-- generating noise in a range similar to that of the original data.
        id_noise = np.arange(N_original, N_original + n_noise_points)
        ra_noise = np.random.uniform(ra_galaxy.min(), ra_galaxy.max(), n_noise_points)
        dec_noise = np.random.uniform(dec_galaxy.min(), dec_galaxy.max(), n_noise_points)
        redshift_noise = np.random.uniform(redshift_galaxy.min(), redshift_galaxy.max(), n_noise_points)
        
        #-- creating the preserved sample by concatenating the original data and the noise
        id_perturbed = np.concatenate([id_galaxy, id_noise])
        ra_perturbed = np.concatenate([ra_galaxy, ra_noise])
        dec_perturbed = np.concatenate([dec_galaxy, dec_noise])
        redshift_perturbed = np.concatenate([redshift_galaxy, redshift_noise])

        #-- running lagasu_raw on the perturbed sample
        result_perturbed = lagasu_raw(
            id_perturbed, ra_perturbed, dec_perturbed, redshift_perturbed, range_cuts, galaxy_separation, n_galaxies, metric_distance, method)
        labels_perturbed = result_perturbed[5] # The perturbed labels are of size N_original + n_noise_points.
        
        #-- selecting only the labels of the original points.
        labels_perturbed_subset = labels_perturbed[:N_original]
        
        #-- Global Stability (ARI)
        #-- comparing the original labels with the labels obtained by the algorithm for the original subset of the perturbed sample.
        ari = adjusted_rand_score(labels_original, labels_perturbed_subset)
        ari_scores.append(ari)

        #-- separating labels and IDs of the groups found in the perturbed sample (original + noise)
        unique_groups_perturbed = [g for g in np.unique(labels_perturbed) if g != -1]

        for g_original in unique_groups_original:
            indices_original = original_group_indices[g_original]
            
            #-- Original group galaxy IDs
            galaxies_in_original_group = id_galaxy[indices_original]
            
            max_jaccard = 0.0
            
            #-- iterating over all groups (including noise) found in the perturbed sample
            for g_perturbed in unique_groups_perturbed:
                #-- indices of the g_perturbed group in the *perturbed sample*
                indices_perturbed_group = np.where(labels_perturbed == g_perturbed)[0]
                galaxies_in_perturbed_group = id_perturbed[indices_perturbed_group]
                
                #-- estimating Jaccard Index
                jaccard = utils.calculate_jaccard_index(
                    galaxies_in_original_group, 
                    galaxies_in_perturbed_group
                )

                if jaccard > max_jaccard:
                    max_jaccard = jaccard
                    
            all_jaccard_scores[g_original].append(max_jaccard)

    print("...")
    print(f"\n... Ending {n_iterations} iterations ...")

    #-- consolidating and formatting the results
    ari_mean = np.mean(ari_scores)
    ari_std = np.std(ari_scores)

    #-- setting individual results (Jaccard Index)
    individual_results = {}
    for g, scores in all_jaccard_scores.items():
        if scores:
            individual_results[g] = {
                'jaccard_mean': np.mean(scores),
                'jaccard_std': np.std(scores),
                'n_galaxies_original': len(original_group_indices[g])
            }
        else:
            individual_results[g] = {'jaccard_mean': 0.0, 'jaccard_std': 0.0, 'n_galaxies_original': len(original_group_indices[g])}

    return {
        'ari': {'mean': ari_mean, 'std': ari_std, 'scores': ari_scores},
        'individual': individual_results
    }


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
            cluster = HDBSCAN(min_cluster_size=n_galaxies, metric=metric_distance, algorithm='ball_tree').fit(X)   
    
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

#####################################################################################################################################################################################
#####################################################################################################################################################################################
