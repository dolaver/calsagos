#!/usr/bin/env python
# version: 1.0 (D. Olave- Rojas 05/07/2021)
# LAGASU: LAbeller of GAlaxies within SUbstructures is a python scripts 
# that assign galaxies to different substructures in and around a galaxy 
# cluster based on their density
# LAGASU uses the Gaussian Mixture Module (GMM) and Density-Based Spatial 
# Clustering of Application with Noise (DBSCAN), both availables from 
# python, to assign galaxies to different substructures found in and 
# around galaxy clusters
# 

#-- Import preexisting python modules
import numpy as np
from sklearn import mixture
from sklearn.cluster import DBSCAN

__author__ = 'daniela.olave@utalca.cl (Daniela Olave-Rojas)'

VERSION = '0.1' 

#####################################################################################################################################################################################
#####################################################################################################################################################################################

def lagasu(id_galaxy, ra_galaxy, dec_galaxy, redshift_galaxy, range_cuts, galaxy_separation, n_galaxies):

    """ LAGASU is a function that assigns galaxies to different 
    susbtructures in and around a galaxy cluster

    This function was developed by D. Olave-Rojas (05/07/2021)

    The input of LAGASU can be a sample of galaxies in a cluster 
    of a sample of galaxies previously selected as potential 
    members of a substructure in and around a single galaxy 
    cluster. The selection of potential members of substructures
    can be done by using the Dressler-Schectamn Test or DS-Test
    (Dressler & Schectman 1988) 

    lagasu(id_galaxy, ra_galaxy, dec_galaxy, redshift_galaxy, 
    range_cuts, galaxy_separation, n_galaxies)

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

    :type ra_galaxy: array
    :type dec_galaxy: array
	:type redshift_galaxy: array
    :type range_cuts: int
    :type galaxy_separation: int, float 
    :type n_galaxies: int, float

	:returns: label to each galaxy,
        which corresponds to a number 
        of substructure or noise
	:rtype: array

	"""
    print("-- starting LAGASU --")

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
#    bars = []
    
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
        #-- creating a transposed array with the poition of galaxies to be used as input in the DBSCAM implementation
        X = np.array([ra, dec]).T

        #-- Performing the DBSCAN
        db = DBSCAN(eps=galaxy_separation, min_samples=n_galaxies).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
     
        #-- putting a label to each galaxy. This label allows us to assign each galaxy to a substructure
        label_dbscan_bic = db.labels_ # the label_dbscan_bic is the label of each odentified substructure. This parameter is a numpy.ndarray

        #-- selecting the labels of the groups found by DBSCAN in each redshift cut
        groups = np.unique(label_dbscan_bic) # number of groups in each redshift cut

        if ii != 0:
            p = np.append(p,groups, axis=0)
            tam_groups = np.append(tam_groups, len(groups))
            labels_dbscan_bic = np.append(labels_dbscan_bic,label_dbscan_bic)
            first_element_groups = np.append(first_element_groups, groups[0])

        #-- process of the first iteration: defining variables
        else:                
            p = groups # array with the unique labels that could be assign to a single galaxy
            tam_groups = len(groups) # size of the each idenfied substructure
            labels_dbscan_bic = label_dbscan_bic # label used to identified each substructure
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
    print("number of substructures =", len(p3))

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

        #print len(groups) 
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


def lagasu_dbscan(id_galaxy, ra_galaxy, dec_galaxy, redshift_galaxy, galaxy_separation, n_galaxies):

    """ LAGASU_DBSCAN is a function that assigns galaxies to 
    different susbtructures in and around a galaxy cluster
    only using DBSCAN implementation

    This function was developed by D. Olave-Rojas (20/08/2021)

    The input of LAGASU can be a sample of galaxies in a cluster 
    of a sample of galaxies previously selected as potential 
    members of a substructure in and around a single galaxy 
    cluster. The selection of potential members of substructures
    can be done by using the Dressler-Schectamn Test or DS-Test
    (Dressler & Schectman 1988) 

    lagasu(id_galaxy, ra_galaxy, dec_galaxy, redshift_galaxy, 
    galaxy_separation, n_galaxies)

	:param ra_galaxy: is the Right Ascension of each galaxy in
        the sample. This parameter must be in degree units
    :param dec_galaxy: is the Declination of each galaxy in
        the sample. This parameter must be in degree units
    :param redshift_galaxy: redshift of each galaxy in the 
        cluster
    :param galaxy_separation: physical separation between 
        galaxies in a substructure the units must be the 
        same as the ra_galaxy and dec_galaxy
    :param n_galaxies: minimum number of galaxies to define a group

    :type ra_galaxy: array
    :type dec_galaxy: array
	:type redshift_galaxy: array
    :type galaxy_separation: int, float 
    :type n_galaxies: int, float

	:returns: label to each galaxy,
        which corresponds to a number 
        of substructure or noise
	:rtype: array

	"""
    print("-- starting LAGASU_DBSCAN --")

    #-- selecting galaxies that are part of a signle cut in redshift
    ra = ra_galaxy
    dec = dec_galaxy

    #-- defining the size of the group in redshift
    size_redshift_groups = len(ra)
    print("size redshift group: ",size_redshift_groups)

    #-- generate sample data
    #-- creating a transposed array with the poition of galaxies to be used as input in the DBSCAM implementation
    X = np.array([ra, dec]).T
        
    #-- Performing the DBSCAN
    db = DBSCAN(eps=galaxy_separation, min_samples=n_galaxies).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    #-- putting a label to each galaxy. This label allows us to assign each galaxy to a substructure
    label_dbscan_bic = db.labels_ # the label_dbscan_bic is the label of each odentified substructure. This parameter is a numpy.ndarray

    #-- printing the labels of the galaxies as the output in the DBSCAN implementation
    print("label dbscan: ", label_dbscan_bic) # if label == -1 the galaxy is noise. If galaxy is != -1 the galaxy is in a substructure

    id_substructures = id_galaxy
    ra_substructures = ra_galaxy
    dec_substructures = dec_galaxy
    zspec_substructures = redshift_galaxy
    label_substructures = label_dbscan_bic


    #-- building matrix with output quantities
    lagasu_parameters = np.array([id_substructures, ra_substructures, dec_substructures, zspec_substructures, label_substructures], dtype=object)

    #-- returning output quantity
    return lagasu_parameters

#####################################################################################################################################################################################
#####################################################################################################################################################################################