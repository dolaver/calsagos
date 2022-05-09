#!/usr/bin/env python
# (D. Olave- Rojas 03/08/2021) 
# CLUMBERI: CLUster MemBER Identifier is a Python script that allows to identify cluster members using Gaussian Mixture Modules (GMM)

# Section dedicated to import python modules
import numpy as np
from astropy.stats import biweight_scale
from astropy.stats import biweight_location
from sklearn import mixture

# Section dedicated to importing the modules from CALSAGOS
from calsagos import cluster_kinematics
from calsagos import utils

# define speed of light in km/s
c = 299792.458

__author__ = 'Pierluigi Cerulo & Daniela Olave-Rojas'
__email__ = 'pcerulo@inf.udec.cl - daniela.olave@utalca.cl '

VERSION = '0.1' 

#####################################################################################################################################################################################
#####################################################################################################################################################################################

def clumberi(id_galaxy, ra_galaxy, dec_galaxy, redshift_galaxy, cluster_initial_redshift, ra_cluster, dec_cluster, range_cuts):
    
    """ CLUMBERI (CLUster MemBER Identifier) is a 
    function that selects members in a single cluster 
    using Gaussian Mixture Modules (GMM) in 3-dimension

    This function was developed by D. Olave-Rojas (03/08/2021)

    CLUMBERI uses as input the position and redshift of the
    galaxies and the central position and redshift of the
    clsuter

    clumberi(id_galaxy, ra_galaxy, dec_galaxy, redshift_galaxy, 
    cluster_initial_redshift, ra_cluster, dec_cluster, range_cuts)

    :param id_galaxy: ID of each galaxy in the field of
        view of the cluster
	:param ra_galaxyRight Ascension (R.A.) of each galaxies
        in degree units
    :param dec_galaxy: Declination (Dec.) of each galaxies
        in degree units
    :param redshift_galaxy: redshift of each galaxy in the 
        field of view of the cluster
	:param cluster_initial_redshift: central redshift of the 
        cluster
    :param ra_cluster: Right Ascension (R.A.) of the cluster
        in degree units
    :param dec_cluster: Declination (Dec.) of the cluster in
        degree units
    :param range_cuts: to perform the cuts in the redshift 
        space is neccesary to give a range that allows find the 
        better number of gaussian to fit, where "range_cuts" is 
        the end of this range which is defined as range(1, range_cuts).
        Therefore, range_cuts must be greater than 1

    :type id_galaxy: array
    :type ra_galaxy: array
    :type dec_galaxy: array
	:type redshift_galaxy: array
    :type cluster_initial_redshift int, float
    :type ra_cluster float
    :type dec_cluster float
    :type range_cuts: int

	:returns: array with the cluster members, redshift, velocity
        dispersion and sigma of the cluster
	:rtype: array

    .. note::

    The output of this module is:
    clumberi(id_galaxy, ra_galaxy, dec_galaxy, redshift_galaxy, 
        cluster_initial_redshift, ra_cluster, dec_cluster, 
        range_cuts)[0]: corresponds to the ID of the cluster 
        members

    clumberi(id_galaxy, ra_galaxy, dec_galaxy, redshift_galaxy, 
        cluster_initial_redshift, ra_cluster, dec_cluster, 
        range_cuts)[1]: corresponds to the R.A. of the cluster 
        members
    
    clumberi(id_galaxy, ra_galaxy, dec_galaxy, redshift_galaxy, 
        cluster_initial_redshift, ra_cluster, dec_cluster, 
        range_cuts)[2]: corresponds to the Dec of the cluster 
        members

    clumberi(id_galaxy, ra_galaxy, dec_galaxy, redshift_galaxy, 
        cluster_initial_redshift, ra_cluster, dec_cluster, 
        range_cuts)[3]: corresponds to the redshift of the 
        cluster members

    clumberi(id_galaxy, ra_galaxy, dec_galaxy, redshift_galaxy, 
        cluster_initial_redshift, ra_cluster, dec_cluster, 
        range_cuts)[4]: corresponds to the redshift of the 
        cluster estimated in the sample of members

    clumberi(id_galaxy, ra_galaxy, dec_galaxy, redshift_galaxy, 
        cluster_initial_redshift, ra_cluster, dec_cluster, 
        range_cuts)[5]: corresponds to the velocity dispersion
        of the cluster estimated in the sample of members

    clumberi(id_galaxy, ra_galaxy, dec_galaxy, redshift_galaxy, 
        cluster_initial_redshift, ra_cluster, dec_cluster, 
        range_cuts)[6]: corresponds to the uncertainty of the 
        velocity dispersion of the cluster estimated in the 
        sample of members

	"""

    print("..starting CLUMBERI..")
    
    #-- creating a transposed array with the redshift_galaxy to be used as input in the GMM implementation
    candidate_galaxies = np.array([ra_galaxy, dec_galaxy, redshift_galaxy]).T #-- This is a 3D-array, which considers the position and redshifts of the galaxies

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
            gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type, tol=1e-2, reg_covar=1e-10, max_iter=1000, n_init=10, random_state=6) #new

            #-- Fit the Gaussian over the data 
            gmm.fit(candidate_galaxies)

            bic.append(gmm.bic(candidate_galaxies))

            #-- selecting the best fit
            if bic[-1] < lowest_bic:

                lowest_bic = bic[-1]

                best_gmm = gmm

    # -- END OF LOOP --

    #-- OTPUT OF THE LOOP
    bic = np.array(bic)

    #-- best fit of the sample
    clf = best_gmm
    
    #-- defining the parameters of the best fit
    n_cuts_bic = clf.n_components # number of cuts do by the algorithms 
    cov_type = clf.covariance_type # best covariance type
    means_bic = clf.means_ # center of each cut in our case represent the central redshift of the cut

    #-- selecting the central values for each gaussian fitted in the whole sample 
    ra_bic = means_bic[:,0]
    dec_bic = means_bic[:,1]
    redshift_bic = means_bic[:,2]

    #-- putting a label to each galaxy. This label allows us to separe and assign each galaxy to a redshift cut
    labels_bic = clf.predict(candidate_galaxies) # using this label we can separate galaxies into n groups that correspond to redshifts cuts

    #-- estimating the euclidean angular distance for each identified gaussian with respect to the central position of the cluster
    angular_distance = utils.calc_angular_distance(ra_bic, dec_bic, ra_cluster, dec_cluster, "degrees")

    #-- estimating the distance between the redshift of each identified Gaussian with respect to the central redshift of the cluster
    distance_redshift = abs(redshift_bic - cluster_initial_redshift)

    #-- selecting the minimum angular distance
    min_angular_distance = min(angular_distance)

    #-- selecting the minimum redshift distance
    min_redshift = min(distance_redshift)

    #-- selecting the closest Gaussian, in redshift and position to the center of the cluster
    dim = angular_distance.size
    
    for ii in range(dim):
        if (angular_distance[ii] == min_angular_distance) and (distance_redshift[ii] == min_redshift):
            
            good_sample = np.where( (angular_distance == min(angular_distance)) & (distance_redshift == min(distance_redshift)) )[0]

        else: 
            good_sample = np.where( (angular_distance == min(angular_distance)) )[0]

    #-- selecting the labels of the multiple Gaussians found by GMM 
    unique_labels = np.unique(labels_bic)

    #-- selecting the label of the closest Gaussian
    label_good = unique_labels[good_sample]
#    print("label good", label_good)

    #-- selecting all galaxies within the closest Gaussian to the center of the cluster
    final_sample = np.where(labels_bic == label_good)[0]

    #-- selecting the redshift of the galaxies within the closest Gaussian to the center of the cluster
    redshift_initial = redshift_galaxy[final_sample]

    #-- estimating the dispersion of the redshift distribution of the closest Gaussian to the center of the cluster
    sigma = biweight_scale(redshift_initial)

    #-- estimating the central redshift of the redshift distribution of the closest Gaussian to the center of the cluster
    redshift = biweight_location(redshift_initial)

    # -- removing all galaxies at more than 3 x sigma from the cluster redshift distribution of galaxies 
    cluster_sample = np.where( (redshift_galaxy > (redshift -3*sigma)) & (redshift_galaxy < + (redshift + 3*sigma)) )[0]

    #-- re-defining sample of cluster members
    id_member = id_galaxy[cluster_sample]
    ra_member = ra_galaxy[cluster_sample]
    dec_member = dec_galaxy[cluster_sample]
    redshift_member = redshift_galaxy[cluster_sample]

    #--computing the cluster redshift
    new_cluster_redshift = biweight_location(redshift_member)

    #-- print the size of the cluster members
    print("cluster members:", len(redshift_member))

    print("sigma members:", round(len(redshift_member)/100.))

    #-- print the central redshift of the cluster estimated using galaxy members selected with GMM
    print("new_cluster_redshift:", new_cluster_redshift)

    #-- calculating the peculiar velocity of the galaxies
    peculiar_velocity = cluster_kinematics.calc_peculiar_velocity(redshift_member, new_cluster_redshift)

    #-- estimating the velocity dispersion of the sample of cluster members selected with GMM
    velocity_dispersion = biweight_scale(peculiar_velocity)
    print("velocity dispersion =", velocity_dispersion, "[km/s]")

    #-- calculating uncertainty on velocity dispersion
    n_iter = 100
    sigma_uncertainty = cluster_kinematics.calc_clumberi_cluster_velocity_dispersion_error(redshift_member, new_cluster_redshift, n_iter)
    print("delta_sigma = ", sigma_uncertainty)

    #-- building matrix with output quantities
    cluster_array = np.array([id_member, ra_member, dec_member, redshift_member, new_cluster_redshift, velocity_dispersion, sigma_uncertainty], dtype=object)

    #-- returning output quantity
    return cluster_array

#####################################################################################################################################################################################
#####################################################################################################################################################################################
