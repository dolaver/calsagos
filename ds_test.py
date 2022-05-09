# Collection of scripts developed to implent the 
# Dressler-Schectamn Test (Dressler & Shectman, 1988)
# in a cluster of galaxies

# Section dedicated to import python modules
import numpy as np
from astropy.stats import biweight_scale

# Section dedicated to importing the modules from CALSAGOS
from . import utils
from . import cluster_kinematics


__author__ = 'D. Olave-Rojas & P. Cerulo'
__email__ = 'daniela.olave@utalca.cl - pcerulo@inf.udec.cl'
VERSION = '0.1'

#####################################################################################################################################################################################
#####################################################################################################################################################################################

def calc_delta_DS(RA, DEC, peculiar_velocity, redshift_member, escape_velocity, cluster_starting_redshift):

    """ calsagos.dressler_schectman_test.calc_delta_DS(RA, DEC, peculiar_velocity, redshift_member, escape_velocity, cluster_starting_redshift)

	Function that computes the delta_i value 
    defined by the Dressler-Schectman Test 
    (see Dressler & Shectman, 1988)

    This function was developed by D. Olave-Rojas
    and P. Cerulo (04/07/2016)

	:param RA: Right Ascension of the galaxies
        in degree units
    :param DEC: Declination of the galaxies 
        in degree units
    :param peculiar_velocity: peculiar velocity
        of the galaxies in km s-1 units
    :param redshift_members: redshift of the
        galaxies in the cluster
    :param escape_velocity: escape velocity of
        the cluster
	:param cluster_starting_redshift: preliminar
        estimation of the redshift of the cluster

    :type RA: array
    :type DEC: array
    :type peculiar_velocity: array
    :type redshift_member: array
	:type escape_velocity: int, float
	:type cluster_starting_redshift: int, float

	:returns: delta_i for each galaxy, critical_value
        threshold_value
	:rtype: numpy array

    .. note::

    The DS-Test verifies the existence of regions 
    kinematically distinct from the main galaxy 
    cluster
    
    The larger the delta_i value, the greater the 
    probability that the galaxy belongs to a 
    substructure.

    The output of this module is:
    
    calc_delta_DS(RA, DEC, peculiar_velocity, redshift_member, escape_velocity, cluster_starting_redshift)[0] corresponds to the 
        delta_i value for each galaxy by Dressler & Shectman, (1988)
    calc_delta_DS(RA, DEC, peculiar_velocity, redshift_member, escape_velocity, cluster_starting_redshift)[1] corresponds to the
        critical value definen by Dressler & Shectman, (1988)
    calc_delta_DS(RA, DEC, peculiar_velocity, redshift_member, escape_velocity, cluster_starting_redshift)[2] corresponds to the
        threshold_value by Dressler & Shectman, (1988)

	"""

    print("starting estimation of delta_i from DS-test")

    #-- estimating the velocity dispersion of the cluster
    sigma_estimate = cluster_kinematics.calc_cluster_velocity_dispersion(redshift_member, escape_velocity, cluster_starting_redshift)
    sigma_cluster = sigma_estimate[1]

    #-- estimating the mean velocity of the cluster
    mean_cluster_velocity = utils.calc_mean_and_standard_error(peculiar_velocity)
    cluster_velocity = mean_cluster_velocity[0]

    #-- defining output quantities
    dim = redshift_member.size # number of elements in the input arrays
    d = np.zeros(dim) # distance between galaxies
    sigma_group = np.zeros(dim) # velocity dispersion of the groups
    delta = np.zeros(dim) # delta_i value to each galaxy

    for ii in range(dim):
        
        for jj in range(dim):
            
            #-- estimating the distance between galaxies in the sample
            d[jj] = np.sqrt((RA[ii] - RA[jj])**2. + (DEC[ii] - DEC[jj])**2.)
                        
            #-- sorting the distance between galaxies with the aim to select the nearest neighbors
            sorted_indices = np.argsort(d)
		           
            #-- synchronise the peculiar velocity array with the sorted distance
            # (P. Cerulo 05/07/2016)
            v_of_d_sorted = peculiar_velocity[sorted_indices]
          
            #-- select the velocity of the ten nearest neighbours in the synchronised peculiar velocity array
            # (P. Cerulo 05/07/2016)
            v_nearest_ten = v_of_d_sorted[0:10]
            
            #-- calc of the mean velocity of the 10-nearest neighbors
            local_velocity_nearest_ten = utils.calc_mean_and_standard_error(v_nearest_ten)
            local_velocity = local_velocity_nearest_ten[0]

            #-- calc of the velocity dispersion of the 10-nearest neighbors
            sigma_local = np.std(v_nearest_ten)
            sigma_group[ii] = np.std(v_nearest_ten)
	    
            #-- function that estimates the delta from Dressler & Shectman (1988)
            delta[ii] = np.sqrt((11./(sigma_cluster**2.))*((local_velocity - cluster_velocity)**2. + (sigma_local - sigma_cluster)**2.))

    print("ending estimation of delta_i from DS-test")

    # -- END OF LOOP --
    delta_obs = delta

    #-- calculating critical value and threshold value
    critical_value = np.sum(delta)
    threshold_value = critical_value/dim

    #-- building matrix with output quantities
    delta_DS_array = np.array([delta_obs, critical_value, threshold_value], dtype=object)

    #-- returning output quantity
    return delta_DS_array

#####################################################################################################################################################################################
#####################################################################################################################################################################################

def calc_delta_shuffle(RA, DEC, peculiar_velocity, redshift_member, escape_velocity, cluster_starting_redshift, n_bootstrap):

    """ calsagos.dressler_schectman_test.calc_delta_DS(RA, DEC, peculiar_velocity, redshift_member, escape_velocity, cluster_starting_redshift)

	Function that randomly shuffles the 
    observed velocities and reassigns 
    these values to the member position
    
    This function was developed by P. Cerulo (31/08/2016)

	:param RA: Right Ascension of the galaxies
        in degree units
    :param DEC: Declination of the galaxies 
        in degree units
    :param peculiar_velocity: peculiar velocity
        of the galaxies in km s-1 units
    :param redshift_members: redshift of the
        galaxies in the cluster
    :param escape_velocity: escape velocity of
        the cluster
	:param cluster_starting_redshift: preliminar
        estimarion of the redshift of the cluster
    :param n_bootstrap: number of of bootstrap 
        iterations

    :type RA: array
    :type DEC: array
    :type peculiar_velocity: array
    :type redshift_member: array
	:type escape_velocity: int, float
	:type cluster_starting_redshift: int, float
    :type n_bootstrap: int

	:returns: simulated critical_value
        and threshold_value
	:rtype: numpy array

    .. note::

    calc_delta_shuffle(RA, DEC, peculiar_velocity, redshift_member, escape_velocity, cluster_starting_redshift, n_bootstrap)[0] corresponds to the
        simulated critical value definen by Dressler & Shectman, (1988)
    calc_delta_shuffle(RA, DEC, peculiar_velocity, redshift_member, escape_velocity, cluster_starting_redshift, n_bootstrap)[1] corresponds to the
        simulated threshold_value by Dressler & Shectman, (1988)

	"""

    #-- number of elements in the input arrays
    dim = redshift_member.size

    #-- defining arrays with output quantities and quantities useful for calculations
    critical_value_sim = np.zeros(n_bootstrap)
    threshold_value_sim = np.zeros(n_bootstrap)
    
    #-- running bootstrap simulations
    for ii in range(n_bootstrap):
        
        # -- select random indices within redshift array
        R = np.random.randint(0, dim, size=dim)
        
        redshift_member_sim = redshift_member[R]
        peculiar_velocity_sim = peculiar_velocity[R]
               
        #-- delta_DS_array_bootstrap = calc_delta_DS(RA, DEC, peculiar_velocity_sim, redshift_member_sim)
        delta_DS_array_bootstrap = calc_delta_DS(RA, DEC, peculiar_velocity_sim, redshift_member_sim, escape_velocity, cluster_starting_redshift)
        critical_value_sim[ii] = delta_DS_array_bootstrap[1]
        threshold_value_sim[ii] = delta_DS_array_bootstrap[2]

    #-- building matrix with output quantities
    delta_shuffle_matrix = np.array([critical_value_sim, threshold_value_sim])

    #-- returning output quantity
    return delta_shuffle_matrix

#####################################################################################################################################################################################
#####################################################################################################################################################################################

# FUNCTION THAT ESTIMATES THE P-VALUE BY COMPARING THE VALUE OF OBSERVED DELTA_DS TO VALUE OF SHUFFLED DELTA_DS 

def probability_value(cumulative_delta, cumulative_delta_shuffle, n_bootstrap):

    """ calsagos.dressler_schectman_test.probability_value(cumulative_delta, cumulative_delta_shuffle, n_bootstrap)

	Function that estimates the P-value
    by comparing the value of observed
    delta_i to value of shuffled delta_i
    
    This function was developed by P. Cerulo (31/08/2016)

	:param cumulative_delta: sum of the all
        observed delta_i in the cluster
    :param cumulative_delta_shuffle: sum of 
        the all shuffled delta_i in the cluster
    :param n_bootstrap: number of bootstrap 
        iterations

    :type cumulative_delta: float
    :type cumulative_delta_shuffle: float
    :type n: int

	:returns: sum of delta_i and delta probability
	:rtype: numpy array

    .. note::

    probability_value(cumulative_delta, cumulative_delta_shuffle, n_bootstrap)[0] corresponds to the
        sum of the delta_i used to estimate the probability of a cluster host substructures. This 
        value is defined by Dressler & Shectman (1988)
    probability_value(cumulative_delta, cumulative_delta_shuffle, n_bootstrap)[1] orresponds to the
        the probability of a cluster host substructures. This value is defined by Dressler & Shectman(1988)

	"""

    #-- defining the criteria to select galaxies that will be used to estimate the probability
    good_values = np.where(cumulative_delta_shuffle >= cumulative_delta)[0]
    
    #-- selecting values to estimate the probability
    delta_good_valued = cumulative_delta_shuffle[good_values]

    #-- estimating the sum of the delta_i values
    sum_delta = np.sum(delta_good_valued)

    #-- estimating the P-value defined in Dressler & Schectman (1988)
    delta_prob = sum_delta/n_bootstrap

    #-- building matrix with output quantities
    delta_probability = np.array([sum_delta, delta_prob])

    #-- returning output quantity
    return delta_probability

#####################################################################################################################################################################################
#####################################################################################################################################################################################

# FUNCTION THAT DETERMINES IF A GALAXY HAS OR NOT A PROBABILITY TO BE PART OF A SUBSTRUCTURE
def calc_delta_outliers(delta_i):

    """ calsagos.ds_test.calc_delta_outliers(delta_i)

	Function that determines if a galaxy
    has or not a probability to be part 
    of a substructure

    This function was developed by D. Olave-Rojas (01/09/2016)
    following Girardi et al. (1996), the galaxies with
    a delta_i >= delta_lim are considered possible members 
    of a substructures

	:param delta_i: delta_i value to each galaxy,
        which was defined following Dressler &
        Schectman (1988)

    :type delta_i: array

	:returns: label that indicates if a galaxy
        has or not a high probability to be 
        part of a substructure
	:rtype: array

    .. note::

    The output of this module is a label that indicates if 
    a galaxy has a value of delta_i that corresponds to 
    an outlier or not in the delta_i distribution

    In this case delta_lim = 3.*sigma_delta, where sigma_delta
    is the dispersion of the delta_i distribution

    sigma_delta is estimated using biweight_scale 
    (Beers et al. 1990) function from astropy version 4.3.1
    
    label = 1: the galaxy is an outlier and it is probable 
        that is hosting in a substructure 
    label = 0: the galaxy is a standard galaxy that is not 
        part of a substructure

	"""

    #-- Determination of the standard deviation of the delta_i distribution
    sigma_delta = biweight_scale(delta_i)

    #-- number of elements in the input arrays
    member_size = len(delta_i)
    label = np.zeros(member_size)
    
    #-- The output parameter label indicates if a galaxy has a value of delta_i that corresponds to an outlier or not in the delta_i distribution
    #-- label = 1: the galaxy is an outlier and it is probable that is hosting in a substructure and label = 0: the galaxy is a standard galaxy that is not part of a substructure

    for ii in range(member_size):
        if (delta_i[ii] <= -3.*sigma_delta) or (delta_i[ii] >= 3.*sigma_delta):

            label[ii] = 1
        
        elif (delta_i[ii] > -3*sigma_delta) and (delta_i[ii] < 3*sigma_delta):

            label[ii] = 0

    #-- returning output quantity
    return np.array(label)

#####################################################################################################################################################################################
#####################################################################################################################################################################################
