# Filename: redshift_boundaries.py
# Here we found a serie of scripts develop to estimate the
# central redshift and velocity dispersion of the cluster
# and to establish the limits in the redshift distribution
import numpy as np

# Section dedicated to importing the modules from SubFind
from . import cluster_kinematics

# define speed of light in km/s
c = 299792.458

#####################################################################################################################################################################################
#####################################################################################################################################################################################

# FUNCTION THAT ESTIMATES THE UPPER AND LOWER BOUNNDS OF THE REDSHIFT DISTRIBUTION OF CLUSTER MEMBERS (3 x velocity dispersion)
# (Pierluigi Cerulo 27/03/2016)

def calc_redshift_boundaries(redshift, escape_velocity, starting_redshift):

    """ redshift_boundaries = calsagos.redshift_boundaries.calc_redshift_boundaries(redshift, escape_velocity, starting_redshift)

    Function that estimates the central redshift, the
    velocity dispersion and the upper and lower bounds 
    of the redshift distribution.
    
    The limits of the redshift distribution are estimated
    using a 3sigma clipping implementation 

    This function was develop by P. Cerulo (27/03/2016)

    :param redshift: redshift of the cluster members
    :param escape_velocity: escape velocity of the cluster 
        in km s-1 units
    :param starting_redshift: preliminary estimation of 
        the redshift of the cluster

	:type redshift: array
	:type escape_velocity: int, float
	:type starting_redshift: int, float

	:returns: the redshift of the cluster and the upper
        and lower bound of the redshift distribution
	:rtype: numpy array

    .. note::

    calc_redshift_boundaries(redshift, escape_velocity, starting_redshift)[0] corresponds to the 
        redshift of the cluster
    calc_redshift_boundaries(redshift, escape_velocity, starting_redshift)[1] corresponds to the
        velocity dispersion of the cluster
    calc_redshift_boundaries(redshift, escape_velocity, starting_redshift)[2] corresponds to the
        lower limit of the redshift
    calc_redshift_boundaries(redshift, escape_velocity, starting_redshift)[3] corresponds to the
        upper limit of the redshift

	"""

    #-- estimate the redshift and cluster velocity dispersion
    sigma_estimate = cluster_kinematics.calc_cluster_velocity_dispersion(redshift, escape_velocity, starting_redshift)

    new_cluster_redshift = sigma_estimate[0]
    sigma = sigma_estimate[1]

    #-- deriving upper and lower bounds of redshift distribution of cluster members
    redshift_upper_bound = new_cluster_redshift + 3*(sigma/c)
    redshift_lower_bound = new_cluster_redshift - 3*(sigma/c)

    #-- returning output quantity
    return np.array([new_cluster_redshift, sigma, redshift_lower_bound, redshift_upper_bound])

#####################################################################################################################################################################################
#####################################################################################################################################################################################





