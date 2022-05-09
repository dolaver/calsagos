# Filename: cluster_kinematics.py
# Here we found a serie of scripts develop to estimate
# the kinemactic properties of the cluster

#import python modules
import numpy as np
from astropy.stats import biweight_scale
from astropy.stats import biweight_location
import astropy.units as u
from astropy import constants as const

# Section dedicated to importing the modules from CALSAGOS
from . import utils

# define speed of light in km/s
c = 299792.458

#####################################################################################################################################################################################
#####################################################################################################################################################################################

def calc_escape_velocity(cluster_mass, cluster_radius):

    """ escape_velocity = cluster_kinematics.calc_escape_velocity(cluster_mass, cluster_radius)

	Function that estimates the escape velocity of a galaxy cluster
    using as input the m200 of the cluster

    This funcion was develop by D. Olave-Rojas (21/06/2016)

    The velocity escape is defined as v_e = sqrt(2GM/r)

	:param cluster_mass: cluster mass defined as m_200. This 
        parameter must be in M_sun
	:param cluster_radius: cluster radius defined as m_200.
        this parameter must be in meteres

	:type cluster_mass: float
	:type cluster_radius: float

	:returns: The escape velocity of the cluster 
	:rtype: int, float

    .. note::
	The returned velocity is in km/s

	:Example:
    >>> import calsagos
    >>> calsagos.cluster_kinematics.calc_escape_velocity(1.279e15, 6.014e22)
    2375.8793080034325

	"""

    # -- defining the gravitational constant
    grav_const = const.G
    g = grav_const.value # in Kg m-3 s-2

    # -- defining the solar mass
    solar_mass = const.M_sun
    ms = solar_mass.value

    # -- converting the mass of the clusters in solar units to mass in kilograms
    mass_kg =  cluster_mass * ms # cluster mass in kg units

    # -- estimate of escape velocity
    esc_vel_mks = np.sqrt(2.* g * mass_kg * (cluster_radius**(-1))) #escape velocity is in m s-1
    escape_velocity = esc_vel_mks/1000. # escape velocity is in km s-1
   
    # -- return output
    return escape_velocity

#####################################################################################################################################################################################
#####################################################################################################################################################################################

def calc_escape_velocity_diaferio(cluster_mass, cluster_radius):

    """ escape_velocity = cluster_kinematics.calc_escape_velocity_diaferio(cluster_mass, cluster_radius)

	Function that estimates the escape velocity of a galaxy cluster
    using as input the m200 of the cluster

    The escape velocity, estimated for a cluster,
    using m200 and r200, is computed as Diaferio (1999)

    This funcion was develop by D. Olave-Rojas (21/06/2016)

	:param cluster_mass: cluster mass defined as m_200. This 
        parameter must be in M_sun
	:param cluster_radius: cluster radius defined as m_200.
        this parameter must be in Mpc

	:type cluster_mass: float
	:type cluster_radius: float

	:returns: The escape velocity of the cluster 
	:rtype: int, float

    .. note::
	The returned velocity is in km/s

	:Example:
    >>> from calsagos import cluster_kinematics
    >>> cluster_kinematics.calc_escape_velocity_diaferio(1.279e15, 1.949)
    2374.7018294642708

	"""

    # -- renaming constants
    K_1 = 92.7e-6
    
    # -- estimate of escape velocity
    escape_velocity =  K_1 * np.sqrt( cluster_mass/cluster_radius)  # escape velocity is in km s-1
   
    # -- return output
    return escape_velocity

#####################################################################################################################################################################################
#####################################################################################################################################################################################

def calc_peculiar_velocity(redshift_array, cluster_redshift):

    """ calsagos.velocity_dispersion.calc_peculiar_velocity(redshift_array, cluster_redshift)

    This function was developed by P. Cerulo (28/11/2015)

	Function that estimates peculiar velocities from
    redshift (as in Harrison 1974)

	:param redshift_array: array with redshift of
        galaxies in the region of a cluster
    :param cluster_redshift: central redshift of
        a galaxy cluster

	:type redshift_array: array
    :type cluster_redshift: int, float

    :returns: peculiar velocity of galaxies
	:rtype: array

    .. note::
	The returned velocity is in km/s 


	"""

    # -- define output quantities
    dim = redshift_array.size

    peculiar_velocity = np.zeros(dim)
  
    # -- compute peculiar velocity
    for ii in range(dim):

        if redshift_array[ii] <= 0.0:

            peculiar_velocity[ii] = -99.9

        elif redshift_array[ii] > 0.0:
            
            peculiar_velocity[ii] = c * ((redshift_array[ii] - cluster_redshift)) / (1.0 + cluster_redshift)

    # -- END OF LOOP

    # -- return cluster velocity dispersion
    return peculiar_velocity

#####################################################################################################################################################################################
#####################################################################################################################################################################################

def calc_cluster_velocity_dispersion(input_redshift_array, escape_velocity, starting_redshift):

    """ cluster_kinematics.calc_cluster_velocity_dispersion(input_redshift_array, escape_velocity, starting_redshift)

    This function was developed by P. Cerulo (28/11/2015)
    following Yahil & Vidal (1977)

    Funcion that estimates the velocity dispersion
    in a sample free of contaminants

    We recomend the user use this function
    when cluster members are selected by 
    using ISOMER

	:param input_redshift_array: array with redshift 
        of spectroscopic members of a cluster
    :param escape_velocity: escape velocity of 
        the cluster
    :param starting_redshift: central redshift 
        of the galaxy cluster

	:type input_redshift_array: array
    :type escape_velocity: int, float
    :type starting_redshift: int, float

    :returns: central redshift of the
        cluster and velocity dispersion
    :rtype: numpy array

    .. note::
	The returned velocity dispersion has km/s units

    calc_peculiar_velocity(redshift_array, cluster_redshift)[0] corresponds to the 
        cluster redshift
    calc_peculiar_velocity(redshift_array, cluster_redshift)[1] corresponds to the
        velocity dispersion of the cluster

        
	"""
   
    #-- removing all bad values in redshift array
    good_values = np.where( input_redshift_array > 0.0 )[0]
    redshift_array = input_redshift_array[good_values]

    #-- removing galaxies at more than 4000 km/s from the cluster initial redshift
    starting_peculiar_velocity = calc_peculiar_velocity(redshift_array, starting_redshift)
    starting_cluster_sample = np.where( (starting_peculiar_velocity > - escape_velocity) & (starting_peculiar_velocity < escape_velocity) )[0]

    #-- defining new set of cluster redshift including only galaxies with -v_esc < v_pec < +v_esc km/s
    cluster_redshift_array = redshift_array[starting_cluster_sample]

    #-- estimating cluster redshift for the new cluster sample
    cluster_redshift = biweight_location(cluster_redshift_array)

    
    while True:
        # -- estimating peculiar velocity and velocity dispersion for cleaned cluster sample
        peculiar_velocity = calc_peculiar_velocity(cluster_redshift_array, cluster_redshift)
        sigma = biweight_scale(peculiar_velocity)

        # -- removing all galaxies at more than 3 x sigma from the cluster redshift
        cluster_sample = np.where( (peculiar_velocity > -3*sigma) & (peculiar_velocity < 3*sigma) )[0]
        outliers = np.where( (peculiar_velocity <= -3*sigma) | (peculiar_velocity >= 3*sigma) )[0]

        # -- re-defining sample of redshifts and computing cluster redshift
        cluster_redshift_array = cluster_redshift_array[cluster_sample]
        cluster_redshift = biweight_location(cluster_redshift_array)

        # -- until there are no longer outliers
        if outliers.size == 0:
            break

    # -- return output array
    return np.array([cluster_redshift, sigma])

#####################################################################################################################################################################################
#####################################################################################################################################################################################

def calc_cluster_velocity_dispersion_error(input_redshift_array, escape_velocity, starting_redshift, n_bootstrap):
    
    """ cluster_kinematics.calc_cluster_velocity_dispersion_error(input_redshift_array, escape_velocity, starting_redshift, n_bootstrap)

    This function was developed by P. Cerulo (10/12/2015)

    Funcion that estimates the uncertainty on
    cluster velocity dispersion using a 
    boostrap technique. 
    
    We recomend the user use this function
    when cluster members are selected by 
    using ISOMER

	:param input_redshift_array: array with redshift 
        of spectroscopic members of a cluster
    :param escape_velocity: escape velocity of 
        the cluster
    :param starting_redshift: central redshift 
        of the galaxy cluster
    :param n_boostrap: number of bootstrap 
        simulations

	:type input_redshift_array: array
    :type escape_velocity: int, float
    :type starting_redshift: int, float
    :type n_bootstrap: int

    :returns: uncertainty on the cluster 
        velocity dispersion
    :rtype: array

    .. note::

	The returned uncertainty on the velocity 
        dispersion is in km/s
        
	"""
    #-- removing all bad values in redshift array
    good_values = np.where( input_redshift_array > 0.0 )[0]
    redshift_array = input_redshift_array[good_values]

    dim = redshift_array.size

    #-- defining arrays with output quantities and quantities useful for calculations
    bootstrap_sigma = np.zeros(n_bootstrap)


    print("starting boostrap estimation of uncertainty on velocity dispersion")

    for ii in range(n_bootstrap):

        # -- select random indices within redshift array

        R = np.random.randint(0, dim, size=dim)
        
        redshift_array_sim = redshift_array[R]

        # -- estimate velocity dispersion for bootstrap sample
        bootstrap_sigma[ii] = calc_cluster_velocity_dispersion(redshift_array_sim, escape_velocity, starting_redshift)[1]

    # -- computing symmetric width of the 68% confidence interval of the bootstrap distribution of velocity dispersion
    delta_sigma = utils.calc_result(bootstrap_sigma, 'symmetric')[1]

    # -- return output array
    return delta_sigma

#####################################################################################################################################################################################
#####################################################################################################################################################################################

def calc_clumberi_cluster_velocity_dispersion_error(input_redshift_array, starting_redshift, n_bootstrap):
    
    """ cluster_kinematics.calc_clumberi_cluster_velocity_dispersion_error(input_redshift_array, starting_redshift, n_bootstrap)

    This function was developed by D. Olave-Rojas (03/07/2021)

    Funcion that estimates the uncertainty on
    cluster velocity dispersion using a 
    boostrap technique
    
    We recomend the user use this function
    when cluster members are selected by 
    using CLUMBERI

	:param input_redshift_array: array with redshift 
        of spectroscopic members of a cluster
    :param escape_velocity: escape velocity of 
        the cluster
    :param starting_redshift: central redshift 
        of the galaxy cluster
    :param n_boostrap: number of bootstrap 
        simulations

	:type input_redshift_array: array
    :type escape_velocity: int, float
    :type starting_redshift: int, float
    :type n_bootstrap: int

    :returns: uncertainty on the cluster 
        velocity dispersion
    :rtype: array

    .. note::

	The returned uncertainty on the velocity 
        dispersion is in km/s
        
	"""
    #-- removing all bad values in redshift array
    good_values = np.where( input_redshift_array > 0.0 )[0]
    redshift_array = input_redshift_array[good_values]

    dim = redshift_array.size

    #-- defining arrays with output quantities and quantities useful for calculations
    bootstrap_sigma = np.zeros(n_bootstrap)
    peculiar_velocity  = np.zeros(n_bootstrap)

    print("starting boostrap estimation of uncertainty on velocity dispersion")

    for ii in range(n_bootstrap):

        # -- select random indices within redshift array

        R = np.random.randint(0, dim, size=dim)
        
        redshift_array_sim = redshift_array[R]

        # -- estimate velocity dispersion for bootstrap sample
        peculiar_velocity = calc_peculiar_velocity(redshift_array_sim, starting_redshift)
        bootstrap_sigma[ii] = biweight_scale(peculiar_velocity)

    # -- computing symmetric width of the 68% confidence interval of the bootstrap distribution of velocity dispersion
    delta_sigma = utils.calc_result(bootstrap_sigma, 'symmetric')[1]

    # -- return output array
    return delta_sigma

#####################################################################################################################################################################################
#####################################################################################################################################################################################





