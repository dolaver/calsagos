#!/usr/bin/env python
# (D. Olave- Rojas 05/07/2021) 
# ISOMER: Identifier of SpectrOscopic MembERs (ISOMER) is a Python script that allows to identify the spectroscopic cluster members 

#--import python modules
import numpy as np

#--import CALSAGOS's modules 
from calsagos import redshift_boundaries
from calsagos import cluster_kinematics
from calsagos import utils

__author__ = 'Pierluigi Cerulo & Daniela Olave-Rojas'
__email__ = 'pcerulo@inf.udec.cl - daniela.olave@utalca.cl '

VERSION = '1.0' 

#####################################################################################################################################################################################
#####################################################################################################################################################################################

def isomer(id_galaxy, ra_galaxy, dec_galaxy, z_spec_galaxy, cluster_mass, cluster_initial_redshift, input_H, input_Omega_L, input_Omega_m):

    """ ISOMER is a function that selects spectroscopic
    members in a single cluster 

    This funcion was develop by P. Cerulo (08/06/2016)

    isomer(id_galaxy, ra_galaxy, dec_galaxy, z_spec_galaxy, 
    cluster_mass, cluster_initial_redshift, input_H, 
    input_Omega_L, input_Omega_m)

    :param id_galaxy: ID of each galaxy in the field of
        view of the cluster
	:param ra_galaxy and dec_galaxy: are the R.A. and Dec., 
        respecively and represent the position of each galaxies 
        in degree units
    :param z_spec_galaxy: spectroscopic redshift of each galaxy 
        in the field of view of the cluster
    :param cluster_mass: mass of the cluster within r200 in 
        M_sun units
	:param cluster_initial_redshift: central redshift of the 
        cluster
    :param input_H: Hubble constant 
    :param input_Omega_L: Omega Lambda cosmology
    :param input_Omega_m: Omega matter cosmology

    :type id_galaxy: array
    :type ra_galaxy: array
    :type dec_galaxy: array
	:type redshift_galaxy: array
    :type cluster_mass: int, float
    :type cluster_initial_redshift int, float
	:type input_H: int, float
    :type intput_Omega_m: int, float
    :type intput_Omega_L: int, float

	:returns: array with the cluster members
	:rtype: array

	"""
    #-- re-defining the array of redshift as a numpy array
    z_spec = np.array(z_spec_galaxy)

    r200 = utils.calc_radius_finn(cluster_mass, cluster_initial_redshift, input_H, input_Omega_L, input_Omega_m, "mts")

    #-- calculating the escape velocity of the cluster
    #-- The input to estimate the escape velocity of the cluster is the mass of the cluster, redshift and cosmology
    #-- The mas must be in solar units
    cluster_escape_velocity = cluster_kinematics.calc_escape_velocity(cluster_mass, r200)
    print("cluster_escape_velocity [km/s] =", cluster_escape_velocity)

    #-- deriving redshift and the limits of redshift distribution
    z_boundaries = redshift_boundaries.calc_redshift_boundaries(z_spec, cluster_escape_velocity, cluster_initial_redshift)

    #-- setting upper and lower bound of redshift distribution
    cluster_z_low = z_boundaries[2]
    cluster_z_high = z_boundaries[3]

    #-- setting the cluster redshift of the sample 
    cluster_redshift = z_boundaries[0]
    print("new_cluster redshift: ", cluster_redshift)

    #-- setting the velocity dispersion of the sample
    velocity_dispersion = z_boundaries[1]
    print("sigma", velocity_dispersion)

    #-- calculating uncertainty on velocity dispersion
    n_iter = 100
    sigma_uncertainty = cluster_kinematics.calc_cluster_velocity_dispersion_error(z_spec, cluster_escape_velocity, cluster_initial_redshift, n_iter)
    print("delta_sigma = ", sigma_uncertainty)
    
    #-- Defining the criteria to select cluster members
    #-- A cluster member is a galaxy with a peculiar velocity lower than the escape velocity of the cluster
    spectroscopic_members = np.where( (z_spec > cluster_z_low) & (z_spec < cluster_z_high) )[0]

    #-- selecting cluster members
    ID_members = id_galaxy[spectroscopic_members]
    RA_members = ra_galaxy[spectroscopic_members]
    DEC_members = dec_galaxy[spectroscopic_members]
    redshift_members = z_spec[spectroscopic_members]

    #-- print the size of the cluster sample
    print('members =', len(RA_members))

    #-- building matrix with output quantities
    cluster_array = np.array([ID_members, RA_members, DEC_members, redshift_members, cluster_redshift, velocity_dispersion, sigma_uncertainty], dtype=object)

    #-- returning output quantity
    return cluster_array

#####################################################################################################################################################################################
##################################################################################################################################################################################### 

