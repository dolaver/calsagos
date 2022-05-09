# Filename: utils.py
# Here we found a serie of utilitaries functions which can be used
# to select cluster members, identify substructures or estimate
# some interesting physical properties of galaxy clusters

from scipy import stats
import numpy as np
import math
from kneebow.rotor import Rotor
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from astropy import constants as const

__author__ = 'D. Olave-Rojas & P. Cerulo'
__email__ = 'daniela.olave@utalca.cl - pcerulo@inf.udec.cl'
VERSION = '0.1'

#####################################################################################################################################################################################
#####################################################################################################################################################################################

def calc_mean_and_standard_error(input_array):

    """ error = calsagos.utils.calc_mean_and_standard_error(value_array, flag)

    This function was developed by P. Cerulo (07/02/2015)

	Function that computes the mean and standard error of a 
    numpy array 

	:param input_array: array with different measurements 
    of a physical quantity which is used to compute the 
    mean value and the standar error of this physical 
    quantity

	:type value_array: array

    :returns: The mean value of the physical quantity 
    and its standard error
	:rtype: numpy array

    .. note::

	The returned variable is a numpy array. The first
    element corresponds to the mean value of the 
    physical quantity and the second element corresponds
    to the error of the mean value

    :Example:
    >>> from calsagos import *
    >>> import numpy
    >>> a = numpy.array([1.5, 1.0, 1.4, 1.6, 1.5])
    >>> utils.calc_mean_and_standard_error(a)
    array([1.4       , 0.09380832])
    >>> 
    >>> mean = utils.calc_mean_and_standard_error(a)[0]
    >>> print("mean:", mean)
    mean: 1.4
    >>> error = utils.calc_mean_and_standard_error(a)[1]
    >>> print("error:", error)
    error: 0.0938083151964686

        
	"""
   
    # -- find size of array
    n = input_array.size

    # -- compute mean
    array_mean = np.mean(input_array)

    # -- compute standard error
    array_delta = np.std(input_array) / math.sqrt(n)

    # -- create output array
    output_quantity = np.array([array_mean, array_delta])

    # -- return output
    return output_quantity

#####################################################################################################################################################################################
#####################################################################################################################################################################################

def calc_result(value_array, flag):

    """ delta_sigma = calsagos.utils.calc_result(value_array, flag)

    This function was developed by P. Cerulo (05/09/2014)

	Function that estimates the median and 1 sigma width of a sample

	:param value_array: array with different measurements 
    of a physical quantity which is used to compute the 
    median value and the uncertainty of this physical 
    quantity
	:param flag: parameter that allows the user to choose 
    between symmetric and asymmetric confidence intervals 
	
    :type value_array: array
	:type flag: string
	
    :returns: The median value of the physical quantity 
    and its symetric or asymmetric error
	:rtype: numpy

    .. note::

	The returned variables are numpy arrays. The first
    element corresponds to the median value of the 
    physical quantity

    The parameter flag must be between quotes, i.e. 
    "symmetric" or "asymmetric"

    :Example:
    >>> from calsagos import *
    >>> import numpy
    >>> a = numpy.array([1.5, 1.0, 1.4, 1.6, 1.5])
    >>> utils.calc_result(a, "symmetric")
    array([1.5 , 0.14])
    >>> 
    >>> utils.calc_result(a, "asymmetric")
    array([1.5  , 0.244, 0.036])
    >>> 
    >>> mean = utils.calc_result(a, "symmetric")[0]
    >>> error = utils.calc_result(a, "symmetric")[1]
    >>> print("mean:", mean)
    mean: 1.5
    >>> print("error:", error)
    error: 0.14000000000000012
    >>> 
    >>> mean = utils.calc_result(a, "asymmetric")[0]
    >>> error_low = utils.calc_result(a, "asymmetric")[1]
    >>> error_high = utils.calc_result(a, "asymmetric")[2]
    >>> print("mean:", mean)
    mean: 1.5
    >>> print("error_low:", error_low)
    error_low: 0.24400000000000022
    >>> print("error_high:", error_high)
    error_high: 0.03600000000000003
        
	"""

    # -- estimate median and error of the median by using the "asymetric" flag
    value_median = stats.scoreatpercentile(value_array, 50)
    value_low = stats.scoreatpercentile(value_array, 16)
    value_high = stats.scoreatpercentile(value_array, 84)

    # -- estimate median and error of the median by using the "symetric" flag
    delta_value = 0.5*(value_high - value_low)
    delta_value_low = (value_median - value_low)
    delta_value_high = (value_high - value_median)
    
    # -- create output array
    result_value = np.array([value_median, delta_value_low, delta_value_high])
    result_value_symmetric = np.array([value_median, delta_value])

    # -- selecting output
    if flag=="symmetric":
        # -- returning output quantity
        return result_value_symmetric

    if flag=="asymmetric":
        # -- returning output quantity
        return result_value

#####################################################################################################################################################################################
#####################################################################################################################################################################################

def convert_kpc_to_angular_distance(distance_in_kpc, redshift, input_H, input_Omega_m, flag):

    """ angular_distance = calsagos.utils.convert_kpc_to_angular_distance(distance_in_kpc, redshift, input_H, input_Omega_m, flag)

	Function that converts a distance in kiloparsec to a distance
    in angular units at the redshift of a single cluster assuming
    a LCDM cosmology

    This function was developed by D. Olave-Rojas (19/06/2016)

	:param distance_in_kp: The distance to convert from kiloparsec 
        to angular distance (i.e dec, min or deg)
	:param redshift: redshift at which the distance will be convert 
    :param input_H: Hubble constant 
    :param intput_Omega_m: Omega matter
	:param flag: The units to convert the distance_in_kpc

	:type distance_in_kpc: int, array, float
	:type redshift: int, float
	:type input_H: int, float
    :type intput_Omega_m: int, float
    :type flag: string ("degrees", "arcminutes" or "arcseconds")

	:returns: The converted distance in angula units
	:rtype: int, float, array

	:Example:
    >>> from calsagos import utils
    >>> utils.convert_kpc_to_angular_distance(1, 0.2, 70., 0.3, "degrees")
    8.418388522320427e-05
    >>> utils.convert_kpc_to_angular_distance(1, 0.2, 70., 0.3, "arcminutes")
    0.005051033113392256
    >>> utils.convert_kpc_to_angular_distance(1, 0.2, 70., 0.3, "arcseconds")
    0.3030619868035354

	"""

    # -- deriving conversion factor from arcminutes to kpc at the redshift of the cluster
    my_cosmo = FlatLambdaCDM(H0=input_H * u.km / u.s / u.Mpc, Om0=input_Omega_m)
    kpc_to_arcmin = my_cosmo.kpc_proper_per_arcmin(redshift)

    # -- converting distance from kpc to arcminutes (P. Cerulo 28/06/2016)
    distance_angular_arcmin = distance_in_kpc / kpc_to_arcmin
    distance_angular_arcmin_value = distance_angular_arcmin.value

    # -- converting distance from arcminutes to arcseconds (D. Olave-Rojas 20/09/2016)
    distance_angular_arcsec = distance_angular_arcmin*60.0
    distance_angular_arcsec_value = distance_angular_arcsec.value

    # -- converting distance from arcminutes to degrees (P. Cerulo 28/06/2016)
    distance_angular_deg = distance_angular_arcmin/60.0
    distance_angular_deg_value = distance_angular_deg.value

    # -- selecting units of output
    if flag=="arcseconds":
        # -- returning output quantity
        return distance_angular_arcsec_value

    if flag=="arcminutes":
        # -- returning output quantity
        return distance_angular_arcmin_value

    if flag=="degrees":
        # -- returning output quantity
        return distance_angular_deg_value
   
#####################################################################################################################################################################################
#####################################################################################################################################################################################

def calc_radius_finn(mass, redshift, input_H0, input_Omega_L, input_Omega_m, flag):

    """ r200 = utils.calc_radius_finn(mass, redshift, input_H0, input_Omega_L, input_Omega_m, flag)

	Function that estimates the r200 of a single cluster
    by using the m200 of the cluster.
    
    The r200 of the cluster is estimated using the 
    equation (7) presented by Finn et al. (2005)

    This funcion was develop by D. Olave-Rojas (07/09/2016)
    
	:param mass: m200 of the cluster, this parameter
        must be in solar units
	:param redshift: redshift of the cluster  
    :param input_H: Hubble constant 
    :param input_Omega_L: Omega Lambda
    :param intput_Omega_m: Omega matter
    :param flag: The units of the output radius


	:type mass: float
	:type redshift: int, float
	:type input_H: int, float
    :type input_Omega_L: int, float
    :type intput_Omega_m: int, float

	:returns: the r200 of the cluster in meters
	:rtype: float

	:Example:
    >>> from calsagos import utils
    >>> utils.calc_radius_finn(1.279e15, 0.396, 70., 0.7, 0.3, "kiloparsec")
    41991.666521983534
    >>> utils.calc_radius_finn(1.279e15, 0.396, 70., 0.7, 0.3, "meters")
    1.2957274399634617e+24

	"""

    # -- renaming constants
    K_1 = 100.**(-1)
   
    # -- defining cosmology
    Hs = input_H0/(1e6*1e-3*u.pc.to(u.m)) #Huble constant in s-1
    Omega_L = input_Omega_L
    Omega_m = input_Omega_m

    # -- defining the gravitational constant
    grav_const = const.G
    g = grav_const.value # in Kg m-3 s-2

    # -- defining the solar mass
    solar_mass = const.M_sun
    ms = solar_mass.value

    # -- converting the mass of the clusters in solar units to mass in kilograms
    mass_kg =  mass * ms # cluster mass in kg units

    # -- groupimg constant terms
    element_1 = (Hs**2) * ( Omega_L + Omega_m * ((1 + redshift)**3))
    element_2 = element_1**(-1)

    # -- estimate of cubic radius
    cubic_radius = (g * K_1) * (mass_kg * element_2) 

    # -- estimate of halo radius
    cluster_radius_meter = cubic_radius**(1/3.) # radius in meters

    # -- converting the cluster radius in meters to kpc
    cluster_radius_kpc = cluster_radius_meter * ((u.m.to(u.pc))/1e3)

    # -- selecting units of output
    if flag=="meters":
        # -- returning output quantity
        return cluster_radius_meter

    if flag=="kiloparsec":
        # -- returning output quantity
        return cluster_radius_kpc


#####################################################################################################################################################################################
#####################################################################################################################################################################################

def calc_angular_distance(RA, DEC, RA_cen, DEC_cen, unit_flag):

    """ angular_distance = calc_angular_distance(RA, DEC, RA_cen, DEC_cen, unit_flag)

    Function that computes the projected angular 
    distance from a fixed position, for example
    with respect to the central position of a
    cluster
    
    The angular distance between galaxies is estimated
    as the euclidean distance

    This funcion was develop by P. Cerulo (13/12/2015)
    
	:param RA: Right Ascention of the galaxy
    :param DEC: Declination fo the galaxy
    :param RA_cen: Right Ascention of the
        fixed position
    :param DEC_cen: Declination of the
        fixed position
    :param unit_flag: set the units for
        the angular distance (deg, arcmin or arsec)

	:type RA: float, array
	:type DEC: float, array
    :type RA_cen: float
    :type DEC_cen: float
    :type unit_flag: string

	:returns: angular distance from a fixed position
	:rtype: array

	:Example:
        >>> import numpy as np
        >>> ra = np.array([64.197184, 64.197281, 64.197996])
        >>> dec = np.array([-24.223233, -23.885687, -23.954148])
        >>> ra0 =  64.0349 
        >>> dec0 = -24.0724
        >>> from calsagos import utils
        >>> utils.calc_angular_distance(ra, dec, ra0, dec0, "degrees")
        array([0.22155516, 0.24744562, 0.20145431])

	"""

    # -- defining useful quantities
    dim = RA.size
    angular_distance = np.zeros(dim)


    # -- calculating angular distance in decimal degrees
    for ii in range(dim):

        if RA[ii] < 0.0 or RA[ii] > 360.0 or RA_cen < 0.0 or RA_cen > 360.0 or DEC[ii] < -90.0 or DEC[ii] > 90.0 or DEC_cen < -90.0 or DEC_cen > 90.0:

            angular_distance[ii] = -99.9

        elif RA[ii] >= 0.0 and RA[ii] <= 360.0 and RA_cen >= 0.0 and RA_cen <= 360.0 and DEC[ii] >= -90.0 and DEC[ii] <= 90.0 and DEC_cen >= -90.0 and DEC_cen <= 90.0:

            angular_distance[ii] = math.sqrt( (RA[ii]-RA_cen)**2 + (DEC[ii]-DEC_cen)**2  )


    # -- selecting units of output
    if unit_flag == "degrees":
        # -- returning output quantity
        output_quantity = angular_distance

    if unit_flag == "arcminutes":
        # -- returning output quantity
        output_quantity = angular_distance*60.0

    if unit_flag == "arcseconds":
        # -- returning output quantity
        output_quantity = angular_distance*3600.0
  
    # -- returning output quantity
    return output_quantity

#####################################################################################################################################################################################
#####################################################################################################################################################################################

def calc_knn_galaxy_distance(RA, DEC, knn):

    """ calsagos.utils.calc_knn_galaxy_distance(RA, DEC)

	Function that computes the angular
    distance between one galaxy and their 
    k-nearest neighbors

    This funcion was develop by D. Olave-Rojas
    (23/07/2021)

	:param RA: Right Ascension of the galaxies
        in degree units
    :param DEC: Declination of the galaxies 
        in degree units
    :param knn: k nearest neighbors

    :type RA: array
    :type DEC: array
    :type knn: int

	:returns: the distance between one galaxy
        and its K.Nearest Neighbors and the
        mean distance between one galaxy and 
        their k-nearest neighbors
    :rtype: numpy array

    .. note::

    The angular galaxy distance is estimated as 
    the euclidian distance between one galaxy and 
    the all galaxies in the sample, then this 
    function selects the K-Nearest Neighbors to a 
    each galaxies.

    The output of this module is:

    calc_knn_angular_distance(RA, DEC, knn)[0] corresponds to the
        distance from a galaxy to its K-Nearest Neighbor

    calc_knn_angular_distance(RA, DEC, knn)[1] corresponds to the
        mean distance between a galaxy and its K-Nearest Neighbors

	:Example:
        >>> import numpy as np
        >>> ra = np.array([64.197184, 64.197281, 64.197996, 
        63.966533, 64.024977, 64.033479, 63.968247, 64.028488, 
        64.027309, 63.972432])
        >>> dec = np.array([-24.223233, -23.885687, -23.954148, 
        -23.922035, -24.072083, -24.150164, -23.944553, 
        -24.084211, -24.145245, -23.940725])
        >>> from calsagos import utils
        >>> distance = utils.calc_knn_galaxy_distance(ra, dec, 3)
        >>> print("knn_distance: ", distance[0])
        knn_distance:  [0.12206446 0.09998393 0.0926183 0.01406066 
        0.02860838 0.02467747 0.00941827 0.02455712 0.02297874 0.0084235 ]
        >>> print("mean_knn_distance: ", distance[1])
        mean_knn_distance:  0.04473908308759126

	"""

    # -- defining output quantities
    dim = RA.size # -- number of elements in the input arrays
    distance = np.zeros(dim) # -- distance between K-Nearest Neighbors
    mean_distance = np.zeros(dim) # -- mean distance between the K-Nearest Neighbors
    d_knn = np.zeros(dim)

    # -- START OF LOOP --
    for ii in range(dim):
        
        for jj in range(dim):
            
            # -- estimating the distance between galaxies in the sample
            distance[jj] = np.sqrt((RA[ii] - RA[jj])**2. + (DEC[ii] - DEC[jj])**2.)
                        
            # -- sorting the distance between galaxies with the aim to select the nearest neighbors
            sorted_indices = np.argsort(distance)
 
            # --synchronise the distance array with the sorted distance
            d_of_d_sorted = distance[sorted_indices]
            
            # -- select the distance of the K-Nearest Neighbours in the synchronised distance array
            d_nearest_k = d_of_d_sorted[0:knn]
      
            # -- estimating the mean distance between the K-Nearest Neighbors
            mean_distance[ii] = np.mean(d_nearest_k)

            # -- estimating the distance from a point to its K-Nearest Neighbor
            d_knn[ii] = d_nearest_k[knn-1]

    # -- END OF LOOP --

    # -- defining the distance between a galaxy from its K-Nearest Neighbor
    knn_distance = d_knn

    # -- defining the mean distance between a galaxy from a its K-Nearest Neighbors
    angular_distance = mean_distance

    # -- building matrix with output quantities
    distance_array = np.array([knn_distance, angular_distance], dtype=object)

    # -- returning output quantity
    return distance_array

#####################################################################################################################################################################################
#####################################################################################################################################################################################

def calc_mean_angular_galaxy_distance(RA, DEC):

    """ calsagos.utils.calc_angular_galaxy_distance(RA, DEC)

	Function that computes the angular
    distance between all galaxies in 
    the sample

    This function was developed by D. Olave-Rojas
    (23/07/2021)

	:param RA: Right Ascension of the galaxies
        in degree units
    :param DEC: Declination of the galaxies 
        in degree units

    :type RA: array
    :type DEC: array

	:returns: mean and median distance between 
        one galaxy and all the others in the 
        sample
	:rtype: array

    .. note::

    The angular galaxy distance 

	:Example:
    >>> import numpy as np
    >>> ra = np.array([64.197184, 64.197281, 64.197996])
    >>> dec = np.array([-24.223233, -23.885687, -23.954148])
    >>> from calsagos import utils
    >>> distance = utils.calc_mean_angular_galaxy_distance(ra, dec)
    >>> print("mean distance:", distance[0])
    mean distance: [0.20221074636463557 0.135336915843549 0.11251698624994522]
    >>> print("median distance:", distance[1])
    median distance: [0.2690862251565477 0.06846473359328796 0.06846473359328796]


	"""

    # -- defining output quantities
    dim = RA.size # number of elements in the input arrays
    distance = np.zeros(dim) # distance between galaxies
    mean_distance = np.zeros(dim)
    median_distance = np.zeros(dim)

    # -- START OF LOOP --
    for ii in range(dim):
        
        for jj in range(dim):
            
            # -- estimating the distance between galaxies in the sample
            distance[jj] = np.sqrt((RA[ii] - RA[jj])**2. + (DEC[ii] - DEC[jj])**2.)

            # -- estimating the mean distance 
            mean_distance[ii] = np.mean(distance)

            # -- estimating the median distance 
            median_distance[ii] = np.median(distance)

    # -- END OF LOOP --

    # -- defining the mean distance between a galaxy and all other galaxies in the sample
    mean_angular_distance = mean_distance

    # -- defining the median distance between a galaxy and all other galaxies in the sample
    median_angular_distance = median_distance

    # -- building matrix with output quantities
    distance_array = np.array([mean_angular_distance, median_angular_distance], dtype=object)

    #-- returning output quantity
    return distance_array

#####################################################################################################################################################################################
#####################################################################################################################################################################################

def best_eps_dbscan(id, distance):

    """ calsagos.utils.best_eps_dbscan(id_galaxy, galaxy_distance)

	Function that finding the elbow distance between 
    galaxies. This distance is used as eps in the 
    LAGASU implementation.

    This implementation is based on the approach 
    suggest by Ester 1996 to selec the best value
    to eps

    This funcion was develop by D. Olave-Rojas
    (07/09/2021)

	:param id: ID of each galaxy in the sample
    :param distance: distance between 
        galaxies

    :type id_galaxy array
    :type galaxy_distance: array

	:returns: elbow distance between galaxies
	:rtype: float

    .. note::

    The distance between galaxies can be determine
    as the projected euclidean distance between 
    galaxies or between a galaxy and its K-Nearest
    Neighbor.

    To determine the projected euclidean distance 
    between galaxies the user can used the function
    calc_mean_angular_galaxy_distance(RA, DEC) or 
    the function calc_knn_galaxy_distance(RA, DEC, knn)
    both are available in utils module as part
    of CALSAGOS package

	:Example:
    >>> import numpy as np
    >>> id = np.array([341000117030202.0, 341000117029276.0, 341000117029404.0, 341000117029483.0, 341000117029983.0, 341000117029614.0, 341000117029149.0, 341000117030088.0])
    >>> ra = np.array([96.4737786539, 96.5948071255, 96.4413750785, 96.5698475786, 96.4802179965, 96.56326854, 96.5878971202, 96.4808231266])
    >>> dec = np.array([-7.78071524633, -7.82582696523, -7.78138696923, -7.77668028787, -7.84946684319, -7.82657346754, -7.80012448146, -7.79229815527])
    >>> from calsagos import utils
    >>> knn_distance = utils.calc_knn_angular_galaxy_distance(ra, dec, 3)[0]
    >>> utils.best_eps_dbscan(id, knn_distance)
    0.04092923746784588


	"""   

    # -- sorting the distances from a galaxy to it k-nearest neighbor
    galaxy_distance = np.sort(distance, axis=0)

    # -- defining output quantities
    dim = len(id)
    number_galaxy = np.zeros(dim)

    # -- creating a list of indices to be used in the estimation of eps
    for ii in range(dim):
        
        number_galaxy[ii] = ii
    
    # -- creating an correlative ID to each galaxy
    number = number_galaxy

    # -- creating an array with index and the distances to the k-nearest neighbor
    array_id_distance = np.array([number, galaxy_distance]).T

    # -- finding the knee of a curve or the elbow of a curve.
    rotor = Rotor()
    rotor.fit_rotate(array_id_distance)
    elbow_idx = rotor.get_elbow_index()
    
    # -- selecting the elbow distance in the distribution
    best = np.where(number == elbow_idx)[0]
    best_eps = galaxy_distance[best]

    # -- defininf the elbow distance
    elbow_distance = float(best_eps)

    # -- returning output quantity
    return elbow_distance

#####################################################################################################################################################################################
#####################################################################################################################################################################################
