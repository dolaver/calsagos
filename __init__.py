"""
CALSAGOS
=====

Clustering ALgorithmS Applied to Galaxies in Overdense Systems
is a python package develop to select cluster members and to
search, find and identify substructures in and around a galaxy 
cluster using the redshift and position of the galaxies.

CALSAGOS was develop to carry out the analysis presented by 
Olave-Rojas et al. (2018). 

CALSAGOS has the following modules:

  1. utils: collection of functions developed to estimate errors,
            distances and to convert quantities
  2. redshift_boundaries: collection of functions developed to 
                          estimate the cluster redshift, the velocity
                          dispersion and the limits in the redshift 
                          distribution
  3. cluster_kinematics: collection of functions developed to estimate 
                         the kinematic properties of the cluster
  4. ds_test: collection of functions developed to implement the 
              Dressler-Shectman Test (DS-Test, Dressler & Shectman, 1988) 
  5. isomer: Identifier of SpectrOscopic MembERs allows to identify the
             spectroscopic cluster members defining as spectroscopic 
             members as those galaxies with a peculiar velocity
             lower than the escape velocity of the cluster
  6. clumberi: CLUster MemBER Identifier allows to identify cluster
               members using a 3D-Gaussian Mixture Modules (GMM)
  7. lagasu: LAbeller of GAlaxies within SUbstructures assigns galaxies to
             different substructures in and around a galaxy cluster

The modules of CALSAGOS can be used separately or together. An example
can be found in folder 'example'. For further information about the package
see Olave-Rojas et al. 2022

If your scientific publication is based on either version of CALSAGOS, 
then please cite Olave-Rojas et al. 2022

---------------------
Important Information
---------------------

CALSAGOS was developed in pyhton 3.8 and uses some pre-existing python 
modules as:

  1. numpy
  2. astropy
  3. matplotlib
  4. sys
  5. math
  6. sklearn
  7. scipy
  8. kneebow

CALSAGOS has been developed and tested by using the
following versions of the pre-existing python modules

  numpy         version '1.21.3'
  astropy       version '4.3.1'
  matplotlib    version '3.4.3'
  sklearn       version '1.0.1'
  scipy         version '1.7.1'

"""

from . import (utils, lagasu, cluster_kinematics, redshift_boundaries, ds_test, isomer, clumberi)

__all__ = ['utils', 'lagasu', 'cluster_kinematics', 'redshift_boundaries', 'ds_test', 'isomer', 'clumberi']
__author__ = 'D. Olave-Rojas & P. Cerulo'
__email__ = 'daniela.olave@utalca.cl - pcerulo@inf.udec.cl'
__ver__ = '0.1'
__date__ = '2021-07-05'
