#!/usr/bin/env python

# import python modules
from astropy.table import Table
import numpy as np
import sys
import time

# import CALSAGOS modules
from calsagos import lagasu
from calsagos import utils
from calsagos import clumberi

#####################################################################################################################################################################################
#####################################################################################################################################################################################

def test_calsagos_clumberi(id_galaxy, ra_galaxy, dec_galaxy, redshift_galaxy, cluster_mass, cluster_initial_redshift, ra_cluster, dec_cluster, input_H, input_Omega_L, input_Omega_m, range_cuts, n_galaxies, flag, final_cluster_catalog):

    #-- Preliminar selection of galaxies in the area of clusters 

    #-- defining the cluster radius
    r200_kpc = utils.calc_radius_finn(cluster_mass, cluster_initial_redshift, input_H, input_Omega_L, input_Omega_m, "kiloparsec")

    #-- converting the radius in kpc to a radius in angular units
    #-- NOTE: if the user has an estimate of the r200 of the cluster it is not necessary to calculate this quantity
    r200_degree = utils.convert_kpc_to_angular_distance(r200_kpc, cluster_initial_redshift, input_H, input_Omega_m, "degrees") 

    #-- estimating the euclidean angular distance for each galaxy with respect to the central position of the cluster
    distance = utils.calc_angular_distance(ra_galaxy, dec_galaxy, ra_cluster, dec_cluster, "degrees")

    #-- setting a cut of galaxies within 3r200
    #-- NOTE: this cut is only to clean the mock catalogue. In a real case the user can decide if apply or not this cut
    good_galaxy = np.where( distance <= 3.*r200_degree)[0]

    #-- selecting galaxies within a cut  3r200
    id_good = id_galaxy[good_galaxy]
    ra_good = ra_galaxy[good_galaxy]
    dec_good = dec_galaxy[good_galaxy]
    redshift_good = redshift_galaxy[good_galaxy]

    #-- select cluster members
    #-- NOTE: if the user has a catalogue with the cluster members it is not necessary to select them newly
    cluster_members = clumberi.clumberi(id_good, ra_good, dec_good, redshift_good, cluster_initial_redshift, ra_cluster, dec_cluster, range_cuts)
    id_member = cluster_members[0]
    ra_member = cluster_members[1]
    dec_member = cluster_members[2]
    redshift_member = cluster_members[3]
    cluster_redshift = cluster_members[4]

    #-- estimating the galaxy separation of galaxies in the cluster sample to be used as input in lagasu
    knn_distance = utils.calc_knn_galaxy_distance(ra_member, dec_member, n_galaxies)

    #-- determining the distance to the k-nearest neighbor of each galaxy in the cluster
    knn_galaxy_distance = knn_distance[0]
    typical_separation = utils.best_eps_dbscan(id_member, knn_galaxy_distance)

    #-- Assign galaxies to each substructures
    label_candidates = lagasu.lagasu(id_member, ra_member, dec_member, redshift_member, range_cuts, typical_separation, n_galaxies, ra_cluster, dec_cluster, cluster_redshift, r200_degree, flag)

    #-- defining output parameters from lagasu
    id_candidates = label_candidates[0] #-- id of reach galaxy in the catalog with cluster members
    ra_candidates = label_candidates[1] #-- R.A. of each galaxy in the catalog with cluster members
    dec_candidates = label_candidates[2] #-- Dec. of each galaxy in the catalog with cluster members
    redshift_candidates = label_candidates[3] #-- redshift of each galaxy in the catalog with cluster members
    label_zcut = label_candidates[4] #-- label of each galaxy in the catalog with cluster members. This parameter is given by GMM 
    label_dbscan = label_candidates[5] #-- label of each galaxy in the catalog with cluster members. This parameter is given by DBSCAN
    label_final = label_candidates[6] #-- label of each galaxy in the catalog with cluster members. Equal to -1 if the galaxy is part of the principal halo and has a value between 0 to N if th galaxy is part of a substructure 

    #-- defining the number of galaxies in the cluster to print a table with output results
    n_members = id_member.size

    #-- Printing data table with all cluster members 
    output_table = open(final_cluster_catalog, 'w')

    sys.stdout = output_table

    print("# ID RA DEC redshift groups")

    for ii in range(n_members):

        print(id_candidates[ii], ra_candidates[ii], dec_candidates[ii], redshift_candidates[ii], label_final[ii])
         
    sys.stdout = sys.__stdout__

    output_table.close()

    print("output in", final_cluster_catalog)

#####################################################################################################################################################################################
#####################################################################################################################################################################################

#-- INPUT PARAMETERS

#-- Catalogue
input_mock_catalog = 'input_mock_SPLUS_catalogue.cat'

# -- reading quantities from the galaxies in mock catalog
catalog_table = Table.read(input_mock_catalog, format='ascii')
ID = catalog_table['galaxyId']
RA = catalog_table['ra']
DEC = catalog_table['dec']
redshift = catalog_table['z_app']

#-- COSMOLOGY
#- S-PLUS mock cosmology
cluster_H_mock = 67.3
cluster_Omega_L_mock = 0.685
cluster_Omega_m_mock = 0.315

#-- GENERAL PARAMETERS
cut = 15 # -- number of Gaussians to be fitted in the CLUMBERI and LAGASU implementation
ngal = 3 # -- number of minimum of galaxies that a group or substructure must have
sample = "zspec"

starting_redshift = 0.337461208999 # -- central redshift of the cluster
mass =  14.889801427038481 # -- log(m_cl/m_sun) of the cluster
cluster_mass = 10.**mass # -- mass of the cluster in kilograms

# -- central position of the cluster 
central_ra = 96.54288696045 # -- right ascention of the cluster in degree units
central_dec = -7.813029470475 # -- declination of the cluster in degree units  

#-- OUTPUT CATALOGS
final_output_catalog = 'output_CALSAGOS_CLUMBERI_catalogue.cat'

#-- RUNNING THE SCRIPT

t = time.localtime()
current_time = time.strftime("%H:%M:%S", t)
print("start time", current_time)

test_calsagos_clumberi(ID, RA, DEC, redshift, cluster_mass, starting_redshift, central_ra, central_dec, cluster_H_mock, cluster_Omega_L_mock, cluster_Omega_m_mock, cut, ngal, sample, final_output_catalog)

t = time.localtime()
current_time = time.strftime("%H:%M:%S", t)
print("end time", current_time)
