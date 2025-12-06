# CALSAGOS

CALSAGOS (Clustering ALgorithmS Applied to Galaxies in Overdense Systems) is a python package developed to select cluster members and to search, find and identify substructures and galaxy groups in  and around galaxy clusters using the redshift and position in the sky of the galaxies. CALSAGOS can determine the cluster members in two ways. One by using ISOMER (Identifier of SpectrOscopic MembERs) which is a function developed to select the spectroscopic cluster members defining cluster members as those galaxies with a peculiar velocity lower than the escape velocity of the cluster. On the other hand, CALSAGOS can select the cluster members by using CLUMBERI (CLUster MemBER Identifier) which is a function developed to select the cluster members using a 3D-Gaussian Mixture Modules (GMM). Both functions remove the field interlopers by using a 3-sigma clipping algorithm.

In order to search, find and identify substructures and groups in and around a galaxy cluster CALSAGOS uses LAGASU (LAbeller of GAlaxies within SUbstructures) which is a function that allows to assign the galaxies in a cluster to different substructures and groups in and around the galaxy cluster. LAGASU is based on clustering algorithms (GMM and DBSCAN) which search areas with high density to define a substructure or groups

## Install

You must install numpy, astropy, matplotlib, sys, math, sklearn, scipy and kneebow prior to install CALSAGOS

```
python3 setup.py install
```

## Updates

- LAGASU module and test.py were updated by D. E. Olave-Rojas on May 22, 2023 to include the correction of the label of substructures directly in this function
- UTILS, LAGASU and test.py were updated by D. E. Olave-Rojas on June 02, 2023 to improve the correction of the label of substructures
- UTILS and LAGASU and were updated by D. E. Olave-Rojas on November 10, 2025 to include metrics "haversine" and "euclidean" and method "dbscan" and "hdbscan"
- UTILS and LAGASU were updated by D. E. Olave-Rojas on December 05, 2025 to include a stability test in the identification of substructures using lagasu

## Example Usage

```py
from calsagos import lagasu
from calsagos import utils
from calsagos import clumberi

#-- select cluster members
cluster_members = clumberi.clumberi(id_galaxy, ra_galaxy, dec_galaxy, redshift_galaxy, cluster_initial_redshift, ra_cluster, dec_cluster, range_cuts)

# -- defining output parameters from clumberi
id_member = cluster_members[0]
ra_member = cluster_members[1]
dec_member = cluster_members[2]
redshift_member = cluster_members[3]
cluster_redshift = cluster_members[4]

cluster_sample = np.array([ra_member, dec_member]).T

#-- estimating the galaxy separation of galaxies in the cluster sample to be used as input in lagasu
neigh = NearestNeighbors(n_neighbors=(n_galaxies+1), metric=metric_distance, algorithm='ball_tree').fit(cluster_sample)
distances, indices = neigh.kneighbors(cluster_sample)

#-- determining the distance to the k-nearest neighbor of each galaxy in the cluster
knn_distance = distances[:,n_galaxies]

typical_separation = utils.best_eps_dbscan(id_member, knn_galaxy_distance)

#-- Assign galaxies to each substructures
label_candidates = lagasu.lagasu(id_member, ra_member, dec_member, redshift_member, range_cuts, typical_separation, n_galaxies, metric_distance, 'dbscan', ra_cluster, dec_cluster, cluster_redshift, r200, flag)

#-- defining output parameters from lagasu
id_candidates = label_candidates[0] #-- id of reach galaxy in the catalog with cluster members
ra_candidates = label_candidates[1] #-- R.A. of each galaxy in the catalog with cluster members
dec_candidates = label_candidates[2] #-- Dec. of each galaxy in the catalog with cluster members
redshift_candidates = label_candidates[3] #-- redshift of each galaxy in the catalog with cluster members
label_zcut = label_candidates[4] #-- label of each galaxy in the catalog with cluster members. This parameter is given by GMM 
label_dbscan = label_candidates[5] #-- label of each galaxy in the catalog with cluster members. This parameter is given by DBSCAN
label_final = label_candidates[6] #-- label of each galaxy in the catalog with cluster members. Equal to -1 if the galaxy is part of the principal halo and has a value between 0 to N if th galaxy is part of a substructure 
    
# all done

```

Which makes:

A catalogue with cluster members and substructure identification

![alt tag](https://i.ibb.co/j8dRSjr/output-catalogue-CALSAGOS.png)

This catalog can be plotted to be able to see in the cluster and its substructures in the sky

![alt tag](https://i.ibb.co/VS5GCNk/output-CALSAGOS.png)

Gray dots correspond to the cluster and the dots in other colors represent the substructures in and around the cluster

## API

<!-- Full CALSAGOS documentation can be viewed in  [pdf](http://www.baryons.org/ezgal/manual.pdf) format. # CAMBIAR -->

If your scientific publication is based on either version of CALSAGOS, then please cite [Olave-Rojas et al. 2023](https://doi.org/10.1093/mnras/stac3762)

## Testing

```
python3 test.py
```
An example can be find [here](https://github.com/dolaver/calsagos/tree/main/test/)


## Contributors

[Daniela E. Olave-Rojas](https://github.com/dolaver/) and Pierlugi Cerulo

## Acknowledgements

We gratefully acknowledge the collaborators of this project for giving us permission to use the data to test this package and for providing us important observations about the work. We especially thank Claudia Mendes de Oliveira and Pablo Araya-Araya for give us access to the mock catalogues from the Southern Photometric Local Universe Survey (S- PLUS; Mendes de Oliveira et al. 2019) and to David Olave-Rojas for help us to develop this package. We are also grateful to Ricardo Demarco, Yara Jaffé and Diego Pallero for their input on this project. Finally, we want to thank everyone who has tested CALSAGOS extensively and provided us with invaluable feedback.


## License

This project is licensed under the terms of the [MIT](https://github.com/dolaver/calsagos/blob/main/LICENSE) license
