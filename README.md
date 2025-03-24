# percolate_mac

This code estimates the electrical conductances of ensembles of amorphous monolayer carbon (AMC) structures using percolation theory.

It was used to obtain the results described in this paper: https://arxiv.org/abs/2411.18041. Check it out (and its Supporting Information) for more details on the theory behind this code and its intended use.

The file `requirements.txt` contains the list of dependencies required to run this code. It can be used to construct a virtual environment using something like:
```
virtualenv --no-download percolate_mac_env
source percolate_mac_env/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
```
Many of the helper modules import functions from `qcnico`, which is another Python module not included in `requirements.txt`. The code for `qcnico` can be found here: https://github.com/ngastellu/qcnico.

## How to run this code
The idea behind our approach is to estimate the ensemble-averaged conductance of a set of many AMC structures (see abovementioned paper for details), whose electronic structure has already been partially obtained using [ArpackMAC](https://github.com/ngastellu/ArpackMAC). 

Before running this code you therefore need:
* A set of .xyz files containing the atomic positions of an ensemble AMC structures. All files corresponding to a given ensemble need to be stored in the same directory, and must be named according the following convention: `<xyz_prefix>-<n>.xyz`, where `<xyz_prefix>` can be defined in `deploy_percolate.load_atomic_positions` and `n` is an integer indexing each AMC structure in the ensemble.
* The NPY files (tight-binding molecular orbitals/energies) produced by `ArpackMAC` , from the structures of the ensemble of interest. Do not rename any of the files produced by `ArpackMAC`.
The calculation of an AMC ensemble's conductance is then done in two steps:
1. Construct the variable range hopping networks for each structure by running `full_run_percolate.py` on each structure in the ensemble.
2. Use the distribution of critical percolation thresholds in the ensemble to calculate its conductance, with `obtain_conductances.py`.
For more information on these two scripts, see the Contents section below.

**N.B.:** `full_run_percolate.py` only processes a single file, and so must be invoked in run separately for each structure in the ensemble. On the other`obtain_conductivities.py` loops over all structures, and thus only needs to be run once per ensemble.

## Contents
### Constructing the hopping conductance networks
The main script is `full_run_percolate.py`. It constructs the hopping sites from the tight-binding eigenstates obtained by the code in `ArpackMAC`, and builds the conductance networks between them edge by edge, until a network connects the left edge of the structure to the right edge.

The script takes three command line arguments:
* `n`: an integer indexing the specific AMC structure to run the percolation calculation on
* `struc_type`: the name of the ensemble to which the AMC structure being considered belongs. Three valid values: `sAMC-300`, `sAMC-q400`, and `sAMC-500`.
* `mo_type`: The type of tight-binding eigenstate used to construct the hopping sites and run the percolation calculation.  Three valid values: `lo` (lowest-energy eigenstates), `hi` (highest energy eigenstates), and `virtual_w_HOMO` (low-lying unoccupied eigenstates).

### Computing the conductance of an ensemble.
Once the percolation networks have been constructed for all of the desired structures in a given ensemble, use the `obtain_conductances.py` script to obtain the ensemble's electrical conductances from the distribution of critical percolation distances in that ensemble:

$$G = \frac{q_e^2\omega_0}{k_{\text{B}}T}\,\int e^{-\xi}\,P(\xi)\,\text{d}\xi\,.$$
The values for the escape frequency $\omega_0$ and the temperature can be set at the top `utils_analysis.py`.
The script accepts a single command line argument: `mo_type`, which is the same as the `mo_type` argument for `full_run_percolate.py`.

## Other modules
The other Python files are helper modules that contain the code doing the actual computations. Here is a brief description of their purpose:
* `MOs2sites.py`: Handles the mapping from molecular orbitals to variable range hopping sites
* `percolate.py`: Defines hopping rates between pairs of sites and constructs the hopping conduction networks.
* `utils_arpackMAC.py`: Processes the output from  the`ArpackMAC` code used to obtain the tight-binding eigenstates of each AMC structure.
* `utils_analysis.py`: Processes the results of the percolation calculation run by `percolate` to obtain the conductance of a given AMC ensemble.

## Some usage notes
The code expects the output of the electronic structure calculation (tight-binding eigenstates, and eigenvalues) to be stored in NPY files with same names as the one produced by the `ArpackMAC` software package (https://github.com/ngastellu/ArpackMAC). However, this can be remedied by appropriately modifying the data loading function `load_arpack_data` in `deploy_percolate.py`.

This code was developed and ran on the [Narval](https://docs.alliancecan.ca/wiki/Narval/en) HPC cluster operated by the Digital Research Alliance of Canada. 

## Citing this code
Please cite the following paper in any work using this code: https://arxiv.org/abs/2411.18041

Gastellu, Nicolas, Ata Madanchi, and Lena Simine. "Disentangling morphology and conductance in amorphous graphene." *arXiv preprint arXiv:2411.18041* (2024).
