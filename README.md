# VC-SQNM
A refactored Python implementation of the (VC)-SQNM optimization algorithm.

The original implementation including Fortran and C++ can be found [here](https://github.com/moritzgubler/vc-sqnm/).

The stabilized quasi Newton method (SQNM) is a fast and reliable optimization method that is well adapted to find local minima on the potential energy surface. When using the SQNM method please cite [the initial SQNM paper](https://aip.scitation.org/doi/10.1063/1.4905665).

This repository also contains an implementation of the variable cell shape SQNM method which optimizes also the lattice vectors of systems with periodic boundary conditions. When the vc-SQNM method is used please cite the [vc-SQNM](https://arxiv.org/abs/2206.07339) paper as well. 
