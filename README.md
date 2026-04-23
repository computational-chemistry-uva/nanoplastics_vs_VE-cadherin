# BayesProtNPSim

Scripts used in the publication work can be found here.
- charmm27.ff: the modified force field
- mdps: .mdp files with simulation settings
- notebooks: Jupyter notebooks used for umbrella integration, force extraction and Bayesian quadrature
- plumed files: files for biased simulations
- protein: structure and topology of cadherin EC1
- scripts: sub-divided into plastic and protein

Required python packages:
- numpy
- matplotlib
- scikit-optimize
- mdtraj
- acpype
- ambertools
- emukit
- scienceplots (optional, used in some notebooks)

Usage:
1. Change the proj_path variable in the following scripts in your cloned repo, according to the actual path:
- scripts/optimize.py
- scripts/plastic/gen.sh
- scripts/prot_plastic/prep_prot_pl.sh
- scripts/prot_plastic/prep_umb.sh

2a. If you would like to use the automatic procedure of finding and evaluating new points using the Bayesian optimiser, copy the scripts/optimize.py script in your main result folder. Please make sure that the initial files needed for your polymer are there (see: scripts/plastic/example_inputs_ps). 

2b. You can also carry out simulations manually with pre-defined COM separation values, which can for example produce simple free-energy profiles for a given plastic length. For this, execute the following scripts in order:
- scripts/plastic/gen.sh
- scripts/prot_plastic/put_together.py
- scripts/prot_plastic/prep_manual_umb.sh
- scripts/prot_plastic/simulate.sh, you can use submit scripts from plumed_files/
- notebooks/integration.ipynb to get the free-energy profiles and block analysis
- notebooks/get_force.ipynb to get the force profile and values for the optimiser / Bayesian quadrature

3. To get a 3D dependence of both COM separation, as well as the plastic length on the free-energy using Bayesian quadrature, follow quadrature.ipynb.

![](notebooks/3Dsurface.gif)

Good luck!:)
