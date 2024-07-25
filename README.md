# Physically Recurrent Neural Networks

**Intact** constitutive models embedded in an encoder-decoder MLP architecture.

If you have accurate material models at the microscale and would like to perform computational homogenization, those same models can be directly embedded into a hybrid architecture to make macroscale predictions.

<p align="center">
<img src="https://raw.githubusercontent.com/MarinaMaia2021/supportMaterial/main/matPointTime_paper2.gif" width="75%" height="75%"/>
</p>

Because the models in the architecture are the exact same as in the micromodel, a number of features can be directly inherited and therefore not learned from data:

- Path dependency (loading/unloading/reloading) without training for it
- Strain rate dependency while training with only a single rate
- Consistent step size dependency (independent for inviscid models; correct dependence for viscous models)
- Between $10\times$ and $100\times$ less training data than RNNs for comparable performance

## Journal papers and preprints

- [1] MA Maia, IBCM Rocha, P Kerfriden, FP van der Meer (2023), [PRNNs for 2D composites, elastoplastic](https://www.sciencedirect.com/science/article/pii/S0045782523000579)

- [2] MA Maia, IBCM Rocha, FP van der Meer (2024), [PRNNs for 3D composites, finite-strain thermoviscoelasticity, creep and fatigue](https://arxiv.org/abs/2404.17583)

- [3] MA Maia, IBCM Rocha, D Kovacevic, FP van der Meer (2024), Reproducing creep and fatigue experiments in thermoplastics using PRNNs -- **COMING SOON**

- [4] N Kovacs, MA Maia, IBCM Rocha, C Furtado, PP Camanho, FP van der Meer (2024), PRNNs for micromodels including distributed cohesive damage -- **COMING SOON**

## In this repository

In this repository you will find the C++ code used for generating the data and training the PRNN models in [2] for archiving purposes. 
An updated version will be released soon with a demonstration of PRNNs for the 3D micromodel with the Eindhoven Glassy Polymer (EGP) for describing the matrix and a transversally isotropic hyperelasticity model (Bonet) for the fibers.

In addition to the source code (`src`), the following datasets for training and testing are made available (`data`):

- `tr_propgp_080.data`: a set of 1100 **proportional GP** paths in random directions in the unit force vector space; 

The loading function consists of the sum of all prescribed displacements on the micromodel, and, in this set, it changes from one path to another according to a Gaussian Process (GP) with a suitable prior. In other words, unloading-reloading can take place at random times for different duration at a fixed direction in the unit force vector space.

- `test_propgp_080.data`: a set of 100 **proportional GP** paths in random directions in the unit force vector space; 

Same loading type as previous set, now with different directions (left).

- `test_nonpropgp_080.data`: set of 100 **non-proportional GP** paths. 

The last dataset contains the most complex type of loading, designed to be as general as possible, with different cycles of unloading-reloading per component. Similar to the approach in [1], each strain component of a path is sampled from a suitable GP prior (right).
 
<p align="center">
<img src="https://raw.githubusercontent.com/MarinaMaia2021/supportMaterial/main/testset_gpalenthick_loadvector.png" width="40%" height="40%"/>
<img src="https://raw.githubusercontent.com/MarinaMaia2021/supportMaterial/main/testset_gpthick_loadvector.png" width="40%" height="40%"/>
</p>

Despite the fixed time increment per path provided in these sets, the network can naturally extrapolate to different time increments as this is part of the inputs of the rate-dependent constitutive model [2].  

## Looking for a python version?

A general repository for all developments related to PRNNs is available at:

https://github.com/SLIMM-Lab/pyprnn.

The repository contains a standalone demonstration in python along with datasets and visualization tools for reproducing results in [1]. 

