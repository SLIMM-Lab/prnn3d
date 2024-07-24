# Physically Recurrent Neural Networks

**Intact** constitutive models embedded in an encoder-decoder MLP architecture.

If you have accurate material models at the microscale and would like to perform computational homogenization, those same models can be directly embedded into a hybrid architecture to make macroscale predictions.

Because the models in the architecture are the exact same as in the micromodel, a number of features can be directly inherited and therefore not learned from data:

- Path dependency (loading/unloading/reloading) without training for it
- Strain rate dependency while training with only a single rate
- Consistent step size dependency (independent for inviscid models; correct dependence for viscous models)
- Between $10\times$ and $100\times$ less training data than RNNs for comparable performance

<p align="center">
<img src="https://raw.githubusercontent.com/ibcmrocha/public/main/materialPointWithAndWithoutIntVars.gif" width="75%" height="75%"/>
</p>

## Journal papers and preprints

- [1] MA Maia, IBCM Rocha, P Kerfriden, FP van der Meer (2023), [PRNNs for 2D composites, elastoplastic](https://www.sciencedirect.com/science/article/pii/S0045782523000579)

- [2] MA Maia, IBCM Rocha, FP van der Meer (2024), [PRNNs for 3D composites, finite-strain thermoviscoelasticity, creep and fatigue](https://arxiv.org/abs/2404.17583)

- [3] MA Maia, IBCM Rocha, D Kovacevic, FP van der Meer (2024), Reproducing creep and fatigue experiments in thermoplastics using PRNNs -- **COMING SOON**

- [4] N Kovacs, MA Maia, IBCM Rocha, C Furtado, PP Camanho, FP van der Meer (2024), PRNNs for micromodels including distributed cohesive damage -- **COMING SOON**

## In this repository

In this repository you will find the C++ code used for generating the data and training the PRNN models in [2] for archiving purposes. 
An updated version will be released soon with a demonstration of PRNNs for the 3D micromodel with the Eindhoven Glossy Polymer (EGP) for describing the matrix and a transversally isotropic hyperelasticity model (Bonet) for the fibers.

In addition to the source code (`src` folder), the following datasets for training and testing can be found in the `demo` folder:

- A set of 100 **proportional** paths in random directions in the unit force vector space and same time increment for all curves. 

As the simplest scenario assessed, the loading function (sum of all prescribed displacements) of all paths is monotonic (left);

- A set of 100 **proportional GP** paths in random directions in the unit force vector space and different time increment per curve. 

This time, the loading function changes from one path to another according to a Gaussian Process (GP) with a suitable prior. This means that despite the constant direction in the unit load vector space, unloading-reloading can take place at random times for different duration (left);

- A set of 1100 **non-proportional GP** paths with different time increments per curve. 

The last dataset contains the most complex type of loading, designed to be as general as possible, with different cycles of unloading-reloading per component. Similar to the approach in [1], each strain component is sampled from a suitable GP prior (right).
 
<p align="center">
<img src="https://surfdrive.surf.nl/files/index.php/apps/files_sharing/ajax/publicpreview.php?x=1920&y=802&a=true&file=testset_gpalenthick_loadvector.png&t=MJ2OEY3a7e7YbF4&scalingup=0" width="40%" height="40%"/>
<img src="https://surfdrive.surf.nl/files/index.php/apps/files_sharing/ajax/publicpreview.php?x=1920&y=802&a=true&file=testset_gpthick_loadvector.png&t=srDAT3xQGS3DIQQ&scalingup=0" width="40%" height="40%"/>
</p>

## Looking for a python version?

A general repository for all developments related to PRNNs is available at https://github.com/SLIMM-Lab/pyprnn.

The repository contains a standalone demonstration in python along with datasets and visualization tools for reproducing results in [1]. 

