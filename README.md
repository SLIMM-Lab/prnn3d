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
 
<p align="center">
<img src="https://surfdrive.surf.nl/files/index.php/apps/files_sharing/ajax/publicpreview.php?x=1920&y=802&a=true&file=testset_gpthick_loadvector.png&t=srDAT3xQGS3DIQQ&scalingup=0" width="75%" height="75%"/>
</p>

## Looking for a python version?

A general repository for all developments related to PRNNs is available at https://github.com/SLIMM-Lab/pyprnn. 
The repository contains a standalone demonstration in python along with datasets and visualization tools for reproducing results in [1]. 

