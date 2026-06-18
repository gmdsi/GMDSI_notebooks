---
layout: default
title: GMDSI Tutorial Notebooks
nav_order: 1
has_children: false
permalink: /
hide:
    - navigation
---

## Welcome to the GMDSI Tutorial Notebooks

__Please be aware that these notebooks are under development and subject to change. Make sure to check back regularly for updates and new content. If you come across major problems or errors, please open a detailed issue in the GitHub repository to let us know.__

GMDSI has committed itself to reducing the slope of the PEST(++) and pyEMU learning curve. It is organizing a series of Jupyter Notebooks that show you how to prepare PEST(++) input datasets, and query PEST(++) output files with ease. All steps are fully explained. So even if you do not intend to use pyEMU for PEST(++) pre/postprocessing, these notebooks will still be worth reading so that you can understand the decision-support modelling workflow. Then, if you like, you can implement the same workflow in a way that is best for you.

The notebooks are being prepared by Rui Hugman (GMDSI), Jeremy White (Intera) and Mike Fienen (USGS). They build upon previous work by the USGS and pyEMU development team. 

You can access completed versions of these tutorials through the links in the side-panel on the left. If you wish to run the notebooks yourself, please head to the * [GitHub repository](https://github.com/gmdsi/GMDSI_notebooks/) and follow the instructions to download files and install the necessary software. Note that you will need to run the notebooks in a specific order, as some notebooks rely on the existence of files created in other tutorials.

## Introductions to Selected Topics  
* [Intro to Regression](intro/intro_to_regression.md)  
* [Intro to pyEMU](intro/intro_to_pyemu.md)  
* [Intro to Geostatistics](intro/intro/intro_to_geostatistics.md)  
* [Intro to Bayes](intro/intro_to_bayes.md)  
* [Intro to SVD](intro/intro_to_svd.md)  

## Introduction to Theory, Concepts and PEST Mechanic  
* [Manual Trial-and-Error](part1/freyberg_trial_and_error.md)  
* [PEST Basics](part1/freyberg_pest_setup.md)  
* [Automated Calibration with PEST](part1/freyberg_k.md)  
* [Calibration with Two Parameters](part1/freyberg_k_and_r.md)  
* [Multiple Observation Types](part1/freyberg_k_r_fluxobs.md)  
* [GLM and the Objective Function Response Surface](part1/freyberg_glm_response_surface.md)  
* [Spatial Parameterisation with Pilot Points - setup](part1/freyberg_pilotpoints_1_setup.md)  
* [Spatial Parameterisation with Pilot Points - run](part1/freyberg_pilotpoints_2_run.md)  
* [Regularization](part1/intro_to_regularization.md)  
* [Intro to FOSM](part1/intro_to_fosm.md)  
* [Local Sensitivity and Identifiability](part1/freyberg_1_local_sensitivity.md)  
* [Global Sensitivity Analysis](part1/freyberg_2_global_sensitivity.md)  
* [Monte Carlo](part1/freyberg_monte_carlo.md)  

## Decision Support Modelling with pyEMU and PEST++  
* [Constructing a High-Dimensional PEST Interface with pyEMU](freyberg_pstfrom_pest_setup.md)  
* [Observation Values, Weights and Noise](freyberg_obs_and_weights.md)  
* [Prior Monte Carlo](freyberg_prior_monte_carlo.md)  
* [PEST++GLM - Calculating a Jacobian Matrix](freyberg_glm_1.md)  
* [FOSM and Data Worth](freyberg_fosm_and_dataworth.md)  
* [PEST++GLM  - Highly-Parameterized Regularized Inversion](freyberg_glm_2.md)  
* [PEST++IES - Basics](freyberg_ies_1_basics.md)  
* [PEST++IES - Localization](freyberg_ies_2_localization.md)  
* [PEST++DA - Getting Ready](freyberg_da_prep.md)  
* [PEST++DA - Sequential Data Assimilation](freyberg_da_run.md)  