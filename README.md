# Decision Support Modelling Notebooks

We have produced a series of tutorial notebooks to assist modellers in setting up and using model-partner software in ways that support the decision-support imperatives of data assimilation and uncertainty quantification. These tutorials provide an introduction to both the concepts and the practicalities of undertaking decision-support groundwater modelling with the PEST/PEST++ and pyEMU suites of software. Their aim is to provide examples of both “how to use” the software as well as “how to think” about using the software. 

## Notebook Contents

We have endeavoured to make these tutorials as accesible to as many people as possible. Workflows demonstrated herein are implemented programmaticaly in Python, employing functionality to interface with PEST/PEST++ available in pyEMU. However, concepts and the general approaches described are not limited to programmatic workflows. If you are interested in understanding how to implement pyEMU workflows, then you are encouraged to complete the jupyter notebooks yourself. If you just want to get a high-level understanding of decision-support modelling concepts and software, then you can simply read through the notebooks without having to run the code yourself. You can access complete version of the notebooks in your web browser [__here__](https://gmdsi.github.io/GMDSI_notebooks/). 

GMDSI has also produced a separate set of tutorials which demonstrate non-programmatic approaches to working with PEST/PEST++ available [here](https://gmdsi.org/education/tutorials/). 

## Pre-requisites (nice to have; not required)
 - Basic understanding of Python 
 - Basic understanding of Jupyter Notebooks
 - Basic understanding of MODFLOW 6

Familiarity with git would be a bonus, but not fundamental.

## Organization of the Tutorial Notebooks
The notebooks in the `tutorial` folder are organized in three main parts:
 - part 0: this is introductory background material that can be referred to for more detail on special topics. This includes the synthetic "freyberg" model used throughout the exercises, the `pyemu` python package that is used to interface with PEST++, and some of the mathematical concepts that come up throughout the tutorial notebooks. Note that the `part1_02_pest_setup/freyberg_pest_setup.ipynb` notebook provides valuable context that will help with understanding the `pyemu` introduction.
 - part 1: this part covers the Gauss-Levenberg Marquardt approach to parameter estimation and associated uncertainty analysis. This was the foundation of the PEST software for multiple decades and the theory continues to resonate through newer techniques
 - part 2: this part expands largely into ensemble-based parameter estimation and uncertainty analysis methods using PEST++. Advanced topics include management optimization and sequential data assimilation.

**To run the Part 0 notebooks**
The part 0 notebooks can be run in any order and do not depend on each other. We recommend, in particular, the first 3 topics as background for the rest of the tutorials, but the remaining part 0 notebooks stand alone for more theoretical basis and a reference for topics covered later.

**To run the Part 1 notebooks**
The part 1 notebooks can also be run in any order and do not depend on each other.

**To run the Part 2 notebooks**

If you are going to go through the Part2 notebooks, you will need to run them in the following order:
 1. part2_01_pstfrom_pest_setup/freyberg_pstfrom_pest_setup.ipynb
 2. part2_02_obs_and_weights/freyberg_obs_and_weights.ipynb

From here you can optionally run each of the following sequences:

Prior Monte Carlo:
 1. part2_03_prior_monte_carlo/freyberg_prior_monte_carlo.ipynb

GLM and data worth:
 1. part2_04_glm/freyberg_glm_1.ipynb
 2. part2_05_fosm_and_dataworth/freyberg_fosm_and_dataworth.ipynb
 3. part2_04_glm/freyberg_glm_2.ipynb

PEST++IES:
 1. part2_06_ies/freyberg_ies_1_basics.ipynb
 2. part2_06_ies/freyberg_ies_2_localization.ipynb

PEST++DA:
 1. part2_07_da/freyberg_da_prep.ipynb
 2. part2_07_da/freyberg_da_run.ipynb

PEST++OPT: 
 1. part2_08_opt/freyberg_opt_1.ipynb
 2. part2_08_opt/freyberg_opt_2.ipynb

PEST++MOU: 
 1. part2_09_mou/freyberg_mou_1.ipynb
 2. part2_09_mou/freyberg_mou_2.ipynb
 
IMPORTANT NOTE: When running some of these notebooks, PEST++ undertakes parallel model runs for computational efficiency. These runs are coordinated using internet communication protocols (e.g. TCP/IP). As a result, it is common (but not universal) that the first time a set of parallel runs is launched, a Windows Defender (or similar popup on other operating systems) box appears warning the user. This box can simply be closed by clicking "cancel" and then the model runs should proceed without issue.

## Installation Instructions

**Download the course repository:**

You can do this in one of two ways. 
 - (1) (easier) Download the repo as a zip file from here: [GMDSI_notebooks](https://github.com/gmdsi/GMDSI_notebooks). Unzip the folder and work from there.
 - (2) (recommended; requires familiarity with git). Install git following directions here: [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git). Sign-up for a git-hub account, then clone the repo [GMDSI_notebooks](https://github.com/gmdsi/GMDSI_notebooks).

**Install Python and dependencies:**
 - If you have already installed Python using Anaconda, you can skip this step. If not, install [Anaconda](https://www.anaconda.com/products/individual) (or [Miniconda](https://docs.conda.io/en/latest/miniconda.html), if you prefer )
 - If you are using __Windows__: go to the start menu and open "Anaconda prompt". An anaconda command line window will open. On __Linux__ or __MacOS__, just use the standard terminal. Navigate to the course repo folder on your machine. You can accomplish this by typing "cd *your folder path*" and pressing < enter >. Replace *your folder path* with the path to the course material folder on your computer.
 - Next, type `conda env create -f environment.yml`. This will create an anaconda environment called "gmdsitut" and install the python dependencies required for this course. It may take a while. Should you wish, you can inspect the *environment.yml* file in the repo folder to see what dependecies are being installed.

**Start jupyter notebook**
You will need to do this step any time you wish to open one of the course notebooks.
To start up the jupyter notebook:
- Windows: open the Anaconda prompt and type `conda activate gmdsitut`
- Mac/Linux: open a termainal and type `conda activate gmdsitut`
- Then navigate to folder where you downloaded the course materials repo and type `jupyter notebook`
A jupyter notebook instance should start within the course repo flder. Using the browser, you can now navigate to the "notebooks" folder and open one.
 
 # Acknowledgments
 
These tutorials were originally developed with support from the U.S Geological Survey (USGS) and support from USGS continues. Continued development and support is funded by the Groundwater Modelling Decision Support Initiative (GMDSI). GMDSI is jointly funded by BHP and Rio Tinto.

# Reference
To reference this material, please use the following:
Hugman, R., White, J.T., Fienen, M.N., and Hunt, R.J., 2022, Parameter Estimation, Uncertainty Analysis, and Optimization with the PEST++ Family of codes: Tutorial Jupyter Notebooks: U.S. Geological Survey data release, https://doi.org/10.5066/P901F9G0.
