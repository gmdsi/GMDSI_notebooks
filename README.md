# (DRAFT) Decision Support Modelling Notebooks

### ***These tutorials/notebooks are still under development.***

We have produced a series of tutorial notebooks to assist modellers in setting up and using model-partner software in ways that support the decision-support imperatives of data assimilation and uncertainty quantification. These tutorials provide an introduction to both the concepts and the practicalities of undertaking decision-support groundwater modelling with the PEST/PEST++ and pyEMU suites of software. Their aim is to provide examples of both “how to use” the software as well as “how to think” about using the software. 

We have endeavoured to make these tutorials as accesible to as many people as possible. Workflows demonstrated herein are implemented programmaticaly in Python, employing functionality to interface with PEST/PEST++ available in pyEMU. However, concepts and the general approaches described are not limited to programmatic workflows. If you are interested in understanding how to implement pyEMU workflows, then you are encouraged to complete the jupyter notebooks yourself. If you just want to get a high-level understanding of decision-support modelling concepts and software, then you can simply read through the notebooks without having to run the code yourself. You can access complete version of the notebooks in your web browser [__here__](https://gmdsi.github.io/GMDSI_notebooks/). 

GMDSI has also produced a separate set of tutorials which demonstrate non-programmatic approaches to working with PEST/PEST++ available [here](https://gmdsi.org/education/tutorials/). 

## Pre-requisites (nice to have; not required)
 - Basic understanding of Python 
 - Basic understanding of Jupyter Notebooks
 - Basic understanding of MODFLOW 6

Familiarity with git would be a bonus, but not fundamental.

## Installation Instructions

**Download the course repository:**

You can do this in one of two ways. 
 - (1) (easier) Download the repo as a zip file from here: [GMDSI_notebooks](https://github.com/rhugman/GMDSI_notebooks). Unzip the folder and work from there.
 - (2) (recommended; requires familiarity with git). Install git following directions here: [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git). Sign-up for a git-hub account, then clone the repo [GMDSI_notebooks](https://github.com/rhugman/GMDSI_notebooks).

**Install Python and dependencies:**
 - If you have already installed Python using Anaconda, you can skip this step. If not, install [Anaconda](https://www.anaconda.com/products/individual) (or [Miniconda](https://docs.conda.io/en/latest/miniconda.html), if you prefer )
 - If you are using __Windows__: go to the start menu and open "Anaconda prompt". An anaconda command line window will open. On __Linux__ or __MacOS__, just use the stndard terminal. Navigate to the course repo folder on your machine. You can accomplish this by typing "cd *your folder path*" and pressing < enter >. Replace *your folder path* with the path to the course material folder on your computer.
 - Next, type `conda env create -f environment.yml`. This will create an anaconda environment called "gmdsitut" and install the python dependencies required for this course. It may take a while. Should you wish, you can inspect the *environment.yml* file in the repo folder to see what dependecies are being installed.

**Start jupyter notebook**
You will need to do this step any time you wish to open one of the course notebooks.
To start up the jupyter notebook:
- Windows: open the Anaconda prompt and type `conda activate pyclass`
- Mac/Linux: open a termainal and type `conda activate pyclass`
- Then navigate to folder where you downloaded the course materials repo and type `jupyter notebook`
A jupyter notebook instance should start within the course repo flder. Using the browser, you can now navigate to the "notebooks" folder and open one.

**Before starting Part 2**

If you are going to go through the Part2 notebooks, you will need to run them in the following order:
 1. freyberg_pstfrom_pest_setup.ipynb
 2. freyberg_obs_and_weights.ipynb

From here you can optionally run each of the following sequences:

Prior Monte Carlo:
 1. freyberg_prior_monte_carlo.ipynb

GLM and data worth:
 1. freyberg_glm_1.ipynb
 2. freyberg_fosm_and_dataworth.ipynb
 3. freyberg_glm_2.ipynb

PEST++IES:
 1. freyberg_ies_1_basics.ipynb
 2. freyberg_ies_2_localization.ipynb
 3. freyberg_ies_3_tot_error_cov.ipynb

PEST++DA:
 1. freyberg_da_prep.ipynb
 2. freyberg_da_run.ipynb
