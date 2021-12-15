# Decision Support Modelling Notebooks

The USGS and GMDSI have produced a series of tutorial notebooks to assist modellers in setting up and using model-partner software in ways that support the decision-support imperatives of data assimilation and uncertainty quantification. These tutorials provide an introduction to both the concepts and the practicalities of undertaking decision-support groundwater modelling with the PEST/PEST++ and pyEMU suites of software. Their aim is to provide examples of both “how to use” the software as well as “how to think” about using the software. 

We have endeavoured to make these tutorials as accesible to as many people as possible. Workflows demonstrated herein are implemented programmaticaly in Python, employing functionality to interface with PEST/PEST++ available in pyEMU. However, concepts and the general approaches described are not limited to programmatic workflows. If you are interested in understanding how to implement pyEMU workflows, then you are encouraged to complete the jupyter notebooks yourself. If you just want to get a high-level understanding of decision-support modelling conceps and software, then you can simply read through the notebooks wihtout having to run the code yourself. GMDSI has also produced a separate set of tutorials which demonstrate non-programmatic approaches to working with PEST/PEST++ (https://gmdsi.org/education/tutorials/). 

Tutorials are designed to be modular and independent of each other. Each tutorial addresses its own specific modelling topic. Hence there is no need to work through them in a pre-ordained sequence. However, they also complement each other. Many employ variations of the same synthetic model, and are based on the same simulator (MODFLOW 6). 

## Pre-requisites
 - Basic understanding of Python 
 - Basic understanding of Jupyter Notebooks
 - Basic understanding of MODFLOW 6
 - Basic understanding of common GIS file formats

Familiarity with git would be a bonus, but not fundamental.

## Installation Instructions

**Download the course repository:**

You can do this in one of two ways. 
 - (1) (easier) Download the repo as a zip file from here [https://github.com/rhugman/GMDSI_notebooks](https://github.com/rhugman/GMDSI_notebooks). Unzip the folder and work from there.
 - (2) (recommended; requires familiarity with git). Install git following directions here: [https://git-scm.com/book/en/v2/Getting-Started-Installing-Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git). Sign-up for a git-hub account, then clone the repo [https://github.com/rhugman/GMDSI_notebooks](https://github.com/rhugman/GMDSI_notebooks).

**Install Python and dependencies:**
 - If you have already installed Python using Anaconda, you can skip this step. If not, install Anaconda https://www.anaconda.com/products/individual (or Miniconda, if you prefer https://docs.conda.io/en/latest/miniconda.html)
 - If you are using Windows: go to the start menu and open "Anaconda prompt". An anaconda command lline window will open. Navigate to the course repo folder on your machine. You can accomplish this by typing "cd *your folder path*" and pressing <enter>. Replace *your folder path* with the  path to the course material repo folder on your computer.
 - Next, type "conda env create -f environment.yml". This will create an anaconda environment called "gmdsituts" and install the python dependencies required for this course. It may take a while. Should you wish, you can inspect the *environment.yml* file in the repo folder to see what dependecies are being installed.

**Start jupyter notebook**
You will need to do this step any time you wish to open one of the course notebooks.
 - In Windows, open the Anaconda prompt. In Mac/Linux, open a terminal. Then, type "conda activate gmdsituts"
 - Next, in the Anaconda prompt or terminal, navigate to the course materials reposiotry folder and type "jupyter notebook". A jupyter notebook instance should start within the course repo flder. Using the browser, you can now navigate to the "notebooks" folder and open one.
