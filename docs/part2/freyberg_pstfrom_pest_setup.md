---
layout: default
title: Constructing a High-Dimensional PEST Interface with pyEMU
parent: Decision Support Modelling with pyEMU and PEST++
nav_order: 1
---

# Setup the PEST(++) interface around the modified Freyberg model

In this notebook, we will construct a complex model independent (non-intrusive) interface around an existing `MODFLOW6` model using `pyEMU`. We assume that the reader is at least partially familiar with PEST(++) file formats and working philosophy. 

The modified Freyberg groundwater flow model has been constructed and is described in a previous notebook from this series. We will construct the entire PEST(++) interface from scratch here. This setup will be built upon in subsequent tutorials. 

We will rely heavily on the `pyemu.PstFrom` class. Although here we employ it with a `MODFLOW6` model, `PstFrom` is designed to be general and software independent (mostly). Some features are only available for `MODFLOW` models (e.g. `SpatialReference`).

The `PstFrom` class automates the construction of high-dimensional PEST(++) interfaces with all the bells and whistles. It provides easy-to-use functions to process model input and output files into PEST(++) datasets. It can assist with setting up spatio-temporaly varying parameters. It handles the generation of geostatisical prior covariance matrices and ensembles. It automates writting a "model run" script. It provides tools to add custom pre- and post-processing functions to this script. It makes adding tweaks and fixes to the PEST(++) interface a breeze. All of this from the comfort of your favourite Python IDE.

During this tutorial we are going to construct a PEST dataset. Amongst other things, we will demonstrate:
 - how to add observations & parameters from model output & input files;
 - how to add pre- and post-processing functions to the "model run" script;
 - how to generate geostatistical structures for spatialy and temporally correlated parameters;
 - how to edit parameter/observation data sections;
 - how to generate a prior parameter covariance matrix and prior parameter ensemble;



First, let's get our model files and sort out some admin.


### 1. Admin & Organize Folders
First some admin. Load the dependencies and organize model folders. 


```python
import sys
import os
import shutil
import platform
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt;


sys.path.append(os.path.join("..", "..", "dependencies"))
import pyemu
import flopy

sys.path.append("..")
import herebedragons as hbd
```

We will be calling a few external programs throughout this tutorial. Namely, MODFLOW 6 and PESTPP-GLM. For the purposes of the tutorial(s), we have included executables in the tutorial repository. They are in the `bin` folder, organized by operating system and will programmatically copied into the working dirs as needed. 

Some may prefer that executables be located in a folder that is cited in your computer’s PATH environment variable. Doing so allows you to run them from a command prompt open to any other folder without having to include the full path to these executables in the command to run them. 

However, in situations where someone has several active projects and each may use difference versions of compiled binary codes, this may not be practical. In such cases, we can simply place the executables in the folder from which they will be executed.  So, let's copy the necessary executables into our working folder using a simple helper function:


Let's copy the original model folder into a new working directory, just to ensure we don't mess up the base files.


```python
# folder containing original model files
org_d = os.path.join('..', '..', 'models', 'freyberg_mf6')

# a dir to hold a copy of the org model files
tmp_d = os.path.join('freyberg_mf6')

if os.path.exists(tmp_d):
    shutil.rmtree(tmp_d)
shutil.copytree(org_d,tmp_d)

# get executables
hbd.prep_bins(tmp_d)
# get dependency folders
hbd.prep_deps(tmp_d)
```

If you inspect the model folder, you will see that all the `MODFLOW6` model files have been written "externally". This is key for working with the `PstFrom` class (or with PEST(++) in general, really). Essentialy, all pertinent model inputs have been written as independent files in either array or list format. This makes it easier for us to programiatically access and re-write the values in these files.

Array files contain a data type (usually floating points). List files will have a few columns that contain index information and then columns of floating point values (they have a tabular format; think `.csv` files or DataFrames). The `PstFrom` class provides methods for processing these file types into a PEST(++) dataset. 




```python
os.listdir(tmp_d)
```




    ['flopy',
     'freyberg6.dis',
     'freyberg6.dis.grb',
     'freyberg6.dis_botm_layer1.txt',
     'freyberg6.dis_botm_layer2.txt',
     'freyberg6.dis_botm_layer3.txt',
     'freyberg6.dis_delc.txt',
     'freyberg6.dis_delr.txt',
     'freyberg6.dis_idomain_layer1.txt',
     'freyberg6.dis_idomain_layer2.txt',
     'freyberg6.dis_idomain_layer3.txt',
     'freyberg6.dis_top.txt',
     'freyberg6.ghb',
     'freyberg6.ghb_stress_period_data_1.txt',
     'freyberg6.ic',
     'freyberg6.ic_strt_layer1.txt',
     'freyberg6.ic_strt_layer2.txt',
     'freyberg6.ic_strt_layer3.txt',
     'freyberg6.ims',
     'freyberg6.lst',
     'freyberg6.nam',
     'freyberg6.npf',
     'freyberg6.npf_icelltype_layer1.txt',
     'freyberg6.npf_icelltype_layer2.txt',
     'freyberg6.npf_icelltype_layer3.txt',
     'freyberg6.npf_k33_layer1.txt',
     'freyberg6.npf_k33_layer2.txt',
     'freyberg6.npf_k33_layer3.txt',
     'freyberg6.npf_k_layer1.txt',
     'freyberg6.npf_k_layer2.txt',
     'freyberg6.npf_k_layer3.txt',
     'freyberg6.oc',
     'freyberg6.rch',
     'freyberg6.rch_recharge_1.txt',
     'freyberg6.rch_recharge_10.txt',
     'freyberg6.rch_recharge_11.txt',
     'freyberg6.rch_recharge_12.txt',
     'freyberg6.rch_recharge_13.txt',
     'freyberg6.rch_recharge_14.txt',
     'freyberg6.rch_recharge_15.txt',
     'freyberg6.rch_recharge_16.txt',
     'freyberg6.rch_recharge_17.txt',
     'freyberg6.rch_recharge_18.txt',
     'freyberg6.rch_recharge_19.txt',
     'freyberg6.rch_recharge_2.txt',
     'freyberg6.rch_recharge_20.txt',
     'freyberg6.rch_recharge_21.txt',
     'freyberg6.rch_recharge_22.txt',
     'freyberg6.rch_recharge_23.txt',
     'freyberg6.rch_recharge_24.txt',
     'freyberg6.rch_recharge_25.txt',
     'freyberg6.rch_recharge_3.txt',
     'freyberg6.rch_recharge_4.txt',
     'freyberg6.rch_recharge_5.txt',
     'freyberg6.rch_recharge_6.txt',
     'freyberg6.rch_recharge_7.txt',
     'freyberg6.rch_recharge_8.txt',
     'freyberg6.rch_recharge_9.txt',
     'freyberg6.sfr',
     'freyberg6.sfr_connectiondata.txt',
     'freyberg6.sfr_packagedata.txt',
     'freyberg6.sfr_perioddata_1.txt',
     'freyberg6.sfr_perioddata_10.txt',
     'freyberg6.sfr_perioddata_11.txt',
     'freyberg6.sfr_perioddata_12.txt',
     'freyberg6.sfr_perioddata_13.txt',
     'freyberg6.sfr_perioddata_14.txt',
     'freyberg6.sfr_perioddata_15.txt',
     'freyberg6.sfr_perioddata_16.txt',
     'freyberg6.sfr_perioddata_17.txt',
     'freyberg6.sfr_perioddata_18.txt',
     'freyberg6.sfr_perioddata_19.txt',
     'freyberg6.sfr_perioddata_2.txt',
     'freyberg6.sfr_perioddata_20.txt',
     'freyberg6.sfr_perioddata_21.txt',
     'freyberg6.sfr_perioddata_22.txt',
     'freyberg6.sfr_perioddata_23.txt',
     'freyberg6.sfr_perioddata_24.txt',
     'freyberg6.sfr_perioddata_25.txt',
     'freyberg6.sfr_perioddata_3.txt',
     'freyberg6.sfr_perioddata_4.txt',
     'freyberg6.sfr_perioddata_5.txt',
     'freyberg6.sfr_perioddata_6.txt',
     'freyberg6.sfr_perioddata_7.txt',
     'freyberg6.sfr_perioddata_8.txt',
     'freyberg6.sfr_perioddata_9.txt',
     'freyberg6.sto',
     'freyberg6.sto_iconvert_layer1.txt',
     'freyberg6.sto_iconvert_layer2.txt',
     'freyberg6.sto_iconvert_layer3.txt',
     'freyberg6.sto_ss_layer1.txt',
     'freyberg6.sto_ss_layer2.txt',
     'freyberg6.sto_ss_layer3.txt',
     'freyberg6.sto_sy_layer1.txt',
     'freyberg6.sto_sy_layer2.txt',
     'freyberg6.sto_sy_layer3.txt',
     'freyberg6.tdis',
     'freyberg6.wel',
     'freyberg6.wel_stress_period_data_1.txt',
     'freyberg6.wel_stress_period_data_10.txt',
     'freyberg6.wel_stress_period_data_11.txt',
     'freyberg6.wel_stress_period_data_12.txt',
     'freyberg6.wel_stress_period_data_13.txt',
     'freyberg6.wel_stress_period_data_14.txt',
     'freyberg6.wel_stress_period_data_15.txt',
     'freyberg6.wel_stress_period_data_16.txt',
     'freyberg6.wel_stress_period_data_17.txt',
     'freyberg6.wel_stress_period_data_18.txt',
     'freyberg6.wel_stress_period_data_19.txt',
     'freyberg6.wel_stress_period_data_2.txt',
     'freyberg6.wel_stress_period_data_20.txt',
     'freyberg6.wel_stress_period_data_21.txt',
     'freyberg6.wel_stress_period_data_22.txt',
     'freyberg6.wel_stress_period_data_23.txt',
     'freyberg6.wel_stress_period_data_24.txt',
     'freyberg6.wel_stress_period_data_25.txt',
     'freyberg6.wel_stress_period_data_3.txt',
     'freyberg6.wel_stress_period_data_4.txt',
     'freyberg6.wel_stress_period_data_5.txt',
     'freyberg6.wel_stress_period_data_6.txt',
     'freyberg6.wel_stress_period_data_7.txt',
     'freyberg6.wel_stress_period_data_8.txt',
     'freyberg6.wel_stress_period_data_9.txt',
     'freyberg6_freyberg.cbc',
     'freyberg6_freyberg.hds',
     'freyberg_mp.mpbas',
     'freyberg_mp.mpend',
     'freyberg_mp.mplst',
     'freyberg_mp.mpnam',
     'freyberg_mp.mpsim',
     'freyberg_mp.ne_layer1.txt',
     'freyberg_mp.ne_layer2.txt',
     'freyberg_mp.ne_layer3.txt',
     'head.obs',
     'heads.csv',
     'inschek.exe',
     'mf5to6.exe',
     'mf6.exe',
     'mfsim.lst',
     'mfsim.nam',
     'mp7.exe',
     'pestchek.exe',
     'pestpp-da.exe',
     'pestpp-glm.exe',
     'pestpp-ies.exe',
     'pestpp-mou.exe',
     'pestpp-opt.exe',
     'pestpp-sen.exe',
     'pestpp-sqp.exe',
     'pestpp-swp.exe',
     'pm.pg1.sloc',
     'pyemu',
     'sfr.csv',
     'sfr.obs',
     'tempchek.exe',
     'zbud6.exe']



Now we need just a tiny bit of info about the spatial discretization of the model - this is needed to work out separation distances between parameters to build a geostatistical prior covariance matrix later.

Here we will load the flopy sim and model instance just to help us define some quantities later - flopy is ***not required*** to use the `PstFrom` class. ***Neither is MODFLOW***. However, at the time of writting, support for `SpatialReference` to spatially locate parameters is limited to structured grid models.

Load the simulation. Run it once to make sure it works and to ***make sure that model output files are in the folder***. 


```python
# load simulation
sim = flopy.mf6.MFSimulation.load(sim_ws=tmp_d)
# load flow model
gwf = sim.get_model()

# run the model once to make sure it works
pyemu.os_utils.run("mf6",cwd=tmp_d)
# run modpath7
pyemu.os_utils.run(r'mp7 freyberg_mp.mpsim', cwd=tmp_d)
```

    loading simulation...
      loading simulation name file...
      loading tdis package...
      loading model gwf6...
        loading package dis...
        loading package ic...
        loading package npf...
        loading package sto...
        loading package oc...
        loading package wel...
        loading package rch...
        loading package ghb...
        loading package sfr...
        loading package obs...
      loading ims package freyberg6...
    

### 2. Spatial Reference
Now we can instantiate a `SpatialReference`. This will later be passed to `PstFrom` to assist with spatially locating parameters (e.g. pilot points and/or cell-by-cell parameters).  You can also use the flopy `modelgrid` class instance that is attached to the simulation, but `SpatialReference` is cleaner and faster for structured grids...


```python
sr = pyemu.helpers.SpatialReference.from_namfile(
        os.path.join(tmp_d, "freyberg6.nam"),
        delr=gwf.dis.delr.array, delc=gwf.dis.delc.array)
sr
```

       could not remove start_datetime
    




    xul:0; yul:10000; rotation:0; proj4_str:None; units:meters; lenuni:2; length_multiplier:1.0



### 3. Instantiate PstFrom

Now we can start to construct the PEST(++) interface by instantiating a `PstFrom` class instance. There are a few things that we need to specify up front:

 - the folder in which we currently have model files (e.g. `tmp_d`). PstFrom will copy all the files from this directory into a new "template" folder.
 - **template folder**: this is a folder in which the PEST dataset will be constructed - this folder will hold the model files plus all of the files needed to run PEST(++). This folder/dataset will form the template for subsequent deployment of PEST(++).
 - **longnames**: for backwards compatibility with PEST and PEST_HP (i.e. non-PEST++ versions), which have upper limits to parameter/obsveration names (PEST++ does not). Setting this value to False is only recommended if required. 
 - Whether the model is `zero based` or not.
 - (optional) the **spatial reference**, as previously discussed. This is only requried if using `pyEMU` to define parameter spatial correlation. Alternatively, you can define these yourself or use utilities available in the PEST-suite. 




```python
# specify a template directory (i.e. the PstFrom working folder)
template_ws = os.path.join("freyberg6_template")
start_datetime="1-1-2008"
# instantiate PstFrom
pf = pyemu.utils.PstFrom(original_d=tmp_d, # where the model is stored
                            new_d=template_ws, # the PEST template folder
                            remove_existing=True, # ensures a clean start
                            longnames=True, # set False if using PEST/PEST_HP
                            spatial_reference=sr, #the spatial reference we generated earlier
                            zero_based=False, # does the MODEL use zero based indices? For example, MODFLOW does NOT
                            start_datetime=start_datetime, # required when specifying temporal correlation between parameters
                            echo=False) # to stop PstFrom from writting lots of infromation to the notebook; experiment by setting it as True to see the difference; usefull for troubleshooting
```


```python
os.listdir(template_ws)
```




    ['flopy',
     'freyberg6.dis',
     'freyberg6.dis.grb',
     'freyberg6.dis_botm_layer1.txt',
     'freyberg6.dis_botm_layer2.txt',
     'freyberg6.dis_botm_layer3.txt',
     'freyberg6.dis_delc.txt',
     'freyberg6.dis_delr.txt',
     'freyberg6.dis_idomain_layer1.txt',
     'freyberg6.dis_idomain_layer2.txt',
     'freyberg6.dis_idomain_layer3.txt',
     'freyberg6.dis_top.txt',
     'freyberg6.ghb',
     'freyberg6.ghb_stress_period_data_1.txt',
     'freyberg6.ic',
     'freyberg6.ic_strt_layer1.txt',
     'freyberg6.ic_strt_layer2.txt',
     'freyberg6.ic_strt_layer3.txt',
     'freyberg6.ims',
     'freyberg6.lst',
     'freyberg6.nam',
     'freyberg6.npf',
     'freyberg6.npf_icelltype_layer1.txt',
     'freyberg6.npf_icelltype_layer2.txt',
     'freyberg6.npf_icelltype_layer3.txt',
     'freyberg6.npf_k33_layer1.txt',
     'freyberg6.npf_k33_layer2.txt',
     'freyberg6.npf_k33_layer3.txt',
     'freyberg6.npf_k_layer1.txt',
     'freyberg6.npf_k_layer2.txt',
     'freyberg6.npf_k_layer3.txt',
     'freyberg6.oc',
     'freyberg6.rch',
     'freyberg6.rch_recharge_1.txt',
     'freyberg6.rch_recharge_10.txt',
     'freyberg6.rch_recharge_11.txt',
     'freyberg6.rch_recharge_12.txt',
     'freyberg6.rch_recharge_13.txt',
     'freyberg6.rch_recharge_14.txt',
     'freyberg6.rch_recharge_15.txt',
     'freyberg6.rch_recharge_16.txt',
     'freyberg6.rch_recharge_17.txt',
     'freyberg6.rch_recharge_18.txt',
     'freyberg6.rch_recharge_19.txt',
     'freyberg6.rch_recharge_2.txt',
     'freyberg6.rch_recharge_20.txt',
     'freyberg6.rch_recharge_21.txt',
     'freyberg6.rch_recharge_22.txt',
     'freyberg6.rch_recharge_23.txt',
     'freyberg6.rch_recharge_24.txt',
     'freyberg6.rch_recharge_25.txt',
     'freyberg6.rch_recharge_3.txt',
     'freyberg6.rch_recharge_4.txt',
     'freyberg6.rch_recharge_5.txt',
     'freyberg6.rch_recharge_6.txt',
     'freyberg6.rch_recharge_7.txt',
     'freyberg6.rch_recharge_8.txt',
     'freyberg6.rch_recharge_9.txt',
     'freyberg6.sfr',
     'freyberg6.sfr_connectiondata.txt',
     'freyberg6.sfr_packagedata.txt',
     'freyberg6.sfr_perioddata_1.txt',
     'freyberg6.sfr_perioddata_10.txt',
     'freyberg6.sfr_perioddata_11.txt',
     'freyberg6.sfr_perioddata_12.txt',
     'freyberg6.sfr_perioddata_13.txt',
     'freyberg6.sfr_perioddata_14.txt',
     'freyberg6.sfr_perioddata_15.txt',
     'freyberg6.sfr_perioddata_16.txt',
     'freyberg6.sfr_perioddata_17.txt',
     'freyberg6.sfr_perioddata_18.txt',
     'freyberg6.sfr_perioddata_19.txt',
     'freyberg6.sfr_perioddata_2.txt',
     'freyberg6.sfr_perioddata_20.txt',
     'freyberg6.sfr_perioddata_21.txt',
     'freyberg6.sfr_perioddata_22.txt',
     'freyberg6.sfr_perioddata_23.txt',
     'freyberg6.sfr_perioddata_24.txt',
     'freyberg6.sfr_perioddata_25.txt',
     'freyberg6.sfr_perioddata_3.txt',
     'freyberg6.sfr_perioddata_4.txt',
     'freyberg6.sfr_perioddata_5.txt',
     'freyberg6.sfr_perioddata_6.txt',
     'freyberg6.sfr_perioddata_7.txt',
     'freyberg6.sfr_perioddata_8.txt',
     'freyberg6.sfr_perioddata_9.txt',
     'freyberg6.sto',
     'freyberg6.sto_iconvert_layer1.txt',
     'freyberg6.sto_iconvert_layer2.txt',
     'freyberg6.sto_iconvert_layer3.txt',
     'freyberg6.sto_ss_layer1.txt',
     'freyberg6.sto_ss_layer2.txt',
     'freyberg6.sto_ss_layer3.txt',
     'freyberg6.sto_sy_layer1.txt',
     'freyberg6.sto_sy_layer2.txt',
     'freyberg6.sto_sy_layer3.txt',
     'freyberg6.tdis',
     'freyberg6.wel',
     'freyberg6.wel_stress_period_data_1.txt',
     'freyberg6.wel_stress_period_data_10.txt',
     'freyberg6.wel_stress_period_data_11.txt',
     'freyberg6.wel_stress_period_data_12.txt',
     'freyberg6.wel_stress_period_data_13.txt',
     'freyberg6.wel_stress_period_data_14.txt',
     'freyberg6.wel_stress_period_data_15.txt',
     'freyberg6.wel_stress_period_data_16.txt',
     'freyberg6.wel_stress_period_data_17.txt',
     'freyberg6.wel_stress_period_data_18.txt',
     'freyberg6.wel_stress_period_data_19.txt',
     'freyberg6.wel_stress_period_data_2.txt',
     'freyberg6.wel_stress_period_data_20.txt',
     'freyberg6.wel_stress_period_data_21.txt',
     'freyberg6.wel_stress_period_data_22.txt',
     'freyberg6.wel_stress_period_data_23.txt',
     'freyberg6.wel_stress_period_data_24.txt',
     'freyberg6.wel_stress_period_data_25.txt',
     'freyberg6.wel_stress_period_data_3.txt',
     'freyberg6.wel_stress_period_data_4.txt',
     'freyberg6.wel_stress_period_data_5.txt',
     'freyberg6.wel_stress_period_data_6.txt',
     'freyberg6.wel_stress_period_data_7.txt',
     'freyberg6.wel_stress_period_data_8.txt',
     'freyberg6.wel_stress_period_data_9.txt',
     'freyberg6_freyberg.cbc',
     'freyberg6_freyberg.hds',
     'freyberg_mp.mpbas',
     'freyberg_mp.mpend',
     'freyberg_mp.mplst',
     'freyberg_mp.mpnam',
     'freyberg_mp.mpsim',
     'freyberg_mp.ne_layer1.txt',
     'freyberg_mp.ne_layer2.txt',
     'freyberg_mp.ne_layer3.txt',
     'head.obs',
     'heads.csv',
     'inschek.exe',
     'mf5to6.exe',
     'mf6.exe',
     'mfsim.lst',
     'mfsim.nam',
     'mp7.exe',
     'mpath7.log',
     'mult',
     'org',
     'pestchek.exe',
     'pestpp-da.exe',
     'pestpp-glm.exe',
     'pestpp-ies.exe',
     'pestpp-mou.exe',
     'pestpp-opt.exe',
     'pestpp-sen.exe',
     'pestpp-sqp.exe',
     'pestpp-swp.exe',
     'pm.pg1.sloc',
     'pyemu',
     'sfr.csv',
     'sfr.obs',
     'tempchek.exe',
     'zbud6.exe']



So we see that when `PstFrom` is instantiated, it starts by copying the `original_d` to the `new_d`.  sweet as!

### 4. Observations

We now have a `PstFrom` instance assigned to the variable `pf`. For now it is only an empty container to which we can start adding "observations", "parameters" and other bits and bobs.

Lets start with observations because they are easier. `MODFLOW6` makes life even easier by recording observations in nicely organized .csv files. Isn't that a peach!

#### 4.1 Freyberg Recap
As you may recall from the "*intro to Freyberg*" tutorial, the model is configured to record time series of head at observation wells, and flux at three locations along the river. These are recorded in external .csv files named `heads.csv` and `sfr.csv`, respectively. You should be able to see these files in the model folder.

Recall that each .csv houses records of observation time-series. Outputs are recorded for each simulated stress-period. The model starts with a single steady-state stress-period, followed by 24 monthly transient stress-periods. The steady-state and first 12 transient stress-periods simulate the history-matching period. The last 12 transient stress periods simulate future conditions (i.e. the prediction period).


```python
# check the output csv file names
for i in gwf.obs:
    print(i.output.obs_names)
```

    ['sfr.csv']
    ['heads.csv']
    

Let's start with the 'heads.csv' file. First load it as a DataFrame to take a look:


```python
df = pd.read_csv(os.path.join(template_ws,"heads.csv"),index_col=0)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TRGW-2-2-15</th>
      <th>TRGW-2-2-9</th>
      <th>TRGW-2-3-8</th>
      <th>TRGW-2-9-1</th>
      <th>TRGW-2-13-10</th>
      <th>TRGW-2-15-16</th>
      <th>TRGW-2-21-10</th>
      <th>TRGW-2-22-15</th>
      <th>TRGW-2-24-4</th>
      <th>TRGW-2-26-6</th>
      <th>...</th>
      <th>TRGW-0-9-1</th>
      <th>TRGW-0-13-10</th>
      <th>TRGW-0-15-16</th>
      <th>TRGW-0-21-10</th>
      <th>TRGW-0-22-15</th>
      <th>TRGW-0-24-4</th>
      <th>TRGW-0-26-6</th>
      <th>TRGW-0-29-15</th>
      <th>TRGW-0-33-7</th>
      <th>TRGW-0-34-10</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3652.5</th>
      <td>34.399753</td>
      <td>34.692968</td>
      <td>34.730206</td>
      <td>35.065796</td>
      <td>34.320937</td>
      <td>34.180708</td>
      <td>34.200684</td>
      <td>34.065937</td>
      <td>34.406587</td>
      <td>34.245567</td>
      <td>...</td>
      <td>35.073084</td>
      <td>34.326872</td>
      <td>34.186027</td>
      <td>34.206589</td>
      <td>34.025500</td>
      <td>34.412947</td>
      <td>34.251605</td>
      <td>33.937893</td>
      <td>34.031870</td>
      <td>33.924572</td>
    </tr>
    <tr>
      <th>3683.5</th>
      <td>34.474597</td>
      <td>34.773873</td>
      <td>34.811043</td>
      <td>35.130920</td>
      <td>34.435795</td>
      <td>34.284363</td>
      <td>34.320775</td>
      <td>34.171539</td>
      <td>34.517134</td>
      <td>34.376046</td>
      <td>...</td>
      <td>35.137941</td>
      <td>34.440950</td>
      <td>34.289268</td>
      <td>34.325758</td>
      <td>34.116346</td>
      <td>34.522520</td>
      <td>34.380738</td>
      <td>34.017081</td>
      <td>34.142435</td>
      <td>34.035987</td>
    </tr>
    <tr>
      <th>3712.5</th>
      <td>34.540686</td>
      <td>34.856926</td>
      <td>34.895437</td>
      <td>35.214427</td>
      <td>34.528656</td>
      <td>34.363909</td>
      <td>34.418080</td>
      <td>34.248171</td>
      <td>34.622236</td>
      <td>34.480890</td>
      <td>...</td>
      <td>35.221267</td>
      <td>34.534811</td>
      <td>34.369785</td>
      <td>34.424073</td>
      <td>34.183016</td>
      <td>34.628106</td>
      <td>34.486650</td>
      <td>34.079395</td>
      <td>34.230419</td>
      <td>34.112732</td>
    </tr>
    <tr>
      <th>3743.5</th>
      <td>34.578877</td>
      <td>34.913862</td>
      <td>34.954331</td>
      <td>35.285693</td>
      <td>34.575597</td>
      <td>34.400771</td>
      <td>34.466995</td>
      <td>34.282155</td>
      <td>34.687775</td>
      <td>34.535632</td>
      <td>...</td>
      <td>35.292600</td>
      <td>34.582875</td>
      <td>34.407581</td>
      <td>34.474190</td>
      <td>34.212731</td>
      <td>34.694595</td>
      <td>34.542664</td>
      <td>34.108744</td>
      <td>34.273212</td>
      <td>34.143653</td>
    </tr>
    <tr>
      <th>3773.5</th>
      <td>34.570039</td>
      <td>34.914817</td>
      <td>34.956630</td>
      <td>35.304308</td>
      <td>34.553737</td>
      <td>34.377572</td>
      <td>34.444166</td>
      <td>34.257522</td>
      <td>34.680793</td>
      <td>34.513612</td>
      <td>...</td>
      <td>35.311425</td>
      <td>34.561764</td>
      <td>34.384903</td>
      <td>34.452207</td>
      <td>34.191712</td>
      <td>34.688486</td>
      <td>34.521646</td>
      <td>34.091720</td>
      <td>34.250685</td>
      <td>34.114481</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>



As you can see, there are many columns, one for each observation site. Conveniently, * *cough* * they are named according to the cell layer, row and column. Note that at every site, there is an observation in both the top and bottom layer (0_ and 2_). We will make use of this later to create "secondary observations" of head differences between layers...but let's not get distracted.

The values in the *.csv* file were generated by running the model. (***IMPORTANT!***) However, `PstFrom` assumes that values in this file are the *target* observation values, and they will be used to populate the PEST(++) dataset.  This lets the user quickly verify that the `PstFrom` process reproduces the same model output files - an important thing to test!

Now, you can and should change the observation values later on for the quantities that correspond to actual observation data.  This is the standard workflow when using `PstFrom` because it allows users to separate the PEST interface setup from the always-important process of setting observation values and weights. We address this part of the workflow in a separate tutorial.

#### 4.1. Adding Observations

First, we will use the `PstFrom.add_observations()` method to add observations to our `pf` object. This method can use ***list-type*** files, where the data are organized in column/tabular format with one or more index columns and one or more data columns.  This method can also use ***array-type*** files, where the data are organized in a 2-D array structure (we will see this one later...)

We are going to tell `pf` which columns of this file contain observations. Values in these columns will be assigned to *observation values*.

We can also inform it if there is an index column (or columns). Values in this column will be included in the *observation names*. 

We could also specify which rows to include as observations. But observations are free...so why not keep them all! 

Let's add observations from `heads.csv`. The first column of this file records the time at which the value is simulated. Let's use that as the index column (this becomes useful later on to post-process results). We want all other columns as observation values.



```python
hds_df = pf.add_observations("heads.csv", # the model output file to read
                            insfile="heads.csv.ins", #optional, the instruction file name
                            index_cols="time", #column header to use as index; can also use column number (zero-based) instead of the header name
                            use_cols=list(df.columns.values), #names of columns that include observation values; can also use column number (zero-based) instead of the header name
                            prefix="hds") #prefix to all observation names; choose something logical and easy o find. We use it later on to select obsevrations
```

Let's inspect what we just created. 

We can see that the `.add_observations()` method returned a dataframe with lots of useful info: 

 - the observation names that were formed (see `obsnme` column); note that these inlcude lots of usefull metadata like the column name, index value and so on;
 - the values that were read from `heads.csv` (see `obsval` column); 
 - some generic weights and group names; note that observations are grouped according to the column of the model output .csv. Alternatively, we could have specified a list of observation group names.


```python
hds_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>obsnme</th>
      <th>obsval</th>
      <th>weight</th>
      <th>obgnme</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3652.5</th>
      <td>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3652.5</td>
      <td>34.326872</td>
      <td>1.0</td>
      <td>oname:hds_otype:lst_usecol:trgw-0-13-10</td>
    </tr>
    <tr>
      <th>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3683.5</th>
      <td>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3683.5</td>
      <td>34.440950</td>
      <td>1.0</td>
      <td>oname:hds_otype:lst_usecol:trgw-0-13-10</td>
    </tr>
    <tr>
      <th>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3712.5</th>
      <td>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3712.5</td>
      <td>34.534811</td>
      <td>1.0</td>
      <td>oname:hds_otype:lst_usecol:trgw-0-13-10</td>
    </tr>
    <tr>
      <th>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3743.5</th>
      <td>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3743.5</td>
      <td>34.582875</td>
      <td>1.0</td>
      <td>oname:hds_otype:lst_usecol:trgw-0-13-10</td>
    </tr>
    <tr>
      <th>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3773.5</th>
      <td>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3773.5</td>
      <td>34.561764</td>
      <td>1.0</td>
      <td>oname:hds_otype:lst_usecol:trgw-0-13-10</td>
    </tr>
  </tbody>
</table>
</div>



At this point, no PEST *control file* has been created, we have simply prepared to add these observations to the control file later. Everything is still only stored in memory. However, a PEST *instruction* file has been created in the template folder (`template_ws`):


```python
[f for f in os.listdir(template_ws) if f.endswith(".ins")]
```




    ['heads.csv.ins']



Blimey, wasn't that easy? Automatically monitoring thousands of model output quantities as observations into a PEST dataset becomes a breeze!

Let's quickly do the same thing for the SFR observations.


```python
df = pd.read_csv(os.path.join(template_ws, "sfr.csv"), index_col=0)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HEADWATER</th>
      <th>TAILWATER</th>
      <th>GAGE-1</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3652.5</th>
      <td>-751.175944</td>
      <td>-530.017751</td>
      <td>1362.956403</td>
    </tr>
    <tr>
      <th>3683.5</th>
      <td>-953.833450</td>
      <td>-674.975248</td>
      <td>1761.276157</td>
    </tr>
    <tr>
      <th>3712.5</th>
      <td>-1111.264769</td>
      <td>-799.212978</td>
      <td>2049.386246</td>
    </tr>
    <tr>
      <th>3743.5</th>
      <td>-1180.723380</td>
      <td>-853.985397</td>
      <td>2163.434411</td>
    </tr>
    <tr>
      <th>3773.5</th>
      <td>-1140.609765</td>
      <td>-823.268420</td>
      <td>2068.666541</td>
    </tr>
  </tbody>
</table>
</div>




```python
# add the observations to pf
sfr_df = pf.add_observations("sfr.csv", # the model output file to read
                            insfile="sfr.csv.ins", #optional, the instruction file name
                            index_cols="time", #column header to use as index; can also use column number (zero-based) instead of the header name
                            use_cols=list(df.columns.values), #names of columns that include observation values; can also use column number (zero-based) instead of the header name
                            prefix="sfr") #prefix to all observation names
```

We also want to add observations of particle travel time and status. Unfortuantely the file written by MODPATH7 is not easily injestible by `PstFrom`. So we are going to need to "manually" construct an instruction file. We could add that now with the `PstFrom.add_observations_from_ins()` method, but we will wait and  add these after constructing the `Pst` object - soon!

### 5. Parameters

The `PstFrom.add_parameters()` method reads model input files and adds parameters to the PEST(++) dataset. Parameterisation can be configured in several ways. 

 - model input files can be in array or list format;
 - parameters can be setup as different "types". Each value in model input files can (1) each be a separate parameter ("grid" scale parameters), (2) be grouped into "zones" or (3) all be treated as a single parameter ("constant" type). Alteratvely, (4) parameters can be assigned to pilot points, from which individual parameter values are subsequently interpolated. `PstFrom` adds the relevant pre-processing steps to assign paramter values directly into the "model run" script.
 - parameter values can be setup as "direct", "multiplier" or "addend". This means the "parameter value" which PEST(++) sees can be (1) the same value the model sees, (2) a multiplier on the value in the existing/original model input file, or (3) a value which is added to the value in the existing/original model input file. This is very nifty and allows for some pretty advanced parameterization schemes by allowing mixtures of different types of parameters. `PstFrom` is designed to preferentially use parameters setup as multipliers (that is the default parameter type). This let us preserve the existing model inputs and treat them as the mean of the prior parameter distribution. Once again, relevant pre-processing scripts are automatically added to the "model run" script (discussed later) so that the multiplicative and additive parameterization process is not something the user has to worry about.


#### 5.1. Freyberg Recap

As discussed, all model inputs are stored in external files. Some are arrays. Others are lists. Recall that our model has 3 layers. It is transient. Hydraulic properties (Kh, Kv, Ss, Sy) vary in space. Recharge varies over both space and time. We have GHBs, SFR and WEL boundary conditions. GHB parameters are constant over time, but vary spatially. SFR inflow varies over time. Pumping rates of individual wells are uncertain in space and and time.

All of these have some degree of spatial and/or temporal correlation.

#### 5.2. Geostatistical Structures

Parameter correlation plays a role in (1) regularization when giving preference to the emergence of patterns of spatial heterogeneity and (2) when specifying the prior parameter probability distribution (which is what regularization is enforcing!). Since we are all sophisticated and recognize the importance of expressing spatial and temporal uncertainty (e.g. heterogeneity) in the model inputs (and the corresponding spatial correlation in those uncertain inputs), let's use geostatistics to express uncertainty. To do that we need to define "geostatistical structures". 

For the sake of this tutorial, let's assume that heterogeneity in all spatially distributed parameters share the same statistical characteristics. Likewise for temporally varying parameters. We will therefore only  construct two geostatisitcal structures.


```python
# exponential variogram for spatially varying parameters
v_space = pyemu.geostats.ExpVario(contribution=1.0, #sill
                                    a=1000, # range of correlation; length units of the model. In our case 'meters'
                                    anisotropy=1.0, #name says it all
                                    bearing=0.0 #angle in degrees East of North corresponding to anisotropy ellipse
                                    )

# geostatistical structure for spatially varying parameters
grid_gs = pyemu.geostats.GeoStruct(variograms=v_space, transform='log') 

# plot the gs if you like:
grid_gs.plot()
```




    <AxesSubplot:xlabel='distance', ylabel='$\\gamma$'>




    
![png](freyberg_pstfrom_pest_setup_files/freyberg_pstfrom_pest_setup_32_1.png)
    



```python
# exponential variogram for time varying parameters
v_time = pyemu.geostats.ExpVario(contribution=1.0, #sill
                                    a=60, # range of correlation; length time units (days)
                                    anisotropy=1.0, #do not change for 1-D time
                                    bearing=0.0 #do not change for 1-D time
                                    )

# geostatistical structure for time varying parameters
temporal_gs = pyemu.geostats.GeoStruct(variograms=v_time, transform='none') 
```

#### 5.3. Add Parameters

Let's start by adding parameters of hydraulic properties that vary in space (but not time) and which are housed in array-type files (e.g. Kh, Kv, Ss, Sy). We will start by demonstrating step-by-step for Kh.

First, find all the external array files that contain Kh values. In our case, these are the files with "npf_k_" in the file name. As you can see below, there is one file for each model layer. 


```python
tag = "npf_k_"
files = [f for f in os.listdir(template_ws) if tag in f.lower() and f.endswith(".txt")]
print(files)
```

    ['freyberg6.npf_k_layer1.txt', 'freyberg6.npf_k_layer2.txt', 'freyberg6.npf_k_layer3.txt']
    

Let's setup multiple spatial scales of parameters for Kh. To do this we will use three of the parameter "types" described above. The coarse scale will be a `constant` single value for each array. The medium scale will `pilot points`. The finest scale will use parameters as the `grid` scale (a unique parameter for each model cell!)

Each scale of parameters will work with the others as multipliers with the existing Kh arrays. (This all happens at runtime as part of the "model run" script.) Think of the scales as dials that PEST(++) can turn to improve the fit. The "coarse" scale is one big dial that alows PEST to move everything at once - that is, change the mean of the entire Kh array. The "medium" dials are few (but not too many) that allow PEST to adjust broad areas, but not making eveything move. The "fine" scales are lots of small dials that allow PEST(++) to have very detailed control, tweaking parameter values within very small areas. 

However, because we are working with parameter `multipliers`, we will need to specify two sets of parameter bounds: 
 - `upper_bound` and `lower_bound` are the standard control file bounds (the bounds on the parameters that PEST sees), while
 - `ult_ubound` and `ult_lbound` are bounds that are applied at runtime to the resulting (multiplied out) model input array that MODFLOW reads. 
 
Since we are using sets of multipliers, it is important to make sure we keep the resulting model input arrays within the range of realistic values.

#### 5.3.1. Array Files

We will first demonstrate steb-by-step for `freyberg6.npf_k_layer1.txt`. We will start with grid scale parameters. These are multipliers assigned to each individual value in the array.

We start by getting the idomain array. As our model has inactive cells, this helps us avoid adding unncessary parameters. It is also required later when generating pilot points.


```python
# as IDOMIAN is the same in all layers, we can use any layer
ib = gwf.dis.idomain.get_data(layer=0)
plt.imshow(ib)
```




    <matplotlib.image.AxesImage at 0x1c7b24693a0>




    
![png](freyberg_pstfrom_pest_setup_files/freyberg_pstfrom_pest_setup_37_1.png)
    



```python
f = 'freyberg6.npf_k_layer1.txt'

# grid (fine) scale parameters
df_gr = pf.add_parameters(f,
                zone_array=ib, #as we have inactie model cells, we can avoid assigning these as parameters
                par_type="grid", #specify the type, these will be unique parameters for each cell
                geostruct=grid_gs, # the gestatisical structure for spatial correlation 
                par_name_base=f.split('.')[1].replace("_","")+"gr", #specify a parameter name base that allows us to easily identify the filename and parameter type. "_gr" for "grid", and so forth.
                pargp=f.split('.')[1].replace("_","")+"gr", #likewise for the parameter group name
                lower_bound=0.2, upper_bound=5.0, #parameter lower and upper bound
                ult_ubound=100, ult_lbound=0.01 # The ultimate bounds for multiplied model input values. Here we are stating that, after accounting for all multipliers, Kh cannot exceed these values. Very important with multipliers
                )
```

As when adding observations,  `pf.add_parameters()` returns a dataframe. Take a look. You may recognize alot of the information that appears in a PEST `*parameter data` section. All of this is still only housed in memory for now. We will write the PEST control file later on.


```python
df_gr.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>parnme</th>
      <th>parval1</th>
      <th>i</th>
      <th>j</th>
      <th>x</th>
      <th>y</th>
      <th>pargp</th>
      <th>tpl_filename</th>
      <th>input_filename</th>
      <th>partype</th>
      <th>partrans</th>
      <th>parubnd</th>
      <th>parlbnd</th>
      <th>parchglim</th>
      <th>offset</th>
      <th>dercom</th>
      <th>scale</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>pname:npfklayer1gr_inst:0_ptype:gr_pstyle:m_i:0_j:0_x:125.00_y:9875.00_zone:1</th>
      <td>pname:npfklayer1gr_inst:0_ptype:gr_pstyle:m_i:0_j:0_x:125.00_y:9875.00_zone:1</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>125.0</td>
      <td>9875.0</td>
      <td>npfklayer1gr</td>
      <td>freyberg6_template\npfklayer1gr_inst0_grid.csv.tpl</td>
      <td>freyberg6_template\mult\npfklayer1gr_inst0_grid.csv</td>
      <td>grid</td>
      <td>log</td>
      <td>5.0</td>
      <td>0.2</td>
      <td>factor</td>
      <td>0.0</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>pname:npfklayer1gr_inst:0_ptype:gr_pstyle:m_i:0_j:1_x:375.00_y:9875.00_zone:1</th>
      <td>pname:npfklayer1gr_inst:0_ptype:gr_pstyle:m_i:0_j:1_x:375.00_y:9875.00_zone:1</td>
      <td>1.0</td>
      <td>0</td>
      <td>1</td>
      <td>375.0</td>
      <td>9875.0</td>
      <td>npfklayer1gr</td>
      <td>freyberg6_template\npfklayer1gr_inst0_grid.csv.tpl</td>
      <td>freyberg6_template\mult\npfklayer1gr_inst0_grid.csv</td>
      <td>grid</td>
      <td>log</td>
      <td>5.0</td>
      <td>0.2</td>
      <td>factor</td>
      <td>0.0</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>pname:npfklayer1gr_inst:0_ptype:gr_pstyle:m_i:0_j:2_x:625.00_y:9875.00_zone:1</th>
      <td>pname:npfklayer1gr_inst:0_ptype:gr_pstyle:m_i:0_j:2_x:625.00_y:9875.00_zone:1</td>
      <td>1.0</td>
      <td>0</td>
      <td>2</td>
      <td>625.0</td>
      <td>9875.0</td>
      <td>npfklayer1gr</td>
      <td>freyberg6_template\npfklayer1gr_inst0_grid.csv.tpl</td>
      <td>freyberg6_template\mult\npfklayer1gr_inst0_grid.csv</td>
      <td>grid</td>
      <td>log</td>
      <td>5.0</td>
      <td>0.2</td>
      <td>factor</td>
      <td>0.0</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>pname:npfklayer1gr_inst:0_ptype:gr_pstyle:m_i:0_j:3_x:875.00_y:9875.00_zone:1</th>
      <td>pname:npfklayer1gr_inst:0_ptype:gr_pstyle:m_i:0_j:3_x:875.00_y:9875.00_zone:1</td>
      <td>1.0</td>
      <td>0</td>
      <td>3</td>
      <td>875.0</td>
      <td>9875.0</td>
      <td>npfklayer1gr</td>
      <td>freyberg6_template\npfklayer1gr_inst0_grid.csv.tpl</td>
      <td>freyberg6_template\mult\npfklayer1gr_inst0_grid.csv</td>
      <td>grid</td>
      <td>log</td>
      <td>5.0</td>
      <td>0.2</td>
      <td>factor</td>
      <td>0.0</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>pname:npfklayer1gr_inst:0_ptype:gr_pstyle:m_i:0_j:4_x:1125.00_y:9875.00_zone:1</th>
      <td>pname:npfklayer1gr_inst:0_ptype:gr_pstyle:m_i:0_j:4_x:1125.00_y:9875.00_zone:1</td>
      <td>1.0</td>
      <td>0</td>
      <td>4</td>
      <td>1125.0</td>
      <td>9875.0</td>
      <td>npfklayer1gr</td>
      <td>freyberg6_template\npfklayer1gr_inst0_grid.csv.tpl</td>
      <td>freyberg6_template\mult\npfklayer1gr_inst0_grid.csv</td>
      <td>grid</td>
      <td>log</td>
      <td>5.0</td>
      <td>0.2</td>
      <td>factor</td>
      <td>0.0</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



This `add_parameters()` call also wrote a template file that PEST(++) will use to populate the multiplier array at runtime:


```python
[f for f in os.listdir(template_ws) if f.endswith(".tpl")]
```




    ['npfklayer1gr_inst0_grid.csv.tpl']



Remember!  no pest control file has been made yet. `PstFrom` is simply preparing to make a control file later...

Now, we add pilot point (medium scale) multiplier parameters to the same model input file. These multipliers are assigned to pilot points, which are subsequently interpolated to values in the array.

You can add pilot points in two ways:

1. `PstFrom` can generate them for you on a regular grid or 
2. you can supply `PstFrom` with existing pilot point location information in the form of a dataframe or a point-coverage shapefile. 

When you change `par_type` to "pilotpoints", by default, a regular grid of pilot points is setup using a default `pp_space` value of 10 (which is every 10th row and column). You can chnge this spacing by passing a integer to `pp_space` (as demonstrated below). 

Alternatively you can specify a filename or dataframe with pilot point locations. If you supply `pp_space` as a `str` it is assumed to be a filename. The extension is the guide: ".csv" for dataframe, ".shp" for shapefile (point-type). Anything else and the file is assumed to be a pilot points file type. The dataframe (or .csv file) must have "name", "x", and "y" as columns - it can have more, but must have those. 


```python
# pilot point (medium) scale parameters
df_pp = pf.add_parameters(f,
                    zone_array=ib,
                    par_type="pilotpoints",
                    geostruct=grid_gs,
                    par_name_base=f.split('.')[1].replace("_","")+"pp",
                    pargp=f.split('.')[1].replace("_","")+"pp",
                    lower_bound=0.2,upper_bound=5.0,
                    ult_ubound=100, ult_lbound=0.01,
                    pp_space=5) # `PstFrom` will generate a unifrom grid of pilot points in every 4th row and column
```

    starting interp point loop for 706 points
    starting 0
    starting 1
    starting 2
    starting 3
    starting 4
    starting 5
    starting 6
    starting 7
    starting 8
    starting 9
    took 3.211963 seconds
    


```python
fig,ax = plt.subplots(1,1,figsize=(4,6))
ax.set_aspect("equal")
ax.pcolormesh(sr.xcentergrid, sr.ycentergrid,ib)
ax.scatter(df_pp.x,df_pp.y)
```




    <matplotlib.collections.PathCollection at 0x1c7b277f2b0>




    
![png](freyberg_pstfrom_pest_setup_files/freyberg_pstfrom_pest_setup_46_1.png)
    


Lastly, add the constant (coarse) parameter multiplier. This is a single multiplier value applied to all values in the array. 


```python
# constant (coarse) scale parameters
df_cst = pf.add_parameters(f,
                    zone_array=ib,
                    par_type="constant",
                    geostruct=grid_gs,
                    par_name_base=f.split('.')[1].replace("_","")+"cn",
                    pargp=f.split('.')[1].replace("_","")+"cn",
                    lower_bound=0.2,upper_bound=5.0,
                    ult_ubound=100, ult_lbound=0.01)
```

Now we see three template files have been created:


```python
[f for f in os.listdir(template_ws) if f.endswith(".tpl")]
```




    ['npfklayer1cn_inst0_constant.csv.tpl',
     'npfklayer1gr_inst0_grid.csv.tpl',
     'npfklayer1pp_inst0pp.dat.tpl']



Feel free to navigate to the `template_ws` and inspect these files.

Let's do that for Kh in the other layers. We are going to be doing this a few times, so lets write a function.


```python
def add_mult_pars(f, lb=0.2, ub=5.0, ulb=0.01, uub=100, add_coarse=True):
    if isinstance(f,str):
        base = f.split(".")[1].replace("_","")
    else:
        base = f[0].split(".")[1]
    # grid (fine) scale parameters
    pf.add_parameters(f,
                    zone_array=ib,
                    par_type="grid", #specify the type, these will be unique parameters for each cell
                    geostruct=grid_gs, # the gestatisical structure for spatial correlation 
                    par_name_base=base+"gr", #specify a parameter name base that allows us to easily identify the filename and parameter type. "_gr" for "grid", and so forth.
                    pargp=base+"gr", #likewise for the parameter group name
                    lower_bound=lb, upper_bound=ub, #parameter lower and upper bound
                    ult_ubound=uub, ult_lbound=ulb # The ultimate bounds for multiplied model input values. Here we are stating that, after accounting for all multipliers, Kh cannot exceed these values. Very important with multipliers
                    )
                    
    # pilot point (medium) scale parameters
    pf.add_parameters(f,
                        zone_array=ib,
                        par_type="pilotpoints",
                        geostruct=grid_gs,
                        par_name_base=base+"pp",
                        pargp=base+"pp",
                        lower_bound=lb, upper_bound=ub,
                        ult_ubound=uub, ult_lbound=ulb,
                        pp_space=5) # `PstFrom` will generate a unifrom grid of pilot points in every 4th row and column
    if add_coarse==True:
        # constant (coarse) scale parameters
        pf.add_parameters(f,
                            zone_array=ib,
                            par_type="constant",
                            geostruct=grid_gs,
                            par_name_base=base+"cn",
                            pargp=base+"cn",
                            lower_bound=lb, upper_bound=ub,
                            ult_ubound=uub, ult_lbound=ulb)
    return
```

A reminder of which files are listed in `files`:


```python
files
```




    ['freyberg6.npf_k_layer1.txt',
     'freyberg6.npf_k_layer2.txt',
     'freyberg6.npf_k_layer3.txt']



Now let's apply our function to the last two:


```python
for f in files[1:]:
    add_mult_pars(f, lb=0.2, ub=5.0, ulb=0.01, uub=100)
```

Let's see what .tpl files have been added:


```python
[f for f in os.listdir(template_ws) if f.endswith(".tpl")]
```




    ['npfklayer1cn_inst0_constant.csv.tpl',
     'npfklayer1gr_inst0_grid.csv.tpl',
     'npfklayer1pp_inst0pp.dat.tpl',
     'npfklayer2cn_inst0_constant.csv.tpl',
     'npfklayer2gr_inst0_grid.csv.tpl',
     'npfklayer2pp_inst0pp.dat.tpl',
     'npfklayer3cn_inst0_constant.csv.tpl',
     'npfklayer3gr_inst0_grid.csv.tpl',
     'npfklayer3pp_inst0pp.dat.tpl']



Well...hot damn, wasn't that easy? Let's speed through the other array parameter files.


```python
# for Kv
tag = "npf_k33"
files = [f for f in os.listdir(template_ws) if tag in f.lower() and f.endswith(".txt")]
for f in files:
    add_mult_pars(f, lb=0.2, ub=5.0, ulb=0.01, uub=100)

# for Ss
tag = "sto_ss"
files = [f for f in os.listdir(template_ws) if tag in f.lower() and f.endswith(".txt")]
# only for layer 2 and 3; we aren't monsters
for f in files[1:]: 
    add_mult_pars(f, lb=0.2, ub=5.0, ulb=1e-6, uub=1e-3)

# For Sy
tag = "sto_sy"
files = [f for f in os.listdir(template_ws) if tag in f.lower() and f.endswith(".txt")]
# only for layer 1
f = files[0]
add_mult_pars(f, lb=0.2, ub=5.0, ulb=0.01, uub=0.4)

# For porosity
tag = "ne_"
files = [f for f in os.listdir(template_ws) if tag in f.lower() and f.endswith(".txt")]
for f in files: 
    add_mult_pars(f, lb=0.2, ub=5.0, ulb=0.01, uub=0.4)

```


```python
[f for f in os.listdir(template_ws) if f.endswith(".tpl")]
```




    ['nelayer1cn_inst0_constant.csv.tpl',
     'nelayer1gr_inst0_grid.csv.tpl',
     'nelayer1pp_inst0pp.dat.tpl',
     'nelayer2cn_inst0_constant.csv.tpl',
     'nelayer2gr_inst0_grid.csv.tpl',
     'nelayer2pp_inst0pp.dat.tpl',
     'nelayer3cn_inst0_constant.csv.tpl',
     'nelayer3gr_inst0_grid.csv.tpl',
     'nelayer3pp_inst0pp.dat.tpl',
     'npfk33layer1cn_inst0_constant.csv.tpl',
     'npfk33layer1gr_inst0_grid.csv.tpl',
     'npfk33layer1pp_inst0pp.dat.tpl',
     'npfk33layer2cn_inst0_constant.csv.tpl',
     'npfk33layer2gr_inst0_grid.csv.tpl',
     'npfk33layer2pp_inst0pp.dat.tpl',
     'npfk33layer3cn_inst0_constant.csv.tpl',
     'npfk33layer3gr_inst0_grid.csv.tpl',
     'npfk33layer3pp_inst0pp.dat.tpl',
     'npfklayer1cn_inst0_constant.csv.tpl',
     'npfklayer1gr_inst0_grid.csv.tpl',
     'npfklayer1pp_inst0pp.dat.tpl',
     'npfklayer2cn_inst0_constant.csv.tpl',
     'npfklayer2gr_inst0_grid.csv.tpl',
     'npfklayer2pp_inst0pp.dat.tpl',
     'npfklayer3cn_inst0_constant.csv.tpl',
     'npfklayer3gr_inst0_grid.csv.tpl',
     'npfklayer3pp_inst0pp.dat.tpl',
     'stosslayer2cn_inst0_constant.csv.tpl',
     'stosslayer2gr_inst0_grid.csv.tpl',
     'stosslayer2pp_inst0pp.dat.tpl',
     'stosslayer3cn_inst0_constant.csv.tpl',
     'stosslayer3gr_inst0_grid.csv.tpl',
     'stosslayer3pp_inst0pp.dat.tpl',
     'stosylayer1cn_inst0_constant.csv.tpl',
     'stosylayer1gr_inst0_grid.csv.tpl',
     'stosylayer1pp_inst0pp.dat.tpl']



Boom!  We just conquered property parameterization in a big way!

#### 5.3.2. Spatial and Temporal Correlation

Now, you may be thinking "shouldn't recharge have temporal correlation as well?". 

Damn straight it should. Now, this requires a little trickery because native handling in spatiotemporal correlation is hard to do.  So what we are going to do is split this correlation into two setup of multiplier parameters.  One set of parameters will be constant in space but vary (and be correlated) in time.  The other set of multiplier parameters will be constant in time but vary (and be correlated) in space.  Since both of these sets of parameters are multipliers, we implicitly represent the concept that recharge is uncertain and correlated in both space and time.  Easy as!

First we need to construct a container of stress period datetimes. (This relies on specifying the start_datetime argument when instantiating `PstFrom`.) These datetime values will specify the postion of parameters on the time-axis.




```python
# build up a container of stress period start datetimes - this will
# be used to specify the datetime of each multipler parameter

dts = pd.to_datetime(start_datetime) + pd.to_timedelta(np.cumsum(sim.tdis.perioddata.array["perlen"]),unit='d')

dts
```




    DatetimeIndex(['2017-12-31 12:00:00', '2018-01-31 12:00:00',
                   '2018-03-01 12:00:00', '2018-04-01 12:00:00',
                   '2018-05-01 12:00:00', '2018-06-01 12:00:00',
                   '2018-07-01 12:00:00', '2018-08-01 12:00:00',
                   '2018-09-01 12:00:00', '2018-10-01 12:00:00',
                   '2018-11-01 12:00:00', '2018-12-01 12:00:00',
                   '2019-01-01 12:00:00', '2019-02-01 12:00:00',
                   '2019-03-01 12:00:00', '2019-04-01 12:00:00',
                   '2019-05-01 12:00:00', '2019-06-01 12:00:00',
                   '2019-07-01 12:00:00', '2019-08-01 12:00:00',
                   '2019-09-01 12:00:00', '2019-10-01 12:00:00',
                   '2019-11-01 12:00:00', '2019-12-01 12:00:00',
                   '2020-01-01 12:00:00'],
                  dtype='datetime64[ns]', freq=None)



If you use the same parameter group name (`pargp`) and same geostruct, `PstFrom` will treat parameters setup across different calls to `add_parameters()` as correlated - ***WARNING*** do not try to express spatial and temporal correlation together - as discussed above, #badtimes.  In this case, we want to express temporal correlation in the recharge multiplier parameters that are "constant" type in space so that there is one recharge multiplier parameter for each stress period that shares a parameter group name across different calls to `add_parameters`. So, we use the same parameter group names for each stress period data file, and specify the `datetime` and `geostruct` arguments.

Including temporal correlation introduces an additional challenge. Interpolation between points that share a common coordinate creates all types of trouble. We are going to have many parameters during each stress period (a single point on the time-axis). To get around this challenge we need to be a bit sneaky.


First, we will apply the multiple *spatial* scales of parameter multiplers (`constant`, `pilot point` and `grid`) as we did for hyraulic properties.  We do this for all recharge files at once, which tells `PstFrom` to broadcast (e.g. share) the same parameters for all of those files: 




```python
# for Recharge; 
tag = "rch_recharge"
files = [f for f in os.listdir(template_ws) if tag in f.lower() and f.endswith(".txt")]
sp = [int(f.split(".")[1].split('_')[-1]) for f in files]
d = {s:f for s,f in zip(sp,files)}
sp.sort()
files = [d[s] for s in sp]
print(files)
# the spatial multiplier parameters; just use the same function
add_mult_pars(files, lb=0.2, ub=5.0, ulb=2e-5, uub=2e-4, add_coarse=False)
    
    
```

    ['freyberg6.rch_recharge_1.txt', 'freyberg6.rch_recharge_2.txt', 'freyberg6.rch_recharge_3.txt', 'freyberg6.rch_recharge_4.txt', 'freyberg6.rch_recharge_5.txt', 'freyberg6.rch_recharge_6.txt', 'freyberg6.rch_recharge_7.txt', 'freyberg6.rch_recharge_8.txt', 'freyberg6.rch_recharge_9.txt', 'freyberg6.rch_recharge_10.txt', 'freyberg6.rch_recharge_11.txt', 'freyberg6.rch_recharge_12.txt', 'freyberg6.rch_recharge_13.txt', 'freyberg6.rch_recharge_14.txt', 'freyberg6.rch_recharge_15.txt', 'freyberg6.rch_recharge_16.txt', 'freyberg6.rch_recharge_17.txt', 'freyberg6.rch_recharge_18.txt', 'freyberg6.rch_recharge_19.txt', 'freyberg6.rch_recharge_20.txt', 'freyberg6.rch_recharge_21.txt', 'freyberg6.rch_recharge_22.txt', 'freyberg6.rch_recharge_23.txt', 'freyberg6.rch_recharge_24.txt', 'freyberg6.rch_recharge_25.txt']
    

Then, we will asign an additional `constant` multiplier parameter for each recharge stress-period file (so, a single multiplier for all recharge paramaters for each stress period). We will specify temporal correlation for these `constant` multipliers. These will all have the same parameter group name, as discussed above. 


```python
for f in files:   
    # multiplier that includes temporal correlation
    # get the stress period number from the file name
    kper = int(f.split('.')[1].split('_')[-1]) - 1  
    # add the constant parameters (with temporal correlation)
    pf.add_parameters(filenames=f,
                    zone_array=ib,
                    par_type="constant",
                    par_name_base=f.split('.')[1]+"tcn",
                    pargp=f.split('.')[1]+"tcn",
                    lower_bound=0.5, upper_bound=1.5,
                    ult_ubound=2e-4, ult_lbound=2e-5,
                    datetime=dts[kper], # this places the parameter value on the "time axis"
                    geostruct=temporal_gs)
```

### 5.3.3. List Files

Adding parameters from list-type files follows similar principles. As with observation files, they must be tabular. Certain columns are specified as index columns and are used to populate parameter names, as well as provide the parameters' spatial location. Other columns are specified as containing parameter values. 

Parameters can be `grid` or `constant`. As before, values can be assigned `directly`, as `multipliers` or as `additives`.

We will demonstrate for the boundary-condition input files. 

Starting off with GHBs. Let's inspect the folder. As you can see, there is a single input file (GHB parameters are assumed to not vary over time).


```python
tag = "ghb_stress_period_data"
files = [f for f in os.listdir(template_ws) if tag in f.lower() and f.endswith(".txt")]
print(files)
```

    ['freyberg6.ghb_stress_period_data_1.txt']
    

Since these boundaries are likely to be very influential, we want to include a robust representation of their uncertainty - both head and conductance and at multiple scales.  

Let's parameterize both GHB conductance and head:

 - For conductance, we shall use two scales of `multiplier` parameters (`constant` and `grid`).

 - For heads, multipliers are not ideal. Insead we will use `additive` parameters. Again, with a coarse and fine scale.

 **ATTENTION!** 
 
 Additive parameters by default get assigned an initial parameter value of zero. This can be problematic later on when computing the derivatives. Be sure to either apply a parameter offset, or use "absolute" increment types in the parameter group section (we will implement the latter option further on in the current tutorial.)


```python
tag = "ghb_stress_period_data"
files = [f for f in os.listdir(template_ws) if tag in f.lower() and f.endswith(".txt")]

for f in files:
    # constant and grid scale multiplier conductance parameters
    name = 'ghbcond'
    pf.add_parameters(f,
                        par_type="grid",
                        geostruct=grid_gs,
                        par_name_base=name+"gr",
                        pargp=name+"gr",
                        index_cols=[0,1,2], #column containing lay,row,col
                        use_cols=[4], #column containing conductance values
                        lower_bound=0.1,upper_bound=10.0,
                        ult_lbound=0.1, ult_ubound=100) #absolute limits
    pf.add_parameters(f,
                        par_type="constant",
                        geostruct=grid_gs,
                        par_name_base=name+"cn",
                        pargp=name+"cn",
                        index_cols=[0,1,2],
                        use_cols=[4],  
                        lower_bound=0.1,upper_bound=10.0,
                        ult_lbound=0.1, ult_ubound=100) #absolute limits

    # constant and grid scale additive head parameters
    name = 'ghbhead'
    pf.add_parameters(f,
                        par_type="grid",
                        geostruct=grid_gs,
                        par_name_base=name+"gr",
                        pargp=name+"gr",
                        index_cols=[0,1,2],
                        use_cols=[3],   # column containing head values
                        lower_bound=-2.0,upper_bound=2.0,
                        par_style="a", # specify additive parameter
                        transform="none", # specify not log-transform
                        ult_lbound=32.5, ult_ubound=42) #absolute limits; make sure head is never lower than the bottom of layer1
    pf.add_parameters(f,
                        par_type="constant",
                        geostruct=grid_gs,
                        par_name_base=name+"cn",
                        pargp=name+"cn",
                        index_cols=[0,1,2],
                        use_cols=[3],
                        lower_bound=-2.0,upper_bound=2.0, 
                        par_style="a", 
                        transform="none",
                        ult_lbound=32.5, ult_ubound=42) 
```

Easy peasy.

Now, this will make some people uncomfortable, but how well do we really ever know historic water use flux rates in space and in time? hmmm, not really! And just a little uncertainty in historic water use can result in large changes in simulated water levels...So lets add parameters to represent that uncertainty in the model inputs.

For wells it may not (or it may...) make sense to include spatial correlation. Here we will assume temporal correlation - its reasonable that pumping rates today will be similar to pumping rates yesterday. 

Pumping rates for different stress periods are in separate files. We will call `.add_parameters()` for each file. But we want to specify correlation between parameters in different files. As explained above for recharge, we do this with the parameter group name.

OK, let's get started.


As discussed above, including temporal correlation introduces an additional challenge. We use the same approach described for recharge parameters:

 - First, we will asign a `constant` multiplier parameter for each WEL stress-period file (so, a single multiplier for all well pumping rates for each stress period). We will specify temporal correlation for these `constant` multipliers.

 - Then, we will also have `grid` type multiplier parameters for each WEL stress period file (so, multipliers for individual well pumping rate during each stress period). These will not include (temporal) correlation. (We could in principle include spatial correlation here if we wanted to; but let's not).


```python
files = [f for f in os.listdir(template_ws) if "wel_stress_period_data" in f and f.endswith(".txt")]
sp = [int(f.split(".")[1].split('_')[-1]) for f in files]
d = {s:f for s,f in zip(sp,files)}
sp.sort()
files = [d[s] for s in sp]

for f in files:
    # get the stress period number from the file name
    kper = int(f.split('.')[1].split('_')[-1]) - 1  
    
    # add the constant parameters (with temporal correlation)
    pf.add_parameters(filenames=f,
                        index_cols=[0,1,2], #columns that specify cell location
                        use_cols=[3],       #columns with parameter values
                        par_type="constant",    #each well will be adjustable
                        par_name_base="welcst",
                        pargp="welcst", 
                        upper_bound = 1.5, lower_bound=0.5,
                        datetime=dts[kper], # this places the parameter value on the "time axis"
                        geostruct=temporal_gs)
    
    # add the grid parameters; each individual well
    pf.add_parameters(filenames=f,
                        index_cols=[0,1,2], #columns that specify cell location 
                        use_cols=[3],       #columns with parameter values
                        par_type="grid",    #each well will be adjustable
                        par_name_base="welgrd",
                        pargp="welgrd", 
                        upper_bound = 1.5, lower_bound=0.5,
                        datetime=dts[kper]) # this places the parameter value on the "time axis"
                     
```

And finally, our favourite (not!) boundary-condition: SFR.

Let's parameterize conductance (time-invariant) and inflow (time-variant).


```python
# SFR conductance
tag = "sfr_packagedata"
files = [f for f in os.listdir(template_ws) if tag in f.lower() and f.endswith(".txt")]
assert len(files) == 1 # There can be only one! It is tradition. Jokes.
print(files)

f = files[0]
# constant and grid scale multiplier conductance parameters
name = "sfrcond"
pf.add_parameters(f,
                par_type="grid",
                geostruct=grid_gs,
                par_name_base=name+"gr",
                pargp=name+"gr",
                index_cols=[0,2,3],
                use_cols=[9],
                lower_bound=0.1,upper_bound=10.0,
                ult_lbound=0.01, ult_ubound=10) #absolute limits
pf.add_parameters(f,
                par_type="constant",
                geostruct=grid_gs,
                par_name_base=name+"cn",
                pargp=name+"cn",
                index_cols=[0,2,3],
                use_cols=[9],
                lower_bound=0.1,upper_bound=10.0,
                ult_lbound=0.01, ult_ubound=10) #absolute limits
```

    ['freyberg6.sfr_packagedata.txt']
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>parnme</th>
      <th>pargp</th>
      <th>covgp</th>
      <th>tpl_filename</th>
      <th>input_filename</th>
      <th>parval1</th>
      <th>partype</th>
      <th>partrans</th>
      <th>parubnd</th>
      <th>parlbnd</th>
      <th>parchglim</th>
      <th>offset</th>
      <th>dercom</th>
      <th>scale</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>pname:sfrcondcn_inst:0_ptype:cn_usecol:9_pstyle:m</th>
      <td>pname:sfrcondcn_inst:0_ptype:cn_usecol:9_pstyle:m</td>
      <td>sfrcondcn</td>
      <td>sfrcondcn</td>
      <td>freyberg6_template\sfrcondcn_inst0_constant.csv.tpl</td>
      <td>freyberg6_template\mult\sfrcondcn_inst0_constant.csv</td>
      <td>1.0</td>
      <td>constant</td>
      <td>log</td>
      <td>10.0</td>
      <td>0.1</td>
      <td>factor</td>
      <td>0.0</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# SFR inflow
files = [f for f in os.listdir(template_ws) if "sfr_perioddata" in f and f.endswith(".txt")]
sp = [int(f.split(".")[1].split('_')[-1]) for f in files]
d = {s:f for s,f in zip(sp,files)}
sp.sort()
files = [d[s] for s in sp]
print(files)
for f in files:
    # get the stress period number from the file name
    kper = int(f.split('.')[1].split('_')[-1]) - 1  
    # add the parameters
    pf.add_parameters(filenames=f,
                        index_cols=[0], #reach number
                        use_cols=[2],   #columns with parameter values
                        par_type="grid",    
                        par_name_base="sfrgr",
                        pargp="sfrgr", 
                        upper_bound = 1.5, lower_bound=0.5, #don't need ult_bounds because it is a single multiplier
                        datetime=dts[kper], # this places the parameter value on the "time axis"
                        geostruct=temporal_gs)
```

    ['freyberg6.sfr_perioddata_1.txt', 'freyberg6.sfr_perioddata_2.txt', 'freyberg6.sfr_perioddata_3.txt', 'freyberg6.sfr_perioddata_4.txt', 'freyberg6.sfr_perioddata_5.txt', 'freyberg6.sfr_perioddata_6.txt', 'freyberg6.sfr_perioddata_7.txt', 'freyberg6.sfr_perioddata_8.txt', 'freyberg6.sfr_perioddata_9.txt', 'freyberg6.sfr_perioddata_10.txt', 'freyberg6.sfr_perioddata_11.txt', 'freyberg6.sfr_perioddata_12.txt', 'freyberg6.sfr_perioddata_13.txt', 'freyberg6.sfr_perioddata_14.txt', 'freyberg6.sfr_perioddata_15.txt', 'freyberg6.sfr_perioddata_16.txt', 'freyberg6.sfr_perioddata_17.txt', 'freyberg6.sfr_perioddata_18.txt', 'freyberg6.sfr_perioddata_19.txt', 'freyberg6.sfr_perioddata_20.txt', 'freyberg6.sfr_perioddata_21.txt', 'freyberg6.sfr_perioddata_22.txt', 'freyberg6.sfr_perioddata_23.txt', 'freyberg6.sfr_perioddata_24.txt', 'freyberg6.sfr_perioddata_25.txt']
    


```python
[f for f in os.listdir(template_ws) if f.endswith(".tpl")]
```




    ['ghbcondcn_inst0_constant.csv.tpl',
     'ghbcondgr_inst0_grid.csv.tpl',
     'ghbheadcn_inst0_constant.csv.tpl',
     'ghbheadgr_inst0_grid.csv.tpl',
     'nelayer1cn_inst0_constant.csv.tpl',
     'nelayer1gr_inst0_grid.csv.tpl',
     'nelayer1pp_inst0pp.dat.tpl',
     'nelayer2cn_inst0_constant.csv.tpl',
     'nelayer2gr_inst0_grid.csv.tpl',
     'nelayer2pp_inst0pp.dat.tpl',
     'nelayer3cn_inst0_constant.csv.tpl',
     'nelayer3gr_inst0_grid.csv.tpl',
     'nelayer3pp_inst0pp.dat.tpl',
     'npfk33layer1cn_inst0_constant.csv.tpl',
     'npfk33layer1gr_inst0_grid.csv.tpl',
     'npfk33layer1pp_inst0pp.dat.tpl',
     'npfk33layer2cn_inst0_constant.csv.tpl',
     'npfk33layer2gr_inst0_grid.csv.tpl',
     'npfk33layer2pp_inst0pp.dat.tpl',
     'npfk33layer3cn_inst0_constant.csv.tpl',
     'npfk33layer3gr_inst0_grid.csv.tpl',
     'npfk33layer3pp_inst0pp.dat.tpl',
     'npfklayer1cn_inst0_constant.csv.tpl',
     'npfklayer1gr_inst0_grid.csv.tpl',
     'npfklayer1pp_inst0pp.dat.tpl',
     'npfklayer2cn_inst0_constant.csv.tpl',
     'npfklayer2gr_inst0_grid.csv.tpl',
     'npfklayer2pp_inst0pp.dat.tpl',
     'npfklayer3cn_inst0_constant.csv.tpl',
     'npfklayer3gr_inst0_grid.csv.tpl',
     'npfklayer3pp_inst0pp.dat.tpl',
     'rch_recharge_10tcn_inst0_constant.csv.tpl',
     'rch_recharge_11tcn_inst0_constant.csv.tpl',
     'rch_recharge_12tcn_inst0_constant.csv.tpl',
     'rch_recharge_13tcn_inst0_constant.csv.tpl',
     'rch_recharge_14tcn_inst0_constant.csv.tpl',
     'rch_recharge_15tcn_inst0_constant.csv.tpl',
     'rch_recharge_16tcn_inst0_constant.csv.tpl',
     'rch_recharge_17tcn_inst0_constant.csv.tpl',
     'rch_recharge_18tcn_inst0_constant.csv.tpl',
     'rch_recharge_19tcn_inst0_constant.csv.tpl',
     'rch_recharge_1gr_inst0_grid.csv.tpl',
     'rch_recharge_1pp_inst0pp.dat.tpl',
     'rch_recharge_1tcn_inst0_constant.csv.tpl',
     'rch_recharge_20tcn_inst0_constant.csv.tpl',
     'rch_recharge_21tcn_inst0_constant.csv.tpl',
     'rch_recharge_22tcn_inst0_constant.csv.tpl',
     'rch_recharge_23tcn_inst0_constant.csv.tpl',
     'rch_recharge_24tcn_inst0_constant.csv.tpl',
     'rch_recharge_25tcn_inst0_constant.csv.tpl',
     'rch_recharge_2tcn_inst0_constant.csv.tpl',
     'rch_recharge_3tcn_inst0_constant.csv.tpl',
     'rch_recharge_4tcn_inst0_constant.csv.tpl',
     'rch_recharge_5tcn_inst0_constant.csv.tpl',
     'rch_recharge_6tcn_inst0_constant.csv.tpl',
     'rch_recharge_7tcn_inst0_constant.csv.tpl',
     'rch_recharge_8tcn_inst0_constant.csv.tpl',
     'rch_recharge_9tcn_inst0_constant.csv.tpl',
     'sfrcondcn_inst0_constant.csv.tpl',
     'sfrcondgr_inst0_grid.csv.tpl',
     'sfrgr_inst0_grid.csv.tpl',
     'sfrgr_inst10_grid.csv.tpl',
     'sfrgr_inst11_grid.csv.tpl',
     'sfrgr_inst12_grid.csv.tpl',
     'sfrgr_inst13_grid.csv.tpl',
     'sfrgr_inst14_grid.csv.tpl',
     'sfrgr_inst15_grid.csv.tpl',
     'sfrgr_inst16_grid.csv.tpl',
     'sfrgr_inst17_grid.csv.tpl',
     'sfrgr_inst18_grid.csv.tpl',
     'sfrgr_inst19_grid.csv.tpl',
     'sfrgr_inst1_grid.csv.tpl',
     'sfrgr_inst20_grid.csv.tpl',
     'sfrgr_inst21_grid.csv.tpl',
     'sfrgr_inst22_grid.csv.tpl',
     'sfrgr_inst23_grid.csv.tpl',
     'sfrgr_inst24_grid.csv.tpl',
     'sfrgr_inst2_grid.csv.tpl',
     'sfrgr_inst3_grid.csv.tpl',
     'sfrgr_inst4_grid.csv.tpl',
     'sfrgr_inst5_grid.csv.tpl',
     'sfrgr_inst6_grid.csv.tpl',
     'sfrgr_inst7_grid.csv.tpl',
     'sfrgr_inst8_grid.csv.tpl',
     'sfrgr_inst9_grid.csv.tpl',
     'stosslayer2cn_inst0_constant.csv.tpl',
     'stosslayer2gr_inst0_grid.csv.tpl',
     'stosslayer2pp_inst0pp.dat.tpl',
     'stosslayer3cn_inst0_constant.csv.tpl',
     'stosslayer3gr_inst0_grid.csv.tpl',
     'stosslayer3pp_inst0pp.dat.tpl',
     'stosylayer1cn_inst0_constant.csv.tpl',
     'stosylayer1gr_inst0_grid.csv.tpl',
     'stosylayer1pp_inst0pp.dat.tpl',
     'welcst_inst0_constant.csv.tpl',
     'welcst_inst10_constant.csv.tpl',
     'welcst_inst11_constant.csv.tpl',
     'welcst_inst12_constant.csv.tpl',
     'welcst_inst13_constant.csv.tpl',
     'welcst_inst14_constant.csv.tpl',
     'welcst_inst15_constant.csv.tpl',
     'welcst_inst16_constant.csv.tpl',
     'welcst_inst17_constant.csv.tpl',
     'welcst_inst18_constant.csv.tpl',
     'welcst_inst19_constant.csv.tpl',
     'welcst_inst1_constant.csv.tpl',
     'welcst_inst20_constant.csv.tpl',
     'welcst_inst21_constant.csv.tpl',
     'welcst_inst22_constant.csv.tpl',
     'welcst_inst23_constant.csv.tpl',
     'welcst_inst24_constant.csv.tpl',
     'welcst_inst2_constant.csv.tpl',
     'welcst_inst3_constant.csv.tpl',
     'welcst_inst4_constant.csv.tpl',
     'welcst_inst5_constant.csv.tpl',
     'welcst_inst6_constant.csv.tpl',
     'welcst_inst7_constant.csv.tpl',
     'welcst_inst8_constant.csv.tpl',
     'welcst_inst9_constant.csv.tpl',
     'welgrd_inst0_grid.csv.tpl',
     'welgrd_inst10_grid.csv.tpl',
     'welgrd_inst11_grid.csv.tpl',
     'welgrd_inst12_grid.csv.tpl',
     'welgrd_inst13_grid.csv.tpl',
     'welgrd_inst14_grid.csv.tpl',
     'welgrd_inst15_grid.csv.tpl',
     'welgrd_inst16_grid.csv.tpl',
     'welgrd_inst17_grid.csv.tpl',
     'welgrd_inst18_grid.csv.tpl',
     'welgrd_inst19_grid.csv.tpl',
     'welgrd_inst1_grid.csv.tpl',
     'welgrd_inst20_grid.csv.tpl',
     'welgrd_inst21_grid.csv.tpl',
     'welgrd_inst22_grid.csv.tpl',
     'welgrd_inst23_grid.csv.tpl',
     'welgrd_inst24_grid.csv.tpl',
     'welgrd_inst2_grid.csv.tpl',
     'welgrd_inst3_grid.csv.tpl',
     'welgrd_inst4_grid.csv.tpl',
     'welgrd_inst5_grid.csv.tpl',
     'welgrd_inst6_grid.csv.tpl',
     'welgrd_inst7_grid.csv.tpl',
     'welgrd_inst8_grid.csv.tpl',
     'welgrd_inst9_grid.csv.tpl']



Damn!  we just parameterized many recognized sources of model input uncertainty at several spatial and temporal scales.  And we expressed spatial and temporal correlation in those parameters.  One last set of parameters that we will need later for sequential data assimilation - initial conditions:


```python
files = [f for f in os.listdir(template_ws) if "ic_strt" in f and f.endswith(".txt")]
files
```




    ['freyberg6.ic_strt_layer1.txt',
     'freyberg6.ic_strt_layer2.txt',
     'freyberg6.ic_strt_layer3.txt']




```python
for f in files:
    base = f.split(".")[1].replace("_","")
    df = pf.add_parameters(f,par_type="grid",par_style="d",
                      pargp=base,par_name_base=base,upper_bound=50,
                     lower_bound=15,zone_array=ib,transform="none")
    print(df.shape)


```

    (706, 17)
    (706, 17)
    (706, 17)
    

### 6. The Forward Run Script

OK! So, we almost have all the base building blocks for a PEST(++) dataset. We have some (1) observations and some (2) parameters. We are still missing (3) the "forward run" script. Recall that in the PEST world, the "model" is not just the numerical model (e.g. MODFLOW). Instead it is a composite of the numerical model (or models) and pre- and post-processing steps, encapsulated in a "forward run" script which can be called from the command line. This command line instruction is what PEST(++) sees as "the model". During execution, PEST(++) writes values to parameter files, runs "the model", and then reads values from the observation files.

`PstFrom` automates the generation of such a script when constructing the PEST control file. The script is written to file named `forward_run.py`. It is written in Python (this is not a PEST(++) requirement, merely a convenience...we are working in Python after all...). 

How about we see that in action? Magic time! Let's create the PEST control file.




```python
pst = pf.build_pst()
```

    noptmax:0, npar_adj:12013, nnz_obs:725
    

Boom! Done. (Well almost.) Check the folder. You should see a new .pst file and the `forward_run.py` file. By default, the .pst file is named after the original model folder name. 


```python
[f for f in os.listdir(template_ws) if f.endswith(".py") or f.endswith(".pst") ]
```




    ['forward_run.py', 'freyberg_mf6.pst']



We will get to the `pst` object later on (see also the "intro to pyemu" tutorial notebook). For now, let's focus on the `forward_run.py` script. It is printed out below.

This script does a bunch of things:
 - it loads necessary dependecies
 - it removes model output files to avoid the possibility of files from a previous model run being read by mistake;
 - it runs pre-processing steps (see `pyemu.helpers.apply_list_and_array_pars()`;
 - it executes system commands (usually running the simulator, i.e. MODFLOW). (*This is still missing. We will demonstrate next.*)
 - it executes post-processing steps; (*for now there aren't any*)
 - ...it washes the dishes (sorry, no it doesn't...this feature is still in development).


```python
_ = [print(line.rstrip()) for line in open(os.path.join(template_ws,"forward_run.py"))]
```

    import os
    import multiprocessing as mp
    import numpy as np
    import pandas as pd
    import pyemu
    def main():
    
        try:
           os.remove(r'heads.csv')
        except Exception as e:
           print(r'error removing tmp file:heads.csv')
        try:
           os.remove(r'sfr.csv')
        except Exception as e:
           print(r'error removing tmp file:sfr.csv')
        pyemu.helpers.apply_list_and_array_pars(arr_par_file='mult2model_info.csv',chunk_len=50)
    
    if __name__ == '__main__':
        mp.freeze_support()
        main()
    
    

That's pretty amazing. But as we just saw, we still need to add commands to actualy run the model.

`PstFrom` allows you to pass a list of system commands which will be executed in sequence. It also has methods for including Python functions that run before or after the system commands. These make pre-/post-processing a piece of cake. In fact, we have already started to add to it. Remember all of the multiplier and additive parameters we setup? These all require pre-processing steps to convert the PEST-generated multipliers into model input values. `PstFrom` will automatically add these functions to the `forward_run.py` script. Nifty, hey?

Next we will demonstrate how to specify the system commands and add Python functions as processing steps.

#### 6.2. Sys Commands

Let's start by adding a command line instruction. These are stored as a list in `PstFrom.mod_sys_cmds`, which is currently empty. 


```python
pf.mod_sys_cmds 
```




    []



To run a MODFLOW6 model from the command line, you can simply execute `mf6` in the model folder. So, we can add this command by appending it to the list. (Do this only once! Every time you append 'mf6' results in an additional call to MODFLOW6, meaning the model would be run multiple times.)

`PstFrom` will add a line to `forward_run.py` w


```python
pf.mod_sys_cmds.append("mf6") #do this only once
pf.mod_sys_cmds
```




    ['mf6']



We also need to run MODPATH7, so we need to add that to the list of system commands. In this case we also need to specify the modpath sim file:


```python
pf.mod_sys_cmds.append("mp7 freyberg_mp.mpsim") #do this only once
pf.mod_sys_cmds
```




    ['mf6', 'mp7 freyberg_mp.mpsim']



OK, now let's re-build the Pst control file and check out the changes ot the `forward_run.py` script.

You should see that `pyemu.os_utils.run(r'mf6')` has been added after the pre-processing functions.


```python
pst = pf.build_pst()

_ = [print(line.rstrip()) for line in open(os.path.join(template_ws,"forward_run.py"))]
```

    noptmax:0, npar_adj:12013, nnz_obs:725
    import os
    import multiprocessing as mp
    import numpy as np
    import pandas as pd
    import pyemu
    def main():
    
        try:
           os.remove(r'heads.csv')
        except Exception as e:
           print(r'error removing tmp file:heads.csv')
        try:
           os.remove(r'sfr.csv')
        except Exception as e:
           print(r'error removing tmp file:sfr.csv')
        pyemu.helpers.apply_list_and_array_pars(arr_par_file='mult2model_info.csv',chunk_len=50)
        pyemu.os_utils.run(r'mf6')
    
        pyemu.os_utils.run(r'mp7 freyberg_mp.mpsim')
    
    
    if __name__ == '__main__':
        mp.freeze_support()
        main()
    
    

#### 6.3. Extra pre- and post-processing functions

You will also certainly need to include some additional processing steps.  These are supported thru the `PstFrom.pre_py_cmds` and `PstFrom.post_py_cmds`, which are lists for pre and post model run python commands and `PstFrom.pre_sys_cmds` and `PstFrom.post_sys_cmds`, which are lists for pre and post model run system commands (these are wrapped in `pyemu.os_utils.run()`.  

But what if your additional steps are actually an entire python function? Well, we got that too! `PstFrom.add_py_function()`. This method allows you to get functions from another (pre-prepared) python source file and add them to the `forward_run.py` script. We will deonstrate this to post-process secondary model observations after each run.

Now lets see this py-sauce in action: we are going to add a little post-processing function to extract the final simulated water level for all model cells for the last stress period from the MF6 binary headsave file and save them to ASCII format so that PEST(++) can read them with instruction files.  And, while we are at it, lets also extract the global water budget info from the MF6 listing file and store it in dataframes - these are ususally good numbers to watch!  We will need the simulated water level arrays later for sequential data assimilation (wouldnt it be nice if MF6 supported the writing of ASCII format head arrays?).  Anyway, this function is stored in the "helpers.py" script..


```python
pf.add_py_function("helpers.py","extract_hds_arrays_and_list_dfs()",is_pre_cmd=False)
```

That last argument - `is_pre_cmd` tells `PstFrom` if the python function should be treated as a pre-processor or a post-processor. So we have added that post-processor, but we still need to setup pest observations for those ASCII head arrays.  Let's do that by first calling that function to operate once within the `template_ws` to generate the arrays and then we can add them with `add_observations()`:  


```python
import helpers
helpers.test_extract_hds_arrays(template_ws)
```


```python
files = [f for f in os.listdir(template_ws) if f.startswith("hdslay")]
files
```




    ['hdslay1_t1.txt',
     'hdslay1_t10.txt',
     'hdslay1_t11.txt',
     'hdslay1_t12.txt',
     'hdslay1_t13.txt',
     'hdslay1_t14.txt',
     'hdslay1_t15.txt',
     'hdslay1_t16.txt',
     'hdslay1_t17.txt',
     'hdslay1_t18.txt',
     'hdslay1_t19.txt',
     'hdslay1_t2.txt',
     'hdslay1_t20.txt',
     'hdslay1_t21.txt',
     'hdslay1_t22.txt',
     'hdslay1_t23.txt',
     'hdslay1_t24.txt',
     'hdslay1_t25.txt',
     'hdslay1_t3.txt',
     'hdslay1_t4.txt',
     'hdslay1_t5.txt',
     'hdslay1_t6.txt',
     'hdslay1_t7.txt',
     'hdslay1_t8.txt',
     'hdslay1_t9.txt',
     'hdslay2_t1.txt',
     'hdslay2_t10.txt',
     'hdslay2_t11.txt',
     'hdslay2_t12.txt',
     'hdslay2_t13.txt',
     'hdslay2_t14.txt',
     'hdslay2_t15.txt',
     'hdslay2_t16.txt',
     'hdslay2_t17.txt',
     'hdslay2_t18.txt',
     'hdslay2_t19.txt',
     'hdslay2_t2.txt',
     'hdslay2_t20.txt',
     'hdslay2_t21.txt',
     'hdslay2_t22.txt',
     'hdslay2_t23.txt',
     'hdslay2_t24.txt',
     'hdslay2_t25.txt',
     'hdslay2_t3.txt',
     'hdslay2_t4.txt',
     'hdslay2_t5.txt',
     'hdslay2_t6.txt',
     'hdslay2_t7.txt',
     'hdslay2_t8.txt',
     'hdslay2_t9.txt',
     'hdslay3_t1.txt',
     'hdslay3_t10.txt',
     'hdslay3_t11.txt',
     'hdslay3_t12.txt',
     'hdslay3_t13.txt',
     'hdslay3_t14.txt',
     'hdslay3_t15.txt',
     'hdslay3_t16.txt',
     'hdslay3_t17.txt',
     'hdslay3_t18.txt',
     'hdslay3_t19.txt',
     'hdslay3_t2.txt',
     'hdslay3_t20.txt',
     'hdslay3_t21.txt',
     'hdslay3_t22.txt',
     'hdslay3_t23.txt',
     'hdslay3_t24.txt',
     'hdslay3_t25.txt',
     'hdslay3_t3.txt',
     'hdslay3_t4.txt',
     'hdslay3_t5.txt',
     'hdslay3_t6.txt',
     'hdslay3_t7.txt',
     'hdslay3_t8.txt',
     'hdslay3_t9.txt']




```python
for f in files:
    pf.add_observations(f,prefix=f.split(".")[0],obsgp=f.split(".")[0])
```


```python
for f in ["inc.csv","cum.csv"]:
    df = pd.read_csv(os.path.join(template_ws,f),index_col=0)
    pf.add_observations(f,index_cols=["totim"],use_cols=list(df.columns.values),
                        prefix=f.split('.')[0],obsgp=f.split(".")[0])
```

Crushed it!



#### 6.3.1. Secondary Observations

Often it is usefull to include "secondary model outcomes" as observations. These can be important components in a history-matching dataset to tease out specific aspects of system behaviour (e.g. head differences between aquifer layers to inform vertical permeabilities). Or they may be simple summaries of modelled outputs which are of interest for a prediction (e.g. minimum simulated head over a given period).

If you inspect the tutorial folder you will find a file named `helpers.py`. This is a python source file which we have prepared for you. (Open it to see how it is organized.) It contains a function named `process_secondary_obs()`. This function reads the model output .csv files, processes them and writes a series of new observation .csv files. These new files contain (1) the temporal-differences between head and SFR observations, and (2) the difference in heads between the top and bottom layers at each observation point. The new .csv files are named `heads.tdiff.csv`,`sfr.tdiff.csv` and `heads.vdiff.csv` respectively.

First, lets load the function here and run it so you can see what happens. (And to make sure that the observation files are in the template folder!) 

Run the next cell, then inspect the template folder. You should see three new csv files. These are the new secondary observations calculated by the post-processing function.


```python
# run the helper function
helpers.process_secondary_obs(ws=template_ws)
```

    Secondary observation files processed.
    


```python
[f for f in os.listdir(template_ws) if f.endswith(".csv")]
```




    ['cum.csv',
     'heads.csv',
     'heads.tdiff.csv',
     'heads.vdiff.csv',
     'inc.csv',
     'mult2model_info.csv',
     'sfr.csv',
     'sfr.tdiff.csv']



OK, so now let's add this function to the `forward_run.py` script.


```python
pf.add_py_function("helpers.py", # the file which contains the function
                    "process_secondary_obs(ws='.')", #the function, making sure to specify any arguments it may requrie
                    is_pre_cmd=False) # whether it runs before the model system command, or after. In this case, after.
```

And, boom! Bob's your uncle. As easy as that.

Now, of course we want to add these observations to `PstFrom` as well:


```python

df = pd.read_csv(os.path.join(template_ws, "sfr.tdiff.csv"), index_col=0)
_ = pf.add_observations("sfr.tdiff.csv", # the model output file to read
                            insfile="sfr.tdiff.csv.ins", #optional, the instruction file name
                            index_cols="time", #column header to use as index; can also use column number (zero-based) instead of the header name
                            use_cols=list(df.columns.values), #names of columns that include observation values; can also use column number (zero-based) instead of the header name
                            prefix="sfrtd") #prefix to all observation 
                            
df = pd.read_csv(os.path.join(template_ws, "heads.tdiff.csv"), index_col=0)
_ = pf.add_observations("heads.tdiff.csv", # the model output file to read
                            insfile="heads.tdiff.csv.ins", #optional, the instruction file name
                            index_cols="time", #column header to use as index; can also use column number (zero-based) instead of the header name
                            use_cols=list(df.columns.values), #names of columns that include observation values; can also use column number (zero-based) instead of the header name
                            prefix="hdstd") #prefix to all observation names

df = pd.read_csv(os.path.join(template_ws, "heads.vdiff.csv"), index_col=0)
_ = pf.add_observations("heads.vdiff.csv", # the model output file to read
                            insfile="heads.vdiff.csv.ins", #optional, the instruction file name
                            index_cols="time", #column header to use as index; can also use column number (zero-based) instead of the header name
                            use_cols=list(df.columns.values), #names of columns that include observation values; can also use column number (zero-based) instead of the header name
                            prefix="hdsvd") #prefix to all observation names
```

Remember to re-build the Pst control file:


```python
pst = pf.build_pst()
```

    noptmax:0, npar_adj:12013, nnz_obs:62225
    


```python
_ = [print(line.rstrip()) for line in open(os.path.join(template_ws,"forward_run.py"))]
```

    import os
    import multiprocessing as mp
    import numpy as np
    import pandas as pd
    import pyemu
    
    # function added thru PstFrom.add_py_function()
    def extract_hds_arrays_and_list_dfs():
        import flopy
        hds = flopy.utils.HeadFile("freyberg6_freyberg.hds")
        for it,t in enumerate(hds.get_times()):
            d = hds.get_data(totim=t)
            for k,dlay in enumerate(d):
                np.savetxt("hdslay{0}_t{1}.txt".format(k+1,it+1),d[k,:,:],fmt="%15.6E")
    
        lst = flopy.utils.Mf6ListBudget("freyberg6.lst")
        inc,cum = lst.get_dataframes(diff=True,start_datetime=None)
        inc.columns = inc.columns.map(lambda x: x.lower().replace("_","-"))
        cum.columns = cum.columns.map(lambda x: x.lower().replace("_", "-"))
        inc.index.name = "totim"
        cum.index.name = "totim"
        inc.to_csv("inc.csv")
        cum.to_csv("cum.csv")
        return
    
    
    
    
    # function added thru PstFrom.add_py_function()
    def process_secondary_obs(ws='.'):
        # load dependencies insde the function so that they get carried over to forward_run.py by PstFrom
        import os
        import pandas as pd
    
        def write_tdif_obs(orgf, newf, ws='.'):
            df = pd.read_csv(os.path.join(ws,orgf), index_col='time')
            df = df - df.iloc[0, :]
            df.to_csv(os.path.join(ws,newf))
            return
    
        # write the tdiff observation csv's
        write_tdif_obs('heads.csv', 'heads.tdiff.csv', ws)
        write_tdif_obs('sfr.csv', 'sfr.tdiff.csv', ws)
    
        #write the vdiff obs csv
        # this is frought with the potential for bugs, but oh well...
        df = pd.read_csv(os.path.join(ws,'heads.csv'), index_col='time')
        df.sort_index(axis=1, inplace=True)
        dh = df.loc[:, [i for i in df.columns if i.startswith('TRGW-0-')]]
        dh = dh - df.loc[:, [i for i in df.columns if i.startswith('TRGW-2-')]].values
        dh.to_csv(os.path.join(ws,'heads.vdiff.csv'))
    
        print('Secondary observation files processed.')
        return
    
    
    
    def main():
    
        try:
           os.remove(r'heads.csv')
        except Exception as e:
           print(r'error removing tmp file:heads.csv')
        try:
           os.remove(r'sfr.csv')
        except Exception as e:
           print(r'error removing tmp file:sfr.csv')
        try:
           os.remove(r'hdslay1_t1.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay1_t1.txt')
        try:
           os.remove(r'hdslay1_t10.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay1_t10.txt')
        try:
           os.remove(r'hdslay1_t11.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay1_t11.txt')
        try:
           os.remove(r'hdslay1_t12.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay1_t12.txt')
        try:
           os.remove(r'hdslay1_t13.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay1_t13.txt')
        try:
           os.remove(r'hdslay1_t14.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay1_t14.txt')
        try:
           os.remove(r'hdslay1_t15.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay1_t15.txt')
        try:
           os.remove(r'hdslay1_t16.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay1_t16.txt')
        try:
           os.remove(r'hdslay1_t17.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay1_t17.txt')
        try:
           os.remove(r'hdslay1_t18.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay1_t18.txt')
        try:
           os.remove(r'hdslay1_t19.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay1_t19.txt')
        try:
           os.remove(r'hdslay1_t2.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay1_t2.txt')
        try:
           os.remove(r'hdslay1_t20.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay1_t20.txt')
        try:
           os.remove(r'hdslay1_t21.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay1_t21.txt')
        try:
           os.remove(r'hdslay1_t22.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay1_t22.txt')
        try:
           os.remove(r'hdslay1_t23.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay1_t23.txt')
        try:
           os.remove(r'hdslay1_t24.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay1_t24.txt')
        try:
           os.remove(r'hdslay1_t25.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay1_t25.txt')
        try:
           os.remove(r'hdslay1_t3.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay1_t3.txt')
        try:
           os.remove(r'hdslay1_t4.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay1_t4.txt')
        try:
           os.remove(r'hdslay1_t5.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay1_t5.txt')
        try:
           os.remove(r'hdslay1_t6.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay1_t6.txt')
        try:
           os.remove(r'hdslay1_t7.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay1_t7.txt')
        try:
           os.remove(r'hdslay1_t8.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay1_t8.txt')
        try:
           os.remove(r'hdslay1_t9.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay1_t9.txt')
        try:
           os.remove(r'hdslay2_t1.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay2_t1.txt')
        try:
           os.remove(r'hdslay2_t10.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay2_t10.txt')
        try:
           os.remove(r'hdslay2_t11.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay2_t11.txt')
        try:
           os.remove(r'hdslay2_t12.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay2_t12.txt')
        try:
           os.remove(r'hdslay2_t13.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay2_t13.txt')
        try:
           os.remove(r'hdslay2_t14.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay2_t14.txt')
        try:
           os.remove(r'hdslay2_t15.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay2_t15.txt')
        try:
           os.remove(r'hdslay2_t16.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay2_t16.txt')
        try:
           os.remove(r'hdslay2_t17.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay2_t17.txt')
        try:
           os.remove(r'hdslay2_t18.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay2_t18.txt')
        try:
           os.remove(r'hdslay2_t19.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay2_t19.txt')
        try:
           os.remove(r'hdslay2_t2.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay2_t2.txt')
        try:
           os.remove(r'hdslay2_t20.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay2_t20.txt')
        try:
           os.remove(r'hdslay2_t21.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay2_t21.txt')
        try:
           os.remove(r'hdslay2_t22.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay2_t22.txt')
        try:
           os.remove(r'hdslay2_t23.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay2_t23.txt')
        try:
           os.remove(r'hdslay2_t24.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay2_t24.txt')
        try:
           os.remove(r'hdslay2_t25.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay2_t25.txt')
        try:
           os.remove(r'hdslay2_t3.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay2_t3.txt')
        try:
           os.remove(r'hdslay2_t4.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay2_t4.txt')
        try:
           os.remove(r'hdslay2_t5.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay2_t5.txt')
        try:
           os.remove(r'hdslay2_t6.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay2_t6.txt')
        try:
           os.remove(r'hdslay2_t7.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay2_t7.txt')
        try:
           os.remove(r'hdslay2_t8.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay2_t8.txt')
        try:
           os.remove(r'hdslay2_t9.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay2_t9.txt')
        try:
           os.remove(r'hdslay3_t1.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay3_t1.txt')
        try:
           os.remove(r'hdslay3_t10.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay3_t10.txt')
        try:
           os.remove(r'hdslay3_t11.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay3_t11.txt')
        try:
           os.remove(r'hdslay3_t12.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay3_t12.txt')
        try:
           os.remove(r'hdslay3_t13.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay3_t13.txt')
        try:
           os.remove(r'hdslay3_t14.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay3_t14.txt')
        try:
           os.remove(r'hdslay3_t15.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay3_t15.txt')
        try:
           os.remove(r'hdslay3_t16.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay3_t16.txt')
        try:
           os.remove(r'hdslay3_t17.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay3_t17.txt')
        try:
           os.remove(r'hdslay3_t18.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay3_t18.txt')
        try:
           os.remove(r'hdslay3_t19.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay3_t19.txt')
        try:
           os.remove(r'hdslay3_t2.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay3_t2.txt')
        try:
           os.remove(r'hdslay3_t20.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay3_t20.txt')
        try:
           os.remove(r'hdslay3_t21.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay3_t21.txt')
        try:
           os.remove(r'hdslay3_t22.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay3_t22.txt')
        try:
           os.remove(r'hdslay3_t23.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay3_t23.txt')
        try:
           os.remove(r'hdslay3_t24.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay3_t24.txt')
        try:
           os.remove(r'hdslay3_t25.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay3_t25.txt')
        try:
           os.remove(r'hdslay3_t3.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay3_t3.txt')
        try:
           os.remove(r'hdslay3_t4.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay3_t4.txt')
        try:
           os.remove(r'hdslay3_t5.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay3_t5.txt')
        try:
           os.remove(r'hdslay3_t6.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay3_t6.txt')
        try:
           os.remove(r'hdslay3_t7.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay3_t7.txt')
        try:
           os.remove(r'hdslay3_t8.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay3_t8.txt')
        try:
           os.remove(r'hdslay3_t9.txt')
        except Exception as e:
           print(r'error removing tmp file:hdslay3_t9.txt')
        try:
           os.remove(r'inc.csv')
        except Exception as e:
           print(r'error removing tmp file:inc.csv')
        try:
           os.remove(r'cum.csv')
        except Exception as e:
           print(r'error removing tmp file:cum.csv')
        try:
           os.remove(r'sfr.tdiff.csv')
        except Exception as e:
           print(r'error removing tmp file:sfr.tdiff.csv')
        try:
           os.remove(r'heads.tdiff.csv')
        except Exception as e:
           print(r'error removing tmp file:heads.tdiff.csv')
        try:
           os.remove(r'heads.vdiff.csv')
        except Exception as e:
           print(r'error removing tmp file:heads.vdiff.csv')
        pyemu.helpers.apply_list_and_array_pars(arr_par_file='mult2model_info.csv',chunk_len=50)
        pyemu.os_utils.run(r'mf6')
    
        pyemu.os_utils.run(r'mp7 freyberg_mp.mpsim')
    
        extract_hds_arrays_and_list_dfs()
        process_secondary_obs(ws='.')
    
    if __name__ == '__main__':
        mp.freeze_support()
        main()
    
    

Now we see that `extract_hds_array_and_list_dfs()` has been added to the forward run script and it is being called after MF6 runs. 


```python
obs = pst.observation_data
obs
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>obsnme</th>
      <th>obsval</th>
      <th>weight</th>
      <th>obgnme</th>
      <th>oname</th>
      <th>otype</th>
      <th>usecol</th>
      <th>time</th>
      <th>i</th>
      <th>j</th>
      <th>totim</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3652.5</th>
      <td>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3652.5</td>
      <td>34.326872</td>
      <td>1.0</td>
      <td>oname:hds_otype:lst_usecol:trgw-0-13-10</td>
      <td>hds</td>
      <td>lst</td>
      <td>trgw-0-13-10</td>
      <td>3652.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3683.5</th>
      <td>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3683.5</td>
      <td>34.440950</td>
      <td>1.0</td>
      <td>oname:hds_otype:lst_usecol:trgw-0-13-10</td>
      <td>hds</td>
      <td>lst</td>
      <td>trgw-0-13-10</td>
      <td>3683.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3712.5</th>
      <td>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3712.5</td>
      <td>34.534811</td>
      <td>1.0</td>
      <td>oname:hds_otype:lst_usecol:trgw-0-13-10</td>
      <td>hds</td>
      <td>lst</td>
      <td>trgw-0-13-10</td>
      <td>3712.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3743.5</th>
      <td>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3743.5</td>
      <td>34.582875</td>
      <td>1.0</td>
      <td>oname:hds_otype:lst_usecol:trgw-0-13-10</td>
      <td>hds</td>
      <td>lst</td>
      <td>trgw-0-13-10</td>
      <td>3743.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3773.5</th>
      <td>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3773.5</td>
      <td>34.561764</td>
      <td>1.0</td>
      <td>oname:hds_otype:lst_usecol:trgw-0-13-10</td>
      <td>hds</td>
      <td>lst</td>
      <td>trgw-0-13-10</td>
      <td>3773.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>oname:hdsvd_otype:lst_usecol:trgw-0-9-1_time:4261.5</th>
      <td>oname:hdsvd_otype:lst_usecol:trgw-0-9-1_time:4261.5</td>
      <td>0.006427</td>
      <td>1.0</td>
      <td>oname:hdsvd_otype:lst_usecol:trgw-0-9-1</td>
      <td>hdsvd</td>
      <td>lst</td>
      <td>trgw-0-9-1</td>
      <td>4261.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>oname:hdsvd_otype:lst_usecol:trgw-0-9-1_time:4291.5</th>
      <td>oname:hdsvd_otype:lst_usecol:trgw-0-9-1_time:4291.5</td>
      <td>0.006136</td>
      <td>1.0</td>
      <td>oname:hdsvd_otype:lst_usecol:trgw-0-9-1</td>
      <td>hdsvd</td>
      <td>lst</td>
      <td>trgw-0-9-1</td>
      <td>4291.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>oname:hdsvd_otype:lst_usecol:trgw-0-9-1_time:4322.5</th>
      <td>oname:hdsvd_otype:lst_usecol:trgw-0-9-1_time:4322.5</td>
      <td>0.005765</td>
      <td>1.0</td>
      <td>oname:hdsvd_otype:lst_usecol:trgw-0-9-1</td>
      <td>hdsvd</td>
      <td>lst</td>
      <td>trgw-0-9-1</td>
      <td>4322.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>oname:hdsvd_otype:lst_usecol:trgw-0-9-1_time:4352.5</th>
      <td>oname:hdsvd_otype:lst_usecol:trgw-0-9-1_time:4352.5</td>
      <td>0.005441</td>
      <td>1.0</td>
      <td>oname:hdsvd_otype:lst_usecol:trgw-0-9-1</td>
      <td>hdsvd</td>
      <td>lst</td>
      <td>trgw-0-9-1</td>
      <td>4352.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>oname:hdsvd_otype:lst_usecol:trgw-0-9-1_time:4383.5</th>
      <td>oname:hdsvd_otype:lst_usecol:trgw-0-9-1_time:4383.5</td>
      <td>0.005251</td>
      <td>1.0</td>
      <td>oname:hdsvd_otype:lst_usecol:trgw-0-9-1</td>
      <td>hdsvd</td>
      <td>lst</td>
      <td>trgw-0-9-1</td>
      <td>4383.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>62225 rows × 11 columns</p>
</div>



### 7. After Building the Control File

At this point, we can do some additional modifications that would typically be done that are problem specific.  Here we can tweak the setup, specifying things such as observation weights, parameter bounds, transforms, control data, etc.

Note that any modifications made after calling `PstFrom.build_pst()` will only exist in memory - you need to call `pf.pst.write()` to record these changes to the control file on disk.  Also note that if you call `PstFrom.build_pst()` after making some changes, these changes will be lost.  

For the current case, the main thing we haven't addressed are the observations from custom *.ins files,  observation weights, parameter group INCTYP's and forecasts.

We will do so now.

#### 7.1. Add Observations from INS files

Recall that we wish to include observations of particle end time and status. As mentioned earlier, MP7 output files are not ina nicely organized tabular format - so we need to construct a custom instruction file. We will do this now:


```python
# write a really simple instruction file to read the MODPATH end point file
out_file = "freyberg_mp.mpend"
ins_file = out_file + ".ins"
with open(os.path.join(template_ws, ins_file),'w') as f:
    f.write("pif ~\n")
    f.write("l7 w w w w !part_status! w w !part_time!\n")

```

Now add these observations to the `Pst`:


```python
pst.add_observations(ins_file=os.path.join(template_ws, ins_file),
                    out_file=os.path.join(template_ws, out_file),
                            pst_path='.')

# and then check what changed                            
obs = pst.observation_data
obs.loc[obs.obsnme=='part_status', 'obgnme'] = 'part'
obs.loc[obs.obsnme=='part_time', 'obgnme'] = 'part'

obs.iloc[-2:]
```

    2 obs added from instruction file freyberg6_template\.\freyberg_mp.mpend.ins
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>obsnme</th>
      <th>obsval</th>
      <th>weight</th>
      <th>obgnme</th>
      <th>oname</th>
      <th>otype</th>
      <th>usecol</th>
      <th>time</th>
      <th>i</th>
      <th>j</th>
      <th>totim</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>part_status</th>
      <td>part_status</td>
      <td>3.0000</td>
      <td>1.0</td>
      <td>part</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>part_time</th>
      <td>part_time</td>
      <td>211849.2446</td>
      <td>1.0</td>
      <td>part</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



#### 7.2. Parameters with Zero as Intial Value

Recall that we assigned additive parameters to the GHB heads. Our initial parameter values for these parameter types were set as 0 (zero). This creates a wee bit of trouble when calculating derivatives. There are a couple of ways we could get around it. One way is to add an "offset" to the parameter intial values and to the parmaeter bounds. Another is to use "absolute" increment types (INCTYP). See the PEST manual or PEST++ user guides for descriptions of increment types. 

We will apply both here. 

We will assign INCTYP as 'absolute'. We will leave DERINC as 0.01 (the default). It is a reasonable value in this case.


```python
head_pargps = [i for i in pst.adj_par_groups if 'head' in i]
head_pargps
```




    ['ghbheadgr', 'ghbheadcn']




```python
pst.parameter_groups.loc[head_pargps, 'inctyp'] = 'absolute'
```

Now add the "offset" to parameter data entries:


```python
par = pst.parameter_data
par_names = par.loc[par.parval1==0].parnme

par.loc[par_names].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>parnme</th>
      <th>partrans</th>
      <th>parchglim</th>
      <th>parval1</th>
      <th>parlbnd</th>
      <th>parubnd</th>
      <th>pargp</th>
      <th>scale</th>
      <th>offset</th>
      <th>dercom</th>
      <th>...</th>
      <th>pstyle</th>
      <th>i</th>
      <th>j</th>
      <th>x</th>
      <th>y</th>
      <th>zone</th>
      <th>usecol</th>
      <th>idx0</th>
      <th>idx1</th>
      <th>idx2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>pname:ghbheadgr_inst:0_ptype:gr_usecol:3_pstyle:a_idx0:1_idx1:39_idx2:8</th>
      <td>pname:ghbheadgr_inst:0_ptype:gr_usecol:3_pstyle:a_idx0:1_idx1:39_idx2:8</td>
      <td>none</td>
      <td>relative</td>
      <td>0.0</td>
      <td>-2.0</td>
      <td>2.0</td>
      <td>ghbheadgr</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>...</td>
      <td>a</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3</td>
      <td>1</td>
      <td>39</td>
      <td>8</td>
    </tr>
    <tr>
      <th>pname:ghbheadgr_inst:0_ptype:gr_usecol:3_pstyle:a_idx0:2_idx1:39_idx2:13</th>
      <td>pname:ghbheadgr_inst:0_ptype:gr_usecol:3_pstyle:a_idx0:2_idx1:39_idx2:13</td>
      <td>none</td>
      <td>relative</td>
      <td>0.0</td>
      <td>-2.0</td>
      <td>2.0</td>
      <td>ghbheadgr</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>...</td>
      <td>a</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3</td>
      <td>2</td>
      <td>39</td>
      <td>13</td>
    </tr>
    <tr>
      <th>pname:ghbheadgr_inst:0_ptype:gr_usecol:3_pstyle:a_idx0:1_idx1:39_idx2:11</th>
      <td>pname:ghbheadgr_inst:0_ptype:gr_usecol:3_pstyle:a_idx0:1_idx1:39_idx2:11</td>
      <td>none</td>
      <td>relative</td>
      <td>0.0</td>
      <td>-2.0</td>
      <td>2.0</td>
      <td>ghbheadgr</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>...</td>
      <td>a</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3</td>
      <td>1</td>
      <td>39</td>
      <td>11</td>
    </tr>
    <tr>
      <th>pname:ghbheadgr_inst:0_ptype:gr_usecol:3_pstyle:a_idx0:1_idx1:39_idx2:14</th>
      <td>pname:ghbheadgr_inst:0_ptype:gr_usecol:3_pstyle:a_idx0:1_idx1:39_idx2:14</td>
      <td>none</td>
      <td>relative</td>
      <td>0.0</td>
      <td>-2.0</td>
      <td>2.0</td>
      <td>ghbheadgr</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>...</td>
      <td>a</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3</td>
      <td>1</td>
      <td>39</td>
      <td>14</td>
    </tr>
    <tr>
      <th>pname:ghbheadgr_inst:0_ptype:gr_usecol:3_pstyle:a_idx0:0_idx1:39_idx2:9</th>
      <td>pname:ghbheadgr_inst:0_ptype:gr_usecol:3_pstyle:a_idx0:0_idx1:39_idx2:9</td>
      <td>none</td>
      <td>relative</td>
      <td>0.0</td>
      <td>-2.0</td>
      <td>2.0</td>
      <td>ghbheadgr</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>...</td>
      <td>a</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3</td>
      <td>0</td>
      <td>39</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
offset = -10
par.loc[par_names, 'offset'] = offset
par.loc[par_names, ['parval1', 'parlbnd', 'parubnd']] -= offset

par.loc[par_names].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>parnme</th>
      <th>partrans</th>
      <th>parchglim</th>
      <th>parval1</th>
      <th>parlbnd</th>
      <th>parubnd</th>
      <th>pargp</th>
      <th>scale</th>
      <th>offset</th>
      <th>dercom</th>
      <th>...</th>
      <th>pstyle</th>
      <th>i</th>
      <th>j</th>
      <th>x</th>
      <th>y</th>
      <th>zone</th>
      <th>usecol</th>
      <th>idx0</th>
      <th>idx1</th>
      <th>idx2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>pname:ghbheadgr_inst:0_ptype:gr_usecol:3_pstyle:a_idx0:1_idx1:39_idx2:8</th>
      <td>pname:ghbheadgr_inst:0_ptype:gr_usecol:3_pstyle:a_idx0:1_idx1:39_idx2:8</td>
      <td>none</td>
      <td>relative</td>
      <td>10.0</td>
      <td>8.0</td>
      <td>12.0</td>
      <td>ghbheadgr</td>
      <td>1.0</td>
      <td>-10.0</td>
      <td>1</td>
      <td>...</td>
      <td>a</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3</td>
      <td>1</td>
      <td>39</td>
      <td>8</td>
    </tr>
    <tr>
      <th>pname:ghbheadgr_inst:0_ptype:gr_usecol:3_pstyle:a_idx0:2_idx1:39_idx2:13</th>
      <td>pname:ghbheadgr_inst:0_ptype:gr_usecol:3_pstyle:a_idx0:2_idx1:39_idx2:13</td>
      <td>none</td>
      <td>relative</td>
      <td>10.0</td>
      <td>8.0</td>
      <td>12.0</td>
      <td>ghbheadgr</td>
      <td>1.0</td>
      <td>-10.0</td>
      <td>1</td>
      <td>...</td>
      <td>a</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3</td>
      <td>2</td>
      <td>39</td>
      <td>13</td>
    </tr>
    <tr>
      <th>pname:ghbheadgr_inst:0_ptype:gr_usecol:3_pstyle:a_idx0:1_idx1:39_idx2:11</th>
      <td>pname:ghbheadgr_inst:0_ptype:gr_usecol:3_pstyle:a_idx0:1_idx1:39_idx2:11</td>
      <td>none</td>
      <td>relative</td>
      <td>10.0</td>
      <td>8.0</td>
      <td>12.0</td>
      <td>ghbheadgr</td>
      <td>1.0</td>
      <td>-10.0</td>
      <td>1</td>
      <td>...</td>
      <td>a</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3</td>
      <td>1</td>
      <td>39</td>
      <td>11</td>
    </tr>
    <tr>
      <th>pname:ghbheadgr_inst:0_ptype:gr_usecol:3_pstyle:a_idx0:1_idx1:39_idx2:14</th>
      <td>pname:ghbheadgr_inst:0_ptype:gr_usecol:3_pstyle:a_idx0:1_idx1:39_idx2:14</td>
      <td>none</td>
      <td>relative</td>
      <td>10.0</td>
      <td>8.0</td>
      <td>12.0</td>
      <td>ghbheadgr</td>
      <td>1.0</td>
      <td>-10.0</td>
      <td>1</td>
      <td>...</td>
      <td>a</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3</td>
      <td>1</td>
      <td>39</td>
      <td>14</td>
    </tr>
    <tr>
      <th>pname:ghbheadgr_inst:0_ptype:gr_usecol:3_pstyle:a_idx0:0_idx1:39_idx2:9</th>
      <td>pname:ghbheadgr_inst:0_ptype:gr_usecol:3_pstyle:a_idx0:0_idx1:39_idx2:9</td>
      <td>none</td>
      <td>relative</td>
      <td>10.0</td>
      <td>8.0</td>
      <td>12.0</td>
      <td>ghbheadgr</td>
      <td>1.0</td>
      <td>-10.0</td>
      <td>1</td>
      <td>...</td>
      <td>a</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3</td>
      <td>0</td>
      <td>39</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>



#### 7.4. Forecasts

For most models there is a forecast/prediction that someone needs. Rather than waiting until the end of the project, the forecast should be entered into your thinking and workflow __right at the beginning__.  Here we do this explicitly by monitoring the forecasts as "observations" in the control file.  This way, for every PEST(++) analysis we do, we can watch what is happening to the forecasts - #winning **

The optional PEST++ `++forecasts` control variable allows us to provide the names of one or more observations featured in the “observation data” section of the PEST control file; these are treated as predictions in FOSM predictive uncertainty analysis by PESTPP-GLM. It is also a convenient way to keep track of "forecast" observations (makes post-processing a wee bit easier later on).

Recall that, for our synthetic case we are interested in forecasting:

 - groundwater level in the upper layer at row 9 and column 1 (site named "trgw-0-9-1") in stress period 22 (time=640);
 - the "tailwater" surface-water/groundwater exchange during stress period 13 (time=367); and
 - the "headwater" surface-water/groundwater exchange at stress period 22 (time=640).
 - the particle travel time.



```python
forecasts =[
            'oname:sfr_otype:lst_usecol:tailwater_time:4383.5',
            'oname:sfr_otype:lst_usecol:headwater_time:4383.5',
            'oname:hds_otype:lst_usecol:trgw-0-9-1_time:4383.5',
            'part_time'
            ]

forecasts
```




    ['oname:sfr_otype:lst_usecol:tailwater_time:4383.5',
     'oname:sfr_otype:lst_usecol:headwater_time:4383.5',
     'oname:hds_otype:lst_usecol:trgw-0-9-1_time:4383.5',
     'part_time']




```python
fobs = obs.loc[forecasts,:]
fobs
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>obsnme</th>
      <th>obsval</th>
      <th>weight</th>
      <th>obgnme</th>
      <th>oname</th>
      <th>otype</th>
      <th>usecol</th>
      <th>time</th>
      <th>i</th>
      <th>j</th>
      <th>totim</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>oname:sfr_otype:lst_usecol:tailwater_time:4383.5</th>
      <td>oname:sfr_otype:lst_usecol:tailwater_time:4383.5</td>
      <td>-519.184506</td>
      <td>1.0</td>
      <td>oname:sfr_otype:lst_usecol:tailwater</td>
      <td>sfr</td>
      <td>lst</td>
      <td>tailwater</td>
      <td>4383.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>oname:sfr_otype:lst_usecol:headwater_time:4383.5</th>
      <td>oname:sfr_otype:lst_usecol:headwater_time:4383.5</td>
      <td>-694.299524</td>
      <td>1.0</td>
      <td>oname:sfr_otype:lst_usecol:headwater</td>
      <td>sfr</td>
      <td>lst</td>
      <td>headwater</td>
      <td>4383.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>oname:hds_otype:lst_usecol:trgw-0-9-1_time:4383.5</th>
      <td>oname:hds_otype:lst_usecol:trgw-0-9-1_time:4383.5</td>
      <td>34.809963</td>
      <td>1.0</td>
      <td>oname:hds_otype:lst_usecol:trgw-0-9-1</td>
      <td>hds</td>
      <td>lst</td>
      <td>trgw-0-9-1</td>
      <td>4383.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>part_time</th>
      <td>part_time</td>
      <td>211849.244600</td>
      <td>1.0</td>
      <td>part</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



We will just set this optional pest++ argument because it will trigger certain automatic behavior later in PESTPP-GLM


```python
pst.pestpp_options['forecasts'] = forecasts
```

#### 7.5. Re-write the Control File!

Make sure to re-**write** the PEST control file. But beware, if you re-**build** the `Pst`, all these changes will be lost.


```python
pst.write(os.path.join(template_ws, 'freyberg_mf6.pst'))
```

    noptmax:0, npar_adj:12013, nnz_obs:62227
    

So that was pretty epic. We now have a (very) high-dimensional PEST interface that includes secondary observations, as well as forecasts, ready to roll. 

If you inspect the folder, you will see PEST control file and all the necessary instruction and template files. Because we have >10k parameters, version 2 of the PEST control file was written by default. 

Shall we check that it works? Let's run PEST once (i.e. with NOPTMAX=0). Now, by default, noptmax is set to zero. But just to check:


```python
pst.control_data.noptmax
```




    0



OK, so when we run PEST it will call the model once and then stop. If the next cell is sucessfull, then eveything is working. Check the folder, you should see PEST output files. (We will go into these and how to process PEST outcomes in subsequent tutorials).


```python
pyemu.os_utils.run('pestpp-glm freyberg_mf6.pst', cwd=template_ws)
```

Recall that we assigned observation values generated from the "base model run"? If we setup everything correctly, this means that PEST should have obtained residuals very close to zero. As mentioned, this is a good way to check for problems early on.

Let's check the Phi recorded in the *.iobj file (could also check the *.rec or *.rei files).


```python
# read the file
iobj = pd.read_csv(os.path.join(template_ws, 'freyberg_mf6.iobj'))

# check value in phi column
iobj.total_phi
```




    0    0
    Name: total_phi, dtype: int64



 Sweet! Zero. All is well.

### 8. Prior Parameter Covariance Matrix

One the major reasons `PstFrom` was built is to help with building the Prior - both covariance matrix and ensemble - with geostatical correlation.  Remember all that business above related to geostatical structures and correlations?  This is where is pays off.

Let's see how this works.  For cases with less than about 30,000 parameters, we can actually generate and visualize the prior parameter covariance matrix.  If you have more parameters, this matrix may not fit in memory.  But, not to worry, `PstFrom` has some trickery to help generate the geostatistical prior ensemble.


```python
# build the prior covariance matrix and store it as a compresed bianry file (otherwise it can get huge!)
# depending on your machine, this may take a while...
if pf.pst.npar < 35000:  #if you have more than about 35K pars, the cov matrix becomes hard to handle
    cov = pf.build_prior(fmt='coo', filename=os.path.join(template_ws,"prior_cov.jcb"))
    # and take a peek at a slice of the matrix
    try: 
        x = cov.x.copy()
        x[x==0] = np.NaN
        plt.imshow(x[:1000,:1000])
    except:
        pass
```


    
![png](freyberg_pstfrom_pest_setup_files/freyberg_pstfrom_pest_setup_146_0.png)
    


snap!  That big block must be a grid-scale parameter group...


```python
cov.row_names[:10]
```




    ['pname:npfklayer1gr_inst:0_ptype:gr_pstyle:m_i:0_j:0_x:125.00_y:9875.00_zone:1',
     'pname:npfklayer1gr_inst:0_ptype:gr_pstyle:m_i:0_j:1_x:375.00_y:9875.00_zone:1',
     'pname:npfklayer1gr_inst:0_ptype:gr_pstyle:m_i:0_j:2_x:625.00_y:9875.00_zone:1',
     'pname:npfklayer1gr_inst:0_ptype:gr_pstyle:m_i:0_j:3_x:875.00_y:9875.00_zone:1',
     'pname:npfklayer1gr_inst:0_ptype:gr_pstyle:m_i:0_j:4_x:1125.00_y:9875.00_zone:1',
     'pname:npfklayer1gr_inst:0_ptype:gr_pstyle:m_i:0_j:5_x:1375.00_y:9875.00_zone:1',
     'pname:npfklayer1gr_inst:0_ptype:gr_pstyle:m_i:0_j:6_x:1625.00_y:9875.00_zone:1',
     'pname:npfklayer1gr_inst:0_ptype:gr_pstyle:m_i:0_j:7_x:1875.00_y:9875.00_zone:1',
     'pname:npfklayer1gr_inst:0_ptype:gr_pstyle:m_i:0_j:8_x:2125.00_y:9875.00_zone:1',
     'pname:npfklayer1gr_inst:0_ptype:gr_pstyle:m_i:0_j:9_x:2375.00_y:9875.00_zone:1']



And now generate a prior parameter ensemble. This step is relevant for using pestpp-ies in subsequent tutorials. Note: you do not have to call `build_prior()` before calling `draw()`!


```python
pe = pf.draw(num_reals=50, use_specsim=True) # draw parameters from the prior distribution
pe.enforce() # enforces parameter bounds
pe.to_binary(os.path.join(template_ws,"prior_pe.jcb")) #writes the paramter ensemble to binary file
assert pe.shape[1] == pst.npar
```

    SpecSim.initialize() summary: full_delx X full_dely: 72 X 72
    SpecSim.initialize() summary: full_delx X full_dely: 72 X 72
    SpecSim.initialize() summary: full_delx X full_dely: 72 X 72
    SpecSim.initialize() summary: full_delx X full_dely: 72 X 72
    SpecSim.initialize() summary: full_delx X full_dely: 72 X 72
    SpecSim.initialize() summary: full_delx X full_dely: 72 X 72
    SpecSim.initialize() summary: full_delx X full_dely: 72 X 72
    SpecSim.initialize() summary: full_delx X full_dely: 72 X 72
    SpecSim.initialize() summary: full_delx X full_dely: 72 X 72
    SpecSim.initialize() summary: full_delx X full_dely: 72 X 72
    SpecSim.initialize() summary: full_delx X full_dely: 72 X 72
    SpecSim.initialize() summary: full_delx X full_dely: 72 X 72
    SpecSim.initialize() summary: full_delx X full_dely: 72 X 72
    SpecSim.initialize() summary: full_delx X full_dely: 72 X 72
    SpecSim.initialize() summary: full_delx X full_dely: 72 X 72
    building diagonal cov
    processing  name:struct1,nugget:0.0,structures:
    name:var1,contribution:1.0,a:1000.0,anisotropy:1.0,bearing:0.0
    
    working on pargroups ['npfklayer1pp']
    build cov matrix
    done
    getting diag var cov 29
    scaling full cov by diag var cov
    working on pargroups ['npfklayer2pp']
    build cov matrix
    done
    getting diag var cov 29
    scaling full cov by diag var cov
    working on pargroups ['npfklayer3pp']
    build cov matrix
    done
    getting diag var cov 29
    scaling full cov by diag var cov
    working on pargroups ['npfk33layer1pp']
    build cov matrix
    done
    getting diag var cov 29
    scaling full cov by diag var cov
    working on pargroups ['npfk33layer2pp']
    build cov matrix
    done
    getting diag var cov 29
    scaling full cov by diag var cov
    working on pargroups ['npfk33layer3pp']
    build cov matrix
    done
    getting diag var cov 29
    scaling full cov by diag var cov
    working on pargroups ['stosslayer2pp']
    build cov matrix
    done
    getting diag var cov 29
    scaling full cov by diag var cov
    working on pargroups ['stosslayer3pp']
    build cov matrix
    done
    getting diag var cov 29
    scaling full cov by diag var cov
    working on pargroups ['stosylayer1pp']
    build cov matrix
    done
    getting diag var cov 29
    scaling full cov by diag var cov
    working on pargroups ['nelayer1pp']
    build cov matrix
    done
    getting diag var cov 29
    scaling full cov by diag var cov
    working on pargroups ['nelayer2pp']
    build cov matrix
    done
    getting diag var cov 29
    scaling full cov by diag var cov
    working on pargroups ['nelayer3pp']
    build cov matrix
    done
    getting diag var cov 29
    scaling full cov by diag var cov
    working on pargroups ['rch_recharge_1pp']
    build cov matrix
    done
    getting diag var cov 29
    scaling full cov by diag var cov
    working on pargroups ['ghbcondgr']
    build cov matrix
    done
    getting diag var cov 10
    scaling full cov by diag var cov
    working on pargroups ['ghbcondgr']
    build cov matrix
    done
    getting diag var cov 10
    scaling full cov by diag var cov
    working on pargroups ['ghbcondgr']
    build cov matrix
    done
    getting diag var cov 10
    scaling full cov by diag var cov
    working on pargroups ['sfrcondgr']
    build cov matrix
    done
    getting diag var cov 40
    scaling full cov by diag var cov
    processing  name:struct1,nugget:0.0,structures:
    name:var1,contribution:1.0,a:60.0,anisotropy:1.0,bearing:0.0
    
    working on pargroups ['welcst']
    build cov matrix
    done
    getting diag var cov 25
    scaling full cov by diag var cov
    working on pargroups ['sfrgr']
    build cov matrix
    done
    getting diag var cov 25
    scaling full cov by diag var cov
    processing  name:struct1,nugget:0.0,structures:
    name:var1,contribution:1.0,a:1000.0,anisotropy:1.0,bearing:0.0
    
    working on pargroups ['ghbheadgr']
    build cov matrix
    done
    getting diag var cov 10
    scaling full cov by diag var cov
    working on pargroups ['ghbheadgr']
    build cov matrix
    done
    getting diag var cov 10
    scaling full cov by diag var cov
    working on pargroups ['ghbheadgr']
    build cov matrix
    done
    getting diag var cov 10
    scaling full cov by diag var cov
    adding remaining parameters to diagonal
    

Let's now test-run one of these geostatistical realizations (always a good idea!).  We do this by replacing the `parval1` values in the control with a row from `pe`:


```python
pst.parameter_data.loc[:,"parval1"] = pe.loc[pe.index[0],pst.par_names].values
pst.parameter_data.parval1.values
```




    array([ 0.72616096,  0.82076147,  1.08770186, ..., 19.43876745,
           37.17823683, 40.77793808])




```python
pst.control_data.noptmax = 0
pst.write(os.path.join(template_ws,"test.pst"))
pyemu.os_utils.run("pestpp-glm test.pst",cwd=template_ws)
```

    noptmax:0, npar_adj:12013, nnz_obs:62227
    

If all went well, that's it! The PEST-interface is setup, tested and we have our prior preprared. We should be good to go!

### Bonus: Understanding Multiplier-Parameters

Now the multiplier files in the "`template_ws`/mult" folder and the MF6 input files in the `template_ws` folder contain the values cooresponding to this realization, so we can visualize the multipler parameter process:


```python
df = pd.read_csv(os.path.join(template_ws,"mult2model_info.csv"))
kh1_df = df.loc[df.model_file.str.contains("npf_k_layer1"),:]
kh1_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>org_file</th>
      <th>model_file</th>
      <th>use_cols</th>
      <th>index_cols</th>
      <th>fmt</th>
      <th>sep</th>
      <th>head_rows</th>
      <th>upper_bound</th>
      <th>lower_bound</th>
      <th>operator</th>
      <th>mlt_file</th>
      <th>zone_file</th>
      <th>fac_file</th>
      <th>pp_file</th>
      <th>pp_fill_value</th>
      <th>pp_lower_limit</th>
      <th>pp_upper_limit</th>
      <th>zero_based</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>org\freyberg6.npf_k_layer1.txt</td>
      <td>freyberg6.npf_k_layer1.txt</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>free</td>
      <td>NaN</td>
      <td>0</td>
      <td>100</td>
      <td>0.01</td>
      <td>m</td>
      <td>mult\npfklayer1gr_inst0_grid.csv</td>
      <td>npfklayer1gr_inst0_grid.csv.zone</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>org\freyberg6.npf_k_layer1.txt</td>
      <td>freyberg6.npf_k_layer1.txt</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>free</td>
      <td>NaN</td>
      <td>0</td>
      <td>100</td>
      <td>0.01</td>
      <td>m</td>
      <td>mult\npfklayer1pp_inst0_pilotpoints.csv</td>
      <td>npfklayer1pp_inst0pp.dat.zone</td>
      <td>npfklayer1pp_inst0pp.fac</td>
      <td>npfklayer1pp_inst0pp.dat</td>
      <td>1.0</td>
      <td>1.000000e-10</td>
      <td>1.000000e+10</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>org\freyberg6.npf_k_layer1.txt</td>
      <td>freyberg6.npf_k_layer1.txt</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>free</td>
      <td>NaN</td>
      <td>0</td>
      <td>100</td>
      <td>0.01</td>
      <td>m</td>
      <td>mult\npfklayer1cn_inst0_constant.csv</td>
      <td>npfklayer1cn_inst0_constant.csv.zone</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
org_arr = np.loadtxt(os.path.join(template_ws,kh1_df.org_file.iloc[0]))
inp_arr = np.loadtxt(os.path.join(template_ws,kh1_df.model_file.iloc[0]))
mlt_arrs = [np.loadtxt(os.path.join(template_ws,afile)) for afile in kh1_df.mlt_file]
arrs = [org_arr]
arrs.extend(mlt_arrs)
arrs.append(inp_arr)
names = ["org"]
names.extend([mf.split('.')[0].split('_')[-1] for mf in kh1_df.mlt_file])
names.append("MF6 input")
fig,axes = plt.subplots(1,kh1_df.shape[0]+2,figsize=(5*kh1_df.shape[0]+2,5))
for i,ax in enumerate(axes.flatten()):
    arr = np.log10(arrs[i])
    arr[ib==0] = np.NaN
    cb = ax.imshow(arr)
    plt.colorbar(cb,ax=ax)
    ax.set_title(names[i],loc="left")
plt.tight_layout()    
    
```


    
![png](freyberg_pstfrom_pest_setup_files/freyberg_pstfrom_pest_setup_157_0.png)
    

