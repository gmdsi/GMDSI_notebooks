---
layout: default
title: PEST++GLM - Calculating a Jacobian Matrix
parent: Decision Support Modelling with pyEMU and PEST++
nav_order: 4
math: mathjax3
---

# Filling a Jacobian Matrix

This notebook is an optional, but recommended, first step for a workflow that implements linear uncertainty analysis (i.e. FOSM), data worth analysis and highly-parameterized regularised inversion (i.e. calibration). It will be relatively short, but provides the foundation for subsequet tutorials. 

Here, we are going to calculate a base Jacobian. In other words, we are going to calculate partial derivatives of model outputs with respect to (adjustable) model parameters. Or "how much each observation value changes for a change in each parameter value".  These partial derivatives (or *sensitivity coefficients*) are fundamental for the implementation of inversion and for linear uncertainty analysis. They form a two-dimensional array of values with as many rows as observations and as many columns as parameters. This array is commonly known as the **Jacobian matrix**. 

PEST and PESTPP-GLM (as well as some other PEST++ versions) calculate and record a Jacobian as part of normal execution. They do so by running "the model" as many times as there are adjustable parameters. Each time, a parameter is adjusted and the corresponding effects on all observations are recorded. These are used to fill in the Jacobian matrix. Once the Jacobian is calculated, the derivative information is used to identify parameter changes that will improve the fit between model outputs and measured data. These are used to update the "calibrated" parameter set. Due to the nonlinear nature of groundwater inverse problems, this process may need to be repeated numerous times during calibration. As you can imagine, if there are many adjustable parameters, this process can take up a lot of computation time. 

Filling the Jacobian is perhaps the main computational cost of derivative-based optimisation methods such as are implemented in PEST and PESTPP-GLM. 

However, this cost is often worth it, as a Jacobian matrix has many uses. Many of these uses are as important as the model calibration process itself. Hence it is not unusual for PEST or PESTPP-GLM to be run purely for the purpose of filling a Jacobian matrix (as we will do here). 

Uses to which a Jacobian matrix may be put include the following:
 - Examination of local sensitivities of model outputs to parameters and/or decision variables.
 - Giving PEST or PESTPP-GLM a “head start” in calibrating a model by providing it with a pre-calculated Jacobian matrix to use in its first iteration. For PESTPP-GLM this is achieved through use of the `base_jacobian()` control variable, as we will demonstrate in a subsequent tutorial.
 - To support the many types of linear analysis implemented by utility programs supplied with PEST, and functions provided by `pyEMU`; these calculate:
    - parameter identifiability;
    - parameter and predictive uncertainty;
    - parameter contributions to predictive uncertainty;
    - data worth;
    - the effects of model defects.


### Admin

Start off with the usual loading of dependencies and preparing model and PEST files. We will be continuing to work with the modified-Freyberg model (see "intro to model" notebook), and the high-dimensional PEST dataset prepared in the "pstfrom pest setup" and "obs and weights" notebooks. 

For the purposes of this notebook, you do not require familiarity with previous notebooks (but it helps...). 

Simply run the next few cells by pressing `shift+enter`.


```python
import os
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import pandas as pd
import matplotlib.pyplot as plt;
import shutil
import psutil

import sys
sys.path.insert(0,os.path.join("..", "..", "dependencies"))
import pyemu
import flopy
assert "dependencies" in flopy.__file__
assert "dependencies" in pyemu.__file__
sys.path.insert(0,"..")
import herebedragons as hbd


```

To maintain continuity in the series of tutorials, we we use the PEST-dataset prepared in the "obs and weigths" tutorial. Run the next cell to copy fthe necessary files across. Note that if you will need to run the previous notebooks in the correct order beforehand.

Specify the path to the PEST dataset template folder. Recall that we will prepare our PEST dataset files in this folder, keeping them separate from the original model files. Then copy across pre-prepared model and PEST files:


```python
# specify the temporary working folder
t_d = os.path.join('freyberg6_template')
if os.path.exists(t_d):
    shutil.rmtree(t_d)

org_t_d = os.path.join("..","part2_2_obs_and_weights","freyberg6_template")
if not os.path.exists(org_t_d):
    raise Exception("you need to run the '/part2_2_obs_and_weights/freyberg_obs_and_weights.ipynb' notebook")

shutil.copytree(org_t_d,t_d)
```




    'freyberg6_template'




```python
pst_path = os.path.join(t_d, 'freyberg_mf6.pst')
```

### Inspect the PEST Dataset

OK. We can now get started.

Load the PEST control file as a `Pst` object. We are going to use the PEST control file that was created in the "pstfrom pest setup" tutorial. This control file has observations with weights equal to the inverse of measurement noise (**not** weighted for visibility!).


```python
pst = pyemu.Pst(pst_path)
```


```python
# check to see if obs&weights notebook has been run
if not pst.observation_data.observed.sum()>0:
    raise Exception("You need to run the '/part2_2_obs_and_weights/freyberg_obs_and_weights.ipynb' notebook")
```

Make a quick parameter summary table as a reminder of what we have in our control file:


```python
pst.write_par_summary_table(filename="none")
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
      <th>type</th>
      <th>transform</th>
      <th>count</th>
      <th>initial value</th>
      <th>lower bound</th>
      <th>upper bound</th>
      <th>standard deviation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ghbcondcn</th>
      <td>ghbcondcn</td>
      <td>log</td>
      <td>1</td>
      <td>0</td>
      <td>-1</td>
      <td>1</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>ghbcondgr</th>
      <td>ghbcondgr</td>
      <td>log</td>
      <td>10</td>
      <td>0</td>
      <td>-1</td>
      <td>1</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>ghbheadcn</th>
      <td>ghbheadcn</td>
      <td>none</td>
      <td>1</td>
      <td>10</td>
      <td>8</td>
      <td>12</td>
      <td>1</td>
    </tr>
    <tr>
      <th>ghbheadgr</th>
      <td>ghbheadgr</td>
      <td>none</td>
      <td>10</td>
      <td>10</td>
      <td>8</td>
      <td>12</td>
      <td>1</td>
    </tr>
    <tr>
      <th>icstrtlayer1</th>
      <td>icstrtlayer1</td>
      <td>none</td>
      <td>706</td>
      <td>32.5287 to 40.1337</td>
      <td>15</td>
      <td>50</td>
      <td>8.75</td>
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
    </tr>
    <tr>
      <th>stosylayer1cn</th>
      <td>stosylayer1cn</td>
      <td>log</td>
      <td>1</td>
      <td>0</td>
      <td>-0.69897</td>
      <td>0.69897</td>
      <td>0.349485</td>
    </tr>
    <tr>
      <th>stosylayer1gr</th>
      <td>stosylayer1gr</td>
      <td>log</td>
      <td>706</td>
      <td>0</td>
      <td>-0.69897</td>
      <td>0.69897</td>
      <td>0.349485</td>
    </tr>
    <tr>
      <th>stosylayer1pp</th>
      <td>stosylayer1pp</td>
      <td>log</td>
      <td>29</td>
      <td>0</td>
      <td>-0.69897</td>
      <td>0.69897</td>
      <td>0.349485</td>
    </tr>
    <tr>
      <th>welcst</th>
      <td>welcst</td>
      <td>log</td>
      <td>25</td>
      <td>0</td>
      <td>-0.60206</td>
      <td>0.60206</td>
      <td>0.30103</td>
    </tr>
    <tr>
      <th>welgrd</th>
      <td>welgrd</td>
      <td>log</td>
      <td>175</td>
      <td>0</td>
      <td>-0.60206</td>
      <td>0.60206</td>
      <td>0.30103</td>
    </tr>
  </tbody>
</table>
<p>103 rows × 7 columns</p>
</div>



Recall that our parameterisation is quite comprehensive, with pilot points and grid based (e.g. cell-by-cell) parameters. 

Let's recall how many adjustable parameters we have:


```python
pst.npar_adj
```




    23786



Quite a lot! How long does the model take to run? Even if it is well under a minute, that can quickly add up. Just to illustrate, let's chekc how long it takes our forwrd run to complete:


```python
import timeit
start_time = timeit.default_timer()

# execute the model forward_run.py script
pyemu.os_utils.run('python forward_run.py', cwd=t_d)

elapsed = timeit.default_timer() - start_time
elapsed
```




    4.694557400000001



So, very roughly, we can estimate how long it will take to fill in a Jacobian matrix. Let's assume we will be running this in parallel with as many agents as we have cores (update `number_of_cpu_cores` according to what you have at your disposal).

You can check how many physical cores you have on your machine with `psutil`:


```python
psutil.cpu_count(logical=False)
```




    10




```python
number_of_cpu_cores = psutil.cpu_count(logical=False)

print(f'Number of hours to fill a jacobian:{pst.npar_adj * elapsed / 60/60 / number_of_cpu_cores}')
```

    Number of hours to fill a jacobian:3.1017983976777783
    

Unless you have many many CPU's at hand, that's still going to be pretty long despite the relatively fast model.

### Good-Bye High-Dimensional Parameterisation!

As previously discussed, the computational cost of conventional model calibration (attained through
adjustment of a single parameter field using partial derivatives calculated using finite parameter
differences) increases with the number of adjustable parameters. This imposes pragmatic limits on the number of adjustable parameters we can have.

We are limited by compute power (e.g. how many parallel model runs can we deploy) and how long each model takes to run. At the end of the day, it will be project time and cost constraints that will pose hard limits on what is acceptible. 

So here comes the painfull part: we can't use these 10's of thousands of parameters. We are going to have to set many of them as "fixed" (e.g. no longer adjustable). We do this by changing the parameter transform value in the `* parameter data` section (e.g. the "partrans" column in `pst.parameter_data`). 


```python
par = pst.parameter_data
```

Let's set all the grid scale parameters as fixed, with the exception of the SFR inflow parameters. That will sort a large amount. The cost is we lose the ability to capture the effects of small-scale heterogeneity. 


```python
# say goodbye to grid-scale pars
gr_pars = par.loc[par.pargp.apply(lambda x: "gr" in x and "sfr" not in x),"parnme"]
par.loc[gr_pars,"partrans"] = "fixed"
pst.npar_adj
```




    1705



Let's fix all recharge pilot point parameters. We will at least still have the layer-scale parameters for these.


```python
rch_pp = [i for i in pst.adj_par_groups if i.startswith('rch') and i.endswith('pp') ]
par.loc[par['pargp'].isin(rch_pp),"partrans"] = "fixed"
pst.npar_adj
```




    980



Fix all those initial head parameters...


```python
icstrt = [i for i in pst.adj_par_groups if i.startswith('icstrt') ]
par.loc[par['pargp'].isin(icstrt),"partrans"] = "fixed"
pst.npar_adj
```




    274



And let's also fix pilot point parameters for storage, and for vertical conductivity ratio in layer 1 and 3. 


```python
fi_grps = [ #'stosslayer3pp',
            'stosslayer2pp',
            #'stosylayer1pp', 
            'npfk33layer1pp',
            'npfk33layer3pp',
            ]
par.loc[par.pargp.apply(lambda x: x in fi_grps),"partrans"] = "fixed"

pst.npar_adj
```




    245



OK, let's check that estimate of run time again...hmm...a bit more manageable. Of course, the cost of this has been a loss of flexibility in our parameterisation scheme. This means we are potentialy less able to fit historical data...but worse, we are also less able to capture the effect of uncertianty from these fixed parameters on model forecasts.


```python
print(f'Number of hours to fill a jacobian:{pst.npar_adj * elapsed / 60/60 / number_of_cpu_cores}')
```

    Number of hours to fill a jacobian:0.03194907119444445
    

OK, if we are happy (#sadface) with the number of parameters, we can move on.

To instruct PEST or PEST++GLM to only calculate the Jacobian and then stop, we assign a value of -1 or -2 to the NOPTMAX control value. Like so:


```python
pst.control_data.noptmax = -1
```

We are now ready to go. Let's re-write the control file. We will record this with a new name: `freyberg_pp.pst`. 


```python
pst.write(os.path.join(t_d,"freyberg_pp.pst"))
```

    noptmax:-1, npar_adj:245, nnz_obs:72
    

### Run PEST++GLM

Alright! Let's run this thing!

As we saw in the "freyberg prior monte carlo" notebook, we can use `pyemu` to deploy PEST in parallel. 

To speed up the process, you will want to distribute the workload across as many parallel agents as possible. Normally, you will want to use the same number of agents (or less) as you have available CPU cores. Most personal computers (i.e. desktops or laptops) these days have between 4 and 10 cores. Servers or HPCs may have many more cores than this. Another limitation to keep in mind is the read/write speed of your machines disk (e.g. your hard drive). PEST and the model software are going to be reading and writting lots of files. This often slows things down if agents are competing for the same resources to read/write to disk. (It also wears through SSD drives...)

The first thing we will do is specify the number of agents we are going to use.

# Attention!

You must specify the number which is adequate for ***your*** machine! Make sure to assign an appropriate value for the following `num_workers` variable:


```python
num_workers = psutil.cpu_count(logical=False) # update according to your available resources!
```

Then specify the folder in which the PEST manager will run and record outcomes. It should be different from the `t_d` folder. 


```python
m_d = os.path.join('master_glm_1')
```

The following cell deploys the PEST agents and manager and then starts the run using `pestpp-glm`. Run it by pressing `shift+enter`.

If you wish to see the outputs in real-time, switch over to the terminal window (the one which you used to launch the `jupyter notebook`). There you should see `pestpp-glm`'s progress. 

If you open the tutorial folder, you should also see a bunch of new folders there named `worker_0`, `worker_1`, etc. These are the agent folders. `pyemu` will remove them when PEST finishes running.

This run should take a while to complete (depending on the number of workers and the speed of your machine). If you get an error, make sure that your firewall or antivirus software is not blocking `pestpp-glm` from communicating with the agents (this is a common problem!).


```python
pyemu.os_utils.start_workers(t_d,"pestpp-glm","freyberg_pp.pst",num_workers=num_workers,worker_root=".",
                           master_dir=m_d)
```
