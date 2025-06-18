---
layout: default
title: PEST++DA - Sequential Data Assimilation
parent: Decision Support Modelling with pyEMU and PEST++
nav_order: 11
math: mathjax3
---

# PESTPP-DA - Generalized Data Assimilation with Ensemble Methods
## Run the beast!

Much Like PESTPP-IES, PESTPP-DA uses ensembles to approximates first-order relationships between inputs (i.e. parameters) and outputs (i.e. observations).  However, PESTPP-DA extends and generalizes the approach of PESTPP-IES (and all other tools in the PEST and PEST++) to the concept of an assimilation "cycle".  Each cycle represents a discrete inverse problems with (potentially) unique parameters and observations.  Most commonly, these cycles represent discrete time intervals, for example, a year.  During each cycle, the parameters active during the cycle are conditioned by assimilating observations active during the cycle. This is referred to as "sequential estimation".  The real mind twister is that the posterior at the end of each cycle is the prior for the next cycle...confused yet?  Conceptually and theoretically, Bayes equation can be split out this way...

The assimilation engine used in each cycle by PESTPP-DA is the same engine used in PESTPP-IES:  the iterative ensemble method, using either the GLM algorithm of Chen and Oliver or the multiple data assimilation algorithm of Emmerick and Reynolds.  

To implement the cycle-based assimilation, users must add a cycle number to parameters (and template files) and observations (and instruction files) in the pest control file.  At runtime, PESTPP-DA does the incredibly painful process of forming a new "sub-problem" using these cycle numbers under the hood.  You are welcome!

But there is something more...if PESTPP-DA takes control of the time advancement process from the underlying simulation model, how do we form a coherent temporal evolution.  This is where the concept of "states" becomes critical.  A "state" is simply a simulated "state" of the system - in groundwater flow modeling, states are the simulated groundwater levels in each active model cell.  In a standard "batch" parameter estimation analysis (where we run the entire historic period at once and let MODFLOW "do its thing"), MODFLOW itself advances the states from stress period to stress period.  That is, the final simulated (hopefully converged) groundwater levels for each active cell at the end of the stress period 1 become the initial heads for stress period 2.  Ok, cool, so how do we make this work with PESTPP-DA where we have defined each stress period to be a cycle?  Well we have to now include the simulated water level in each active cell as an "observation" in the control file and we need to also add the initial groundwater level (i.e. the `strt` quantity in the MODFLOW world) for each active cell as a "parameter" in the control file.  And then we also need to tell PESTPP-DA how each of these state observations and state parameters map to each other - that is, how the observed state in model cell in layer 1/row 1/column 1 maps to the initial state parameter that is in layer 1/row 1/column 1.  Just some book keeping...right?  In this case, since we are using each stress period as an assimilation cycle, we have also changed the underlying model so that it is just a single stress-period...

So at this point, maybe you are thinking "WTF - this is insane.  Who would ever want to use sequential estimation".  Well, it turns out, if you are interested in making short-term, so-called "one-step-ahead" forecasts, sequential estimation is the optimal approach.  And that is because, just like we estimate parameters for things like HK and SS, we can also estimate the initial condition parameters!  WAT?!  That's right - we can estimate the initial groundwater levels in each active model cell for each cycle along with the static properties.  This results in the model being especially tuned at making a forecast related to the system behavior in the near term - these near-term forecasts depend as much on having the system state optimal as they do on having the properties optimal (maybe even more). So for short-term/near-term forecasts, if you have the groundwater levels nearly right, then you are probably going to do pretty well for forecasting something happening in the near future.  The combined form of estimation is referred to as "joint state-parameter estimation".  

In this example problem, we are estimating static properties during all cycles, as well as some cycle-specific forcing parameters like recharge and extraction rates, plus the initial groundwater level states.  With regard to the static properties, like HK and SS, the sequential estimation problem implies that the optimal static properties values may, and do, change for each cycle!  That's because what is optimal for cycle 1 in terms of HK and SS differs from what is optimal for cycle 2.  This may cause some of you to question the validity of sequential estimation.  But from a Bayesian perspective, its perfectly valid, and from the stand point of improved forecasting skill, its optimal.  



## The Current Tutorial

In the current notebook we are going to pick up after the horrific notebook that modifies the existing interface to one that is designed for sequential estimation - "freyberg_da_prep.ipynb". 

In this notebook, we will actually run PESTPP-DA for a sequential, joint state-parameter estimation problem where each monthly stress period in the original batch interface is now a discrete assimilation cycle.

### Admin

The next couple of cells load necessary dependencies and call a convenience function to prepare the PEST dataset folder for you. Simply press `shift+enter` to run the cells.


```python

import os
import shutil
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt;
import sys
sys.path.insert(0,os.path.join("..", "..", "dependencies"))
import pyemu
import flopy
assert "dependencies" in flopy.__file__
assert "dependencies" in pyemu.__file__
sys.path.insert(0,"..")
import herebedragons as hbd


```


```python
# specify the temporary working folder
t_d = os.path.join('freyberg6_da_template')

if not os.path.exists(t_d):
    raise Exception("you need to run the 'freyberg_da_prep.ipynb' notebook")


```


```python
pst = pyemu.Pst(os.path.join(t_d,"freyberg_mf6.pst"))
```

For this simple fast running model, there is considerable overhead time related to the file and model interface operations that PESTPP-DA compared to the model runtime.  So, to make the notebook experience more enjoyable, let's limit the number of realizations, the number of iterations, and the lambdas we want to test (with less non-zero weighted observations per cycle, we should be ok to use less realizations related to spurious correlation?)


```python
pst.pestpp_options['ies_parameter_ensemble'] = 'prior_pe.jcb'
pst.pestpp_options["ies_num_reals"] = 20
pst.pestpp_options["ies_lambda_mults"] = [0.1,1.0]
pst.pestpp_options["lambda_scale_fac"] = 1.0
num_workers = 10
pst.control_data.noptmax = 2
pst.write(os.path.join(t_d, 'freyberg_mf6.pst'))
m_d = os.path.join('master_da')
```


```python
pyemu.os_utils.start_workers(t_d, # the folder which contains the "template" PEST dataset
                            'pestpp-da', #the PEST software version we want to run
                            'freyberg_mf6.pst', # the control file to use with PEST
                            num_workers=num_workers, #how many agents to deploy
                            worker_root='.', #where to deploy the agent directories; relative to where python is running
                            master_dir=m_d, #the manager directory
                            port=4269)
```

## Inspecting PESTPP-DA
Like you probably have realized, all things get more complicated with sequential estimation, this includes post processing as well...

Let's load the prior observation ensemble files for each cycle.  The naming strategy is `<case>.<cycle>.<iteration>.obs.csv` for cycle-specific observation ensembles and `<case>.<cycle>.<iteration>.par.csv` for parameter ensembles


```python
files = [f for f in os.listdir(m_d) if ".0.obs.csv" in f]
pr_oes = {int(f.split(".")[1]):pd.read_csv(os.path.join(m_d,f),index_col=0) for f in files}
print(files)
```


```python
files = [f for f in os.listdir(m_d) if ".{0}.obs.csv".format(pst.control_data.noptmax) in f]
pt_oes = {int(f.split(".")[1]):pd.read_csv(os.path.join(m_d,f),index_col=0) for f in files}
```

Now load the obs and weight cycle tables so we can get the obsvals and weights for each cycle (since these change across the cycles)


```python
otab = pd.read_csv(os.path.join(m_d,"obs_cycle_table.csv"),index_col=0)
wtab = pd.read_csv(os.path.join(m_d,"weight_cycle_table.csv"),index_col=0)
```


```python
obs = pst.observation_data
obs = obs.loc[pst.nnz_obs_names,:]
obs
```

This file was made during the PESTPP-DA prep process - it contains all of the observation values.  Its just to help with plotting here...


```python
ad_df = pd.read_csv(os.path.join(t_d,"alldata.csv"),index_col=0)
ad_df
```


```python
for o in pst.nnz_obs_names:
    fig,axes = plt.subplots(2,1,figsize=(10,8))
    
    for kper,oe in pr_oes.items():
        axes[0].scatter([kper]*oe.shape[0],oe.loc[:,o].values,marker=".",c="0.5",alpha=0.5)
        axes[1].scatter([kper]*oe.shape[0],oe.loc[:,o].values,marker=".",c="0.5",alpha=0.5)
    for kper,oe in pt_oes.items():
        axes[1].scatter([kper]*oe.shape[0],oe.loc[:,o].values,marker=".",c="b",alpha=0.5)
    axes[1].scatter([kper]*oe.shape[0],oe.loc[:,o].values,marker=".",c="b",alpha=0.5,label="posterior")
    axes[1].scatter([kper]*pr_oes[kper].shape[0],pr_oes[kper].loc[:,o].values,marker=".",c="0.5",alpha=0.5,label="prior")
    
    ovals = otab.loc[o,:].values
    wvals = wtab.loc[o,:].values
    ylim = axes[1].get_ylim()
    xlim = axes[1].get_xlim()
    ovals[wvals==0] = np.nan
    axes[0].scatter(otab.columns.values,ovals,marker='^',c='r',s=60,label="observed")
    axes[1].scatter(otab.columns.values,ovals,marker='^',c='r',s=60,label="observed")
    axes[0].set_ylim(ylim)
    axes[0].set_xlim(xlim)
    axes[0].set_title("A) prior only: "+o,loc="left")
    axes[0].set_xlabel("kper")
    axes[1].set_ylim(ylim)
    axes[1].set_xlim(xlim)
    axes[1].set_title("B) pior and post: "+o,loc="left")
    axes[1].set_xlabel("kper")
    
    avals = ad_df.loc[:,o]
    axes[0].scatter(ad_df.index.values,avals,marker='.',facecolor='none',edgecolor="r",s=200,label="unseen truth")
    axes[1].scatter(ad_df.index.values,avals,marker='.',facecolor='none',edgecolor="r",s=200,label="unseen truth")
    
    axes[1].legend(loc="upper right")
    plt.tight_layout()
    
```

These plots look very different don't they... What we are showing in each pair is the prior simulated results on top and then the prior and posterior simulated results on the bottom.   Red circles are truth values not used for conditioning (we usually don't have these...), the red triangles are observations that were assimilated in a given cycle.  The reason we shows vertically stacked points instead of connected lines is because in the sequential estimation framework, the parameter and observation ensembles pertain only to the current cycle. Remembering that each "prior" simulated output ensemble is the forecast from the previous cycle to the current cycle without having "seen" any observations for the current cycle.  So we can see that after the first cycle with observations (cycle = 1), the model starts "tracking" the dynamics and it is pretty good a predicting the next cycles value.


```python
obs = pst.observation_data
forecasts = obs.loc[obs.obsnme.apply(lambda x: "headwater" in x or "tailwater" in x),"obsnme"]
forecasts
for o in forecasts:
    fig,axes = plt.subplots(2,1,figsize=(10,8))
    
    for kper,oe in pr_oes.items():
        axes[0].scatter([kper]*oe.shape[0],oe.loc[:,o].values,marker=".",c="0.5",alpha=0.5)
        axes[1].scatter([kper]*oe.shape[0],oe.loc[:,o].values,marker=".",c="0.5",alpha=0.5)
    for kper,oe in pt_oes.items():
        axes[1].scatter([kper]*oe.shape[0],oe.loc[:,o].values,marker=".",c="b",alpha=0.5)
    
    axes[1].scatter([kper]*oe.shape[0],oe.loc[:,o].values,marker=".",c="b",alpha=0.5,label="posterior")
    axes[1].scatter([kper]*pr_oes[kper].shape[0],pr_oes[kper].loc[:,o].values,marker=".",c="0.5",alpha=0.5,label="prior")
    
    
    axes[0].set_title("A) prior only: "+o,loc="left")
    axes[0].set_xlabel("kper")
    axes[1].set_title("B) pior and post: "+o,loc="left")
    axes[1].set_xlabel("kper")
    
    avals = ad_df.loc[:,o]
    axes[0].scatter(ad_df.index.values,avals,marker='.',facecolor='none',edgecolor="r",s=200,label="unseen truth")
    axes[1].scatter(ad_df.index.values,avals,marker='.',facecolor='none',edgecolor="r",s=200,label="unseen truth")
    
    axes[1].legend(loc="upper right")
    
    plt.tight_layout()
```

So that's pretty impressive right?  We are bracketing the sw/gw flux behavior for each cycle in the "one-step-ahead" sense (i.e. the prior plots).  And, for this single truth we are using, we also do pretty well through the 12-month/12-cycle forecast period (the last 12 cycles/months).  

Is PESTPP-DA worth the cost (in terms of cognitive load and increased computational burden)?  As always, "it depends"!
