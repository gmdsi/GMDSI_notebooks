---
layout: default
title: Multiple Observation Types
parent: Introduction to Theory, Concepts and PEST Mechanic
nav_order: 5
math: mathjax3
---

# History match the Freyberg model using a two parameters ``K`` and ``R`` using head and flux observations

#### Where are we on the Goldilocks complexity curve? 

<img src="freyberg_k_r_fluxobs_files\Hunt1998_sweetspot.png" style="float: center">

The runs so far were intended to be greatly oversimplified so as to be a starting point for adding complexity. However, when we added just __*one more parameter*__ for a total of 2 parameters uncertainty for some forecasts got appreciably __worse__.  And these parameters cover the entire model domain, which is unrealistic for the natural world!  Are we past the "sweetspot" and should avoid any additional complexity even if our model looks nothing like reality?  

Adding parameters in and of itself is not the real problem.  Rather, it is adding parameters that influence forecasts but which are unconstrained by observations so that they are free to wiggle and ripple uncertainty to our forecasts.  If observations are added that help constrain the parameters, the forecast observation will be more certain. That is, the natural flip side of adding parameters is constraining them, with data (first line of defense) or soft-knowledge and problem dimension reduciton (SVD).  

Anderson et al. (2015) suggest that at a minimum groundwater models be history matched to heads and fluxes.  There is a flux observation in our PEST control file, but it was given zero weight.  Let's see what happens if we move our model to the minimum calibration criteria of Anderson et al.

#### Objectives for this notebook are to:
1) Add a flux observation to the measurement objective function of our Freyberg model
2) Explore the effect of adding the observation to history matching, parameter uncertainty, and forecast uncertainty

### Admin
We have provided some pre-cooked PEST dataset files, wrapped around the modified Freyberg model. This is the same dataset introduced in the "freyberg_pest_setup" and "freyberg_k" notebooks. 

The functions in the next cell import required dependencies and prepare a folder for you. This folder contains the model files and a preliminary PEST setup. Run the cells, then inspect the new folder named "freyberg_k" which has been created in your tutorial directory. (Just press `shift+enter` to run the cells). 


```python
import sys
import os
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt;

import shutil

sys.path.insert(0,os.path.join("..", "..", "dependencies"))
import pyemu
import flopy
assert "dependencies" in flopy.__file__
assert "dependencies" in pyemu.__file__
sys.path.insert(0,"..")
import herebedragons as hbd

plt.rcParams['font.size'] = 10
pyemu.plot_utils.font =10
```


```python
# folder containing original model files
org_d = os.path.join('..', '..', 'models', 'monthly_model_files_1lyr_newstress')
# a dir to hold a copy of the org model files
tmp_d = os.path.join('freyberg_mf6')
if os.path.exists(tmp_d):
    shutil.rmtree(tmp_d)
shutil.copytree(org_d,tmp_d)
# get executables
hbd.prep_bins(tmp_d)
# get dependency folders
hbd.prep_deps(tmp_d)
# run our convenience functions to prepare the PEST and model folder
hbd.prep_pest(tmp_d)
```

Load the PEST control file:


```python
pst = pyemu.Pst(os.path.join(tmp_d,'freyberg.pst'))
```

Before we get started, just run PEST++ to repeat the last tutorial. We do this to have access to files for comparison.

As we did in the last tutorial, set `rch0` parameter transform to `log` and update NOPTMAX to 20:


```python
#update parameter transform
par = pst.parameter_data
par.loc['rch0', 'partrans'] = 'log'
# update noptmax
pst.control_data.noptmax = 20
# write
pst.write(os.path.join(tmp_d, 'freyberg_k_r.pst'))
# run pestpp
pyemu.os_utils.run("pestpp-glm freyberg_k_r.pst", cwd=tmp_d)
```

### Let's look at all observations in the PEST run


```python
# echo the observation data
pst.observation_data.head()
```

Wow!  that's a lot of observations.  Why so many?  Answer:  we are "carrying" lots of model outputs that may be of interest to us later __(not just places and times where we have actual measurements)__.  These outputs include forecasts as well as *"potential" observation* locations we will use in dataworth analysis (more on that later)

But, the calibration only uses observations where you assign weights.  Let's get a listing of just those.


```python
# filter the output based on non-zero weights
pst.observation_data.loc[pst.nnz_obs_names,:].head()
```

#### Here we have only head calibration targets (calhead).  But it is recommended that we calibrate to heads and fluxes.  

Let's give the observation ``gage-1`` a non-zero weight.  You can do this in a text editor but we'll do it in the next block and see the report out for convenience. We chose a new weight of 0.05 for this problem, but we'll spend more time on the concepts involved with observation weighting in a later notebook.


```python
obs = pst.observation_data
obs.loc[(obs.obgnme=="gage-1") & (obs['gage-1'].astype(float)<=3804.5), "weight"] = 0.05 #super subjective
obs.loc[obs.obgnme=="gage-1"].head()
```

Re-write and run PEST++:


```python
pst.write(os.path.join(tmp_d, 'freyberg_k_r_flxo.pst'))
```

Watch the terminal window where you launched this notebook to see the progress of PEST++.  Advance through the code blocks when you see a 0 returned.


```python
pyemu.os_utils.run("pestpp-glm freyberg_k_r_flxo.pst", cwd=tmp_d)
```

Let's explore the results, how did we do with fit (lowering PHI)?


```python
df_obj = pd.read_csv(os.path.join(tmp_d,"freyberg_k_r_flxo.iobj"),index_col=0)
df_obj
```

Egads!  Our Phi is a bit larger!  Are we moving backwards? Oh wait, we added a new weighted observation, so we can't compare it directly to what we had with only head observations.

#### Okay, what did it do to our parameter uncertainty?

As a reminder, let's load in the parameter uncertainty from the previous calibration (in which we only used head observations).


```python
# make a dataframe from the old run that had K and R but head-only calibration
df_paru_base = pd.read_csv(os.path.join(tmp_d, "freyberg_k_r.par.usum.csv"),index_col=0)
# echo out the dataframe
df_paru_base
```

OK, and now the new parameter uncertainty from the new calibration run (with added flux observations):


```python
df_paru = pd.read_csv(os.path.join(tmp_d,"freyberg_k_r_flxo.par.usum.csv"),index_col=0)
df_paru
```

Let's plot these up like before and compare them side by side. Here are the prior (dahsed grey lines) and posterior standard deviations (blue is with flux observations; green is with only head observations):


```python
figs,axes = pyemu.plot_utils.plot_summary_distributions(df_paru,subplots=True)
for pname,ax in zip(pst.adj_par_names,axes):
    pyemu.plot_utils.plot_summary_distributions(df_paru_base.loc[[pname],:],ax=ax,pt_color="g")
figs[0].tight_layout()
```

So, the blue shaded areas are taller and thinner than the green. This implies that, from an uncertainty standpoint, including the flux observations has helped us learn a lot about the recharge parameter. As recharge and `hk1` are correlated, this has in turn reduced uncertainty in `hk1`. #dividends

But, as usual, what about the forecast uncertainty?

### Forecasts

Let's look at our forecast uncertainty for both calibration runs. Load the original forecast uncertainties (head observations only):


```python
df_foreu_base = pd.read_csv(os.path.join(tmp_d,"freyberg_k_r.pred.usum.csv"),index_col=0)
df_foreu_base.loc[:,"reduction"] = 100.0 *  (1.0 - (df_foreu_base.post_stdev / df_foreu_base.prior_stdev))
df_foreu_base
```

And for the version with the flux observations:


```python
df_foreu = pd.read_csv(os.path.join(tmp_d,"freyberg_k_r_flxo.pred.usum.csv"),index_col=0)
df_foreu.loc[:,"reduction"] = 100.0 *  (1.0 - (df_foreu.post_stdev / df_foreu.prior_stdev))
df_foreu
```

Right then, let's plot these up together. Again, the original calibration forecast uncertainty are shaded in green. Forecast uncertaines from the version with head + flux observations are shaded in blue.


```python
for forecast in pst.forecast_names:
    ax1 = plt.subplot(111)
    pyemu.plot_utils.plot_summary_distributions(df_foreu.loc[[forecast],:],ax=ax1)
    pyemu.plot_utils.plot_summary_distributions(df_foreu_base.loc[[forecast],:],
                                             ax=ax1,pt_color='g')       
    ax1.set_title(forecast)
    plt.show()
figs[0].tight_layout()
```

As you can see, the information in the flux observations has reduced forecast uncertainty significantly for the `headwater`, but not so much for the `tailwater` forecast. So, at first glance we that the same model/observtion data set can make some forecasts better....but not others! i.e. calibration is sometimes worth it, sometimes it isn't.

But have we succeeded? Let's plot with the true forecast values:


```python
figs, axes = pyemu.plot_utils.plot_summary_distributions(df_foreu,subplots=True)
for ax in axes:
    fname = ax.get_title().lower()
    ylim = ax.get_ylim()
    v = pst.observation_data.loc[fname,"obsval"]
    ax.plot([v,v],ylim,"k--")
    ax.set_ylim(0,ylim[-1])
figs[0].tight_layout()
```

...._sigh_...still not winning.

### Hold up!

Isn't there a major flaw in our approach to uncertainty here? We freed `rch0` (which affects recharge in the calibration period). But we left the recharge in the forecast period (`rch1`) fixed - which is saying we know it perfectly. 

Damn...Ok...let's repeat all of the above but with `rch1` freed.


```python
par.loc['rch1', 'partrans'] = 'log'
# write
pst.write(os.path.join(tmp_d, 'freyberg_k_r_flxo.pst'))
# run pestpp
pyemu.os_utils.run("pestpp-glm freyberg_k_r_flxo.pst", cwd=tmp_d)
```


```python
# get parameter uncertainty
df_paru_new = pd.read_csv(os.path.join(tmp_d,"freyberg_k_r_flxo.par.usum.csv"),index_col=0)
# get forecast uncertainty
df_foreu_new = pd.read_csv(os.path.join(tmp_d,"freyberg_k_r_flxo.pred.usum.csv"),index_col=0)
df_foreu_new
```

Check out the parameter uncertainties. This time, pink shaded areas are the previous calibration run (with `rch1` fixed), blue is the new run (all parameters freed).

The `hk1` and `rch0` parameter uncertainties are the same as before. Note that the `rch1` prior and posterior uncertainty is the same. Which makes sense...the observation data which we have contains no information that informs what recharge will be in the future (well..it might...but let's not get side tracked).


```python
figs,axes = pyemu.plot_utils.plot_summary_distributions(df_paru_new,subplots=True)
for pname,ax in zip(pst.adj_par_names,axes):
    if pname == "rch1":
        continue
    pyemu.plot_utils.plot_summary_distributions(df_paru.loc[[pname],:],ax=ax,pt_color="fuchsia")
plt.ylim(0)
figs[0].tight_layout()
```

So how does this new source of uncertainty ripple out to our forecasts? Plot up the forecast uncertainties. 


```python
figs,axes = pyemu.plot_utils.plot_summary_distributions(df_foreu_new,subplots=True)
for forecast,ax in zip(sorted(pst.forecast_names),axes):
    pyemu.plot_utils.plot_summary_distributions(df_foreu.loc[[forecast],:],ax=ax,pt_color="fuchsia")
figs[0].tight_layout()
```

Oh hello! Forecast uncertainty for the `headwater` and `tailwater` forecasts have increased a lot (the `trgw` and `part_time` forecast did as well, but less noticeably). 

We see that the posterior for most forecasts is increased because of including future recharge uncertainty.  Intutitively, it makes sense because future recharge directly influences water levels and fluxes in the future.  And since calibration (history-matching) can't tell us anything about future recharge.  This means there is no data we can collect to reduce this source of uncertainty.

So..success? What do you think? Better...at least some truth values are bracketed by the predections but..._sigh_... still not.


```python
figs, axes = pyemu.plot_utils.plot_summary_distributions(df_foreu_new,subplots=True)
for ax in axes:
    fname = ax.get_title().lower()
    ylim = ax.get_ylim()
    v = pst.observation_data.loc[fname,"obsval"]
    ax.plot([v,v],ylim,"k--")
    ax.set_ylim(0,ylim[-1])
figs[0].tight_layout()
```
