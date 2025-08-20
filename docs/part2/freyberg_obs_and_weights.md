---
layout: default
title: Observation Values, Weights and Noise
parent: Decision Support Modelling with pyEMU and PEST++
nav_order: 2
math: mathjax3
---

# Formulating the Objective Function and The Dark Art of Weighting

The objective function expresses model-to-measurement misfit for use in the solution of an inverse problem through the *weighted squared difference between measured and simulated observations*. There is no formula for a universally "best approach" to formulate an objective function. However, the universal underlying principle is to ensure that as much information as possible is transferred to the parameters of a model, in as "safe" a way as possible (Doherty, 2015). 

**Observation Types**

In most history matching contexts a “multicomponent” objective function is recommended. Each component of this objective function is calculated on the basis of different groups of observations or of the same group of observations processed in different ways. In a nut-shell, this means as many (useful!) types of observation as are available should be included in the parameter-estimation process. This does not **just** mean different "measurement types"! It also means teasing out components *within* a given measurement type. These "secondary" components often contain information that is otherwise lost or overshadowed. 

**Observation Grouping** 

Using a multicomponent approach can extract as much information from an observation dataset as possible and transfer this information to estimated parameters. When constructing a PEST dataset, it is often useful (and convenient) to group observations by type. This makes it easier to customize objective function design and track the flow of information from data to parameters (and subsequently to predictions). Ideally, each observation grouping should illuminate and constrain parameter estimation related to a separate aspect of the system being modelled (Doherty and Hunt, 2010). For example, absolute values of heads may inform parameters that control horizontal flow patterns, whilst vertical differences between heads in different aquifers may inform parameters that control vertical flow patterns. 

**Observation Weighting**

A user must decide how to weight observations before estimating parameters with PEST(++). In some cases, it is prudent to strictly weight observations based on the inverse of the standard deviation of measurement noise. Observations with higher credibility should, without a doubt, be given more weight than those with lower credibility. However, in many history-matching contexts, model defects are just as important as noise in inducing model-to-measurement misfit as field measurements. Some types of model outputs are more affected by model imperfections than others. (Notably, the effects of imperfections on model output _differences_ are frequently less than their effects on raw model outputs.) In some cases, accommodating the "structural" nature of model-to-measurement misfit (i.e. the model is inherently better at fitting some measurements than others) is preferable rather than attempting to strictly respect the measurement credibility. 

The PEST Book (Doherty, 2015) and the USGS published report "A Guide to Using PEST for Groundwater-Model Calibration" (Doherty et al 2010) go into detail on formulating an objective function and discuss common issues with certain data-types. 

**References and Recommended Reading:**
>  - Doherty, J., (2015). Calibration and Uncertainty Analysis for Complex Environmental Models. Watermark Numerical Computing, Brisbane, Australia. ISBN: 978-0-9943786-0-6.
>  - <div class="csl-entry">White, J. T., Doherty, J. E., &#38; Hughes, J. D. (2014). Quantifying the predictive consequences of model error with linear subspace analysis. <i>Water Resources Research</i>, <i>50</i>(2), 1152–1173. https://doi.org/10.1002/2013WR014767</div>
>  - <div class="csl-entry">Doherty, J., &#38; Hunt, R. (2010). <i>Approaches to Highly Parameterized Inversion: A Guide to Using PEST for Groundwater-Model Calibration: U.S. Geological Survey Scientific Investigations Report 2010–5169</i>. https://doi.org/https://doi.org/10.3133/sir20105169</div>


### Recap: the modified-Freyberg PEST dataset

The modified Freyberg model is introduced in another tutorial notebook (see "intro to freyberg model"). The current notebook picks up following the "pstfrom pest setup" notebook, in which a high-dimensional PEST dataset was constructed using `pyemu.PstFrom`. You may also wish to go through the "intro to pyemu" notebook beforehand.

We will now address assigning observation target values from "measured" data and observation weighting prior to history matching. 

Recall from the "pstfrom pest setup" notebook that we included several observation types in the history matching dataset, namely:
 - head time-series
 - river gage time-series
 - temporal differences between both heads and gage time-series

We also included many observations of model outputs for which we do not have measured data. We kept these to make it easier to keep track of model outputs (this becomes a necessity when working with ensembles). We also included observations of "forecasts", i.e. model outputs of management interest.

The next couple of cells load necessary dependencies and call a convenience function to prepare the PEST dataset folder for you. This is the same dataset that was constructed during the "pstfrom pest setup" tutorial. 

Simply press `shift+enter` to run the cells.


```python
import os
import shutil
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt;
import matplotlib

import sys
sys.path.insert(0,os.path.join("..", "..", "dependencies"))
import pyemu
import flopy
assert "dependencies" in flopy.__file__
assert "dependencies" in pyemu.__file__
sys.path.insert(0,"..")
import herebedragons as hbd



plt.rcParams.update({'font.size': 10})

```


```python
# specify the temporary working folder
t_d = os.path.join('freyberg6_template')
```


```python
# use the conveninece function to get the pre-preprepared PEST dataset;
# this is the same dataset consutrcted in the "pstfrom" tutorial
if os.path.exists(t_d):
        shutil.rmtree(t_d)
org_t_d = os.path.join("..","part2_1_pstfrom_pest_setup",t_d)
assert os.path.exists(org_t_d),"you need to run the '/part2_1_pstfrom_pest_setup/freyberg_pstfrom_pest_setup.ipynb' notebook"
print("using files at ",org_t_d)
shutil.copytree(org_t_d,t_d)
```

Let's load in the `Pst` control file we constructed during the "pstfrom" tutorial:


```python
pst_file = "freyberg_mf6.pst"
pst = pyemu.Pst(os.path.join(t_d, pst_file))
```

When we constructed the PEST dataset (in the "pstfrom" tutorial) we simply identified what model outputs we wanted PEST to "observe". In doing so, `pyemu.PstFrom` assigned observation target values that it found in the existing model output files. (Which conveniently allowed us to test whether out PEST setup was working correctly). All observation weights were assigned a default value of 1.0. 

As a reminder:


```python
obs = pst.observation_data
obs.head()
```

As mentioned above, we need to do several things:
 - replace observation target values (the `obsval` column) with corresponding values from "measured data";
 - assign meaningful weights to history matching target observations (the `weight` column);
 - assign zero weight to observations that should not affect history matching.

Let's start off with the basics. First set all weights to zero. We will then go through and assign meaningful weights only to relevant target observations. 


```python
#check for nonzero weights
obs.weight.value_counts()
```


```python
# assign all weight zero
obs.loc[:, 'weight'] = 0

# check for non zero weights
obs.weight.unique()
```

### Measured Data

In most data assimilation contexts you will have some relevant measured data (e.g. water levels, river flow rates, etc.) which correspond to simulated model outputs. These will probably not coincide exactly with your model outputs. Are the wells at the same coordinate as the center of the model cell? Do measurement times line up nicely with model output times? Doubt it. And if they do, are single measurements that match model output times biased? And so on... 

A modeller needs to ensure that the observation values assigned in the PEST control file are aligned with simulated model outputs. This will usually require some case-specific pre-processing. Here we are going to demonstrate __an example__ - but remember, every case is different!

First, let's access our dataset of "measured" observations.


```python
obs_csv = os.path.join('..', '..', 'models', 'daily_freyberg_mf6_truth',"obs_data.csv")
assert os.path.exists(obs_csv)
obs_data = pd.read_csv(obs_csv)
obs_data.set_index('site', inplace=True)
obs_data.iloc[:5]
```

As you can see, we have measured data at daily intervals. But our model simulates monthly stress periods. So what observation value do we use? 

One option is to simply sample measured values from the data closest to our simulated output. The next cell does this, with a few checks along the way:


```python
#just pick the nearest to the sp end
model_times = pst.observation_data.time.dropna().astype(float).unique()
# get the list of osb names for which we have data
obs_sites =  obs_data.index.unique().tolist()

# restructure the observation data 
es_obs_data = []
for site in obs_sites:
    site_obs_data = obs_data.loc[site,:].copy()
    if isinstance(site_obs_data, pd.Series):
        site_obs_data.loc["site"] = site_obs_data.index.values
    elif isinstance(site_obs_data, pd.DataFrame):
        site_obs_data.loc[:,"site"] = site_obs_data.index.values
        site_obs_data.index = site_obs_data.time
        site_obs_data = site_obs_data.reindex(model_times,method="nearest")

    if site_obs_data.shape != site_obs_data.dropna().shape:
        print("broke",site)
    es_obs_data.append(site_obs_data)
es_obs_data = pd.concat(es_obs_data,axis=0,ignore_index=True)
es_obs_data.shape
```

Right then...let's plot our down-sampled measurement data and compare it to the original high-frequency time series.

The next cell generates plots for each time series of measured data. Blue lines are the original high-frequency data. The marked red line is the down-sampled data. What do you think? Does sampling to the "closest date" capture the behaviour of the time series? Doesn't look too good...It does not seem to capture the general trend very well.

Let's try something else instead.


```python
for site in obs_sites:
    #print(site)
    site_obs_data = obs_data.loc[site,:]
    es_site_obs_data = es_obs_data.loc[es_obs_data.site==site,:].copy()
    es_site_obs_data.sort_values(by="time",inplace=True)
    #print(site,site_obs_data.shape)
    fig,ax = plt.subplots(1,1,figsize=(10,2))
    ax.plot(site_obs_data.time,site_obs_data.value,"b-",lw=0.5)
    #ax.plot(es_site_obs_data.datetime,es_site_obs_data.value,'r-',lw=2)
    ax.plot(es_site_obs_data.time,es_site_obs_data.value,'r-',lw=1,marker='.',ms=10)
    ax.set_title(site)
plt.show()
```

This time, let's try using a moving-average instead. Effectively this is applying a low-pass filter to the time-series, smoothing out some of the spiky noise. 

The next cell re-samples the data and then plots it. Measured data sampled using a low-pass filter is shown by the marked green line. What do you think? Better? It certainly does a better job at capturing the trends in the original data! Let's go with that.


```python
ess_obs_data = {}
for site in obs_sites:
    #print(site)
    site_obs_data = obs_data.loc[site,:].copy()
    if isinstance(site_obs_data, pd.Series):
        site_obs_data.loc["site"] = site_obs_data.index.values
    if isinstance(site_obs_data, pd.DataFrame):
        site_obs_data.loc[:,"site"] = site_obs_data.index.values
        site_obs_data.index = site_obs_data.time
        sm = site_obs_data.value.rolling(window=20,center=True,min_periods=1).mean()
        sm_site_obs_data = sm.reindex(model_times,method="nearest")
    #ess_obs_data.append(pd.DataFrame9sm_site_obs_data)
    ess_obs_data[site] = sm_site_obs_data
    
    es_site_obs_data = es_obs_data.loc[es_obs_data.site==site,:].copy()
    es_site_obs_data.sort_values(by="time",inplace=True)
    fig,ax = plt.subplots(1,1,figsize=(10,2))
    ax.plot(site_obs_data.time,site_obs_data.value,"b-",lw=0.25)
    ax.plot(es_site_obs_data.time,es_site_obs_data.value,'r-',lw=1,marker='.',ms=10)
    ax.plot(sm_site_obs_data.index,sm_site_obs_data.values,'g-',lw=0.5,marker='.',ms=10)
    ax.set_title(site)
plt.show()
ess_obs_data = pd.DataFrame(ess_obs_data)
ess_obs_data.shape
```

### Update Target Observation Values in the Control File

Right then - so, these are our smoothed-sampled observation values:


```python
ess_obs_data.head()
```

Now we are confronted with the task of getting these _processed_ measured observation values into the `Pst` control file. Once again, how you do this will end up being somewhat case-specific and will depend on how your observation names were constructed. For example, in our case we can use the following function (we made it a function because we are going to repeat it a few times):


```python
def update_pst_obsvals(obs_names, obs_data):
    """obs_names: list of selected obs names
       obs_data: dataframe with obs values to use in pst"""
    # for checking
    org_nnzobs = pst.nnz_obs
    # get list of times for obs name suffixes
    time_str = obs_data.index.map(lambda x: f"time:{x}").values
    # empyt list to keep track of missing observation names
    missing=[]
    for col in obs_data.columns:
        # get obs list suffix for each column of data
        obs_sufix = col.lower()+"_"+time_str
        for string, oval, time in zip(obs_sufix,obs_data.loc[:,col].values, obs_data.index.values):
                
                if not any(string in obsnme for obsnme in obs_names):
                    if string.startswith("trgw-2"):
                        pass
                    else:
                        missing.append(string)
                # if not, then update the pst.observation_data
                else:
                    # get a list of obsnames
                    obsnme = [ks for ks in obs_names if string in ks] 
                    assert len(obsnme) == 1,string
                    obsnme = obsnme[0]
                    # assign the obsvals
                    obs.loc[obsnme,"obsval"] = oval
                    # assign a generic weight
                    if time > 3652.5 and time <=4018.5:
                        obs.loc[obsnme,"weight"] = 1.0
    # checks
    #if (pst.nnz_obs-org_nnzobs)!=0:
    #    assert (pst.nnz_obs-org_nnzobs)==obs_data.count().sum()
    if len(missing)==0:
        print('All good.')
        print('Number of new nonzero obs:' ,pst.nnz_obs-org_nnzobs) 
        print('Number of nonzero obs:' ,pst.nnz_obs)  
    else:
        raise ValueError('The following obs are missing:\n',missing)
    return
```


```python
pst.nnz_obs_groups
```


```python
# subselection of observation names; this is because several groups share the same obs name suffix
obs_names = obs.loc[obs.oname.isin(['hds', 'sfr']), 'obsnme']

# run the function
update_pst_obsvals(obs_names, ess_obs_data)
```


```python
pst.nnz_obs_groups
```


```python
pst.observation_data.oname.unique()
```

So that has sorted out the absolute observation groups. But remember the 'sfrtd' and 'hdstd' observation groups? Yeah that's right, we also added in a bunch of other "secondary observations" (the time difference between observations) as well as postprocessing functions to get them from model outputs. We need to get target values for these observations into our control file as well!

Let's start by calculating the secondary values from the absolute measured values. In our case, the easiest is to populate the model output files with measured values and then call our postprocessing function.

Let's first read in the SFR model output file, just so we can see what is happening:


```python
obs_sfr = pd.read_csv(os.path.join(t_d,"sfr.csv"),
                    index_col=0)

obs_sfr.head()
```

Now update the model output csv files with the smooth-sampled measured values:


```python
def update_obs_csv(obs_csv):
    obsdf = pd.read_csv(obs_csv, index_col=0)
    check = obsdf.copy()
    # update values in reelvant cols
    for col in ess_obs_data.columns:
        if col in obsdf.columns:
            obsdf.loc[:,col] = ess_obs_data.loc[:,col]
    # keep only measured data columns; helps for vdiff and tdiff obs later on
    #obsdf = obsdf.loc[:,[col for col in ess_obs_data.columns if col in obsdf.columns]]
    # rewrite the model output file
    obsdf.to_csv(obs_csv)
    # check 
    obsdf = pd.read_csv(obs_csv, index_col=0)
    assert (obsdf.index==check.index).all()
    return obsdf

# update the SFR obs csv
obs_srf = update_obs_csv(os.path.join(t_d,"sfr.csv"))
# update the heads obs csv
obs_hds = update_obs_csv(os.path.join(t_d,"heads.csv"))
```

OK...now we can run the postprocessing function to update the "tdiff" model output csv's. Copy across the `helpers.py` we used during the `PstFrom` tutorial. Then import it and run the `process_secondary_obs()` function.


```python
shutil.copy2(os.path.join('..','part2_1_pstfrom_pest_setup', 'helpers.py'),
            os.path.join('helpers.py'))

import helpers
helpers.process_secondary_obs(ws=t_d)
```


```python
# the oname column in the pst.observation_data provides a useful way to select observations in this case
obs.oname.unique()
```


```python
org_nnzobs = pst.nnz_obs
    #if (pst.nnz_obs-org_nnzobs)!=0:
    #    assert (pst.nnz_obs-org_nnzobs)==obs_data.count().sum()
```


```python
print('Number of nonzero obs:', pst.nnz_obs)

diff_obsdict = {'sfrtd': "sfr.tdiff.csv", 
                'hdstd': "heads.tdiff.csv",
                }

for keys, value in diff_obsdict.items():
    print(keys)
    # get subselct of obs names
    obs_names = obs.loc[obs.oname.isin([keys]), 'obsnme']
    # get df
    obs_csv = pd.read_csv(os.path.join(t_d,value),index_col=0)
    # specify cols to use; make use of info recorded in pst.observation_data to only select cols with measured data
    usecols = list(set((map(str.upper, obs.loc[pst.nnz_obs_names,'usecol'].unique()))) & set(obs_csv.columns.tolist()))
    obs_csv = obs_csv.loc[:, usecols]
    # for checking
    org_nnz_obs_names = pst.nnz_obs_names
    # run the function
    update_pst_obsvals(obs_names,
                        obs_csv)
    # verify num of new nnz obs
    print(pst.nnz_obs)
    print(len(org_nnz_obs_names))
    print(len(usecols))
    assert (pst.nnz_obs-len(org_nnz_obs_names))==12*len(usecols), [i for i in pst.nnz_obs_names if i not in org_nnz_obs_names]
```


```python
pst.nnz_obs_groups
```


```python
pst.nnz_obs_names
```

The next cell does some sneaky things in the background to populate `obsvals` for forecast observations just so that we can keep track of the truth. In real-world applications you might assign values that reflect decision-criteria (such as limits at which "bad things" happen, for example) simply as a convenience. For the purposes of history matching, these values have no impact. They can play a role in specifying constraints when undertaking optimisation problems.  


```python
pst.observation_data.loc[pst.forecast_names]
```


```python
hbd.prep_forecasts(pst)
```


```python
pst.observation_data.loc[pst.forecast_names]
```

## Observation Weights

We are going to start off by taking a look at our current objective function value and the relative contributions from the various observation groups. Recall that this is the objective function value with **initial parameter values** and the default observations weights.

First off, we need to get PEST to run the model once so that the objective function can be calculated. Let's do that now. Start by reading the control file and checking that NOPTMAX is set to zero:




```python
# check noptmax
pst.control_data.noptmax
```

You got a zero? Alrighty then! Let's write the uprated control file and run PEST again and see what that has done to our Phi:


```python
pst.write(os.path.join(t_d,pst_file))
```


```python
pyemu.os_utils.run("pestpp-ies.exe {0}".format(pst_file),cwd=t_d)
```

Now we need to reload the `Pst` control file so that the residuals are updated:


```python
pst = pyemu.Pst(os.path.join(t_d, pst_file))
pst.phi
```

Jeepers - that's large! Before we race off and start running PEST to lower it we should compare simulated and measured values and take a look at the components of Phi. 

Let's start with taking a closer look. The `pst.phi_components` attribute returns a dictionary of the observation group names and their contribution to the overall value of Phi. 


```python
pst.phi_components
```


Unfortunately, in this case we have too many observation groups to easily display (we assigned each individual time series to its own observation group; this is a default setting in `pyemu.PstFrom`). 

So let's use `Pandas` to help us summarize this information (note: `pyemu.plot_utils.res_phi_pie()` does the same thing, but it looks a bit ugly because of the large number of observation groups). To make it easier, we are going to just look at the nonzero observation groups:


```python
nnz_phi_components = {k:pst.phi_components[k] for k in pst.nnz_obs_groups}
nnz_phi_components
```

And while we are at it, plot these in a pie chart. 

If you wish, try displaying this with `pyemu.plot_utils.res_phi_pie()` instead. Because of the large number of columns it's not going to be pretty, but it gets the job done.


```python
phicomp = pd.Series(nnz_phi_components)
plt.pie(phicomp, labels=phicomp.index.values);
#pyemu.plot_utils.res_phi_pie(pst,);
```

Well that is certainly not ideal - phi is dominated by the SFR observation groups. Why? Because the observation values of these are much larger than those of the other observation groups...and we assigned the same weight to all of them...

Let's try error-based weighting combined with common sense (e.g. subjectivity) instead. For each observation type we will assign a weight equal to the inverse of error. Conceptually, this error reflects our estimate of measurement noise. In practice, it reflects both measurement noise and model error.


```python
nz_obs = pst.observation_data.loc[pst.nnz_obs_names,:]
nz_obs.oname.unique()
```


```python
obs = pst.observation_data
obs.loc[nz_obs.loc[(nz_obs.oname=='hds'), 'obsnme'], 'weight'] = 1/0.1
obs.loc[nz_obs.loc[(nz_obs.oname=='hdstd'), 'obsnme'], 'weight'] = 1/0.05
obs.loc[nz_obs.loc[(nz_obs.oname=='hdsvd'), 'obsnme'], 'weight'] = 1/0.01
# fanciness alert: heteroskedasticity; error will be proportional to the observation value for SFR obs
obs.loc[nz_obs.loc[(nz_obs.oname=='sfr'), 'obsnme'], 'weight'] = 1/ abs(0.1 * nz_obs.loc[(nz_obs.oname=='sfr'), 'obsval'])
obs.loc[nz_obs.loc[(nz_obs.oname=='sfrtd'), 'obsnme'], 'weight'] = 1/100 #abs(0.5* nz_obs.loc[(nz_obs.oname=='sfrtd'), 'obsval'])
```

Let's take a look at how that has affected the contributions to Phi:


```python
plt.figure(figsize=(7,7))
phicomp = pd.Series({k:pst.phi_components[k] for k in pst.nnz_obs_groups})
plt.pie(phicomp, labels=phicomp.index.values);
plt.tight_layout()
```

Better! Now weights reflect the quality of the data. Another aspect to consider is if observation group contributions to phi are relatively balanced...do you think any of these observation groups will dominate phi (or be lost)? 

The next cell adds in a column to the `pst.observation_data` for checking purposes in subsequent tutorials. Pretend you didn't see it :)


```python
pst.observation_data.loc[pst.nnz_obs_names,'observed'] = 1
```

Don't forget to re-write the .pst file!



```python
pst.write(os.path.join(t_d,pst_file),version=2)
```

### Understanding Observation Weights and Measurement Noise

Let's visualize what these weights actually imply. 

We are going to generate an ensemble of observation values. These values will be sampled from a Gaussian distribution, in which the observation value in the `Pst` control file is the mean and the observation weights reflect the inverse of standard deviation. (See the "intro to pyemu" tutorial for an introduction to  `pyemu.ObservationEnsemble`)



```python
oe = pyemu.ObservationEnsemble.from_gaussian_draw(pst=pst, 
                                                num_reals=50) 
```

The function in the next cell does a bunch of funky things in order to plot observation time series (convenient that we stored all the necessary information in observation names, hey?). 



```python
def plot_obs_ts(pst, oe, obsgrps=pst.nnz_obs_groups):
    # get times and "measured values"
    nz_obs = pst.observation_data.loc[pst.nnz_obs_names,:].copy()
    nz_obs['time'] = nz_obs['time'].astype(float)
    nz_obs.sort_values(['obgnme','time'], inplace=True)
    
    # to plot crrent model outputs
    res = pst.res.copy()
    res['time'] = pst.observation_data['time'].astype(float)
    res.sort_values(['group','time'], inplace=True)

    for nz_group in obsgrps:
        nz_obs_group = nz_obs.loc[nz_obs.obgnme==nz_group,:]
        nz_obs_meas = res.loc[(res['group']==nz_group) & res['weight']!=0]

        fig,ax = plt.subplots(1,1,figsize=(10,2))
        # plot ensemble of time series of measured + noise
        [ax.plot(nz_obs_group.time, oe.loc[r, nz_obs_group.obsnme] ,color="r",lw=0.3) for r in oe.index]
        # plot simulated time series
        ax.plot(nz_obs_group.time,nz_obs_group.obsval,"b")
        # plot measured time series
        ax.plot(nz_obs_meas.time, nz_obs_meas.modelled, 'k', linestyle='--')
        ax.set_title(nz_group)
    plt.show()
    return
```

Let's visualize the time series of observations for the SFR measurement types. Run the next cell.

The blue line is the time series of "field measured" values (in the control file). The red lines are time series from the observation ensemble.The dashed black line is the model output with initial parameter values. What we are saying with these weights is that any of these red lines could also have been "measured". Due to the noise (e.g. uncertainty) in our measurements, the true value can fall any where in that range.


```python
ts_obs = [i for i in pst.nnz_obs_groups if 'sfr' in i]
plot_obs_ts(pst, oe, ts_obs)
```

### An Aside: A Convenience of Error-Based Weighting

A convenience of weighting with the inverse of the measurement uncertainty is that it is easy to know what the ideal Phi should be: it should be equal to the number of non-zero weighted observations. This of course assumes that all model-to-measurement misfit is due to *measurement* uncertainty. In practice, model error usually plays a larger role, as we will see in other tutorials. 

Just to demonstrate what we mean, let's quickly do some math. Imagine that we have:
 - 2 observations
 - both have measured values of 1
 - with measurement standard deviation (e.g. uncertainty) of 0.25
 - therefore, they are assigned a weight of 1/0.25 = 4

So, all that we know is that the true measured values are most likely to be somewhere between 0.75 and 1.25. Therefore, the best that a modelled value should be expected to achieve is 1 +/- 0.25. 

Let's say we obtain modelled values for each of observation: 
 - 1.25
 - 0.75

 Let's calculate Phi for such a case:


```python
weight = 1/0.25
# res = (weight * (meas - sim)) ^ 2
res1 = (weight * (1 - 1.25))**2
res2 = (weight * (1 - 0.75))**2
# sum the squared weighted residuals
phi = res1 + res2
phi
```

And there you have it, the value of Phi is equal to the number of observations. Getting a better fit than that means we are just "fitting noise".

### Back to Freyberg.

So, how many non-zero observations do we have in the dataset? Recall this number is recorded in the control file and easily accessible through the `Pst.nnz_obs` attribute. So our current Phi (see `pst.phi`) is  a bit higher than that. Hopefully history matching will help us bring it down.


```python
print('Target phi:',pst.nnz_obs)
print('Current phi:', pst.phi)
```

Right, so we have or observation weights sorted out. Let's just make a quick check of how model outputs and measured values compare by looking at a 1to1 plot for each observation group:


```python
figs = pst.plot(kind="1to1");
pst.res.loc[pst.nnz_obs_names,:]
plt.show()
```

### Autocorrelated Transient Noise

(Sounds like a pretty cool name for a band...) 

Lastly, not so common, let's also generate our observation covariance matrix. Often this step is omitted,  if the observation weights in the `* observation data` section reflect observation uncertainty. Here we will implement it expressly so as to demonstrate how to add autocorrelated transient noise! When is the last time you saw that in the wild?

First let us start by generating a covariance matrix, using the weights recorded in the `pst.observation_data` section. This is accomplished with `pyemu.Cov.from_observation_data()`. This will generate a covariance matrix for **all** observations in the control file, even the zero-weighted ones. 

However, we are only interested in keeping the non-zero weighted observations. We can use `pyemu` convenient Matrix manipulation functions to `.get()` the covariance matrix rows/columns that we are interested in.


```python
obs = pst.observation_data

# generate cov matrix for all obs using weights in the control file
obs_cov = pyemu.Cov.from_observation_data(pst, )

# reduce cov down to only include non-zero observations
obs_cov = obs_cov.get(row_names=pst.nnz_obs_names, col_names=pst.nnz_obs_names, )

# side note: 
# we are saving the diagonal (e.g no correlation) observation uncertainty cov matrix
# to an external file for use in a later tutorial
obs_cov.to_coo(os.path.join(t_d,"obs_cov_diag.jcb"))

# make it a dataframe to make life easier
df = obs_cov.to_dataframe()

df.head()
```

So this returned a diagonal covariance matrix (e.g. only values in the diagonal are non-zero). This implies that there is no covariance between observation noise. 

You can plot the matrix, but due to the scale it won't look particularly interesting:


```python
plt.imshow(df.values)
plt.colorbar()
```

Hmm...you can kinda see something there (that's the sfr flow obs...large variance). Let's instead just look at a small part, focusing on observations that make up a single time series. 

So as you can see below, the matrix diagonal reflects individual observation uncertainties, whilst all the off-diagonals are zero (e.g. no correlation).


```python
obs_select = obs.loc[(obs.obgnme=='oname:hds_otype:lst_usecol:trgw-0-26-6') & (obs.weight>0)]

plt.imshow(df.loc[obs_select.obsnme,obs_select.obsnme].values)
plt.colorbar()
```

But recall that most of the observations are time-series. Here we are saying "how wrong the measurement is today, has nothing to do with how wrong it was yesterday". Is that reasonable? For some noise (e.g. white noise), yes, surely. But perhaps not for all noise. So how can we express that "if my measurement was wrong yesterday, then it is *likely* to be wrong in the same way today"? Through covariance, that's how. We can use the same principles that we employed to specify parameter spatial/temporal covariance earlier on.

We will now demonstrate for the single time series we looked at above. 

First, construct a geostatistical structure and variogram. Here we are using a range of 90 days; for an exponential variogram `a` is range/3. From these, construct a generic covariance matrix for generic parameters with "coordinates" corresponding to the observation times. 


```python
v = pyemu.geostats.ExpVario(a=730,contribution=1.0)
x = obs_select.time.astype(float).values #np.arange(float(obs.time.values[:11].max()))
y = np.zeros_like(x)
names = ["obs_{0}".format(xx) for xx in x]
cov = v.covariance_matrix(x,y,names=names)
```

Then, scale the values in the generic covariance matrix to reflect the observation uncertainties and take a peek.

And there you have it, off-diagonal elements are no longer zero and show greater correlation between uncertainties of observation which happen closer together.


```python
x_group = cov.x.copy()
w = obs_select.weight.mean()
v = (1./w)**2
x_group *= v
df.loc[obs_select.obsnme,obs_select.obsnme] = x_group

plt.imshow(df.loc[obs_select.obsnme,obs_select.obsnme].values);
plt.colorbar();
```

Now implement that to loop over all the non-zero time series observation groups:


```python
for obgnme in pst.nnz_obs_groups:
    obs_select = obs.loc[(obs.obgnme==obgnme) & (obs.weight>0)]

    v = pyemu.geostats.ExpVario(a=30,contribution=1.0)
    x = obs_select.time.astype(float).values #np.arange(float(obs.time.values[:11].max()))
    y = np.zeros_like(x)
    names = ["obs_{0}".format(xx) for xx in x]
    cov = v.covariance_matrix(x,y,names=names)


    x_group = cov.x.copy()
    w = obs_select.weight.mean()
    v = (1./w)**2
    x_group *= v
    df.loc[obs_select.obsnme,obs_select.obsnme] = x_group

```

And then re-construct a `Cov` object and record it to an external file:


```python
obs_cov_tv = pyemu.Cov.from_dataframe(df)
obs_cov_tv.to_coo(os.path.join(t_d,"obs_cov.jcb"))
```

If you made it this far, good on you! Let's finish with some eye candy! 


```python
x = obs_cov_tv.to_pearson().x
x[np.abs(x)<0.00001]= np.nan
plt.imshow(x);
```

beautiful!

Last thing here, let's just see what this actually implies for our observation ensembles. As before, let's generate an ensemble of observations, but this time we pass the `obs_cov_tv` covariance matrix:


```python
oe_tv = pyemu.ObservationEnsemble.from_gaussian_draw(pst=pst, 
                                                num_reals=50,
                                                cov=obs_cov_tv) 
```

Let's plot the same ensemble of obs timeseries with non-correlated noise we saw earlier, just to make it easy to compare.


```python
print('Non correlated observation noise:')
plot_obs_ts(pst, oe, ts_obs)
```

And now the ensemble of obs timeseries with autocorrelated noise. Note how these are smoother and less of a jumble of spaghetti.


```python
print('Autocorrelated observation noise:')
ts_obs = [i for i in pst.nnz_obs_groups if 'sfr' in i]
plot_obs_ts(pst, oe_tv, ts_obs)
```

### A Final Aside: Weighting for Visibility

(The following is an aside and does not apply for subsequent tutorials.)

As discussed above, a practical means of accommodating this situation is to weight all observation groups
such that they contribute an equal amount to the starting measurement objective function. In this
manner, no single group dominates the objective function, or is dominated by others; the information
contained in each group is therefore equally “visible” to PEST.

The `Pst.adjust_weights()` method provides a mechanism to fine tune observation weights according to their contribution to the objective function. (*Side note: the PWTADJ1 utility from the PEST-suite automates this same process of "weighting for visibility".*) 

We start by creating a dictionary of non-zero weighted observation group names and their respective contributions to the objective function. For example, we will specify that we want each group to contribute a value of 100 to the objective function. (Why 100? No particular reason. Could just as easily be 1000. Or 578. Doesn't really matter. 100 is a nice round number though.)


```python
# create a dictionary of group names and weights
balanced_groups = {grp:100 for grp in pst.nnz_obs_groups}
balanced_groups
```

We can now use the `pst.adjust_weights()` method to adjust the observation weights in the `pst` control file object. (*Remember! This all only happens in memory. It does not get written to the PEST control file yet!*)


```python
# make all non-zero weighted groups have a contribution of 100.0
pst.adjust_weights(obsgrp_dict=balanced_groups,)
```

If we now plot the phi components again, voila! We have a nice even distribution of phi from each observation group.


```python
phicomp = pd.Series(pst.phi_components)
plt.pie(phicomp, labels=phicomp.index.values);
```

Some **caution** is required here. Observation weights and how these pertain to history-matching *versus* how they pertain to generating an observation ensemble for use with `pestpp-ies` or FOSM is a frequent source of confusion.

 - when **history-matching**, observation weights listed in the control file determine their contribution to the objective function, and therefore to the parameter estimation process. Here, observation weights may be assigned to reflect observation uncertainty, the balance required for equal "visibility", or other modeller-defined (and perhaps subjective...) measures of observation worth.  
 - when undertaking **uncertainty analysis**, weights should reflect ideally the inverse of observation error (or the standard deviation of measurement noise). Keep this in mind when using `pestpp-glm` or `pestpp-ies`. If the observations in the control file do not have error-based weighting then (1) care is required if using `pestpp-glm` for FOSM and (2) make sure to provide an adequately prepared observation ensemble to `pestpp-ies`.



Made it this far? Congratulations! Have a cookie :)


```python

```
