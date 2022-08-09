---
layout: default
title: Monte Carlo
parent: Introduction to Theory, Concepts and PEST Mechanic
nav_order: 13
math: mathjax3
---

# NonLinear Uncertainty Analysis - Monte Carlo

As we've seen, First-Order-Second-Moment (FOSM) is quick and insightful.  But FOSM depends on an assumption that the relation between the model and the forecast uncertainty is linear.  But many times the world is nonlinear. Short cuts like FOSM need assumptions, but we can free ourselves by taking the brute force approach.  That is define the parameters that are important, provide the prior uncertainty, sample those parameters many times, run the model many times, and then summarize the results.  

On a more practical standpoint, the underlying theory for FOSM and why it results in shortcomings can be hard to explain to stakeholders.  Monte Carlo, however, is VERY straightforward, its computational brute force notwithstanding. 

Here's a flowchart from Anderson et al. (2015):

<img src="freyberg_monte_carlo_files/Fig10.14_MC_workflow.png" style="float: center">


## The Current Tutorial

In this notebook we will:
1. Run Monte Carlo on the Freyberg model
2. Look at parameter and forecast uncertainty 
3. Look at the effect of prior parameter uncertainty covariance
4. Start thinking of the advantages and disadvantages of linear and nonlinear uncertainty methods

This notebook is going to be computationaly tough and may tax your computer. So buckle up. 


### Admin

First the usual admin of preparing folders and constructing the model and PEST datasets.


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
working_dir = os.path.join('freyberg_mf6')
if os.path.exists(working_dir):
    shutil.rmtree(working_dir)
shutil.copytree(org_d,working_dir)
# get executables
hbd.prep_bins(working_dir)
# get dependency folders
hbd.prep_deps(working_dir)
# run our convenience functions to prepare the PEST and model folder
hbd.prep_pest(working_dir)
# convenience function that builds a new control file with pilot point parameters for hk
hbd.add_ppoints(working_dir)
```



Monte Carlo uses a lots and lots of forward runs so we don't want to make the mistake of burning the silicon for a PEST control file that is not right.  Here we make doubly sure that the control file has the recharge freed (not "fixed' in the PEST control file).  

### Load the `pst` control file

Let's double check what parameters we have in this version of the model using `pyemu` (you can just look in the PEST control file too.).

We have adjustable parameters that control SFR inflow rates, well pumping rates, hydraulic conductivity and recharge rates. Recall that by setting a parameter as "fixed" we are stating that we know it perfectly (should we though...?). Currently fixed parameters include porosity and future recharge.

For the sake of this tutorial, and as we did in the "sensitivity" tutorials, let's set all the parameters free:


```python
pst_name = "freyberg_pp.pst"
# load the pst
pst = pyemu.Pst(os.path.join(working_dir,pst_name))
#update parameter data
par = pst.parameter_data
#update paramter transform
par.loc[:, 'partrans'] = 'log'
```


```python
# check if the model runs
pst.control_data.noptmax=0
# rewrite the control file
pst.write(os.path.join(working_dir,pst_name))
# run the model once
pyemu.os_utils.run('pestpp-glm freyberg_pp.pst', cwd=working_dir)
```

Reload the control file:


```python
pst = pyemu.Pst(os.path.join(working_dir,'freyberg_pp.pst'))
pst.phi
```

# Monte Carlo

In it's simplest form, Monte Carlo boils down to: 1) "draw" many random samples of parameters from the prior probability distribution, 2) run the model, 3) look at the results.

So how do we "draw", or sample, parameters? (Think "draw" as in "drawing a card from a deck"). We need to randomly sample parameter values from a range. This range is defined by the _prior parameter probability distribution_. As we did for FOSM, let's assume that the bounds in the parameter data section define the range of a Gaussian (or normal) distribution, and that the intial values define the mean. 

### The Prior

We can use `pyemu` to sample parameter values from such a distribution. First, construct a covariance matrix from the parameter data in the `pst` control file:


```python
prior_cov = pyemu.Cov.from_parameter_data(pst)
```

What did we do there? We generated a covariance matrix from the parameter bound values. The next cell displays an image of the matrix. As you can see, all off-diagonal values are zero. Therefore no parameter correaltion is accounted for. 

This covariance matrix is the _prior_ (before history matching) parameter covariance. 


```python
x = prior_cov.as_2d.copy()
x[x==0] = np.nan
plt.imshow(x)
plt.colorbar();
```

OK, let's now sample 500 parameter sets from the probability distribution described by this covariance matrix and the mean values (e.g. the initial parameter values in the `pst` control file).


```python
parensemble = pyemu.ParameterEnsemble.from_gaussian_draw(pst=pst,cov=prior_cov,num_reals=500,)
# ensure that the samples respect parameter bounds in the pst control file
parensemble.enforce()
```

Here's an example of the first 5 parameter sets of our 500 created by our draw ("draw" here is like "drawing" a card


```python
parensemble.head()
```

What does this look like in terms of spatially varying parameters? Let's just plot the hydraulic conductivity from one of these samples. (Note the log-transformed values of K):


```python
par = pst.parameter_data
parnmes = par.loc[par.pargp=='hk1'].parnme.values
pe_k = parensemble.loc[:,parnmes].copy()

# use the hbd convenienc function to plot several realisations
hbd.plot_ensemble_arr(pe_k, working_dir, 10)
```

That does not look very realistic. Do these look "right" (from a geologic stand point)? Lots of "random" variation (pilot points spatially near each other can have very different values)...not much structure...why? Because we have not specified any parameter correlation. Each pilot point is statisticaly independent. 

How do we express that in the prior? We need to express parameter spatial covariance with a geostatistical structure. Much the same as we did for regularization. Let's build a covariance matrix for pilot point parameters.

You should be familiar with these by now:



```python
v = pyemu.geostats.ExpVario(contribution=1.0,a=2500,anisotropy=1.0,bearing=0.0)
gs = pyemu.utils.geostats.GeoStruct(variograms=[v])
pp_tpl = os.path.join(working_dir,"hkpp.dat.tpl")
cov = pyemu.helpers.geostatistical_prior_builder(pst=pst, struct_dict={gs:pp_tpl})
# display
plt.imshow(cov.to_pearson().x,interpolation="nearest")
plt.colorbar()
cov.to_dataframe().head()
```

Now re-sample using the geostatisticaly informed prior:


```python
parensemble = pyemu.ParameterEnsemble.from_gaussian_draw(pst=pst, cov=cov, num_reals=500,)
# ensure that the samples respect parameter bounds in the pst control file
parensemble.enforce()
```

Here's an example of the first 5 parameter sets of our 500 created by our draw ("draw" here is like "drawing" a card


```python
parensemble.head()
```

Now when we plot the spatialy distributed parameters (`hk1`) we can see some structure and points which are near to each other are morel ikely to be similar:


```python
pe_k = parensemble.loc[:,parnmes].copy()
# use the hbd convenienc function to plot several realisations
hbd.plot_ensemble_arr(pe_k, working_dir, 10)
```

Let's look at some of the distributions. Note that distributions are log-normal, because parameters in the `pst` are log-transformed:


```python
for pname in pst.par_names[:5]:
    ax = parensemble.loc[:,pname].hist(bins=20)
    print(parensemble.loc[:,pname].min(),parensemble.loc[:,pname].max(),parensemble.loc[:,pname].mean())
    ax.set_title(pname)
    plt.show()
```

Notice anything funny? Compare these distributions to the uper/lower bounds in the `pst.parameter_data`. There seem to be many parameters "bunched up" at the bounds. This is due to the gaussian distribution being truncated at the parameter bounds.


```python
pst.parameter_data.head()
```

### Run Monte Carlo

So we now have 500 different random samples of parameters. Let's run them! This is going to take some time, so let's do it in parallel using [PEST++SWP](https://github.com/usgs/pestpp/blob/master/documentation/pestpp_users_manual.md#10-pestpp-swp).

First write the ensemble to an external CSV file:


```python
parensemble.to_csv(os.path.join(working_dir,"sweep_in.csv"))
```

As usual, make sure to specify the number of agents to use. This value must be assigned according to the capacity of youmachine:


```python
num_workers = 10
```


```python
# the master directory
m_d='master_mc'
```

Run the next cell to call PEST++SWP. It'll take a while.


```python
pyemu.os_utils.start_workers(working_dir, # the folder which contains the "template" PEST dataset
                            'pestpp-swp', #the PEST software version we want to run
                            pst_name, # the control file to use with PEST
                            num_workers=num_workers, #how many agents to deploy
                            worker_root='.', #where to deploy the agent directories; relative to where python is running
                            master_dir=m_d, #the manager directory
                            )
```

Alright - let's see some MC results.  For these runs, what was the Phi?


```python
df_out = pd.read_csv(os.path.join(m_d,"sweep_out.csv"),index_col=0)
df_out = df_out.loc[df_out.failed_flag==0,:] #drop any failed runs
df_out = df_out.loc[~df_out.le(-2.e27).any(axis=1)] #drop extreme values
df_out.columns = [c.lower() for c in df_out.columns]
df_out.head()
```

Wow, some are pretty large. What was Phi with the initial parameter values??



```python
pst.phi
```


Let's plot Phi for all 500 runs:


```python
df_out.phi.hist(bins=100);
```

Wow, some of those models are really REALLY bad fits to the observations.  So, when we only use our prior knowledge to sample the parameters we get a bunch of models with unacceptably Phi, and we can consider them not reasonable.  Therefore, we should NOT include them in our uncertainty analysis.


## Conditioning (a.k.a GLUE)

__IMPORTANT:__  this is super important - in this next block we are "conditioning" our Monte Carlo run by removing the bad runs based on a Phi cutoff. So here is where we choose which realizations we consider __good enough__ with respect to fitting the observation data. 

Those realizations with a phi bigger than our cutoff are out, no matter how close to the cutoff; those that are within the cutoff are all considered equally good regardless of how close to the cutoff they reside.

We started out with a Phi of:


```python
pst.phi
```

Let's say any Phi below 2x `pst.phi` is aceptable (which is pretty poor by the way...):


```python
acceptable_phi = pst.phi * 2
good_enough = df_out.loc[df_out.phi<acceptable_phi].index.values
print("number of good enough realisations:", good_enough.shape[0])
```

These are the run number of the 500 that met this cutoff.  Sheesh - that's very few!

Here is a __major problem with "rejection sampling" in high dimensions__: you have to run the model many many many many many times to find even a few realizations that fit the data acceptably well.  

With all these parameters, there are so many possible combinations, that very few realizations fit the data very well...we will address this problem later, so for now, let bump our "good enough" threshold to some realizations to plot.

Let's plot just these "good" ones:


```python
#ax = df_out.phi.hist(alpha=0.5)
df_out.loc[good_enough,"phi"].hist(color="g",alpha=0.5)
plt.show()
```

Now the payoff - using our good runs, let's make some probablistic plots!  Here's our parameters

Gray blocks the full the range of the realizations.  These are within our bounds of reasonable given what we knew when we started, so those grey boxes represent our prior

The blue boxes show the runs that met our criteria, so that distribution represents our posterior


```python
for parnme in pst.par_names[:5]:
    ax = plt.subplot(111)
    parensemble.loc[:,parnme].hist(bins=10,alpha=0.5,color="0.5",ax=ax,)
    parensemble.loc[good_enough,parnme].hist(bins=10,alpha=0.5,color="b",ax=ax,)   
    ax.set_title(parnme)
    plt.ylabel('count')
    plt.tight_layout()
    plt.show()
```

Similar to the FOSM results, the future recharge (`rch1`) and porosity (`ne1`) are not influenced by calibration. The conditioned parameter values should cover the same range as unconditioned values. 

## Let's look at the forecasts

In the plots below, prior forecast distributions are shaded grey, posteriors are graded blue and the "true" value is shown as dashed black line.


```python

```


```python
for forecast in pst.forecast_names:
    ax = plt.subplot(111)
    df_out.loc[:,forecast].hist(bins=10,alpha=0.5,color="0.5",ax=ax)
    df_out.loc[good_enough,forecast].hist(bins=10,alpha=0.5,color="b",ax=ax)
    v = pst.observation_data.loc[forecast,"obsval"]
    ylim = ax.get_ylim()
    ax.plot([v,v],ylim,"k--",lw=2.0)
    ax.set_ylabel('count')
    ax.set_title(forecast)
    plt.tight_layout()
    plt.show()
```

We see for some of the foreacsts that the prior and posterior are similar - indicating that "calibration" hasn't helped reduce forecast uncertainty. That being said, given the issues noted above for high-dimensional conditioned-MC, the very very few realisations we are using here to assess the posterior make it a bit dubious.

And, as you can see, some of the true forecast values are not being covered by the posterior. So - failing.

Its hard to say how the posterior compares to the prior with so few "good enough" realizations.  To fix this problem, we have two choices:
 - run the model more times for Monte Carlo (!)
 - generate realizations that fix the data better before hand

# Advanced Monte Carlo - sampling from the linearized posterior

In the previous section, we saw that none of the realizations fit the observations anywhere close to ``phimlim`` because of the dimensionality of the pilot point problem.  

Here, we will use so linear algebra trickeration to "pre-condition" the realizations so that they have a better chance of fitting the observations. As we all know now, "linear algebra" = Jacobian!

First we need to run the calibration process to get the calibrated parameters and last Jacobian. Let's do that quick-sticks now. The next cell repeats what we did during the "freyberg regularization" tutorial. It can take a while.


```python
hbd.prep_mc(working_dir)
```


```python
pst_name = "freyberg_reg.pst"
```


```python
m_d = 'master_amc'
```


```python
pyemu.os_utils.start_workers(working_dir, # the folder which contains the "template" PEST dataset
                            'pestpp-glm', #the PEST software version we want to run
                            pst_name, # the control file to use with PEST
                            num_workers=num_workers, #how many agents to deploy
                            worker_root='.', #where to deploy the agent directories; relative to where python is running
                            master_dir=m_d, #the manager directory
                            )
```


```python
pst = pyemu.Pst(os.path.join(m_d, pst_name))
```

###  ```schur``` bayesian monte carlo

Here, we will swap out the prior parameter covariance matrix ($\boldsymbol{\Sigma}_{\theta}$) for the FOSM-based posterior parameter covariance matrix ($\overline{\boldsymbol{\Sigma}}_{\theta}$).  Everything else is exactly the same (sounds like a NIN song)


```python
sc = pyemu.Schur(jco=os.path.join(m_d,pst_name.replace(".pst",".jcb")),
                pst=pst,
                parcov=cov) # geostatistically informed covariance matrix we built earlier
```

Update the control file with the "best fit" parameters (e.g. the calibrated parameters). These values are the mean ofthe posterior parameter probability distribution.


```python
sc.pst.parrep(os.path.join(m_d,pst_name.replace(".pst",".parb")))
```


```python
parensemble = pyemu.ParameterEnsemble.from_gaussian_draw(pst=sc.pst,
                                                        cov=sc.posterior_parameter, # posterior parameter covariance
                                                        num_reals=500,)
```


```python
parensemble.to_csv(os.path.join(working_dir,"sweep_in.csv"))
```


```python

```


```python
pyemu.os_utils.start_workers(working_dir, # the folder which contains the "template" PEST dataset
                            'pestpp-swp', #the PEST software version we want to run
                            pst_name, # the control file to use with PEST
                            num_workers=num_workers, #how many agents to deploy
                            worker_root='.', #where to deploy the agent directories; relative to where python is running
                            master_dir=m_d, #the manager directory
                            )
```


```python
df_out = pd.read_csv(os.path.join(m_d,"sweep_out.csv"),index_col=0)
df_out = df_out.loc[df_out.failed_flag==0,:] #drop any failed runs
#df_out = df_out.loc[~df_out.le(-2.e27).any(axis=1)] #drop extreme values
df_out.columns = [c.lower() for c in df_out.columns]
df_out.head()
```


```python
df_out.info()
```


```python
df_out.phi.hist(bins=100);
```

Now we see, using the same Phi threshold we obtain a larger number of "good enough" parameter sets:


```python
good_enough = df_out.loc[df_out.meas_phi<acceptable_phi].index.values
print("number of good enough realisations:", good_enough.shape[0])
```


```python
#ax = df_out.phi.hist(alpha=0.5)
df_out.loc[good_enough,"phi"].hist(color="g",alpha=0.5)
plt.show()
```

Howevver, even so we are failing to entirely capture the truth values of all forecasts (see "part_time" forecast). So our forecast posterior is overly narrow (non-conservative). 

Why? Because our uncertainty analysis is __not robust__ for all our forecasts, even with non-linear Monte Carlo. The parameterization scheme is still to coarse. ...segway to high-dimension PEST setup in PART2.


```python
for forecast in pst.forecast_names:
    ax = plt.subplot(111)
    df_out.loc[:,forecast].hist(bins=10,alpha=0.5,color="0.5",ax=ax)
    df_out.loc[good_enough,forecast].hist(bins=10,alpha=0.5,color="b",ax=ax)
    v = pst.observation_data.loc[forecast,"obsval"]
    ylim = ax.get_ylim()
    ax.plot([v,v],ylim,"k--",lw=2.0)
    ax.set_ylabel('count')
    ax.set_title(forecast)
    plt.tight_layout()
    plt.show()
```


```python

```


```python

```
