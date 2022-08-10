---
layout: default
title: PEST++IES - Basics
parent: Decision Support Modelling with pyEMU and PEST++
nav_order: 7
math: mathjax3
---

# PESTPP-IES - History Matching For "Nothing", Uncertainty Analysis for Free

PESTPP-IES embodies an approach to data assimilation and uncertainty analysis which is a signifcant departure from the “calibrate first and do uncertainty analysis later”. PESTPP-IES does away with the search for a "unique" parameter field that calibrates a model. Instead, its goal is to obtain an _ensemble_ of parameter fields, all of which adequately reflect measured data and expert knowledge. 

Unlike PESTPP-GLM and PEST(\_HP), PESTPP-IES does not calculate derivatives using finite parameter differences. Instead, it calculates approximate partial derivatives from cross-covariances between parameter values and model outputs that are calculated using members of the ensemble. A major benefit that is forthcoming from this approach to calculating partial derivatives is that the number of model runs required to history-match an ensemble is independent of the number of adjustable parameters. This means that it is feasible for a model to employ a large number of adjustable parameters (history matching for "nothing"...). Conceptually, this reduces propensity for predictive bias at the same time as it protects against uncertainties of decision-critical model predictions being underestimated. 

PESTPP-IES commences by using a suite of random parameter fields sampled from the prior parameter probability distribution. Each parameter field is referred to as a “realisation”. The suite of realisations is referred to as an “ensemble”. Using an iterative procedure, PESTPP-IES modifies the parameters that comprise each realisation so that each is better able to better reflect historical data. In other words, PESTPP-IES adjusts the parameters of each realisation in order to reduce the misfit between simulated and measured observation values. 

But there is more! PESTPP-IES also accounts for the influence of observation noise. Each parameter realisation is (optionaly) adjusted to fit a slightly different set of observation values. (An observation ensemble can be provided by the user or generated automatically by PESTPP-IES.) Thus, the uncertainty incurred through observation noise gets carried through to the predictions' posterior uncertainty. 

The outcome of this multi-realisation parameter adjustment process is an ensemble of parameter fields, all of which allow the model to adequately replicate observed system behaviour. These parameter fields can be considered samples of the posterior parameter probability distribution. By simulating a forecast with this ensemble of models, a sample of the _posterior forecast probability distribution_ is obtained (uncertainty analysis for free...). 


## The Current Tutorial

In the current notebook we are going to pick up after the "observation and weights" tutorial. We have prepared a high-dimensional PEST dataset and are ready to begin history matching. Here we are going to continue with the same PEST(++) control file, add some PESTPP-IES specific options and then run it.

Then, we are going to take the opportunity to revisit some of the concepts of history matching induced bias. In subsequent tutorials, we will demonstrate several strategies to mitigate the potential for history matching to introduce bias into our forecasts.

### Admin

The modified Freyberg model is introduced in another tutorial notebook (see "freyberg intro to model"). The current notebook picks up following the "freyberg observation and weights" notebook, in which a high-dimensional PEST dataset was constructed using `pyemu.PstFrom`. You may also wish to go through the "intro to pyemu" notebook beforehand.

The next couple of cells load necessary dependencies and call a convenience function to prepare the PEST dataset folder for you. Simply press `shift+enter` to run the cells.


```python
import os
import shutil
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt;
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

Prepare the template directory:


```python
# specify the temporary working folder
t_d = os.path.join('freyberg6_template')
# get the previously generated PEST dataset
org_t_d = os.path.join("..","part2_2_obs_and_weights","freyberg6_template")
if not os.path.exists(org_t_d):
    raise Exception("you need to run the '/part2_2_obs_and_weights/freyberg_obs_and_weights.ipynb' notebook")
if os.path.exists(t_d):
    shutil.rmtree(t_d)
shutil.copytree(org_t_d,t_d)
```

## Preparing for PESTPP-IES

We shall start by running PESTPP-IES in standard, basic mode. 

First we need to load the existing control file and add some PESTPP-IES specific options. As with most PEST++ options, these mostly have pretty decent default values; however, depending on your setup you may wish to change them. We highly recommend reading the [PEST++ user manual](https://github.com/usgs/pestpp/blob/master/documentation/pestpp_users_manual.md) for full descriptions of the options and their default values.

Load the PEST control file as a `Pst` object:


```python
pst_path = os.path.join(t_d, 'freyberg_mf6.pst')
pst = pyemu.Pst(pst_path)
```

The next cell is just in place to make sure you are running the tutorials in order. Check that we are at the right stage to run PESTPP-IES:


```python
# check to see if obs&weights notebook has been run
if not pst.observation_data.observed.sum()>0:
    raise Exception("You need to run the '/part2_2_obs_and_weights/freyberg_obs_and_weights.ipynb' notebook")
```

A quick check of what PEST++ options are already in the control file:


```python
pst.pestpp_options
```

### Number of realisations
As previously mentioned, an “ensemble” refers to a set of parameter fields. Each of these parameter fields is referred to as a “realisation”. In general, the larger the ensemble, the better is the history-matching performance of PESTPP-IES. In practice, the number of realisations is limited by computing resources (more realisations require more model runs per iteration). 

The number of parameter fields comprising an ensemble should, at least, be larger than the dimensionality of the solution space of the inverse problem that is posed by the history-matching process. This ensures that the iterative ensemble smoother has access to the directions in parameter space that it requires in order to provide a good fit with the calibration dataset.

Normally, this number can only be guessed. It is usually less than the number of observations comprising a calibration dataset, and can never be greater than this. Furthermore, the dimensionality of the solution space decreases with the amount of noise that accompanies the dataset. If a Jacobian matrix is available (it may remain from a previous calibration exercise), the pyEMU (or the SUPCALC utility from the PEST suite) can be used to assess solution space dimensionality. 

For the purposes of this tutorial, lets just use 50 realizations to speed things up (feel free to use less or more - choose your own adventure!):


```python
pst.nnz_obs
```


```python
pst.pestpp_options["ies_num_reals"] = 50
```

### Prior Parameter Ensemble

As already mentioned, PESTPP-IES starts from an ensemble of parameter realisations. You can provide an ensemble yourself. If you do _not_ provide a prior parameter ensemble, PESTPP-IES will generate one itself by sampling parameter values from a multi-Gaussian distribution. In doing so, it will assume that parameter values listed in the control file "parameter data" section reflect the mean of the distribution. If no other information is provided, PESTPP-IES will calculate the standard deviation assuming that parameter upper and lower bounds reflect the 95% confidence interval, and that all parameters are statistically independent.

Now, we are all sophisticated people that recognize the importance of heterogneity and spatial (and temporal) correlation between parameters. So, alterantively, we can inform PESTPP-IES of these covariances by providing a covariance matrix to the pest++ control variable `parcov()`. We prepared a geostatistical prior parameter covariance matrix during the "pstfrom pest setup" tutorial (the file named `prior_cov.jcb`). 

Alternatively, we can provide PESTPP-IES with a pre-preprepared ensemble of parameter realisations. We also prepared one during the "pstfrom pest setup" tutorial and recorded it in binary format. It is the file named `prior_pe.jcb`.

Load the prior parameter ensemble we generated previously:


```python
[f for f in os.listdir(t_d) if f.endswith(".jcb")]
```


```python
pe = pyemu.ParameterEnsemble.from_binary(pst=pst,filename=os.path.join(t_d,"prior_pe.jcb"))
pe.shape
```

OK then. We need to tell PESTPP-IES which file to use. Easy enough, just assign the file name to the `ies_parameter_ensemble()` pestpp control variable:


```python
pst.pestpp_options['ies_parameter_ensemble'] = 'prior_pe.jcb'
```

### Observation plus noise Ensemble

We also can also include an ensemble of observation plus noise realizations; these realizations allows measurement noise to be incorporated into the posterior parameter (and therefore forecast) ensemble.  Conceptually, each parameter realization in the prior ensemble is paired with a realization of observed values plus measurement noise.  This means that each parameter realization is seeking to match a (slightly) different set of observed values.  This is the reason pestpp-ies report a "measured" objective function summary and an "actual" objective function summary; the former includes the noise realizations, the latter does not.

As for parameters, PESTPP-IES will by default generate noise realizations from `weight` values in the control file by assuming the weights are the inverse of the standard deviation of noise, and tht noise has a normal distribution. The observation ensemble is generated by adding the ensemble of noise to the observation target values in the control file. (The "obs and weights" tutorial notebook covers these topics in more detail.)

There are also other options to supply pestpp-ies with information related to measurement noise.  Using the version 2 control file format, users can add a `standard_deviation` column to the `* observation data` section.  This will allow users to seperate the noise realizations from the weights so that the weights can be used to amplify certain parts of the observation data, while the noise can represent expected uncertainty/error in the observations (and the model...).  Users can also supply a ready-made observation ensemble by specifying a file name in the `ies_observation_ensemble()` PEST++ control variable . Let's do that. In the "obs and weights" tutorial we generated a covariance matrix auto-correlated noise and recorded it in a file named `obs_cov.jcb`. Let's now use this to construct an observation ensemble:


```python
# load the covarince matrix file
obscov = pyemu.Cov.from_binary(os.path.join(t_d, 'obs_cov.jcb'))
# generate the ensemble
oe = pyemu.ObservationEnsemble.from_gaussian_draw(pst, cov=obscov, num_reals=pe.shape[0])
# record the ensemble file in the template folder
oe.to_csv(os.path.join(t_d, 'oe.csv'))
```

And assign the relevant PEST++ control variable:


```python
#pst.pestpp_options["ies_no_noise"] = True
pst.pestpp_options["ies_observation_ensemble"] = "oe.csv"
```

Right then, let's do a pre-flight check to make sure eveything is working. It's always good to do the 'ole `noptmax=0` test. Set NOPTMAX to zero and run PEST++IES once:


```python
pst.control_data.noptmax = 0
pst.write(os.path.join(t_d, 'freyberg_mf6.pst'))

pyemu.os_utils.run("pestpp-ies freyberg_mf6.pst",cwd=t_d)
```

If that was sucessfull, we can re-load it and just check the Phi:


```python
pst = pyemu.Pst(os.path.join(t_d, 'freyberg_mf6.pst'))
pst.phi
```

### Other options that are usually a good idea

PESTPP-IES has heaps of options that can be brought to bare for any given problem but most of these have good default values.  However, two options that can be very important for highly nonlinear inverse problems are related to how and when to give up on realizations that are just not doing very well with respect to phi.  These lagging realizations can dominate the obective function and therefore influence the choice of lambda values, and, ultimately, the success of the PESTPP-IES analysis. These two options are `ies_bad_phi` and `ies_bad_phi_sigma`.  The first one, `ies_bad_phi` is an absolute tolerance on the maximum phi value that will be allowed for any realization in the ensemble at any point in the algorithm.  If a realization yields a phi greater than `ies_bad_phi`, it will be "dropped" from the ensemble.  This can be an important option if cells go dry and yield an enormous phi.  

The second option, `ies_bad_phi_sigma` is a relative tolerance on the phi of a realization vs the mean phi of all the realizations in the ensemble.  The "sigma" tells us standard deviation is involved in this option:  `ies_bad_phi_sigma` is the standard deviation "distance" from the mean phi that realizations are allowed to have before they are dropped.  For example, if the mean phi is 100 and standard deviation around this mean is 20, and `ies_bad_phi_sigma` = 2, then any realization with a phi greater than 100 + (2 * 20) = 140 is dropped.  This "adaptive" rejection filter adapts as the ensemble evolves across iterations, and, for highly nonlinear problems, this option can make a huge difference.  Values between 1.5 (more agressive) and 2.5 (more tolerant) seem to work well...


```python
pst.pestpp_options["ies_bad_phi_sigma"] = 2.0
```

## Run PESTPP-IES

Right then, let's do this!


```python
# update NOPTMAX again and re-write the control file
pst.control_data.noptmax = 3
pst.write(os.path.join(t_d, 'freyberg_mf6.pst'))
```

To speed up the process, you will want to distribute the workload across as many parallel agents as possible. Normally, you will want to use the same number of agents (or less) as you have available CPU cores. Most personal computers (i.e. desktops or laptops) these days have between 4 and 10 cores. Servers or HPCs may have many more cores than this. Another limitation to keep in mind is the read/write speed of your machines disk (e.g. your hard drive). PEST and the model software are going to be reading and writting lots of files. This often slows things down if agents are competing for the same resources to read/write to disk.

The first thing we will do is specify the number of agents we are going to use.

__Attention!__

You must specify the number which is adequate for ***your*** machine! Make sure to assign an appropriate value for the following `num_workers` variable - if its too large for your machine, #badtimes:

You can check the number of physical cores avalable on your machine using `psutils`:


```python
psutil.cpu_count(logical=False)
```


```python
num_workers = 8 #update this according to your resources
```

Next, we shall specify the PEST run-manager/master directory folder as `m_d`. This is where outcomes of the PEST run will be recorded. It should be different from the `t_d` folder, which contains the "template" of the PEST dataset. This keeps everything separate and avoids silly mistakes.


```python
m_d = os.path.join('master_ies_1')
```

The following cell deploys the PEST agents and manager and then starts the run using PESTPP-IES. Run it by pressing `shift+enter`. If you wish to see the outputs in real-time, switch over to the terminal window (the one which you used to launch the `jupyter notebook`). There you should see PESTPP-IES's progress. 

If you open the tutorial folder, you should also see a bunch of new folders there named `worker_0`, `worker_1`, etc. These are the agent folders. The `master_ies` folder is where the manager is running. 

This run should take several minutes to complete (depending on the number of workers and the speed of your machine). If you get an error, make sure that your firewall or antivirus software is not blocking PESTPP-IES from communicating with the agents (this is a common problem!).

> **Pro Tip**: Running PEST from within a `jupyter notebook` has a tendency to slow things down and hog alot of RAM. When modelling in the "real world" it is often more efficient to implement workflows in scripts which you can call from the command line. 


```python
pyemu.os_utils.start_workers(t_d, # the folder which contains the "template" PEST dataset
                            'pestpp-ies', #the PEST software version we want to run
                            'freyberg_mf6.pst', # the control file to use with PEST
                            num_workers=num_workers, #how many agents to deploy
                            worker_root='.', #where to deploy the agent directories; relative to where python is running
                            master_dir=m_d, #the manager directory
                            )
```

## Explore the Outcomes

Right then. PESTPP-IES completed sucessfully. Let's take a look at some of the outcomes.

First lets open the "freyberg_mf6.rec" and make sure we understand what is going on.  Make sure to look for :
 - initial phi summary
 - number of runs per iteration
 - number of active realizations per iteration
 - parameter change summary per iteration
 
Do you understand the flow of the algorithm? 

It is usually usefull to take a look at how the ensemble performed overall at fitting historical data and how fit evolved at each PESTPP-IES iteration. Let's make a cheap Phi progress plot. PESTPP-IES recorded the Phi ($\Phi$) for each iteration in the file named `freyberg_mf6.phi.actual.csv`. Here you will find a summary of the ensembles' Phi, as well as the Phi from each individual realization. Note that the Phi in this file is calculated from the residual between simulated values and the observation values in the _control file_ (i.e. the measured data). These differ from values in `freyberg_mf6.phi.meas.csv`, which records Phi values calculated from the residual between simulated values and observation values + realizations of noise.  

Let's make a plot of the Phi progress from each realization against the total number of model runs.

Wow! Check that out. With a measly few hundered model runs we have a pretty decent fit for most of the ensemble. (How does this compare to the fit achieved with derivative-based methods in the "glm part 2" tutorial?) Recall here we are using >10k parameters...pretty amazing.


```python
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(10,3.5))
# left
ax = axes[0]
phi = pd.read_csv(os.path.join(m_d,"freyberg_mf6.phi.actual.csv"),index_col=0)
phi.index = phi.total_runs
phi.iloc[:,6:].apply(np.log10).plot(legend=False,lw=0.5,color='k', ax=ax)
ax.set_title(r'Actual $\Phi$')
ax.set_ylabel(r'log $\Phi$')
# right
ax = axes[-1]
phi = pd.read_csv(os.path.join(m_d,"freyberg_mf6.phi.meas.csv"),index_col=0)
phi.index = phi.total_runs
phi.iloc[:,6:].apply(np.log10).plot(legend=False,lw=0.2,color='r', ax=ax)
ax.set_title(r'Measured+Noise $\Phi$')
fig.tight_layout()
```

Whilst we are at it, why not plot a histogram of Phi from the last iteration. Good to get a depiction of how Phi is distributed across the ensemble:


```python
plt.figure()
phi.iloc[-1,6:].hist()
plt.title(r'Final $\Phi$ Distribution');
```

PESTPP-IES has conveniently kept track of all our observaton data, measurement noise and the model outputs from each realization at each iteration. This now allows us to go back and look at all this information in detail should we wish to do so. We are interested in looking at (1) how model outputs compare to measured data+noise and (2) the distribution of model outps for forecast observations.

Since PESTPP-IES evaluates a prior parameter ensemble, we can use the model outputs from that iteration (iteration zero) as a sample of the prior. We treat the model outputs from the ensemble for the final (best?) iteration as a sample of the posterior. Let's read in the files which PESTPP-IES recorded:


```python
pr_oe = pyemu.ObservationEnsemble.from_csv(pst=pst,filename=os.path.join(m_d,"freyberg_mf6.0.obs.csv"))
pt_oe = pyemu.ObservationEnsemble.from_csv(pst=pst,filename=os.path.join(m_d,"freyberg_mf6.{0}.obs.csv".format(pst.control_data.noptmax)))
noise = pyemu.ObservationEnsemble.from_csv(pst=pst,filename=os.path.join(m_d,"freyberg_mf6.obs+noise.csv"))
```

Make a quick comparison of Phi between the prior and posterior. Nice improvement overall!


```python
fig,ax = plt.subplots(1,1)
pr_oe.phi_vector.apply(np.log10).hist(ax=ax,fc="0.5",ec="none",alpha=0.5,density=False)
pt_oe.phi_vector.apply(np.log10).hist(ax=ax,fc="b",ec="none",alpha=0.5,density=False)
_ = ax.set_xlabel("$log_{10}\\phi$")
```

Finally, what we really want to see. Let's plot timeseries of simulated versus measured observation values. We are going to do this many times in this notebook, so let's make a function:


```python
def plot_tseries_ensembles(pr_oe, pt_oe, noise, onames=["hds","sfr"]):
    pst.try_parse_name_metadata()
    # get the observation data from the control file and select 
    obs = pst.observation_data.copy()
    # onames provided in oname argument
    obs = obs.loc[obs.oname.apply(lambda x: x in onames)]
    # only non-zero observations
    obs = obs.loc[obs.obgnme.apply(lambda x: x in pst.nnz_obs_groups),:]
    # make a plot
    ogs = obs.obgnme.unique()
    fig,axes = plt.subplots(len(ogs),1,figsize=(10,2*len(ogs)))
    ogs.sort()
    # for each observation group (i.e. timeseries)
    for ax,og in zip(axes,ogs):
        # get values for x axis
        oobs = obs.loc[obs.obgnme==og,:].copy()
        oobs.loc[:,"time"] = oobs.time.astype(float)
        oobs.sort_values(by="time",inplace=True)
        tvals = oobs.time.values
        onames = oobs.obsnme.values
        # plot prior
        [ax.plot(tvals,pr_oe.loc[i,onames].values,"0.5",lw=0.5,alpha=0.5) for i in pr_oe.index]
        # plot posterior
        [ax.plot(tvals,pt_oe.loc[i,onames].values,"b",lw=0.5,alpha=0.5) for i in pt_oe.index]
        # plot measured+noise 
        oobs = oobs.loc[oobs.weight>0,:]
        tvals = oobs.time.values
        onames = oobs.obsnme.values
        [ax.plot(tvals,noise.loc[i,onames].values,"r",lw=0.5,alpha=0.5) for i in noise.index]
        ax.plot(oobs.time,oobs.obsval,"r-",lw=2)
        ax.set_title(og,loc="left")
    fig.tight_layout()
    return fig
```

OK, now plot the time series of non-zero observation groups. Let's just plot the timeseries of absolute values of heads and stream gage flow.

In the plots below, light grey lines are timeseries simulated with the prior parameter ensemble. Blue lines are model outputs simulted using the posterior parameter ensemble. Red lines are the ensemble of measurement + noise. 

Looks like we are getting an excellent fit. All the blue lines are within the same areas as the red lines. This implies we have acheived a level of fit comensurate with measurment noise. Sounds very positive. What do you think? Success? 


```python
fig = plot_tseries_ensembles(pr_oe, pt_oe, noise, onames=["hds","sfr"])
```

### (Optional) Some additional filtering

Often a few realizations perform particularily poorly. In such cases it can be good practice to remove them. Easy enough to do. For example, the cell below drops any realizations that did not achieve a Phi lower than the  threshold value assigned to the variable `thresh`.


```python
# threshold Phi
thres = phi.iloc[-1,6:].quantile(0.9)
# drop reals with Phi > thresh
pv = pt_oe.phi_vector
keep = pv.loc[pv<thres]
if keep.shape[0] != pv.shape[0]:
    print("reducing posterior ensemble from {0} to {1} realizations".format(pv.shape[0],keep.shape[0]))
    pt_oe = pt_oe.loc[keep.index,:]
    fig,ax = plt.subplots(1,1)
    pr_oe.phi_vector.apply(np.log10).hist(ax=ax,fc="0.5",ec="none",alpha=0.5,density=False)
    pt_oe.phi_vector.apply(np.log10).hist(ax=ax,fc="b",ec="none",alpha=0.5,density=False)
    _ = ax.set_xlabel("$log_{10}\\phi$")
else:
    print('No realizations dropped.')
if pt_oe.shape[0] == 0:
    print("filtered out all posterior realization #sad")
```

## Forecasts

As usual, we bring this story back to the forecasts - after all they are why we are modelling. As this is a synthetic case and we know the "truth", we have benefit of being able to check the reliability of our forecast. Let's do that now.

A quick reminder of the observations that record our forecast value of interest:


```python
pst.forecast_names
```

Now, we are going to plot histograms of the forecast values simulated by the model using parmaeters from the (1) prior and (2) posterior ensembles. Simulated forecast values are recorded in the observation ensembles we read in earlier (as are all observations listed in the PEST contorl file).

Again, we are going to do this alot in the current and subsequent tutorials, so let's just make a function:


```python
def plot_forecast_hist_compare(pt_oe,pr_oe, last_pt_oe=None,last_prior=None ):
        num_plots = len(pst.forecast_names)
        num_cols = 1
        if last_pt_oe!=None:
            num_cols=2
        fig,axes = plt.subplots(num_plots, num_cols, figsize=(5*num_cols,num_plots * 2.5), sharex='row',sharey='row')
        for axs,forecast in zip(axes, pst.forecast_names):
            # plot first column with currrent outcomes
            if num_cols==1:
                axs=[axs]
            ax = axs[0]
            # just for aesthetics
            bin_cols = [pt_oe.loc[:,forecast], pr_oe.loc[:,forecast],]
            if num_cols>1:
                bin_cols.extend([last_pt_oe.loc[:,forecast],last_prior.loc[:,forecast]])
            bins=np.histogram(pd.concat(bin_cols),
                                         bins=20)[1] #get the bin edges
            pr_oe.loc[:,forecast].hist(facecolor="0.5",alpha=0.5, bins=bins, ax=ax)
            pt_oe.loc[:,forecast].hist(facecolor="b",alpha=0.5, bins=bins, ax=ax)
            ax.set_title(forecast)
            fval = pst.observation_data.loc[forecast,"obsval"]
            ax.plot([fval,fval],ax.get_ylim(),"r-")
            # plot second column with other outcomes
            if num_cols >1:
                ax = axs[1]
                last_prior.loc[:,forecast].hist(facecolor="0.5",alpha=0.5, bins=bins, ax=ax)
                last_pt_oe.loc[:,forecast].hist(facecolor="b",alpha=0.5, bins=bins, ax=ax)
                ax.set_title(forecast)
                fval = pst.observation_data.loc[forecast,"obsval"]
                ax.plot([fval,fval],ax.get_ylim(),"r-")
        # set ax column titles
        if num_cols >1:
            axes.flatten()[0].text(0.5,1.2,"Current Attempt", transform=axes.flatten()[0].transAxes, weight='bold', fontsize=12, horizontalalignment='center')
            axes.flatten()[1].text(0.5,1.2,"Previous Attempt", transform=axes.flatten()[1].transAxes, weight='bold', fontsize=12, horizontalalignment='center')
        fig.tight_layout()
        return fig
```

Plot the forecast histograms. Grey columns are the prior. Blue columns are the posterior:


```python
fig = plot_forecast_hist_compare(pt_oe=pt_oe, pr_oe=pr_oe)
```

Ruh roh!  The posterior isn't covering the correct values for some of forecasts. Major bad times. 

But hold on! The prior does (nearly) cover the true values for all forecasts. So that implies there is somewhere between the prior and posterior we have now, which is optimal with respect to all forecasts. Hmmm...so this means that history matching made our prediction worse. We have incurred forecast-sensitive bias through the parameter adjustment process. How can we fit historical data so well but get the "wrong" answer for some of the forecasts?

Here we have seen a very important concept: when you are using an imperfect model (compared to the truth), the link between a "good fit" and robust forecast is broken. A good fit does not mean a good forecaster! This is particularily the case for forecasts that are sensitive to combinations of parameters that occupy the history-matching null space (see [Doherty and Moore (2020)](https://s3.amazonaws.com/docs.pesthomepage.org/documents/model_complexity_monograph.pdf) for a discussion of these concepts). In other words, forecasts which rely on (combinations of) parameters that are not informed by available observation data. (In our case, an example is the "headwater" forecast.)

### Underfitting
So, somewhere between the prior and the final iteration is the optimal amount of parameter adjustment that (1) reduces uncertainty but (2) does not incur forecast bias. We saw that the posterior for our last iteration achieved a level of fit comensurate with measreument error. So we achieved as good a fit as could be expected with the data. But now we have seen that getting that fit incurred bias. So what happnes if we "underfit"? i.e. accept a level of fit which is _worse_ than can be explained by noise in the _measured_ data.

Luckily, we can just load up a previous iteration of PESTPP-IES results and use those! Let's see if that resolves our predicament.

Check the outcomes from the first iteration:


```python
iter_to_use_as_posterior = 1
pt_oe_iter = pyemu.ObservationEnsemble.from_csv(pst=pst,filename=os.path.join(m_d,"freyberg_mf6.{0}.obs.csv".\
                                                                         format(iter_to_use_as_posterior)))

```

So, as you'd expect, some improvement in Phi, but not as much as for the final iteration:


```python
fig,ax = plt.subplots(1,1)
pr_oe.phi_vector.apply(np.log10).hist(ax=ax,fc="0.5",ec="none",alpha=0.5,density=False)
pt_oe_iter.phi_vector.apply(np.log10).hist(ax=ax,fc="b",ec="none",alpha=0.5,density=False)
_ = ax.set_xlabel("$log_{10}\phi$")
```

Let's take a look at those time series again. 

There we go...much less satisfying. Clearly not "as good" a replica of observed behaviour. We also see more variance in the simulated equivalents (blue lines) to the observations, meaning we arent fitting the historic observations as well...basically, we have only eliminated the extreme prior realizations - we can call this "light" conditioning or "underfitting"...


```python
fig = plot_tseries_ensembles(pr_oe, pt_oe_iter,noise, onames=["hds","sfr"])
```

Finaly, let's see what has happened to the forecasts. The next cell will plot the forecast histograms from the current "posterior" (right column of plots), alongside those from the previous attempt (left column of plots)


```python
fig = plot_forecast_hist_compare(pt_oe=pt_oe_iter,pr_oe=pr_oe,
                                last_pt_oe=pt_oe,last_prior=pr_oe
                                )
```

Interesting! That's a bit better. What have we done? We've accepted "more uncertainty" for a reduced propensity of inducing forecast bias. Perhaps if we had more realizations we would have gotten a wider sample of the posterior? But even if we hadn't failed to capture the truth, in the real-world how would we know? So...should we just stick with prior? (Assuming the prior is adequately described...) Feeling depressed yet? Worry not, in our next tutorial we will introduce some coping strategies. 

In summary, we have learnt:
 - How to configure and run PESTPP-IES
 - How to explore some of the outcomes
 - __Most importantly__, we have learnt that getting a good fit does not a good predictor make. Trying to fit measured data with an imperfect model (which every model is...) can induce bias. 

If prior uncertainty is suficient for decision-support purposes, it may be more robust to forgo history matching entirely. However, if uncertainty reduction is required, additional strategies to avoid inducing bias are needed. We will address some of these in subsequent tutorials.

## Optional

#### General Question: How do we do better with ensemble methods?  

#### General Answer: use more realizations


```python
pst.pestpp_options["ies_num_reals"] = 150
m_d = "master_ies_1a"
pst.write(os.path.join(t_d,"freyberg_mf6.pst"),version=2)
pyemu.os_utils.start_workers(t_d, # the folder which contains the "template" PEST dataset
                            'pestpp-ies', #the PEST software version we want to run
                            'freyberg_mf6.pst', # the control file to use with PEST
                            num_workers=num_workers, #how many agents to deploy
                            worker_root='.', #where to deploy the agent directories; relative to where python is running
                            master_dir=m_d, #the manager directory
                            )
```


```python
pr_oe = pyemu.ObservationEnsemble.from_csv(pst=pst,filename=os.path.join(m_d,"freyberg_mf6.0.obs.csv"))
pt_oe = pyemu.ObservationEnsemble.from_csv(pst=pst,filename=os.path.join(m_d,"freyberg_mf6.{0}.obs.csv".format(pst.control_data.noptmax)))
noise = pyemu.ObservationEnsemble.from_csv(pst=pst,filename=os.path.join(m_d,"freyberg_mf6.obs+noise.csv"))
```


```python
fig = plot_forecast_hist_compare(pt_oe=pt_oe_iter,pr_oe=pr_oe,
                                last_pt_oe=pt_oe,last_prior=pr_oe
                                )
```

having more realizations has two benefits:  more samples for uncertainty analysis and better resolution of the empirical first-order relations between parameters and observations.  In this case, these two combined effects have helped us (nearly) bracket the true value for each forecast - yeh!  So always use a many realizations as you can tolerate!


```python

```
