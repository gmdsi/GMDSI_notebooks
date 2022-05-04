# PESTPP-IES

words here

### 1. The modified Freyberg PEST dataset

The modified Freyberg model is introduced in another tutorial notebook (see "freyberg intro to model"). The current notebook picks up following the "freyberg psfrom pest setup" notebook, in which a high-dimensional PEST dataset was constructed using `pyemu.PstFrom`. You may also wish to go through the "intro to pyemu" notebook beforehand.

The next couple of cells load necessary dependencies and call a convenience function to prepare the PEST dataset folder for you. This is the same dataset that was constructed during the "freyberg pstfrom pest setup" tutorial. Simply press `shift+enter` to run the cells.


```python
import os
import shutil
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import pyemu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt;

import sys
sys.path.append(os.path.join("..", "..", "dependencies"))
import pyemu
import flopy

sys.path.append("..")
import herebedragons as hbd
```

Some file and dir manipulations to prepare:


```python
# specify the temporary working folder
t_d = os.path.join('freyberg6_template')

org_t_d = os.path.join("..","part2_2_obs_and_weights","freyberg6_template")
if not os.path.exists(org_t_d):
    raise Exception("you need to run the '/part2_2_obs_and_weights/freyberg_obs_and_weights.ipynb' notebook")

if os.path.exists(t_d):
    shutil.rmtree(t_d)
shutil.copytree(org_t_d,t_d)
                       

```

Load the PEST control file as a `Pst` object.


```python
pst_path = os.path.join(t_d, 'freyberg_mf6.pst')
pst = pyemu.Pst(pst_path)
pst.observation_data.columns
```

Check that we are at the right stage to run ies:


```python
if "observed" not in pst.observation_data.columns:
    raise Exception("you need to run the '/part2_2_obs_and_weights/freyberg_obs_and_weights.ipynb' notebook")
```

Load the prior parameter ensemble we generated previously:


```python
[f for f in os.listdir(t_d) if f.endswith(".jcb")]
```


```python
pe = pyemu.ParameterEnsemble.from_binary(pst=pst,filename=os.path.join(t_d,"prior_pe.jcb"))
pe.shape
```

### 3. Run PESTPP-IES in Parallel

Here we go...this is gonna epic!

We need to tell PESTPP-IES to use the geostatistical prior parameter ensemble we generated previously. And lets just use 50 realizations to speed things up (feel free to use less or more - choose your own adventure!)


```python
pst.pestpp_options['ies_parameter_ensemble'] = 'prior_pe.jcb'
pst.pestpp_options["ies_num_reals"] = 50
```

Then, re-write the PEST control file. If you open `freyberg_mf6.pst` in a text editor, you'll see a new PEST++ control variable has been added.


```python
pst.control_data.noptmax = 0
pst.write(os.path.join(t_d, 'freyberg_mf6.pst'))
```

Always good to do the 'ole `noptmax=0` test:


```python
pyemu.os_utils.run("pestpp-ies freyberg_mf6.pst",cwd=t_d)
```


```python
pst = pyemu.Pst(os.path.join(t_d, 'freyberg_mf6.pst'))
assert np.abs(pst.phi - 257.328) < 1.0e-1,pst.phi
```

To speed up the process, you will want to distribute the workload across as many parallel agents as possible. Normally, you will want to use the same number of agents (or less) as you have available CPU cores. Most personal computers (i.e. desktops or laptops) these days have between 4 and 10 cores. Servers or HPCs may have many more cores than this. Another limitation to keep in mind is the read/write speed of your machines disk (e.g. your hard drive). PEST and the model software are going to be reading and writting lots of files. This often slows things down if agents are competing for the same resources to read/write to disk.

The first thing we will do is specify the number of agents we are going to use.

# Attention!

You must specify the number which is adequate for ***your*** machine! Make sure to assign an appropriate value for the following `num_workers` variable - if its too large for your machine, #badtimes:


```python
num_workers = 10
```

Next, we shall specify the PEST run-manager/master directory folder as `m_d`. This is where outcomes of the PEST run will be recorded. It should be different from the `t_d` folder, which contains the "template" of the PEST dataset. This keeps everything separate and avoids silly mistakes.


```python
pst.control_data.noptmax = 3
pst.write(os.path.join(t_d, 'freyberg_mf6.pst'))
m_d = os.path.join('master_ies')
```

The following cell deploys the PEST agents and manager and then starts the run using `pestpp-ies`. Run it by pressing `shift+enter`.

If you wish to see the outputs in real-time, switch over to the terminal window (the one which you used to launch the `jupyter notebook`). There you should see `pestpp-ies`'s progress. 

If you open the tutorial folder, you should also see a bunch of new folders there named `worker_0`, `worker_1`, etc. These are the agent folders. The `master_ies` folder is where the manager is running. 

This run should take several minutes to complete (depending on the number of workers and the speed of your machine). If you get an error, make sure that your firewall or antivirus software is not blocking `pestpp-ies` from communicating with the agents (this is a common problem!).

> **Pro Tip**: Running PEST from within a `jupyter notebook` has a tendency to slow things down and hog alot of RAM (at least if you are using Visual Studio Code, as I am). When modelling in the "real world" it is more efficient to implement workflows in scripts which you can call from the command line. For example, for this case it took me 20min when running `pestpp-ies` from the `jupyter notebook`, but only 5min when running form the comand line. If you inspect the tutorial folder, you will find a file named `run.py` that accomplishes this. 


```python
pyemu.os_utils.start_workers(t_d, # the folder which contains the "template" PEST dataset
                            'pestpp-ies', #the PEST software version we want to run
                            'freyberg_mf6.pst', # the control file to use with PEST
                            num_workers=num_workers, #how many agents to deploy
                            worker_root='.', #where to deploy the agent directories; relative to where python is running
                            master_dir=m_d, #the manager directory
                            )
```

### 4. Explore the Outcomes

words here


```python
pr_oe = pyemu.ObservationEnsemble.from_csv(pst=pst,filename=os.path.join(m_d,"freyberg_mf6.0.obs.csv"))
pt_oe = pyemu.ObservationEnsemble.from_csv(pst=pst,filename=os.path.join(m_d,"freyberg_mf6.{0}.obs.csv".format(pst.control_data.noptmax)))
noise = pyemu.ObservationEnsemble.from_csv(pst=pst,filename=os.path.join(m_d,"freyberg_mf6.obs+noise.csv"))

```

We can take a look at the distribution of Phi for both prior ensemble and posterior ensemble:


```python
fig,ax = plt.subplots(1,1)
pr_oe.phi_vector.apply(np.log10).hist(ax=ax,fc="0.5",ec="none",alpha=0.5,density=False)
pt_oe.phi_vector.apply(np.log10).hist(ax=ax,fc="b",ec="none",alpha=0.5,density=False)
_ = ax.set_xlabel("$log_{10}\\phi$")
```

Finally, let's plot the obs vs sim timeseries - everyone's fav!


```python
pst.try_parse_name_metadata()
obs = pst.observation_data.copy()
obs = obs.loc[obs.oname.apply(lambda x: x in ["hds","sfr"])]
obs = obs.loc[obs.obgnme.apply(lambda x: x in pst.nnz_obs_groups),:]
obs.obgnme.unique()
```


```python
ogs = obs.obgnme.unique()
fig,axes = plt.subplots(len(ogs),1,figsize=(10,5*len(ogs)))
ogs.sort()
for ax,og in zip(axes,ogs):
    oobs = obs.loc[obs.obgnme==og,:].copy()
    oobs.loc[:,"time"] = oobs.time.astype(float)
    oobs.sort_values(by="time",inplace=True)
    tvals = oobs.time.values
    onames = oobs.obsnme.values
    [ax.plot(tvals,pr_oe.loc[i,onames].values,"0.5",lw=0.1,alpha=0.5) for i in pr_oe.index]
    [ax.plot(tvals,pt_oe.loc[i,onames].values,"b",lw=0.1,alpha=0.5) for i in pt_oe.index]
       
    oobs = oobs.loc[oobs.weight>0,:]
    tvals = oobs.time.values
    onames = oobs.obsnme.values
    [ax.plot(tvals,noise.loc[i,onames].values,"r",lw=0.1,alpha=0.5) for i in noise.index]
    ax.plot(oobs.time,oobs.obsval,"r-",lw=2)
    ax.set_title(og,loc="left")
```

How do we feel about these plots?  In general, its a really (really!) good fit...is that ok?  

### optional additional filtering

apply an optional additional phi filter to remove poor fitting realizations - usually a good idea in practice


```python
thres = 100
pv = pt_oe.phi_vector
keep = pv.loc[pv<thres]
if keep.shape[0] != pv.shape[0]:
    print("reducing posterior ensemble from {0} to {1} realizations".format(pv.shape[0],keep.shape[0]))
    pt_oe = pt_oe.loc[keep.index,:]
    fig,ax = plt.subplots(1,1)
    pr_oe.phi_vector.apply(np.log10).hist(ax=ax,fc="0.5",ec="none",alpha=0.5,density=False)
    pt_oe.phi_vector.apply(np.log10).hist(ax=ax,fc="b",ec="none",alpha=0.5,density=False)
    _ = ax.set_xlabel("$log_{10}\\phi$")
if pt_oe.shape[0] == 0:
    print("filtered out all posterior realization #sad")
```

### 5. Forecasts

As usual, we bring this story back to the forecasts - after all they are why we are modelling.


```python
pst.forecast_names
```


```python
for forecast in pst.forecast_names:
    plt.figure()
    ax = pr_oe.loc[:,forecast].hist(facecolor="0.5",alpha=0.5)
    ax = pt_oe.loc[:,forecast].hist(facecolor="b",alpha=0.5)
    
    ax.set_title(forecast)
    fval = pst.observation_data.loc[forecast,"obsval"]
    ax.plot([fval,fval],ax.get_ylim(),"r-")
```

Ruh roh!  The posterior isnt covering the correct values for several forecasts. But the prior does, so that implies there is somewhere between the prior and posterior we have now that is optimal with respect to the forecasts.  Luckily, we can just load up a previous iteration of ies results and use those!


```python
iter_to_use_as_posterior = 1
pt_oe = pyemu.ObservationEnsemble.from_csv(pst=pst,filename=os.path.join(m_d,"freyberg_mf6.{0}.obs.csv".\
                                                                         format(iter_to_use_as_posterior)))

```


```python
fig,ax = plt.subplots(1,1)
pr_oe.phi_vector.apply(np.log10).hist(ax=ax,fc="0.5",ec="none",alpha=0.5,density=False)
pt_oe.phi_vector.apply(np.log10).hist(ax=ax,fc="b",ec="none",alpha=0.5,density=False)
_ = ax.set_xlabel("$log_{10}\phi$")
```

The posterior phi values are more similar to the prior....


```python


ogs = obs.obgnme.unique()
fig,axes = plt.subplots(len(ogs),1,figsize=(10,5*len(ogs)))
ogs.sort()
for ax,og in zip(axes,ogs):
    oobs = obs.loc[obs.obgnme==og,:].copy()
    oobs.loc[:,"time"] = oobs.time.astype(float)
    oobs.sort_values(by="time",inplace=True)
    tvals = oobs.time.values
    onames = oobs.obsnme.values
    [ax.plot(tvals,pr_oe.loc[i,onames].values,"0.5",lw=0.1,alpha=0.5) for i in pr_oe.index]
    [ax.plot(tvals,pt_oe.loc[i,onames].values,"b",lw=0.1,alpha=0.5) for i in pt_oe.index]
       
    oobs = oobs.loc[oobs.weight>0,:]
    tvals = oobs.time.values
    onames = oobs.obsnme.values
    [ax.plot(tvals,noise.loc[i,onames].values,"r",lw=0.1,alpha=0.5) for i in noise.index]
    ax.plot(oobs.time,oobs.obsval,"r-",lw=2)
    ax.set_title(og,loc="left")
```

Now we see more variance in the simulated equivalents to the observations, meaning we arent fitting the historic observations as well...basically, we have only elimiated the extreme prior realizations - we can call this "light" conditioning or "underfitting"...

Let's see what has happened to the forecasts:


```python
for forecast in pst.forecast_names:
    plt.figure()
    ax = pr_oe.loc[:,forecast].hist(facecolor="0.5",alpha=0.5)
    ax = pt_oe.loc[:,forecast].hist(facecolor="b",alpha=0.5)
    
    ax.set_title(forecast)
    fval = pst.observation_data.loc[forecast,"obsval"]
    ax.plot([fval,fval],ax.get_ylim(),"r-")
```

Ok, now things are getting interesting - the posterior is covering the truth...success?


```python

```
