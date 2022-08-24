---
layout: default
title: PEST++DA - Getting Ready
parent: Decision Support Modelling with pyEMU and PEST++
nav_order: 10
math: mathjax3
---

# Prepare for sequential data assimilation

Sequential state-parameter estimation is a whole new beast for the PEST world.  Every other tool in PEST and PEST++ operate on the concept of "batch" estimation, where the model is run forward for the full simulation period and PEST(++) simply calls the model and reads the results.  In sequential estimation, PESTPP-DA takes control of the advancing of simulation time.  This opens up some powerful new analyses but requires us to heavily modify the PEST interface and model itself.  This horrible notebook does that...

### The modified Freyberg PEST dataset

The modified Freyberg model is introduced in another tutorial notebook (see "freyberg intro to model"). The current notebook picks up following the "freyberg psfrom pest setup" notebook, in which a high-dimensional PEST dataset was constructed using `pyemu.PstFrom`. You may also wish to go through the "intro to pyemu" notebook beforehand.

The next couple of cells load necessary dependencies and call a convenience function to prepare the PEST dataset folder for you. This is the same dataset that was constructed during the "freyberg pstfrom pest setup" tutorial. Simply press `shift+enter` to run the cells.

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

org_t_d = os.path.join("..","part2_2_obs_and_weights","freyberg6_template")
if not os.path.exists(org_t_d):
    raise Exception("you need to run the '/part2_2_obs_and_weights/freyberg_obs_and_weights.ipynb' notebook")

if os.path.exists(t_d):
    shutil.rmtree(t_d)
shutil.copytree(org_t_d,t_d)
```

## Modify the model itself

There are several modifications we need to make to both the model and pest interface in order to go from batch estimation to sequential estimation.  First, we need to make the model a single stress period model - PESTPP-DA will take control of the advancement of simulation time...


```python
with open(os.path.join(t_d,"freyberg6.tdis"),'w') as f:
    f.write("# new tdis written hastily at {0}\n]\n".format(datetime.now()))
    f.write("BEGIN options\n  TIME_UNITS days\nEND options\n\n")
    f.write("BEGIN dimensions\n  NPER 1\nEND dimensions\n\n")
    f.write("BEGIN perioddata\n  1.0  1 1.0\nEND perioddata\n\n")

          
```

Now, just make sure we havent done something dumb (er than usual):


```python
pyemu.os_utils.run("mf6",cwd=t_d)
```

# Now for the hard part: modifying the interface from batch to sequential

## This is going to be rough...

First, let's assign cycle numbers to the time-varying parameters and their template files.  The "cycle" concept is core to squential estimation with PESTPP-DA.  A cycle can be thought of as a unit of simulation time that we are interested in. In the PEST interface, a cycle defines a set of parameters and observations, so you can think of a cycle as a "sub-problem" in the PEST since - PESTPP-DA creates this subproblem under the hood for us. For a given cycle, we will "assimilate" all non-zero weighted obsevations in that cycle using the adjustable parameters and states in that cycle.  If a parameter/observation (and associated input/outputs files) are assigned a cycle value of -1, that means it applies to all cycles. 


```python
pst_path = os.path.join(t_d, 'freyberg_mf6.pst')
pst = pyemu.Pst(pst_path)
if "observed" not in pst.observation_data.columns:
    raise Exception("you need to run the '/part2_obs_and_weights/freyberg_obs_and_weights.ipynb' notebook")
```


```python
df = pst.model_input_data
df
```


```python
df.loc[:,"cycle"] = -1
```

Here we want to assign the template file info associated with time-varying SFR parameters to the appropriate cycle - this includes resetting the actual model-input filename since we only have only stress period in the model now


```python
sfrdf = df.loc[df.pest_file.apply(lambda x: "sfr" in x and "cond" not in x),:]
sfrdf.loc[:,"inst"] = sfrdf.pest_file.apply(lambda x: int(x.split("inst")[1].split("_")[0]))
sfrdf.loc[:,"model_file"] = sfrdf.model_file.iloc[0]
sfrdf.head()
```


```python
df.loc[sfrdf.index,"cycle"] = sfrdf.inst.values
df.loc[sfrdf.index,"model_file"] = sfrdf.model_file.values

df.loc[sfrdf.index,:]
```

And the same for the template files associated with the WEL package time-varying parameters 


```python
weldf = df.loc[df.pest_file.str.contains('wel'),:]
df.loc[weldf.index,"cycle"] = weldf.pest_file.apply(lambda x: int(x.split("inst")[1].split("_")[0]))
grdf = weldf.loc[weldf.pest_file.str.contains("welgrd"),:]
df.loc[grdf.index,"model_file"] = grdf.model_file.iloc[0]
cndf = weldf.loc[weldf.pest_file.str.contains("welcst"),:]
df.loc[cndf.index,"model_file"] = cndf.model_file.iloc[0]

df.loc[weldf.index,:]
```

And the same for the template files associated with the RCH package time-varying parameters


```python
rchdf = df.loc[df.pest_file.apply(lambda x: "rch" in x and "tcn" in x),:]
df.loc[rchdf.index,"cycle"] = rchdf.pest_file.apply(lambda x: int(x.split("tcn")[0].split("_")[-1])-1)
df.loc[rchdf.index,"model_file"] = rchdf.model_file.iloc[0]
df.loc[rchdf.index,:].head()
```

and for rch pp and grd:


```python
df.loc[df.pest_file.apply(lambda x: "rch" in x ),:]
```


```python
rchgrdf = df.loc[df.pest_file.apply(lambda x: "rch" in x and "gr" in x),:]
df.loc[rchgrdf.index,"cycle"] = rchgrdf.pest_file.apply(lambda x: int(x.split("gr")[0].split("rchrecharge")[-1])-1)
df.loc[rchgrdf.index,"model_file"] = rchgrdf.model_file.iloc[0]
df.loc[rchgrdf.index,:].head()
```


```python
rchppdf = df.loc[df.pest_file.apply(lambda x: "rch" in x and "pp" in x),:]
df.loc[rchppdf.index,"cycle"] = rchppdf.pest_file.apply(lambda x: int(x.split("pp")[0].split("rchrecharge")[-1])-1)
df.loc[rchppdf.index,"model_file"] = rchppdf.model_file.iloc[0]
df.loc[rchppdf.index,:].head()
```

Now we need to set the cycle numbers for the parmaeters themselves - good luck doing this with recarrays!


```python
par = pst.parameter_data
par.loc[:,"cycle"] = -1
```

time-varying well parameters - the parmaeter instance ("inst") value assigned by `PstFrom` correspond to the zero-based stress period number, so we can just use that as the cycle value - nice!


```python
wpar = par.loc[par.parnme.str.contains("wel"),:]
par.loc[wpar.index,"cycle"] = wpar.inst.astype(int)
```

Same for sfr time-varying parameters:


```python
spar = par.loc[par.parnme.apply(lambda x: "sfr" in x and "cond" not in x),:]
par.loc[spar.index,"cycle"] = spar.inst.astype(int)
```

And the same for time-varying recharge parameters


```python
rpar = par.loc[par.parnme.apply(lambda x: "rch" in x and "tcn" in x),:]
par.loc[rpar.index,"cycle"] = rpar.parnme.apply(lambda x: int(x.split("tcn")[0].split("_")[-1])-1)


rgrpar = par.loc[par.parnme.apply(lambda x: "rch" in x and "gr" in x),:]
par.loc[rgrpar.index,"cycle"] = rgrpar.parnme.apply(lambda x: int(x.split("gr")[0].split("rchrecharge")[-1])-1)


rpppar = par.loc[par.parnme.apply(lambda x: "rch" in x and "pp" in x),:]
par.loc[rpppar.index,"cycle"] = rpppar.parnme.apply(lambda x: int(x.split("pp")[0].split("rchrecharge")[-1])-1)

```

Now we need to add a special parameter that will be used to control the length of the stress period that the single-stress-period model will simulate.  As usual, we do this with a template file:


```python
with open(os.path.join(t_d,"freyberg6.tdis.tpl"),'w') as f:
    f.write("ptf ~\n")
    f.write("# new tdis written hastily at {0}\n]\n".format(datetime.now()))
    f.write("BEGIN options\n  TIME_UNITS days\nEND options\n\n")
    f.write("BEGIN dimensions\n  NPER 1\nEND dimensions\n\n")
    f.write("BEGIN perioddata\n  ~  perlen  ~  1 1.0\nEND perioddata\n\n")
```


```python
pst.add_parameters(os.path.join(t_d,"freyberg6.tdis.tpl"),pst_path=".")
```

Let's also add a dummy parameter that is the cycle number - this will be written into the working dir at runtime and can help us get our pre and post processors going for sequential estimation


```python
tpl_file = os.path.join(t_d,"cycle.dat.tpl")
with open(tpl_file,'w') as f:
    f.write("ptf ~\n")
    f.write("cycle_num ~  cycle_num   ~\n")
pst.add_parameters(tpl_file,pst_path=".")
```


```python
pst.parameter_data.loc["perlen","partrans"] = "fixed"
pst.parameter_data.loc["perlen","cycle"] = -1
pst.parameter_data.loc["cycle_num","partrans"] = "fixed"
pst.parameter_data.loc["cycle_num","cycle"] = -1
pst.model_input_data.loc[pst.model_input_data.index[-2],"cycle"] = -1
pst.model_input_data.loc[pst.model_input_data.index[-1],"cycle"] = -1
pst.model_input_data.tail()
```

Since `perlen` needs to change over cycles (month to month), we a way to tell PESTPP-DA to change it.  We could setup separate parameters and template for each cycle (e.g. `perlen_0`,`perlen_1`,`perlen_2`, etc, for cycle 0,1,2, etc), but this is cumbersome.  Instead, we can use a parameter cycle table to specific the value of the `perlen` parameter for each cycle (only fixed parameters can be treated this way...):


```python
sim = flopy.mf6.MFSimulation.load(sim_ws=org_t_d,load_only=["dis"])
org_perlen = sim.tdis.perioddata.array["perlen"]
org_perlen
```


```python
df = pd.DataFrame({"perlen":org_perlen},index=np.arange(org_perlen.shape[0]))
df.loc[:,"cycle_num"] = df.index.values
df
```


```python
df.T.to_csv(os.path.join(t_d,"par_cycle_table.csv"))
pst.pestpp_options["da_parameter_cycle_table"] = "par_cycle_table.csv"
```

### Observation data

Now for the observation data - yuck!  In the existing PEST interface, we include simulated GW level values in all active cells as observations, but then we also used the MF6 head obs process to make it easier for us to get the obs v sim process setup.  Here, we will ditch the MF6 head obs process and just rely on the arrays of simulated GW levels - these will be included in every cycle.  The arrays of simulated groundwater level in every active model cell will serve two roles: outputs to compare with data for assimilation (at specific locations in space and time) and also as dynamic states that will be linked to the initial head parameters - this is where things will get really exciting...


```python
obs = pst.observation_data
obs
```


```python
pst.model_output_data
```

Unfortunately, there is not an easy way to carry the particle-based forecasts, so let's drop those...


```python
pst.drop_observations(os.path.join(t_d,"freyberg_mp.mpend.ins"),pst_path=".")
```

Same for temporal-based difference observations....


```python
pst.drop_observations(os.path.join(t_d,"sfr.tdiff.csv.ins"),pst_path=".")
```


```python
pst.drop_observations(os.path.join(t_d,"heads.tdiff.csv.ins"),pst_path=".")
```

Here is where we will drop the MF6 head obs type observations - remember, we will instead rely on the arrays of simulated GW levels


```python
hdf = pst.drop_observations(os.path.join(t_d,"heads.csv.ins"),pst_path=".")

#sdf = pst.drop_observations(os.path.join(t_d,"sfr.csv.ins"),pst_path=".")
```


```python
#[i for i in pst.model_output_data.model_file if i.startswith('hdslay')]
```


```python
pst.model_output_data
```

Now for some really nasty hackery:  we are going to modify the remaining stress-period-based instruction files to only include one row of instructions (since we only have one stress period now):


```python
sfrdf = None
for ins_file in pst.model_output_data.pest_file:
    if ins_file.startswith("hdslay") and ins_file.endswith("_t1.txt.ins"):
        print('not dropping:', ins_file)
        continue
    elif ins_file.startswith("hdslay"):
        df = pst.drop_observations(os.path.join(t_d,ins_file),pst_path=".")
        print('dropping:',ins_file)
    else:
        lines = open(os.path.join(t_d,ins_file),'r').readlines()
        df = pst.drop_observations(os.path.join(t_d,ins_file),pst_path=".")
        if ins_file == "sfr.csv.ins":
            sfrdf = df
        with open(os.path.join(t_d,ins_file),'w') as f:
            for line in lines[:3]:
                f.write(line.replace("_totim:3652.5","").replace("_time:3652.5",""))
        pst.add_observations(os.path.join(t_d,ins_file),pst_path=".")
assert sfrdf is not None
```


```python
[i for i in pst.model_output_data.model_file if i.startswith('hdslay')]
```


```python
pst.model_output_data
```

### Assigning observation values and weights

Time to work out a mapping from the MF6 head obs data (that have the actual head observations and weight we want) to the array based GW level observations.  We will again use a special set of PESTPP-DA specific options to help us here.  Since the observed value of GW level and the weights change through time (e.g. across cycles) but we are recording the array-based GW level observations every cycle, we need a way to tell PESTPP-DA to use specific `obsval`s and `weight`s for a given cycle.  `da_observation_cycle_table` and `da_weight_cycle_table` to the rescue!


```python
hdf.loc[:,"k"] = hdf.usecol.apply(lambda x: int(x.split("-")[1]))
hdf.loc[:,"i"] = hdf.usecol.apply(lambda x: int(x.split("-")[2]))
hdf.loc[:,"j"] = hdf.usecol.apply(lambda x: int(x.split("-")[3]))
hdf.loc[:,"time"] = hdf.time.astype(float)
```


```python
sites = hdf.usecol.unique()
sites.sort()
sites
```

In this code bit, we will process each MF6 head obs record (which includes `obsval` and `weight` for each stress period at each L-R-C location) and align that with corresponding (L-R-C) array-based GW level observation.  Then just collate those records into obs and weight cycle table. Note: we only want to include sites that have at least one non-zero weighted observation.  easy as!


```python
pst.try_parse_name_metadata()
obs = pst.observation_data
hdsobs = obs.loc[obs.obsnme.str.contains("hdslay"),:].copy()
hdsobs.loc[:,"i"] = hdsobs.i.astype(int)
hdsobs.loc[:,"j"] = hdsobs.j.astype(int)
hdsobs.loc[:,"k"] = hdsobs.oname.apply(lambda x: int(x[-1])-1)
odata = {}
wdata = {}
alldata = {}
for site in sites:
    sdf = hdf.loc[hdf.usecol==site,:].copy()
    #print(sdf.weight)
    
    sdf.sort_values(by="time",inplace=True)
    k,i,j = sdf.k.iloc[0],sdf.i.iloc[0],sdf.j.iloc[0]
    hds = hdsobs.loc[hdsobs.apply(lambda x: x.i==i and x.j==j and x.k==k,axis=1),:].copy()
    #assert hds.shape[0] == 1,site
    obname = hds.obsnme.iloc[0]
    print(obname)
    alldata[obname] = sdf.obsval.values
    if sdf.weight.sum() == 0:
        continue
    odata[obname] = sdf.obsval.values
    wdata[obname] = sdf.weight.values
    #print(site)
        
```

Same for the SFR "gage-1" observations


```python
sfrobs = obs.loc[obs.obsnme.str.contains("oname:sfr"),:].copy()
sites = sfrdf.usecol.unique()
sites.sort()
sites
```


```python
for site in sites:
    sdf = sfrdf.loc[sfrdf.usecol==site,:].copy()
    sdf.loc[:,"time"] = sdf.time.astype(float)
    
    sdf.sort_values(by="time",inplace=True)
    sfr = sfrobs.loc[sfrobs.usecol==site,:].copy()
    assert sfr.shape[0] == 1,sfr
    alldata[sfr.obsnme.iloc[0]] = sdf.obsval.values
    if sdf.weight.sum() == 0:
        continue
    odata[sfr.obsnme.iloc[0]] = sdf.obsval.values
    wdata[sfr.obsnme.iloc[0]] = sdf.weight.values
        
```


```python
odata
```

### Buidling the observation and weight cycle tables

Since we have observations at the same spatial locations across cycles, but we have only one "observation" (and there for `obsval` and `weight`) for that location in the control file.  So we can use the pestpp-da specific options: the observation and weight cycle table.

Form the obs cycle table as a dataframe:


```python
df = pd.DataFrame(odata)
df.index.name = "cycle"
df
```


```python
df.T.to_csv(os.path.join(t_d,"obs_cycle_table.csv"))
pst.pestpp_options["da_observation_cycle_table"] = "obs_cycle_table.csv"
```

Prep for the weight cycle table also.  As a safety check, PESTPP-DA requires any observation quantity that ever has a non-zero weight for any cycle to have a non-zero weight in `* observation data` (this weight value is not used, its more of just a flag).


```python
obs = pst.observation_data
obs.loc[:,"weight"] = 0
obs.loc[:,"cycle"] = -1
df = pd.DataFrame(wdata)
df.index.name = "cycle"
wsum = df.sum()
wsum = wsum.loc[wsum>0]
print(wsum)
obs.loc[wsum.index,"weight"] = 1.0

df.T.to_csv(os.path.join(t_d,"weight_cycle_table.csv"))
pst.pestpp_options["da_weight_cycle_table"] = "weight_cycle_table.csv"
df
```

Nothing to see here...let's save `alldata` to help us plot the results of PESTPP-DA later WRT forecasts and un-assimilated observations


```python
df = pd.DataFrame(alldata)
df.index.name = "cycle"
df.to_csv(os.path.join(t_d,"alldata.csv"))
```

### The state mapping between pars and obs

Ok, now for our next trick...

We need to tell PESTPP-DA that we want to use dynamic states.  This is tricky concept for us "batch" people, but conceptually, these states allow PESTPP-DA to coherently advance the model in time.  Just like MF6 would take the final simulated GW levels at the end of stress period and set them as the starting heads for the next stress, so too must PESTPP-DA. Otherwise, there would be no temporal coherence in the simulated results.  What is exciting about this is that PESTPP-DA also has the opportunity to "estimate" the start heads for each cycle, along with the other parameters.  Algorithmically, PESTPP-DA sees these "states" just as any other parameter to estimate for a given cycle.  Conceptually, treating the initial states for each cycle as uncertain and therefore adjustable, is one way to explicitly acknowledge that the model is "imperfect" and therefore the initial conditions for each cycle are "imperfect" e.g. uncertain!  How cool!

The way we tell PESTPP-DA about the dynamic state linkage between observations and parameters is by either giving the parameters and observations identical names, or by adding a column to the `* observation data` dataframe that names the parameter that the observation links to.  We will do the latter here - this column must be named "state_par_link":


```python
obs = pst.observation_data
obs.loc[:,"state_par_link"] = ""
hdsobs = obs.loc[obs.obsnme.str.contains("hdslay"),:].copy()
hdsobs.loc[:,"i"] = hdsobs.i.astype(int)
hdsobs.loc[:,"j"] = hdsobs.j.astype(int)
hdsobs.loc[:,"k"] = hdsobs.oname.apply(lambda x: int(x[-1])-1)
hdsobs.loc[:,"kij"] = hdsobs.apply(lambda x: (x.k,x.i,x.j),axis=1)
```


```python
par = pst.parameter_data
strtpar = par.loc[par.parnme.str.contains("strt"),:].copy()
strtpar.loc[:,"i"] = strtpar.i.astype(int)
strtpar.loc[:,"j"] = strtpar.j.astype(int)
strtpar.loc[:,"k"] = strtpar.pname.apply(lambda x: int(x[-1])-1)
strtpar.loc[:,"kij"] = strtpar.apply(lambda x: (x.k,x.i,x.j),axis=1)
spl = {kij:name for kij,name in zip(strtpar.kij,strtpar.parnme)}
```


```python
obs.loc[hdsobs.obsnme,"state_par_link"] = hdsobs.kij.apply(lambda x: spl.get(x,""))
```


```python
obs.loc[hdsobs.obsnme,:]
```

One last thing: we need to modify the multiplier-parameter process since we now have a single-stress-period model.  This is required if you are using `PstFrom`:


```python
df = pd.read_csv(os.path.join(t_d,"mult2model_info.csv"),index_col=0)
ifiles = set(pst.model_input_data.model_file.tolist())
#print(df.mlt_file.unique())
new_df = df.loc[df.mlt_file.apply(lambda x: pd.isna(x) or x in ifiles),:]
#new_df.shape,df.shape
#new_df.to_csv(os.path.join(t_d,"mult2model_info.csv"))
new_df
```


```python
df.loc[:,"cycle"] = -1
```


```python
sfr = df.loc[df.model_file.str.contains("sfr_perioddata"),:].copy()
df.loc[sfr.index,"cycle"] = sfr.model_file.apply(lambda x: int(x.split("_")[-1].split(".")[0])-1)
df.loc[sfr.index.values[1:],"model_file"] = sfr.model_file.iloc[0]
df.loc[sfr.index]

```


```python
rch = df.loc[df.model_file.str.contains("rch"),:]
df.loc[rch.index,"cycle"] = rch.model_file.apply(lambda x: int(x.split('_')[-1].split(".")[0])-1)
df.loc[rch.index.values[1:],"model_file"] = rch.model_file.iloc[0]
```


```python
wel = df.loc[df.model_file.str.contains("wel"),:].copy()
df.loc[wel.index,"cycle"] = wel.model_file.apply(lambda x: int(x.split('_')[-1].split(".")[0])-1)
df.loc[wel.index.values[1:],"model_file"] = wel.model_file.iloc[0]
```


```python
df.loc[wel.index]
```


```python
df.loc[df.cycle!=-1,["org_file","model_file","cycle"]]
```


```python
df.to_csv(os.path.join(t_d,"mult2model_info.global.csv"))
```


```python
shutil.copy2("prep_mult.py",os.path.join(t_d,"prep_mult.py"))
```


```python
lines = open(os.path.join(t_d,"forward_run.py"),'r').readlines()
with open(os.path.join(t_d,"forward_run.py"),'w') as f:
    for line in lines:
        if "apply_list_and_array_pars" in line:
            f.write("    pyemu.os_utils.run('python prep_mult.py')\n")
        f.write(line)
```

# OMG that was brutal


```python
pst.pestpp_options.pop("forecasts",None)
pst.control_data.noptmax = 0
pst.write(os.path.join(t_d,"freyberg_mf6.pst"),version=2)
pyemu.os_utils.run("pestpp-da freyberg_mf6.pst",cwd=t_d)
```

Wow, that takes a lot longer...this is the price of sequential estimation...


```python
files = [f for f in os.listdir(t_d) if ".base.obs.csv" in f]
files.sort()
print(files)
pr_oes = {int(f.split(".")[1]):pd.read_csv(os.path.join(t_d,f),index_col=0) for f in files[:-1]}

```


```python
otab = pd.read_csv(os.path.join(t_d,"obs_cycle_table.csv"),index_col=0)
wtab = pd.read_csv(os.path.join(t_d,"weight_cycle_table.csv"),index_col=0)
ad_df = pd.read_csv(os.path.join(t_d,"alldata.csv"),index_col=0)
```


```python
obs = pst.observation_data
nzobs = obs.loc[pst.nnz_obs_names,:]
nzobs
```


```python

```


```python
for o in pst.nnz_obs_names:
    fig,axes = plt.subplots(1,1,figsize=(10,3))
    axes=[axes]
    for kper,oe in pr_oes.items():
        axes[0].scatter([kper]*oe.shape[0],oe.loc[:,o].values,marker=".",c="0.5",alpha=0.5)

    ovals = otab.loc[o,:].values
    wvals = wtab.loc[o,:].values
    ylim = axes[0].get_ylim()

 
    xlim = axes[0].get_xlim()

    ovals[wvals==0] = np.nan
    axes[0].scatter(otab.columns.values,ovals,marker='^',c='r')
    oval_lim = (np.nanmin(ovals),np.nanmax(ovals))
    d = [ylim, (np.nanmin(ovals),np.nanmax(ovals))]
    ylim =  min(d, key = lambda t: t[1])[0], max(d, key = lambda t: t[1])[-1]
    axes[0].set_ylim(ylim)
    axes[0].set_xlim(xlim)
    axes[0].set_title("A) prior only: "+o,loc="left")
    axes[0].set_xlabel("kper")

    
    avals = ad_df.loc[:,o]
    axes[0].scatter(ad_df.index.values,avals,marker='.',c='r')

    plt.tight_layout()
```
