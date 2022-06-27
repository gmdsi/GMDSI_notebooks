---
layout: default
title: Intro to Freyberg
parent: Introductions to Selected Topics
nav_order: 1
math: mathjax3
---

# Intro to the model

## Freyberg (1988) - "An Exercise in Ground-Water Model Calibration and Prediction" 


> "*The goal of parameter identification is rarely the parameter estimates. Rather, the ultimate goal is nearly always a prediction .*"
>-David Freyberg (1988)


The following series of tutorials make use of a synthetic model. This model is a variant of the model originally created by David Freyberg at Stanford University in the late 1980s. David Freyberg designed a simple model to give to a graduate class and asked them each to "calibrate" the model. Students were provided with an extensive amount of data:

1. the water level (perfectly represented) in a number of wells
2. the bedrock elevation at those same well locations
3. all the lateral aquifer geometry
4. boundary conditions including lateral flows 
5. well pumping rates 

The forecast of interest was the head if the river channel was lined (e.g. conductance reduced greatly).

There are interesting insights in the paper, but perhaps the most interesting is illustrated by the figure below: just because a model is good at fitting measurement data, does not mean it is good at making a prediction!

<img src=".\intro_to_freyberg_model_files\cal_pred.png" style="float: center; width: 75%;  margin-bottom: 0.5em;">


You can read the original paper here:

> *Freyberg, David L. 1988. “AN EXERCISE IN GROUND-WATER MODEL CALIBRATION AND PREDICTION.” Ground Water 26 (3): 350–60. doi:10.1111/j.1745-6584.1988.tb00399.x.*

And more recently, the same exercise was revisited in a contemporary context:

> *Hunt, Randall J., Michael N. Fienen, and Jeremy T. White. 2019. “Revisiting ‘An Exercise in Groundwater Model Calibration and Prediction’ After 30 Years: Insights and New Directions.” Groundwater, July, gwat.12907. doi:10.1111/gwat.12907.* 
   

## The Modified-Freyberg Model

Using a synthetic model allows us to know the "truth". It also allows us to design it to be fast-running. Both usefull characteristics for a tutorial model. For the current set of tutorials we will be using a variant of the Freyberg model. This is similar to the model described in the PEST++ documentation:
> White, J.T., Hunt, R.J., Fienen, M.N., and Doherty, J.E., 2020, Approaches to Highly Parameterized > Inversion: PEST++ Version 5, a Software Suite for Parameter Estimation, Uncertainty Analysis, Management > Optimization and Sensitivity Analysis: U.S. Geological Survey Techniques and Methods 7C26, 52 p., https://> doi.org/10.3133/tm7C26.

Some of the parameterisation and selected observation data are different. We also include additional particle tracking simulated using MODPATH7. 

Let's get acquainted with it.

### Admin
First some admin. You will see this in most of the tutorial notebooks. Load the dependencies and organize model folders. Let's copy the original model folder into a new working directory, just to ensure we don't mess up the base files. Simply run the next cells by pressing `shift+enter`.


```python
import os
import shutil
import platform
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.join("..", "..", "dependencies"))
import flopy
```


```python
# folder containing original model files
org_ws = os.path.join('..', '..', 'models', 'freyberg_mf6')

# set a new workspace folder to avoid breaking things by mistake
sim_ws = os.path.join('freyberg_mf6')

# remove existing folder
if os.path.exists(sim_ws):
    shutil.rmtree(sim_ws)

# copy the original model folder across
shutil.copytree(org_ws, sim_ws)

# get the necessary executables; OS agnostic
bin_dir = os.path.join('..','..','bin')
exe_files=['mf6', 'mp7']

for exe_file in exe_files:
    if "window" in platform.platform().lower():
        exe_file = exe_file+'.exe'
    shutil.copy2(os.path.join(bin_dir, exe_file), os.path.join(sim_ws,exe_file))

```

Load and run the simulation. 

It should take less than a second. (If only all models were so fast!) As you can see, the model is fast and numerically stable. When undertaking highly-parameterized inversion, a model will be simulated many, many times; and run-times add up quickly! A modeller needs to take this factor into account during model design.


```python
# load simulation
sim = flopy.mf6.MFSimulation.load(sim_ws=sim_ws, verbosity_level=0)

# load flow model
gwf = sim.get_model()

# run the model
sim.run_simulation()
```




    (True, [])



### Model Domain, BCs and Properties

The figure belows shows the model domain and boundary conditions. The model has 3 layers, 40 rows and 20 columns. Cell dimensions are 250m x 250m. There are inactive outcrop areas within the model domain (shown in black in the figure).


```python
dis = gwf.dis
print(f'layers:{dis.nlay.get_data()} nrows:{dis.nrow.get_data()} columns:{dis.ncol.get_data()}')
```

    layers:3 nrows:40 columns:20
    

There is a GHB along the southern boundary in all layers. All other external boundaries are no-flow. 

The surface-water system consists of a straight stream flowing north to south, which is simulated using the Streamflow Routing (SFR) package. SFR reaches raverse the model domain from row 1 to row 40 in column 16. Surface-water flow observations are monitored in reach 40 (the terminal reach). 

Six groundwater extraction wells are placed in the bottom layer 3. There are several monitoring wells screened in layer 1 and layer 3. 

Water enters the model domain as recharge and stream leakage in layer 1. It leaves through groundwater discharge to the surface-water, groundwater extraction and through the downgradient GHB. 


```python
# plot
fig = plt.figure(figsize=(7, 7))

ax = fig.add_subplot(1, 1, 1, aspect='equal')
mm = flopy.plot.PlotMapView(model=gwf, ax=ax, layer=0)

mm.plot_grid()
mm.plot_inactive()
# Plot grid 
# you can plot BC cells using the plot_bc() 
mm.plot_bc('ghb')
mm.plot_bc('sfr')

# Plot wells in layer 3
mm = flopy.plot.PlotMapView(model=gwf, ax=ax, layer=2)
mm.plot_bc('wel');
```


    
![png](freyberg_intro_model_files/freyberg_intro_model_10_0.png)
    


Take a quick look at everyone's favourite parameter, hydraulic conductivity (K). Values in the plot below show K in layer 1. If you check each layer (by changing the mflay value), you will see different initial values in each. 

Layer 3 is the most permeable. Layer 2 has low permeabilities. Layer 1 is somewhere in between. 


```python
for lay in range(gwf.dis.nlay.array):
    print(f'K in layer {lay+1}:', "{:10.1f}".format(gwf.npf.k.get_data(lay).mean()))
```

    K in layer 1:        3.0
    K in layer 2:        0.3
    K in layer 3:       30.0
    


```python
gwf.npf.k.plot(colorbar=True, mflay=0);
```


    
![png](freyberg_intro_model_files/freyberg_intro_model_13_0.png)
    


Surface topography and the bottom elevation are not uniform (see plots below). The middle layer has a constant thickness of 2.5m, with a top and bottom elevation of 32.5m and 30m, respectively.


```python
# plot model top
gwf.dis.top.plot(colorbar=True, masked_values=[-1049.99])

# plot bottom of bottom layer
gwf.dis.botm.plot(colorbar=True, mflay=2);
```


    
![png](freyberg_intro_model_files/freyberg_intro_model_15_0.png)
    



    
![png](freyberg_intro_model_files/freyberg_intro_model_15_1.png)
    


### Time Discretisation

The model simulates 25 stress-periods: 1 steady-state, followed by 24 transient stress periods. 

Conceptualy, the first 12 transient stress periods represent the "historical" conditions. Simulated outputs from this period (using the "true" parameter field) are used as "observations" for history matching. These represent field measurments in from our fictional site.

The last 12 transient stress periods conceptualy represent the unmeasured, future condition. The period for which predictions are required. Selected model outputs simulated during this period form a set of "forecasts" or "predicitons" of management interest. 

## Observation Data

The following field data are available as "observations" for the purposes of history matching:
 - surface-water flow at the terminal reach (stress period 2 to 13);
 - groundwater levels at two sites (stress period 2 to 13); 

"Measured" data was generated using the same model, but with higher spatial and temporal resolution (i.e. a finer grid and daily stress periods) and randomly generated parameters.


```python
obs_data = pd.read_csv(os.path.join('..','..','models','daily_freyberg_mf6_truth','obs_data.csv'))
obs_data = obs_data.pivot(index='time', columns='site', values='value')
obs_data.head()
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
      <th>site</th>
      <th>GAGE-1</th>
      <th>TRGW-0-26-6</th>
      <th>TRGW-0-3-8</th>
      <th>TRGW-2-26-6</th>
      <th>TRGW-2-3-8</th>
    </tr>
    <tr>
      <th>time</th>
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
      <td>1713.876690</td>
      <td>34.780818</td>
      <td>34.621020</td>
      <td>34.382895</td>
      <td>34.612454</td>
    </tr>
    <tr>
      <th>3653.5</th>
      <td>2476.327373</td>
      <td>34.588687</td>
      <td>34.477117</td>
      <td>34.627812</td>
      <td>34.563783</td>
    </tr>
    <tr>
      <th>3654.5</th>
      <td>1505.541231</td>
      <td>34.505069</td>
      <td>34.554786</td>
      <td>34.573804</td>
      <td>34.420837</td>
    </tr>
    <tr>
      <th>3655.5</th>
      <td>2210.090691</td>
      <td>34.770207</td>
      <td>34.603493</td>
      <td>34.510403</td>
      <td>34.656066</td>
    </tr>
    <tr>
      <th>3656.5</th>
      <td>2207.018734</td>
      <td>34.724382</td>
      <td>34.623075</td>
      <td>34.626356</td>
      <td>34.584529</td>
    </tr>
  </tbody>
</table>
</div>



Model simulated counterparts to these observations are recorded in external CSV files. Let's take a look:


```python
# check the output csv file names
for i in gwf.obs:
    print(i.output.obs_names)
```

    ['sfr.csv']
    ['heads.csv']
    

We can read the `sfr.csv` output file, and inspect the values:


```python
sfr_obs = pd.read_csv(os.path.join(sim_ws, 'sfr.csv'))
sfr_obs.head()
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
      <th>time</th>
      <th>HEADWATER</th>
      <th>TAILWATER</th>
      <th>GAGE-1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3652.5</td>
      <td>-751.85</td>
      <td>-531.46</td>
      <td>1365.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3683.5</td>
      <td>-952.74</td>
      <td>-674.01</td>
      <td>1759.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3712.5</td>
      <td>-1109.70</td>
      <td>-798.24</td>
      <td>2046.9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3743.5</td>
      <td>-1181.90</td>
      <td>-854.51</td>
      <td>2165.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3773.5</td>
      <td>-1138.00</td>
      <td>-821.46</td>
      <td>2064.4</td>
    </tr>
  </tbody>
</table>
</div>



Simulated values for surface-water flow at the terminal reach are recorded in the "GAGE1" column:


```python
site = 'GAGE-1'
fig,ax=plt.subplots(1,1)
obs_data.plot(y=site, ax=ax, label='measured')
sfr_obs.plot(x='time', y=site, ax=ax, label='modelled')
ax.set_title(site);
```


    
![png](freyberg_intro_model_files/freyberg_intro_model_24_0.png)
    


Simulated groundwater levels are recorded in the "heads.csv" file. Several monitoring sites are simulated, however there is measured data for a only a few of these. 

The sites for which "measured data" are available are named:
 - TRGW-0-26-6
 - TRGW-2-26-6
 - TRGW-0-3-8
 - TRGW-2-3-8

The site naming convention is: "TRGW-layer-row-column". Thus, the four sites listed above pertain to two observation locations, with measurments from both the top and bottom layers. 


```python
hds_obs = pd.read_csv(os.path.join(sim_ws, 'heads.csv'))
hds_obs.head()
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
      <th>time</th>
      <th>TRGW-2-2-15</th>
      <th>TRGW-2-2-9</th>
      <th>TRGW-2-3-8</th>
      <th>TRGW-2-9-1</th>
      <th>TRGW-2-13-10</th>
      <th>TRGW-2-15-16</th>
      <th>TRGW-2-21-10</th>
      <th>TRGW-2-22-15</th>
      <th>TRGW-2-24-4</th>
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
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3652.5</td>
      <td>34.399290</td>
      <td>34.691525</td>
      <td>34.728569</td>
      <td>35.061060</td>
      <td>34.320957</td>
      <td>34.180851</td>
      <td>34.200938</td>
      <td>34.066219</td>
      <td>34.405733</td>
      <td>...</td>
      <td>35.068338</td>
      <td>34.326896</td>
      <td>34.186174</td>
      <td>34.206846</td>
      <td>34.025747</td>
      <td>34.412096</td>
      <td>34.251506</td>
      <td>33.937832</td>
      <td>34.030710</td>
      <td>33.923445</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3683.5</td>
      <td>34.473764</td>
      <td>34.772123</td>
      <td>34.809126</td>
      <td>35.126827</td>
      <td>34.434959</td>
      <td>34.283819</td>
      <td>34.319984</td>
      <td>34.171011</td>
      <td>34.515566</td>
      <td>...</td>
      <td>35.133817</td>
      <td>34.440124</td>
      <td>34.288730</td>
      <td>34.324980</td>
      <td>34.115906</td>
      <td>34.520964</td>
      <td>34.379532</td>
      <td>34.016423</td>
      <td>34.141237</td>
      <td>34.034923</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3712.5</td>
      <td>34.539742</td>
      <td>34.855185</td>
      <td>34.893566</td>
      <td>35.211003</td>
      <td>34.527661</td>
      <td>34.363167</td>
      <td>34.417107</td>
      <td>34.247456</td>
      <td>34.620546</td>
      <td>...</td>
      <td>35.217810</td>
      <td>34.533812</td>
      <td>34.369042</td>
      <td>34.423098</td>
      <td>34.182401</td>
      <td>34.626410</td>
      <td>34.485296</td>
      <td>34.078743</td>
      <td>34.229400</td>
      <td>34.111873</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3743.5</td>
      <td>34.577505</td>
      <td>34.912040</td>
      <td>34.952401</td>
      <td>35.282335</td>
      <td>34.573685</td>
      <td>34.398859</td>
      <td>34.465330</td>
      <td>34.280521</td>
      <td>34.686225</td>
      <td>...</td>
      <td>35.289225</td>
      <td>34.580980</td>
      <td>34.405657</td>
      <td>34.472538</td>
      <td>34.211387</td>
      <td>34.693043</td>
      <td>34.541211</td>
      <td>34.107372</td>
      <td>34.271897</td>
      <td>34.142187</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3773.5</td>
      <td>34.568882</td>
      <td>34.913084</td>
      <td>34.954807</td>
      <td>35.301600</td>
      <td>34.552253</td>
      <td>34.376324</td>
      <td>34.442806</td>
      <td>34.256405</td>
      <td>34.679251</td>
      <td>...</td>
      <td>35.308694</td>
      <td>34.560270</td>
      <td>34.383636</td>
      <td>34.450839</td>
      <td>34.190749</td>
      <td>34.686945</td>
      <td>34.520264</td>
      <td>34.090823</td>
      <td>34.249617</td>
      <td>34.113480</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>



Let's make a quick plot of time series of simulated groundwater levels from both layers at a single location:


```python
for site in ['TRGW-0-26-6','TRGW-2-26-6']:
    fig,ax=plt.subplots(1,1)
    obs_data.plot(y=site, ax=ax, label='measured')
    hds_obs.plot(x='time', y=site, ax=ax, label='modelled')
    ax.set_title(site);
```


    
![png](freyberg_intro_model_files/freyberg_intro_model_28_0.png)
    



    
![png](freyberg_intro_model_files/freyberg_intro_model_28_1.png)
    


Whilst we are at it, lets just make a plot of the spatial distribution of simulated heads in the upper layer. (Heads in the other layers are very similar.)


```python
hdobj = gwf.output.head()
times = hdobj.get_times()
hdobj.plot(mflay=0, colorbar=True, totim=times[-1], masked_values=[1e30]);
```


    
![png](freyberg_intro_model_files/freyberg_intro_model_30_0.png)
    


## Forecasts

Three model simulated outputs are included as forecast "observations". These represent predictions of management interest for our imaginary case. Simulated forecasts are:
 - aggregated surface-water/grounwdater exchange for reaches 1-20 (recorded under "headwater" in the sfr.csv file) during stress period 22;
 - aggregated surface-water/grounwdater exchange for reaches 21-40 (recorded under "tailwater" in the sfr.csv file) during stress period 22;
 - groundwater level at TRGW_0_9_1 at the end of stress period 22;
 - travel time for a particle released in the north-west of the domain to exit the model domain.

These forecasts were selected to represent model outputs that are informed in varying degrees by the history matching data. Throughout the tutorials we will compare our simulated forecast and uncertainty against "true" values obtained by running the same model with which we generated "measured" observation data. 
