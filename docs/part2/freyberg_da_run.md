---
layout: default
title: PEST++DA - Sequential Data Assimilation
parent: Decision Support Modelling with pyEMU and PEST++
nav_order: 11
math: mathjax3
---

# PESTPP-DA - Generalized Data Assimilation with Ensemble Methods
## Run the beast!

Much Like PESTPP-IES, PESTPP-DA uses ensembles to approximates first-order relationships between inputs (i.e. parameters) and outputs (i.e. observations).  However, PESTPP-DA extends and generalizes the approach of PESTPP-IES (and all other tools in the PEST and PEST++) to the concept of an assimilation "cycle".  Each cycle represents a discrete inverse problems with (potentially) unique parameters and observations.  Most commonly, these cycles represent discrete time intervals, for example, a year.  During each cycle, the parameters active during the cycle are conditioned by assimilating observations active during the cycle. This is referred to as "sequential estimation".  The real mind twister is that the posterior at the end of each cycle is the prior for the next cycle...confused yet?  Conceptually and theorically, Bayes equation can be split out this way...

The assimilation engine used in each cycle by PESTPP-DA is the same engine used in PESTPP-IES:  the iterative ensemble method, using either the GLM algorithm of Chen and Oliver or the multiple data assimilation algorithm of Emmerick and Reynolds.  

To implement the cycle-based assimilation, users must add a cycle number to parameters (and template files) and observations (and instruction files) in the pest control file.  At runtime, PESTPP-DA does the incredibly painful process of forming a new "sub-problem" using these cycle numbers under the hood.  You are welcome!

But there is something more...if PESTPP-DA takes control of the time advancement process from the underlying simulation model, how do we form a coherent temporal evolution.  This is where the concept of "states" becomes critical.  A "state" is simply a simulated "state" of the system - in groundwater flow modeling, states are the simulated groundwater levels in each active model cell.  In a standard "batch" parameter estimation analysis (where we run the entire historic period at once and let MODFLOW "do its thing"), MODFLOW itself advances the states from stress period to stress period.  That is, the final simulated (hopefully converged) groundwater levels for each active cell at the end of the stress period 1 become the initial heads for stress period 2.  Ok, cool, so how do we make this work with PESTPP-DA where we have defined each stress period to be a cycle?  Well we have to now include the simulated water level in each active cell as an "observation" in the control file and we need to also add the initial groundwater level (i.e. the `strt` quantity in the MODFLOW world) for each active cell as a "parameter" in the control file.  And then we also need to tell PESTPP-DA how each of these state observations and state parameters map to each other - that is, how the observed state in model cell in layer 1/row 1/column 1 maps to the initial state parameter that is in layer 1/row 1/column 1.  Just some book keeping...right?  In this case, since we are using each stress period as an assimilation cycle, we have also changed the underlying model so that it is just a single stress-period...

So at this point, maybe you are thinking "WTF - this is insane.  Who would ever want to use sequential estimation".  Well, it turns out, if you are interested in making short-term, so-called "one-step-ahead" forecasts, sequential estimation is the optimal approach.  And that is because, just like we estimate parameters for things like HK and SS, we can also estimate the initial condition parameters!  WAT?!  That's right - we can estimate the initial groundwater levels in each active model cell for each cycle along with the static properties.  This results in the model being especially tuned at making a forecast related to the system behavior in the near term - these near-term forecasts depend as much on having the system state optimal as they do on having the properties optimal (maybe even more). So for short-term/near-term forecasts, if you have the groundwater levels nearly right, then you are probably going to do pretty well for forecasting something happening in the near future.  The combined form of estimation is referred to as "joint state-parameter estimation".  

In this example problem, we are estimating static properties during all cycles, as well as some cycle-specific forcing parameters like recharge and extraction rates, plus the initial groundwater level states.  With regard to the static properties, like HK and SS, the sequential estimation problem implies that the optimal static properties values may, and do, change for each cycle!  Thats because what is optimal for cycle 1 in terms of HK and SS differs from what is optimal for cycle 2.  This may cause some of you to question the validity of sequential estimation.  But from a Bayesian perspetive, its perfectly valid, and from the stand point of improved forecasting skill, its optimal.  



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

    noptmax:2, npar_adj:23786, nnz_obs:3
    


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

    ['freyberg_mf6.0.0.obs.csv', 'freyberg_mf6.1.0.obs.csv', 'freyberg_mf6.10.0.obs.csv', 'freyberg_mf6.11.0.obs.csv', 'freyberg_mf6.12.0.obs.csv', 'freyberg_mf6.13.0.obs.csv', 'freyberg_mf6.14.0.obs.csv', 'freyberg_mf6.15.0.obs.csv', 'freyberg_mf6.16.0.obs.csv', 'freyberg_mf6.17.0.obs.csv', 'freyberg_mf6.18.0.obs.csv', 'freyberg_mf6.19.0.obs.csv', 'freyberg_mf6.2.0.obs.csv', 'freyberg_mf6.20.0.obs.csv', 'freyberg_mf6.21.0.obs.csv', 'freyberg_mf6.22.0.obs.csv', 'freyberg_mf6.23.0.obs.csv', 'freyberg_mf6.24.0.obs.csv', 'freyberg_mf6.3.0.obs.csv', 'freyberg_mf6.4.0.obs.csv', 'freyberg_mf6.5.0.obs.csv', 'freyberg_mf6.6.0.obs.csv', 'freyberg_mf6.7.0.obs.csv', 'freyberg_mf6.8.0.obs.csv', 'freyberg_mf6.9.0.obs.csv']
    


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
      <th>observed</th>
      <th>cycle</th>
      <th>state_par_link</th>
    </tr>
    <tr>
      <th>obsnme</th>
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
      <th>oname:hdslay1_t1_otype:arr_i:26_j:6</th>
      <td>oname:hdslay1_t1_otype:arr_i:26_j:6</td>
      <td>35.06628</td>
      <td>1.0</td>
      <td>hdslay1_t1</td>
      <td>hdslay1</td>
      <td>arr</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>26</td>
      <td>6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1</td>
      <td>pname:icstrtlayer1_inst:0_ptype:gr_pstyle:d_i:26_j:6_x:1625.00_y:3375.00_zone:1</td>
    </tr>
    <tr>
      <th>oname:hdslay1_t1_otype:arr_i:3_j:8</th>
      <td>oname:hdslay1_t1_otype:arr_i:3_j:8</td>
      <td>35.71549</td>
      <td>1.0</td>
      <td>hdslay1_t1</td>
      <td>hdslay1</td>
      <td>arr</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3</td>
      <td>8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1</td>
      <td>pname:icstrtlayer1_inst:0_ptype:gr_pstyle:d_i:3_j:8_x:2125.00_y:9125.00_zone:1</td>
    </tr>
    <tr>
      <th>oname:sfr_otype:lst_usecol:gage-1</th>
      <td>oname:sfr_otype:lst_usecol:gage-1</td>
      <td>4065.54321</td>
      <td>1.0</td>
      <td>obgnme</td>
      <td>sfr</td>
      <td>lst</td>
      <td>gage-1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



This file was made during the PESTPP-DA prep process - it contains all of the observation values.  Its just to help with plotting here...


```python
ad_df = pd.read_csv(os.path.join(t_d,"alldata.csv"),index_col=0)
ad_df
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
      <th>oname:hdslay1_t1_otype:arr_i:13_j:10</th>
      <th>oname:hdslay1_t1_otype:arr_i:15_j:16</th>
      <th>oname:hdslay1_t1_otype:arr_i:2_j:15</th>
      <th>oname:hdslay1_t1_otype:arr_i:2_j:9</th>
      <th>oname:hdslay1_t1_otype:arr_i:21_j:10</th>
      <th>oname:hdslay1_t1_otype:arr_i:22_j:15</th>
      <th>oname:hdslay1_t1_otype:arr_i:24_j:4</th>
      <th>oname:hdslay1_t1_otype:arr_i:26_j:6</th>
      <th>oname:hdslay1_t1_otype:arr_i:29_j:15</th>
      <th>oname:hdslay1_t1_otype:arr_i:3_j:8</th>
      <th>oname:hdslay1_t1_otype:arr_i:33_j:7</th>
      <th>oname:hdslay1_t1_otype:arr_i:34_j:10</th>
      <th>oname:hdslay1_t1_otype:arr_i:9_j:1</th>
      <th>oname:sfr_otype:lst_usecol:gage-1</th>
      <th>oname:sfr_otype:lst_usecol:headwater</th>
      <th>oname:sfr_otype:lst_usecol:tailwater</th>
    </tr>
    <tr>
      <th>cycle</th>
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
      <th>0</th>
      <td>34.720563</td>
      <td>34.633412</td>
      <td>34.623412</td>
      <td>35.013130</td>
      <td>34.640962</td>
      <td>34.498465</td>
      <td>35.157157</td>
      <td>34.852666</td>
      <td>34.423990</td>
      <td>35.056132</td>
      <td>34.619055</td>
      <td>34.547056</td>
      <td>35.445192</td>
      <td>3809.880113</td>
      <td>-1199.387892</td>
      <td>-2111.037057</td>
    </tr>
    <tr>
      <th>1</th>
      <td>34.722037</td>
      <td>34.638673</td>
      <td>34.630134</td>
      <td>35.020444</td>
      <td>34.647106</td>
      <td>34.504557</td>
      <td>35.165706</td>
      <td>34.795085</td>
      <td>34.432867</td>
      <td>35.047742</td>
      <td>34.625792</td>
      <td>34.551285</td>
      <td>35.450551</td>
      <td>3867.951586</td>
      <td>-1204.609384</td>
      <td>-2124.825850</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34.714429</td>
      <td>34.630976</td>
      <td>34.632897</td>
      <td>35.030673</td>
      <td>34.639387</td>
      <td>34.498111</td>
      <td>35.169473</td>
      <td>34.768398</td>
      <td>34.433925</td>
      <td>35.069538</td>
      <td>34.625897</td>
      <td>34.548921</td>
      <td>35.471283</td>
      <td>3757.724334</td>
      <td>-1193.483236</td>
      <td>-2118.685728</td>
    </tr>
    <tr>
      <th>3</th>
      <td>34.782017</td>
      <td>34.677580</td>
      <td>34.680302</td>
      <td>35.084482</td>
      <td>34.690776</td>
      <td>34.543350</td>
      <td>35.219354</td>
      <td>34.896937</td>
      <td>34.476912</td>
      <td>35.129611</td>
      <td>34.689569</td>
      <td>34.608954</td>
      <td>35.540237</td>
      <td>3988.281927</td>
      <td>-1285.172211</td>
      <td>-2241.788169</td>
    </tr>
    <tr>
      <th>4</th>
      <td>34.841318</td>
      <td>34.731719</td>
      <td>34.728793</td>
      <td>35.156624</td>
      <td>34.743566</td>
      <td>34.589426</td>
      <td>35.286827</td>
      <td>34.979874</td>
      <td>34.518895</td>
      <td>35.164669</td>
      <td>34.744762</td>
      <td>34.655012</td>
      <td>35.626434</td>
      <td>4135.608082</td>
      <td>-1372.204659</td>
      <td>-2364.340674</td>
    </tr>
    <tr>
      <th>5</th>
      <td>34.881352</td>
      <td>34.763381</td>
      <td>34.759789</td>
      <td>35.222994</td>
      <td>34.778145</td>
      <td>34.617693</td>
      <td>35.342980</td>
      <td>34.957829</td>
      <td>34.546224</td>
      <td>35.239166</td>
      <td>34.783479</td>
      <td>34.688770</td>
      <td>35.717698</td>
      <td>4324.840909</td>
      <td>-1426.133447</td>
      <td>-2444.304401</td>
    </tr>
    <tr>
      <th>6</th>
      <td>34.867546</td>
      <td>34.756265</td>
      <td>34.755255</td>
      <td>35.246785</td>
      <td>34.772147</td>
      <td>34.610456</td>
      <td>35.365945</td>
      <td>35.019307</td>
      <td>34.539875</td>
      <td>35.298738</td>
      <td>34.776164</td>
      <td>34.679452</td>
      <td>35.767874</td>
      <td>4321.169528</td>
      <td>-1413.691590</td>
      <td>-2425.631131</td>
    </tr>
    <tr>
      <th>7</th>
      <td>34.813972</td>
      <td>34.715593</td>
      <td>34.720277</td>
      <td>35.225713</td>
      <td>34.724995</td>
      <td>34.568851</td>
      <td>35.337949</td>
      <td>34.935632</td>
      <td>34.497482</td>
      <td>35.264201</td>
      <td>34.716097</td>
      <td>34.622441</td>
      <td>35.766159</td>
      <td>4124.844324</td>
      <td>-1342.689753</td>
      <td>-2304.539075</td>
    </tr>
    <tr>
      <th>8</th>
      <td>34.678659</td>
      <td>34.585434</td>
      <td>34.621217</td>
      <td>35.127060</td>
      <td>34.590068</td>
      <td>34.449349</td>
      <td>35.208911</td>
      <td>34.767495</td>
      <td>34.385429</td>
      <td>35.171774</td>
      <td>34.579590</td>
      <td>34.493716</td>
      <td>35.675578</td>
      <td>3660.938812</td>
      <td>-1138.454075</td>
      <td>-1989.103860</td>
    </tr>
    <tr>
      <th>9</th>
      <td>34.502779</td>
      <td>34.405765</td>
      <td>34.489830</td>
      <td>34.981513</td>
      <td>34.411655</td>
      <td>34.299358</td>
      <td>34.954871</td>
      <td>34.596834</td>
      <td>34.242937</td>
      <td>35.010369</td>
      <td>34.408483</td>
      <td>34.330603</td>
      <td>35.516042</td>
      <td>2991.023913</td>
      <td>-872.264316</td>
      <td>-1597.403848</td>
    </tr>
    <tr>
      <th>10</th>
      <td>34.312501</td>
      <td>34.222510</td>
      <td>34.351600</td>
      <td>34.814063</td>
      <td>34.218187</td>
      <td>34.138023</td>
      <td>34.657407</td>
      <td>34.346105</td>
      <td>34.096046</td>
      <td>34.797311</td>
      <td>34.225203</td>
      <td>34.160159</td>
      <td>35.310812</td>
      <td>2327.725610</td>
      <td>-592.795345</td>
      <td>-1193.652178</td>
    </tr>
    <tr>
      <th>11</th>
      <td>34.171973</td>
      <td>34.102068</td>
      <td>34.251466</td>
      <td>34.660221</td>
      <td>34.088571</td>
      <td>34.028009</td>
      <td>34.458634</td>
      <td>34.206008</td>
      <td>33.997107</td>
      <td>34.660226</td>
      <td>34.094958</td>
      <td>34.048192</td>
      <td>35.101419</td>
      <td>1854.845135</td>
      <td>-408.757143</td>
      <td>-925.219092</td>
    </tr>
    <tr>
      <th>12</th>
      <td>34.125174</td>
      <td>34.079343</td>
      <td>34.213698</td>
      <td>34.547508</td>
      <td>34.048163</td>
      <td>33.995011</td>
      <td>24.319226</td>
      <td>34.080781</td>
      <td>33.968724</td>
      <td>34.550172</td>
      <td>34.036444</td>
      <td>34.003281</td>
      <td>34.916955</td>
      <td>1649.794674</td>
      <td>-353.205790</td>
      <td>-837.013264</td>
    </tr>
    <tr>
      <th>13</th>
      <td>34.171184</td>
      <td>34.144603</td>
      <td>34.252303</td>
      <td>34.512315</td>
      <td>34.087980</td>
      <td>34.039629</td>
      <td>21.294895</td>
      <td>34.131926</td>
      <td>34.013335</td>
      <td>34.529361</td>
      <td>34.068364</td>
      <td>34.045688</td>
      <td>34.810067</td>
      <td>1914.579309</td>
      <td>-441.846623</td>
      <td>-947.892909</td>
    </tr>
    <tr>
      <th>14</th>
      <td>34.296284</td>
      <td>34.257165</td>
      <td>34.339166</td>
      <td>34.563781</td>
      <td>34.204525</td>
      <td>34.144521</td>
      <td>24.152679</td>
      <td>34.219315</td>
      <td>34.112406</td>
      <td>34.553075</td>
      <td>34.179505</td>
      <td>34.156759</td>
      <td>34.810100</td>
      <td>2311.935890</td>
      <td>-618.760193</td>
      <td>-1213.194881</td>
    </tr>
    <tr>
      <th>15</th>
      <td>34.461999</td>
      <td>34.399711</td>
      <td>34.453803</td>
      <td>34.684916</td>
      <td>34.353880</td>
      <td>34.277074</td>
      <td>23.958488</td>
      <td>34.335170</td>
      <td>34.237223</td>
      <td>34.678382</td>
      <td>34.346276</td>
      <td>34.309551</td>
      <td>34.912484</td>
      <td>2851.089921</td>
      <td>-842.203764</td>
      <td>-1560.380246</td>
    </tr>
    <tr>
      <th>16</th>
      <td>34.607066</td>
      <td>34.525875</td>
      <td>34.558676</td>
      <td>34.822993</td>
      <td>34.489013</td>
      <td>34.392501</td>
      <td>25.057221</td>
      <td>34.538285</td>
      <td>34.342468</td>
      <td>34.853028</td>
      <td>34.489351</td>
      <td>34.438145</td>
      <td>35.060213</td>
      <td>3355.250128</td>
      <td>-1045.439319</td>
      <td>-1859.202847</td>
    </tr>
    <tr>
      <th>17</th>
      <td>34.702365</td>
      <td>34.612209</td>
      <td>34.630243</td>
      <td>34.944571</td>
      <td>34.576656</td>
      <td>34.465556</td>
      <td>24.176593</td>
      <td>34.631675</td>
      <td>34.409394</td>
      <td>34.916840</td>
      <td>34.577513</td>
      <td>34.514188</td>
      <td>35.210403</td>
      <td>3717.193199</td>
      <td>-1180.901238</td>
      <td>-2043.168578</td>
    </tr>
    <tr>
      <th>18</th>
      <td>34.702938</td>
      <td>34.612720</td>
      <td>34.639484</td>
      <td>34.999141</td>
      <td>34.571422</td>
      <td>34.460077</td>
      <td>23.193792</td>
      <td>34.642223</td>
      <td>34.404673</td>
      <td>34.995033</td>
      <td>34.568143</td>
      <td>34.503903</td>
      <td>35.300411</td>
      <td>3591.846998</td>
      <td>-1183.779701</td>
      <td>-2024.799485</td>
    </tr>
    <tr>
      <th>19</th>
      <td>34.648399</td>
      <td>34.569221</td>
      <td>34.602834</td>
      <td>34.995695</td>
      <td>34.522153</td>
      <td>34.416743</td>
      <td>19.960147</td>
      <td>34.545620</td>
      <td>34.356994</td>
      <td>35.025554</td>
      <td>34.506081</td>
      <td>34.443503</td>
      <td>35.326985</td>
      <td>3453.274031</td>
      <td>-1106.697470</td>
      <td>-1891.100556</td>
    </tr>
    <tr>
      <th>20</th>
      <td>34.526180</td>
      <td>34.453074</td>
      <td>34.521736</td>
      <td>34.922179</td>
      <td>34.401185</td>
      <td>34.309956</td>
      <td>22.490135</td>
      <td>34.469354</td>
      <td>34.254442</td>
      <td>34.926636</td>
      <td>34.381196</td>
      <td>34.325694</td>
      <td>35.262639</td>
      <td>2963.423841</td>
      <td>-930.938120</td>
      <td>-1607.095294</td>
    </tr>
    <tr>
      <th>21</th>
      <td>34.368686</td>
      <td>34.285951</td>
      <td>34.398886</td>
      <td>34.793701</td>
      <td>34.241898</td>
      <td>34.172459</td>
      <td>23.784032</td>
      <td>34.287215</td>
      <td>34.124560</td>
      <td>34.819036</td>
      <td>34.229451</td>
      <td>34.183542</td>
      <td>35.126756</td>
      <td>2433.011487</td>
      <td>-688.761781</td>
      <td>-1258.308390</td>
    </tr>
    <tr>
      <th>22</th>
      <td>34.191969</td>
      <td>34.112291</td>
      <td>34.271561</td>
      <td>34.645317</td>
      <td>34.057410</td>
      <td>34.018258</td>
      <td>20.532427</td>
      <td>34.051586</td>
      <td>33.984681</td>
      <td>34.648611</td>
      <td>34.061895</td>
      <td>34.024722</td>
      <td>34.951145</td>
      <td>1768.502601</td>
      <td>-430.119069</td>
      <td>-882.598102</td>
    </tr>
    <tr>
      <th>23</th>
      <td>34.058366</td>
      <td>33.997078</td>
      <td>34.175324</td>
      <td>34.506956</td>
      <td>33.931314</td>
      <td>33.910463</td>
      <td>19.148287</td>
      <td>33.849665</td>
      <td>33.886879</td>
      <td>34.517857</td>
      <td>33.931640</td>
      <td>33.908138</td>
      <td>34.766514</td>
      <td>1409.641409</td>
      <td>-251.600610</td>
      <td>-623.406276</td>
    </tr>
    <tr>
      <th>24</th>
      <td>33.993800</td>
      <td>33.948988</td>
      <td>34.127358</td>
      <td>34.404633</td>
      <td>33.880646</td>
      <td>33.865132</td>
      <td>22.772972</td>
      <td>33.810257</td>
      <td>33.843995</td>
      <td>34.420952</td>
      <td>33.868331</td>
      <td>33.860516</td>
      <td>34.615595</td>
      <td>1169.056830</td>
      <td>-179.032036</td>
      <td>-514.965935</td>
    </tr>
  </tbody>
</table>
</div>




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


    
![png](freyberg_da_run_files/freyberg_da_run_16_0.png)
    



    
![png](freyberg_da_run_files/freyberg_da_run_16_1.png)
    



    
![png](freyberg_da_run_files/freyberg_da_run_16_2.png)
    


These plots look very different dont they... What we are showing in each pair is the prior simulated results on top and then the prior and posterior simulted results on the bottom.   Red circles are truth values not used for conditioning (we usually dont have these...), the red triangles are obseravtions that were assimilated in a given cycle.  The reason we shows vertically stacked points instead of connected lines is because in the sequential estimation framework, the parameter and observation ensembles pertain only to the current cycle. Remembering that each "prior" simulated output ensemble is the forecast from the previous cycle to the current cycle without having "seen" any observations for the current cycle.  So we can see that after the first cycle with observations (cycle = 1), the model starts "tracking" the dynamics and it is pretty good a predicting the next cycles value.


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


    
![png](freyberg_da_run_files/freyberg_da_run_18_0.png)
    



    
![png](freyberg_da_run_files/freyberg_da_run_18_1.png)
    


So thats pretty impressive right?  We are bracketing the sw/gw flux behavior for each cycle in the "one-step-ahead" sense (i.e. the prior plots).  And, for this single truth we are using, we also do pretty well through the 12-month/12-cycle forecast period (the last 12 cycles/months).  

Is PESTPP-DA worth the cost (in terms of cognitive load and increased computational burden)?  As always, "it depends"!


```python

```
