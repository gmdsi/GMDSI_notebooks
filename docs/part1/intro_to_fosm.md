---
layout: default
title: Intro to FOSM
parent: Introduction to Theory, Concepts and PEST Mechanic
nav_order: 10
math: mathjax3
---

# FOSM - a brief overview (with equations!)

Throughout the previous tutorial notebooks we have explored how parameter estimation affects posterior parameter and forecast uncertainty. This notebook goes through some of the detail of how these uncertainties are calculated by PEST++ and `pyemu`. 

FOSM stands for "First Order, Second Moment", which is the mathematical description of what is being described. In PEST documentation (and other GMDSI tutorials), it is sometimes referred to as "linear analysis". See also page 460 in [Anderson et al. (2015)](https://linkinghub.elsevier.com/retrieve/pii/B9780080916385000018). 

> <div class="csl-entry">Anderson, M. P., Woessner, W. W., &#38; Hunt, R. J. (2015). Applied Groundwater Modeling: Simulation of Flow and Advective Transport. In <i>Applied Groundwater Modeling</i> (2nd ed.). Elsevier. https://linkinghub.elsevier.com/retrieve/pii/B9780080916385000018</div>

Pages 461-465 of Anderson et al. use the PREDUNC equation of PEST to discuss an applied view of FOSM, what goes into it, and what it means in practice.  Here we will look more closely at these.  The objective is to get a better feel for what is going on under the hood in linear uncertainty analyses. 

> __Side Note__: in Part2 of this series of tutorial notebooks we demonstrate a complete FOSM and Data-worth analysis workflow using `pyemu` and PEST++. The current notebook merely aims to provide a very high level introduction to some of the concepts.

<img src="intro_to_fosm_files/bayes.png" style="inline: left; width: 25%; margin-right: 1%; margin-bottom: 0.5em;">
<img src="intro_to_fosm_files/jacobi.jpg" style="inline: left; width: 25%; margin-right: 1%; margin-bottom: 0.5em;">
<img src="intro_to_fosm_files/gauss.jpg" style="inline: left; width: 22%; margin-right: 1%; margin-bottom: 0.5em;">
<img src="intro_to_fosm_files/schur.jpg" style="inline: left; width: 22%; margin-right: 1%; margin-bottom: 0.5em;">


FOSM provides approximate mathematical characterisation of prior predictive probability distributions, and of posterior parameter and predictive probability distributions. It has other uses as well. It can be used to demonstrate how the history-matching process bestows worth on data. It can also be deployed to track the flow of information from field measurements of system state to parameters, and ultimately from parameters to model predictions. 

It does all of these things by implementing Bayes equation under the following assumptions:
 -  The prior probability distribution of parameters is multiGaussian.
 - “Measurement noise” (including structural noise) is also characterized by a Gaussian distribution.
 - The relationships between model outputs that correspond to measurements of system state and parameters employed by a model can be approximated by the action of a matrix on a vector.
 - Model outputs that correspond to predictions of management interest can be calculated using another matrix that acts on model parameters.
 
Ideally linear analysis is undertaken after a model has been calibrated. However, if a model is truly linear (which it never is), the outcomes of FOSM are independent of parameter values and can therefore, in theory, be applied with the user-supplied prior mean parameter values.

If calibration has been undertaken, then minimum-error variance (i.e. calibrated) parameter values should be assigned to parameters as their initial parameters in the “parameter data” section of the PEST control file on which linear analysis is based. The Jacobian matrix should be calculated using these parameters. And, if the uncertainty of a prediction is going to be examined, then the model output that pertains to this prediction must be included as an “observation” in the PEST input dataset; sensitivities of this model output to model parameters will therefore appear in the Jacobian matrix.

FOSM tasks may include:
 - approximate parameter and predictive uncertainty quantification;
 - data worth analysis;
 - identifying parameters that are most salient for forecasts of interest, 
 - identifying parameter contributions to predictive uncertainty and 
 - assessing parameter identifiability. 

Outcomes of these analyses can provide easily understood insights into what history-matching can and cannot achieve with the available information. These insights can be used to streamline the data assimilation process and guide further site characterisation studies. Of particular interest is data worth analysis. The worth of data is measured by their ability to reduce the uncertainties of model predictions that we care about. Because the equations on which FOSM relies do not require that an observation value already be known, data worth can be assessed on as-of-yet ungathered data. 

But we are getting ahead of ourselves, let's take this back to basics.


## The famous Bayes Rule in a nutshell:

We update our knowledge by comparing what we know/believe with measured/observed data. What we know now, is a function of what we knew before, compared to what we learned from measured data.

$$\underbrace{P(\boldsymbol{\theta}|\textbf{d})}_{\substack{\text{what we} \\ \text{know now}}} \propto \underbrace{\mathcal{L}(\boldsymbol{\theta} | \textbf{d})}_{\substack{\text{what we} \\ \text{learned}}} \underbrace{P(\boldsymbol{\theta})}_{\substack{\text{what we} \\ \text{knew}}}$$


We can also think of this graphically, as taken from Anderson et al. (2015) in slightly different notation but the same equation and concept:

<img src="intro_to_fosm_files/Fig10.3_Bayes_figure.png" style="inline:center;width:500px;"/>

The problem is, for real-world problems, the likelihood function ("what we learned") is high-dimensional and non-parametric, requiring non-linear (typically Monte Carlo) integration for rigorous Bayes. Unfortunately, non-linear methods are computationally expensive and inefficient as we will see in a subsequent notebook. 

But, we can make some assumptions and greatly reduce computational burden. This is why we often suggest using these linear methods first before burning the silicon on the non-linear ones like Monte Carlo.  

## How do we reduce the computational burden? 

By assuming that:

### 1. There is an approximate linear relation between parameters and observations:

<img src="intro_to_fosm_files/jacobi.jpg" style="inline: left; width: 5%; margin-right: 1%; margin-bottom: 0.5em;">

### <center> $\mathbf{J} \approx \text{constant}$, $\frac{\partial\text{obs}}{\partial\text{par}} \approx \text{constant}$</center>

### 2. The parameter and forecast prior and posterior distributions are approximately Gaussian:

<img src="intro_to_fosm_files/gauss.jpg" style="inline: left; width: 5%; margin-right: 1%; margin-bottom: 0.5em;">

###  <center>  $ P(\boldsymbol{\theta}|\mathbf{d}) \approx \mathcal{N}(\overline{\boldsymbol{\mu}}_{\boldsymbol{\theta}},\overline{\boldsymbol{\Sigma}}_{\boldsymbol{\theta}})$ </center>

Armed with these two assumptions, from Bayes equations, one can derive the Schur complement for conditional uncertainty propagation:

<img src="intro_to_fosm_files/schur.jpg" style="inline: left; width: 5%; margin-right: 1%; margin-bottom: 0.5em;">

### <center> $\underbrace{\overline{\boldsymbol{\Sigma}}_{\boldsymbol{\theta}}}_{\substack{\text{what we} \\ \text{know now}}} = \underbrace{\boldsymbol{\Sigma}_{\boldsymbol{\theta}}}_{\substack{\text{what we} \\ \text{knew}}} - \underbrace{\boldsymbol{\Sigma}_{\boldsymbol{\theta}}\bf{J}^T\left[\bf{J}\boldsymbol{\Sigma}_{\boldsymbol{\theta}}\bf{J}^T + \boldsymbol{\Sigma}_{\boldsymbol{\epsilon}}\right]^{-1}\bf{J}\boldsymbol{\Sigma}_{\boldsymbol{\theta}}}_{\text{what we learned}}$ </center>

### Highlights:
1. There are no parameter values or observation values in these equations!
2. "us + data" = $\overline{\Sigma}_{\theta}$; "us" = $\Sigma_{\theta}$. This accounts for information from both data and expert knowledge.
3. The '-' on the right-hand-side shows that we are (hopefully) collapsing the probability manifold in parameter space by "learning" from the data. Or put another way, we are subtracting from the uncertainty we started with (we started with the Prior uncertainty)
4. Uncertainty in our measurements of the world is encapsulated in $\Sigma_{\epsilon}$. If the "observations" are highly uncertain, then parameter "learning" decreases because $\Sigma_{\epsilon}$ is in the denominator. Put another way, if our measured data are made (assumed) to be accurate and precise, then uncertainty associated with the parameters that are constrained by these measured data is reduced - we "learn" more. 
5. What quantities are needed? $\bf{J}$, $\boldsymbol{\Sigma}_{\theta}$, and $\boldsymbol{\Sigma}_{\epsilon}$
6. The diagonal of $\Sigma_{\theta}$ and $\overline{\Sigma}_{\theta}$ are the Prior and Posterior uncertainty (variance) of each adjustable parameter

# But what about forecasts? 

<img src="intro_to_fosm_files/jacobi.jpg" style="inline: left; width: 5%; margin-right: 1%; margin-bottom: 0.5em;">
<img src="intro_to_fosm_files/gauss.jpg" style="inline: left; width: 5%; margin-right: 1%; margin-bottom: 0.5em;">


We can use the same assumptions:
    
- prior forecast uncertainty (variance): $\sigma^2_{s} = \mathbf{y}^T\boldsymbol{\Sigma}_{\boldsymbol{\theta}}\mathbf{y}$
- posterior forecast uncertainty (variance): $\overline{\sigma}^2_{s} = \mathbf{y}^T\overline{\boldsymbol{\Sigma}}_{\boldsymbol{\theta}}\mathbf{y}$


### Highlights:
- Again, no parameter values or forecast values!
- What's needed? $\bf{y}$, which is the __sensitivity of a given forecast__ to each adjustable parameter. Each forecast will have its own $\bf{y}$.
- How do I get $\bf{y}$? the easiest way is to include your forecast(s) as an observation in the control file - then we get the $\bf{y}$'s for free during the parameter estimation process.

# Mechanics of calculating FOSM parameter and forecast uncertainty estimates

__in the PEST world:__

In the origingal PEST (i.e., not PEST++) documentation, FOSM is referred to as linear analysis. Implementing the various linear analyses relies a suite of utility software and a series of user-input-heavy steps, as illustrated in the figure below. 

<img src="intro_to_fosm_files/workflow.png" style="inline: left; width: 50%; margin-right: 1%; margin-bottom: 0.5em;">



__in PEST++__:

In the PEST++ world, life is much easier. By default, PEST++GLM implements FOSM on-the-fly (it can be deactivated if the user desires) and records parameter and forecast uncertainties throughout the parameter estimation process.

Let's take a closer look and get a feel what is going on. 

# FOSM with PEST++ Demo

In the tutorial directory there is a folder containing the outcomes a PEST++GLM parameter estimation run. (These are based on the model and PEST setup constructed in the "_part1_freyberg_pilotpoints_" notebooks.) In the following section we will access several of these files using `pyemu`. It is assumed that the reader is familiar with the basics of `pyemu`.

Parameter estimation has already been undertaken with PEST++GLM. So we already have at our disposal a `jacobian matrix`, and the parameter and forecast uncertainty files written by PEST++GLM.


```python
import sys
import os
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt;
import psutil
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

The folder with model and pest files:



```python
working_dir = "master_pp"
```

The PEST control file name:


```python
pst_name = "freyberg_pp.pst"
```

Load the PEST control file:


```python
pst = pyemu.Pst(os.path.join(working_dir, pst_name))
```

### Let's look at the parameter uncertainty summary written by pestpp:

PEST++GLM records a parameter uncertainty file named _casename.par.usum.csv_. It records the prior and posterior means, bounds and standard deviations.


```python
df = pd.read_csv(os.path.join(working_dir,pst_name.replace(".pst",".par.usum.csv")),index_col=0)
df.tail()
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
      <th>prior_mean</th>
      <th>prior_stdev</th>
      <th>prior_lower_bound</th>
      <th>prior_upper_bound</th>
      <th>post_mean</th>
      <th>post_stdev</th>
      <th>post_lower_bound</th>
      <th>post_upper_bound</th>
    </tr>
    <tr>
      <th>name</th>
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
      <th>rch_i:17_j:2_zone:1.0</th>
      <td>0.0</td>
      <td>0.150515</td>
      <td>-0.30103</td>
      <td>0.30103</td>
      <td>0.30103</td>
      <td>0.149751</td>
      <td>0.001528</td>
      <td>0.600532</td>
    </tr>
    <tr>
      <th>rch_i:17_j:12_zone:1.0</th>
      <td>0.0</td>
      <td>0.150515</td>
      <td>-0.30103</td>
      <td>0.30103</td>
      <td>-0.30103</td>
      <td>0.150025</td>
      <td>-0.601081</td>
      <td>-0.000979</td>
    </tr>
    <tr>
      <th>rch_i:37_j:17_zone:1.0</th>
      <td>0.0</td>
      <td>0.150515</td>
      <td>-0.30103</td>
      <td>0.30103</td>
      <td>-0.30103</td>
      <td>0.150258</td>
      <td>-0.601546</td>
      <td>-0.000514</td>
    </tr>
    <tr>
      <th>rch_i:27_j:12_zone:1.0</th>
      <td>0.0</td>
      <td>0.150515</td>
      <td>-0.30103</td>
      <td>0.30103</td>
      <td>-0.30103</td>
      <td>0.150081</td>
      <td>-0.601192</td>
      <td>-0.000868</td>
    </tr>
    <tr>
      <th>rch_i:12_j:17_zone:1.0</th>
      <td>0.0</td>
      <td>0.150515</td>
      <td>-0.30103</td>
      <td>0.30103</td>
      <td>-0.30103</td>
      <td>0.149847</td>
      <td>-0.600725</td>
      <td>-0.001335</td>
    </tr>
  </tbody>
</table>
</div>



We can visualize this with probability distributions. In the plot below, prior parameter distributions are shown by the dashed grey lines. Posterior parameter distributions are the blue shaded areas. Each plot shows distributions for parameters in the same group:


```python
par = pst.parameter_data
df_paru = pd.read_csv(os.path.join(working_dir,pst_name.replace(".pst",".par.usum.csv")),index_col=0)

fig, axes=plt.subplots(1,len(pst.adj_par_groups),figsize=(15,5))

for pargp, ax in zip(pst.adj_par_groups, axes):
    hk_pars = [p for p in pst.par_names if p.startswith("hk")]
    pars = par.loc[par.pargp==pargp].parnme.values
    df_par = df_paru.loc[pars,:]
    ax = pyemu.plot_utils.plot_summary_distributions(df_par,label_post=False, ax=ax)
    mn = np.log10(pst.parameter_data.loc[pars[0].lower(),"parlbnd"])
    mx = np.log10(pst.parameter_data.loc[pars[0].lower(),"parubnd"])
    ax.set_title(pargp)
```


    
![png](intro_to_fosm_files/intro_to_fosm_26_0.png)
    


### There is a similar file for forecasts:
_casename.pred.usum.csv_


```python
axes = pyemu.plot_utils.plot_summary_distributions(os.path.join(working_dir,pst_name.replace(".pst",".pred.usum.csv")),subplots=True)
```


    
![png](intro_to_fosm_files/intro_to_fosm_28_0.png)
    


### Where do the prior parameter distributions come from?

Prior parameter distributions can come from one of two sources. 

1. If no other information is provided, PEST++GLM assumes that all adjustable parameters are statistically independent. In this case, by default, the prior standard deviation of each parameter is calculated as a quarter of the difference between its upper and lower bounds in the PEST control file.(This is the case here)
2. Alternatively, the name of a prior parameter covariance matrix file can be provided to the `parcov()` control variable.


### Where do the prior forecast distributions come from?

At the first iteration of the parameter estimation process, PEST++GLM calculates sensitivities based on initial parameter values. These are used to determine the prior parameter and forecast uncertainty.


### Why are are the posterior distributions different than the priors?

Recall Bayes' Rule? By comparing model outputs to measured data we have "learnt" information about model parameters, thus "updating our prior" and reducing parameter (and forecast) uncertainty.


# FOSM with pyEMU

Now, `pyemu` does the same calculations, but also allows you to do other, more exciting things! 

We need three ingredients for FOSM:
 - parameter covariance matrix 
 - observation noise covariance matrix
 - jacobian matrix 


The ``Schur`` object is one of the primary object for FOSM in pyEMU and the only one we will talk about in this tutorial.


```python
sc = pyemu.Schur(jco=os.path.join(working_dir,pst_name.replace(".pst",".jcb")),verbose=False)
```

Now that seemed too easy, right?  Well, underhood the ``Schur`` object found the control file ("freyberg_pp.pst") and used it to build the prior parameter covariance matrix, from the parameter bounds and the observation noise covariance matrix from the observation weights.  These are the ``Schur.parcov`` and ``Schur.obscov`` attributes.  

The ``Schur`` object also found the "++forecasts()" optional pestpp argument in the control, found the associated rows in the Jacobian matrix file and extracted those rows to serve as forecast sensitivity vectors:


```python
sc.pst.pestpp_options['forecasts']
```




    'headwater:4383.5,tailwater:4383.5,trgw-0-9-1:4383.5,part_time'



### The Jacobian Matrix and Forecast Sensitivity Vectors

Recall that a Jacobian matrix looks at the changes in observations as a parameter is changed.  Therefore the Jacobian matrix has parameters in the columns and observations in the rows.  The bulk of the matrix is made up of the difference in  observations between a base run and a run where the parameter at the column head was perturbed (typically 1% from the base run value - controlled by the "parameter groups" info).  Now we'll plot out the Jacobian matrix as a `DataFrame`:


```python
sc.jco.to_dataframe().loc[sc.pst.nnz_obs_names,:].head()
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
      <th>strinf</th>
      <th>wel0</th>
      <th>wel2</th>
      <th>wel4</th>
      <th>wel5</th>
      <th>wel1</th>
      <th>wel3</th>
      <th>hk_i:12_j:2_zone:1.0</th>
      <th>hk_i:32_j:12_zone:1.0</th>
      <th>hk_i:2_j:17_zone:1.0</th>
      <th>...</th>
      <th>rch_i:2_j:7_zone:1.0</th>
      <th>rch_i:2_j:17_zone:1.0</th>
      <th>rch_i:7_j:17_zone:1.0</th>
      <th>rch_i:22_j:12_zone:1.0</th>
      <th>rch_i:22_j:2_zone:1.0</th>
      <th>rch_i:17_j:2_zone:1.0</th>
      <th>rch_i:17_j:12_zone:1.0</th>
      <th>rch_i:37_j:17_zone:1.0</th>
      <th>rch_i:27_j:12_zone:1.0</th>
      <th>rch_i:12_j:17_zone:1.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>gage-1:3652.5</th>
      <td>2716.948265</td>
      <td>-4.548055</td>
      <td>-4.719683</td>
      <td>-5.232870</td>
      <td>-4.498392</td>
      <td>-4.689892</td>
      <td>-5.017089</td>
      <td>3.445071</td>
      <td>31.416424</td>
      <td>15.240922</td>
      <td>...</td>
      <td>190.031485</td>
      <td>115.306638</td>
      <td>118.042265</td>
      <td>119.114189</td>
      <td>223.313201</td>
      <td>231.708826</td>
      <td>138.388709</td>
      <td>63.774836</td>
      <td>113.687036</td>
      <td>121.292411</td>
    </tr>
    <tr>
      <th>gage-1:3683.5</th>
      <td>2774.639984</td>
      <td>-181.311316</td>
      <td>-166.978297</td>
      <td>-10.342791</td>
      <td>-87.590161</td>
      <td>-110.276389</td>
      <td>-32.921654</td>
      <td>4.402686</td>
      <td>32.094217</td>
      <td>15.121772</td>
      <td>...</td>
      <td>221.004388</td>
      <td>134.915061</td>
      <td>136.592731</td>
      <td>130.330209</td>
      <td>238.872431</td>
      <td>252.221559</td>
      <td>153.398371</td>
      <td>69.859064</td>
      <td>123.139058</td>
      <td>138.266282</td>
    </tr>
    <tr>
      <th>gage-1:3712.5</th>
      <td>2780.020916</td>
      <td>-276.473740</td>
      <td>-245.350759</td>
      <td>-20.599377</td>
      <td>-151.432872</td>
      <td>-185.385447</td>
      <td>-67.049198</td>
      <td>4.654596</td>
      <td>36.865047</td>
      <td>18.845729</td>
      <td>...</td>
      <td>229.133526</td>
      <td>153.496081</td>
      <td>156.089054</td>
      <td>145.632341</td>
      <td>242.263018</td>
      <td>256.949881</td>
      <td>169.704385</td>
      <td>83.319417</td>
      <td>138.394319</td>
      <td>158.253214</td>
    </tr>
    <tr>
      <th>gage-1:3743.5</th>
      <td>2783.890941</td>
      <td>-335.620872</td>
      <td>-291.893149</td>
      <td>-35.153415</td>
      <td>-197.641178</td>
      <td>-238.817336</td>
      <td>-101.203897</td>
      <td>4.971822</td>
      <td>46.180032</td>
      <td>24.684773</td>
      <td>...</td>
      <td>242.111479</td>
      <td>176.567497</td>
      <td>180.120737</td>
      <td>164.500045</td>
      <td>247.881326</td>
      <td>264.289167</td>
      <td>190.624789</td>
      <td>98.052362</td>
      <td>156.877133</td>
      <td>182.798667</td>
    </tr>
    <tr>
      <th>gage-1:3773.5</th>
      <td>2785.355604</td>
      <td>-374.821849</td>
      <td>-322.374110</td>
      <td>-50.609492</td>
      <td>-228.410582</td>
      <td>-276.715310</td>
      <td>-129.809297</td>
      <td>5.299181</td>
      <td>55.300079</td>
      <td>29.438466</td>
      <td>...</td>
      <td>256.583441</td>
      <td>194.019730</td>
      <td>198.089350</td>
      <td>178.866029</td>
      <td>255.292138</td>
      <td>272.878837</td>
      <td>207.569112</td>
      <td>107.102969</td>
      <td>170.503931</td>
      <td>201.037663</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 65 columns</p>
</div>



This reports changes in observations to a change in a parameter.  We can report how  forecasts of interests change as the parameter is perturbed.  Note `pyemu` extracted the forecast rows from the Jacobian on instantiation:


```python
sc.forecasts.to_dataframe()
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
      <th>headwater:4383.5</th>
      <th>tailwater:4383.5</th>
      <th>trgw-0-9-1:4383.5</th>
      <th>part_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>strinf</th>
      <td>4.525783</td>
      <td>15.049001</td>
      <td>0.019712</td>
      <td>-31.396749</td>
    </tr>
    <tr>
      <th>wel0</th>
      <td>14.641942</td>
      <td>3.898550</td>
      <td>-0.100690</td>
      <td>-0.731877</td>
    </tr>
    <tr>
      <th>wel2</th>
      <td>11.608075</td>
      <td>10.579199</td>
      <td>-0.095968</td>
      <td>-21.285655</td>
    </tr>
    <tr>
      <th>wel4</th>
      <td>18.162402</td>
      <td>33.111962</td>
      <td>-0.175127</td>
      <td>-72.519641</td>
    </tr>
    <tr>
      <th>wel5</th>
      <td>8.105663</td>
      <td>17.741830</td>
      <td>-0.097510</td>
      <td>-36.384135</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>rch_i:17_j:2_zone:1.0</th>
      <td>-101.045614</td>
      <td>-61.271020</td>
      <td>0.718966</td>
      <td>3973.941965</td>
    </tr>
    <tr>
      <th>rch_i:17_j:12_zone:1.0</th>
      <td>-11.717836</td>
      <td>-7.108638</td>
      <td>0.056490</td>
      <td>52.716389</td>
    </tr>
    <tr>
      <th>rch_i:37_j:17_zone:1.0</th>
      <td>-0.778950</td>
      <td>-1.143194</td>
      <td>0.004321</td>
      <td>17.939060</td>
    </tr>
    <tr>
      <th>rch_i:27_j:12_zone:1.0</th>
      <td>-7.106006</td>
      <td>-8.015407</td>
      <td>0.041517</td>
      <td>183.053088</td>
    </tr>
    <tr>
      <th>rch_i:12_j:17_zone:1.0</th>
      <td>-4.895285</td>
      <td>-1.892905</td>
      <td>0.020423</td>
      <td>-33.987577</td>
    </tr>
  </tbody>
</table>
<p>65 rows × 4 columns</p>
</div>



Each of these columns in a $\bf{y}$ vector used in the FOSM calculations...that's it! 

###  The prior parameter covariance matrix - $\boldsymbol{\Sigma}_{\theta}$

Because we have inherent uncertainty in the parameters, the forecasts also have uncertainty.  Here's what we have defined for parameter uncertainty - the Prior.  As discussed above, it was constructed on-the-fly from the parameter bounds in the control file: 


```python
sc.parcov.to_dataframe()
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
      <th>strinf</th>
      <th>wel0</th>
      <th>wel2</th>
      <th>wel4</th>
      <th>wel5</th>
      <th>wel1</th>
      <th>wel3</th>
      <th>hk_i:12_j:2_zone:1.0</th>
      <th>hk_i:32_j:12_zone:1.0</th>
      <th>hk_i:2_j:17_zone:1.0</th>
      <th>...</th>
      <th>rch_i:2_j:7_zone:1.0</th>
      <th>rch_i:2_j:17_zone:1.0</th>
      <th>rch_i:7_j:17_zone:1.0</th>
      <th>rch_i:22_j:12_zone:1.0</th>
      <th>rch_i:22_j:2_zone:1.0</th>
      <th>rch_i:17_j:2_zone:1.0</th>
      <th>rch_i:17_j:12_zone:1.0</th>
      <th>rch_i:37_j:17_zone:1.0</th>
      <th>rch_i:27_j:12_zone:1.0</th>
      <th>rch_i:12_j:17_zone:1.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>strinf</th>
      <td>0.25</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>wel0</th>
      <td>0.00</td>
      <td>0.238691</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>wel2</th>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.238691</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>wel4</th>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.238691</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>wel5</th>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.238691</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>rch_i:17_j:2_zone:1.0</th>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.022655</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>rch_i:17_j:12_zone:1.0</th>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.022655</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>rch_i:37_j:17_zone:1.0</th>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.022655</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>rch_i:27_j:12_zone:1.0</th>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.022655</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>rch_i:12_j:17_zone:1.0</th>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.022655</td>
    </tr>
  </tbody>
</table>
<p>65 rows × 65 columns</p>
</div>



> Page 463-464 in Anderson et al. (2015) spends some time on what is shown above.  

For our purposes, a diagonal Prior -  numbers only along the diagonal - shows that we expect the uncertainty for each parameter to only results from itself - there is no covariance with other parameters. The numbers themselves reflect "the innate parameter variability", and is input into the maths as a standard deviation around the parameter value.  This is called the "C(p) matrix of innate parameter variability" in PEST parlance.

> __IMPORTANT POINT__:  Again, how did PEST++ and pyEMU get these standard deviations shown in the diagonal?  From the *parameter bounds* that were specified for each parameter in the PEST control file.

### The  matrix  of observation noise - $C{\epsilon}$

Forecast uncertainty has to take into account the noise/uncertainty in the observations.   Similar to the parameter Prior - the $\Sigma_{\theta}$ matrix -, it is a covariance matrix of measurement error associated with the observations.  This is the same as  $\Sigma_{\epsilon}$ that we discussed above. For our Freyberg problem, the $C{\epsilon}$ matrix would look like:


```python
sc.obscov.to_dataframe().loc[sc.pst.nnz_obs_names,sc.pst.nnz_obs_names].head()
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
      <th>gage-1:3652.5</th>
      <th>gage-1:3683.5</th>
      <th>gage-1:3712.5</th>
      <th>gage-1:3743.5</th>
      <th>gage-1:3773.5</th>
      <th>gage-1:3804.5</th>
      <th>gage-1:3834.5</th>
      <th>gage-1:3865.5</th>
      <th>gage-1:3896.5</th>
      <th>gage-1:3926.5</th>
      <th>...</th>
      <th>trgw-0-3-8:3743.5</th>
      <th>trgw-0-3-8:3773.5</th>
      <th>trgw-0-3-8:3804.5</th>
      <th>trgw-0-3-8:3834.5</th>
      <th>trgw-0-3-8:3865.5</th>
      <th>trgw-0-3-8:3896.5</th>
      <th>trgw-0-3-8:3926.5</th>
      <th>trgw-0-3-8:3957.5</th>
      <th>trgw-0-3-8:3987.5</th>
      <th>trgw-0-3-8:4018.5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>gage-1:3652.5</th>
      <td>40000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>gage-1:3683.5</th>
      <td>0.0</td>
      <td>40000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>gage-1:3712.5</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>40000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>gage-1:3743.5</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>40000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>gage-1:3773.5</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>40000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 37 columns</p>
</div>



> __IMPORTANT POINT__:  How did PEST++ and pyEMU get these standard deviations shown in the diagonal?  From the *weights* that were specified for each observation in the PEST control file.

> __IMPORTANT POINT__: You can use FOSM in the "pre-calibration" state to design an objective function (e.g. weights) to maximize forecast uncertainty reduction.

> __IMPORTANT POINT__: In PEST++, if a given observation has a larger-than-expected residual, the variance of said observation is reset to the variance implied by the residual.  That is, the diagonal elements of $\Sigma_{\epsilon}$ are reset according to the residuals

## Posterior Parameter Uncertainty - ${\overline{\boldsymbol{\Sigma}}_{\boldsymbol{\theta}}} $

Okay, enough emphasis.  Here's the point.  When we apply FOSM using the matrices above, we can see how our uncertainty changes during calibration, first for parameters and then for forecasts. 

Here, we are updating parameter covariance following notional calibration as represented by the Jacobian matrix and both prior parameter and observation noise covariance matrices. 

In other words, given prior parameter uncertainty and the inherent noise in measurements, we calculate the expected parameter uncertainty __after__ calibration. This assumes that _calibration achieves a fit commensurate with measurement noise, parameter linearity, etc_.

The posterior parameter covariance matrix is stored as a `pyemu.Cov` object in the `sc.posterior_parameter` attribute. The diagonal of this matrix contains the posterior __variance__ for each parameter. The off-diagonals the parameter covariances. 


```python
sc.posterior_parameter.to_dataframe().head()
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
      <th>strinf</th>
      <th>wel0</th>
      <th>wel2</th>
      <th>wel4</th>
      <th>wel5</th>
      <th>wel1</th>
      <th>wel3</th>
      <th>hk_i:12_j:2_zone:1.0</th>
      <th>hk_i:32_j:12_zone:1.0</th>
      <th>hk_i:2_j:17_zone:1.0</th>
      <th>...</th>
      <th>rch_i:2_j:7_zone:1.0</th>
      <th>rch_i:2_j:17_zone:1.0</th>
      <th>rch_i:7_j:17_zone:1.0</th>
      <th>rch_i:22_j:12_zone:1.0</th>
      <th>rch_i:22_j:2_zone:1.0</th>
      <th>rch_i:17_j:2_zone:1.0</th>
      <th>rch_i:17_j:12_zone:1.0</th>
      <th>rch_i:37_j:17_zone:1.0</th>
      <th>rch_i:27_j:12_zone:1.0</th>
      <th>rch_i:12_j:17_zone:1.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>strinf</th>
      <td>0.006971</td>
      <td>0.004594</td>
      <td>0.004228</td>
      <td>0.000588</td>
      <td>0.002349</td>
      <td>0.003672</td>
      <td>0.001565</td>
      <td>-0.001135</td>
      <td>-0.005610</td>
      <td>0.000060</td>
      <td>...</td>
      <td>-0.001500</td>
      <td>-0.000722</td>
      <td>-0.000726</td>
      <td>-0.000746</td>
      <td>-0.001539</td>
      <td>-0.001735</td>
      <td>-0.000883</td>
      <td>-0.000382</td>
      <td>-0.000699</td>
      <td>-0.000737</td>
    </tr>
    <tr>
      <th>wel0</th>
      <td>0.004594</td>
      <td>0.176378</td>
      <td>-0.052375</td>
      <td>-0.008618</td>
      <td>-0.037732</td>
      <td>-0.048709</td>
      <td>-0.022273</td>
      <td>0.001038</td>
      <td>0.007925</td>
      <td>-0.003280</td>
      <td>...</td>
      <td>0.000872</td>
      <td>0.000442</td>
      <td>0.000427</td>
      <td>0.000363</td>
      <td>0.000935</td>
      <td>0.000851</td>
      <td>0.000491</td>
      <td>0.000094</td>
      <td>0.000315</td>
      <td>0.000409</td>
    </tr>
    <tr>
      <th>wel2</th>
      <td>0.004228</td>
      <td>-0.052375</td>
      <td>0.193076</td>
      <td>-0.010004</td>
      <td>-0.032230</td>
      <td>-0.039082</td>
      <td>-0.020505</td>
      <td>0.000689</td>
      <td>0.005073</td>
      <td>-0.001778</td>
      <td>...</td>
      <td>0.000942</td>
      <td>0.000499</td>
      <td>0.000489</td>
      <td>0.000407</td>
      <td>0.000726</td>
      <td>0.000772</td>
      <td>0.000527</td>
      <td>0.000167</td>
      <td>0.000359</td>
      <td>0.000477</td>
    </tr>
    <tr>
      <th>wel4</th>
      <td>0.000588</td>
      <td>-0.008618</td>
      <td>-0.010004</td>
      <td>0.213220</td>
      <td>-0.014696</td>
      <td>-0.012287</td>
      <td>-0.026425</td>
      <td>-0.000075</td>
      <td>-0.017551</td>
      <td>-0.004137</td>
      <td>...</td>
      <td>0.000140</td>
      <td>-0.000374</td>
      <td>-0.000394</td>
      <td>-0.000244</td>
      <td>0.000513</td>
      <td>0.000774</td>
      <td>-0.000210</td>
      <td>-0.000294</td>
      <td>-0.000281</td>
      <td>-0.000399</td>
    </tr>
    <tr>
      <th>wel5</th>
      <td>0.002349</td>
      <td>-0.037732</td>
      <td>-0.032230</td>
      <td>-0.014696</td>
      <td>0.210984</td>
      <td>-0.032978</td>
      <td>-0.023670</td>
      <td>0.000649</td>
      <td>-0.000570</td>
      <td>-0.003115</td>
      <td>...</td>
      <td>0.000668</td>
      <td>0.000037</td>
      <td>0.000014</td>
      <td>0.000024</td>
      <td>0.000546</td>
      <td>0.000647</td>
      <td>0.000115</td>
      <td>-0.000110</td>
      <td>-0.000021</td>
      <td>-0.000002</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 65 columns</p>
</div>



But...is calibration worth pursuing or not? Let's explore what the notional calibration is expected to do for parameter uncertainty. We accomplish this by comparing prior and posterior parameter uncertainty. Using `.get_parameter_summary()` makes this easy:


```python
df = sc.get_parameter_summary()
df.head()
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
      <th>prior_var</th>
      <th>post_var</th>
      <th>percent_reduction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>strinf</th>
      <td>0.250000</td>
      <td>0.006971</td>
      <td>97.211534</td>
    </tr>
    <tr>
      <th>wel0</th>
      <td>0.238691</td>
      <td>0.176378</td>
      <td>26.106274</td>
    </tr>
    <tr>
      <th>wel2</th>
      <td>0.238691</td>
      <td>0.193076</td>
      <td>19.110462</td>
    </tr>
    <tr>
      <th>wel4</th>
      <td>0.238691</td>
      <td>0.213220</td>
      <td>10.671466</td>
    </tr>
    <tr>
      <th>wel5</th>
      <td>0.238691</td>
      <td>0.210984</td>
      <td>11.607882</td>
    </tr>
  </tbody>
</table>
</div>



We can plot that up:


```python
df.percent_reduction.plot(kind="bar", figsize=(15,3));
```


    
![png](intro_to_fosm_files/intro_to_fosm_50_0.png)
    


### Do these results make sense?  Why are some parameters unaffected by calibration?

As the name suggests, the `percent_reduction` column shows the  percentage decrease in uncertainty expected through calibration for each parameter.

From the plot above we can see that calibrating the model with available data definitely reduces uncertainty of some parameters. Some parameters are informed by observation data...however calibration does not affect all parameters equally. Available observation data does not contain information that affects these parameters. Calibration will not help us reduce their uncertainty.

##  Forecast Uncertainty

So far we have seen that some parameter uncertainty will be reduced. Uncertainty for other parameters will not. That's great and all, but what we really care about are our forecast uncertainties. Do the parameters that are informed by calibration affect the forecast of interest? And will calibrating reduce the uncertainty of these forecast?

Let's examine the prior and posterior variance of our forecasts. Recall that they are recorded as observations in the `Pst` control file and also listed in the pest++ `forecast` control variable:


```python
forecasts = sc.pst.forecast_names
forecasts
```




    ['headwater:4383.5', 'tailwater:4383.5', 'trgw-0-9-1:4383.5', 'part_time']



As before, `pyemu` has already done much of the heavy-lifting. We can get a summary of the forecast prior and posterior variances with `.get_forecast_summary()`:


```python
df = sc.get_forecast_summary()
df
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
      <th>prior_var</th>
      <th>post_var</th>
      <th>percent_reduction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>headwater:4383.5</th>
      <td>1.840786e+04</td>
      <td>1.043837e+04</td>
      <td>43.293944</td>
    </tr>
    <tr>
      <th>tailwater:4383.5</th>
      <td>1.854221e+04</td>
      <td>7.894541e+03</td>
      <td>57.423952</td>
    </tr>
    <tr>
      <th>trgw-0-9-1:4383.5</th>
      <td>1.106435e+01</td>
      <td>4.617277e+00</td>
      <td>58.268859</td>
    </tr>
    <tr>
      <th>part_time</th>
      <td>8.931397e+07</td>
      <td>7.014413e+07</td>
      <td>21.463434</td>
    </tr>
  </tbody>
</table>
</div>



And we can make a cheeky little plot of that. As you can see, unsurprisingly some forecasts benefit more from calibration than others. So, depending on the forecast of interest, calibration may or may not be worthwhile...


```python
# get the forecast summary then make a bar chart of the percent_reduction column
fig = plt.figure()
ax = plt.subplot(111)
ax = df.percent_reduction.plot(kind='bar',ax=ax,grid=True)
ax.set_ylabel("percent uncertainy\nreduction from calibration")
ax.set_xlabel("forecast")
```




    Text(0.5, 0, 'forecast')




    
![png](intro_to_fosm_files/intro_to_fosm_57_1.png)
    


## Parameter contribution to forecast uncertainty

Information flows from observations to parameters and then out to forecasts. Information contained in observation data constrains parameter uncertainty, which in turn constrains forecast uncertainty. For a given forecast, we can evaluate which parameter contributes the most to uncertainty. This is accomplished by assuming a parameter (or group of parameters) is perfectly known and then assessing forecast uncertainty under that assumption. Comparing uncertainty obtained in this manner, to the forecast uncertainty under the base assumption (in which no parameter is perfectly known), the contribution from that parameter (or parameter group) is obtained. 

Now, this is a pretty big assumption - in practice a parameter is never perfectly known. Nevertheless, this metric can provide useful insights into the flow of information from data to forecast uncertainty, which can help guide data assimilation design as well as future data collection efforts. 

In `pyemu` we can  evaluate parameter contributions to forecast uncertainty with groups of parameters by type using `.get_par_group_contribution()`:


```python
par_contrib = sc.get_par_group_contribution()
par_contrib.head()
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
      <th>headwater:4383.5</th>
      <th>tailwater:4383.5</th>
      <th>trgw-0-9-1:4383.5</th>
      <th>part_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>base</th>
      <td>10438.368644</td>
      <td>7894.540701</td>
      <td>4.617277</td>
      <td>7.014413e+07</td>
    </tr>
    <tr>
      <th>hk1</th>
      <td>544.264823</td>
      <td>191.279078</td>
      <td>0.022420</td>
      <td>1.243777e+06</td>
    </tr>
    <tr>
      <th>rchpp</th>
      <td>8897.659624</td>
      <td>7731.591543</td>
      <td>4.578167</td>
      <td>6.915516e+07</td>
    </tr>
    <tr>
      <th>strinf</th>
      <td>10046.303913</td>
      <td>7333.006237</td>
      <td>4.490364</td>
      <td>7.014090e+07</td>
    </tr>
    <tr>
      <th>wel</th>
      <td>10280.959926</td>
      <td>7838.238188</td>
      <td>4.566155</td>
      <td>6.994418e+07</td>
    </tr>
  </tbody>
</table>
</div>



We can see the relative contribution by normalizing to the base case (e.g. in which no parameters/groups are perfectly known):


```python
base = par_contrib.loc["base",:]
par_contrib = 100.0 * (base - par_contrib) / base
par_contrib.sort_index().head()
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
      <th>headwater:4383.5</th>
      <th>tailwater:4383.5</th>
      <th>trgw-0-9-1:4383.5</th>
      <th>part_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>base</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>hk1</th>
      <td>94.785921</td>
      <td>97.577071</td>
      <td>99.514439</td>
      <td>98.226827</td>
    </tr>
    <tr>
      <th>rchpp</th>
      <td>14.760056</td>
      <td>2.064074</td>
      <td>0.847050</td>
      <td>1.409910</td>
    </tr>
    <tr>
      <th>strinf</th>
      <td>3.755996</td>
      <td>7.112947</td>
      <td>2.748656</td>
      <td>0.004603</td>
    </tr>
    <tr>
      <th>wel</th>
      <td>1.507982</td>
      <td>0.713183</td>
      <td>1.107201</td>
      <td>0.285045</td>
    </tr>
  </tbody>
</table>
</div>



Understanding the links between parameters and forecast uncertainties can be useful - in particular to gain insight into the system dynamics. But we are still missing a step to understand what _observation_ data affects the forecast. It is often more straightforward to quantify how observation information impacts forecast uncertainty so that we can explore the worth of observation data directly.

# Data worth analysis

> __Note__: We will _not_ demonstrate data worth analysis here. See the respective notebook in Part2 of these tutorials.

The worth of data is measured by their ability to reduce the uncertainties of model predictions that we care about. Linear analysis is particularly useful for exploring data worth. This is because the equations that it uses to calculate predictive uncertainty do not include terms that represent the actual values of observations or of parameters; only sensitivities of model outputs to parameters are required. Therefore, linear analysis can be used to assess the ability (or otherwise) of yet-ungathered data to reduce the uncertainties of decision-critical predictions.

### <center> This is __Huge__. Let me say it again.<center>

<center>  We can assess the relative worth of an observation without knowing the value of the observation. </center>


This means that potential field measurements that correspond to one or many outputs of a model can be assessed for their worth. For example, it is possible to assess the worth of observations of head in every single model cell at every time step of a model run with a relatively small computational burden. This makes linear analysis a useful tool for designing and comparing strategies for data-collection, when data acquisition seeks to reduce the uncertainties of one or a number of decision-critical predictions. 

There are two main applications for data worth analysis:
 1.	ranking of the relative worth of existing observations by calculating predictive uncertainty with selected individual or combined observations removed from a calibration dataset. 
 2.	ranking of the relative worth of __potential__ new observations by calculating predictive uncertainty with selected individual or combined observations added to an existing calibration dataset.
