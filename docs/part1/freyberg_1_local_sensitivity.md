---
layout: default
title: Local Sensitivity and Identifiability
parent: Introduction to Theory, Concepts and PEST Mechanic
nav_order: 11
math: mathjax3
---

# Local Parameter Sensitivity and Identifiability

Sensitivity analysis maks use of a Joacoban matrix to determine statistical insights into a model. We have already discussed the Jacobian matrix in a few places. It is calculated by perturbing the parameter (usually 1%) and tracking what happens to each observation.  In a general form the sensitivity equation looks like eq. 9.7 Anderson et al. 2015:

<img src="freyberg_1_local_sensitivity_files/Sensitivity_eq.png" style="float: center">

This is key for derivative-based parameter estimation because, as we've seen, it allows us to efficiently compute upgraded parameters to try during the lambda search.  But the Jacobian matrix can give us insight about the model in and of itself.  

### Parameter Sensitivty

Recall that a Jacobian matrix stores parameter-to-observation sensitivities.  For each parameter-observation combination, we can see how much the observation value changes due to a small change in the parameter. If $y$ are the observations and $x$ are the parameters, the equation for the $i^th$ observation with respect to the $j^th$ parameter is:  
$$\frac{\partial y_i}{\partial x_j}$$
This can be approximated by finite differences as :  
$$\frac{\partial y_i}{\partial x_j}~\frac{y\left(x+\Delta x \right)-y\left(x\right)}{\Delta x}$$

___Insensitive parameters___ (i.e. parameters which are not informed by calibration) are defined as those which have sensitivity coefficients larger than a modeller specified value (Note: this is subjective! In practice, insensitive parameters are usually defined has having a sensitiveity coeficient two orders of magnitude lower than the most sensitive parameter.)


### Parameter Identifiability
Sensitivity analyses can mask other artifacts that affect calibration and uncertainty. A primary issues is correlation between parameters.  For example, we saw that in a heads-only calibration we can't estimate both recharge and hydraulic conductivity independently - the parameters are correlated so that an increase in one can be offset with an increase in the other.  To address this shortcoming, Doherty and Hunt (2009) show that singular value decomposition can extend the sensitivity insight into __*parameter identifiability*__.  Parameter identifiability combines parameter insensitivity and correlation information, and reflects the robustness with which particular parameter values in a model might be calibrated. That is, an identifiable parameter is both sensitive and relatively uncorrelated and thus is more likely to be estimated (identified) than an insensitive and/or correlated parameter. 

Parameter identifiability is considered a "linear method" in that it assumes the Jacobian matrix sensitivities hold over a range of reasonable parameter values.  It is able to address parameter correlation through singular value decomposition (SVD), exactly as we've seen earlier in this course.  Parameter identifiability ranges from 0 (perfectly unidentifiable with the observations available) to 1.0 (fully identifiable). So, we typically plot identifiability using a stacked bar chart which is comprised of the included singular value contributions. Another way to think of it: if a parameter is strongly in the SVD solution space (low singular value so above the cutoff) it will have a higher identifiability. However, as Doherty and Hunt (2009) point out, identifiability is qualitative in nature because the singular value cutoff is user specified.  

### Limitations

As has been mentioned a couple of times now, sensitivity and identifiability assumes that the relation between parameter and obsevration changes are linear. These sensitivities are tested for a single parameter set (i.e. the parameter values listed in the PEST control file). This parameter set can be before or after calibration (or any where in between). The undelying assumption is that the linear relation holds. 

In practice, this is rarely the case. Thus, sensitivities obtained during derivative-based parameter estimation is ___local___. It only holds up in the vicinity of the current parameters. That being said, it is a quick and computationaly efficient method to gain insight into the model and links between parameters and simualted outputs. 

An alternative is to employ _global_ sensitivity analysis methods, which we introduce in a subsequent notebook.

> __References:__
>
> - Doherty, John, and Randall J. Hunt. 2009. “Two Statistics for Evaluating Parameter Identifiability and Error Reduction.” Journal of Hydrology 366 (1–4): 119–27. doi:10.1016/j.jhydrol.2008.12.018.
> - Anderson, Mary P., William W. Woessner, and Randall J. Hunt. 2015. Applied Groundwater Modeling: Simulation of Flow and Advective Transport. Applied Groundwater Modeling. 2nd ed. Elsevier. https://linkinghub.elsevier.com/retrieve/pii/B9780080916385000018.

## Admin

We are going to pick up where the "Freyberg pilot points" tutorials left off. The cells bellow prepare the model and PEST files.


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

### Load the `pst` control file

Let's double check what parameters we have in this version of the model using `pyemu` (you can just look in the PEST control file too.).


```python
pst_name = "freyberg_pp.pst"
# load the pst
pst = pyemu.Pst(os.path.join(working_dir,pst_name))
# what parameter groups?
pst.par_groups
```

Which are adjustable?


```python
# what adjustable parameter groups?
pst.adj_par_groups
```

We have adjustable parameters that control SFR inflow rates, well pumping rates, hydraulic conductivity and recharge rates. Recall that by setting a parameter as "fixed" we are stating that we know it perfectly (should we though...?). Currently fixed parameters include porosity and future recharge.

For the sake of this tutorial, let's set all the parameters free:


```python
par = pst.parameter_data
#update paramter transform
par.loc[:, 'partrans'] = 'log'
#check adjustable parameter groups again
pst.adj_par_groups
```

## Calculate the Jacobian

First Let's calculate a single Jacobian by changing the NOPTMAX = -2.  This will need npar+1 runs. The Jacobian matrix we get is the local-scale sensitivity information


```python
# change noptmax
pst.control_data.noptmax = -2
# rewrite the contorl file!
pst.write(os.path.join(working_dir,pst_name))
```


```python
num_workers = 10
m_d = 'master_local'
pyemu.os_utils.start_workers(working_dir, # the folder which contains the "template" PEST dataset
                            'pestpp-glm', #the PEST software version we want to run
                            pst_name, # the control file to use with PEST
                            num_workers=num_workers, #how many agents to deploy
                            worker_root='.', #where to deploy the agent directories; relative to where python is running
                            master_dir=m_d, #the manager directory
                            )
```

# Sensitivity

Okay, let's examing the *local sensitivities* by looking at the local gradients of parameters with respect to observations (the Jacobian matrix from the PEST++ NOPTMAX = -2 run)

We'll use `pyemu` to do this:


```python
#read the jacoban matrix
jco = pyemu.Jco.from_binary(os.path.join(m_d,pst_name.replace(".pst",".jcb")))
jco_df = jco.to_dataframe()
# inspect the matrix as a dataframe
jco_df = jco_df.loc[pst.nnz_obs_names,:]
jco_df.head()
```

We can see that some parameters (e.g. `rch0`) have a large effect on the observations used for calibration.  The future recharge (`rch1`) has no effect on the calibration observations, but that makes sense as none of the calibration observations are in that future stress period!

## How about Composite Scaled Sensitivities
As can be seen above, parameter sensitivity for any given parameter is split among all the observations in the Jacobian matrix, but the parameter sensitivity that is most important for parameter estimation is the *total* parameter sensitivity, which reflects contributions from all the observations.  

How to sum the individual sensitivities in the Jacobian matrix in the most meaningful way?  In the traditional, overdetermined regression world, CSS was a popular metric. CSS is Composite Scaled Sensitivitity. It sums the observation *weighted* sensitivity to report a single number for each parameter.

In Hill and Tiedeman (2007) this is calculated as: 
$${css_{j}=\sqrt{\left(\sum_{i-1}^{ND}\left(\frac{\partial y'_{i}}{\partial b_{j}}\right)\left|b_{j}\right|\sqrt{w_{ii}}\right)/ND}}$$

In PEST and PEST++, it is calculated slightly differently in that scaling by the parameter values happens automatically when the parameter is subjected to a log-transform (and we can see above that all our parameters are logged). This is due to a correction that must be made in calculating the Jacobian matrix and follows from the chain rule of derivatives.  Seems somewhat academic, but let's compare the two.

Let's instantiate a `Schur` object (see the "intro to fosm" tutorials) to calculate the sensitivities.


```python
# instantiate schur
sc = pyemu.Schur(jco=os.path.join(m_d,pst_name.replace(".pst",".jcb")))

# calcualte the parameter CSS
css_df = sc.get_par_css_dataframe()
css_df.sort_values(by='pest_css', ascending=False).plot(kind='bar', figsize=(13,3))
plt.yscale('log')
```

Note that the reltive ranks of the `hk` parameters agree between the two...but no so for the `rchpp` ilot point parameters, nor the `rch0` parameter. According to `pest_css` the `rch0` is the most sensitive. Not so for the `hill_css`.  Why might this be?

> hint: what is the initial value of `rch0`?  What is the log of that initial value?  

###  Okay, let's look at just the PEST CSS and rank/plot it:

What does this show us? It shows which parameters have the greatest effect on the _objective function_. In other words,  sensitive parameters are those which are informed by observation data. They will be affected by calibration.

Parameters which are _insensitive_ (e.g. `rch1` and `ne1`), are not affected by calibration. Why? Because we have no observation data that affects them. Makes sense: `rch1` is recharge in the future, `ne1` is not informed by head or flow data - the only observations we have.


```python
plt.figure(figsize=(12,3))
ax = css_df['pest_css'].sort_values(ascending=False).plot(kind='bar')
ax.set_yscale('log')
```

## So how do these parameter sensitivities affect the forecasts?  

Recall that the sensitivity is calculated by differencing the two model outputs, so any model output can have a sensitivity calculated even if we don't have a measured value.  So, because we included the forecasts as observations we have sensitivities for them in our Jacobian matrix.  Let's use `pyemu` to pull just these forecasts!


```python
jco_fore_df = sc.forecasts.to_dataframe()
jco_fore_df.head()
```

Note that porosity is 0.0, except for the travel time forecast (`part_time`), which makes sense.  

Perhaps less obvious is `rch0` - why does it have sensitivity when all the forecasts are in the period that has `rch1` recharge? Well what happened in the past will affect the future...

### Consider posterior covariance and parameter correlation

Again, use `pyemu` to construct a posterior parameter covariance matrix:


```python
covar = pyemu.Cov(sc.xtqx.x, names=sc.xtqx.row_names)
covar.df().head()
```

For covariance, very small numbers reflect that the parameter doesn't covary with another.  (Does it make sense that `rch1` does not covary with other parameters?)


### We can visualize the correlation betwen the two parameters using a correlation coefficient


```python
R = covar.to_pearson()
plt.imshow(R.df(), interpolation='nearest', cmap='viridis')
plt.colorbar()
```

As expected, the parameters are correlated perfectly to themselves (1.0 along the yellow diagonal) buth they also can have appreciable correlation to each other, both positively and negatively.

#### Inspect correlation for a single parameter

Using pilot point `hk_i:12_j:12_zone:1.0`, let's look only at the parameters that have correlation > 0.5


```python
cpar = 'hk_i:12_j:12_zone:1.0'
R.df().loc[cpar][np.abs(R.df().loc[cpar])>.5]
```

Saying parameters are correlated is really saying that when a parameter changes it has a similar effect on the observations as the other parameter(s). So in this case that means that when `hk_i:12_j:12_zone:1.0` increases it has a similar effect on observations as increasing `hk_i:2_j:12_zone:1.0`.  If we add a new observation type (or less powerfully, an observation at a new location) we can break the correlation.  And we've seen this:  adding a flux observation broke the correlation between R and K!

We can use this `pyemu` picture to interrogate the correlation - here we say plot this but cut out all that correlations under 0.5.  Play with this by putting other numbers between 0.3 and 1.0 and re-run the block below.


```python
R_plot = R.as_2d.copy()
R_plot[np.abs(R_plot)>0.5] = np.nan
plt.imshow(R_plot, interpolation='nearest', cmap='viridis')
plt.colorbar();
```

In practice, correlation >0.95 or so becomes a problem for obtainning a unique solution to the parameter estimation problem. (A problem which can be avoided with regularization.)

# Identifiability

Parameter identifiability combines parameter insensitivity and correlation information, and reflects the robustness with which particular parameter values in a model might be calibrated. That is, an identifiable parameter is both sensitive and relatively uncorrelated and thus is more likely to be estimated (identified) than an insensitive and/or correlated parameter. 

One last cool concept about identifiability the Doherty and Hunt (2009) point out: Because parameter identifiability uses the Jacobian matrix it is the *sensitivity* that matters, not the actual value specified. This means you can enter *hypothetical observations* to the existing observations, re-run the Jacobian matrix, and then re-plot identifiability. In this way identifiability becomes a quick but qualitative way to look at the worth of future data collection - an underused aspect of our modeling!   

To look at identifiability we will need to create an `ErrVar` object in `pyemu`


```python
ev = pyemu.ErrVar(jco=os.path.join(m_d,pst_name.replace(".pst",".jcb")))
```

We can get a dataframe of identifiability for any singular value cutoff (`singular_value` in the cell below). (recall that the minimum number of singular values will the number of non-zero observations; SVD regularization)


```python
pst.nnz_obs
```

Try playing around with `singular_value` in the cell below:


```python
singular_value= 37

id_df = ev.get_identifiability_dataframe(singular_value=singular_value).sort_values(by='ident', ascending=False)
id_df.head()
```

It's easy to  visualize parameter _identifiability_  as stacked bar charts. Here we are looking at the identifiability with `singular_value` number of singular vectors:


```python
id = pyemu.plot_utils.plot_id_bar(id_df, figsize=(14,4))
```

However, it can be more meaningful to look at a singular value cutoff:


```python
id = pyemu.plot_utils.plot_id_bar(id_df, nsv=10, figsize=(14,4))
```

# Display Spatially

It can be usefull to display sensitivities or identifiabilities spatially. Pragmatically this can be acomplished by assigning sensitivity/identifiability values to pilot points (as if they were parameter values) and then interpolating to the model grid.

The next cells do this in the background for the `hk1` pilot parameter identifiability.


```python
# get the identifiability values of rthe hk1 pilot points
hkpp_parnames = par.loc[par.pargp=='hk1'].parnme.tolist()
ident_vals = id_df.loc[ hkpp_parnames, 'ident'].values
```


```python
# use the conveninec function to interpolate to and then plot on the model grid
hbd.plot_arr2grid(ident_vals, working_dir)
```

We can do the same for sensitivity:


```python
# and the same for sensitivity CSS
css_hk = css_df.loc[ hkpp_parnames, 'pest_css'].values
# use the conveninec function to interpolate to and then plot on the model grid
hbd.plot_arr2grid(css_hk, working_dir, title='Sensitivity')
```

# So what?

So what is this usefull for? Well if we have a very long-running model and many adjstable parameters - _and if we really need to use derivative-based parameter estimation methods (i.e. PEST++GLM)_ - we could now identify parameters that can be fixed and/or omitted during parameter estimation (but not uncertainty analysis!). 

Looking at forecast-to-parameter sensitivities can also inform us of which parameters matter for our forecast. If a a parameter doesnt affect a forecast, then we are less concerned with it. If it does...then we may wish to give it more attention. These methods provide usefull tools for assessing details of model construction and their impact on decision-support forecasts of interest (e.g. assessing whether a boundary condition affects a forecast).


But! As has been mentioned - these methods are _local_ and assume a linear relation between parameter and observation changes. As we have seen, this is usualy not the case. A more robust measure of parameter sensitivity requires the use of global sensitivity analysis methods, discussed in the next tutorial.
