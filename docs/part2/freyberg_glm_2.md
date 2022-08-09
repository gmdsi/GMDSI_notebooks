
---
layout: default
title: PEST++GLM  - Highly-Parameterized Regularized Inversion
parent: Decision Support Modelling with pyEMU and PEST++
nav_order: 6
---
                    ---
layout: default
title: PEST++GLM  - Highly-Parameterized Regularized Inversion
parent: Decision Support Modelling with pyEMU and PEST++
nav_order: 6
math: mathjax3
---

# Highly Parameterized Inversion

The current tutorial continues where the "freyberg glm part1" notebook left off. The "freyberg intro to model" and "freyberg pstfrom pest setup" provide details on the model and PEST dataset. The "freyberg glm 1" notebooks introduced changes to the PEST setup that are relevant to the current tutorial. You do not need to know the details of all these noteboks to follow along with general points of the current tutorial - but it helps! 

In this tutorial we will add the final tweaks to a PEST dataset and then calibrate our model using PESTPP-GLM. Much like PEST, PESTPP-GLM undertakes highly parameterized inversion. However, it streamlines alot of the user-input process. It also automates FOSM and FOSM-based Monte Carlo uncertainty analyses. This drastically reduces requirements for user input, making workflows easier to implement. However, it also increases the number of moving pieces that a user must be familiar with (no free lunch!).

Here we will discuss some PESTPP-GLM specific options and their implications, calibrate the model and then explore outputs - all using a programatic interface. 

### Admin

Start off with the usual loading of dependencies and preparing model and PEST files. We will be continuing to work with the MODFLOW6 modified-Freyberg model (see "freyberg intro to model" notebook), and the high-dimensional PEST dataset prepared in the "freyberg glm 1" notebook. For the purposes of this notebook, you do not require familiarity with previous notebooks (but it helps...). 

Simply run the next few cells by pressing `shift+enter`.


```python
import os
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt;
import shutil 
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

Specify the path to the PEST dataset template folder:


```python
# specify the temporary working folder
t_d = os.path.join('freyberg6_template')
```

Copy across pre-prepared model and PEST files:


```python
org_t_d = os.path.join("master_glm_1")
if not os.path.exists(org_t_d):
    raise Exception("you need to run the '/part2_4_glm/freyberg_glm_1.ipynb' notebook")

if os.path.exists(t_d):
    shutil.rmtree(t_d)
shutil.copytree(org_t_d,t_d)
```




    'freyberg6_template'




```python
pst_path = os.path.join(t_d, 'freyberg_pp.pst')
```

Right then, let's load in our PEST control file.

Just as a reminder, this is the control fle prepared in the "glm 1" notebook:
 - We have reduced our very high dimensional PEST dataset down from 10s of thousands to several hundreds of parameters (#sad).
 - We continue to have lots of spatially distributed parameters (pilot points), which we expect to have a degree of spatial (and in some cases, temporal) correlation. 
 - We have weighted observations to reflect the inverse of measurement noise standard deviation. 
 - We have already calculated a Jacobian matrix, with parameter sensitivities based on intial parameter values. It is stored in a file named "freyberg_pp.jcb".



```python
pst = pyemu.Pst(pst_path)
```


```python
# check to see if obs&weights notebook has been run
if not pst.observation_data.observed.sum()>0:
    raise Exception("You need to run the '/part2_2_obs_and_weights/freyberg_obs_and_weights.ipynb' notebook")
```


```python
print(f'number of observations: {pst.nnz_obs} \nnumber of parameters: {pst.npar_adj}')
```

    number of observations: 72 
    number of parameters: 245
    

## Regularisation

We have more unkown parameters than observations. Thus, we have an ill-posed inverse problem. The mathematical term for the process through which a unique solution is sought for a nonunique inverse problem is “regularisation”. The goal of regularised inversion is to seek a unique parameter field that results in a suitable fit between model outputs and field measurements, whilst minimizing the potential for wrongness in model predictions. That is, out of all the ways to fit a calibration dataset, regularized inversion seeks the parameter set of minimum error variance.

There are two main approaches used to stabilize this problem: 
 1) adding soft knowledge and/or 
 2) reducing the problem dimensionality. 

These methods can be used by themselves, but are most commonly applied together.

#### Tikhonov Regularisation (e.g. "soft knowledge")

One way to seek a parameter field of minimum error variance is to seek a parameter field that allows the model to fit the calibration dataset, but whose values are also as close as possible to a set of “preferred parameter values”. Ideally, preferred parameter values should also be initial parameter values as listed in the “parameter data” section of the PEST control file. These preferred parameter values are normally close to the centre of the prior parameter probability distribution. At the same time, scales and patterns of departures from the preferred parameter set that are required for model outputs to fit a calibration dataset should be achieved in ways that make “geological sense”.

PEST provides a user with a great deal of flexibility in how Tikhonov constraints can be introduced to an inversion process. The easiest way is to do this is through the use of prior information equations. When prior information equations are employed, Tikhonov constraints are expressed through preferred values that are assigned to linear relationships between parameters. (Equality is the simplest type of linear relationship.) Weights must be assigned to these equations. As is described in PEST documentation, when PEST is run in “regularisation” mode, it makes internal adjustments to the weights that are assigned to any observations or prior information equations that belong to special observation groups that are referred to as “regularisation groups”. 


#### Promoting "Geologicaly Reasonable" Patterns

PEST (and PEST_HP) provide the option to replace prior information equation weights with covariance matrices. When prior information equations embody Tikhonov constraints, they can be used to ensure that patterns of parameter heterogeneity that emerge from the inversion process are geologically reasonable. (See the GMDSI Tutorials for examples of how to accompish this with PEST_HP: https://gmdsi.org/blog/calibration-a-simple-model/)

PEST++GLM (which we will be using here) does things a bit differently. You *can* specify prior information equations and assign relevant weights and run PEST++GLM in "regularisation" mode. (See the "intro to pyemu" notebook for an example of how to specify prior information equations.) However, PEST++GLM does not accept covariance matrices for the purposes of specifying preferred parameter heterogeneity.

Alternatively, PEST++GLM offers users the option of using prior parameter covariance matrix based regularisation directly in the parameter upgrade calculation process - referred to as Regularized-Gauss-Levenburg-Marquardt. Instead of using the contribution of regularisation to the objective function to avoid "unrealistic" parameter patterns (as is done in PEST), this approach tries to maintain "realistic" patterns directly when calculating parameter upgrades. 

As we will show further on, this is implemented by specifying a prior parameter covariance matrix and the relevant `pest++` options. If this option is enabled, then prior information equation based regularisation cannot also be employed. Regularisation is controlled by the max singular values and eigen threshold control variables in the SVD section.

Pragmatically, the latter approach is the easier one to implement within a PEST++ workflow. A modeller will likley have already prepared a prior parameter covariance marix (we have uncertianty on our mind after all...). Thus, only a few lines need to be added to the PEST control file. It also removes the need to determine and dynamically adjust prior information weights, making the process numerically more efficient. 

The Regularized-Gauss-Levenburg-Marquardt option is activated with the `glm_normal_form(prior)` pest++ option and by providing a prior parameter covariance matrix in `parcov()`.

Let's start by preparing this covariance matrix. Recall that we created a prior covariance matrix in the "freyberg pstfrom pest setup" tutorial and saved it in a file named "prior_cov.jcb". That matrix contains values for *all* the parameters. First, we need to reduce it down to only the adjustable parameters in our current "pilot points" control file:


```python
# read the orginal high-dimensional covariance matrix
cov = pyemu.Cov.from_binary(os.path.join(t_d,"prior_cov.jcb"))

# reduce it down to the currently adjustable parameters
cov = cov.get(pst.adj_par_names)

# write the reduced covariance matrix to an external file
cov.to_ascii(os.path.join(t_d,"glm_prior.cov"))
```

Now that the covariance matrix is ready, let's add the pest++ control variables to the `pst` control file.

Start by providing the control variable specifying the filename for the prior parameter covariance matrix. This matrix is used for both regularisation and during FOSM:


```python
# specify the prior parameter covariance matrix file
pst.pestpp_options["parcov"] = "glm_prior.cov"
```

Then specify the option to employ Regularized-Gauss-Levenburg-Marquardt:


```python
# activate the regularized-GLM solution
pst.pestpp_options["glm_normal_form"] = "prior"
```

Boom! Done.

###  SVD

Tikhonov regularisation adds information to the calibration process in order to achieve numerical stability. In contrast, subspace methods do so by reducing the dimensionality of the problem, removing and/or combining prameters. When employing SVD in calibration, only parameters and linear combinations of parameters that are suficiently constrained by measured data are estimated. These parameters are said to reside in the *solution space*. Chossing which parameter combinations to estimate is accomplished via singular value decomposition (SVD). SVD-based parameter estimation fixes intial values for parameters/parameter combinations that are not estimable (reside in the *null space*) and does not adjust them during inversion. (So make sure initial parameter values are sensible!)  

Unlike PEST, by default members of the PEST++ suite employ singular value decomposition (or methods closely related to it) for solution of the inverse problem. Unless otherwise specifed, default options are employed. PEST++GLm offers two numerical libraries for implementing SVD; the default option will usually suffice (see the manual for details).

Two variables affect how SVD is deployed: MAXSING and EIGTHRESH. These are recorded in the `* singular value decomposition` section of the PEST control file. By default, MAXSING is equal to the number of adjustable parameters and EIGHTRESH is 1.0E-7. 

When employing Regularized-Gauss-Levenburg-Marquardt (i.e. if the `glm_normal_form(prior)` option is specified, as we have done here), the values of MAXSING and EIGTHRESH control the degree of regularisation. 

For the purposes of this tutorial, we will use default values. However, if desired, these variables could be assigned thus:


```python
# MAXSING
pst.svd_data.maxsing = pst.npar_adj

#EIGTHRESH
pst.svd_data.eigthresh = 1e-7
```

###  SVD-assist

Use of PEST’s “SVD-assist” methodology can promulgate significant increases in the numerical efficiency of highly parameterized inversion. In implementing this methodology, PEST estimates the values of so-called “super parameters” in place of the parameters themselves. 

Estimating super parameters can reduce the computational cost of highly parameterized inversion considerably. However, a Jacobian matrix based on the full parameter set must be calculated before the super parameter inversion process.

We have already done so in the "freyberg glm 1" tutorial. We can now make use of the pre-calculated Jacobian matrix to save time. We specify this with the `base_jacobian` pest++ option. Note that this is **not** an SVD-Assist specific option. It can be used any time.


```python
# make a copy of the Jacobian file with a different name
shutil.copy2(os.path.join(t_d,"freyberg_pp.jcb"),
             os.path.join(t_d,"freyberg_reuse.jcb"))
```




    'freyberg6_template\\freyberg_reuse.jcb'




```python
# specify file to use as the base jacobian
pst.pestpp_options["base_jacobian"] = "freyberg_reuse.jcb"
```

Unfortunately, the large computational savings gained from SVD-assisted inversion come with a number of costs. Chief among these is that, for a nonlinear model, the partitioning of parameter space into solution and null spaces based on singular decomposition of a full Jacobian matrix calculated on the basis of initial parameter values may not remain valid as parameter values change. Hence, as the super parameter inversion process progresses, it may become necessary to recalculate a full Jacobian matrix so that super parameters can be re-defined. 

For moderately to highly nonlinear models, super parameter redefinition may be required every few iterations. With intermittent super parameter redefinition, model run savings accrued through SVD-assisted inversion may still be considerable; however, they will not be as great as for a linear model where re-computation of a full Jacobian matrix is never required.

In PEST++GLM, the `n_iter_base` and `n_iter_super` pestpp options determine the number of sequential base and super parameter iterations, respectively. Choosing values for these variables will be case specific and may require some trial and error. Relatively linear problems may work well a single base iteration. Non linear problems will likley benefit from fewer super iterations, interspersed with base iterations. Ideal values may also depend on the maximum number of super parameters (discussed further on).

For our case we have found that a single base parameter iteration, followed by a few super iterations is sufficient. 


```python
# for convenience, specify a maximum number of iterations. Avoids wasting cpu-time for the purposes of the tutorial; in the real-world you will probably use more than this
pst.control_data.noptmax = 3

# specify the number of base parameter iterations; 
# by passing a value of -1, PEST++GLM will calculate a single iteration and then switch to super parameter iterations
pst.pestpp_options["n_iter_base"] = -1

# specify the number of subsequent super parameter iterations; set equal to NOPTMAX
pst.pestpp_options["n_iter_super"] = pst.control_data.noptmax
```

The number of super parameters determines the number of model runs undertaken during a super parameter iteration. The number of super parameters to estimate can be set using either or both of the `max_n_super()` and ­`super_eigthresh()` pest++ control variable.  The upper limit of super parameters to form is controlled by `max_n_super()`. The higher this value is set, the greater the flexibility that PEST++GLM will have to adjust parameters and achieve a good fit with observation data. But, as we discuss quite often over the course of these tutorials, a good fit does not guarantee a good forecast. If you try to esimate more super parameters than there are valid dimensions in the calibration solution space, you run the risk of incurring numerical instability and overfitting. 

Ideally an upper limit to `max_n_super` should reflect the amount of information contained in the observation dataset (e.g. the calibration solution space). It cannot be higher than the number of non-zero weighted observations. It will likely be lower. As we have previoulsy calcualted a Jacobian matrix, we can get an estimate of the dimensions of the solution space by inspecting the singular value spectrum using `pyemu.ErrVar` (or the SUPCALC utility form the PEST suite).

First create a linear analysis object.  We will use `ErrVar`  derived type, which replicates the behavior of the `PREDVAR` suite of PEST utilities. We pass it the name of the jacobian matrix file.  If we don't pass an explicit argument for `parcov` or `obscov`, `pyemu` attempts to build them from the parameter bounds and observation weights in a pest control file (.pst) with the same base case name as the jacobian. Seeing as we have a prior covariance matrix, let's pass that in explicitly. 


```python
la = pyemu.ErrVar(jco=os.path.join(t_d, "freyberg_pp.jcb"), 
                    parcov=os.path.join(t_d,"glm_prior.cov"))
print(la.jco.shape) #without the omitted parameter or the prior info
```

    (21248, 245)
    

We can inspect the singular spectrum of $\mathbf{Q}^{\frac{1}{2}}\mathbf{J}$, where $\mathbf{Q}$ is the cofactor matrix and $\mathbf{J}$ is the jacobian:


```python
s = la.qhalfx.s
```

If we plot the singular spectrum, we can see that it decays rapidly (note the y-axis is log-scaled). We can really only support about `max_sing_val` right singular vectors even though we have several hundred adjustable parameters. Should we be using mre super-parameters than that then? Doing so runs the risk of overfitting. Overfitting leads to bias. Bias leads to suffering...


```python
max_sing_val = np.argmax(s.x<1e-8)
print("Solution space dimensions: ",max_sing_val)

# plot sing sepctrum
figure = plt.figure(figsize=(10, 4))
ax = plt.subplot(111)
ax.plot(s.x)
ax.set_title("singular spectrum")
ax.set_ylabel("power")
ax.set_xlabel("singular value")
ax.set_xlim(0,pst.nnz_obs)
ax.set_yscale('log')
plt.show()
```

    Solution space dimensions:  39
    


    
![png](freyberg_glm_2_files/freyberg_glm_2_33_1.png)
    


That's already a decent amount of savings in terms of run-time! Now, for each iteration instead of running the model several hundred times (i.e. the number of adjustable base parameters; see `pst.npar_adj`) we only need to run the model a few tens of times. An order of magnitude less!

In practice, `max_n_super` may be further limited to reflect available computational resources - if you only have time/resources to run the modle 10 times...well then that is what you should set it at. (See the PEST Manual Part 1 for a pragmatic discussion of how to choose the number of super parameters.) 


```python
pst.pestpp_options["max_n_super"] = max_sing_val
```


### FOSM
 

PEST++GLM makes FOSM parameter and predictive uncertainty analysis a breeze. 

At the end of each iteration PEST++GLM implements FOSM analysis, as long as the `uncertainty()` control variable is set to `true` (it is by default). If present, it ignores regularisation and prior information equations (not present in our case). By default, prior parameter uncertainties are calculated from parameter bounds. Alternatively, a user can supply a prior parameter covariance matrix with the `parcov()` variable; this permits inclusion of prior parameter covariances. Recall that we have already specified this option: 


```python
pst.pestpp_options['parcov']
```




    'glm_prior.cov'



PEST++GLM calculates a posterior parameter covariance matrix at the end of each iteration. Each covariance matrix is recorded in an external file. PEST++GLM also provides summaries of prior and posterior parameter uncertainties (means, standard deviations and bounds) for each iteration.

If any observations are listed as forecasts (in the `forecast()` variable), PEST++GLM will also undertake predictive uncertainty analysis. By default, if no forecasts are provided, PEST++GLM will assume all zero-weighted observations are forecasts - so it is usually a good idea to specify forecasts explicitly (if you have many many zero-weighted obsevrations FOSM can cost some computation time). Recall that we specified several observations as forecasts:


```python
pst.pestpp_options['forecasts']
```




    'oname:sfr_otype:lst_usecol:tailwater_time:4383.5,oname:sfr_otype:lst_usecol:headwater_time:4383.5,oname:hds_otype:lst_usecol:trgw-0-9-1_time:4383.5,part_time'



FOSM implemented by PEST++GLM assumes that the standard deviation of measurement noise associated with each observation is proportional to current observation residual. This accounts for the model's ability to reproduce an observation. Effectively, it assumes that the residual is a measure of measurement noise + model error. Thus, observation weights used during FOSM are calculated as the inverse of the residual. Note that this "residual weight" never increases weights beyond those which are specified in the control file! The assumption is that weighs in the control file represent the inverse of measurment noise standard deviation - and it would be illogical to decrease noise beyond this level. 

It is important to keep this in mind. If observation weights in the control file do **not** represent measurement noise, then it may be preferable to not use PEST++GLM to undertake FOSM during parameter estimation. In our case, weights represent the inverse of measurment standard deviations - so we are all good!

### FOSM-informed Monte Carlo

PEST++GLM also has the ability to undertake nonlinear Monte Carlo uncertainty analysis. FOSM-based posterior Monte Carlo (also called Bayes-linear Monte Carlo) is implemented by drawing samples of parameters from the posterior parameter distribution (described by the posterior parameter covariance matrix and assuming best-fit parameter values as the mean). Each of these parameter realisation is then simulated. As long as forecasts are included as observations in the control file, then the Monte Carlo process provides an ensemble of forecast values. With enough samples of foreacast the posterior predictive uncertainty can be described. In principle, using FOSM-based Monte Carlo to evaluate forecast uncertainty relaxes the assumption of linearity between and foreacsts and parameters - making it more robust.

Activating this option is as easy as adding the `glm_num_reals()` option to the control file and specifying the number of Monte Carlo realisation to sample: 


```python
pst.pestpp_options["glm_num_reals"] = 50
```

### Extra Utility Options

There are numerous "utility" options available in PEST++GLM. The user manual provides descriptions of all of them. The following two specify options for halting PEST++GLM due to model-run failure:


```python
# consider a model to have failed if it takes 5 times the average model run time
pst.pestpp_options["overdue_giveup_fac"] = 5.0
# attemp to repeat a failed model a maximum of 3 times; if it still fails, halt PEST++
pst.pestpp_options["max_run_fail"] = 3
```

### Re-write Control File and Run PEST++GLM

Re-write the control file with the updated pestpp options.


```python
# write PST 
case = 'freyberg_pp'
pst.write(os.path.join(t_d,f"{case}.pst"))
```

    noptmax:3, npar_adj:245, nnz_obs:72
    

Now, deploy PEST++GLM in parallel.

To speed up the process, you will want to distribute the workload across as many parallel agents as possible. Normally, you will want to use the same number of agents (or less) as you have available CPU cores. Most personal computers (i.e. desktops or laptops) these days have between 4 and 10 cores. Servers or HPCs may have many more cores than this. Another limitation to keep in mind is the read/write speed of your machines disk (e.g. your hard drive). PEST and the model software are going to be reading and writting lots of files. This often slows things down if agents are competing for the same resources to read/write to disk.

The first thing we will do is specify the number of agents we are going to use.

# Attention!

You must specify the number which is adequate for ***your*** machine! Make sure to assign an appropriate value for the following `num_workers` variable. (You can check how many physical cores you have on your local machine with  `psutil`.)


```python
print(psutil.cpu_count(logical=False))
```

    10
    


```python
num_workers = 10 #update this according to your resources

m_d = os.path.join('master_glm_2')
```


```python
# run glm in parallel
pyemu.os_utils.start_workers(t_d,"pestpp-glm",f"{case}.pst",
                           num_workers=num_workers,
                           worker_root=".",
                           master_dir=m_d)
```

To see PEST++'s progress, switch to the command line window from which you launched this notebook. Wait until it has completed. This may take several minutes. Feel free to go make a cup of coffee.

## Postprocess

During inversion PEST++ records a lot of usefull information in external files. We shall not go through each of these here (see the PEST++ user manual for detailed descriptions). The following are some common outputs a modeller is likely to inspect.

Let's first re-read the control file so as to update the `.phi` and `.res` attributes.


```python
pst = pyemu.Pst(os.path.join(m_d,'freyberg_pp.pst'))
pst.phi
```




    47.79859025261253



Recall that observations are weighted according to the inverse of measurment noise. Conceptualy, we hope to achieve a fit between simulated and measured values comensurate with measurment error. As we saw in the "obs and weights" tutorial, such a fit would result in a Phi equal to the number of non-zero weighted observations. Did we achieve that? No?! Shocking! (Why not, you ask? Short answer: not enough parameters.) Model-to-measurment misfit is often dominated by model error (and by "model" we also mean the parameterisation of the model) and not just measurement noise. 


```python
print(f"Phi: {pst.phi} \nNumber of non-zero obs: {pst.nnz_obs}")
```

    Phi: 47.79859025261253 
    Number of non-zero obs: 72
    

A usefull way to track PEST's performance is to look at the evolution of Phi throughout the inversion. This is recorded in a file with the extension `*.iobj`  (e.g. `freyperg_pp.iobj`). You can read this file whilst PEST++ is working if you like. PEST++ will update the file after every iteration, thus it provides an easy way to keep an eye on the inversion progress.

The file has a tabular format, making it easy to read as a `Pandas` dataframe:


```python
# make a dataframe "df_obj" that shows the contents of the pst file casename with the extension .iobj
# .iobj = PEST++ output file that has the objective function by iteration 
df_obj = pd.read_csv(os.path.join(m_d, "freyberg_pp.iobj"),index_col=0)
# echo out the dataframe
df_obj.head()
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
      <th>model_runs_completed</th>
      <th>total_phi</th>
      <th>measurement_phi</th>
      <th>regularization_phi</th>
      <th>oname:hds_otype:lst_usecol:trgw-0-13-10</th>
      <th>oname:hds_otype:lst_usecol:trgw-0-15-16</th>
      <th>oname:hds_otype:lst_usecol:trgw-0-2-15</th>
      <th>oname:hds_otype:lst_usecol:trgw-0-2-9</th>
      <th>oname:hds_otype:lst_usecol:trgw-0-21-10</th>
      <th>oname:hds_otype:lst_usecol:trgw-0-22-15</th>
      <th>...</th>
      <th>oname:hdstd_otype:lst_usecol:trgw-0-21-10</th>
      <th>oname:hdstd_otype:lst_usecol:trgw-0-22-15</th>
      <th>oname:hdstd_otype:lst_usecol:trgw-0-24-4</th>
      <th>oname:hdstd_otype:lst_usecol:trgw-0-26-6</th>
      <th>oname:hdstd_otype:lst_usecol:trgw-0-29-15</th>
      <th>oname:hdstd_otype:lst_usecol:trgw-0-3-8</th>
      <th>oname:hdstd_otype:lst_usecol:trgw-0-33-7</th>
      <th>oname:hdstd_otype:lst_usecol:trgw-0-34-10</th>
      <th>oname:hdstd_otype:lst_usecol:trgw-0-9-1</th>
      <th>part</th>
    </tr>
    <tr>
      <th>iteration</th>
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
      <td>0</td>
      <td>13061.7000</td>
      <td>13061.7000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>206.7330</td>
      <td>0</td>
      <td>148.43200</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11</td>
      <td>656.9730</td>
      <td>656.9730</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>445.1770</td>
      <td>0</td>
      <td>98.41460</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>61</td>
      <td>47.7986</td>
      <td>47.7986</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>27.5609</td>
      <td>0</td>
      <td>5.72848</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>111</td>
      <td>118.3940</td>
      <td>118.3940</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>27.3603</td>
      <td>0</td>
      <td>17.45060</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>4 rows × 80 columns</p>
</div>



A quick and dirty plot to see the evolution of Phi per iteration. Ideally Phi should decrease for each sucessive iteration. If this plot bounces up and down it is often a sign of trouble. When using SVD-Assist, this often occurs if the inverse problem is highy nonlinear. In such cases it may be worth experimenting with fewer sucessive super parameter iterations and/or a greater number of super parameters. Lack of convergence can also be a sign of "dirty derivatives", often due to model output files which PEST is reading being written with insufficient precision.


```python
# plot out the dataframe that was shown as a table above
df_obj.loc[:,["total_phi","model_runs_completed"]].plot(subplots=True)
```




    array([<AxesSubplot:xlabel='iteration'>, <AxesSubplot:xlabel='iteration'>],
          dtype=object)




    
![png](freyberg_glm_2_files/freyberg_glm_2_61_1.png)
    


### Residuals

We may also wish to compare the measured versus simulated observation values obtained using the "best-fit" parameter set.

PEST++ stores observation residuals in a `*.rei` file. When instantiating a `Pst` class from an existing control file, `pyemu` will attemp to read a corresponding `*.rei` file. Data from the rei file is stored in the `Pst.res` attribute as a `Pandas` `DataFrame`. This makes it easy to access and postprocess. We can also read in residuals after instatinating a `Pst` object by using the `Pst.set_res()` method. 


```python
pst.res.head()
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
      <th>name</th>
      <th>group</th>
      <th>measured</th>
      <th>modelled</th>
      <th>residual</th>
      <th>weight</th>
    </tr>
    <tr>
      <th>name</th>
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
      <th>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3652.5</th>
      <td>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3652.5</td>
      <td>oname:hds_otype:lst_usecol:trgw-0-13-10</td>
      <td>35.870909</td>
      <td>36.420135</td>
      <td>-0.549226</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3683.5</th>
      <td>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3683.5</td>
      <td>oname:hds_otype:lst_usecol:trgw-0-13-10</td>
      <td>35.791150</td>
      <td>36.294485</td>
      <td>-0.503335</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3712.5</th>
      <td>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3712.5</td>
      <td>oname:hds_otype:lst_usecol:trgw-0-13-10</td>
      <td>35.784015</td>
      <td>36.277202</td>
      <td>-0.493186</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3743.5</th>
      <td>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3743.5</td>
      <td>oname:hds_otype:lst_usecol:trgw-0-13-10</td>
      <td>35.818347</td>
      <td>36.328078</td>
      <td>-0.509731</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3773.5</th>
      <td>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3773.5</td>
      <td>oname:hds_otype:lst_usecol:trgw-0-13-10</td>
      <td>35.884582</td>
      <td>36.412482</td>
      <td>-0.527899</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



And then display 1-to-1 and residual plots for each of the non-zero weighted observation groups. This is a quick way to explore how well the model is able to replicate measured data. These plots you'll see often.  The left plot is a "1:1" plot that has simulated values on the x-axis and measured values on the y-axis; a perfect fit would be all circles on the black diagonal line.  The right plot has the residual (y-axis) compared to the observation magnitude (x-axis). The closer the circle is to the black line the better the fit.  The mean residual is shown as a red line. Ideally this red line should plot on y=0. 

Scroll down through the plot below. How well does the model replicate historical data? Within the range of "measurement error"? Seems good overall!

What about the residuals? are they evenly distributed around zero? If not, then this is a sign of bias. It means there is something wrong with the model or with the assumptions used for history matching. (In fact, because this is a synthetic model, we know what the cause is: under-parameterisation. Recall all the grid and pilot point parameters that we set as "fixed" during the "freyberg_glm_1" tutorial.) By looking for a "good" fit with our imperfect model, we may be inducing bias in our predictions...?


```python
# use pyemu's plot utilities to plot 1:1 line and the residuals as fxn of observation magnitude
pyemu.plot_utils.res_1to1(pst);
```


    <Figure size 576x756 with 0 Axes>



    
![png](freyberg_glm_2_files/freyberg_glm_2_65_1.png)
    



    
![png](freyberg_glm_2_files/freyberg_glm_2_65_2.png)
    


### Posterior Monte Carlo 

Because we included FOSM-based Monte Carlo options in the control file, when PEST++GLM concluded the inversion, it subsequently simulated an ensemble of parameters sampled from the linearized posterior parameter probability distribution. Observations from these simulations are recorded in the file named `freyberg_pp.post.obsen.csv`.
Let's read this file and instantiate a `pyemu.ObservationEnsemble` object to make it easy to plot them:


```python
df = df=pd.read_csv(os.path.join(m_d,"freyberg_pp.post.obsen.csv"),index_col=0)
oe = pyemu.ObservationEnsemble.from_dataframe(pst=pst,df=df)

oe.head()
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
      <th>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3652.5</th>
      <th>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3683.5</th>
      <th>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3712.5</th>
      <th>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3743.5</th>
      <th>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3773.5</th>
      <th>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3804.5</th>
      <th>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3834.5</th>
      <th>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3865.5</th>
      <th>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3896.5</th>
      <th>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3926.5</th>
      <th>...</th>
      <th>oname:hdstd_otype:lst_usecol:trgw-0-9-1_time:4169.5</th>
      <th>oname:hdstd_otype:lst_usecol:trgw-0-9-1_time:4199.5</th>
      <th>oname:hdstd_otype:lst_usecol:trgw-0-9-1_time:4230.5</th>
      <th>oname:hdstd_otype:lst_usecol:trgw-0-9-1_time:4261.5</th>
      <th>oname:hdstd_otype:lst_usecol:trgw-0-9-1_time:4291.5</th>
      <th>oname:hdstd_otype:lst_usecol:trgw-0-9-1_time:4322.5</th>
      <th>oname:hdstd_otype:lst_usecol:trgw-0-9-1_time:4352.5</th>
      <th>oname:hdstd_otype:lst_usecol:trgw-0-9-1_time:4383.5</th>
      <th>part_status</th>
      <th>part_time</th>
    </tr>
    <tr>
      <th>real_name</th>
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
      <td>36.3796</td>
      <td>36.2651</td>
      <td>36.2221</td>
      <td>36.2353</td>
      <td>36.3013</td>
      <td>36.3894</td>
      <td>36.4878</td>
      <td>36.4227</td>
      <td>36.2736</td>
      <td>36.0851</td>
      <td>...</td>
      <td>0.317252</td>
      <td>0.342115</td>
      <td>0.269281</td>
      <td>0.135337</td>
      <td>-0.021468</td>
      <td>-0.189139</td>
      <td>-0.324796</td>
      <td>-0.382808</td>
      <td>5</td>
      <td>222851.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>36.8739</td>
      <td>36.7338</td>
      <td>36.7396</td>
      <td>36.7981</td>
      <td>36.8933</td>
      <td>37.0025</td>
      <td>37.0336</td>
      <td>36.9469</td>
      <td>36.7490</td>
      <td>36.5025</td>
      <td>...</td>
      <td>0.112141</td>
      <td>0.084943</td>
      <td>0.044979</td>
      <td>-0.162401</td>
      <td>-0.399399</td>
      <td>-0.644954</td>
      <td>-0.825335</td>
      <td>-0.904149</td>
      <td>5</td>
      <td>49490.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>36.8321</td>
      <td>36.7919</td>
      <td>36.7847</td>
      <td>36.8238</td>
      <td>36.8996</td>
      <td>36.9714</td>
      <td>37.0006</td>
      <td>36.9317</td>
      <td>36.8216</td>
      <td>36.6193</td>
      <td>...</td>
      <td>0.455377</td>
      <td>0.512456</td>
      <td>0.439978</td>
      <td>0.282058</td>
      <td>0.089265</td>
      <td>-0.115376</td>
      <td>-0.272259</td>
      <td>-0.334552</td>
      <td>5</td>
      <td>159596.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>36.7126</td>
      <td>36.5110</td>
      <td>36.4849</td>
      <td>36.5394</td>
      <td>36.6174</td>
      <td>36.7140</td>
      <td>36.8055</td>
      <td>36.7205</td>
      <td>36.5493</td>
      <td>36.2799</td>
      <td>...</td>
      <td>0.079958</td>
      <td>0.033099</td>
      <td>-0.049624</td>
      <td>-0.246669</td>
      <td>-0.464928</td>
      <td>-0.693805</td>
      <td>-0.869428</td>
      <td>-0.894001</td>
      <td>5</td>
      <td>114285.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>36.4516</td>
      <td>36.3152</td>
      <td>36.3038</td>
      <td>36.3754</td>
      <td>36.4958</td>
      <td>36.6144</td>
      <td>36.7244</td>
      <td>36.6701</td>
      <td>36.4928</td>
      <td>36.1727</td>
      <td>...</td>
      <td>0.318807</td>
      <td>0.398570</td>
      <td>0.370773</td>
      <td>0.264358</td>
      <td>0.127420</td>
      <td>-0.024509</td>
      <td>-0.147014</td>
      <td>-0.178043</td>
      <td>5</td>
      <td>211444.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21252 columns</p>
</div>



Let's plot a histogram of Phi achieved by this ensemble. Some have a good fit (low Phi). Others not so much. (The more linear the problem, the more likley that more will have a good fit.) So should we use all of them? Probably not.


```python
oe.phi_vector.sort_values().hist()
```




    <AxesSubplot:>




    
![png](freyberg_glm_2_files/freyberg_glm_2_69_1.png)
    


Theoretically, as observations were weighted with the inverse of the standard deviation of measurement noise, we should accept a Phi equal to the number of nonzero observations. In practice, because of model error, we rarely reach the ideal value of Phi. For the purposes of this tutorial, we are going to arbitrarily take the 30 best realisations and use these as our "posterior ensemble".


```python
oe_pt = oe.loc[oe.phi_vector.sort_values().index[:30],:] #just take the 30 lowest phi realizations
```

Most of our observations are time series. Let's take a look at how well time series from the ensemble of outputs match those of measured values. This allows us to get a quick look at where the ensmbles may not be capturing model behaviour well. Run the following cell to generate the plots. Each plot displays an obsevration time series. The red lines are the measured values. The simulated values from each posterior realisation are displayed as blue lines.

What do you think? Are we happy with these results? There are some more indications of potential problems. The posterior ensemble fails to cover some of the measured data. And the simulated results do not seem to match the pattern in the measured data time series. The model does not seem to be capturing some details of system behaviour - possibly due to over-simplification of model structure or lack of flexibility in the parameterisation scheme. How confortable are we with trying to fit parameters under these conditions? Sketchy.


```python
nz_obs = pst.observation_data.loc[pst.nnz_obs_names,:].copy()

for nz_group in pst.nnz_obs_groups:
    # select obs values
    nz_obs_group = nz_obs.loc[nz_obs.obgnme==nz_group,:]
    nz_obs_group.time = nz_obs_group.time.astype(float)
    nz_obs_group.sort_values('time', inplace=True)
    fig,ax = plt.subplots(1,1,figsize=(10,2))
    # plot measured values
    ax.plot(nz_obs_group.time,nz_obs_group.obsval,"r-")
    # plot simulated values from post ensemble
    [ax.plot(nz_obs_group.time,oe_pt.loc[r,nz_obs_group.obsnme],color="b",lw=0.1,alpha=0.5) for r in oe_pt.index]

    ax.set_title(nz_group)
 
plt.show()
```


    
![png](freyberg_glm_2_files/freyberg_glm_2_73_0.png)
    



    
![png](freyberg_glm_2_files/freyberg_glm_2_73_1.png)
    



    
![png](freyberg_glm_2_files/freyberg_glm_2_73_2.png)
    



    
![png](freyberg_glm_2_files/freyberg_glm_2_73_3.png)
    



    
![png](freyberg_glm_2_files/freyberg_glm_2_73_4.png)
    



    
![png](freyberg_glm_2_files/freyberg_glm_2_73_5.png)
    


### The Minimum Error Variance Parameter Field

We have inspected the models' ability to replicate measured data. It is also a good idea to inspect how "sensible" are the obtained parameter values. A common easy check is to visualy inspect the spatial distirbution of hydrualic property parameters. One can often find unexpected insight from how parameter patterns emerge during calibration. 

First start by updating the control file with the calibrated parameter values using the `Pst.parrep()` method:


```python
pst.parrep(parfile=os.path.join(m_d, 'freyberg_pp.par'))
```

    Updating parameter values from master_glm_2\freyberg_pp.par
    parrep: updating noptmax to 0
    

Then write the parameter values to the model input files and run the model once:


```python
# updates the model input files with parameter values
pst.write_input_files(pst_path=m_d)

# run the model forward run; this applies all the multipler paarameters, executes MODFLOW6 and MODPATH7 and then postprocess observations
pyemu.os_utils.run('python forward_run.py', cwd=m_d)
```

Now we can use `flopy` to load the model and plot some of the parameters. Let's plot some maps of the spatial distribution of K. (Run the next cell.) 

Looks a bit "blotchy"...perhaps not particularily realistic, hey? In part, this is due to using the regularized Gauss Levenburg Marquardt option which tends to not look as "smooth" as using geostatistics-informed Tikhonov regularisation explicitly. In this case, it is also likley due to parameter values compensating for model structural error (e.g. we have n imperfect model and a corase parmaeterisation scheme).


```python
# load simulation
sim = flopy.mf6.MFSimulation.load(sim_ws=m_d, verbosity_level=0)
# load flow model
gwf = sim.get_model()
gwf.npf.k.plot(colorbar=True)
#gwf.sto.ss.plot()
#gwf.sto.sy.plot()
```




    <AxesSubplot:title={'center':'k'}>




    
![png](freyberg_glm_2_files/freyberg_glm_2_79_1.png)
    


Another quick check is to look for parameters which are at their bounds. (Run the next cell.) 

Some parameters are at their bounds. Along with the extreme values and blotchiness in parameter spatial distributions, this can often be a sign of either a strucutral problem with the model or an inflexible parameterisation scheme. 


```python
# idealy, this should return an empty list
pst.get_adj_pars_at_bounds()
```




    (['pname:rch_recharge_13tcn_inst:0_ptype:cn_pstyle:m'], [])



### Forecast Uncertainty

So far we have looked at the fit with measured data. But what we are realy intersted in are the forecasts. We instructed PEST++GLM to undertake both FOSM and FOSM-based Monte Carlo uncertainty analysis. We have already loaded the ensemble of forecasts from the Monte Carlo analysis (`oe_pt`). Let's also load and plot the FOSM forecast results along side of the ensemble results. FOSM forecast results are recorded in the file named `freyberg_pp.pred.usum.csv`. 


```python
f_df = pd.read_csv(os.path.join(m_d,"freyberg_pp.pred.usum.csv"),index_col=0)
f_df.index = f_df.index.map(str.lower)
f_df
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
      <th>oname:hds_otype:lst_usecol:trgw-0-9-1_time:4383.5</th>
      <td>35.5598</td>
      <td>1.3548</td>
      <td>32.8502</td>
      <td>38.2694</td>
      <td>39.1139</td>
      <td>0.353508</td>
      <td>38.4069</td>
      <td>39.8209</td>
    </tr>
    <tr>
      <th>oname:sfr_otype:lst_usecol:headwater_time:4383.5</th>
      <td>-127.1930</td>
      <td>467.6410</td>
      <td>-1062.4700</td>
      <td>808.0890</td>
      <td>-719.2990</td>
      <td>274.946000</td>
      <td>-1269.1900</td>
      <td>-169.4080</td>
    </tr>
    <tr>
      <th>oname:sfr_otype:lst_usecol:tailwater_time:4383.5</th>
      <td>92.9114</td>
      <td>398.7560</td>
      <td>-704.6000</td>
      <td>890.4220</td>
      <td>-369.0800</td>
      <td>274.225000</td>
      <td>-917.5300</td>
      <td>179.3710</td>
    </tr>
    <tr>
      <th>part_time</th>
      <td>99358.4000</td>
      <td>48689.3000</td>
      <td>1979.8600</td>
      <td>196737.0000</td>
      <td>117996.0000</td>
      <td>47419.500000</td>
      <td>23157.3000</td>
      <td>212835.0000</td>
    </tr>
  </tbody>
</table>
</div>



Now, let's make some plots of the forecast uncertainties obtained from FOSM (e.g. linear analysis) and from FOSM-based Monte Carlo (e.g. nonlinear analyis). This let's us investigate the validity of the assumed linear relation between parameters and forecasts (the forecast sensitivity vectors). If it holds up, then the FOSM posterior should cover the Monte Carlo posterior (note that we used very few realisations for Monte Carlo, so in our case this assumption is a bit shaky). 

We will superimpose the two posteriors on the *prior* forecast probability distribution to illustrate the uncertainty reduction gained from history matching. Because we are using a synthetic case, we know the true forecast values. So let's also plot these. If all went well, both posteriors should cover the truth.

Run the next cell to generate plots for each of the forecasts referenced in the control file:


```python
obs = pst.observation_data
fnames = pst.pestpp_options["forecasts"].split(",")
for forecast in fnames:
    ax = plt.subplot(111)
    # plot Monte Carlo posterior
    oe_pt.loc[:,forecast].hist(ax=ax,color="b",alpha=0.5,density=True)
    # plot truth
    ax.plot([obs.loc[forecast,"obsval"],obs.loc[forecast,"obsval"]],ax.get_ylim(),"r")
    # plot FOSM prior
    axt = plt.twinx()
    x,y = pyemu.plot_utils.gaussian_distribution(f_df.loc[forecast,"prior_mean"],f_df.loc[forecast,"prior_stdev"])
    axt.fill_between(x,0,y,facecolor="0.5",alpha=0.25)
    # plot FOSM posterior
    x,y = pyemu.plot_utils.gaussian_distribution(f_df.loc[forecast,"post_mean"],f_df.loc[forecast,"post_stdev"])
    axt.fill_between(x,0,y,facecolor="b",alpha=0.25)
    axt.set_ylim(0,axt.get_ylim()[1])
    axt.set_yticks([])
    #ax.set_xlim(oe_pt.loc[:,forecast].min() * .1,oe_pt.loc[:,forecast].max() * 1.5)
    #axt.set_xlim(oe_pt.loc[:,forecast].min() * .10,oe_pt.loc[:,forecast].max() * 1.5)
    ax.set_title(forecast)
    plt.show()
```


    
![png](freyberg_glm_2_files/freyberg_glm_2_85_0.png)
    



    
![png](freyberg_glm_2_files/freyberg_glm_2_85_1.png)
    



    
![png](freyberg_glm_2_files/freyberg_glm_2_85_2.png)
    



    
![png](freyberg_glm_2_files/freyberg_glm_2_85_3.png)
    


Does the FOSM posterior (blue-shaded area) overlap the Monte Carlo posterior (blue columns) for all forecasts? If so, that's good news. It means the assumed linear parameter-forecast relation in the FOSM calculations holds true. If it does not, then the usefullness of FOSM for forecast uncertainty analysis is erroded.

Because we are working with a synthetic case for which we know the truth, we can also assess whether forecast posteriors include the true value. Naturally, this would not be possible for a real-world case. Do each of the red lines fall within the FOSM (blue shaded area) and Monte Carlo (blue column) posteriors? If so, then technically uncertainty analysis has not failed. #winning. If not, then it has failed. #major bad times.

In this case we have done a pretty good job. We seem to have avoided underestimating uncertainty, even for the difficult null-space dependent forecasts. Counterintuitively, we did so by _not_ getting a really good fit with with measured data. Only _as good_ a fit as is reasonable given the information content therein. 
