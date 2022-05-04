# Intro to pyEMU

This notebook provides a quick run through some of the capabilities of `pyemu` for working with PEST(++). This run through is very "high level". We will not go into detail, merely show some of the core functionality as a primer for subsequent tutorials. We assume the reader has at least some understanding of PEST(++), common file structures and workflows.

We will make use of an existing PEST(++) interface. You do not need to be familiar with the details of the setup for the purposes of the current tutorial. 

Throughout the notebook we will:
 - introduce how to access and edit an existing PEST control file and setup using the `Pst` class.
 - explore some of the built-in methods for customing PEST setups and post-processing outcomes.
 - introduce geostats in `pyemu`.
 - introduce methods for handling matricies with the `Matrix` class, as special instances with the `Cov` and `Jco` classes.
 - introduce classes that facilitate generating and handling parameter and obsveration Ensembles.

Here we **do not** demonstrate how to setup a PEST interface from scratch. See the "part2_pstfrom_pest_setup" tutorial for a demonstration on how to use the `PstFrom` class to do so.

### 1. Admin


```python
import os
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append(os.path.join("..", "..", "dependencies"))
import pyemu
```

In the tutorial directory there is a folder named `handling_files`. This folder contains the PEST(++) dataset which we will access. Feel free to explore it if you wish. 

Amongst other things, it includes:

 - PEST(++) control file: `freyberg_pp.pst` 
 - A Jacobian matrix produced by PEST: `freyberg_pp.jcb` 
 - Parameters assigned using pilot points. These are recorded in the pilot point file: `hkpp.dat.tpl`


```python
# the pest folder
f_d = os.path.join('handling_files')
```

### 2. The `Pst` class

`pyEMU` encapsulates the PEST control file in the `Pst` class. Let's read the `freyberg_pp.pst` control file in the `f_d` folder.


```python
pst = pyemu.Pst(os.path.join(f_d,"freyberg_pp.pst"))
```

From the `pst` instance, we can access all the "*" sections of the control file. Each of these is stored as an attribute.  PEST variable names are used for consistency. 

For example, the `* parameter data` section can be accessed by calling `pst.parameter_data`. This returns the parameter data as a `Pandas` DataFrame, making it easy to access and edit. 


```python
pst.parameter_data.head()
```

The same for `* observation data`:


```python
pst.observation_data.head()
```

You can access, slice and edit `pst.parameter_data` or `pst.observation_data` just as you would a DataFrame. You can add columns, these will not be writen to the control file. `pyemu` is clever like that.


```python
pst.parameter_data.columns
```

It's best not to mess around with parameter names (`parnme`). If you do, you will need to make sure that the corresponding names appear in a `.tpl` file. (The same applies for observation names and `.ins` files.) 

You can edit values like so:



```python
#assing values to all parameter intial values
pst.parameter_data['parval1'] = 1.1

# the pandas .loc method makes for easy slicing and dicing
pst.parameter_data.iloc[:3,:] 
```

#### 2.1. Control Data Section

The `* control data` section is handled by a special class that tries to prevent mistakes. `pyemu` will not allow you to assign an illogical value to a control data variable. This avoids silly mistakes. But it also requires that the user must know what type(s) can be passed to each data variable.

For example, NOPTMAX can only be assigned an integer. If you try to assign text, you will get an error. Try replacing the value 0 in the next cell with the string "zero":



```python
pst.control_data.noptmax = 0 # replace 0 with "zero" and see what happens
```

We can inspect all control data values using the `pst.control_data.formatted_values` attribute:


```python
pst.control_data.formatted_values
```

#### 2.2. PEST++ Options

PEST++ options are stored in a dictionary in which the `keys` are the PEST++ Control Variable name (see the PEST++ user manual for names of these variables and their descriptions). Values must be asigned according to what PEST++ expects as input.

You can access the existing PEST++ options like so:


```python
pst.pestpp_options
```

You can change the values of existing control variables like so:


```python
# changes the value for the PEST++ option 'ies_parameter_ensemble'
pst.pestpp_options['ies_parameter_ensemble'] = 'dummy_ies_par_ensemble.csv'

# check the dictionary again
pst.pestpp_options
```

Or add new PEST++ variables like so:


```python
# A few examples of adding PEST++ options of different types:
# pestpp-ies; the number of realizations to draw in order to form parameter and observation ensembles.
pst.pestpp_options['ies_num_reals'] = 50

# specifies a list of values for the Marquardt lambda used in calculation of parameter upgrades. 
pst.pestpp_options["lambdas"] = [0.1, 1, 10, 100, 1000]

# pestpp-da; True/False, specify whether to use the simulated states at the end of each cycle as the initial states for the next cycle.   
pst.pestpp_options['da_use_simulated_states'] = True

# check the dictionary again
pst.pestpp_options
```

#### 2.3. Writing the .pst control file

All of these edits are kept in memory untill explicitly written to a .pst file. This is accomplished with the `Pst.write()` method.

The control file can be written as version 1 or 2 (see the PEST++ user manual for descriptions of versions). Unlike the original PEST version 1, version 2 control files have each of the "*" sections stored in external csv files. This makes them easier to access and manipulate, either programatically or using common spreadsheet software. PEST and PEST_HP only accept version 1. Only PEST++ accepts version 2. If there are more than 10k parameters, version 2 is written by default. 

You can specify the version by passing the relevant argument. Run the cells below then inspect the folder to see the differences.


```python
pst.write(os.path.join(f_d, 'temp.pst'), version=1)
```


```python
pst.write(os.path.join(f_d, "temp_v2.pst"), version=2)
```


```python
[f for f in os.listdir(f_d) if f.endswith('.pst')]
```

#### 2.4. Adding Parameters/Observations from .tpl/.ins files

In another tutorial we demonstrate the use of the `PstFrom` to automate construction of a complete PEST interface, starting from scratch.

Alternatively, it is possible to construct a PEST control file from template (.tpl) and instruction (.ins) files. The `Pst` class includes methods to read .tpl/.ins files and add parameters/observations to the control file. This enables construction of a PEST dataset, or simply adding new ones to an existing control file. It is particularily usefull for pesky model input/output files with inconvenient file structures (e.g. that are not array or tabular formats).

The cell bellow writes a "dummy" template file to the tutorial folder for demonstration purposes. 


```python
tpl_filename = os.path.join(f_d,"special_pars.dat.tpl")
with open(tpl_filename,'w') as f:
    f.write("ptf ~\n")
    f.write("special_par1  ~  special_par1   ~\n")
    f.write("special_par2  ~  special_par2   ~\n")
```

Adding the parameters from "special_pars.dat.tpl" to the control file is a simple matter of calling the `.add_parameters()` method. This method adds the parameters to the `pst.parameter_data` section, and updates other relevant sections of the `pst` control file; it also returns a DataFrame. Note that parameter names come from the .tpl file. However, intial values, bounds and group name are assigned default values; you will need to specify correct values before writing the control file to disk.


```python
pst.add_parameters(tpl_filename, pst_path=".") #pst_path is the relative path from the control file to the .tp file
```

You can then adjust parameter data details:


```python
# assign to variable to make code easier to read
par = pst.parameter_data

# adjust parameter bounds; don't worry about this now
par.loc[par['pargp'] == 'pargp', ['parlbnd', 'parubnd']] = 0.1, 10

par.loc[par['pargp'] == 'pargp']
```

#### 2.5. Tying Parameters

We may on ocasion need to tie parameters in the control file. In the `pyemu` world, tied parametes are specified in the `Pst.parameter_data` dataframe. Start by adding a `partied` column and, for parameters you want to tie, changing "partrans" to "tied" and adding the name of the parameter to tie to in the "partied" column. 

We will demonstrate step-by-step by tying "special_par2" to "special_par1" (the parameters we just added from the .tpl file):


```python
# see the parameters
par.loc[par['pargp'] == 'pargp']
```


```python
# set the partrans for "special_par2" as "tied"
par.loc['special_par2', 'partrans'] = 'tied'

# add a new column named "partied" and assign the parameter name to which to tie "special_par2"
par.loc['special_par2', 'partied'] = 'special_par1'

# display for comparison; see partrans and partied columns
par.loc[par['pargp'] == 'pargp', ['partrans', 'partied']]
```

### 3. Utilities

`pyemu` has several built-in methods to make your PEST-life easier. Several of these handle similar tasks as utilities from the PEST-suite, such as adjusting observation weights and assigning prior information equations. Others provide usefull tables or plots that summarize details of the PEST setup and/or outcomes.

#### 3.1. Par and Obs Summaries
You can access quick summaries of observation and paramaeter names, group names, etc thorugh the respective `pst` attributes:


```python
# non-zero weighted observation groups, returns a list. 
# Here it sliced to the first 5 elements to keep it short
pst.nnz_obs_groups[:5]
```


```python
# number of non-zero observations and adjustable parmaeters
pst.nnz_obs, pst.npar_adj
```


```python
# adjustble parameter group names
pst.adj_par_groups[:5]
```

You can write a parameter or observation summary table wth the `Pst.write_par_symmary_table()` and `Pst.write_obs_symmary_table()` methods, respectively. Quite usefull when preparing those pesky reports. 

These methods return a Pandas DataFrame and (by default) write the table to an external file. Parameters and observations are summarized by group name.


```python
pst.write_par_summary_table()
```


```python
pst.write_obs_summary_table()
```

### 3.2. Phi and residuals

The `Pst` class tries to load a residuals file iduring construction. It looks for a file in the same folder as the control file and with the same base name, but with the extension ".rei". Alterantaively, you can specify the name of the residual file when constructing the `Pst`. (e.g. `pyemu.Pst("controlfile.pst", resfile="residualfile.rei")`)

If that file is found, you can access some pretty cool stuff.  The `Pst.res` attribute is stored as a Pandas DataFrame. 

Of course, all of this relies on PEST(++) having been run at least once before hand to record the residuals file. For the purposes of this tutorial, we have already done so. When we constructed `pst` at the beggining of this notebook, `pyemu` also loaded the residuals file. 

Inspect it by running the cell bellow. As you can see, the DataFrame lists all observations and group names, their modelled and measured values, weights and of course the residual:



```python
pst.res.head()
```

A somewhat clunky (and meaningless) look at everyones favourite "good fit" plot:


```python
pst.res.plot.scatter('measured', 'modelled')
```

Or a clunky look at the residuals for selected observations:


```python
pst.res.iloc[:10].loc[:, 'residual'].plot(kind='bar')
```

There are built in routines for some common plots. These can be called with the `Pst.plot()` method and specifying the `kind` argument. For example, a 1to1 plot for each observation group: 


```python
# 1to1 plots are displayed for each observationg group with non-zero weighted observations
pst.plot(kind='1to1');
```

The weighted sum of square residuals (Phi) is also stored in the respective `Pst` attribute. 


```python
# the value of the objective function
pst.phi
```

We can access the components of the objective function as a dictionary. These allow us to breakdown the contributions to Phi from each observation group:


```python
# observation group contributions to Phi
pst.phi_components
```

They can also be displayed with a plot, like so:


```python
pst.plot(kind="phi_pie");
```

These values can be recalculated for different observation weights by simply changing the weights in the `pst.observation_data`. No need to re-run PEST!


```python
obs = pst.observation_data
# change all observation weights
obs['weight'] = 1.0

# check the phi contributions again; compare to vaues displayed above
pst.phi_components
```

### 3.3. Adjusting Weights for "Visibility"

Prior to estimating parameters using PEST(++), a user must decide how to weight observations. In some
cases, it is wise to weight observations strictly according to the inverse of the standard deviation of
measurement noise. Certainly, observations of greater credibility should be given greater weights
than those of lower credibility. 

However, when history-matching, this approach can result in the loss of information contained in some observations, due to their contribution to the objective function being over-shadowed by the contribution from other obsevrations. An alternative approach is to weight observations (or observation groups) such that, at the start of the history-matching process, they each contribute the same amount to the objective function. The information content of each of these groups is thereby given equal right of entry to the parameter estimation process. (This matter is extensively discussed in the PEST Book.)

As stated above, a practical means of accommodating this situation is to weight all observation groups
such that they contribute an equal amount to the starting measurement objective function. In this
manner, no single group dominates the objective function, or is dominated by others; the information
contained in each group is therefore equally “visible” to PEST(++).

The `Pst.adjust_weights()` method provides a mechanism to fine tune observation weights according to their contribution to the objective function. (*Side note: the PWTADJ1 utility from the PEST-suite automates this same process of "weighting for visibility".*) 

Some **caution** is required here. Observation weights and how these pertain to history-matching *versus* how they pertain to generating an observation ensemble for use with `pestpp-ies` is a frequent source of confusion.

 - when **history-matching**, observation weights listed in the control file determine their contribution to the objective function, and therefore to the parameter estiamtion process. Here, observation weights may be assigned to reflect observation uncertainty, the balance required for equal "visibility", or other modeller-defined (and perhaps subjective...) measures of observation worth.  
 - when undertaking  **generating an observation ensemble**, weights should reflect the inverse of the standard deviation of measurement noise. Unless instructed otherwise, `pestpp-ies` will generate the observation ensemble *using observation weights in the PEST control file*. Therefore, when history-matching with `pestpp-ies` and using weights that **do not** reflect observation uncertainty, it is important to provide `pestpp-ies` with a previously prepared observation ensemble (we will demonstrate this further on).

OK, so let's adjust some observation weights using `Pst.adjust_weights()`. This method allows us to adjust weights for individual observations, or for entire observation groups. We do so by passing a dictionary with observation names (or group names) as keys, and the correspding value they contribute to the objective function as values.

Let's get started.


```python
# check the phi contributions
pst.phi_components
```

Now, let's create the dictionary of non-zero weighted observation groups. We will specify that we want each group to contribute a value of 100 to the objective function. (Why 100? No particular reason. Could just as easily be 1000. Or 578. Doesn't really matter. 100 is a nice round number though.)


```python
balanced_groups = {grp:100 for grp in pst.nnz_obs_groups}
balanced_groups
```

Now we can simply pass this dictionary as an argument:


```python
# make all non-zero weighted groups have a contribution of 100.0
pst.adjust_weights(obsgrp_dict=balanced_groups,)
```

And voila. Run the cell below to see that phi components form each group are (roughly) 100. The same approach can be implemented for individual observations (see the `obs_dict` argument in `pst.adjust_weights()`). 


```python
# check the phi contributions; comapre to those above
pst.phi_components
```


```python
# comapre this plot to the one we generated earlier; this one is much more balanced
pst.plot(kind="phi_pie");
```

### 3.4. Discrepancy based weight adjustment

In a perfect (model and algorithm) world, we would acheive a final objective function that is equal to the number of (non-zero weighted) observations. But because of model error and simplifying assumptions in the algorithms we use for history matching, this is rarely the case.  More often, the final objective function is much larger than the number of observations.  This implies that we were not able to "fit" as well as we thought we could (where "thought" is incapsulated in the observations weights in the control file, representing the inverse of measurment noise).  This really matters when we do posterior uncertainty analyses following a PEST run (this will be discussed further in the FOSM and data-worth notebooks). 

The simpliest way to try to rectify this situation is to adjust the weights in the control file so that the resulting contribution to the objective function from each observation (or optional observation group) is equal to 1 (or the number of members of the group).  This is related to Morozov's discrepancy principal (google it!).  `pyEMU` has a built in routine to help with this: `Pst.adjust_weights_discrepancy()` - great name!

*Note 1: dont make this adjustment until after you are through with history matching! The point is for weights to represent the inverse of observation uncertainty, which includes both measurement error **and** model error.*

*Note 2: the PWTADJ2 utility from the PEST-suite acomplishes a similar task.*  


```python
# see current phi and the number of non-zero observations
pst.phi, pst.nnz_obs
```


```python
pst.adjust_weights_discrepancy(original_ceiling=True) # default
# check the resulting phi
pst.phi
```

So we were expecting Phi to be equal to `nnz_obs` (number of non zero observations). This did not happen due to the `original_ceiling` argument being set to `True` (which is the default value).

What this means is that, for some observations, weights would have to be *increased* to achieve a contribution to Phi of 1.0. Which is illogical if the original weight is assumed to be the inverse of the measurement noise. In some cases, this requirement may not apply; such as when observatons are weighted for visibility.

### 4. Geostatistics in pyEMU

The `pyemu.geostats` module provides tools for implementing geostatistics in the `pyemu` world. These have similar functions to the PPCOV* utilities from the PEST suite. 

In the PEST world, geostatistics are used (1) to describe how parameters vary in space and/or time and (2) when assigning parameter values from pilot points to model input files. Front-end users will mostly be interested in the former. A separate tutorial notebook delves into geostatistics in greater detail (see the "intro to geostats" notebook). The "part1 pest setup" and "part2 pstfrom pest setup" tutorials provide examples of their use in the wild.

At the heart of geostatistics is some kind of model expressing the variability of properties in a field. This is a "variogram". `pyemu` supports spherical, exponential and gaussian variograms.


```python
# exponential variogram for spatially varying parameters
v = pyemu.geostats.ExpVario(contribution=1.0, #sill
                                    a=1000, # range of correlation
                                    anisotropy=1.0, #name says it all
                                    bearing=0.0 #angle in degrees East of North corresponding to anisotropy ellipse
                                    )
```


```python
# spherical variogram for spatially varying parameters
v2 = pyemu.geostats.SphVario(contribution=1.0, #sill
                                    a=500, # range of correlation
                                    anisotropy=1.5, #name says it all
                                    bearing=45.0, #angle in degrees East of North corresponding to anisotropy ellipse
                                    name='var2') #optional name
```

A `pyemu`  geostatistical structure object that mimics the behavior of a PEST geostatistical structure (see the PEST manual for details).  The object contains variogram instances, information on parameter transforms and
(optionally) nugget information. Multiple nested variograms are supported.


```python
# geostatistical structure for spatially varying parameters
gs = pyemu.geostats.GeoStruct(variograms=[v], transform='log') 

# plot the gs if you like:
gs.plot()
```

Using the geostatistical structure (or a variogram), a covariance matrix describing the statistical relationship between spatialy distributed parameters can be constructed. These covariance matrices are used (1) to describe prior parameter probability distributions, and (2) specifying the weights of prior information equations as a regularisation device. 

For example, let's create a covariance matrix for a set of pilot point parameters from the `gs` structure. This requires a list of X and Y coordinates. Let's make use of the coordinates from a pilot points file in the `f_d` folder.

First, read the pilot points file to get X and Y values:


```python
df = pyemu.pp_utils.pp_tpl_to_dataframe(os.path.join(f_d,"hkpp.dat.tpl"))
df.head()
```

Now simply pass the respective values from the pilot points file as arguments to `gs.covariance_matrix()`. Conveniently, we can also take the parameter names from the pilot point file. This returns a `pyemu.Cov` object (which we will discuss later). The same can be complished with a single variogram (e.g. `v.covariance_matrix()`)


```python
cov = gs.covariance_matrix(x=df.x, y=df.y, names=df.parnme)
```


```python
# just for a bit of eye-candy; bright yellow indicates higher covariance.
c = plt.imshow(cov.x)
plt.colorbar(c)
```

We can do the same thing for a 1-D sequence (think time-series), to get a covariance matrix for parameters distributed in time:


```python
# let's make up a time-series
times = np.arange(0,365,1) # this is the "X" coordinate
y = np.ones_like(times)    # this is the "Y" coordinate
names = ["t_"+str(t) for t in times] # 'parameter" names

# make the variogram
exp_vario = pyemu.geostats.ExpVario(contribution=1.0,
                                    a=5 #range in time units (e.g. days)
                                    )
cov_t = exp_vario.covariance_matrix(times,y,names)
plt.imshow(cov_t.x)
```

### 5. Prior Information Equations

The mathematical term for the process through which a unique solution is sought for a nonunique
inverse problem is “regularisation”. The goal of regularised inversion is to seek a unique parameter
field that results in a suitable fit between model outputs and field measurements, whilst minimizing
the potential for wrongness in model predictions. That is, out of all the ways to fit a calibration dataset, regularized inversion seeks the parameter set of minimum error variance. (*Regularisation is discussed in greater detail in a separate notebook.*)

One way to seek a parameter field of minimum error variance is to seek a parameter field that allows
the model to fit the calibration dataset, but whose values are also as close as possible to a set of
“preferred parameter values”. Ideally, preferred parameter values should also be initial parameter
values as listed in the “parameter data” section of the PEST control file. These preferred parameter
values are normally close to the centre of the prior parameter probability distribution. At the same
time, scales and patterns of departures from the preferred parameter set that are required for model
outputs to fit a calibration dataset should be achieved in ways that make “geological sense”.

PEST provides a user with a great deal of flexibility in how Tikhonov constraints can be introduced to
an inversion process. The easiest way is to do this is through the use of prior information equations.
When prior information equations are employed, Tikhonov constraints are expressed through
preferred values that are assigned to linear relationships between parameters. (Equality is the simplest type of linear relationship.) Weights must be assigned to these equations. As is described in PEST documentation, when PEST is run in “regularisation” mode, it makes internal adjustments to the weights that are assigned to any observations or prior information equations that belong to special observation groups that are referred to as “regularisation groups”.

*Note: in a similar manner, the PEST-utilities ADDREG1 and ADDREG2 also automate the addition of prior information equations to a PEST-control file.*

### 5.1. Preffered value or Zero Order Tikhonov

`pyemu` provides utilities to apply preferred value prior equations to a PEST control file. Note though, pyemu doesn't call it "preferred value"! Rather, it uses the mathematical term "Zero Order Tikhonov".

Before we do so, we can inspect the control file `* prior information` section. As you may have guessed, this is accessed using the `pst.prior_information` attribute, which returns a `DataFrame`:



```python
pst.prior_information.head()
```


```python
# use pyemu to apply preferred value (aka zero order Tikhonov) to all adjustable parameter groups
pyemu.helpers.zero_order_tikhonov(pst,
                                par_groups=pst.adj_par_groups, # par groups for which prior inf eq are added
                                reset=True) # whether to remove existing prior equations first; default is true
```

Now, as you can see below, prior information equations have been added for all adjustable parameters. The "preferred value" is obtained from the parameter initial value in the `* parameter_data` section. The weight is calculated from the parameter bounds (this behaviour can be changed with arguments in `pyemu.helpers.zero_order_tikhonov()`).


```python
pst.prior_information.head()
```

### 5.2. Preferred difference or First Order Pearson Tikhonov

We may wish (almost certaintly) to express a preference for similarity between parameters. For example, hydraulic properties of two points close together are more likley to be similar to each other, than two points which are far apart. We describe this relationship using geostatistics, encapsulated in a covariance matrix.

As previously described, `pyemu.geostats` module provides tools for generating such matrices. The PEST suite also includes many utilities for this purpose (see the PPCOV* set of utilities.)
Let's use the `cov` covariance matrix we constructed earlier for the set of pilot points.


```python
# a reminder
plt.imshow(cov.x)
```

Now, we can assign prior information equations for preferred difference. Note that the preferred difference = 0, which means our preferred difference regularization is really a preferred *homogeneity* condition! If observation data doesn't say otherwise, parameters which are close together should be similar to each other.

The weights on the prior information equations are the Pearson correlation coefficients implied by the covariance matrix.


```python
# then assign cov pror
pyemu.helpers.first_order_pearson_tikhonov(pst, 
                                            cov=cov,     # the covariance matrix; these can be for some OR all parameters in pst
                                            reset=False, # so as to have both prefered value and prefered differnece eqs
                                            abs_drop_tol=0.01) # drop pi eqs that have small weight
```


```python
# note the additional number of prior information equations
pst.prior_information.tail()
```

### 5.3. Custom Prior Information Equations

`Pst.add_pi_equation()` is a helper to construct prior information equations. We demonstrate below using the "special_par"s we added earlier.


```python
# reminder
par = pst.parameter_data
# let's just un-tie special_par2, otherwise we can't assign a prior info euqation using it
par.loc[par['pargp']=='pargp', 'partrans'] = 'log'
#display
par.loc[par['pargp']=='pargp']
```


```python
pst.add_pi_equation(
                    par_names=['special_par1', 'special_par2'], # list of parameter names included on the left hand side of the equation
                    pilbl='new_pi_eq',
                    rhs=0.0, # the value on the right hand side of the equation; make sure that this value is logical. you may also wish to check if it conflicts with existing prior information equations
                    weight=1.0, # the weight assigned to the prior information equation
                    obs_group='regul_special', # name to assign to the prior information "observationg group"
                    coef_dict = {'special_par1':1.0, 'special_par2':0.5} # dictionary of parameter coeficients; try specifying different values, note how the equation changes
                    )

# let's take a look; just print the last row (e.g. the latest prior info equation)
pst.prior_information.iloc[-1:]

```

### 6. Matrices

The `pyemu.Matrix` class is the backbone of `pyemu`. The `Matrix` class does some fancy things in the background to make manipulating matrices and linear algebra easier. The class overloads all common mathematical operators and also uses an "auto-align" functionality to line up matrices for multiplication, addition, etc. The `pyemu.Jco` class provides a wrapper to deal with PEST Jacobian matrices. It functions the same as the `Matrix` class.

The `pyemu.Cov` class is a special class designed specifically to handle covariance matrices. (We have already seen examples of the `pyemu.Cov` earlier.) The `Cov` class makes some assumptions, such as symmetry (and accordingly that matrix rows and columns names are equal) and provides additional functionality.

All classes provide functionality to record matrices to, as well as read from, external files in PEST-compatible formats (e.g. ASCII or binary). `Cov` can also write PEST uncertainty (*.unc) files. The `Cov` class has additional functionality that allows covariance matrices to be constructed from information contained in PEST control files (e.g. from parameter or observation bounds). 


```python
# generate covariance matrix from parameter bounds
parcov = pyemu.Cov.from_parameter_data(pst)

# generate a covariance matrix from observation data (e.g. weights):
obscov = pyemu.Cov.from_observation_data(pst)
```

As you can see below, the `Cov` object retains the parameter names as row and column names.


```python
# the first 5 row and col names 
parcov.row_names[:5], parcov.col_names[:5], 
```

We can check if the matrix is diagonal with the `.isdiagonal` attribute. 


```python
parcov.isdiagonal
```

How about the `cov` covariance matrix we generated earlier for spatialy correlated parameters?

Recall that, a matrix is "diagonal" when all the entries off the diagonal are zero. This means that elements of the matrix are **uncorrelated**. By generating a covariance matrix from parameter data, the only information we have is the uncertainty of each individual parameter (expressed by the parameter bounds). There is no information on correlation between parameters. Therefore `cov_pb.isdiagonal` is `True`. When we generated `cov`, we specified correlation between parameters; so, off-diagonal entries in `cov` are non-zero. Therefore, `cov.isdiagonal` is `False`.


```python
cov.isdiagonal
```

The values of the matrix are accessed in the `.x` attribute:


```python
parcov.x
```

Note that `parcov.x` is 1-dimensional. Again, this is because `parcov` is "diagonal". So `.x` only returns the diagonal entries. On the other hand, `cov.x` is 2-dimensional:


```python
parcov.x.shape, cov.x.shape
```

You can access the full 2-dimensional matrix with the `.as_2d` attribute (see all the off-diagonals are zero):


```python
parcov.as_2d
```

It may be easier to visualize this in a plot. Because `parcov` has many parameters it may be hard to see. So let's first use the `get()` method to get a submatrix. We can use this method to extract specific row and column names. We will get the same pilot point parameters that are in the `cov` matrix to compare.


```python
# make a list of parameter names
parnames = par.loc[par['pargp']=='hk', 'parnme'].tolist()

# get a submatrix from cov_pb; both row and column names must be the same if we want to retain a square matrix (which we do...)
subcov = parcov.get(row_names=parnames, col_names=parnames, drop=False)


# plot  both matrices side by side
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# plot submatrix of diagonal matrix
c = ax1.imshow(subcov.as_2d)
plt.colorbar(c, ax=ax1, shrink=0.3)
# plot cov matrix of correlated parameters
c = ax2.imshow(0.25 * cov.as_2d) #scaled to parameter unc
plt.colorbar(c, ax=ax2, shrink=0.3)
```

The singular values are the diagonal entries of the matrix, arranged in descending order. These can be accessed with the `.s` attribute. This still returns a `Matrix` object (technically a vector). Values in the matrix can be accessed in the same fashion as described above.


```python
# get the singular values vector
cov_singular_values = cov.s

# access the entries in the vector
cov_singular_values.x
```

"Right singular vectors" are obtained with `.v` attribute:


```python
cov.v
```

Matrices can be written to, and read from, external files. These can be ASCII or binary type files. As you saw above, we can access numpy arrays with the `.as_2d` attribute. They can also be converted to a Pandas Dataframe.


```python
covdf = cov.to_dataframe()
covdf.head()
```


```python
# write to ascii; .to_binary() for binary format
cov.to_ascii(filename=os.path.join(f_d,'ppoint.cov'))
```


```python
# read from ascii
cov = cov.from_ascii(filename=os.path.join(f_d,'ppoint.cov'))
```


```python
# a PEST uncertainty file
cov.to_uncfile(os.path.join(f_d,'test.unc'))
```

Jacobian matrices, recorded by PEST(++) during parameter estimation, can be read and manipulated with the `Jco` class. This calss has the same functioanlity as the `Matrix` class.


```python
jco = pyemu.Jco.from_binary(os.path.join(f_d,"freyberg_pp.jcb"))

jco.shape
```

### 7. Linear Analysis or FOSM

FOSM stands for "First Order, Second Moment". You may also see this referred to as "linear analysis" (e.g. in PEST documentation). We will delve into FOSM in more detail in another tutorial. Here we merely provide a brief introduction.

The ``pyemu.Schur`` object is one of the primary object for FOSM in `pyemu`. Instantiating a `Schur` object requires, at minimum, a Jacobian matrix and a PEST(++) control file.  From these, `pyemu` builds the prior parameter covariance matrix (from parameter bounds) and the observation noise covariance matrix (from observation weights). Alternatively, the parameter and observation covariance matrices can be provided explicitly.

Optionaly, observation names can be specified as forecasts. The `Schur` object extracts the corresponding rows from the Jacobian matrix to serve as forecast sensitivity vectors.

As we saw earlier, there is a Jacobian matrix in the tutorial folder, recorded in the file named `freyberg_pp.jcb`. 


```python
sc = pyemu.Schur(os.path.join(f_d,'freyberg_pp.jcb'), verbose=False)
```

The prior parameter covariance matrix is stored in the `.parcov` attribute:


```python
sc.parcov.to_dataframe().head()
```

The same for the observation noise covariance matrix:


```python
sc.obscov.to_dataframe().head()
```

The **posterior** parameter covariance matrix is calculated and stored in the `.posterior_parameter` attribute:


```python
sc.posterior_parameter.to_dataframe().head()
```

Let's record the prior and posterior coavariance matrix to external files (we will use these later):


```python
prior_cov = sc.parcov
prior_cov.to_ascii(os.path.join(f_d, 'freyberg_pp.prior.cov'))

post_cov = sc.posterior_parameter
post_cov.to_ascii(os.path.join(f_d, 'freyberg_pp.post.cov'))
```

The ``Schur`` object found the "++forecasts()" optional pestpp argument in the control, found the associated rows in the Jacobian matrix file and extracted those rows to serve as forecast sensitivity vectors. lterantively, we can also pass a list of observation names to use as forecasts when instantiating the `Schur` object. (e.g.`pyemu.Schur("jacobian.jcb", forecasts=[obsname1, obsname2, etc...])`.)


```python
sc.forecast_names
```


```python
# forecast sensitivity vectors stores as `Matrix` objects:
sc.forecasts.to_dataframe().head()
```


```python
# summary of forecast prior and posterior uncertainty
sc.get_forecast_summary()
```

`pyemu` makes FOSM easy to undertake. It has lot's of usefull functionality. We will not go into further detail here. See the "intro to fosm" tutorial for a deeper dive.

### 8. Ensembles

The `pyemu.ParameterEnsemble` and `pyemu.ObservationEnsemble` ensemble classes store parameter or observation ensembles, respectively. These classes are `DataFrames` under the hood, allowing us to use all the baked in Pandas conveniences.

We will focus on the `pyemu.ParameterEnsemble` class. Similar concepts apply to `pyemu.ObservationEnsemble`.

Ensembles can be read from (and written to) external files, or generated by `pyemu` using one of several methods. These methods `draw` stochastic values from (multivariate) (log) gaussian, uniform and triangular distributions for parameters in a `Pst` control file.  Much of what we do is predicated on the gaussian distribution, so let's use that here by generating an ensemble with the `.from_gaussian_draw()` method. 

Note that these `draw` methods use initial parameter values in the control file (the `Pst.parameter_data.parval1` attribute) the $\boldsymbol{\mu}$  (mean) prior parameter vector.  Unless otherwise specified, the parameter bounds are used to approximate the prior parameter covariance matrix and these bounds are assumed to represent six standard deviations of the parameter probability distribution (i.e. ~99% of parameter values lie within the lower and upper bounds).  Users can change this assumption by creating their own prior parameter covariance matrix and passing it to `.from_gaussian_draw()`. 


```python
# before continuing, we are going to re-load the pest control file to get rid of any changes we introduced
pst = pyemu.Pst(os.path.join(f_d,"freyberg_pp.pst"))
```

Now let's just draw a prior parameter ensemble using the parameter bounds:


```python
pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst, # the Pst control file object, parameter data here will be used to characterize the ensembles' parameter statisctical distribution
                                                num_reals=200,) # the number of realisations to generate
                                                
```

We can express prior parameter correlation by passing a covariance matrix to the `cov` argument. In doing so, the covariance matrix describes the second moement (the standard deviation and optionally prior correlation between parameters) of the gaussian distribution. This allows us to draw parameter ensembles respecting prior parameter covariance. 


```python
pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst,           # the Pst control file object, parameter data here will be used to characterize the ensembles' parameter statisctical distribution
                                                num_reals=200, # the number of realisations to generate
                                                cov=prior_cov)    # specify parameter covariance; in this case, `parcov` doesn't actualy specify any covariance, but you get the idea...
```

Because the Ensemble is stored as a DataFrame, we can easily access it. Each row is an individual realisation. Each column is a parameter.


```python
pe.head()
```

Let's plot a histogram of values generated for one of the parameters:


```python
pe.loc[:, 'rch_0'].hist()
plt.xlabel('$log_{10}$ parameter value')
plt.ylabel('number of realisations')
```

We can see that this parameter has a "log" shape distribution, as expected based on it's `partrans` entry in the control file:


```python
par.loc["rch_0","partrans"]
```

Let's see its histogram in log space then:


```python
pe.loc[:, 'rch_0'].apply(np.log10).hist()
plt.xlabel('parameter value')
plt.ylabel('number of realisations')
```


```python
# chekc the parmeter bounds; do any of the histogram bnis fall above/below the upper/lower bound value?
par.loc['rch_0', ['parlbnd','parubnd']]
```


```python
pe.loc[:,"rch_0"].min(),pe.loc[:,"rch_0"].max()
```

As you can see, parameter bounds may be violated when drawing the ensemble because gaussian distributions are continuous. The `.enforce()` method goes through it and makes sure that bounds are respected. It does this by assigning the bound value to any parameter values which exceed the bound.  Conceptaully this results in a "truncated gaussian distribution"

As you can see in the subsequent plot, bounds are now respected:


```python
# enforce parmeter bounds
pe.enforce()

# plot again
pe.loc[:, 'rch_0'].hist()
plt.xlabel('parameter value')
plt.ylabel('number of realisations')
```

There are also built in functions to automate plotting of ensembles:


```python
# generates a A4 page of histograms for specified columns (e.g. parameters)

# slect column (e.g. parameter) names
plot_cols = pe.columns[0:8].tolist() 

# plot histograms
pe.plot(bins=10, 
        plot_cols=plot_cols, # specifyes which columns to plot
        filename=None, )     # external filename to record plot
```

You can record an ensemble to external files (.csv or binary).  Note the `.to_binary()` method saves the ensemble into the same extended compressed sparse storage file format as the jacobian matrix (".jcb") and the extension of the file is used to determine its type in PEST++.


```python
pe.to_csv(os.path.join(f_d, 'prior_pe.csv'))
pe.to_binary(os.path.join(f_d,'prior_pe.jcb'))
```

And of course, read from external files (csv, binary) or even from pandas DataFrames:


```python
pe = pyemu.ParameterEnsemble.from_csv(pst, filename=os.path.join(f_d, 'prior_pe.csv'))
pe_b = pyemu.ParameterEnsemble.from_binary(pst, filename=os.path.join(f_d, 'prior_pe.jcb'))
```

We can even form an empirical covariance matrix from an ensemble!  This is some for shadowing for the ensemble methods that are covered in later notebooks... 


```python
emp_cov = pe.covariance_matrix()

# display for fun
x = emp_cov.x.copy()
x[x<1.0e-2] = np.NaN
c = plt.imshow(x)
plt.colorbar(c)
```

and a reminder of what the initial parameter covariance matrix looked like (the one used to generate the ensemble):


```python
x = prior_cov.as_2d.copy()
x[x<1.0e-2] = np.NaN
plt.imshow(x)
```

So we can see that with 200 realizations, we can recover the diagonal pretty well, but we have some "spurious" covariances in the off diagonals...

#### 8.1 Bayes Linear Monte Carlo

We can use the bayes linear posterior parameter covariance matrix (aka Schur compliment) to "precondition" the realizations using linear algebra so that they hopefully yield a lower phi.  The trick is we just need to pass this
posterior covariance matrix to the draw method.  Note this covariance matrix is the second moment of the posterior (under the FOSM assumptions) and the final parameter values is the first moment.

In other GMDSI educational material, and in the PEST Roadmaps, this approach is sometimes referred to as "using the linearized posterior parameter distribution". 

Applying Bayes linear Monte Carlo requires that we have previously calibrated a model and calculated a post-calibration Jacobian matrix.  From the Jacobian, we can obtain the post-calibration parameter covariance. The "calibrated" parameter values represent the mean of the posterior parameter probability distribution. By centering the distribution on values that already provide a good fit with measurment data, we are increasing the likelihood that the realisatiosn that we draw will also fit measured data well. This can be usefull in reducing subsequent history-matching computation time.



```python
pe_post = pyemu.ParameterEnsemble.from_gaussian_draw(pst=pst,
                                                    cov=post_cov, 
                                                    num_reals=200)
pe_post.enforce()
```


```python
# comapre the prior and the posterior
pe.loc[:, 'rch_0'].hist(alpha=0.5,color="0.5")
pe_post.loc[:, 'rch_0'].hist(alpha=0.5,color="b")
#pe_post.loc[:, 'hk00'].plot(kind="hist",bins=20,ax=ax,alpha=0.5)
```

We see that the uncertainty in the recharge parameter `rch_0` has decreased substantially from prior (grey) to posterior (blue)


```python
# plots the change between two ensembles
pyemu.plot_utils.ensemble_change_summary(pe, pe_post, pst)
```

Or for comparing histograms from several ensembles. (Can also be used for observation ensembles; see additional method arguments):


```python
pyemu.plot_utils.ensemble_helper(ensemble={"0.5":pe, "b":pe_post,},
                                     filename=None,
                                     plot_cols=plot_cols,
                                     )
```

This is just a basic introduction in to handling ensembles in pyemu, we will see more later...


```python

```
