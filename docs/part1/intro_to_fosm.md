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

<img src="intro_to_fosm_files/bayes.png" style="float: left; width: 25%; margin-right: 1%; margin-bottom: 0.5em;">
<img src="intro_to_fosm_files/jacobi.jpg" style="float: left; width: 25%; margin-right: 1%; margin-bottom: 0.5em;">
<img src="intro_to_fosm_files/gauss.jpg" style="float: left; width: 22%; margin-right: 1%; margin-bottom: 0.5em;">
<img src="intro_to_fosm_files/schur.jpg" style="float: left; width: 22%; margin-right: 1%; margin-bottom: 0.5em;">


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

Outcomes of these analyses can provide easily understood insights into what history-matching can and cannot achieve with the available information. These insights can be used to streamline the data assimlation process and guide further site characterisation studies. Of particular interest is data worth analysis. The worth of data is measured by their ability to reduce the uncertainties of model predictions that we care about. Because the equations on which FOSM relies do not require that an observation value already be known, data worth can be assessed on as-of-yet ungathered data. 

But we are getting ahead of ourselves, let's take this back to basics.


## The famous Bayes Rule in a nutshell:

We update our knowledge by comparing what we know/believe with measured/observed data. What we know now, is a function of what we knew before, compared to what we learned from measured data.

### $\underbrace{P(\boldsymbol{\theta}|\textbf{d})}_{\substack{\text{what we} \\ \text{know now}}} \propto \underbrace{\mathcal{L}(\boldsymbol{\theta} | \textbf{d})}_{\substack{\text{what we} \\ \text{learned}}} \underbrace{P(\boldsymbol{\theta})}_{\substack{\text{what we} \\ \text{knew}}} $


We can also think of this graphically, as taken from Anderson et al. (2015) in slightly different notation but the same equation and concept:

<img src="intro _to_fosm_files/Fig10.3_Bayes_figure.png" style="float: center;width:500px;"/>

The problem is, for real-world problems, the likelihood function  $\mathcal{L}(\theta | \textbf{D})$ is high-dimensional and non-parameteric, requiring non-linear (typically Monte Carlo) integration for rigorous Bayes. Unfortunatley, non-linear methods are computationaly expensive and ineficient. 

But, we can make some assumptions and greatly reduce computational burden. This is why we often suggest using these linear methods first before burning the silicon on the non-linear ones like Monte Carlo.  

## How do we reduce the computational burden? 

By assuming that:

### 1. There is an approximate linear relation between parameters and observations:

<img src="intro _to_fosm_files/jacobi.jpg" style="float: left; width: 5%; margin-right: 1%; margin-bottom: 0.5em;">

### <center> $\mathbf{J} \approx \text{constant}$, $\frac{\partial\text{obs}}{\partial\text{par}} \approx \text{constant}$</center>

### 2. The parameter and forecast prior and posterior distributions are approximately Gaussian:

<img src="intro _to_fosm_files/gauss.jpg" style="float: left; width: 5%; margin-right: 1%; margin-bottom: 0.5em;">

###  <center>  $ P(\boldsymbol{\theta}|\mathbf{d}) \approx \mathcal{N}(\overline{\boldsymbol{\mu}}_{\boldsymbol{\theta}},\overline{\boldsymbol{\Sigma}}_{\boldsymbol{\theta}})$ </center>

Armed with these two assumptions, from Bayes equations, one can derive the Schur complement for conditional uncertainty propogation:

<img src="intro _to_fosm_files/schur.jpg" style="float: left; width: 5%; margin-right: 1%; margin-bottom: 0.5em;">

### <center> $\underbrace{\overline{\boldsymbol{\Sigma}}_{\boldsymbol{\theta}}}_{\substack{\text{what we} \\ \text{know now}}} = \underbrace{\boldsymbol{\Sigma}_{\boldsymbol{\theta}}}_{\substack{\text{what we} \\ \text{knew}}} - \underbrace{\boldsymbol{\Sigma}_{\boldsymbol{\theta}}\bf{J}^T\left[\bf{J}\boldsymbol{\Sigma}_{\boldsymbol{\theta}}\bf{J}^T + \boldsymbol{\Sigma}_{\boldsymbol{\epsilon}}\right]^{-1}\bf{J}\boldsymbol{\Sigma}_{\boldsymbol{\theta}}}_{\text{what we learned}}$ </center>

### Highlights:
1. There are no parameter values or observation values in these equations!
2. "us + data" = $\overline{\Sigma}_{\theta}$; "us" = $\Sigma_{\theta}$. This accounts for information from both data and expert knowledge.
3. The '-' on the right-hand-side shows that we are (hopefully) collapsing the probability manifold in parameter space by "learning" from the data. Or put another way, we are subtracting from the uncertainty we started with (we started with the Prior uncertainty)
4. Uncertainty in our measurements of the world is encapsulated in $\Sigma_{\epsilon}$. If the "observations" are highly uncertain, then parameter "learning" decreases because $\Sigma_{\epsilon}$ is in the denominator. Put another way, if our measured data are made (assumed) to be accurate and precise, then uncertainty associated with the parameters that are constrained by these measured data is reduced - we "learn" more. 
5. What quantities are needed? $\bf{J}$, $\boldsymbol{\Sigma}_{\theta}$, and $\boldsymbol{\Sigma}_{\epsilon}$
6. The diagonal of $\Sigma_{\theta}$ and $\overline{\Sigma}_{\theta}$ are the Prior and Posterior uncertainty (variance) of each adjustable parameter

# But what about forecasts? 

<img src="intro _to_fosm_files/jacobi.jpg" style="float: left; width: 5%; margin-right: 1%; margin-bottom: 0.5em;">
<img src="intro _to_fosm_files/gauss.jpg" style="float: left; width: 5%; margin-right: 1%; margin-bottom: 0.5em;">


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

<img src="intro _to_fosm_files/workflow.png" style="float: left; width: 50%; margin-right: 1%; margin-bottom: 0.5em;">



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


    ---------------------------------------------------------------------------

    Exception                                 Traceback (most recent call last)

    Input In [4], in <cell line: 1>()
    ----> 1 pst = pyemu.Pst(os.path.join(working_dir, pst_name))
    

    File D:\Workspace\hugm0001\github\GMDSI_notebooks_fork\tutorials\part1_10_intro_to_fosm\..\..\dependencies\pyemu\pst\pst_handler.py:119, in Pst.__init__(self, filename, load, resfile)
        117 if load:
        118     if not os.path.exists(filename):
    --> 119         raise Exception("pst file not found:{0}".format(filename))
        121     self.load(filename)
    

    Exception: pst file not found:master_pp\freyberg_pp.pst


### Let's look at the parameter uncertainty summary written by pestpp:

PEST++GLM records a parameter uncertainty file named _casename.par.usum.csv_. It records the prior and posterior means, bounds and standard deviations.


```python
df = pd.read_csv(os.path.join(working_dir,pst_name.replace(".pst",".par.usum.csv")),index_col=0)
df.tail()
```


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    Input In [5], in <cell line: 1>()
    ----> 1 df = pd.read_csv(os.path.join(working_dir,pst_name.replace(".pst",".par.usum.csv")),index_col=0)
          2 df.tail()
    

    File D:\Workspace\hugm0001\anaconda\envs\gmdsitut\lib\site-packages\pandas\util\_decorators.py:311, in deprecate_nonkeyword_arguments.<locals>.decorate.<locals>.wrapper(*args, **kwargs)
        305 if len(args) > num_allow_args:
        306     warnings.warn(
        307         msg.format(arguments=arguments),
        308         FutureWarning,
        309         stacklevel=stacklevel,
        310     )
    --> 311 return func(*args, **kwargs)
    

    File D:\Workspace\hugm0001\anaconda\envs\gmdsitut\lib\site-packages\pandas\io\parsers\readers.py:680, in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, squeeze, prefix, mangle_dupe_cols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, error_bad_lines, warn_bad_lines, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options)
        665 kwds_defaults = _refine_defaults_read(
        666     dialect,
        667     delimiter,
       (...)
        676     defaults={"delimiter": ","},
        677 )
        678 kwds.update(kwds_defaults)
    --> 680 return _read(filepath_or_buffer, kwds)
    

    File D:\Workspace\hugm0001\anaconda\envs\gmdsitut\lib\site-packages\pandas\io\parsers\readers.py:575, in _read(filepath_or_buffer, kwds)
        572 _validate_names(kwds.get("names", None))
        574 # Create the parser.
    --> 575 parser = TextFileReader(filepath_or_buffer, **kwds)
        577 if chunksize or iterator:
        578     return parser
    

    File D:\Workspace\hugm0001\anaconda\envs\gmdsitut\lib\site-packages\pandas\io\parsers\readers.py:933, in TextFileReader.__init__(self, f, engine, **kwds)
        930     self.options["has_index_names"] = kwds["has_index_names"]
        932 self.handles: IOHandles | None = None
    --> 933 self._engine = self._make_engine(f, self.engine)
    

    File D:\Workspace\hugm0001\anaconda\envs\gmdsitut\lib\site-packages\pandas\io\parsers\readers.py:1217, in TextFileReader._make_engine(self, f, engine)
       1213     mode = "rb"
       1214 # error: No overload variant of "get_handle" matches argument types
       1215 # "Union[str, PathLike[str], ReadCsvBuffer[bytes], ReadCsvBuffer[str]]"
       1216 # , "str", "bool", "Any", "Any", "Any", "Any", "Any"
    -> 1217 self.handles = get_handle(  # type: ignore[call-overload]
       1218     f,
       1219     mode,
       1220     encoding=self.options.get("encoding", None),
       1221     compression=self.options.get("compression", None),
       1222     memory_map=self.options.get("memory_map", False),
       1223     is_text=is_text,
       1224     errors=self.options.get("encoding_errors", "strict"),
       1225     storage_options=self.options.get("storage_options", None),
       1226 )
       1227 assert self.handles is not None
       1228 f = self.handles.handle
    

    File D:\Workspace\hugm0001\anaconda\envs\gmdsitut\lib\site-packages\pandas\io\common.py:789, in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
        784 elif isinstance(handle, str):
        785     # Check whether the filename is to be opened in binary mode.
        786     # Binary mode does not support 'encoding' and 'newline'.
        787     if ioargs.encoding and "b" not in ioargs.mode:
        788         # Encoding
    --> 789         handle = open(
        790             handle,
        791             ioargs.mode,
        792             encoding=ioargs.encoding,
        793             errors=errors,
        794             newline="",
        795         )
        796     else:
        797         # Binary mode
        798         handle = open(handle, ioargs.mode)
    

    FileNotFoundError: [Errno 2] No such file or directory: 'master_pp\\freyberg_pp.par.usum.csv'


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


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Input In [6], in <cell line: 1>()
    ----> 1 par = pst.parameter_data
          2 df_paru = pd.read_csv(os.path.join(working_dir,pst_name.replace(".pst",".par.usum.csv")),index_col=0)
          4 fig, axes=plt.subplots(1,len(pst.adj_par_groups),figsize=(15,5))
    

    NameError: name 'pst' is not defined


### There is a similar file for forecasts:
_casename.pred.usum.csv_


```python
axes = pyemu.plot_utils.plot_summary_distributions(os.path.join(working_dir,pst_name.replace(".pst",".pred.usum.csv")),subplots=True)
```


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    Input In [7], in <cell line: 1>()
    ----> 1 axes = pyemu.plot_utils.plot_summary_distributions(os.path.join(working_dir,pst_name.replace(".pst",".pred.usum.csv")),subplots=True)
    

    File D:\Workspace\hugm0001\github\GMDSI_notebooks_fork\tutorials\part1_10_intro_to_fosm\..\..\dependencies\pyemu\plot\plot_utils.py:80, in plot_summary_distributions(df, ax, label_post, label_prior, subplots, figsize, pt_color)
         77 import matplotlib.pyplot as plt
         79 if isinstance(df, str):
    ---> 80     df = pd.read_csv(df, index_col=0)
         81 if ax is None and not subplots:
         82     fig = plt.figure(figsize=figsize)
    

    File D:\Workspace\hugm0001\anaconda\envs\gmdsitut\lib\site-packages\pandas\util\_decorators.py:311, in deprecate_nonkeyword_arguments.<locals>.decorate.<locals>.wrapper(*args, **kwargs)
        305 if len(args) > num_allow_args:
        306     warnings.warn(
        307         msg.format(arguments=arguments),
        308         FutureWarning,
        309         stacklevel=stacklevel,
        310     )
    --> 311 return func(*args, **kwargs)
    

    File D:\Workspace\hugm0001\anaconda\envs\gmdsitut\lib\site-packages\pandas\io\parsers\readers.py:680, in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, squeeze, prefix, mangle_dupe_cols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, error_bad_lines, warn_bad_lines, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options)
        665 kwds_defaults = _refine_defaults_read(
        666     dialect,
        667     delimiter,
       (...)
        676     defaults={"delimiter": ","},
        677 )
        678 kwds.update(kwds_defaults)
    --> 680 return _read(filepath_or_buffer, kwds)
    

    File D:\Workspace\hugm0001\anaconda\envs\gmdsitut\lib\site-packages\pandas\io\parsers\readers.py:575, in _read(filepath_or_buffer, kwds)
        572 _validate_names(kwds.get("names", None))
        574 # Create the parser.
    --> 575 parser = TextFileReader(filepath_or_buffer, **kwds)
        577 if chunksize or iterator:
        578     return parser
    

    File D:\Workspace\hugm0001\anaconda\envs\gmdsitut\lib\site-packages\pandas\io\parsers\readers.py:933, in TextFileReader.__init__(self, f, engine, **kwds)
        930     self.options["has_index_names"] = kwds["has_index_names"]
        932 self.handles: IOHandles | None = None
    --> 933 self._engine = self._make_engine(f, self.engine)
    

    File D:\Workspace\hugm0001\anaconda\envs\gmdsitut\lib\site-packages\pandas\io\parsers\readers.py:1217, in TextFileReader._make_engine(self, f, engine)
       1213     mode = "rb"
       1214 # error: No overload variant of "get_handle" matches argument types
       1215 # "Union[str, PathLike[str], ReadCsvBuffer[bytes], ReadCsvBuffer[str]]"
       1216 # , "str", "bool", "Any", "Any", "Any", "Any", "Any"
    -> 1217 self.handles = get_handle(  # type: ignore[call-overload]
       1218     f,
       1219     mode,
       1220     encoding=self.options.get("encoding", None),
       1221     compression=self.options.get("compression", None),
       1222     memory_map=self.options.get("memory_map", False),
       1223     is_text=is_text,
       1224     errors=self.options.get("encoding_errors", "strict"),
       1225     storage_options=self.options.get("storage_options", None),
       1226 )
       1227 assert self.handles is not None
       1228 f = self.handles.handle
    

    File D:\Workspace\hugm0001\anaconda\envs\gmdsitut\lib\site-packages\pandas\io\common.py:789, in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
        784 elif isinstance(handle, str):
        785     # Check whether the filename is to be opened in binary mode.
        786     # Binary mode does not support 'encoding' and 'newline'.
        787     if ioargs.encoding and "b" not in ioargs.mode:
        788         # Encoding
    --> 789         handle = open(
        790             handle,
        791             ioargs.mode,
        792             encoding=ioargs.encoding,
        793             errors=errors,
        794             newline="",
        795         )
        796     else:
        797         # Binary mode
        798         handle = open(handle, ioargs.mode)
    

    FileNotFoundError: [Errno 2] No such file or directory: 'master_pp\\freyberg_pp.pred.usum.csv'


### Where do the prior parameter distributions come from?

Prior parameter distributions can come from one of two sources. 

1. If no other information is provided, PEST++GLM assumes that all adjustable parameters are statistically independent. In this case, by default, the prior standard deviation of each parameter is calculated as a quarter of the difference between its upper and lower bounds in the PEST control file.(This is the case here)
2. Alternatively, the name of a prior parameter covariance matrix file can be provided to the `parcov()` control variable.


### Where do the prior forecast distributions come from?

At the first iteration of the parameter estimation process, PEST++GLM calculates sensitivities based on initial parameter values. These are used to determine the prior parameter and forecast uncertianty.


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


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    Input In [8], in <cell line: 1>()
    ----> 1 sc = pyemu.Schur(jco=os.path.join(working_dir,pst_name.replace(".pst",".jcb")),verbose=False)
    

    File D:\Workspace\hugm0001\github\GMDSI_notebooks_fork\tutorials\part1_10_intro_to_fosm\..\..\dependencies\pyemu\sc.py:68, in Schur.__init__(self, jco, **kwargs)
         66 self.__posterior_prediction = None
         67 self.__posterior_parameter = None
    ---> 68 super(Schur, self).__init__(jco, **kwargs)
    

    File D:\Workspace\hugm0001\github\GMDSI_notebooks_fork\tutorials\part1_10_intro_to_fosm\..\..\dependencies\pyemu\la.py:125, in LinearAnalysis.__init__(self, jco, pst, parcov, obscov, predictions, ref_var, verbose, resfile, forecasts, sigma_range, scale_offset, **kwargs)
        123 self.log("pre-loading base components")
        124 if jco is not None:
    --> 125     self.__load_jco()
        126 if pst is not None:
        127     self.__load_pst()
    

    File D:\Workspace\hugm0001\github\GMDSI_notebooks_fork\tutorials\part1_10_intro_to_fosm\..\..\dependencies\pyemu\la.py:256, in LinearAnalysis.__load_jco(self)
        254     self.__jco = self.jco_arg
        255 elif isinstance(self.jco_arg, str):
    --> 256     self.__jco = self.__fromfile(self.jco_arg, astype=Jco)
        257 else:
        258     raise Exception(
        259         "linear_analysis.__load_jco(): jco_arg must "
        260         + "be a matrix object or a file name: "
        261         + str(self.jco_arg)
        262     )
    

    File D:\Workspace\hugm0001\github\GMDSI_notebooks_fork\tutorials\part1_10_intro_to_fosm\..\..\dependencies\pyemu\la.py:187, in LinearAnalysis.__fromfile(self, filename, astype)
        181 def __fromfile(self, filename, astype=None):
        182     """a private method to deduce and load a filename into a matrix object.
        183     Uses extension: 'jco' or 'jcb': binary, 'mat','vec' or 'cov': ASCII,
        184     'unc': pest uncertainty file.
        185 
        186     """
    --> 187     assert os.path.exists(filename), (
        188         "LinearAnalysis.__fromfile(): " + "file not found:" + filename
        189     )
        190     ext = filename.split(".")[-1].lower()
        191     if ext in ["jco", "jcb"]:
    

    AssertionError: LinearAnalysis.__fromfile(): file not found:master_pp\freyberg_pp.jcb


Now that seemed too easy, right?  Well, underhood the ``Schur`` object found the control file ("freyberg_pp.pst") and used it to build the prior parameter covariance matrix, $\boldsymbol{\Sigma}_{\theta}$, from the parameter bounds and the observation noise covariance matrix ($\boldsymbol{\Sigma}_{\epsilon}$) from the observation weights.  These are the ``Schur.parcov`` and ``Schur.obscov`` attributes.  

The ``Schur`` object also found the "++forecasts()" optional pestpp argument in the control, found the associated rows in the Jacobian matrix file and extracted those rows to serve as forecast sensitivity vectors:


```python
sc.pst.pestpp_options['forecasts']
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Input In [9], in <cell line: 1>()
    ----> 1 sc.pst.pestpp_options['forecasts']
    

    NameError: name 'sc' is not defined


### The Jacobian Matrix and Forecast Sensitivity Vectors

Recall that a Jacobian matrix looks at the changes in observations as a parameter is changed.  Therefore the Jacobian matrix has parameters in the columns and observations in the rows.  The bulk of the matrix is made up of the difference in  observations between a base run and a run where the parameter at the column head was perturbed (typically 1% from the base run value - controlled by the "parameter groups" info).  Now we'll plot out the Jacobian matrix as a `DataFrame`:


```python
sc.jco.to_dataframe().loc[sc.pst.nnz_obs_names,:].head()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Input In [10], in <cell line: 1>()
    ----> 1 sc.jco.to_dataframe().loc[sc.pst.nnz_obs_names,:].head()
    

    NameError: name 'sc' is not defined


This reports changes in observations to a change in a parameter.  We can report how  forecasts of interests change as the parameter is perturbed.  Note `pyemu` extracted the forecast rows from the Jacobian on instantiation:


```python
sc.forecasts.to_dataframe()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Input In [11], in <cell line: 1>()
    ----> 1 sc.forecasts.to_dataframe()
    

    NameError: name 'sc' is not defined


Each of these columns in a $\bf{y}$ vector used in the FOSM calculations...that's it! 

###  The prior parameter covariance matrix - $\boldsymbol{\Sigma}_{\theta}$

Because we have inherent uncertainty in the parameters, the forecasts also have uncertainty.  Here's what we have defined for parameter uncertainty - the Prior.  As discussed above, it was constructed on-the-fly from the parameter bounds in the control file: 


```python
sc.parcov.to_dataframe()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Input In [12], in <cell line: 1>()
    ----> 1 sc.parcov.to_dataframe()
    

    NameError: name 'sc' is not defined


> Page 463-464 in Anderson et al. (2015) spends some time on what is shown above.  

For our purposes, a diagonal Prior -  numbers only along the diagonal - shows that we expect the uncertainty for each parameter to only results from itself - there is no covariance with other parameters. The numbers themselves reflect "the innate parameter variability", and is input into the maths as a standard deviation around the parameter value.  This is called the "C(p) matrix of innate parameter variability" in PEST parlance.

> __IMPORTANT POINT__:  Again, how did PEST++ and pyEMU get these standard deviations shown in the diagonal?  From the *parameter bounds* that were specified for each parameter in the PEST control file.

### The  matrix  of observation noise - $C{\epsilon}$

Forecast uncertainty has to take into account the noise/uncertainty in the observations.   Similar to the parameter Prior - the $\Sigma_{\theta}$ matrix -, it is a covariance matrix of measurement error associated with the observations.  This is the same as  $\Sigma_{\epsilon}$ that we discussed above. For our Fryberg problem, sthe $C{\epsilon}$ matrix would look like:


```python
sc.obscov.to_dataframe().loc[sc.pst.nnz_obs_names,sc.pst.nnz_obs_names].head()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Input In [13], in <cell line: 1>()
    ----> 1 sc.obscov.to_dataframe().loc[sc.pst.nnz_obs_names,sc.pst.nnz_obs_names].head()
    

    NameError: name 'sc' is not defined


> __IMPORTANT POINT__:  How did PEST++ and pyEMU get these standard deviations shown in the diagonal?  From the *weights* that were specified for each observation in the PEST control file.

> __IMPORTANT POINT__: You can use FOSM in the "pre-calibration" state to design an objective function (e.g. weights) to maximize forecast uncertainty reduction.

> __IMPORTANT POINT__: In PEST++, if a given observation has a larger-than-expected residual, the variance of said observation is reset to the variance implied by the residual.  That is, the diagonal elements of $\Sigma_{\epsilon}$ are reset according to the residuals

## Posterior Parameter Uncertainty - ${\overline{\boldsymbol{\Sigma}}_{\boldsymbol{\theta}}} $

Okay, enough emphasis.  Here's the point.  When we apply FOSM using the matrices above, we can see how our uncertainty changes during calibration, first for parameters and then for forecasts. 

Here, we are updating parameter covariance following notional calibration as represented by the Jacobian matrix and both prior parameter and observation noise covariance matrices. 

In other words, given prior parameter uncertainty (expressed by $\boldsymbol{\Sigma}_{\theta}$) and the inherent noise in measurments (expressed by $C{\epsilon}$), we calculate the expected parameter uncertainty __after__ calibration. This assumes that _calibration achieves a fit comensurate with measurement noise, parameter linearity, etc_.

The posterior parameter covariance matrix is stored as a `pyemu.Cov` object in the `sc.posterior_parameter` attribute. The diagonal of this matrix contains the posterior __variance__ for each parameter. The off-diagonals the parameter covariances. 


```python
sc.posterior_parameter.to_dataframe().head()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Input In [14], in <cell line: 1>()
    ----> 1 sc.posterior_parameter.to_dataframe().head()
    

    NameError: name 'sc' is not defined


But...is calibration worth pursuing or not? Let's explore what the notional calibration is expected to do for parameter uncertainty. We accomplish this by comparing prior and posterior parameter uncertainty. Using `.get_parameter_summary()` makes this easy:


```python
df = sc.get_parameter_summary()
df.head()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Input In [15], in <cell line: 1>()
    ----> 1 df = sc.get_parameter_summary()
          2 df.head()
    

    NameError: name 'sc' is not defined


We can plot that up:


```python
df.percent_reduction.plot(kind="bar", figsize=(15,3));
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Input In [16], in <cell line: 1>()
    ----> 1 df.percent_reduction.plot(kind="bar", figsize=(15,3))
    

    NameError: name 'df' is not defined


### Do these results make sense?  Why are some parameters unaffected by calibration?

As the name suggests, the `percent_reduction` column shows the  percentage decrease in uncertainty expected through calibration for each parameter.

From the plot above we can see that calibrating the model with available data definetly reduces uncertainty of some parameters. Some parameters are informed by observation data...however calibration does not affect all parameters equally. Available observation data does not contain information that affects these parameters. Calibration will not help us reduce their uncertainty.

##  Forecast Uncertainty

So far we have seen that some parameter uncertainty will be reduced. Uncertainty for other parameters will not. That's great and all, but what we really care about are our forecast uncertainties. Do the parameters that are informed by calibration affect the forecast of interest? And will calibrating reduce the uncertainty of these forecast?

Let's examine the prior and posterior variance of our forecasts. Recall that they are recorded as observations in the `Pst` control file and also listed in the pest++ `forecast` control variable:


```python
forecasts = sc.pst.forecast_names
forecasts
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Input In [17], in <cell line: 1>()
    ----> 1 forecasts = sc.pst.forecast_names
          2 forecasts
    

    NameError: name 'sc' is not defined


As before, `pyemu` has already done much of the heavy-lifting. We can get a summary of the forecast prior and posterior variances with `.get_forecast_summary()`:


```python
df = sc.get_forecast_summary()
df
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Input In [18], in <cell line: 1>()
    ----> 1 df = sc.get_forecast_summary()
          2 df
    

    NameError: name 'sc' is not defined


And we can make a cheeky little plot of that. As you can see, unsurprisingly some forecasts benefit more from calibration than others. So, depending on the foreacst of interest, calibration may or may not be worthwhile...


```python
# get the forecast summary then make a bar chart of the percent_reduction column
fig = plt.figure()
ax = plt.subplot(111)
ax = df.percent_reduction.plot(kind='bar',ax=ax,grid=True)
ax.set_ylabel("percent uncertainy\nreduction from calibration")
ax.set_xlabel("forecast")
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Input In [19], in <cell line: 4>()
          2 fig = plt.figure()
          3 ax = plt.subplot(111)
    ----> 4 ax = df.percent_reduction.plot(kind='bar',ax=ax,grid=True)
          5 ax.set_ylabel("percent uncertainy\nreduction from calibration")
          6 ax.set_xlabel("forecast")
    

    NameError: name 'df' is not defined



    
![png](intro_to_fosm_files/intro_to_fosm_57_1.png)
    


## Parameter contribution to forecast uncertainty

Information flows from observations to parameters and then out to forecasts. Information contained in observation data constrains parameter uncertainty, which in turn constrains forecast uncertainty. For a given forecast, we can evaluate which parameter contributes the most to uncertainty. This is accomplished by assuming a parameter (or group of parameters) is perfectly known and then assessing forecast uncertainty under that assumption. Comparing uncertainty obtained in this manner, to the forecast uncertainty under the base assumption (in which no parameter is perfectly known), the contribution from that parameter (or parameter group) is obtained. 

Now, this is a pretty big assumption - in practice a parameter is never perfectly known. Nevertheless, this metric can provide usefull insights into the flow of information from data to forecast uncertainty, which can help guide data assimilation design as well as future data collection efforts. 

In `pyemu` we can  evaluate parameter contributions to forecast uncertainty with groups of parameters by type using `.get_par_group_contribution()`:


```python
par_contrib = sc.get_par_group_contribution()
par_contrib.head()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Input In [20], in <cell line: 1>()
    ----> 1 par_contrib = sc.get_par_group_contribution()
          2 par_contrib.head()
    

    NameError: name 'sc' is not defined


We can see the relatve contribution by normalizing to the base case (e.g. in which no parameters/groups are perfectly known):


```python
base = par_contrib.loc["base",:]
par_contrib = 100.0 * (base - par_contrib) / base
par_contrib.sort_index().head()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Input In [21], in <cell line: 1>()
    ----> 1 base = par_contrib.loc["base",:]
          2 par_contrib = 100.0 * (base - par_contrib) / base
          3 par_contrib.sort_index().head()
    

    NameError: name 'par_contrib' is not defined


Understanding the links between parameters and forecast uncertainties can be usefull - in particular to gain insight into the system dynamics. But we are still missing a step to understand what _observation_ data affects the forecast. It is often more straightforward to quantify how observation information imapcts forecast uncertianty so that we can explore the worth of observation data directly.

# Data worth analysis

> __Note__: We will _not_ demonstrate data worth analysis here. See the respective notebook in Part2 of these tutorials.

The worth of data is measured by their ability to reduce the uncertainties of model predictions that we care about. Linear analysis is particularly useful for exploring data worth. This is because the equations that it uses to calculate predictive uncertainty do not include terms that represent the actual values of observations or of parameters; only sensitivities of model outputs to parameters are required. Therefore, linear analysis can be used to assess the ability (or otherwise) of yet-ungathered data to reduce the uncertainties of decision-critical predictions.

### <center> This is __Huge__. Let me say it again.<center>

#### <center>  We can assess the relative worth of an observation ___without knowing the value of the observation___. </center>


This means that potential field measurements that correspond to one or many outputs of a model can be assessed for their worth. For example, it is possible to assess the worth of observations of head in every single model cell at every time step of a model run with a relatively small computational burden. This makes linear analysis a useful tool for designing and comparing strategies for data-collection, when data acquisition seeks to reduce the uncertainties of one or a number of decision-critical predictions. 

There are two main applications for data worth analysis:
 1.	ranking of the relative worth of existing observations by calculating predictive uncertainty with selected individual or combined observations removed from a calibration dataset. 
 2.	ranking of the relative worth of __potential__ new observations by calculating predictive uncertainty with selected individual or combined observations added to an existing calibration dataset.
