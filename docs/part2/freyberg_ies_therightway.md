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
import psutil

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




    'freyberg6_template'



Load the PEST control file as a `Pst` object.


```python
pst_path = os.path.join(t_d, 'freyberg_mf6.pst')
pst = pyemu.Pst(pst_path)
pst.observation_data.columns
```




    Index(['obsnme', 'obsval', 'weight', 'obgnme', 'oname', 'otype', 'usecol',
           'time', 'i', 'j', 'totim', 'observed'],
          dtype='object')



Check that we are at the right stage to run ies:


```python
# check to see if obs&weights notebook has been run
if not pst.observation_data.observed.sum()>0:
    raise Exception("You need to run the '/part2_2_obs_and_weights/freyberg_obs_and_weights.ipynb' notebook")
```

Load the prior parameter ensemble we generated previously:


```python
[f for f in os.listdir(t_d) if f.endswith(".jcb")]
```




    ['obs_cov.jcb', 'obs_cov_diag.jcb', 'prior_cov.jcb', 'prior_pe.jcb']




```python
pe = pyemu.ParameterEnsemble.from_binary(pst=pst,filename=os.path.join(t_d,"prior_pe.jcb"))
pe.shape
```




    (50, 29653)




```python
#obscov = pyemu.Cov.from_binary(os.path.join(t_d, 'obs_cov.jcb'))
#oe = pyemu.ObservationEnsemble.from_gaussian_draw(pst, cov=obscov, num_reals=50)
#oe.to_csv(os.path.join(t_d, 'oe.csv'))
```

### 3. Run PESTPP-IES in Parallel

Here we go...this is gonna epic!

We need to tell PESTPP-IES to use the geostatistical prior parameter ensemble we generated previously. And lets just use 50 realizations to speed things up (feel free to use less or more - choose your own adventure!)


```python
obscov = pyemu.Cov.from_binary(os.path.join(t_d, 'obs_cov.jcb'))
oe = pyemu.ObservationEnsemble.from_gaussian_draw(pst, cov=obscov, num_reals=50)
oe.to_csv(os.path.join(t_d, 'oe.csv'))
```


```python
pst.pestpp_options
```




    {'forecasts': 'oname:sfr_otype:lst_usecol:tailwater_time:4383.5,oname:sfr_otype:lst_usecol:headwater_time:4383.5,oname:hds_otype:lst_usecol:trgw-0-9-1_time:4383.5,part_time'}




```python
pst.pestpp_options['ies_parameter_ensemble'] = 'prior_pe.jcb'
pst.pestpp_options["ies_num_reals"] = 50 #enough?
pst.pestpp_options["ies_bad_phi_sigma"] = 2.0
pst.pestpp_options["overdue_giveup_fac"] = 1.5
# the time to calc each lambda upgrade is substantial, so just use one...
pst.pestpp_options["ies_lambda_mults"] = 1.0
pst.pestpp_options["ies_save_rescov"] = True
pst.pestpp_options["ies_drop_conflicts"] = True
pst.pestpp_options["ies_pdc_sigma_distance"] = 2.0
#pst.pestpp_options["ies_no_noise"] = True
pst.pestpp_options["ies_observation_ensemble"] = "oe.csv"
pst.pestpp_options['ies_num_threads'] = 10 # make sure it is less than the number of phisycal cores on your machine
pst.pestpp_options["ies_autoadaloc"] = True
```


```python
pst.control_data.noptmax = 3
```


```python

pst.write(os.path.join(t_d, 'freyberg_mf6.pst'))
```

    noptmax:3, npar_adj:29653, nnz_obs:144
    


```python
num_workers = psutil.cpu_count(logical=False) #update this according to your resources
```


```python
pst.write(os.path.join(t_d, 'freyberg_mf6.pst'))
m_d = os.path.join('master_ies')
```

    noptmax:3, npar_adj:29653, nnz_obs:144
    


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

Finally, let's plot the obs vs sim timeseries - everyone's fav!


```python
def plot_tseries_ensembles(pr_oe, pt_oe, noise, onames=["hds","sfr"]):
    pst.try_parse_name_metadata()
    obs = pst.observation_data.copy()
    obs = obs.loc[obs.oname.apply(lambda x: x in onames)]
    obs = obs.loc[obs.obgnme.apply(lambda x: x in pst.nnz_obs_groups),:]
    obs.obgnme.unique()

    ogs = obs.obgnme.unique()
    fig,axes = plt.subplots(len(ogs),1,figsize=(10,2*len(ogs)))
    ogs.sort()
    for ax,og in zip(axes,ogs):
        oobs = obs.loc[obs.obgnme==og,:].copy()
        oobs.loc[:,"time"] = oobs.time.astype(float)
        oobs.sort_values(by="time",inplace=True)
        tvals = oobs.time.values
        onames = oobs.obsnme.values
        [ax.plot(tvals,pr_oe.loc[i,onames].values,"0.5",lw=0.5,alpha=0.5) for i in pr_oe.index]
        [ax.plot(tvals,pt_oe.loc[i,onames].values,"b",lw=0.5,alpha=0.5) for i in pt_oe.index]
        
        oobs = oobs.loc[oobs.weight>0,:]
        tvals = oobs.time.values
        onames = oobs.obsnme.values
        [ax.plot(tvals,noise.loc[i,onames].values,"r",lw=0.5,alpha=0.5) for i in noise.index]
        ax.plot(oobs.time,oobs.obsval,"r-",lw=2)
        ax.set_title(og,loc="left")
    fig.tight_layout()
    return fig
```


```python
# checking which obs groups have poor fit
df = pd.read_csv(os.path.join('master_ies_therightway','freyberg_mf6.phi.group.csv'))
df.loc[df.iteration==pst.control_data.noptmax, pst.nnz_obs_groups].sum()
```




    oname:hds_otype:lst_usecol:trgw-0-26-6       380.025120
    oname:hds_otype:lst_usecol:trgw-0-3-8       1260.970421
    oname:hds_otype:lst_usecol:trgw-2-26-6       352.995330
    oname:hds_otype:lst_usecol:trgw-2-3-8       1303.093360
    oname:sfr_otype:lst_usecol:gage-1            161.478582
    oname:sfrtd_otype:lst_usecol:gage-1          198.807602
    oname:hdstd_otype:lst_usecol:trgw-0-26-6     147.905511
    oname:hdstd_otype:lst_usecol:trgw-0-3-8      995.461501
    oname:hdstd_otype:lst_usecol:trgw-2-26-6     224.833592
    oname:hdstd_otype:lst_usecol:trgw-2-3-8     1019.513049
    oname:hdsvd_otype:lst_usecol:trgw-0-26-6     546.966253
    oname:hdsvd_otype:lst_usecol:trgw-0-3-8       83.519813
    dtype: float64




```python

```


```python
fig = plot_tseries_ensembles(pr_oe, pt_oe, noise, onames=["hds","sfr",])# 'hdsv', 'hdstd', 'sfrtd'])
```


    
![png](freyberg_ies_therightway_files/freyberg_ies_therightway_26_0.png)
    



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
            bins=np.histogram(pr_oe.loc[:,forecast], bins=20)[1] #get the bin edges
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
            axes.flatten()[1].text(0.5,1.2,"Last Attempt", transform=axes.flatten()[1].transAxes, weight='bold', fontsize=12, horizontalalignment='center')
        fig.tight_layout()
        return fig

```


```python
fig = plot_forecast_hist_compare(pt_oe=pt_oe, pr_oe=pr_oe)
```


    
![png](freyberg_ies_therightway_files/freyberg_ies_therightway_28_0.png)
    



```python
res_cov_to_mod_file = os.path.join(t_d,'freyberg_mf6_shrunk_res_to_mod.csv')
if not os.path.exists(res_cov_to_mod_file):
    res_cov_file = os.path.join(m_d,"freyberg_mf6.{0}.shrunk_res.cov".format(pst.control_data.noptmax))
    assert os.path.exists(res_cov_file)
    shutil.copy2(res_cov_file, res_cov_to_mod_file)
    
res_cov = pyemu.Cov.from_ascii(res_cov_to_mod_file)
x = res_cov.to_pearson().x.copy()
x[np.abs(x) < 0.2] = np.NaN
x[x==1.0] = np.NaN

fig,ax = plt.subplots(1,1,figsize=(10,10))
cb = ax.imshow(x,cmap="plasma")
plt.colorbar(cb, shrink=0.8)
```




    <matplotlib.colorbar.Colorbar at 0x26910de6c40>




    
![png](freyberg_ies_therightway_files/freyberg_ies_therightway_29_1.png)
    



```python
obs=pst.observation_data
minvar = ((1./obs.loc[res_cov.names,"weight"])**2).min()
shrink = np.zeros(res_cov.shape)
np.fill_diagonal(shrink,minvar)
lamb = 2. / (pt_oe.shape[0] + 1)
lamb = 0.2
print(lamb)
shrunk = (lamb * shrink) + ((1.-lamb) * res_cov.x)
shrunk = pyemu.Cov(x=shrunk,names=res_cov.names)

# write residual covariance matrix to file
shrunk.to_ascii(os.path.join(t_d,"shrunk_obs.cov"))
x = shrunk.to_pearson().x.copy()
x[x==0.0] = np.NaN
plt.imshow(x,cmap="plasma")
plt.colorbar(cb, shrink=0.8);
```

    0.2
    

    C:\Users\hugm0001\AppData\Local\Temp\ipykernel_15320\950571858.py:16: MatplotlibDeprecationWarning: Starting from Matplotlib 3.6, colorbar() will steal space from the mappable's axes, rather than from the current axes, to place the colorbar.  To silence this warning, explicitly pass the 'ax' argument to colorbar().
    


    
![png](freyberg_ies_therightway_files/freyberg_ies_therightway_30_2.png)
    



```python
pst.pestpp_options["ies_no_noise"] = False
pst.pestpp_options["ies_drop_conflicts"] = False
pst.pestpp_options["obscov"] = "shrunk_obs.cov"
pst.pestpp_options["ies_group_draws"] = False
```


```python
# run for a single iteration?
pst.control_data.noptmax = 3
```


```python
pst.write(os.path.join(t_d,"freyberg_mf6.pst"))
```

    noptmax:3, npar_adj:29653, nnz_obs:144
    


```python
pyemu.os_utils.start_workers(t_d,"pestpp-ies","freyberg_mf6.pst",num_workers=num_workers,master_dir=m_d,worker_root='.')
```


```python
pr_oe_rescov = pyemu.ObservationEnsemble.from_csv(pst=pst,filename=os.path.join(m_d,"freyberg_mf6.0.obs.csv"))
pt_oe_rescov = pyemu.ObservationEnsemble.from_csv(pst=pst,filename=os.path.join(m_d,"freyberg_mf6.{0}.obs.csv".format(pst.control_data.noptmax)))
noise_rescov = pyemu.ObservationEnsemble.from_csv(pst=pst,filename=os.path.join(m_d,"freyberg_mf6.obs+noise.csv"))
```


```python

```


```python
fig = plot_tseries_ensembles(pr_oe_rescov, pt_oe_rescov, noise_rescov, onames=["hds","sfr"])
```


    
![png](freyberg_ies_therightway_files/freyberg_ies_therightway_37_0.png)
    



```python
fig = plot_forecast_hist_compare(pt_oe=pt_oe_rescov,
                                pr_oe=pr_oe_rescov,
                                last_pt_oe=pt_oe,
                                last_prior=pr_oe
                                )
```


    
![png](freyberg_ies_therightway_files/freyberg_ies_therightway_38_0.png)
    


Autoadaloc isnt perfect...


```python
m_d=os.path.join('master_ies_therightway')
pe_pr = pd.read_csv(os.path.join(m_d,"freyberg_mf6.0.par.csv"),index_col=0)
pe_pt = pd.read_csv(os.path.join(m_d,"freyberg_mf6.{0}.par.csv".format(pst.control_data.noptmax)),index_col=0)
par = pst.parameter_data
pdict = par.groupby("pargp").groups

d = [i for i in pst.par_groups if any(i.startswith(s) for s in ['ne'])]
pdict = {k:pdict[k] for k in pdict if k in d}

pyemu.plot_utils.ensemble_helper({"0.5":pe_pr,"b":pe_pt},plot_cols=pdict)
```


    <Figure size 576x756 with 0 Axes>



    
![png](freyberg_ies_therightway_files/freyberg_ies_therightway_40_1.png)
    



    
![png](freyberg_ies_therightway_files/freyberg_ies_therightway_40_2.png)
    



```python

```
