import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,os.path.join("..", "..", "dependencies"))
import pyemu
import flopy
assert "dependencies" in flopy.__file__
assert "dependencies" in pyemu.__file__
sys.path.insert(0,"..")
import herebedragons as hbd


WORKING_DIR = 'freyberg_mf6'
MODEL_NAM = "freyberg.nam"
PST_NAME = 'freyberg.pst'
NUM_STEPS_RESPSURF = 40

def run_respsurf(par_names=None, pstfile='freyberg.pst', WORKING_DIR='freyberg_mf6'):
    pst = pyemu.Pst(os.path.join(WORKING_DIR,pstfile))
    par = pst.parameter_data
    pst.pestpp_options['sweep_parameter_csv_file'] = pstfile.replace('.pst', "sweep_in.csv")
    pst.pestpp_options['sweep_output_csv_file'] = pstfile.replace('.pst', "sweep_out.csv")
    pst.write(os.path.join(WORKING_DIR,pstfile))
    icount = 0
    if par_names is None:
        parnme1 = pst.adj_par_names[0]
        parnme2 = pst.adj_par_names[1]
    else:
        parnme1 = par_names[0]
        parnme2 = par_names[1]
    p1 = np.linspace(par.loc[parnme1,"parlbnd"],par.loc[parnme1,"parubnd"],NUM_STEPS_RESPSURF).tolist()
    p2 = np.linspace(par.loc[parnme2,"parlbnd"],par.loc[parnme2,"parubnd"],NUM_STEPS_RESPSURF).tolist()
    p1_vals,p2_vals = [],[]
    for p in p1:
        p1_vals.extend(list(np.zeros(NUM_STEPS_RESPSURF)+p))
        p2_vals.extend(p2)
    df = pd.DataFrame({parnme1:p1_vals,parnme2:p2_vals})
    for cp in par.parnme.values:
        if cp not in df.columns:
            df[cp] = par.loc[cp].parval1
    df.to_csv(os.path.join(WORKING_DIR,pstfile.replace('.pst',"sweep_in.csv")))


    num_workers=8
    pyemu.os_utils.start_workers(WORKING_DIR, # the folder which contains the "template" PEST dataset
                            'pestpp-swp', #the PEST software version we want to run
                            pstfile, # the control file to use with PEST
                            num_workers=num_workers, #how many agents to deploy
                            worker_root='.', #where to deploy the agent directories; relative to where python is running
                            master_dir=WORKING_DIR, reuse_master=True #the manager directory
                            )
    return

def plot_response_surface(parnames=['hk1','rch0'], pstfile='freyberg.pst', WORKING_DIR='freyberg_mf6',
                          nanthresh=None,alpha=0.5, label=True, maxresp=None,
                          figsize=(5,5),levels=None, cmap="nipy_spectral"):
    p1,p2 = parnames
    df_in = pd.read_csv(os.path.join(WORKING_DIR, pstfile.replace('.pst',"sweep_in.csv")))
    df_out = pd.read_csv(os.path.join(WORKING_DIR, pstfile.replace('.pst',"sweep_out.csv")))
    resp_surf = np.zeros((NUM_STEPS_RESPSURF, NUM_STEPS_RESPSURF))
    p1_values = df_in[p1].unique()
    p2_values = df_in[p2].unique()
    c = 0
    for i, v1 in enumerate(p1_values):
        for j, v2 in enumerate(p2_values):
            resp_surf[j, i] = df_out.loc[c, "phi"]
            c += 1
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    X, Y = np.meshgrid(p1_values, p2_values)
    if nanthresh is not None:
        resp_surf = np.ma.masked_where(resp_surf > nanthresh, resp_surf)
    if maxresp is None:
        maxresp = np.max(resp_surf)
    if levels is None:
        levels = np.array([0.001, 0.01, 0.02, 0.05, .1, .2, .5])*maxresp
    
    import matplotlib.colors as colors
    vmin=np.min(resp_surf)
    vmax=maxresp
    p = ax.pcolor(X, Y, resp_surf, alpha=alpha, cmap=cmap,# vmin=vmin, vmax=vmax,
                norm=colors.LogNorm(vmin=vmin, vmax=vmax))
    plt.colorbar(p)
    c = ax.contour(X, Y, resp_surf,
                   levels=levels,
                   colors='k', alpha=0.5)
    plt.title('min $\\Phi$ = {0:.2f}'.format(np.nanmin(resp_surf)))
    if label:
        plt.clabel(c)
    ax.set_xlim(p1_values.min(), p1_values.max())
    ax.set_ylim(p2_values.min(), p2_values.max())
    ax.set_xlabel(p1)
    ax.set_ylabel(p2)
    return fig, ax, resp_surf

def add_trajectory_to_plot(fig,ax, title, working_dir='freyberg_mf6', pst_name='freyberg.pst', pars2plot=['hk1','rch0']):
    obfun = pd.read_csv(os.path.join(working_dir,pst_name.replace('.pst','.iobj')))
    pars=pd.read_csv(os.path.join(working_dir,pst_name.replace('.pst','.ipar')))
    ax.plot(pars[pars2plot[0]].values,pars[pars2plot[1]].values, 'kx-')
    ax.set_title(title)
    return pars, obfun