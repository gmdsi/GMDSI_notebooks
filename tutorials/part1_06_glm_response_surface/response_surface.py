import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
# sys.path.insert(0,os.path.join("..", "..", "dependencies"))
import pyemu
import flopy
assert "dependencies" in flopy.__file__
assert "dependencies" in pyemu.__file__
sys.path.insert(0,"..")
import herebedragons as hbd
import flopy.plot.styles as wtf

WORKING_DIR = 'freyberg_mf6'
MODEL_NAM = "freyberg.nam"
PST_NAME = 'freyberg.pst'
NUM_STEPS_RESPSURF = 40

def run_respsurf(par_names=None, pstfile='freyberg.pst', WORKING_DIR='freyberg_mf6',num_workers=8,port=4004):
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


    pyemu.os_utils.start_workers(WORKING_DIR, # the folder which contains the "template" PEST dataset
                            'pestpp-swp', #the PEST software version we want to run
                            pstfile, # the control file to use with PEST
                            num_workers=num_workers, #how many agents to deploy
                            worker_root='.', #where to deploy the agent directories; relative to where python is running
                            master_dir=WORKING_DIR,
                            port=port,
                            reuse_master=True #the manager directory
                            )
    return

def plot_response_surface(parnames=['hk1','rch0'], pstfile='freyberg.pst', WORKING_DIR='freyberg_mf6',
                          nanthresh=None,alpha=0.5, label=True, maxresp=None,
                          figsize=(5,5),levels=None, cmap="magma",title=None,ax=None):
    
    with wtf.USGSPlot():
        font = {'size'   : 12}

        mpl.rc('font', **font)
        p1,p2 = parnames
        df_in = pd.read_csv(os.path.join(WORKING_DIR, pstfile.replace('.pst',"sweep_in.csv")))
        df_out = pd.read_csv(os.path.join(WORKING_DIR, pstfile.replace('.pst',"sweep_out.csv")))
        resp_surf = np.zeros((NUM_STEPS_RESPSURF, NUM_STEPS_RESPSURF))
        p1_values = df_in[p1].unique()
        p2_values = df_in[p2].unique()
        c = 0
        minval=np.inf
        min_i = np.inf
        min_j = np.inf
        min1,min2 = np.inf,np.inf
        for i, v1 in enumerate(p1_values):
            for j, v2 in enumerate(p2_values):
                resp_surf[j, i] = df_out.loc[c, "phi"]
                if resp_surf[j, i] < minval:
                    min_i,min_j = i,j
                    min1 = v1
                    min2 = v2
                    minval = resp_surf[j, i]
                c += 1
        fig = None
        if ax is None:
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
        cb = plt.colorbar(p)
        cb.ax.get_yaxis().labelpad = 10
        cb.ax.tick_params(labelsize = 12)
        
        cb.set_label('Objective Function $\\Phi$', rotation=90, fontsize=12)
        c = ax.contour(X, Y, resp_surf,
                       levels=levels,
                       colors='k', alpha=0.5)
        ax.plot(min1,min2, 'r*',zorder=100,markersize=15)
        if title is None:
            ax.set_title('min $\\Phi$ = {0:.2f}'.format(np.nanmin(resp_surf)),fontsize=12)
        else:
            ax.set_title(title,fontsize=12)
        if label:
            plt.clabel(c)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        ax.set_xlim(p1_values.min(), p1_values.max())
        ax.set_ylim(p2_values.min(), p2_values.max())
        #ax.set_xlabel(p1)
        #ax.set_ylabel(p2)
        ax.set_xlabel('Hydraulic Conductivity', fontsize=12)
        ax.set_ylabel('Recharge Multiplier', fontsize=12)
        return fig, ax, resp_surf
        
def plot_ies_and_resp_par_forecast_results(resp_d,ies_d,pst,title=None,fig_name=None):
    r_inp = pd.read_csv(os.path.join(resp_d,"freybergsweep_in.csv"),index_col=0)
    r_out = pd.read_csv(os.path.join(resp_d,"freybergsweep_out.csv"),index_col=1)
    r_out.loc[:,"likelihood"] = 1.0/r_out.phi.values**2
    phidf = pd.read_csv(os.path.join(ies_d,"freyberg.phi.actual.csv"))
    iiter = int(phidf.iteration.max())
    print("using iter",iiter)
    pe = pd.read_csv(os.path.join(ies_d,"freyberg.{0}.par.csv".format(iiter)),index_col=0)
    oe_pt = pd.read_csv(os.path.join(ies_d,"freyberg.{0}.obs.csv".format(iiter)),index_col=0)
    oe_pr = pd.read_csv(os.path.join(ies_d,"freyberg.0.obs.csv"),index_col=0)
    r_inp.loc[:,"phi"] = r_out.likelihood
    
    fig, ax, resp_surf = plot_response_surface(figsize=(7,7),WORKING_DIR=resp_d,title=title) #maxresp=1e3,
    pes = []
    for i in range(iiter+1):
        fname = os.path.join(ies_d,"freyberg.{0}.par.csv".format(i))
        if not os.path.exists(fname):
            break
        pe = pd.read_csv(fname,index_col=0)    
        pes.append(pe)
    for real in pes[-1].index:
        xvals  = [pe.loc[real,"hk1"] for pe in pes]
        yvals  = [pe.loc[real,"rch0"] for pe in pes]
        ax.plot(xvals,yvals,marker=".",c="w",lw=0.5,markersize=3)
    xvals = pes[-1].loc[:,"hk1"].values
    yvals = pes[-1].loc[:,"rch0"].values
    ax.scatter(xvals,yvals,marker=".",c="b",s=15,zorder=5)
    plt.tight_layout()
    if fig_name is not None:
        plt.savefig(fig_name)
    plt.show()

    fig,axes = plt.subplots(2,1,figsize=(6,6))
    hk1 = r_inp.groupby("hk1").sum().loc[:,"phi"]
    rch0 = r_inp.groupby("rch0").sum().loc[:,"phi"]
    hk1_space = hk1.index[1] - hk1.index[0]
    rch0_space = rch0.index[1] - rch0.index[0]
    axes[0].bar(hk1.index,hk1.values,width=hk1_space,alpha=0.1,fc="0.5")
    axes[1].bar(rch0.index,rch0.values,width=rch0_space,alpha=0.1,fc="0.5")
    axt0 = plt.twinx(axes[0])
    axt0.hist(pe.loc[:,"hk1"].values,density=True,alpha=0.5,fc="b")
    axt1 = plt.twinx(axes[1])
    axt1.hist(pe.loc[:,"rch0"].values,density=True,alpha=0.5,fc="b")
    axes[0].set_title("hk1",loc="left")
    axes[1].set_title("rch0",loc="left")
    for ax in [axes[0],axes[1],axt0,axt1]:
        ax.set_yticks([])
    
    for forecast in pst.pestpp_options["forecasts"].split(","):
        fig,ax = plt.subplots(1,1,figsize=(6,3))
        #ax.hist(r_out.loc[:,forecast].values,weights=r_out.likelihood.values,alpha=0.5,fc="0.5",density=True)
        ax.hist(oe_pr.loc[:,forecast].values,alpha=0.5,fc="0.5",density=True)
        ax.hist(oe_pt.loc[:,forecast].values,alpha=0.5,fc="b",density=True)
        ax.set_yticks([])
        ax.set_title(forecast,loc="left")
        ylim = ax.get_ylim()
        #fval = pst.observation_data.loc[forecast,"obsval"]
        #ax.plot([fval,fval],ylim,"r-",lw=2)
def add_trajectory_to_plot(fig,ax, title, working_dir='freyberg_mf6', pst_name='freyberg.pst', pars2plot=['hk1','rch0']):
    obfun = pd.read_csv(os.path.join(working_dir,pst_name.replace('.pst','.iobj')))
    pars=pd.read_csv(os.path.join(working_dir,pst_name.replace('.pst','.ipar')))
    #ax.plot(pars[pars2plot[0]].values,pars[pars2plot[1]].values, 'kx-')
    ax.plot(pars[pars2plot[0]].values,pars[pars2plot[1]].values,'.-', color="w",lw=0.5)
    ax.plot(pars[pars2plot[0]].values[-1],pars[pars2plot[1]].values[-1], '.',markersize=10,color='b',zorder=10)
    
    ax.set_title(title,fontsize=12)
    return pars, obfun