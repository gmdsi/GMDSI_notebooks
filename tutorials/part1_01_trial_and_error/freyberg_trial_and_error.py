import os
import sys
import flopy
import shutil
import matplotlib.pyplot as plt


import platform
import numpy as np
import pandas as pd
import pyemu

sys.path.append("..")
import herebedragons as hbd

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def add_1to1(ax):
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    # now plot both limits against eachother
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    return



def get_model():
    # folder containing original model files
    org_ws = os.path.join('..', '..', 'models', 'monthly_model_files_1lyr_newstress')

    # set a new workspace folder to avoid breaking things by mistake
    sim_ws = os.path.join('freyberg_mf6')

    # remove existing folder
    if os.path.exists(sim_ws):
        shutil.rmtree(sim_ws)

    # copy the original model folder across
    shutil.copytree(org_ws, sim_ws)

    hbd.prep_bins(sim_ws)

    pyemu.os_utils.run('mf6', cwd=sim_ws)

    # get measured data
    get_meas_data(sim_ws)

    return print(f'model files are in: {sim_ws}')


def update_par(k1=3, rch_factor=1, sfrplot=False):
    
    # load simulation
    org_ws = os.path.join('..', '..', 'models', 'monthly_model_files_1lyr_newstress')
    sim_ws = os.path.join('freyberg_mf6')
    sim = flopy.mf6.MFSimulation.load(sim_ws=org_ws, verbosity_level=0)
    
    sim.set_sim_path(sim_ws)

    # load flow model
    gwf = sim.get_model()
    
    gwf.npf.k.set_data(k1)
    gwf.npf.set_all_data_external()
    #gwf.npf.write()

    rch = gwf.rch.recharge.get_data()

    rch.update((x, y*rch_factor) for x, y in rch.items())
    gwf.rch.recharge.set_data(rch)
    gwf.rch.set_all_data_external()
    #gwf.rch.write()

    # run the model
    sim.write_simulation()
    #sim.run_simulation()
    pyemu.os_utils.run("mf6",cwd=sim_ws)

    # plot results
    plot_simvsmeas(sim_ws, sfrplot)

    return

def add_plot_formatting():
    plt.xlabel('measured')
    plt.ylabel('simulated')
    plt.grid()
    plt.axis('square')
    return

def plot_simvsmeas(sim_ws, sfrplot):
    plt.rcParams.update({'font.size': 12})
    fig = plt.figure(figsize=(15,5))
    
    addcol=0
    if sfrplot:
        addcol=1

    ax = fig.add_subplot(1,2+addcol,1)
    #meas = pd.read_csv(os.path.join(sim_ws, 'heads.meas.csv')).iloc[:12, 1:].values
    obs_data = pd.read_csv(os.path.join(sim_ws, 'obs_data_ess.csv'))
    sim_heads = pd.read_csv(os.path.join(sim_ws, 'heads.csv'))

    head_sites=['TRGW-0-3-8', 'TRGW-0-26-6']
    simvals = sim_heads[head_sites].iloc[:12].values
    measvals = obs_data[[i.lower() for i in head_sites]].iloc[:12].values

    ax.scatter( measvals, simvals)
    ax.set_title('Heads')
    add_1to1(ax)
    add_plot_formatting()
    ax.text(x=.1, y=.95, s=f'RMSE:{round(rmse(simvals, measvals),2)}',
            transform=ax.transAxes, ha='left')

    sim_sfr = pd.read_csv(os.path.join(sim_ws, 'sfr.csv'))
    if sfrplot==True:
        ax = fig.add_subplot(1,3,2)
        simvals = sim_sfr['GAGE-1'].iloc[:12].values
        measvals = obs_data[('GAGE-1').lower()].iloc[:12]
        ax.scatter( measvals,simvals)
        ax.set_title('SFR')
        add_1to1(ax)
        add_plot_formatting()
        plt.text(x=.1, y=.95, s=f'RMSE:{round(rmse(simvals, measvals),2)}',
                transform=ax.transAxes, ha='left')
    
    ax = fig.add_subplot(1,2+addcol,2+addcol)
    meas = obs_data[('GAGE-1').lower()].iloc[-1]
    sim = sim_sfr['GAGE-1'].iloc[-1]
    ax.scatter( meas,sim)
    ax.set_ylim(0)
    ax.set_title('Forecast SFR')
    add_1to1(ax)
    add_plot_formatting()
    
    ax.text(x=.1, y=.95, s=f'RMSE:{round(rmse(sim, meas),2)}',transform=ax.transAxes, ha='left')

    fig.tight_layout()
    return

def get_meas_data(tmp_d='freyberg_mf6'):
    # geat meas values
    shutil.copy2(os.path.join('..', '..', 'models', 'daily_freyberg_mf6_truth','obs_data.csv'),
                            os.path.join(tmp_d, 'obs_data.csv'))
    obs_data = pd.read_csv(os.path.join(tmp_d, 'obs_data.csv'))
    obs_data.site = obs_data.site.str.lower()
    obs_data.set_index('site', inplace=True)
    
    # restructure the obsevration data 
    obs_sites = obs_data.index.unique().tolist()
    #model_times = pst.observation_data.time.dropna().astype(float).unique()
    model_times = pd.read_csv(os.path.join(tmp_d, 'heads.csv')).time.values
    ess_obs_data = {}
    for site in obs_sites:
        #print(site)
        site_obs_data = obs_data.loc[site,:].copy()
        if isinstance(site_obs_data, pd.Series):
            site_obs_data.loc["site"] = site_obs_data.index.values
        if isinstance(site_obs_data, pd.DataFrame):
            site_obs_data.loc[:,"site"] = site_obs_data.index.values
            site_obs_data.index = site_obs_data.time
            sm = site_obs_data.value.rolling(window=20,center=True,min_periods=1).mean()
            sm_site_obs_data = sm.reindex(model_times,method="nearest")
        #ess_obs_data.append(pd.DataFrame9sm_site_obs_data)
        ess_obs_data[site] = sm_site_obs_data
    ess_obs_data = pd.DataFrame(ess_obs_data)
    ess_obs_data.to_csv(os.path.join(tmp_d, 'obs_data_ess.csv'))

if __name__ == "__main__":
    print('This is not the command you are looking for...')