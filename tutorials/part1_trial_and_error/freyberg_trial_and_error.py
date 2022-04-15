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
    org_ws = os.path.join('..', '..', 'models', 'freyberg_mf6')

    # set a new workspace folder to avoid breaking things by mistake
    sim_ws = os.path.join('freyberg_mf6')

    # remove existing folder
    if os.path.exists(sim_ws):
        shutil.rmtree(sim_ws)
    
    os.mkdir(sim_ws)

    # copy the original model folder across
    #shutil.copytree(org_ws, sim_ws)

    # get the necessary executables; OS agnostic
    #bin_dir = os.path.join('..','..','bin')
    #exe_file='mf6'
    #if "window" in platform.platform().lower():
    #    exe_file = exe_file+'.exe'
    #shutil.copy2(os.path.join(bin_dir, exe_file), os.path.join(sim_ws,exe_file))
    hbd.prep_bins(sim_ws)

    # get measured data
    truth_dir = os.path.join('..', '..', 'models', 'freyberg_mf6_truth')
    for f in ['heads.meas.csv', 'sfr.meas.csv']:
        shutil.copy2(os.path.join(truth_dir, f), os.path.join(sim_ws,f))

    return print(f'model files are in: {sim_ws}')


def update_par(k1=3,k2=0.3,k3=30, rch_factor=1, sfrplot=True):
    
    # load simulation
    org_ws = os.path.join('..', '..', 'models', 'freyberg_mf6')
    sim = flopy.mf6.MFSimulation.load(sim_ws=org_ws, verbosity_level=0)

    sim_ws=os.path.join('freyberg_mf6')
    sim.set_sim_path(sim_ws)
    # load flow model
    gwf = sim.get_model()
    

    k = gwf.npf.k.get_data()
    k[0] = k1
    k[1] = k2
    k[2] = k3

    # reset data
    gwf.npf.k.set_data(k)
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
    plot_simvsmeas(sim_ws,sfrplot)

    return


def plot_simvsmeas(sim_ws,sfrplot=True):
    fig = plt.figure(figsize=(10,5))

    ax = fig.add_subplot(131)
    meas = pd.read_csv(os.path.join(sim_ws, 'heads.meas.csv')).iloc[:12, 1:].values
    sim = pd.read_csv(os.path.join(sim_ws, 'heads.csv')).iloc[:12, 1:].values
    plt.scatter( meas, sim)
    plt.title('Heads')
    plt.xlabel('measured')
    plt.ylabel('simulated')
    plt.grid()
    add_1to1(ax)
    plt.axis('square')
    plt.text(x=.95, y=.05, s=f'RMSE:{round(rmse(sim, meas),2)}',transform=ax.transAxes, ha='right')

    if sfrplot==True:
        ax = fig.add_subplot(132)
        meas = pd.read_csv(os.path.join(sim_ws, 'sfr.meas.csv')).iloc[:12, 1:].values
        sim = pd.read_csv(os.path.join(sim_ws, 'sfr.csv')).iloc[:12, 1:].values
        plt.scatter( meas,sim)
        plt.title('SFR')
        plt.xlabel('measured')
        plt.ylabel('simulated')
        plt.grid()
        add_1to1(ax)
        plt.axis('square')
        plt.text(x=.95, y=.05, s=f'RMSE:{round(rmse(sim, meas),2)}',transform=ax.transAxes, ha='right')
    

    ax = fig.add_subplot(133)
    meas = pd.read_csv(os.path.join(sim_ws, 'sfr.meas.csv')).iloc[-1, 1]
    sim = pd.read_csv(os.path.join(sim_ws, 'sfr.csv')).iloc[-1, 1]
    plt.scatter( meas,sim)
    plt.title('Forecast SFR')
    plt.xlabel('truth')
    plt.ylabel('simulated')
    plt.grid()
    add_1to1(ax)
    plt.axis('square')
    plt.text(x=.95, y=.05, s=f'RMSE:{round(rmse(sim, meas),2)}',transform=ax.transAxes, ha='right')


    fig.tight_layout()
    return

if __name__ == "__main__":
    print('This is not the command you are looking for...')