import pyemu
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd


def update_K(inpstname):
    optpst = pyemu.Pst(inpstname)
    # reset the parameter values using the best parameters from the last run
    optpst.parrep(inpstname.replace('.pst', '.bpa'))
    # set NOPTMAX=0 to run the model just a single time
    optpst.control_data.noptmax=0
    # write out the PST file with a new name
    optpst.write(inpstname.replace('.pst', '_opt.pst'))
    # run the new version of the model
    pyemu.os_utils.run('pestpp {}'.format(inpstname.replace('.pst', '_opt.pst')))


def plot_K_results(working_dir, inpstname):
    ib = np.loadtxt(os.path.join(working_dir,'ibound_layer_1.ref'))
    HK_truth = np.loadtxt(os.path.join(working_dir,'hk.truth.ref'))
    HK_best = np.loadtxt(os.path.join(working_dir,'hk_layer_1.ref'))
    HK_best[ib == 0] = np.nan
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4))
    im1 = ax1.imshow(HK_truth, interpolation='nearest', cmap='viridis', vmin=np.nanmin(HK_truth),
                     vmax=np.nanmax(HK_truth))
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="20%", pad=0.05)
    cbar1 = plt.colorbar(im1, cax=cax1, format="%.2f")

    im2 = ax2.imshow(HK_best, interpolation='nearest', cmap='viridis', vmin=np.nanmin(HK_truth),
                     vmax=np.nanmax(HK_truth))
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="20%", pad=0.05)
    cbar2 = plt.colorbar(im2, cax=cax2, format="%.2f")

    im3 = ax3.imshow(HK_best, interpolation='nearest', cmap='viridis')
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="20%", pad=0.05)
    cbar3 = plt.colorbar(im3, cax=cax3, format="%.2f")
    plt.tight_layout()
    plt.savefig(inpstname + 'HK_.pdf')

    plt.figure()
    inphi = pd.read_csv(os.path.join(working_dir,inpstname + '.iobj'))
    inphi[['total_phi', 'measurement_phi', 'regularization_phi']].plot()
    plt.gca().set_yscale('log')
    plt.ylabel('PHI')
    plt.xlabel('Iteration Number')
