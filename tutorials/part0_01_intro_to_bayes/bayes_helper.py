import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,os.path.join("..", "..", "dependencies"))
import pyemu
import flopy
assert "dependencies" in flopy.__file__
assert "dependencies" in pyemu.__file__
import matplotlib as mpl

#--modify default matplotlib settings
mpl.rcParams['font.sans-serif']          = 'Univers 57 Condensed'
mpl.rcParams['font.serif']               = 'Times'
mpl.rcParams['pdf.compression']          = 0
mpl.rcParams['pdf.fonttype']             = 42
#--figure text sizes
mpl.rcParams['legend.fontsize']  = 12
mpl.rcParams['axes.labelsize']   = 12
mpl.rcParams['xtick.labelsize']  = 12
mpl.rcParams['ytick.labelsize']  = 12

def gaussian_multiply(mu1,std1,mu2,std2):
    var1,var2 = std1**2,std2**2
    mean = (var1*mu2 + var2*mu1) / (var1 + var2)
    variance = (var1 * var2) / (var1 + var2)
    return mean, np.sqrt(variance)


def plot_posterior(prior_mean, prior_std, likeli_mean, likeli_std, legend=True, savefigure=False):
    plt.figure()

    post_mean, post_std = gaussian_multiply(prior_mean, prior_std, likeli_mean, likeli_std)

    xs, ys = pyemu.plot.plot_utils.gaussian_distribution(prior_mean, prior_std)
    plt.plot(xs, ys, color='k', ls='--', lw=3.0, label='prior')

    xs, ys = pyemu.plot.plot_utils.gaussian_distribution(likeli_mean, likeli_std)
    plt.plot(xs, ys, color='g', ls='--', lw=3.0, label='likelihood')

    xs, ys = pyemu.plot.plot_utils.gaussian_distribution(post_mean, post_std)
    plt.fill_between(xs, 0, ys, label='posterior', color='b', alpha=0.25)
    if legend:
        plt.legend();
    ax = plt.gca()
    ax.set_xlabel("Parameter")
    ax.set_yticks([])


    if savefigure:
        plt.savefig('probs.pdf')
    plt.show()