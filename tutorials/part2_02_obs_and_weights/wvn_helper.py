
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
modeled = np.random.normal(loc=5, scale=1, size=500)

def plot_mod_obs(truth=6, noisy=None, pdc=None, std=None):
    truth1 = 6
    plt.hist(modeled, bins=50, label='modeled')
    if pdc is not None:
        pdc=': PRIOR DATA CONFLICT'
    else:
        pdc= ''
    if noisy is None:
        noiseflag = 'No '
        plt.axvline(truth, lw=2, c='orange', label='observed')
    elif not isinstance(std, list):
        noiseflag = ''
        obs = np.random.normal(loc=truth, scale=std, size=500)
        plt.hist(obs, color='orange', bins=50, label='observed')
    elif isinstance(std, list):
        noiseflag = ''
        obs1 = np.random.normal(loc=truth, scale=std[0], size=500)
        obs2 = np.random.normal(loc=truth, scale=std[1], size=500)
        plt.hist(obs1, color='orange', bins=50, label="observed $\\sigma$")
        plt.hist(obs2, color='green', bins=50, label="observed $\\frac{1}{\\omega}$", alpha=.3)
        
    plt.legend()
    plt.title(f'Fit With {noiseflag}Observation Noise{pdc}')
    plt.xlim([-10,25])
    plt.yticks([])