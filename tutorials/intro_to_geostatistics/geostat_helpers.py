import numpy as np
from scipy.interpolate import griddata, NearestNDInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pyemu
import pandas as pd

contribution=100.0
a=350.0


def data_cooker(domain_pts=50, n_sample_pts=50):
    # make a "true" variogram
    print('Initializing a variogram model')
    v = pyemu.geostats.SphVario(contribution=contribution, a=a)
    gs = pyemu.geostats.GeoStruct(variograms=v)

    # make a domain on which to work
    print('Making the domain')
    x = np.linspace(1, 1000, domain_pts)
    y = x.copy()
    X, Y = np.meshgrid(x, y)

    # get the covariance matrix as a realization over the domain
    print('Initializing covariance model')
    Q = gs.covariance_matrix(X.ravel(), Y.ravel(), names=[str(i) for i in range(len(Y.ravel()))])

    # draw from it to obtain the truth
    print('Drawing from the Geostatistical Model')
    #Z = Q.draw(mean=100).reshape(X.shape)
    ss = pyemu.geostats.SpecSim2d(np.ones_like(x),np.ones_like(y),gs, )
    Z = ss.draw_arrays(1, mean_value=100)[0]


    # sample some "true" values from that domain and add some noise to them
    xd = np.random.uniform(0, 1000, n_sample_pts)
    yd = np.random.uniform(0, 1000, n_sample_pts)
    names = ['p{0}'.format(i) for i in range(n_sample_pts)]
    zd = sample_from_grid(X, Y, Z, xd, yd)

    data_mean = np.mean(Z)
    noise_level = data_mean/10

    zd_noisy = zd +  np.random.randn(len(zd)) * noise_level
    sample_df = pd.DataFrame({'x': xd, 'y': yd, 'z':zd, 'z_noisy':zd_noisy, 'name': names})
    sample_df.index = sample_df['name']
    return X, Y, Z, v, gs, sample_df


# get sampled points using nearest neighbor from a grid of data
def sample_from_grid(X,Y,Z,xs,ys):
    '''
    params:
    X and Y:  point locations on the regular grid
    Z: the data at locations X and Y
    xs and ys: the points at which samples are requested
    
    returns:
    zs: the sampled data at locations xs and ys
    '''
    
    # first make an interpolator object for the regular grid
    fullgrid=NearestNDInterpolator((X.ravel(),Y.ravel()),Z.ravel())
    
    # now sample it at the xs and ys locations
    zs = np.array([fullgrid(point) for point in zip(xs.ravel(),ys.ravel())])
    
    return zs
    
# calculate and plot empirical variogram
def plot_empirical_variogram(x,y,data, nbins=25):

    X,Y = np.meshgrid(x,y)
    h = np.sqrt((X-X.T)**2 + (Y-Y.T)**2).ravel()
    d,d1 = np.meshgrid(data, data)
    gam = (1/2.0*(d-d.T)**2).ravel()
    if nbins>0:
        
        bindiffs = np.ones(nbins)*np.max(h)/2/nbins
        bins = np.hstack(([0],np.cumsum(bindiffs)))
        bindiffs[0] = bins[1]/2
        bincenters = np.cumsum(bindiffs)
    
        empirical_vario = list()
        for i in range(len(bincenters)):
            cinds = np.where((h>bins[i]) & (h<=bins[i+1]))
            empirical_vario.append(np.mean(gam[cinds]))
        h=bincenters
        gam = np.array(empirical_vario)

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    plt.scatter(h,gam)
    plt.xlabel('Separation Distance')
    plt.ylabel('Empirical Variogram')
    plt.grid()
    return h, gam, ax

# scatter plotter for nice-looking plots
def field_scatterplot(x,y,z=None, s=100, xlim=1000, ylim=1000, title=None):
    fig = plt.figure(figsize=(6,6))
    ax = plt.gca()
    if z is not None:
        sc = plt.scatter(x,y,s=s, c=z)
        plt.colorbar(sc)
    
    else:
        plt.scatter(x,y,s=s)
        
    plt.xlim([0,xlim])
    plt.ylim([0,ylim])
    if title is not None:
        plt.title(title)
    ax.set_aspect('equal')
    return ax

def grid_plot(X,Y,Z, vlims=None, title=None):
    plt.figure(figsize=(6,6))
    ax = plt.gca()
    if vlims is None:
        im = ax.pcolormesh(X,Y,Z)
    else:
        im = ax.pcolormesh(X, Y, Z,vmin=vlims[0], vmax=vlims[1])
    ax.set_aspect('equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)    
    plt.colorbar(im, cax=cax)
    if title is not None:
        plt.suptitle(title)
    return ax    
    

def geostat_interpolate(X,Y,interp_data, data_df):
    '''

    :param X: grid of X grid locations
    :param Y: grid of Y grid locations
    :param interp_data: dataframe of interpolation data greated by pyemu.geostats.OrdinaryKrige.calc_factors()
    :param data_df: xd, yd, zd, and names of points

    :return: Z -- an interpolated field
    '''

    dims = X.shape
    X = X.ravel()
    Y = Y.ravel()
    Z = np.zeros_like(X)
    i = 0
    for cp in interp_data.index:
        # alot going on in the next line! Get the locations from sample_df of the names for current point (cp)
        # then sort the results to be the same order as ifacts
        # then grab the zd values
        # make it 2d to take advantage of vectorization later
        cpts = np.atleast_2d(data_df.loc[data_df.name.isin(interp_data.loc[cp].inames)].loc[
            interp_data.loc[cp].inames].z.values)
        Z[i] = np.squeeze(cpts.dot(np.atleast_2d(interp_data.loc[cp].ifacts).T))
        i+=1
    
    return Z.reshape(dims)

if __name__ == "__main__":
    data_cooker()