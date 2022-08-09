
---
layout: default
title: Intro to Geostatistics
parent: Introductions to Selected Topics
nav_order: 5
---
                    ---
layout: default
title: Intro to Geostatistics
parent: Introductions to Selected Topics
nav_order: 5
math: mathjax3
---

```python
import pyemu
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import geostat_helpers as gh
import pandas as pd
from scipy.stats.mstats import normaltest
import scipy.stats as sps
```

# Geostatistics 

This notebook is a very high-level introduction to geostatisics. Some definitions from Geoff Bohling http://people.ku.edu/~gbohling/cpe940/Variograms.pdf

> “_Geostatistics: study of phenomena that vary in space and/or time_”
(Deutsch, 2002)

> “_Geostatistics can be regarded as a collection of numerical techniques that deal with the characterization of spatial attributes, employing primarily random models in a manner similar to the way in which time series analysis characterizes temporal data._”
(Olea, 1999)

> “_Geostatistics offers a way of describing the spatial continuity of natural phenomena and provides adaptations of classical regression techniques to take advantage of this continuity._” 
(Isaaks and Srivastava, 1989)

> Geostatistics deals with spatially _autocorrelated_ data.

> "_Autocorrelation: correlation between elements of a series and others from the same series separated from them by a given interval._"
(Oxford American Dictionary)

# Main Concepts

1. Variogram modeling -- a way to characterize spatial correlation
2. Kriging -- a best linear unbiased estimate (BLUE) for interpolation with minimum variance. There are several flavors - we will focus on Ordinary Kriging
3. Stochastic Simulation -- http://petrowiki.org/Geostatistical_conditional_simulation
4. Beyond this multi-Gaussian approach focused on the relationships among pairs of points, there is _multiple point geostatistics_ as well using training images and more complex shapes

These concepts each build on each other. We will briefly touch on the first two.

### Generate a Field
Let's cook up a quick random field and explore the spatial structure. 


```python
X,Y,Z,v,gs,sample_df = gh.data_cooker()
```

    Initializing a variogram model
    Making the domain
    Initializing covariance model
    Drawing from the Geostatistical Model
    SpecSim.initialize() summary: full_delx X full_dely: 2162 X 2162
    

### Visualize the Field
Pretend (key word!) that this is a hydraulic conductivity field. What do you think? Any _autocorrelation_ here? 
Note how values spread _continuously_. Points which are close together have similar values. They are not _entirely_ random.


```python
gh.grid_plot(X,Y,Z);
```


    
![png](intro_to_geostatistics_files/intro_to_geostatistics_6_0.png)
    


### Link to the Real-World

In practice, we would typically only know the values at a few points (and probably not perfectly). (Think pumping tests or other point-sample site characterisation methods.) So how do we go from these "few" samples to a continuous parameter field?

>note: the default number of samples we use here is 50.


```python
gh.field_scatterplot(sample_df.x,sample_df.y,sample_df.z);
```


    
![png](intro_to_geostatistics_files/intro_to_geostatistics_8_0.png)
    


## Main Assumptions:
   1. The values are second order stationary (the mean and variance are relatively constant) 
   2. The values are multi-Gaussian (e.g. normally distributed)

If we inspect our generated data, we see that it is normally distributed, so that's good. (_side note: of course it is, we generated it using geostaticsits..so we are cheating here..._)


```python
plt.hist(Z.ravel(), bins=50)
x=np.linspace(70,130,100)
plt.plot(x,sps.norm.pdf(x, np.mean(Z),np.std(Z))*len(Z.ravel()));
```


    
![png](intro_to_geostatistics_files/intro_to_geostatistics_10_0.png)
    


What about our sample?


```python
plt.hist(sample_df.z, bins=50)
x=np.linspace(70,130,100)
plt.plot(x,sps.norm.pdf(x, np.mean(sample_df.z),np.std(sample_df.z))*len(sample_df.z))
```




    [<matplotlib.lines.Line2D at 0x1aa5c82bc70>]




    
![png](intro_to_geostatistics_files/intro_to_geostatistics_12_1.png)
    


Purity is commendable, but in practice we are going to violate some of these assumptions for sure. 

### Variograms
At the heart of geostatistics is some kind of model expressing the variability of properties in a field. This is a "variogram" and we can explore it based on the following empirical formula:

 $$\hat{\gamma}\left(h\right)=\frac{1}{2\left(h\right)}\left(z\left(x_1\right)-z\left(x_2\right)\right)^2$$
 
where $x_1$ and $x_2$ are the locations of two $z$ data points separated by distance $h$.



If we plot these up we get something called a cloud plot showing $\hat\gamma$ for all pairs in the dataset


```python
h,gam,ax=gh.plot_empirical_variogram(sample_df.x,sample_df.y,sample_df.z,0)
```


    
![png](intro_to_geostatistics_files/intro_to_geostatistics_16_0.png)
    


This is pretty messy, so typically it is evaluated in bins, and usually only over half the total possible distance


```python
h,gam,ax=gh.plot_empirical_variogram(sample_df.x,sample_df.y,sample_df.z,50)
```


    
![png](intro_to_geostatistics_files/intro_to_geostatistics_18_0.png)
    


Also note that this was assuming perfect observations. What if there was ~10% noise?


```python
h,gam,ax=gh.plot_empirical_variogram(sample_df.x,sample_df.y,sample_df.z_noisy,30)
```


    
![png](intro_to_geostatistics_files/intro_to_geostatistics_20_0.png)
    


Geostatistics is making the assumption that you can model the variability of this field using a variogram. The variogram is closely related to covariance. We take advantage of a few assumptions to come up with a few functional forms that should characterize this behavior.

## Variograms in `pyemu`
`pyemu` supports three variogram models. (As do most of the utilities in the PEST-suite of software.)
This follows the _GSLIB_ terminology:
 1. *Spherical*  
 $\gamma\left(h\right)=c\times\left[1.5\frac{h}{a}-0.5\frac{h}{a}^3\right]$ if $h<a$
 $\gamma\left(h\right)=c$ if $h \ge a$  
  
 2. *Exponential*  
 $\gamma\left(h\right)=c\times\left[1-\exp\left(-\frac{h}{a}\right)\right]$  
  
 3. *Gaussian*  
 $\gamma\left(h\right)=c\times\left[1-\exp\left(-\frac{h^2}{a^2}\right)\right]$  
  
 $h$ is the separation distance, and $a$ is the range. `contribution` is the variogram value at which the variogram levels off. Also called the `sill`, this value is the maximum variability between points.
 The sill is reached at about $a$ for the *Spherical* model, $2a$ for the *Gaussian*, and $3a$ for the *Exponential*

### What do these look like?

For a consistent set of parameters:
 > a=500, c=10
 
 We can use `pyemu` to setup a geostatistical model


```python
a=500
c=10
```

Set up a variogram object and, from that, build a geostatistical structure

### _Spherical_


```python
v = pyemu.geostats.SphVario(contribution=c, a=a)
gs = pyemu.geostats.GeoStruct(variograms=v)
```


```python
gs.plot()
plt.plot([v.a,v.a],[0,v.contribution],'r')
plt.grid()
```


    
![png](intro_to_geostatistics_files/intro_to_geostatistics_28_0.png)
    



```python
Q= gs.covariance_matrix(X.ravel(), Y.ravel(), names=[str(i) for i in range(len(Y.ravel()))])
plt.figure(figsize=(6,6))
plt.imshow(Q.x)
plt.colorbar();
```


    
![png](intro_to_geostatistics_files/intro_to_geostatistics_29_0.png)
    


### _Exponential_


```python
v = pyemu.geostats.ExpVario(contribution=c, a=a)
gs = pyemu.geostats.GeoStruct(variograms=v)
gs.plot()
plt.plot([v.a,v.a],[0,v.contribution],'r')
plt.plot([3*v.a,3*v.a],[0,v.contribution],'r:')
plt.grid();
```


    
![png](intro_to_geostatistics_files/intro_to_geostatistics_31_0.png)
    



```python
Q= gs.covariance_matrix(X.ravel(), Y.ravel(), names=[str(i) for i in range(len(Y.ravel()))])
plt.figure(figsize=(6,6))
plt.imshow(Q.x)
plt.colorbar();
```


    
![png](intro_to_geostatistics_files/intro_to_geostatistics_32_0.png)
    


### _Gaussian_


```python
v = pyemu.geostats.GauVario(contribution=c, a=a)
gs = pyemu.geostats.GeoStruct(variograms=v)
gs.plot()
plt.plot([v.a,v.a],[0,v.contribution],'r')
plt.plot([7/4*v.a,7/4*v.a],[0,v.contribution],'r:')
plt.grid();
```


    
![png](intro_to_geostatistics_files/intro_to_geostatistics_34_0.png)
    



```python
Q= gs.covariance_matrix(X.ravel(), Y.ravel(), names=[str(i) for i in range(len(Y.ravel()))])
plt.figure(figsize=(6,6))
plt.imshow(Q.x)
plt.colorbar();
```


    
![png](intro_to_geostatistics_files/intro_to_geostatistics_35_0.png)
    


## Interpolating from Sparse Data
So how do we go from a sample of measurments (i.e. our 50 points, sampled frmo the field at the start of the notebook) and generate a continuous filed? If we fit an appropriate model ($\gamma$) to the empirical variogram ($\hat\gamma$), we can use that structure for interpolation from sparse data.

Experiment below with changing the `new_a` and `new_c` variables and/or the variogram type.


```python
h,gam,ax=gh.plot_empirical_variogram(sample_df.x,sample_df.y,sample_df.z,50)
new_c=10
new_a=500.0

v_fit = pyemu.geostats.ExpVario(contribution=new_c,a=new_a)
gs_fit = pyemu.geostats.GeoStruct(variograms=v_fit)
gs_fit.plot(ax=ax);
```


    
![png](intro_to_geostatistics_files/intro_to_geostatistics_37_0.png)
    



```python
Q = gs_fit.covariance_matrix(X.ravel(), Y.ravel(), names=[str(i) for i in range(len(Y.ravel()))])
```


```python
plt.figure(figsize=(6,6))
plt.imshow(Q.x)
plt.colorbar();
```


    
![png](intro_to_geostatistics_files/intro_to_geostatistics_39_0.png)
    


Now we can perform Kriging to interpolate using this variogram and our "sample data". First make an Ordinary Kriging object:


```python
k = pyemu.geostats.OrdinaryKrige(gs_fit,sample_df)
```


```python
sample_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>z_noisy</th>
      <th>name</th>
    </tr>
    <tr>
      <th>name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>p0</th>
      <td>287.116655</td>
      <td>699.455696</td>
      <td>85.372835</td>
      <td>83.771018</td>
      <td>p0</td>
    </tr>
    <tr>
      <th>p1</th>
      <td>253.732612</td>
      <td>910.336327</td>
      <td>82.353507</td>
      <td>96.801400</td>
      <td>p1</td>
    </tr>
    <tr>
      <th>p2</th>
      <td>274.886538</td>
      <td>167.553939</td>
      <td>82.260247</td>
      <td>82.146898</td>
      <td>p2</td>
    </tr>
    <tr>
      <th>p3</th>
      <td>32.513697</td>
      <td>966.814451</td>
      <td>81.683278</td>
      <td>79.377728</td>
      <td>p3</td>
    </tr>
    <tr>
      <th>p4</th>
      <td>565.624540</td>
      <td>767.730971</td>
      <td>83.725429</td>
      <td>85.923798</td>
      <td>p4</td>
    </tr>
  </tbody>
</table>
</div>



Next we need to calculate factors (we only do this once - takes a few seconds)


```python
kfactors = k.calc_factors(X.ravel(),Y.ravel())
```

    starting interp point loop for 2500 points
    took 5.955451 seconds
    

It's easiest to think of these factors as weights on surrounding point to calculate a weighted average of the surrounding values. The weight is a function of the distance - points father away have smaller weights.


```python
kfactors.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
      <th>idist</th>
      <th>inames</th>
      <th>ifacts</th>
      <th>err_var</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.000000</td>
      <td>1.0</td>
      <td>[184.7051991534765, 207.40971945627004, 386.50730584099625, 491.60458345801544, 580.295266631326...</td>
      <td>[p38, p20, p40, p37, p36, p33, p10, p35, p18, p0, p43]</td>
      <td>[0.4327770142438521, 0.3417529484244285, 0.004059230296319384, 0.04778912246916971, 0.0501610806...</td>
      <td>2.767218</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21.387755</td>
      <td>1.0</td>
      <td>[169.14795023838278, 187.0266048520901, 370.70802497724264, 471.21711009351276, 575.554567272516...</td>
      <td>[p38, p20, p40, p37, p36, p33, p10, p35, p18, p0, p43]</td>
      <td>[0.43604287579758993, 0.36167431286832813, 0.0020604545233822214, 0.04483723075668418, 0.0440368...</td>
      <td>2.574679</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41.775510</td>
      <td>1.0</td>
      <td>[154.71706002108814, 166.64462558472334, 450.82966221072473, 571.5022698624969, 650.332363132082...</td>
      <td>[p38, p20, p37, p36, p33, p35, p18, p10, p0, p43]</td>
      <td>[0.43557284599104173, 0.3857123856407392, 0.041766258070064934, 0.03765454657692189, 0.022814952...</td>
      <td>2.374874</td>
    </tr>
    <tr>
      <th>3</th>
      <td>62.163265</td>
      <td>1.0</td>
      <td>[141.7569376414895, 146.264256283779, 430.4422434303959, 568.1531044744195, 630.8486277459325, 6...</td>
      <td>[p38, p20, p37, p36, p33, p35, p18, p10, p43, p0]</td>
      <td>[0.42882967662347826, 0.41442810902652116, 0.038478280336504585, 0.03109342162082103, 0.02008586...</td>
      <td>2.169980</td>
    </tr>
    <tr>
      <th>4</th>
      <td>82.551020</td>
      <td>1.0</td>
      <td>[125.8862788886181, 130.70581927985793, 410.0548580933408, 565.5195636810955, 645.0133254748288,...</td>
      <td>[p20, p38, p37, p36, p35, p18, p10, p43, p30]</td>
      <td>[0.4523516266403694, 0.41636993457552113, 0.03176622888874157, 0.02508944553343011, 0.0014195050...</td>
      <td>2.009520</td>
    </tr>
  </tbody>
</table>
</div>



Now interpolate from our sampled points to a grid:


```python
Z_interp = gh.geostat_interpolate(X,Y,k.interp_data, sample_df)
```


```python
ax=gh.grid_plot(X,Y,Z_interp, title='reconstruction', vlims=[72,92])
ax.plot(sample_df.x,sample_df.y, 'ko');
```


    
![png](intro_to_geostatistics_files/intro_to_geostatistics_49_0.png)
    



```python
gh.grid_plot(X,Y,Z,title='truth', vlims=[72,92])
```




    <AxesSubplot:>




    
![png](intro_to_geostatistics_files/intro_to_geostatistics_50_1.png)
    



```python
ax=gh.grid_plot(X,Y,kfactors.err_var.values.reshape(X.shape), title='Variance of Estimate')
ax.plot(sample_df.x,sample_df.y, 'ko');
```


    
![png](intro_to_geostatistics_files/intro_to_geostatistics_51_0.png)
    



```python
ax=gh.grid_plot(X,Y,np.abs(Z-Z_interp), title='Actual Differences')
ax.plot(sample_df.x,sample_df.y, 'yo');
```


    
![png](intro_to_geostatistics_files/intro_to_geostatistics_52_0.png)
    


## What if our data were noisy?

Try and get a good fit by adjusting the `new_c` and `new_a` values:


```python
h,gam,ax=gh.plot_empirical_variogram(sample_df.x,sample_df.y,sample_df.z_noisy,30)
new_c=50.0
new_a=350.0

# select which kind of variogram here because in reality we don't know, right?
v_fit = pyemu.geostats.ExpVario(contribution=new_c,a=new_a)
gs_fit = pyemu.geostats.GeoStruct(variograms=v_fit, nugget=50)
gs_fit.plot(ax=ax);
```


    
![png](intro_to_geostatistics_files/intro_to_geostatistics_54_0.png)
    



```python
Q = gs_fit.covariance_matrix(X.ravel(), Y.ravel(), names=[str(i) for i in range(len(Y.ravel()))])
plt.figure(figsize=(6,6))
plt.imshow(Q.x)
plt.colorbar();
```


    
![png](intro_to_geostatistics_files/intro_to_geostatistics_55_0.png)
    


Again make the Kriging Object and the factors and interpolate


```python
k = pyemu.geostats.OrdinaryKrige(gs_fit,sample_df)
kfactors = k.calc_factors(X.ravel(),Y.ravel())
Z_interp = gh.geostat_interpolate(X,Y,k.interp_data, sample_df)
```

    starting interp point loop for 2500 points
    took 5.975461 seconds
    


```python
ax=gh.grid_plot(X,Y,Z_interp, vlims=[72,92], title='reconstruction')
ax.plot(sample_df.x,sample_df.y, 'ko')
```




    [<matplotlib.lines.Line2D at 0x1aa5bce5460>]




    
![png](intro_to_geostatistics_files/intro_to_geostatistics_58_1.png)
    



```python
gh.grid_plot(X,Y,Z, vlims=[72,92],title='truth');
```


    
![png](intro_to_geostatistics_files/intro_to_geostatistics_59_0.png)
    



```python
ax=gh.grid_plot(X,Y,kfactors.err_var.values.reshape(X.shape), title='Variance of Estimate')
ax.plot(sample_df.x,sample_df.y, 'ko');
```


    
![png](intro_to_geostatistics_files/intro_to_geostatistics_60_0.png)
    



```python
ax=gh.grid_plot(X,Y,np.abs(Z-Z_interp), title='Actual Differences')
ax.plot(sample_df.x,sample_df.y, 'yo')
```




    [<matplotlib.lines.Line2D at 0x1aa5b1e86a0>]




    
![png](intro_to_geostatistics_files/intro_to_geostatistics_61_1.png)
    


### Spectral simulation

Because pyemu is pure python (and because the developers are lazy), it only implments spectral simulation for grid-scale field generation.  For regular grids without anisotropy and without conditioning data ("known" property values), it is identical to sequential gaussian simulation.

Each of hte plots below illustrate the effect of different values of `a`. Experiment with changing `a`,  `contribution`, etc to get a feel for how they affect spatial patterns.


```python
ev = pyemu.geostats.ExpVario(1.0,1, )
gs = pyemu.geostats.GeoStruct(variograms=ev)
ss = pyemu.geostats.SpecSim2d(np.ones(100),np.ones(100),gs)
plt.imshow(ss.draw_arrays()[0]);
```

    SpecSim.initialize() summary: full_delx X full_dely: 116 X 116
    


    
![png](intro_to_geostatistics_files/intro_to_geostatistics_63_1.png)
    



```python
ev = pyemu.geostats.ExpVario(1.0,5)
gs = pyemu.geostats.GeoStruct(variograms=ev)
ss = pyemu.geostats.SpecSim2d(np.ones(100),np.ones(100),gs)
plt.imshow(ss.draw_arrays()[0]);
```

    SpecSim.initialize() summary: full_delx X full_dely: 132 X 132
    


    
![png](intro_to_geostatistics_files/intro_to_geostatistics_64_1.png)
    



```python
ev = pyemu.geostats.ExpVario(1.0,500)
gs = pyemu.geostats.GeoStruct(variograms=ev)
ss = pyemu.geostats.SpecSim2d(np.ones(100),np.ones(100),gs)
plt.imshow(ss.draw_arrays()[0]);
```

    SpecSim.initialize() summary: full_delx X full_dely: 3108 X 3108
    


    
![png](intro_to_geostatistics_files/intro_to_geostatistics_65_1.png)
    


# Further resources and information
1. These concepts are used for pilot point interpolation in PEST:
    - In the GW utilities in PEST (http://www.pesthomepage.org/Groundwater_Utilities.php) 
    - The main tools are also available in `pyemu` -- we'll use that in the class
2. The Stanford Geostatistical Modeling Software (SGeMS: http://sgems.sourceforge.net/) is a nice GUI for geostatistical modeling, but it's not being maintained anymore.
3. Python libraries for geostistics:
    - [`pysgems`](https://github.com/robinthibaut/pysgems) uses SGEMS within Python 
    - [`Scikit-GStat`](https://github.com/mmaelicke/scikit-gstat). A tutorial can be found [here](https://guillaumeattard.com/geostatistics-applied-to-hydrogeology-with-scikit-gstat/)
4. `R` has a package: http://rgeostats.free.fr/
