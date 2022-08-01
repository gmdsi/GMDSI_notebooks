import os
import shutil
import pandas as pd
import numpy as np
import pyemu
import flopy
pp_file = 'hkpp.dat'
hk_arr = pyemu.geostats.fac2real(pp_file, factors_file=pp_file+'.fac',out_file='freyberg6.npf_k_layer1.txt')
pp_file = 'rchpp.dat'
hk_arr = pyemu.geostats.fac2real(pp_file, factors_file=pp_file+'.fac',out_file='rch0_fac.txt')
rch0 = np.loadtxt('rch0_fac.txt')
files = [i for i in os.listdir('.') if '.rch_recharge' in i and int(i.split('.')[-2].split('_')[-1])<13]
for f in files:
    a = np.loadtxt(os.path.join('org_f',f))
    a = a*rch0
    np.savetxt(f, a, fmt='%1.6e')
pyemu.os_utils.run('mf6')
pyemu.os_utils.run('mp7 freyberg_mp.mpsim')
