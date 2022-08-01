import os
import shutil
import pandas as pd
import numpy as np
import pyemu
import flopy
pp_file = 'hkpp.dat'
hk_arr = pyemu.geostats.fac2real(pp_file, factors_file=pp_file+'.fac',out_file='freyberg6.npf_k_layer1.txt')
pyemu.os_utils.run('mf6')
pyemu.os_utils.run('mp7 freyberg_mp.mpsim')
