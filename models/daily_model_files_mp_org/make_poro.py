import os
import numpy as np
import flopy

nrow,ncol = 120,60
for k in range(3):
    np.savetxt("freyberg_mp.ne_layer{0}.txt".format(k+1),np.zeros((nrow,ncol))+0.1)   

files = [f for f in os.listdir(".") if "wel_stress_period_data" in f and f.endswith(".txt")]
for fname in files:
	print(fname)
	lines = open(fname,'r').readlines()
	with open(fname,'w') as f:
		for line in lines:
			f.write(line.replace(',',' ')) 
