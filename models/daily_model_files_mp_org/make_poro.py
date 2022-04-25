import numpy as np
nrow,ncol = 120,60
for k in range(3):
    np.savetxt("freyberg_mp.ne_layer{0}.txt".format(k+1),np.zeros((nrow,ncol))+0.1)   
