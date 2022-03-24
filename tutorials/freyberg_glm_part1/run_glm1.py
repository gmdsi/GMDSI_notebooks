import os
import shutil
import pyemu
import sys
sys.path.append("..")
# import pre-prepared convenience functions
import herebedragons as hbd



# specify the temporary working folder
t_d = os.path.join('freyberg6_template')

# use the convenience function to get the pre-preprepared PEST dataset;
# this is the same dataset consutructed in the "obs and weights" tutorial
hbd.dir_cleancopy(org_d=os.path.join('..','..', 'models','freyberg_obs_and_weights'), 
                new_d=t_d)
# get and unzip the prior covariance JCB file
hbd.unzip(os.path.join('..','..','models','prior_cov.zip'), os.path.join(t_d))


pst = pyemu.Pst(os.path.join(t_d, 'freyberg_wv.pst'))

# fix pars
par = pst.parameter_data
# say goodbye to grid-scale pars
gr_pars = par.loc[par.pargp.apply(lambda x: "gr" in x),"parnme"]
par.loc[gr_pars,"partrans"] = "fixed"

rch_pp_future = [i for i in pst.adj_par_groups if i.startswith('rch') and int(i.split('_')[-2]) > 12]
par.loc[par['pargp'].isin(rch_pp_future),"partrans"] = "fixed"

fi_grps = ["sto_ss_layer3_pp",'sto_ss_layer2_pp',"sto_sy_layer1_pp", 'npf_k33_layer1_pp','npf_k33_layer3_pp']
par.loc[par.pargp.apply(lambda x: x in fi_grps),"partrans"] = "fixed"

pst.control_data.noptmax = -1
pst.write(os.path.join(t_d,"freyberg_pp.pst"))

# master
num_workers = 5
m_d=os.path.join('master_glm')
pyemu.os_utils.start_workers(t_d,"pestpp-glm","freyberg_pp.pst",num_workers=num_workers,worker_root=".",
                           master_dir=m_d)