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
# this continues off from the "freyberg glm 1" tutorial 
hbd.dir_cleancopy(org_d=os.path.join('freyberg_glm_1_uncw'), 
                new_d=t_d)

# get and unzip the prior covariance JCB file, prepared in "freyberg pstfrom" tutorial
hbd.unzip(os.path.join('..','..','models','prior_cov.zip'), os.path.join(t_d))


pst = pyemu.Pst(os.path.join(t_d, 'freyberg_pp.pst'))

cov = pyemu.Cov.from_binary(os.path.join(t_d,"prior_cov.jcb"))
cov = cov.get(pst.adj_par_names)
cov.to_ascii(os.path.join(t_d,"glm_prior.cov"))


pst.control_data.noptmax = 4
print(pst.pestpp_options)
#pst.pestpp_options = {"forecasts":pst.pestpp_options["forecasts"]}
#pst.pestpp_options['uncertainty'] = False
pst.pestpp_options["n_iter_base"] = -1
pst.pestpp_options["n_iter_super"] = pst.control_data.noptmax
pst.pestpp_options["glm_num_reals"] = 50
pst.pestpp_options["parcov"] = "glm_prior.cov"
pst.pestpp_options["base_jacobian"] = "freyberg_reuse.jcb"
pst.pestpp_options["glm_normal_form"] = "prior"
pst.pestpp_options["max_n_super"] = 30
pst.pestpp_options["overdue_giveup_fac"] = 5.0
pst.pestpp_options["max_run_fail"] = 3
#pst.svd_data.maxsing = 30

case = 'freyberg_pp'

pst.write(os.path.join(t_d,f"{case}.pst"))

shutil.copy2(os.path.join(t_d,"freyberg_pp.jcb"),
             os.path.join(t_d,"freyberg_reuse.jcb"))


## master
#num_workers = 5
#m_d=os.path.join('master_glm_2')
#pyemu.os_utils.start_workers(t_d,"pestpp-glm",f"{case}.pst",num_workers=num_workers,worker_root=".",
#                           master_dir=m_d)

                           
# master
num_workers = 10
m_d=os.path.join('master_glm_2_uncw')
pyemu.os_utils.start_workers(t_d,"pestpp-glm",f"{case}.pst",
                            num_workers=num_workers,
                            worker_root=os.path.join('R:\Temp'),
                           master_dir=m_d)