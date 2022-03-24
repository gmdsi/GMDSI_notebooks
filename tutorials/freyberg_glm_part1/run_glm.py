import os
import shutil
import pyemu

# specify the temporary working folder
t_d = os.path.join('freyberg6_template')




pst = pyemu.Pst(os.path.join(t_d, 'freyberg_pp.pst'))


cov = pyemu.Cov.from_binary(os.path.join(t_d,"prior_cov.jcb"))
cov = cov.get(pst.adj_par_names)
cov.to_ascii(os.path.join(t_d,"glm_prior.cov"))


pst.control_data.noptmax = 6
print(pst.pestpp_options)
pst.pestpp_options = {"forecasts":pst.pestpp_options["forecasts"]}
pst.pestpp_options["n_iter_base"] = -1
pst.pestpp_options["n_iter_super"] = 3#pst.control_data.noptmax
pst.pestpp_options["glm_num_reals"] = 50
pst.pestpp_options["parcov"] = "glm_prior.cov"
pst.pestpp_options["base_jacobian"] = "freyberg_reuse.jcb"
pst.pestpp_options["glm_normal_form"] = "diag"
pst.pestpp_options["max_n_super"] = 80
pst.pestpp_options["overdue_giveup_fac"] = 5.0
pst.pestpp_options["max_run_fail"] = 3
#pst.svd_data.maxsing = 30
pst.write(os.path.join(t_d,"freyberg_pp.pst"))

shutil.copy2(os.path.join("master_glm","freyberg_pp.jcb"),
             os.path.join(t_d,"freyberg_reuse.jcb"))


# master
num_workers = 4
m_d=os.path.join('master_glm_run')
pyemu.os_utils.start_workers(t_d,"pestpp-glm","freyberg_pp.pst",num_workers=num_workers,worker_root=".",
                           master_dir=m_d)