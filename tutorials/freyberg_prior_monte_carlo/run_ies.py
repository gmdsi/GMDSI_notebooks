import os
import pyemu

t_d = os.path.join('freyberg6_template')
num_workers = 4
m_d = "master_diagonal_prior_monte_carlo" 
pyemu.os_utils.start_workers(t_d,"pestpp-ies","freyberg_diagprior.pst",num_workers=num_workers,
                             master_dir=m_d)