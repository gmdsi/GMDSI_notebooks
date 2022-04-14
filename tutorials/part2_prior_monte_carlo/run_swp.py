import os
import pyemu

t_d = os.path.join('freyberg6_template')
num_workers = 5
m_d = os.path.join('master_priormc')
pyemu.os_utils.start_workers(t_d, # the folder which contains the "template" PEST dataset
                            'pestpp-swp', #the PEST software version we want to run
                            'priormc.pst', # the control file to use with PEST
                            num_workers=num_workers, #how many agents to deploy
                            worker_root='.', #where to deploy the agent directories; relative to where python is running
                            master_dir=m_d, #the manager directory
                            )