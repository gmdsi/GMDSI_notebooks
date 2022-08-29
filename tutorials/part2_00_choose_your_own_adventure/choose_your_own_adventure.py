
import sys
import os
import shutil
import platform
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt;
import psutil
sys.path.insert(0,"..")
import herebedragons as hbd               
sys.path.insert(0,os.path.join("..","..","dependencies"))                               
import pyemu
import flopy


def setup_pst():
    # folder containing original model files
    org_d = os.path.join('..','..', 'models', 'daily_model_files_newstress')

    # a dir to hold a copy of the org model files
    tmp_d = os.path.join('daily_freyberg_mf6')

    if os.path.exists(tmp_d):
        shutil.rmtree(tmp_d)
    shutil.copytree(org_d,tmp_d)

    hbd.prep_bins(tmp_d)
    hbd.prep_deps(tmp_d)

    # load simulation
    sim = flopy.mf6.MFSimulation.load(sim_ws=tmp_d)
    # load flow model
    gwf = sim.get_model()

    #fix the fucking wrapped format bullshit from the 1980s
    sim.simulation_data.max_columns_of_data = gwf.modelgrid.ncol
    sim.write_simulation()

    # run the model once to make sure it works
    pyemu.os_utils.run("mf6",cwd=tmp_d)
    # run modpath7
    pyemu.os_utils.run(r'mp7 freyberg_mp.mpsim', cwd=tmp_d)


    sr = pyemu.helpers.SpatialReference.from_namfile(
            os.path.join(tmp_d, "freyberg6.nam"),
            delr=gwf.dis.delr.array, delc=gwf.dis.delc.array)

    # specify a template directory (i.e. the PstFrom working folder)
    template_ws = os.path.join("freyberg6_template")
    start_datetime="1-1-2008"
    # instantiate PstFrom
    pf = pyemu.utils.PstFrom(original_d=tmp_d, # where the model is stored
                                new_d=template_ws, # the PEST template folder
                                remove_existing=True, # ensures a clean start
                                longnames=True, # set False if using PEST/PEST_HP
                                spatial_reference=sr, #the spatial reference we generated earlier
                                zero_based=False, # does the MODEL use zero based indices? For example, MODFLOW does NOT
                                start_datetime=start_datetime, # required when specifying temporal correlation between parameters
                                echo=False) # to stop PstFrom from writting lots of infromation to the notebook; experiment by setting it as True to see the difference; usefull for troubleshooting


    df = pd.read_csv(os.path.join(template_ws,"heads.csv"),index_col=0)
    hds_df = pf.add_observations("heads.csv", # the model output file to read
                                insfile="heads.csv.ins", #optional, the instruction file name
                                index_cols="time", #column header to use as index; can also use column number (zero-based) instead of the header name
                                use_cols=list(df.columns.values), #names of columns that include observation values; can also use column number (zero-based) instead of the header name
                                prefix="hds") #prefix to all observation names; choose something logical and easy o find. We use it later on to select obsevrations






    df = pd.read_csv(os.path.join(template_ws, "sfr.csv"), index_col=0)
    sfr_df = pf.add_observations("sfr.csv", # the model output file to read
                                insfile="sfr.csv.ins", #optional, the instruction file name
                                index_cols="time", #column header to use as index; can also use column number (zero-based) instead of the header name
                                use_cols=list(df.columns.values), #names of columns that include observation values; can also use column number (zero-based) instead of the header name
                                prefix="sfr") #prefix to all observation names



    # exponential variogram for spatially varying parameters
    v_space = pyemu.geostats.ExpVario(contribution=1.0, #sill
                                        a=1000, # range of correlation; length units of the model. In our case 'meters'
                                        anisotropy=1.0, #name says it all
                                        bearing=0.0 #angle in degrees East of North corresponding to anisotropy ellipse
                                        )

    # geostatistical structure for spatially varying parameters
    grid_gs = pyemu.geostats.GeoStruct(variograms=v_space, transform='log') 

    # exponential variogram for time varying parameters
    v_time = pyemu.geostats.ExpVario(contribution=1.0, #sill
                                        a=60, # range of correlation; length time units (days)
                                        anisotropy=1.0, #do not change for 1-D time
                                        bearing=0.0 #do not change for 1-D time
                                        )

    # geostatistical structure for time varying parameters
    temporal_gs = pyemu.geostats.GeoStruct(variograms=v_time, transform='none') 


    ib = gwf.dis.idomain.get_data(layer=0)

    def add_mult_pars(f, lb=0.2, ub=5.0, ulb=0.01, uub=100, add_coarse=True):
        if isinstance(f,str):
            base = f.split(".")[1].replace("_","")
        else:
            base = f[0].split(".")[1]
        # grid (fine) scale parameters
        pf.add_parameters(f,
                        zone_array=ib,
                        par_type="grid", #specify the type, these will be unique parameters for each cell
                        geostruct=grid_gs, # the gestatisical structure for spatial correlation 
                        par_name_base=base+"gr", #specify a parameter name base that allows us to easily identify the filename and parameter type. "_gr" for "grid", and so forth.
                        pargp=base+"gr", #likewise for the parameter group name
                        lower_bound=lb, upper_bound=ub, #parameter lower and upper bound
                        ult_ubound=uub, ult_lbound=ulb # The ultimate bounds for multiplied model input values. Here we are stating that, after accounting for all multipliers, Kh cannot exceed these values. Very important with multipliers
                        )
                        
        # pilot point (medium) scale parameters
        pf.add_parameters(f,
                            zone_array=ib,
                            par_type="pilotpoints",
                            geostruct=grid_gs,
                            par_name_base=base+"pp",
                            pargp=base+"pp",
                            lower_bound=lb, upper_bound=ub,
                            ult_ubound=uub, ult_lbound=ulb,
                            pp_space=5) # `PstFrom` will generate a unifrom grid of pilot points in every 4th row and column
        if add_coarse==True:
            # constant (coarse) scale parameters
            pf.add_parameters(f,
                                zone_array=ib,
                                par_type="constant",
                                geostruct=grid_gs,
                                par_name_base=base+"cn",
                                pargp=base+"cn",
                                lower_bound=lb, upper_bound=ub,
                                ult_ubound=uub, ult_lbound=ulb)
        return


    tag = "npf_k_"
    files = [f for f in os.listdir(template_ws) if tag in f.lower() and f.endswith(".txt")]
    for f in files:
        add_mult_pars(f, lb=0.2, ub=5.0, ulb=0.01, uub=100)


    # for Kv
    tag = "npf_k33"
    files = [f for f in os.listdir(template_ws) if tag in f.lower() and f.endswith(".txt")]
    for f in files:
        add_mult_pars(f, lb=0.2, ub=5.0, ulb=0.01, uub=100)

    # for Ss
    tag = "sto_ss"
    files = [f for f in os.listdir(template_ws) if tag in f.lower() and f.endswith(".txt")]
    # only for layer 2 and 3; we aren't monsters
    for f in files[1:]: 
        add_mult_pars(f, lb=0.2, ub=5.0, ulb=1e-6, uub=1e-3)

    # For Sy
    tag = "sto_sy"
    files = [f for f in os.listdir(template_ws) if tag in f.lower() and f.endswith(".txt")]
    # only for layer 1
    f = files[0]
    add_mult_pars(f, lb=0.2, ub=5.0, ulb=0.01, uub=0.4)

    # For porosity
    tag = "ne_"
    files = [f for f in os.listdir(template_ws) if tag in f.lower() and f.endswith(".txt")]
    for f in files: 
        add_mult_pars(f, lb=0.2, ub=5.0, ulb=0.01, uub=0.4)



    dts = pd.to_datetime(start_datetime) + pd.to_timedelta(np.cumsum(sim.tdis.perioddata.array["perlen"]),unit='d')

    dts

    # for Recharge; 
    tag = "rch_recharge"
    files = [f for f in os.listdir(template_ws) if tag in f.lower() and f.endswith(".txt")]
    sp = [int(f.split(".")[1].split('_')[-1]) for f in files]
    d = {s:f for s,f in zip(sp,files)}
    sp.sort()
    files = [d[s] for s in sp]
    #for f in files:
    # doing it for all spds takes too long
    add_mult_pars(files, lb=0.2, ub=5.0, ulb=2e-6, uub=2e-4, add_coarse=False)
        

    for f in files:   
        # multiplier that includes temporal correlation
        # get the stress period number from the file name
        kper = int(f.split('.')[1].split('_')[-1]) - 1  
        # add the constant parameters (with temporal correlation)
        pf.add_parameters(filenames=f,
                        zone_array=ib,
                        par_type="constant",
                        par_name_base=f.split('.')[1]+"tcn",
                        pargp=f.split('.')[1]+"tcn",
                        lower_bound=0.5, upper_bound=1.5,
                        ult_ubound=2e-4, ult_lbound=2e-5,
                        datetime=dts[kper], # this places the parameter value on the "time axis"
                        geostruct=temporal_gs)



    tag = "ghb_stress_period_data"
    files = [f for f in os.listdir(template_ws) if tag in f.lower() and f.endswith(".txt")]

    for f in files:
        # constant and grid scale multiplier conductance parameters
        name = 'ghbcond'
        pf.add_parameters(f,
                            par_type="grid",
                            geostruct=grid_gs,
                            par_name_base=name+"gr",
                            pargp=name+"gr",
                            index_cols=[0,1,2], #column containing lay,row,col
                            use_cols=[4], #column containing conductance values
                            lower_bound=0.1,upper_bound=10.0,
                            ult_lbound=0.1, ult_ubound=100) #absolute limits
        pf.add_parameters(f,
                            par_type="constant",
                            geostruct=grid_gs,
                            par_name_base=name+"cn",
                            pargp=name+"cn",
                            index_cols=[0,1,2],
                            use_cols=[4],  
                            lower_bound=0.1,upper_bound=10.0,
                            ult_lbound=0.1, ult_ubound=100) #absolute limits

        # constant and grid scale additive head parameters
        name = 'ghbhead'
        pf.add_parameters(f,
                            par_type="grid",
                            geostruct=grid_gs,
                            par_name_base=name+"gr",
                            pargp=name+"gr",
                            index_cols=[0,1,2],
                            use_cols=[3],   # column containing head values
                            lower_bound=-2.0,upper_bound=2.0,
                            par_style="a", # specify additive parameter
                            transform="none", # specify not log-transform
                            ult_lbound=32.5, ult_ubound=42) #absolute limits; make sure head is never lower than the bottom of layer1
        pf.add_parameters(f,
                            par_type="constant",
                            geostruct=grid_gs,
                            par_name_base=name+"cn",
                            pargp=name+"cn",
                            index_cols=[0,1,2],
                            use_cols=[3],
                            lower_bound=-2.0,upper_bound=2.0, 
                            par_style="a", 
                            transform="none",
                            ult_lbound=32.5, ult_ubound=42) 



    files = [f for f in os.listdir(template_ws) if "wel_stress_period_data" in f and f.endswith(".txt")]
    sp = [int(f.split(".")[1].split('_')[-1]) for f in files]
    d = {s:f for s,f in zip(sp,files)}
    sp.sort()
    files = [d[s] for s in sp]

    for f in files:
        # get the stress period number from the file name
        kper = int(f.split('.')[1].split('_')[-1]) - 1  
        
        # add the constant parameters (with temporal correlation)
        pf.add_parameters(filenames=f,
                            index_cols=[0,1,2], #columns that specify cell location
                            use_cols=[3],       #columns with parameter values
                            par_type="constant",    #each well will be adjustable
                            par_name_base="welcst",
                            pargp="welcst", 
                            upper_bound = 1.5, lower_bound=0.5,
                            datetime=dts[kper], # this places the parameter value on the "time axis"
                            geostruct=temporal_gs)
        
        # add the grid parameters; each individual well
        pf.add_parameters(filenames=f,
                            index_cols=[0,1,2], #columns that specify cell location 
                            use_cols=[3],       #columns with parameter values
                            par_type="grid",    #each well will be adjustable
                            par_name_base="welgrd",
                            pargp="welgrd", 
                            upper_bound = 1.5, lower_bound=0.5,
                            datetime=dts[kper]) # this places the parameter value on the "time axis"
                         

    # SFR conductance
    tag = "sfr_packagedata"
    files = [f for f in os.listdir(template_ws) if tag in f.lower() and f.endswith(".txt")]
    assert len(files) == 1 # There can be only one! It is tradition. Jokes.
    print(files)

    f = files[0]
    # constant and grid scale multiplier conductance parameters
    name = "sfrcond"
    pf.add_parameters(f,
                    par_type="grid",
                    geostruct=grid_gs,
                    par_name_base=name+"gr",
                    pargp=name+"gr",
                    index_cols=[0,2,3],
                    use_cols=[9],
                    lower_bound=0.1,upper_bound=10.0,
                    ult_lbound=0.01, ult_ubound=10) #absolute limits
    pf.add_parameters(f,
                    par_type="constant",
                    geostruct=grid_gs,
                    par_name_base=name+"cn",
                    pargp=name+"cn",
                    index_cols=[0,2,3],
                    use_cols=[9],
                    lower_bound=0.1,upper_bound=10.0,
                    ult_lbound=0.01, ult_ubound=10) #absolute limits



    # SFR inflow
    files = [f for f in os.listdir(template_ws) if "sfr_perioddata" in f and f.endswith(".txt")]
    sp = [int(f.split(".")[1].split('_')[-1]) for f in files]
    d = {s:f for s,f in zip(sp,files)}
    sp.sort()
    files = [d[s] for s in sp]
    for f in files:
        # get the stress period number from the file name
        kper = int(f.split('.')[1].split('_')[-1]) - 1  
        # add the parameters
        pf.add_parameters(filenames=f,
                            index_cols=[0], #reach number
                            use_cols=[2],   #columns with parameter values
                            par_type="grid",    
                            par_name_base="sfrgr",
                            pargp="sfrgr", 
                            upper_bound = 1.5, lower_bound=0.5, #don't need ult_bounds because it is a single multiplier
                            datetime=dts[kper], # this places the parameter value on the "time axis"
                            geostruct=temporal_gs)


    # files = [f for f in os.listdir(template_ws) if "ic_strt" in f and f.endswith(".txt")]
    # for f in files:
    #     ar = np.loadtxt(os.path.join(template_ws,f))
    #     ar = ar.reshape((gwf.modelgrid.nrow,gwf.modelgrid.ncol))
    #     np.savetxt(os.path.join(template_ws,f),ar)
    #     base = f.split(".")[1].replace("_","")
    #     df = pf.add_parameters(f,par_type="grid",par_style="d",
    #                       pargp=base,par_name_base=base,upper_bound=50,
    #                      lower_bound=15,zone_array=ib,transform="none")


    pf.mod_sys_cmds.append("mf6") #do this only once
    pf.mod_sys_cmds.append("mp7 freyberg_mp.mpsim") #do this only once


    shutil.copy2(os.path.join("..","part2_1_pstfrom_pest_setup","helpers.py"),"helpers.py")

    pf.add_py_function("helpers.py","extract_hds_arrays_and_list_dfs()",is_pre_cmd=False)

    import helpers
    helpers.test_extract_hds_arrays(template_ws)

    files = [f for f in os.listdir(template_ws) if f.startswith("hdslay") and f.endswith("_t1.txt")]
    for f in files:
        pf.add_observations(f,prefix=f.split(".")[0],obsgp=f.split(".")[0])


    # In[ ]:


    for f in ["inc.csv","cum.csv"]:
        df = pd.read_csv(os.path.join(template_ws,f),index_col=0)
        pf.add_observations(f,index_cols=["totim"],use_cols=list(df.columns.values),
                            prefix=f.split('.')[0],obsgp=f.split(".")[0])




    # run the helper function
    helpers.process_secondary_obs(ws=template_ws)


    pf.add_py_function("helpers.py", # the file which contains the function
                        "process_secondary_obs(ws='.')", #the function, making sure to specify any arguments it may requrie
                        is_pre_cmd=False) # whether it runs before the model system command, or after. In this case, after.


    df = pd.read_csv(os.path.join(template_ws, "sfr.tdiff.csv"), index_col=0)
    _ = pf.add_observations("sfr.tdiff.csv", # the model output file to read
                                insfile="sfr.tdiff.csv.ins", #optional, the instruction file name
                                index_cols="time", #column header to use as index; can also use column number (zero-based) instead of the header name
                                use_cols=list(df.columns.values), #names of columns that include observation values; can also use column number (zero-based) instead of the header name
                                prefix="sfrtd") #prefix to all observation 
                                
    df = pd.read_csv(os.path.join(template_ws, "heads.tdiff.csv"), index_col=0)
    _ = pf.add_observations("heads.tdiff.csv", # the model output file to read
                                insfile="heads.tdiff.csv.ins", #optional, the instruction file name
                                index_cols="time", #column header to use as index; can also use column number (zero-based) instead of the header name
                                use_cols=list(df.columns.values), #names of columns that include observation values; can also use column number (zero-based) instead of the header name
                                prefix="hdstd") #prefix to all observation names



    pst = pf.build_pst()

    obs = pst.observation_data
    obs


    # write a really simple instruction file to read the MODPATH end point file
    out_file = "freyberg_mp.mpend"
    ins_file = out_file + ".ins"
    with open(os.path.join(template_ws, ins_file),'w') as f:
        f.write("pif ~\n")
        f.write("l7 w w w w !part_status! w w !part_time!\n")


    # Now add these observations to the `Pst`:

    # In[ ]:


    pst.add_observations(ins_file=os.path.join(template_ws, ins_file),
                        out_file=os.path.join(template_ws, out_file),
                                pst_path='.')

    # and then check what changed                            
    obs = pst.observation_data
    obs.loc[obs.obsnme=='part_status', 'obgnme'] = 'part'
    obs.loc[obs.obsnme=='part_time', 'obgnme'] = 'part'

    obs.iloc[-2:]

    head_pargps = [i for i in pst.adj_par_groups if 'head' in i]
    head_pargps

    pst.parameter_groups.loc[head_pargps, 'inctyp'] = 'absolute'

    par = pst.parameter_data
    par_names = par.loc[par.parval1==0].parnme


    offset = -10
    par.loc[par_names, 'offset'] = offset
    par.loc[par_names, ['parval1', 'parlbnd', 'parubnd']] -= offset

    par.loc[par_names].head()


    forecasts =[
                'oname:sfr_otype:lst_usecol:tailwater_time:4383.5',
                'oname:sfr_otype:lst_usecol:headwater_time:4383.5',
                'oname:hds_otype:lst_usecol:trgw-0-9-1_time:4383.5',
                'part_time'
                ]
                


    fobs = obs.loc[forecasts,:]
    assert fobs.shape[0] == len(forecasts)
    print(fobs)
    pst.pestpp_options['forecasts'] = forecasts
    pst.control_data.noptmax = 0
    pst.pestpp_options["ies_parameter_ensemble"] = "prior_pe.jcb"
    pst.pestpp_options["save_binary"] = True
    pst.write(os.path.join(template_ws, 'freyberg_mf6.pst'))
    pyemu.os_utils.run('pestpp-glm freyberg_mf6.pst', cwd=template_ws)

    pe = pf.draw(num_reals=20, use_specsim=True) # draw parameters from the prior distribution
    pe.enforce() # enforces parameter bounds
    pe.to_binary(os.path.join(template_ws,"prior_pe.jcb")) #writes the paramter ensemble to binary file
    assert pe.shape[1] == pst.npar

    pst.parameter_data.loc[:,"parval1"] = pe.loc[pe.index[0],pst.par_names].values
    pst.parameter_data.parval1.values

    pst.control_data.noptmax = 0
    pst.write(os.path.join(template_ws,"test.pst"))
    pyemu.os_utils.run("pestpp-glm test.pst",cwd=template_ws)


def run_prior_mc(t_d):
    pst = pyemu.Pst(os.path.join(t_d,"freyberg_mf6.pst"))
    pst.control_data.noptmax = -1
    pst.pestpp_options["ies_num_reals"] = 20
    #pst.pestpp_options["overdue_giveup_fac"] = 5
    pst.write(os.path.join(t_d,"freyberg_mf6.pst"))
    num_workers = 5 #psutil.cpu_count(logical=False)
    pyemu.os_utils.start_workers(t_d,"pestpp-ies","freyberg_mf6.pst",num_workers=num_workers,worker_root=".",master_dir="master_pmc")


def pick_truth(m_d,t_d):
    pst = pyemu.Pst(os.path.join(m_d,"freyberg_mf6.pst"))
    oe = pyemu.ObservationEnsemble.from_binary(pst=pst,filename=os.path.join(m_d,"freyberg_mf6.0.obs.jcb"))
    pe = pyemu.ParameterEnsemble.from_binary(pst=pst,filename=os.path.join(m_d,"freyberg_mf6.0.par.jcb"))
    forecasts = pst.pestpp_options["forecasts"].split(",")
    print(forecasts)
    hw_fore = [f for f in forecasts if "headwater" in f]
    assert len(hw_fore) == 1
    hw_fore = hw_fore[0]
    #use the worst hw value
    hw_vals = oe.loc[:,hw_fore].values.copy()
    #mx = hw_vals.max()
    #amx = np.argmax(hw_vals)
    amx = int(hw_vals.shape[0]/2)
    mx = hw_vals[amx]
    print(hw_vals)
    print(mx,amx)
    ovals = oe._df.iloc[amx,:]
    print(ovals)
    pst.observation_data.loc[:,"obsval"] = ovals
    pvals = pe._df.iloc[amx,:]
    pst.parameter_data.loc[:,"parval1"] = pvals
    pst.control_data.noptmax = 0
    truth_d = "truth_template"
    if os.path.exists(truth_d):
        shutil.rmtree(truth_d)
    shutil.copytree(t_d,truth_d)
    pst.write(os.path.join(truth_d,"truth.pst"),version=2)
    pyemu.os_utils.run("pestpp-ies truth.pst",cwd=truth_d)

    # fig, axes = plt.subplots(len(forecasts), 1, figsize=(5 * len(forecasts),5))
    # for forecast,ax in zip(forecasts,axes):
    #     ax.hist(oe.loc[:,forecast].values)
    #     ylim = ax.get_ylim()
    #     ax.plot([ovals[forecast],ovals[forecast]],ylim,"r-")
    #     ax.set_title(forecast,loc="left")
    # plt.show()

def prep_obs_data(truth_d):
    hds_df = pd.read_csv(os.path.join(truth_d, 'heads.csv'))
    sfr_df = pd.read_csv(os.path.join(truth_d, 'sfr.csv'))
    mp_obs = pd.read_csv(os.path.join(truth_d, 'freyberg_mp.mpend'), skiprows=6, header=None, usecols=[3,5], delim_whitespace=True)
    
    # prep calib obs
    obs_sites = ['GAGE-1','TRGW-0-26-6','TRGW-2-26-6','TRGW-0-3-8','TRGW-2-3-8']
    truth_data = pd.DataFrame()
    truth_data = pd.concat([truth_data, pd.melt(hds_df, id_vars=['time'], var_name='site')])
    truth_data = pd.concat([truth_data, pd.melt(sfr_df, id_vars=['time'], var_name='site')])
    truth_data.set_index('site', inplace=True)
    truth_data.loc['part_time', ['time', 'value']] = 1e30, mp_obs.iloc[:,-1].values[0]
    
    # prep hm obs
    obs_data = truth_data.loc[obs_sites]
    # add a wee bit of extra random noise
    for i in obs_sites:
        if i=='GAGE-1':
            scale = abs(0.1*truth_data.loc[i, 'value'].mean())
            obs_data.loc[i, 'value'] = np.random.normal(truth_data.loc[i, 'value'], scale=scale)
        else:
            obs_data.loc[i, 'value'] = np.random.normal(truth_data.loc[i, 'value'], scale=0.1)
    

    #nnz obs
    pred_data = truth_data.loc[~truth_data.index.isin(obs_sites)]
    # write files
    obs_data.to_csv(os.path.join(truth_d, 'obs_data.csv'))
    pred_data.to_csv(os.path.join(truth_d, 'pred_data.csv'))


def store_truth_model(truth_d):
    # folder containing original model files
    org_d = os.path.join(truth_d)
    # a dir to hold a copy of the org model files
    tmp_d = os.path.join('..','..', 'models', 'daily_freyberg_mf6_truth')
    if os.path.exists(tmp_d):
        shutil.rmtree(tmp_d)
    #shutil.copytree(org_d,tmp_d)
    os.makedirs(tmp_d)
    pst = pyemu.Pst(os.path.join(truth_d,"truth.pst"))
    pst.write(os.path.join(tmp_d,"truth.pst"),version=2)
    shutil.copy2(os.path.join(truth_d,"obs_data.csv"),os.path.join(tmp_d,"obs_data.csv"))
    shutil.copy2(os.path.join(truth_d,"pred_data.csv"),os.path.join(tmp_d,"pred_data.csv"))

def store_truth_k(truth_d):
    sim = flopy.mf6.MFSimulation.load(sim_ws=truth_d, verbosity_level=0) #modflow.Modflow.load(fs.MODEL_NAM,model_ws=working_dir,load_only=[])
    gwf= sim.get_model()

    k = gwf.npf.k.get_data()
    top = gwf.dis.top.get_data()
    bot = gwf.dis.botm.get_data()

    thick = np.zeros_like(bot)
    thick[gwf.dis.idomain.get_data()==0] = np.nan
    thick[0] = top-bot[0]
    thick[1] = bot[0]-bot[1]
    thick[2] = bot[1]-bot[2]
    thick[gwf.dis.idomain.get_data()==0] = np.nan
    k_eff = (thick * k ).sum(axis=0) / thick.sum(axis=0)

    np.savetxt(os.path.join('..','..', 'models', 'daily_freyberg_mf6_truth','truth_hk.txt'), k_eff)
    return

if __name__ == "__main__":
    setup_pst()
    run_prior_mc("freyberg6_template")
    pick_truth("master_pmc","freyberg6_template")
    prep_obs_data("truth_template")
    store_truth_model("truth_template")
    store_truth_k("truth_template")
    