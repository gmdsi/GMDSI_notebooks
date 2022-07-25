import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import shutil
import platform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import shutil
import pyemu
import flopy

def prep_forecasts(pst, model_times=False):
    pred_csv = os.path.join('..', '..', 'models', 'daily_freyberg_mf6_truth',"pred_data.csv")
    assert os.path.exists(pred_csv)
    pred_data = pd.read_csv(pred_csv)
    pred_data.set_index('site', inplace=True)
    
    if type(model_times) == bool:
        model_times = [float(i) for i in pst.observation_data.time.unique()]
        
    ess_obs_data = {}
    for site in pred_data.index.unique().values:
        site_obs_data = pred_data.loc[site,:].copy()
        if isinstance(site_obs_data, pd.Series):
            site_obs_data.loc["site"] = site_obs_data.index.values
        if isinstance(site_obs_data, pd.DataFrame):
            site_obs_data.loc[:,"site"] = site_obs_data.index.values
            site_obs_data.index = site_obs_data.time
            sm = site_obs_data.value.rolling(window=20,center=True,min_periods=1).mean()
            sm_site_obs_data = sm.reindex(model_times,method="nearest")
        #ess_obs_data.append(pd.DataFrame9sm_site_obs_data)
        ess_obs_data[site] = sm_site_obs_data
    obs_data = pd.DataFrame(ess_obs_data)

    obs = pst.observation_data
    obs_names = [o for o in pst.obs_names if o not in pst.nnz_obs_names]

    # get list of times for obs name sufixes
    time_str = obs_data.index.map(lambda x: f"time:{x}").values
    # empyt list to keep track of misssing observation names
    missing=[]
    for col in obs_data.columns:
        if col.lower()=='part_time':
            obs_sufix = col.lower()
        else:
        # get obs list sufix for each column of data
            obs_sufix = col.lower()+"_"+time_str
        if type(obs_sufix)==str:
            obs_sufix=[obs_sufix]

        for string, oval, time in zip(obs_sufix,obs_data.loc[:,col].values, obs_data.index.values):
                if not any(string in obsnme for obsnme in obs_names):
                    missing.append(string)
                # if not, then update the pst.observation_data
                else:
                    # get a list of obsnames
                    obsnme = [ks for ks in obs_names if string in ks] 
                    if type(obsnme) == str:
                        obsnme=[obsnme]
                    obsnme = obsnme[0]
                    if obsnme=='part_time':
                        oval = pred_data.loc['part_time', 'value']
                    # assign the obsvals
                    obs.loc[obsnme,"obsval"] = oval
                        ## assign a generic weight
                        #if time > 3652.5 and time <=4018.5:
                        #    obs.loc[obsnme,"weight"] = 1.0      
    return 

def prep_deps(template_ws, dep_dir=None):
    dep_dir=os.path.join('..','..','dependencies')
    for org_d in [os.path.join(dep_dir,"flopy"),os.path.join(dep_dir,"pyemu")]:
        #org_d = i.path
        new_d = os.path.join(template_ws, os.path.basename(org_d))
        if os.path.exists(new_d):
            shutil.rmtree(new_d)
        shutil.copytree(org_d, new_d)
    return


if "linux" in platform.platform().lower():
    bin_path = os.path.join("..","..", "bin_new", "linux")
elif "darwin" in platform.platform().lower() or "macos" in platform.platform().lower():
    bin_path = os.path.join("..","..", "bin_new", "mac")
else:
    bin_path = os.path.join("..", "..", "bin_new", "win")

def prep_bins(dest_path):
    files = os.listdir(bin_path)
    for f in files:
        if os.path.exists(os.path.join(dest_path,f)):
            os.remove(os.path.join(dest_path,f))
        shutil.copy2(os.path.join(bin_path,f),os.path.join(dest_path,f))

def run_notebook(notebook_filename, path):
    notebook_filename = os.path.join(path,notebook_filename)
    with open(notebook_filename, encoding="utf8") as f:
        nb = nbformat.read(f, nbformat.NO_CONVERT)

    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': os.path.join(path)}})
    with open(notebook_filename, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    print(f'notebook complete:{notebook_filename}')
    return

def dir_cleancopy(org_d, new_d, delete_orgdir=False):
    # remove existing folder
    if os.path.exists(new_d):
        shutil.rmtree(new_d)
    # copy the original model folder across
    shutil.copytree(org_d, new_d)
    print(f'Files copied from:{org_d}\nFiles copied to:{new_d}')

    if delete_orgdir==True:
        shutil.rmtree(org_d)
        print(f'Hope you did that on purpose. {org_d} has been deleted.')
    #prep_bins(new_d)
    return

def unzip(path_to_zip_file,directory_to_extract_to):
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)
    return


def make_truth(truth_d):
    """Gets a realisation from the pest_setup backup folder 
        and prepares the 'truth' model folder."""
    # remove existing folder
    if os.path.exists(truth_d):
        shutil.rmtree(truth_d)
    # make truth dir
    os.mkdir(truth_d)

    # pest setup template folder
    t_d = os.path.join('..', 'models', 'freyberg_pstfrom_pest_setup')
    pst = pyemu.Pst(os.path.join(t_d, 'freyberg_mf6.pst'))

    # choose realisation; this one gives headwater forecast > 95%
    real=187
    pst.parrep(parfile=os.path.join(t_d, 'prior_pe.jcb'), real_name=real, binary_ens_file=True)
    pst.write_input_files(pst_path=t_d)

    # run forward run so that parameter input files are updated; 
    pyemu.os_utils.run('python forward_run.py', cwd=t_d)

    # load, rewrite and run simulation
    sim = flopy.mf6.MFSimulation.load(sim_ws=t_d, verbosity_level=0)

    sim.set_sim_path(truth_d)
    sim.set_all_data_external(check_data=True)
    sim.write_simulation()

    # run mf6 so that model output files are available
    pyemu.os_utils.run('mf6', cwd=truth_d)

    # copy modpath7 model files
    files = [f for f in os.listdir(t_d) if f.startswith('freyberg_mp') or f.startswith('pm.pg1')]
    for f in files:
        shutil.copy2(os.path.join(t_d, f),os.path.join(truth_d,f))

    # run mp7
    pyemu.os_utils.run("mp7 freyberg_mp.mpsim", cwd=truth_d)


    ## rename model output files because of silly design decisions a while back
    #for f in ['heads.csv', 'sfr.csv']:
    #    os.rename(os.path.join(truth_d, f), os.path.join(truth_d, f.split('.')[0]+'.meas.csv'))
   
    ##rename output file
    #f='freyberg_mp.mpend'
    #os.rename(os.path.join(truth_d, f), os.path.join(truth_d, f+'.meas'))
    def make_obsdata(obs_df, obs_data, obs_sites=[], noise_scale=None):
        for site in obs_df.columns:
            if noise_scale==None:
                noise_scale=abs(0.2*obs_df[site].mean())
            y_true = obs_df[site].values
            times_true = obs_df.index.values
            if site in obs_sites:
                x = np.linspace(obs_df.index[0], obs_df.index[-1], 500)
                y = np.random.normal(np.interp(x, times_true, y_true), scale=noise_scale)
            else:
                x = times_true
                y = y_true
            arr = np.array([x.shape[0]*[site],x,y])
            obs_data = pd.concat([obs_data, pd.DataFrame(arr.T)], axis=0, ignore_index=True)
        obs_data.iloc[:,1:] = obs_data.iloc[:,1:].astype('float')
        
        return obs_data

        
    meas_sfr = pd.read_csv(os.path.join(truth_d,"sfr.csv"),
                        index_col=0)
    meas_hds = pd.read_csv(os.path.join(truth_d,"heads.csv"),
                        index_col=0)

    obs_sites = ['GAGE-1','TRGW-0-26-6','TRGW-2-26-6','TRGW-0-3-8','TRGW-2-3-8']

    obs_data = pd.DataFrame()
    obs_data = make_obsdata(meas_sfr, obs_data, obs_sites, noise_scale=None)
    obs_data = make_obsdata(meas_hds, obs_data, obs_sites, noise_scale=0.1)
    obs_data.columns=['site', 'time', 'value']
    obs_data.set_index('site', inplace=True)

    # add particle time obs
    mp_obs = pd.read_csv(os.path.join(truth_d, 'freyberg_mp.mpend'), skiprows=6, header=None, usecols=[3,5], delim_whitespace=True)
    obs_data.loc['part_time', ['time', 'value']] = '', mp_obs.iloc[:,-1].values[0]
    obs_data.to_csv(os.path.join(truth_d, 'obs_data.csv'))
    
    return (print('Truth is updated.'))

# instruction files
def make_ins_from_csv(csvfile, tmp_d):
    with open(os.path.join(tmp_d, csvfile),'r') as f:
        lines = f.readlines()
    colnames = lines[0].strip().lower().split(',')[1:]
    with open(os.path.join(tmp_d, csvfile+'.ins'),'w') as f:
        f.write("pif ~\n")
        f.write("l1\n")
        for line in lines[1:]:
            row_time = float(line.split(',')[0])
            line='l1 '
            for i in colnames:
                line+= f'~,~ !{i}:{row_time}! ' 
            line+='\n'
            f.write(line)
    try:
        pyemu.utils.run(f'inschek {csvfile}.ins {csvfile}', cwd=tmp_d)
        print(f'ins file for {csvfile} prepared.')
    except: 
        print('something is wrong with the observation & instruction pair. See INSCHEK.')
    return 

def clean_pst4pestchek(pstfile, par):
    """Hack to bypass NUMCOM/DERCOM conflict with PESTCHEK 
        for pyemu-written control files. Could be better."""
    with open(pstfile, 'r') as f:
        lines = f.readlines()
    lines = [i.replace('point         1\n', 'point\n') for i in lines ]    
    lines = [i.replace('0.0000000000E+00      1          \n', '0.0000000000E+00      \n') if any(i.startswith(xs) for xs in par['parnme']) else i for i in lines ]
    with open(pstfile, 'w') as f:
        for line in lines:
            f.write(line)
    return

def make_part_ins(tmp_d):
    # write a really simple instruction file to read the MODPATH end point file
    out_file = "freyberg_mp.mpend"
    ins_file = out_file + ".ins"
    with open(os.path.join(tmp_d, ins_file),'w') as f:
        f.write("pif ~\n")
        f.write("l7 w w w w w w !part_time!\n")
    return


def prep_pest(tmp_d):
    """Prepares the PEST setup for part 1 of the tutorials.
        Used by the freyberg_pest_setup notebook."""
    
    pyemu.os_utils.run('mf6', cwd=tmp_d)
    pyemu.os_utils.run(r'mp7 freyberg_mp.mpsim', cwd=tmp_d)

    # load simulation
    sim = flopy.mf6.MFSimulation.load(sim_ws=tmp_d,load_only=['DIS'], verbosity_level=0)
    # load flow model
    gwf = sim.get_model()
    # get dis info
    nrow, ncol = gwf.dis.nrow.get_data(), gwf.dis.ncol.get_data()
    nlay=gwf.dis.nlay.get_data()

    # make hk pars tpl
    for lay in range(nlay):
        filename = f'freyberg6.npf_k_layer{lay+1}.txt.tpl'
        with open(os.path.join(tmp_d,filename),'w+') as f:
            f.write("ptf ~\n")
            for i in range(nrow):
                for j in range(ncol):
                    f.write(f" ~     hk{lay+1}   ~")
                f.write("\n")
        filename = f'freyberg_mp.ne_layer{lay+1}.txt.tpl'
        with open(os.path.join(tmp_d,filename),'w+') as f:
            f.write("ptf ~\n")
            for i in range(nrow):
                for j in range(ncol):
                    f.write(f" ~     ne{lay+1}   ~")
                f.write("\n")
    # rch multiplier pars tpl
    spdfiles = [f'freyberg6.rch_recharge_{i}.txt' for i in range(14)]
    with open(os.path.join(tmp_d,'freyberg6.rch'),'r') as f:
        lines = f.readlines()
    lines = [item.replace('1.0', " ~     rch0   ~") if any(spd in item for spd in spdfiles) else item.replace('1.0', " ~     rch1   ~") for item in lines]
    with open(os.path.join(tmp_d,'freyberg6.rch.tpl'),'w') as f:
        f.write("ptf ~\n")
        for l in lines:
            f.write(l)

    # make ins files
    make_ins_from_csv('heads.csv', tmp_d)
    make_ins_from_csv('sfr.csv', tmp_d)
    make_part_ins(tmp_d)

    # build lists of tpl, in, ins, and out files
    tpl_files = [os.path.join(tmp_d, f) for f in os.listdir(tmp_d) if f.endswith(".tpl")]
    in_files = [f.replace(".tpl","") for f in tpl_files]
    ins_files = [os.path.join(tmp_d, f) for f in os.listdir(tmp_d) if f.endswith(".ins")]
    out_files = [f.replace(".ins","") for f in ins_files]

    # build a control file
    pst = pyemu.Pst.from_io_files(tpl_files,in_files,
                                            ins_files,out_files, pst_path='.')
    pst.try_parse_name_metadata()
    #tidy up
    par=pst.parameter_data
    par.loc[par['parnme'].str.startswith('hk'), ['parlbnd','parval1','parubnd', 'pargp']] = 0.05, 5, 500, 'hk'
    par.loc['rch0', ['parlbnd','parval1','parubnd', 'partrans','pargp']] = 0.5, 1, 2, 'fixed', 'rch'
    par.loc['rch1', ['parlbnd','parval1','parubnd', 'partrans','pargp']] = 0.5, 1, 2, 'fixed', 'rch'
    par.loc['ne1', ['parlbnd','parval1','parubnd', 'partrans','pargp']] = 0.005, 0.01, 0.02, 'fixed', 'porosity'
    
    obs=pst.observation_data
    obs['weight'] = 0
    #obs.loc[:,"time"] = obs.obsnme.apply(lambda x: float(x.split(':')[-1]))
    obs.loc[:,"obgnme"] = obs.obsnme.apply(lambda x: x.split(':')[0])

    with open(os.path.join(tmp_d, 'runmodel.bat'), 'w+') as f:
        f.write('mf6\n')
        f.write('mp7 freyberg_mp.mpsim')

    pst.model_command = 'runmodel.bat'
    pst.control_data.noptmax=0
    pst.pestpp_options['forecasts'] = ['headwater:4383.5','tailwater:4383.5','trgw-0-9-1:4383.5', 'part_time']

    ###-- Set OBSERVATION DATA AND WEIGHTS --###
    # geat meas values
    shutil.copy2(os.path.join('..', '..', 'models', 'daily_freyberg_mf6_truth','obs_data.csv'),
                            os.path.join(tmp_d, 'obs_data.csv'))
    obs_data = pd.read_csv(os.path.join(tmp_d, 'obs_data.csv'))
    obs_data.site = obs_data.site.str.lower()
    obs_data.set_index('site', inplace=True)
    
    # restructure the obsevration data 
    obs_sites = obs_data.index.unique().tolist()
    model_times = obs.loc[:,obs_sites[0]].astype(float)
    ess_obs_data = {}
    for site in obs_sites:
        #print(site)
        site_obs_data = obs_data.loc[site,:].copy()
        if isinstance(site_obs_data, pd.Series):
            site_obs_data.loc["site"] = site_obs_data.index.values
        if isinstance(site_obs_data, pd.DataFrame):
            site_obs_data.loc[:,"site"] = site_obs_data.index.values
            site_obs_data.index = site_obs_data.time
            sm = site_obs_data.value.rolling(window=20,center=True,min_periods=1).mean()
            sm_site_obs_data = sm.reindex(model_times,method="nearest")
        #ess_obs_data.append(pd.DataFrame9sm_site_obs_data)
        ess_obs_data[site] = sm_site_obs_data
    ess_obs_data = pd.DataFrame(ess_obs_data)
        
    ## set the obs values
    obs_names = pst.observation_data.obsnme.tolist()
    obs_data = ess_obs_data.copy()
    # for checking
    org_nnzobs = pst.nnz_obs
    org_nobs = pst.nobs
    # empyt list to keep track of misssing observation names
    missing=[]
    for col in obs_data.columns:
        # get obs list sufix for each column of data
        obs_sufix = obs_data.index.map(lambda x: col.lower()+f':{x}' ).values
        for string, oval, time in zip(obs_sufix, obs_data.loc[:,col].values, obs_data.index.values):
            # get a list of obsnames
            obsnme = [ks for ks in obs_names if string in ks] 
            # assign the obsvals
            obs.loc[obsnme,"obsval"] = oval
            # assign a generic weight
            if time > 3652.5 and time <=4018.5:
                obs.loc[obsnme,"weight"] = 1.0
    # checks
    assert org_nobs-pst.nobs==0, 'oh oh, new observations.'
    assert len(missing)==0, f'The following obs are missing:\n{missing}'

    # set weights
    obs = pst.observation_data
    #obs.loc[obs.obgnme.str.startswith('gage-1'), 'weight'] = 1/ abs(0.1 *obs.loc[obs.obgnme.str.startswith('gage-1')].obsval)
    obs.loc[obs.obgnme.str.startswith('gage-1'), 'weight'] = 0
    obs.loc[obs.obgnme.str.startswith('trgw-0-26-6 '), 'weight'] = 1/ 0.1
    prep_forecasts(pst, model_times)

    # write and run pst check
    pstfile = os.path.join(tmp_d,'freyberg.pst')
    pst.write(pstfile)
    clean_pst4pestchek(pstfile, par)
    #pyemu.utils.run(f'pestchek {os.path.basename(pstfile)}', cwd=tmp_d)
    print(f'written pest control file: {pstfile}')

    return pst


def plot_freyberg(tmp_d):
    # load simulation
    sim = flopy.mf6.MFSimulation.load(sim_ws=tmp_d, verbosity_level=0)
    # load flow model
    gwf = sim.get_model()

    cols = pd.read_csv(os.path.join(tmp_d, 'heads.csv')).columns[1:].tolist()
    obsxy = pd.DataFrame([i.replace('TRGW-','').split('-') for i in cols], columns=['layer','row','col'])
    obsxy['x'] = [gwf.modelgrid.xycenters[0][int(i)+1] for i in obsxy['col'].values]
    obsxy['y'] = [gwf.modelgrid.xycenters[1][int(i)+1] for i in obsxy['row'].values]

    hdobj = gwf.output.head()
    times = hdobj.get_times()
    hds = hdobj.get_data(totim=times[-1])
    hds[hds==1e30]=np.nan

    fig = plt.figure(figsize=(10, 7))

    ax = fig.add_subplot(1, 2, 1, aspect='equal')
    mm = flopy.plot.PlotMapView(model=gwf, ax=ax, layer=0)

    mm.plot_grid(alpha=0.5)
    mm.plot_inactive()
    # Plot grid 
    # you can plot BC cells using the plot_bc() 
    mm.plot_bc('ghb')
    mm.plot_bc('sfr')
    mm.plot_bc('wel')

    levels=np.linspace(np.nanmin(hds), np.nanmax(hds), 10)
    ca = mm.contour_array(hds, masked_values=[1e30], levels=levels, colors='blue', linestyles ='dashed', linewidths=1)
    plt.clabel(ca, fmt='%.1f', colors='k', fontsize=8)

    ax.scatter(obsxy['x'],obsxy['y'], marker='x', c='k',)
    plt.draw()
    ax.set_yticklabels(labels=ax.get_xticklabels(), rotation=90)

    # plo crossection
    ax = fig.add_subplot(1, 2, 2, aspect=200)
    mm = flopy.plot.PlotCrossSection(model=gwf, ax=ax, line={'row':26})
    arr = mm.plot_array(hds, masked_values=[1e30], cmap='Blues')
    cb = plt.colorbar(arr, shrink=0.25, )
    mm.plot_grid()
    mm.plot_inactive()
    mm.plot_bc('ghb')
    mm.plot_bc('sfr')
    mm.plot_bc('wel')
    ax.set_ylim(0,45)
    ax.set_ylabel('elevation (m)')
    ax.set_xlabel('x-axis (m)')
    ax.set_title('cross-section; row:26');
    return 


####
def prep_notebooks(rebuild_truth=True):
    """Runs notebooks, prepares model folders, etc."""

    # removes all the .ipynb checkpoint folders
    for cdir, cpath, cf in os.walk('.'):
        if os.path.basename(cdir).startswith('.ipynb'):
            if os.path.isdir(cdir):
                print('removing {}'.format(cdir))
                shutil.rmtree(cdir)

    # make sure there is a truth model; if not make one
    truth_d = os.path.join('..','models','freyberg_mf6_truth')
    if not os.path.exists(truth_d):
        dir_cleancopy(org_d=os.path.join('..','models','freyberg_mf6'), 
                    new_d=truth_d)
        pyemu.os_utils.run('mf6', cwd=truth_d)
        # rename model output csv because of silly design decisions
        for f in [f for f in os.listdir(truth_d) if f.endswith('.csv')]:
            os.rename(os.path.join(truth_d, f), os.path.join(truth_d, f.split('.')[0]+'.meas.csv'))

        f='freyberg_mp.mpend'
        os.rename(os.path.join(truth_d, f), os.path.join(truth_d, f+'.meas'))

    redo_part1=False
    if redo_part1==True:
        # run the intro_to_regression
        run_notebook('intro_to_regression.ipynb', 'intro_to_regression')

        # run the intro_to_pyemu
        run_notebook('intro_to_pyemu.ipynb', 'intro_to_pyemu')

        # run the sequence of Freyberg model notebooks
        # run the freyberg model
        run_notebook('freyberg_intro_model.ipynb', 'part1_intro_to_model')

        # trial and error
        run_notebook('freyberg_trial_and_error.ipynb', 'part1_trial_and_error')

        # k only calib; takes a few minutes
        run_notebook('freyberg_k.ipynb', 'part1_k')
        dir_cleancopy(org_d=os.path.join('part1_k', 'freyberg_k'), 
                    new_d=os.path.join('..','models','freyberg_k'), 
                    delete_orgdir=True) # reduce occupied disk space

    ## Part 2
    # run the base pest setup and make a backup
    run_notebook('freyberg_pstfrom_pest_setup.ipynb', 'part2_1_pstfrom_pest_setup')
    dir_cleancopy(org_d=os.path.join('part2_1_pstfrom_pest_setup', 'freyberg6_template'), 
                new_d=os.path.join('..','models','freyberg_pstfrom_pest_setup'),
                delete_orgdir=True) # reduce occupied disk space

    if rebuild_truth==True:
        print('Rebuilding truth.')
        ### Generate the truth model; chicken and egg situation going on here.
        # Need to re-run the pest setup notebook again to ensure that the correct Obs are used.
        # Alternative is to accept some manual input here and just make sure the "truth" is setup correctly beforehand?
        #...or just update the obs data...meh...this way burns a bit more silicon, but keeps things organized
        make_truth(truth_d)

        ### Run PEST setup again with correct obs values for consistency...
        # run the base pest setup and make a backup
        run_notebook('freyberg_pstfrom_pest_setup.ipynb', 'part2_1_pstfrom_pest_setup')
        dir_cleancopy(org_d=os.path.join('part2_1_pstfrom_pest_setup', 'freyberg6_template'), 
                    new_d=os.path.join('..','models','freyberg_pstfrom_pest_setup'),
                    delete_orgdir=True) # reduce occupied disk space
        
        # zip the prior cov; it is >100mb. prior_cov.jcb is in gitignore
        import zipfile
        with zipfile.ZipFile(os.path.join('..','models','prior_cov.zip'), 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(os.path.join('..','models','freyberg_pstfrom_pest_setup','prior_cov.jcb'), 
            arcname='prior_cov.jcb')
            zipf.close()
    

    # run the obs&weights notebook and make a backup
    run_notebook('freyberg_obs_and_weights.ipynb', 'part2_2_obs_and_weights')
    dir_cleancopy(org_d=os.path.join('part2_2_obs_and_weights', 'freyberg6_template'), 
                new_d=os.path.join('..','models','freyberg_obs_and_weights'),
                delete_orgdir=True) # reduce occupied disk space



    return print('Notebook folders ready.')


if __name__ == "__main__":
    #make_truth(os.path.join('..','models','freyberg_mf6_truth'))
    prep_notebooks(rebuild_truth=True)
    #prep_pest(os.path.join("pest_files"))

