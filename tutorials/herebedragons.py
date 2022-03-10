# Oh hello! Is it me you are looking for? 
# This script runs the tutorial notebooks and does some tidying up around the place.
# Functions herein are accessed by some notebooks when they are run

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import shutil
import pyemu
import flopy 
import platform
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

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

def dir_cleancopy(org_d, new_d):
    # remove existing folder
    if os.path.exists(new_d):
        shutil.rmtree(new_d)
    # copy the original model folder across
    shutil.copytree(org_d, new_d)
    print(f'Files copied from:{org_d}\nFiles copied to:{new_d}')
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
    t_d = os.path.join('..', 'models', 'freyberg_pest_setup')
    pst = pyemu.Pst(os.path.join(t_d, 'freyberg_mf6.pst'))

    # choose realisation; this one gives extreme-ish headwater predictions
    real=61
    pst.parrep(parfile=os.path.join(t_d, 'prior.jcb'), real_name=real, binary_ens_file=True)
    pst.write_input_files(pst_path=t_d)

    # run forward run so that parameter input files are updated; 
    pyemu.os_utils.run('python forward_run.py', cwd=t_d)

    # load, rewrite and run simulation
    sim = flopy.mf6.MFSimulation.load(sim_ws=t_d, verbosity_level=0)

    sim.set_sim_path(truth_d)
    sim.set_all_data_external(check_data=True)
    sim.write_simulation()

    # run mf6 so that model output files are avilable
    pyemu.os_utils.run('mf6', cwd=truth_d)

    # rename model output files because of silly design decisions a while back
    for f in ['heads.csv', 'sfr.csv']:
        os.rename(os.path.join(truth_d, f), os.path.join(truth_d, f.split('.')[0]+'.meas.csv'))
    
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
    lines = [i.replace('0.0000000000E+00      1          \n', '0.0000000000E+00      \n') if any(i.startswith(xs) for xs in par['parnme']) else i for i in lines ]
    with open(pstfile, 'w') as f:
        for line in lines:
            f.write(line)
    return


def prep_pest(tmp_d):
    """Prepares the PEST setup for part 1 of the tutorials.
        Used by the freyberg_pest_setup notebook."""
    # get the necessary executables; OS agnostic
    bin_dir = os.path.join('..','..','bin')
    if "window" in platform.platform().lower():
        exe_files = [f for f in os.listdir(bin_dir) if f.endswith('exe')]
    else:
        exe_files = [f for f in os.listdir(bin_dir) if not f.endswith('exe')]
    # remove existing folder if it already exists
    if os.path.exists(tmp_d):
        shutil.rmtree(tmp_d)
    # make the folder
    os.mkdir(tmp_d)
    # copy executables across
    for exe_file in exe_files:
        shutil.copy2(os.path.join(bin_dir, exe_file),os.path.join(tmp_d,exe_file))
    
    # folder containing original model files
    org_d = os.path.join('..', '..', 'models', 'freyberg_mf6')
    # copy files across to the temp folder
    for f in os.listdir(org_d):
        shutil.copy2(os.path.join(org_d,f), os.path.join(tmp_d,f))

    # geat meas values
    for csv in ['heads.meas.csv', 'sfr.meas.csv']:
        df = pd.read_csv(os.path.join('..', '..', 'models', 'freyberg_mf6_truth',csv),
                        index_col=0)
        df.to_csv(os.path.join(tmp_d,csv.replace('.meas','')))
    # load simulation
    sim = flopy.mf6.MFSimulation.load(sim_ws=tmp_d,load_only=['DIS'], verbosity_level=0)
    # load flow model
    gwf = sim.get_model()
    # get dis info
    nrow, ncol = gwf.dis.nrow.get_data(), gwf.dis.ncol.get_data()
    nlay=gwf.dis.nlay.get_data()

    # make hk pars tpl
    for lay in range(nlay):
        filename = f'freyberg6.npf_k_layer{lay+1}.tpl'
        with open(os.path.join(tmp_d,filename),'w') as f:
            f.write("ptf ~\n")
            for i in range(nrow):
                for j in range(ncol):
                    f.write(f" ~     hk{lay+1}   ~")
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

    # build lists of tpl, in, ins, and out files
    tpl_files = [os.path.join(tmp_d, f) for f in os.listdir(tmp_d) if f.endswith(".tpl")]
    in_files = [f.replace(".tpl",".txt") for f in tpl_files]
    ins_files = [os.path.join(tmp_d, f) for f in os.listdir(tmp_d) if f.endswith(".ins")]
    out_files = [f.replace(".ins","") for f in ins_files]

    # build a control file
    pst = pyemu.Pst.from_io_files(tpl_files,in_files,
                                            ins_files,out_files, pst_path='.')
    #tidy up
    par=pst.parameter_data
    par.loc[par['parnme'].str.startswith('hk'), ['parlbnd','parval1','parubnd', 'pargp']] = 0.05, 5, 50, 'hk'
    par.loc['rch0', ['parlbnd','parval1','parubnd', 'partrans','pargp']] = 0.5, 1, 2, 'fixed', 'rch'
    par.loc['rch1', ['parlbnd','parval1','parubnd', 'partrans','pargp']] = 0.5, 1, 2, 'fixed', 'rch'
    obs=pst.observation_data
    obs['weight'] = 0
    obs.loc[obs['obsnme'].str.startswith('gage'), 'obgnme'] = 'flux'
    obs.loc[obs['obsnme'].str.startswith('headwater'), 'obgnme'] = 'flux'
    obs.loc[obs['obsnme'].str.startswith('tailwater'), 'obgnme'] = 'flux'
    obs.loc[obs['obsnme'].str.startswith('trgw'), 'obgnme'] = 'hds'
    obs.loc[(obs['obgnme']=='hds') & (obs['1'].astype('float') <=367), 'weight'] = 1/0.1

    pst.model_command = 'mf6'
    pst.control_data.noptmax=0

    pst.pestpp_options['forecasts'] = ['headwater:732.0','tailwater:732.0',
                                    'trgw_2_9_1:732.0','trgw_0_9_1:732.0']

    pstfile = os.path.join(tmp_d,'freyberg.pst')
    pst.write(pstfile)

    clean_pst4pestchek(pstfile, par)
    pyemu.utils.run(f'pestchek {os.path.basename(pstfile)}', cwd=tmp_d)

    return print(f'written pest control file: {pstfile}')


def plot_freyberg(tmp_d):
    # load simulation
    sim = flopy.mf6.MFSimulation.load(sim_ws=tmp_d, verbosity_level=0)
    # load flow model
    gwf = sim.get_model()

    cols = pd.read_csv(os.path.join(tmp_d, 'heads.csv')).columns[1:].tolist()
    obsxy = pd.DataFrame([i.replace('TRGW_','').split('_') for i in cols], columns=['layer','row','col'])
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

    # Plot wells in layer 3
    mm = flopy.plot.PlotMapView(model=gwf, ax=ax, layer=2)
    #arr = mm.plot_array(hds[0], masked_values=[1e30], cmap='Blues')
    #cb = plt.colorbar(arr, shrink=0.5)
    levels=np.linspace(np.nanmin(hds), np.nanmax(hds), 10)
    ca = mm.contour_array(hds, masked_values=[1e30], levels=levels, colors='blue', linestyles ='dashed', linewidths=1)
    plt.clabel(ca, fmt='%.1f', colors='k', fontsize=8)

    mm.plot_bc('wel')
    ax.scatter(obsxy['x'],obsxy['y'], marker='x', c='k',)
    plt.draw()
    ax.set_yticklabels(labels=ax.get_xticklabels(), rotation=90)


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
    ax.set_title('cross-section; row:30');
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


    # run the intro_to_regression
    run_notebook('intro_to_regression.ipynb', 'intro_to_regression')

    # run the intro_to_pyemu
    run_notebook('intro_to_pyemu.ipynb', 'intro_to_pyemu')

    # run the sequence of Freyberg model notebooks
    # run the freyberg model
    run_notebook('freyberg_intro_model.ipynb', 'freyberg_intro_to_model')

    # trial and error
    run_notebook('freyberg_trial_and_error.ipynb', 'freyberg_trial_and_error')

    ##...



    # run the base pest setup and make a backup
    run_notebook('freyberg_pstfrom_pest_setup.ipynb', 'freyberg_pstfrom_pest_setup')
    dir_cleancopy(org_d=os.path.join('freyberg_pstfrom_pest_setup', 'freyberg6_template'), 
                new_d=os.path.join('..','models','freyberg_pstfrom_pest_setup'))

    if rebuild_truth==True:
        ### Generate the truth model; chicken and egg situation going on here.
        # Need to re-run the pest setup notebook again to ensure that the correct Obs are used.
        # Alternative is to accept some manual input here and just make sure the "truth" is setup correctly beforehand?
        #...or just update the obs data...meh...this way burns a bit more silicon, but keeps things organized
        make_truth(truth_d)

        ### Run PEST setup again with correct obs values for consistency...
        # run the base pest setup and make a backup
        run_notebook('freyberg_pstfrom_pest_setup.ipynb', 'freyberg_pstfrom_pest_setup')
        dir_cleancopy(org_d=os.path.join('freyberg_pstfrom_pest_setup', 'freyberg6_template'), 
                    new_d=os.path.join('..','models','freyberg_pstfrom_pest_setup'))
    
    return print('Notebook folders ready.')


if __name__ == "__main__":
    prep_notebooks(rebuild_truth=True)