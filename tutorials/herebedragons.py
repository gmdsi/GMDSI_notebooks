# Oh hello! Is it me you are looking for? 
# This script runs the tutorial notebooks and does some tidying up around the place.

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import shutil
import pyemu
import flopy 

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
    """Gets a realisation from the pest_setup backup folder and prepares a 'truth' model folder."""
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