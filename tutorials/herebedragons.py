# Oh hello! Is it me you are looking for? 
# This script runs notebooks and does some tidying up around the place.

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import shutil

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


# removes all the .ipynb checkpoint folders
for cdir, cpath, cf in os.walk('.'):
    if os.path.basename(cdir).startswith('.ipynb'):
        if os.path.isdir(cdir):
            print('removing {}'.format(cdir))
            shutil.rmtree(cdir)

# run the freyberg model
run_notebook('freyberg_intro_model.ipynb', 'freyberg_intro_to_model')

# run the base pest setup and make a backup
run_notebook('freyberg_setup_pest_interface.ipynb', 'freyberg_pest_setup')
dir_cleancopy(org_d=os.path.join('freyberg_pest_setup'), 
              new_d=os.path.join('..','models','freyberg_pest_setup'))