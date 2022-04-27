import os
cwd = os.getcwd()
print(cwd)
tutdir=os.path.join(cwd, "..","tutorials")
if os.path.basename(os.path.normpath(cwd))!='autotest':
    tutdir=os.path.join(cwd, ".","tutorials")

def run_nb(nb_file, nb_dir):
    assert nb_dir
    assert os.path.join(nb_dir, nb_file)
    os.chdir(nb_dir)
    os.system(f"jupyter nbconvert --execute --ExecutePreprocessor.timeout=1800 --allow-errors --inplace {nb_file}")    
    os.system(f"jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --allow-errors --inplace {nb_file}")
    os.chdir(cwd)
    print('ran: ', nb_file)
    return

nb_dir=os.path.join(tutdir,'part2_1_pstfrom_pest_setup')
nb_file="freyberg_pstfrom_pest_setup.ipynb"
run_nb(nb_file, nb_dir)

nb_dir=os.path.join(tutdir,'part2_2_obs_and_weights')
nb_file="freyberg_obs_and_weights.ipynb"
run_nb(nb_file, nb_dir)