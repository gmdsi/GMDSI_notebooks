import os
import shutil

html_dir = os.path.abspath(os.path.join("..", "docs"))

cwd = os.getcwd()

clear = False
pdf = False
html = True


def run_nb(nb_file, nb_dir): 
    os.chdir(nb_dir)
    os.system("jupyter nbconvert --execute --ExecutePreprocessor.timeout=1800 --allow-errors --inplace {0}".format(nb_file))
    if html:
        os.system("jupyter nbconvert --to html {0}".format(nb_file))
        md_file = nb_file.replace('.ipynb', '.html')
        shutil.move(md_file, os.path.join(html_dir, md_file))
        print('preped htmlfile: ', os.path.join(html_dir, md_file))
    if pdf:
        os.system("jupyter nbconvert --to pdf {0}".format(nb_file))
    if clear:
        os.system("jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --allow-errors --inplace {0}".format(nb_file))
    os.chdir(cwd)
    return


nb_dir = "intro_to_regression"
nb_file = "intro_to_regression.ipynb"
run_nb(nb_file, nb_dir)

nb_dir = "intro_to_freyberg_model"
nb_file = "freyberg_intro_model.ipynb"
run_nb(nb_file, nb_dir)

nb_dir = "intro_to_pyemu"
nb_file = "intro_to_pyemu.ipynb"
run_nb(nb_file, nb_dir)

nb_dir = "part1_pest_setup"
nb_file = "freyberg_pest_setup.ipynb"
run_nb(nb_file, nb_dir)

nb_dir = "part1_trial_and_error"
nb_file = "freyberg_trial_and_error.ipynb"
run_nb(nb_file, nb_dir)

nb_dir = "part1_k"
nb_file = "freyberg_k.ipynb"
run_nb(nb_file, nb_dir)