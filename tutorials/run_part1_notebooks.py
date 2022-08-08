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

intro_dirs = [f.name for f in os.scandir('.') if f.is_dir() and f.name.startswith('intro_')]
part1_dirs = [f.name for f in os.scandir('.') if f.is_dir() and f.name.startswith('part1_')]

print(intro_dirs)

for dir in intro_dirs:
    nbfiles = [i for i in os.listdir(dir) if i.endswith('.ipynb')]
    for nb in nbfiles:
        run_nb(nb, dir)

for dir in part1_dirs:
    nbfiles = [i for i in os.listdir(dir) if i.endswith('.ipynb')]
    for nb in nbfiles:
        run_nb(nb, dir)

#nb_dir = "intro_to_freyberg_model"
#nb_file = "freyberg_intro_model.ipynb"
#run_nb(nb_file, nb_dir)
#
#nb_dir = "intro_to_regression"
#nb_file = "intro_to_regression.ipynb"
#run_nb(nb_file, nb_dir)
#
#nb_dir = "intro_to_geostatistics"
#nb_file = "intro_to_geostatistics.ipynb"
#run_nb(nb_file, nb_dir)
#
#nb_dir = "intro_to_pyemu"
#nb_file = "intro_to_pyemu.ipynb"
#run_nb(nb_file, nb_dir)
#
#nb_dir = "part1_1_trial_and_error"
#nb_file = "freyberg_trial_and_error.ipynb"
#run_nb(nb_file, nb_dir)
#
#nb_dir = "part1_2_pest_setup"
#nb_file = "freyberg_pest_setup.ipynb"
#run_nb(nb_file, nb_dir)
#
#nb_dir = "part1_3_calibrate_k"
#nb_file = "freyberg_k.ipynb"
#run_nb(nb_file, nb_dir)
#
#nb_dir = "part1_4_calibrate_k_and_r"
#nb_file = "freyberg_k_and_r.ipynb"
#run_nb(nb_file, nb_dir)
#
#nb_dir = "part1_5_calibrate_k_r_fluxobs"
#nb_file = "freyberg_k_r_fluxobs.ipynb"
#run_nb(nb_file, nb_dir)