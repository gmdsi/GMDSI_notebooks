import os
import shutil

html_dir = os.path.abspath(os.path.join("..", "docs"))

cwd = os.getcwd()

clear = False
pdf = False
html = True
allow_errors = True

def run_nb(nb_file, nb_dir): 
    os.chdir(nb_dir)
    worker_dirs = [d for d in os.listdir(".") if os.path.isdir(d) and d.startswith("worker")]
    for worker_dir in worker_dirs:
        shutil.rmtree(worker_dir)
    if allow_errors:
        os.system("jupyter nbconvert --execute --ExecutePreprocessor.timeout=180000 --allow-errors --inplace {0}".format(nb_file))
    else:
        os.system("jupyter nbconvert --execute --ExecutePreprocessor.timeout=180000 --inplace {0}".format(nb_file))
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

nb_dir = "part2_01_pstfrom_pest_setup"
nb_file = "freyberg_pstfrom_pest_setup.ipynb"
run_nb(nb_file, nb_dir)


nb_dir = "part2_02_obs_and_weights"
nb_file = "freyberg_obs_and_weights.ipynb"
run_nb(nb_file, nb_dir)

nb_dir = "part2_03_prior_monte_carlo"
nb_file = "freyberg_prior_monte_carlo.ipynb"
run_nb(nb_file, nb_dir)

nb_dir = "part2_04_glm"
nb_file = "freyberg_glm_1.ipynb"
run_nb(nb_file, nb_dir)

nb_dir = "part2_05_fosm_and_dataworth"
nb_file = "freyberg_fosm_and_dataworth.ipynb"
run_nb(nb_file, nb_dir)

nb_dir = "part2_04_glm"
nb_file = "freyberg_glm_2.ipynb"
run_nb(nb_file, nb_dir)

nb_dir = "part2_06_ies"
nb_file = "freyberg_ies_1_basics.ipynb"
run_nb(nb_file, nb_dir)
nb_file = "freyberg_ies_2_localization.ipynb"
run_nb(nb_file, nb_dir)
nb_file = "freyberg_ies_3_restarting.ipynb"
run_nb(nb_file, nb_dir)

nb_dir = "part2_07_da"
nb_file = "freyberg_da_prep.ipynb"
run_nb(nb_file, nb_dir)
nb_file = "freyberg_da_run.ipynb"
run_nb(nb_file, nb_dir)

nb_dir = "part2_08_opt"
nb_file = "freyberg_opt_1.ipynb"
run_nb(nb_file, nb_dir)

nb_dir = "part2_08_opt"
nb_file = "freyberg_opt_2.ipynb"
run_nb(nb_file, nb_dir)

nb_dir = "part2_09_mou"
nb_file = "freyberg_mou_1.ipynb"
run_nb(nb_file, nb_dir)

nb_dir = "part2_09_mou"
nb_file = "freyberg_mou_2.ipynb"
run_nb(nb_file, nb_dir)




