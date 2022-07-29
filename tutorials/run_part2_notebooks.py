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

nb_dir = "part2_1_pstfrom_pest_setup"
nb_file = "freyberg_pstfrom_pest_setup.ipynb"
run_nb(nb_file, nb_dir)

nb_dir = "part2_2_obs_and_weights"
nb_file = "freyberg_obs_and_weights.ipynb"
run_nb(nb_file, nb_dir)

nb_dir = "part2_3_prior_monte_carlo"
nb_file = "freyberg_prior_monte_carlo.ipynb"
run_nb(nb_file, nb_dir)

nb_dir = "part2_4_glm"
nb_file = "freyberg_glm_1.ipynb"
run_nb(nb_file, nb_dir)

nb_dir = "part2_5_fosm_and_dataworth"
nb_file = "freyberg_fosm_and_dataworth.ipynb"
run_nb(nb_file, nb_dir)

nb_dir = "part2_4_glm"
nb_file = "freyberg_glm_2.ipynb"
run_nb(nb_file, nb_dir)

nb_dir = "part2_6_ies"
nb_file = "freyberg_ies_1_basics.ipynb"
run_nb(nb_file, nb_dir)
nb_file = "freyberg_ies_2_localization.ipynb"
run_nb(nb_file, nb_dir)

nb_dir = "part2_7_da"
nb_file = "freyberg_da_prep.ipynb"
run_nb(nb_file, nb_dir)
nb_file = "freyberg_da_run.ipynb"
run_nb(nb_file, nb_dir)