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
    os.system(f"jupyter nbconvert --execute --ExecutePreprocessor.timeout=1800 --inplace {nb_file}")
    os.system(f"jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --inplace {nb_file}")
    os.chdir(cwd)
    print('ran: ', nb_file)
    return

# part0 - theory notebooks
nb_dir=os.path.join(tutdir,'part0_00_intro_to_regression')
run_nb("intro_to_regression.ipynb", nb_dir)

nb_dir=os.path.join(tutdir,'part0_01_intro_to_bayes')
run_nb("intro_to_bayes.ipynb", nb_dir)

nb_dir=os.path.join(tutdir,'part0_01_intro_to_bayes')
run_nb("simple_bayes_demo.ipynb", nb_dir)

nb_dir=os.path.join(tutdir,'part0_02_intro_to_freyberg_model')
run_nb("intro_freyberg_model.ipynb", nb_dir)

nb_dir=os.path.join(tutdir,'part0_intro_to_eva_and_dsi')
run_nb("intro_to_eva_and_dsi.ipynb", nb_dir)

nb_dir=os.path.join(tutdir,'part0_intro_to_geostatistics')
run_nb("intro_to_geostatistics.ipynb", nb_dir)

nb_dir=os.path.join(tutdir,'part0_intro_to_geostatistics')
run_nb("understanding_variograms_and_realizations.ipynb", nb_dir)

nb_dir=os.path.join(tutdir,'part0_intro_to_pyemu')
run_nb("intro_to_pyemu.ipynb", nb_dir)

nb_dir=os.path.join(tutdir,'part0_intro_to_svd')
run_nb("intro_to_svd.ipynb", nb_dir)

# part1 - foundational methods (order matters for some)
nb_dir=os.path.join(tutdir,'part1_00_get_bins')
run_nb("get_bins.ipynb", nb_dir)

nb_dir=os.path.join(tutdir,'part1_01_trial_and_error')
run_nb("freyberg_trial_and_error.ipynb", nb_dir)

nb_dir=os.path.join(tutdir,'part1_02_pest_setup')
run_nb("freyberg_pest_setup.ipynb", nb_dir)

nb_dir=os.path.join(tutdir,'part1_02_pest_setup')
run_nb("pstfrom_sneakpeak.ipynb", nb_dir)

nb_dir=os.path.join(tutdir,'part1_03_calibrate_k')
run_nb("freyberg_k.ipynb", nb_dir)

nb_dir=os.path.join(tutdir,'part1_04_calibrate_k_and_r')
run_nb("freyberg_k_and_r.ipynb", nb_dir)

nb_dir=os.path.join(tutdir,'part1_05_calibrate_k_r_fluxobs')
run_nb("freyberg_k_r_fluxobs.ipynb", nb_dir)

nb_dir=os.path.join(tutdir,'part1_06_glm_response_surface')
run_nb("freyberg_glm_response_surface.ipynb", nb_dir)

nb_dir=os.path.join(tutdir,'part1_06_glm_response_surface')
run_nb("freyberg_glm_response_surface_ies.ipynb", nb_dir)

nb_dir=os.path.join(tutdir,'part1_07_pilotpoints_setup')
run_nb("freyberg_pilotpoints_1_setup.ipynb", nb_dir)

nb_dir=os.path.join(tutdir,'part1_08_pilotpoints_run')
run_nb("freyberg_pilotpoints_2_run.ipynb", nb_dir)

nb_dir=os.path.join(tutdir,'part1_09_regularization')
run_nb("intro_to_regularization.ipynb", nb_dir)

nb_dir=os.path.join(tutdir,'part1_10_intro_to_fosm')
run_nb("intro_to_fosm.ipynb", nb_dir)

nb_dir=os.path.join(tutdir,'part1_11_local_and_global_sensitivity')
run_nb("freyberg_1_local_sensitivity.ipynb", nb_dir)

nb_dir=os.path.join(tutdir,'part1_11_local_and_global_sensitivity')
run_nb("freyberg_2_global_sensitivity.ipynb", nb_dir)

nb_dir=os.path.join(tutdir,'part1_12_monte_carlo')
run_nb("freyberg_monte_carlo.ipynb", nb_dir)

# part2 - advanced workflows
nb_dir=os.path.join(tutdir,'part2_01_pstfrom_pest_setup')
nb_file="freyberg_pstfrom_pest_setup.ipynb"
run_nb(nb_file, nb_dir)

nb_dir=os.path.join(tutdir,'part2_02_obs_and_weights')
nb_file="freyberg_obs_and_weights.ipynb"
run_nb(nb_file, nb_dir)
