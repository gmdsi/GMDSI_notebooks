import os
cwd = os.getcwd()

os.chdir("part2_pstfrom_pest_setup")
os.system("jupyter nbconvert --execute --ExecutePreprocessor.timeout=1800 --allow-errors --inplace freyberg_pstfrom_pest_setup.ipynb")
os.system("jupyter nbconvert --to pdf freyberg_pstfrom_pest_setup.ipynb")
os.chdir(cwd)

os.chdir("part2_obs_and_weights")
os.system("jupyter nbconvert --execute --ExecutePreprocessor.timeout=1800 --allow-errors --inplace freyberg_obs_and_weights.ipynb")
os.system("jupyter nbconvert --to pdf freyberg_obs_and_weights.ipynb")
os.chdir(cwd)

os.chdir("part2_ies")
os.system("jupyter nbconvert --execute --ExecutePreprocessor.timeout=1800 --allow-errors --inplace freyberg_ies.ipynb")
os.system("jupyter nbconvert --to pdf freyberg_ies.ipynb")
os.chdir(cwd)

os.chdir("part2_da")
os.system("jupyter nbconvert --execute --ExecutePreprocessor.timeout=1800 --allow-errors --inplace freyberg_da_prep.ipynb")
os.system("jupyter nbconvert --to pdf freyberg_da_prep.ipynb")
os.chdir(cwd)

os.chdir("part2_da")
os.system("jupyter nbconvert --execute --ExecutePreprocessor.timeout=1800 --allow-errors --inplace freyberg_da_run.ipynb")
os.system("jupyter nbconvert --to pdf freyberg_da_run.ipynb")
os.chdir(cwd)