
import os
cwd = os.getcwd()

clear = False
pdf = False

os.chdir("part2_pstfrom_pest_setup")
os.system("jupyter nbconvert --execute --ExecutePreprocessor.timeout=1800 --allow-errors --inplace freyberg_pstfrom_pest_setup.ipynb")
if pdf:
    os.system("jupyter nbconvert --to pdf freyberg_pstfrom_pest_setup.ipynb")
if clear:
    os.system("jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --allow-errors --inplace freyberg_pstfrom_pest_setup.ipynb.ipynb")

os.chdir(cwd)

os.chdir("part2_obs_and_weights")
os.system("jupyter nbconvert --execute --ExecutePreprocessor.timeout=1800 --allow-errors --inplace freyberg_obs_and_weights.ipynb")
if pdf:
    os.system("jupyter nbconvert --to pdf freyberg_obs_and_weights.ipynb")
if clear:
    os.system("jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --allow-errors --inplace freyberg_obs_and_weights.ipynb")
os.chdir(cwd)

os.chdir("part2_glm")
# glm1
os.system("jupyter nbconvert --execute --ExecutePreprocessor.timeout=180000 --allow-errors --inplace freyberg_glm_1.ipynb")
if pdf:
    os.system("jupyter nbconvert --to pdf freyberg_glm_1.ipynb")
if clear:
    os.system("jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --allow-errors --inplace freyberg_glm_1.ipynb")
# glm2
os.system("jupyter nbconvert --execute --ExecutePreprocessor.timeout=180000 --allow-errors --inplace freyberg_glm_2.ipynb")
if pdf:
    os.system("jupyter nbconvert --to pdf freyberg_glm_2.ipynb")
if clear:
    os.system("jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --allow-errors --inplace freyberg_glm_2.ipynb")
os.chdir(cwd)

os.chdir("part2_ies")
os.system("jupyter nbconvert --execute --ExecutePreprocessor.timeout=1800 --allow-errors --inplace freyberg_ies.ipynb")
if pdf:
    os.system("jupyter nbconvert --to pdf freyberg_ies.ipynb")
if clear:
    os.system("jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --allow-errors --inplace freyberg_ies.ipynb")
os.chdir(cwd)

os.chdir("part2_da")
os.system("jupyter nbconvert --execute --ExecutePreprocessor.timeout=1800 --allow-errors --inplace freyberg_da_prep.ipynb")
if pdf:
    os.system("jupyter nbconvert --to pdf freyberg_da_prep.ipynb")
if clear:
    os.system("jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --allow-errors --inplace freyberg_da_prep.ipynb")
os.chdir(cwd)

os.chdir("part2_da")
os.system("jupyter nbconvert --execute --ExecutePreprocessor.timeout=1800000 --allow-errors --inplace freyberg_da_run.ipynb")
if pdf:
    os.system("jupyter nbconvert --to pdf freyberg_da_run.ipynb")
if clear:
    os.system("jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --allow-errors --inplace freyberg_da_run.ipynb")
os.chdir(cwd)