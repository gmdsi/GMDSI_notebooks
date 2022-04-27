import os

def run_nb(nb_file):
    os.system(f"jupyter nbconvert --execute --ExecutePreprocessor.timeout=1800 --allow-errors --inplace {nb_file}")
    return

run_nb("freyberg_glm_1.ipynb")
run_nb("freyberg_glm_2.ipynb")
