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
    if allow_errors:
        os.system("jupyter nbconvert --execute --ExecutePreprocessor.timeout=1800 --allow-errors --inplace {0}".format(nb_file))
    else:
        os.system("jupyter nbconvert --execute --ExecutePreprocessor.timeout=1800 --inplace {0}".format(nb_file))
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

intro_dirs = [f.name for f in os.scandir('.') if f.is_dir() and f.name.startswith('part0_')]
part1_dirs = [f.name for f in os.scandir('.') if f.is_dir() and f.name.startswith('part1_')]



for dir in intro_dirs:
    nbfiles = [i for i in os.listdir(dir) if i.endswith('.ipynb')]
    for nb in nbfiles:
        run_nb(nb, dir)

iesrespsurf_notebook = None
for d in part1_dirs:
    nbfiles = [i for i in os.listdir(d) if i.endswith('.ipynb')]
    for nb in nbfiles:
        if "ies" in nb and "resp" in nb:
            iesrespsurf_notebook = (nb,d)
        else:
            run_nb(nb, d)
run_nb(iesrespsurf_notebook[0],iesrespsurf_notebook[1])