import os
import shutil
html_dir = os.path.join("..", "docs")

dirs = [d for d in os.listdir(".") if os.path.isdir(d)]
for d in dirs:
    nb_files = [os.path.join(d,f) for f in os.listdir(d) if f.lower().endswith(".ipynb")]
    for nb_file in nb_files:
        os.system("jupyter nbconvert --to html {0}".format(nb_file))
        html_file = os.path.basename(nb_file).replace('.ipynb', '.html')
        shutil.move(os.path.join(d, html_file), os.path.join(html_dir, html_file))
        print('preped htmlfile: ', os.path.join(html_dir, html_file))

