import os
import shutil
html_dir = os.path.join("..", "docs")

dirs = [d for d in os.listdir(".") if os.path.isdir(d)]
for d in dirs:
    nb_files = [os.path.join(d,f) for f in os.listdir(d) if f.lower().endswith(".ipynb")]
    for nb_file in nb_files:
        print(nb_file)
        os.system("jupyter nbconvert --to markdown {0}".format(nb_file))
        html_file = os.path.basename(nb_file).replace('.ipynb', '.md')
        shutil.move(os.path.join(d, html_file), os.path.join(html_dir, html_file))
        figs_dir = os.path.join(d,html_file.replace(".md", "_files"))
        if os.path.exists(figs_dir):
            shutil.move(figs_dir, os.path.join(html_dir, figs_dir))
        print('preped htmlfile: ', os.path.join(html_dir, html_file))

