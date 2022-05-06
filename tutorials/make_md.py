import os
import shutil
import pandas as pd

def add_header(md_file, parent, nav_order, title):
    with open(md_file, "r+",encoding="utf8") as f:
        first_line = f.readline()
        if first_line != "---":
            lines = f.readlines()
            f.seek(0)
            f.write(
                    f"""
---
layout: default
title: {title}
parent: {parent}
nav_order: {nav_order}
---
                    """
                    )
            f.write(first_line)
            f.writelines(lines) 
    return


cwd = os.getcwd()
docs_dir = os.path.join("..", "docs")

dirs = [d for d in os.listdir(".") if os.path.isdir(d)]
for d in dirs:
    nb_files = [os.path.join(d,f) for f in os.listdir(d) if f.lower().endswith(".ipynb")]
    for nb_file in nb_files:
        print(nb_file)
        os.system("jupyter nbconvert --to markdown {0}".format(nb_file))
        # set new name
        html_file = os.path.basename(nb_file).replace('.ipynb', '.md')
        # set the path to md file
        md_dir = os.path.join(docs_dir, d.split("_")[0])
        ## remove existing file in docs folder
        #if os.path.exists(os.path.join(md_dir, html_file)):
        #    os.remove(os.path.join(md_dir, html_file))
        # move new file to docs folder
        shutil.copy(os.path.join(d, html_file), os.path.join(md_dir, html_file))
        os.remove(os.path.join(d, html_file))
        # move the figures to the docs folder
        figs_dir = html_file.replace(".md", "_files")
        # remove existing fig files
        if os.path.exists(os.path.join(d, figs_dir)):
            if os.path.exists(os.path.join(md_dir, figs_dir)):
                shutil.rmtree(os.path.join(md_dir, figs_dir)) 
            else:
                os.mkdir(os.path.join(md_dir, figs_dir))
            # move new fig files
            shutil.move(os.path.join(d, figs_dir), os.path.join(md_dir, figs_dir))
        #shutil.rmtree(os.path.join(d, figs_dir))
        print('preped htmlfile: ', os.path.join(md_dir, html_file))

os.chdir(docs_dir)
md_order = pd.read_csv("notebook_order.csv")
for folder in md_order.folder.unique():
    print(folder)
    md_order_select = md_order.loc[md_order.folder==folder]
    for md_file in md_order_select.file.unique():
        parent = md_order_select.loc[md_order_select.file==md_file, "parent"].values[0]
        nav_order = md_order_select.loc[md_order_select.file==md_file, "nav_order"].values[0]
        title = md_order_select.loc[md_order_select.file==md_file, "title"].values[0]
        md_file = os.path.join(folder, md_file)
        add_header(md_file, parent, nav_order, title)
os.chdir(cwd)