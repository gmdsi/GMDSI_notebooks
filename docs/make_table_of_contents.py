import os
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


md_order = pd.read_csv("notebook_order.csv")

for folder in md_order.folder.unique():
    print(folder)
    md_order_select = md_order.loc[md_order.folder==folder]
    for md_file in md_order_select.file.unique():
        parent = md_order_select.loc[md_order_select.file==md_file, "parent"].values[0]
        nav_order = md_order_select.loc[md_order_select.file==md_file, "nav_order"].values[0]
        title = md_order_select.loc[md_order_select.file==md_file, "title"].values[0]
        md_file = os.path.join(folder, md_file)
        print(title, nav_order, parent)
        add_header(md_file, parent, nav_order, title)
    

