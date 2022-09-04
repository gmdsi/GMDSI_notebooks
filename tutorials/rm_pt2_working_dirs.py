import os
import shutil
dirs = [d for d in os.listdir(".") if os.path.isdir(d)]
for d in dirs:
    dirs = [os.path.join(d,dd) for dd in os.listdir(d) if os.path.isdir(os.path.join(d,dd)) and ("master" in dd or "template" in dd or "worker" in dd)]
    for dd in dirs:
        shutil.rmtree(dd)

