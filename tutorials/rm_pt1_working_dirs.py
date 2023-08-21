import os
import shutil
dirs = [d for d in os.listdir(".") if os.path.isdir(d)]
for d in dirs:
    dirs = [os.path.join(d,dd) for dd in os.listdir(d) if os.path.isdir(os.path.join(d,dd)) and ("master" in dd or "template" in dd or "worker" in dd)]
    for dd in dirs:
        if "part1_" not in dd:
            continue
        try:
            shutil.rmtree(dd,ignore_errors=True)
        except Exception as e:
            print("error removing '{0}':{1}".format(dd,str(e)))

