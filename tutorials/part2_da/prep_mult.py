import os
import pandas as pd

cycle = int(float(open("cycle.dat",'r').readline().split()[-1]))

df = pd.read_csv("mult2model_info.global.csv",index_col=0)
df.loc[:,"cycle"] = df.cycle.astype(int)
df = df.loc[df.cycle.apply(lambda x: x == -1 or x == cycle),:]
df.to_csv(os.path.join("mult2model_info.csv"))
