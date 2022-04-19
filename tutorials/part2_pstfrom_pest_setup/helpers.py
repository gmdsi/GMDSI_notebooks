import os
import numpy as np


def process_secondary_obs(ws='.'):
    # load dependencies insde the function so that they get carried over to forward_run.py by PstFrom
    import os 
    import pandas as pd

    def write_tdif_obs(orgf, newf, ws='.'):
        df = pd.read_csv(os.path.join(ws,orgf), index_col='time')
        df = df - df.iloc[0, :]
        df.to_csv(os.path.join(ws,newf))
        return
    
    # write the tdiff observation csv's
    write_tdif_obs('heads.csv', 'heads.tdiff.csv', ws)
    write_tdif_obs('sfr.csv', 'sfr.tdiff.csv', ws)

    #write the vdiff obs csv
    # this is frought with the potential for bugs, but oh well...
    df = pd.read_csv(os.path.join(ws,'heads.csv'), index_col='time')
    df.sort_index(axis=1, inplace=True)
    dh = df.loc[:, [i for i in df.columns if i.startswith('TRGW-0-')]]
    dh = dh - df.loc[:, [i for i in df.columns if i.startswith('TRGW-2-')]].values
    dh.to_csv(os.path.join(ws,'heads.vdiff.csv'))

    print('Secondary observation files processed.')
    return 


def extract_hds_arrays_and_list_dfs():
    import flopy
    hds = flopy.utils.HeadFile("freyberg6_freyberg.hds")
    arr = hds.get_data()
    for i,a in enumerate(arr):
        np.savetxt("hdslay{0}.txt".format(i+1),a,fmt="%15.6E")
    lst = flopy.utils.Mf6ListBudget("freyberg6.lst")
    inc,cum = lst.get_dataframes(diff=True,start_datetime=None)
    inc.columns = inc.columns.map(lambda x: x.lower().replace("_","-"))
    cum.columns = cum.columns.map(lambda x: x.lower().replace("_", "-"))
    inc.index.name = "totim"
    cum.index.name = "totim"
    inc.to_csv("inc.csv")
    cum.to_csv("cum.csv")
    

def test_extract_hds_arrays(d):
    cwd = os.getcwd()
    os.chdir(d)
    extract_hds_arrays_and_list_dfs()
    os.chdir(cwd)


if __name__ == "__main__":
	#process_secondary_obs()
    test_extract_hds_arrays("freyberg6_template")