import os
import multiprocessing as mp
import numpy as np
import pandas as pd
import pyemu

# function added thru PstFrom.add_py_function()
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
    dh = df.loc[:, [i for i in df.columns if i.startswith('TRGW_0_')]]
    dh = dh - df.loc[:, [i for i in df.columns if i.startswith('TRGW_2_')]].values
    dh.to_csv(os.path.join(ws,'heads.vdiff.csv'))

    print('Secondary observation files processed.')
    return 



def main():

    try:
       os.remove(r'heads.csv')
    except Exception as e:
       print(r'error removing tmp file:heads.csv')
    try:
       os.remove(r'sfr.csv')
    except Exception as e:
       print(r'error removing tmp file:sfr.csv')
    try:
       os.remove(r'sfr.tdiff.csv')
    except Exception as e:
       print(r'error removing tmp file:sfr.tdiff.csv')
    try:
       os.remove(r'heads.tdiff.csv')
    except Exception as e:
       print(r'error removing tmp file:heads.tdiff.csv')
    try:
       os.remove(r'heads.vdiff.csv')
    except Exception as e:
       print(r'error removing tmp file:heads.vdiff.csv')
    pyemu.helpers.apply_list_and_array_pars(arr_par_file='mult2model_info.csv',chunk_len=50)
    pyemu.os_utils.run(r'mf6')

    pyemu.os_utils.run(r'mp7 freyberg_mp.mpsim')

    process_secondary_obs(ws='.')

if __name__ == '__main__':
    mp.freeze_support()
    main()

