import os
import numpy as np
import pandas as pd
import pyemu
import matplotlib.pyplot as plt
import freyberg_setup as frey_mod

WORKING_DIR = frey_mod.WORKING_DIR_KR
MODEL_NAM = "freyberg.nam"
PST_NAME = frey_mod.PST_NAME_KR
NUM_SLAVES = 15
NUM_STEPS_RESPSURF = 10

def run_respsurf(par_names=None):
    pst = pyemu.Pst(os.path.join(WORKING_DIR,PST_NAME))
    par = pst.parameter_data
    icount = 0
    if par_names is None:
        parnme1 = par.parnme[0]
        parnme2 = par.parnme[1]
    else:
        parnme1 = par_names[0]
        parnme2 = par_names[1]
    p1 = np.linspace(par.loc[parnme1,"parlbnd"],par.loc[parnme1,"parubnd"],NUM_STEPS_RESPSURF).tolist()
    p2 = np.linspace(par.loc[parnme2,"parlbnd"],par.loc[parnme2,"parubnd"],NUM_STEPS_RESPSURF).tolist()
    p1_vals,p2_vals = [],[]
    for p in p1:
        p1_vals.extend(list(np.zeros(NUM_STEPS_RESPSURF)+p))
        p2_vals.extend(p2)
    df = pd.DataFrame({parnme1:p1_vals,parnme2:p2_vals})
    df.to_csv(os.path.join(WORKING_DIR,"sweep_in.csv"))

    os.chdir(WORKING_DIR)
    pyemu.helpers.start_slaves('.', 'pestpp-swp', PST_NAME, num_slaves=NUM_SLAVES, master_dir='.')
    os.chdir("..")

def plot_response_surface():
    df_in = pd.read_csv(os.path.join(WORKING_DIR, "sweep_in.csv"))
    df_out = pd.read_csv(os.path.join(WORKING_DIR, "sweep_out.csv"))
    resp_surf = np.zeros((NUM_STEPS_RESPSURF, NUM_STEPS_RESPSURF))
    hk_values = df_in.hk1.unique()
    rch_values = df_in.rch_0.unique()
    c = 0
    for i, v1 in enumerate(hk_values):
        for j, v2 in enumerate(rch_values):
            resp_surf[j, i] = df_out.loc[c, "phi"]
            c += 1
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    X, Y = np.meshgrid(hk_values, rch_values)
    #resp_surf = np.ma.masked_where(resp_surf > 5, resp_surf)
    p = ax.pcolor(X, Y, resp_surf, alpha=0.5, cmap="nipy_spectral")
    plt.colorbar(p)
    c = ax.contour(X, Y, resp_surf, levels=[0.1, 0.2, 0.5, 1, 2, 5], colors='k')
    plt.clabel(c)
    ax.set_xlim(hk_values.min(), hk_values.max())
    ax.set_ylim(rch_values.min(), rch_values.max())
    ax.set_xlabel("hk1 ($\\frac{L}{T}$)")
    ax.set_ylabel("rch ($L$)")
    return resp_surf

def rerun_new_pars(hk=5.5, rch_0 = 1.0):
    pst = pyemu.Pst(os.path.join(WORKING_DIR,PST_NAME))
    pst.control_data.noptmax = 0
    pars = pst.parameter_data
    pars.loc[pars.parnme == 'hk', 'parval1'] = hk
    pars.loc[pars.parnme == 'rch_0', 'parval1'] = rch_0
    pars.loc['hk','parlbnd'] = hk / 2
    pars.loc['hk', 'parubnd'] = hk * 2
    pars.loc['rch_0','parlbnd'] = rch_0 / 2
    pars.loc['rch_0','parubnd'] = rch_0 * 2

    obs = pst.observation_data
    obs.loc[obs.obgnme=='calhead', 'weight'] = 1.0
    obs.loc[obs.obgnme=='calflux', 'weight'] = 0.0

    pst.write(os.path.join(WORKING_DIR,'onerun.pst'))

    if os.path.exists(os.path.join(WORKING_DIR,'onerun.rei')):
        os.remove(os.path.join(WORKING_DIR,'onerun.rei'))
    os.chdir(WORKING_DIR)
    pyemu.os_utils.run('pestpp onerun.pst')
    os.chdir('..')

    if not os.path.exists(os.path.join(WORKING_DIR, 'onerun.rei')):
        print('Hey! your model blew up. Rein back in those parameters :-)!')
        return

    newpst = pyemu.Pst(os.path.join(WORKING_DIR,'onerun.pst'))
    res = newpst.res
    print('The root mean squared error is: {:.2f}'.format(np.sqrt(newpst.phi/pst.nnz_obs)))
    fig = plt.figure(figsize=(12,12))
    ax1 = fig.add_subplot(221)
    cal = res.loc[res.group == 'calhead']
    plt.plot(cal.measured, cal.modelled, '.')
    minmin = np.min([cal.measured.min(),cal.modelled.min()])
    maxmax = np.max([cal.measured.max(), cal.modelled.max()])
    plt.plot([minmin, maxmax],[minmin,maxmax], 'r')
    plt.xlabel('measured')
    plt.ylabel('modeled')
    plt.title('Calibration Head')
    ax1.set_aspect('equal')

    ax2 = fig.add_subplot(222)
    fore = res.loc[res.apply(lambda x: x.name.startswith("c00") and x.group=="forecast",axis=1)]
    fore.loc[:,["measured","modelled"]].plot(kind="bar",ax=ax2)
    ax2.set_title("head forecasts")
    #plt.plot(fore.measured, fore.modelled, '.')
    #minmin = np.min([fore.measured.min(), fore.modelled.min()])
    #maxmax = np.max([fore.measured.max(), fore.modelled.max()])
    #plt.plot([minmin, maxmax],[minmin,maxmax], 'r')
    
    #plt.xlabel('measured')
    #plt.ylabel('modeled')
    #plt.title('Forecast Head')
    #ax2.set_aspect('equal')
    fore_trav = res.loc[res.name == 'travel_time']
    fore_flux = res.loc[res.name == 'fa_headwaters_0001']
    
    ax3 = fig.add_subplot(223)
    res.loc[res.name == 'travel_time',['measured', 'modelled']].plot(kind='bar', ax=ax3, rot=45)
    ax3.set_title('travel time forecasts')

    #res.loc[((res.name == 'fa_headwaters_0001') | 
    ax4 = fig.add_subplot(224)
    res.loc[res.name == 'fa_headwaters_0001',['measured', 'modelled']].plot(kind='bar', ax=ax4, rot=45)
    ax4.set_title('sw-gw flux forecasts')





    plt.tight_layout()

    plt.show()


def run_ies():
    pass


if __name__ == "__main__":
    #setup_model()
    #setup_pest()
    #run_pe()
    run_fosm()
    run_dataworth()
