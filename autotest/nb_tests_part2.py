import os
import time
cwd = os.getcwd()
print(cwd)
tutdir=os.path.join(cwd, "..","tutorials")
if os.path.basename(os.path.normpath(cwd))!='autotest':
    tutdir=os.path.join(cwd, ".","tutorials")

timings = []

def run_nb(nb_file, nb_dir):
    assert nb_dir
    assert os.path.join(nb_dir, nb_file)
    os.chdir(nb_dir)
    t0 = time.time()
    os.system(f"jupyter nbconvert --execute --ExecutePreprocessor.timeout=1800 --inplace {nb_file}")
    elapsed = time.time() - t0
    timings.append((nb_file, elapsed))
    os.system(f"jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --inplace {nb_file}")
    os.chdir(cwd)
    print(f'ran: {nb_file} ({elapsed/60:.1f} min)')
    return

nb_dir=os.path.join(tutdir,'part2_01_pstfrom_pest_setup')
run_nb("freyberg_pstfrom_pest_setup.ipynb", nb_dir)

nb_dir=os.path.join(tutdir,'part2_02_obs_and_weights')
run_nb("freyberg_obs_and_weights.ipynb", nb_dir)

print("\n" + "="*60)
print("TIMING SUMMARY (sorted by duration)")
print("="*60)
for nb, t in sorted(timings, key=lambda x: x[1], reverse=True):
    print(f"  {t/60:6.1f} min  {nb}")
print(f"\n  TOTAL: {sum(t for _,t in timings)/60:.1f} min")
