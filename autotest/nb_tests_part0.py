import os
import subprocess
import sys
import time
cwd = os.getcwd()
print(cwd)
tutdir=os.path.join(cwd, "..","tutorials")
if os.path.basename(os.path.normpath(cwd))!='autotest':
    tutdir=os.path.join(cwd, ".","tutorials")

timings = []
failures = []

def run_nb(nb_file, nb_dir):
    if not os.path.isdir(nb_dir):
        raise FileNotFoundError(f"Notebook directory does not exist: {nb_dir}")
    nb_path = os.path.join(nb_dir, nb_file)
    if not os.path.isfile(nb_path):
        raise FileNotFoundError(f"Notebook file does not exist: {nb_path}")
    os.chdir(nb_dir)
    try:
        t0 = time.time()
        result = subprocess.run(
            ["jupyter", "nbconvert", "--execute",
             "--ExecutePreprocessor.timeout=1800", "--inplace", nb_file],
            check=False
        )
        elapsed = time.time() - t0
        timings.append((nb_file, elapsed))
        if result.returncode != 0:
            failures.append(nb_file)
            print(f'FAILED: {nb_file} ({elapsed/60:.1f} min)')
        else:
            print(f'ran: {nb_file} ({elapsed/60:.1f} min)')
        subprocess.run(
            ["jupyter", "nbconvert",
             "--ClearOutputPreprocessor.enabled=True",
             "--ClearMetadataPreprocessor.enabled=True",
             "--inplace", nb_file],
            check=False
        )
    finally:
        os.chdir(cwd)
    return

nb_dir=os.path.join(tutdir,'part0_00_intro_to_regression')
run_nb("intro_to_regression.ipynb", nb_dir)

nb_dir=os.path.join(tutdir,'part0_01_intro_to_bayes')
run_nb("intro_to_bayes.ipynb", nb_dir)

nb_dir=os.path.join(tutdir,'part0_01_intro_to_bayes')
run_nb("simple_bayes_demo.ipynb", nb_dir)

nb_dir=os.path.join(tutdir,'part0_02_intro_to_freyberg_model')
run_nb("intro_freyberg_model.ipynb", nb_dir)

nb_dir=os.path.join(tutdir,'part0_intro_to_eva_and_dsi')
run_nb("intro_to_eva_and_dsi.ipynb", nb_dir)

nb_dir=os.path.join(tutdir,'part0_intro_to_geostatistics')
run_nb("intro_to_geostatistics.ipynb", nb_dir)

nb_dir=os.path.join(tutdir,'part0_intro_to_geostatistics')
run_nb("understanding_variograms_and_realizations.ipynb", nb_dir)

nb_dir=os.path.join(tutdir,'part0_intro_to_pyemu')
run_nb("intro_to_pyemu.ipynb", nb_dir)

nb_dir=os.path.join(tutdir,'part0_intro_to_svd')
run_nb("intro_to_svd.ipynb", nb_dir)

print("\n" + "="*60)
print("TIMING SUMMARY (sorted by duration)")
print("="*60)
for nb, t in sorted(timings, key=lambda x: x[1], reverse=True):
    print(f"  {t/60:6.1f} min  {nb}")
print(f"\n  TOTAL: {sum(t for _,t in timings)/60:.1f} min")

if failures:
    print("\n" + "="*60)
    print(f"FAILURES ({len(failures)}):")
    for f in failures:
        print(f"  - {f}")
    print("="*60)
    sys.exit(1)
