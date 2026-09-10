"""Run tutorial notebooks by part/section prefix.

Usage:
    python run_notebooks.py part0          # run all part0 notebooks
    python run_notebooks.py part2_01 part2_02  # run specific sections in order
    python run_notebooks.py part1          # run all part1 notebooks

    python run_notebooks.py --keep-output --all part1

Notebooks within each section are sorted and run sequentially.
Sections are run in the order given on the command line.

Two modes:

  CI (default)   Notebooks are patched down to a size that finishes on a
                 runner (fewer realizations, noptmax=3), executed, then
                 RESTORED and their output cleared. Nothing is left behind.

  --keep-output  No patching, no restore, no clearing: notebooks are executed
                 at their real settings and the results stay in the file.
                 This is what you want when publishing the rendered notebooks.
                 Add --all to also run the sections and notebooks that the CI
                 skip lists exclude.
"""
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

TUTORIALS = Path(__file__).resolve().parent.parent / "tutorials"
# Seconds per notebook. The heavier part2 sections (MOU under uncertainty in
# particular) can exceed 30 minutes on a laptop; override with NB_TIMEOUT.
TIMEOUT = int(os.environ.get("NB_TIMEOUT", 1800))

# Cap on PANTHER agents during CI. The notebooks ask for 10-20, sized for a
# workstation; a GitHub runner has 2-4 vCPUs. Oversubscribing them makes model
# runs fail sporadically, and PESTPP-OPT cannot tolerate even one lost run - its
# sequential LP needs a complete response matrix (freyberg_opt_2 has failed on
# windows-latest for exactly this reason).
#
# Windows is capped hardest: it is where the failures show up, with per-file
# locking and antivirus scanning every model output. Override with NB_WORKER_CAP;
# set it to 0 to disable capping entirely.
_DEFAULT_CAP = 4 if sys.platform.startswith("win") else 6
WORKER_CAP = int(os.environ.get("NB_WORKER_CAP", _DEFAULT_CAP))

# Sections and individual notebooks to skip during testing.
SKIP_SECTIONS = {
    "part2_07_da",
    "part2_09_mou",
}
SKIP_NOTEBOOKS = {
    # interactive widgets don't work in headless CI
    "understanding_variograms_and_realizations.ipynb",
    "intro_to_svd.ipynb",
    "simple_bayes_demo.ipynb",
    "intro_to_bayes.ipynb",
    "intro_to_regression.ipynb",
    # ies notebooks incompatible with reduced noptmax
    "freyberg_ies_2_localization.ipynb",
    "freyberg_ies_3_restarting.ipynb",
}

# Response-matrix / Jacobian builders (and their dependents) are the most expensive
# notebooks and run much slower on the macOS and Windows GitHub Actions runners, so we
# skip them there and rely on the Ubuntu job for coverage.  freyberg_fosm_and_dataworth
# is included because it loads the prior covariance from master_glm_1.
SKIP_ON_MAC_WIN = {
    "freyberg_glm_1.ipynb",
    "freyberg_glm_2.ipynb",
    "freyberg_glm_response_surface.ipynb",
    "freyberg_glm_response_surface_ies.ipynb",
    "freyberg_fosm_and_dataworth.ipynb",
}
# GitHub Actions sets RUNNER_OS; fall back to platform detection for local runs.
_RUNNER_OS = os.environ.get("RUNNER_OS", "")
IS_MAC_WIN = _RUNNER_OS in ("Windows", "macOS") or (
    not _RUNNER_OS and platform.system() in ("Darwin", "Windows")
)

# Ordering within sections where it matters.
# Keys are section directory prefixes; values are ordered notebook filenames.
# Sections not listed here have their notebooks sorted alphabetically.
SECTION_ORDER = {
    "part1_07_pilotpoints_setup": [
        "freyberg_pilotpoints_1_setup.ipynb",
    ],
    "part1_08_pilotpoints_run": [
        "freyberg_pilotpoints_2_run.ipynb",
    ],
    "part1_10_intro_to_fosm": [
        "intro_to_fosm.ipynb",
    ],
    "part2_01_pstfrom_pest_setup": [
        "freyberg_pstfrom_pest_setup.ipynb",
    ],
    "part2_02_obs_and_weights": [
        "freyberg_obs_and_weights.ipynb",
        "weights_vs_noise.ipynb",
    ],
    "part2_04_glm": [
        "freyberg_glm_1.ipynb",
        "freyberg_glm_2.ipynb",
    ],
    "part2_06_ies": [
        "freyberg_ies_1_basics.ipynb",
        "freyberg_ies_2_localization.ipynb",
        "freyberg_ies_3_restarting.ipynb",
        "freyberg_ies_4_noise.ipynb",
    ],
    "part2_07_da": [
        "freyberg_da_prep.ipynb",
        "freyberg_da_run.ipynb",
    ],
    "part2_08_opt": [
        "simple_LP_example.ipynb",
        "freyberg_opt_1.ipynb",
        "freyberg_opt_2.ipynb",
    ],
    "part2_09_mou": [
        "freyberg_mou_1.ipynb",
        "freyberg_mou_2.ipynb",
        "mou_viz.ipynb",
    ],
    "part2_10_eva_and_dsi": [
        "1_freyberg_ensemble_dataworth.ipynb",
        "2_freyberg_ensemble_data_space_inversion.ipynb",
    ],
}

# Part1 section run order. Order matters because later sections reuse results from
# earlier ones: part1_10 (fosm) needs the pilotpoints runs, and part1_14 (dsi) uses
# the prior ensemble that part1_13 (ies) leaves in its master directory.
PART1_ORDER = [
    "part1_01", "part1_02", "part1_03", "part1_04", "part1_05",
    "part1_06", "part1_07", "part1_08", "part1_09", "part1_10",
    "part1_11", "part1_12", "part1_13", "part1_14",
]


def get_sections(prefix, run_all=False):
    """Find tutorial section directories matching a prefix."""
    skip = set() if run_all else SKIP_SECTIONS
    sections = sorted(
        d for d in TUTORIALS.iterdir()
        if d.is_dir() and d.name.startswith(prefix)
        and d.name not in skip
    )
    return sections


def get_notebooks(section_dir, run_all=False):
    """Get ordered list of notebooks for a section directory."""
    dirname = section_dir.name
    skip = set() if run_all else SKIP_NOTEBOOKS
    if dirname in SECTION_ORDER:
        ordered = [nb for nb in SECTION_ORDER[dirname] if nb not in skip]
        nbs = [section_dir / nb for nb in ordered if (section_dir / nb).exists()]
        if run_all:
            # anything in the directory the explicit order does not mention
            named = set(ordered)
            nbs += sorted(nb for nb in section_dir.glob("*.ipynb")
                          if nb.name not in named)
    else:
        nbs = sorted(nb for nb in section_dir.glob("*.ipynb")
                     if nb.name not in skip)
    if IS_MAC_WIN and not run_all:
        nbs = [nb for nb in nbs if nb.name not in SKIP_ON_MAC_WIN]
    return nbs


def patch_ies_notebook(nb_path):
    """Patch IES notebooks to use fewer realizations and iterations for CI."""
    import json
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    changed = False
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        new_source = []
        for line in cell["source"]:
            orig = line
            if "ies_num_reals" in line and "=" in line and not line.lstrip().startswith("#"):
                # Replace any ies_num_reals assignment value with 20
                import re
                line = re.sub(
                    r'(ies_num_reals["\']?\s*[\])]?\s*=\s*)\d+',
                    r'\g<1>20', line
                )
            if "noptmax" in line and "=" in line and not line.lstrip().startswith("#"):
                # Replace positive noptmax values with 1, leave -1 and -2 alone
                import re
                line = re.sub(
                    r'(noptmax\s*=\s*)([2-9]\d*|[1-9]\d+)',
                    r'\g<1>3', line
                )
            if line != orig:
                changed = True
            new_source.append(line)
        cell["source"] = new_source
    if changed:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print(f"  Patched IES settings in {nb_path.name}")
    return changed


def patch_part1_ies_notebook(nb_path):
    """Patch the part1 basic-ies notebook to run cheaply in CI: cap ensemble size at
    20 realizations (both the gaussian-draw `num_reals=` and `ies_num_reals`) and cap
    positive noptmax at 2 iterations (leaving 0/1 and any negative values untouched).
    Returns True if the notebook was changed."""
    import json
    import re
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    def cap_noptmax(m):
        return m.group(1) + (m.group(2) if int(m.group(2)) <= 2 else "2")

    changed = False
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        new_source = []
        for line in cell["source"]:
            orig = line
            if line.lstrip().startswith("#"):
                new_source.append(line)
                continue
            # ies_num_reals = N  and  num_reals=N (gaussian draw) -> 20
            if "ies_num_reals" in line and "=" in line:
                line = re.sub(r'(ies_num_reals["\']?\s*[\])]?\s*=\s*)\d+',
                              r'\g<1>20', line)
            elif "num_reals" in line and "=" in line:
                line = re.sub(r'(num_reals\s*=\s*)\d+', r'\g<1>20', line)
            # positive noptmax -> capped at 2 (0/1 and negatives left alone)
            if "noptmax" in line and "=" in line:
                line = re.sub(r'(noptmax\s*=\s*)(\d+)', cap_noptmax, line)
            if line != orig:
                changed = True
            new_source.append(line)
        cell["source"] = new_source
    if changed:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print(f"  Patched part1 IES settings in {nb_path.name}")
    return changed


def patch_noptmax(nb_path, value):
    """Set positive noptmax assignments to `value` (leaving 0 and negative values,
    e.g. -1/-2 Jacobian-only runs, untouched). Used to shorten expensive GLM runs
    in CI. Returns True if the notebook was changed."""
    import json
    import re
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    changed = False
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        new_source = []
        for line in cell["source"]:
            orig = line
            if "noptmax" in line and "=" in line and not line.lstrip().startswith("#"):
                line = re.sub(r'(noptmax\s*=\s*)([1-9]\d*)',
                              r'\g<1>{0}'.format(value), line)
            if line != orig:
                changed = True
            new_source.append(line)
        cell["source"] = new_source
    if changed:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print(f"  Set noptmax={value} in {nb_path.name}")
    return changed


def uses_panther(nb):
    """Return True if any code cell in the loaded notebook invokes start_workers."""
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        for line in cell["source"]:
            if "start_workers" in line and not line.lstrip().startswith("#"):
                return True
    return False


def patch_overdue_giveup_fac(nb_path):
    """Inject overdue_giveup_fac=1e10 before each pst.write call in notebooks that
    use the panther parallel run manager. Avoids spurious agent timeouts on slow
    runners (e.g. GitHub Actions macOS)."""
    import json
    import re
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    if not uses_panther(nb):
        return False
    changed = False
    pattern = re.compile(r'^(\s*)pst\.write\(')
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        new_source = []
        for line in cell["source"]:
            stripped = line.lstrip()
            if not stripped.startswith("#"):
                m = pattern.match(line)
                if m:
                    indent = m.group(1)
                    new_source.append(
                        f'{indent}pst.pestpp_options["overdue_giveup_fac"] = 1e10\n'
                    )
                    changed = True
            new_source.append(line)
        cell["source"] = new_source
    if changed:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print(f"  Injected overdue_giveup_fac into {nb_path.name}")
    return changed


def cap_num_workers(nb_path, cap):
    """Cap the PANTHER agent count in a notebook.

    The notebooks ask for 10-20 agents, which is right on a workstation and far
    too many on a CI runner. A GitHub windows-latest box has 4 vCPUs, so 15
    agents is ~4 per core, all of them writing model output into their own
    directory while Defender scans it. Runs then fail sporadically - and
    PESTPP-OPT cannot absorb that the way PESTPP-IES can, because its sequential
    LP needs a complete response matrix, so one lost run aborts the notebook.

    Rewrites `num_workers = N` where N exceeds the cap. Expressions such as
    `num_workers = psutil.cpu_count(logical=False)` are left alone - they
    already scale to the machine.
    """
    import json
    import re

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    pattern = re.compile(r'^(\s*num_workers\s*=\s*)(\d+)(.*)$')
    changed = False
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        new_source = []
        for line in cell["source"]:
            m = pattern.match(line)
            if m and int(m.group(2)) > cap:
                new_source.append(f"{m.group(1)}{cap}{m.group(3)}\n".rstrip("\n") + "\n")
                changed = True
            else:
                new_source.append(line)
        cell["source"] = new_source
    if changed:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print(f"  Capped num_workers at {cap} in {nb_path.name}")
    return changed


def run_notebook(nb_path, keep_output=False):
    """Execute a notebook in place. Returns True on success.

    With keep_output the notebook is run at its real settings and the results
    are left in the file; otherwise it is patched for CI, then restored and
    cleared.
    """
    print(f"\n{'='*60}")
    print(f"Running: {nb_path.relative_to(TUTORIALS.parent)}")
    print(f"{'='*60}")
    t0 = time.time()

    # Patch notebooks for testing, keeping a backup to restore after.
    # IES-specific (reduced realizations/iterations) is path-scoped; the panther
    # overdue_giveup_fac injection is content-scoped via uses_panther().
    backup = None
    if not keep_output:
        # every one of these shrinks the notebook for CI, so none of them may
        # run when the point is to publish real results
        backup = nb_path.read_bytes()
        if "part2_06_ies" in str(nb_path):
            patch_ies_notebook(nb_path)
        if "part1_13" in str(nb_path):
            patch_part1_ies_notebook(nb_path)
        if nb_path.name == "freyberg_glm_2.ipynb":
            patch_noptmax(nb_path, 1)
        patch_overdue_giveup_fac(nb_path)
        if WORKER_CAP:
            cap_num_workers(nb_path, WORKER_CAP)

    env = os.environ.copy()
    if not keep_output:
        # The response-surface notebook sweeps a NUM_STEPS_RESPSURF**2 grid for each
        # of three surfaces - 4,800 forward runs at the teaching default of 40, which
        # is most of what put the ubuntu part1 job over its 2-hour limit. 12 steps is
        # 432 runs and still exercises the whole sweep/plot path. Setting the flag
        # `run_response_surfaces = False` instead is not an option in CI: the
        # *_respsurf directories are gitignored, so a fresh clone has nothing to plot
        # and the follow-on IES notebook asserts on their existence.
        env.setdefault("RESPSURF_STEPS", "12")
    if keep_output:
        # MPLBACKEND=Agg (set for CI, where output is thrown away anyway)
        # overrides ipykernel's inline backend, and every matplotlib figure is
        # then silently discarded instead of being embedded in the notebook.
        # These are plot-heavy teaching notebooks, so in keep-output mode the
        # figures ARE the result - never let that variable through.
        env.pop("MPLBACKEND", None)

    result = subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--execute",
            f"--ExecutePreprocessor.timeout={TIMEOUT}",
            "--inplace",
            str(nb_path),
        ],
        cwd=str(nb_path.parent),
        capture_output=False,
        env=env,
    )

    elapsed = time.time() - t0
    status = "PASS" if result.returncode == 0 else "FAIL"
    print(f"{status}: {nb_path.name} ({elapsed:.0f}s)")

    if not keep_output:
        # Restore original notebook content (reverts any patching above)
        nb_path.write_bytes(backup)

        # Clear output and metadata regardless of success
        subprocess.run(
            [
                sys.executable, "-m", "jupyter", "nbconvert",
                "--ClearOutputPreprocessor.enabled=True",
                "--ClearMetadataPreprocessor.enabled=True",
                "--inplace",
                str(nb_path),
            ],
            cwd=str(nb_path.parent),
            capture_output=True,
        )

    return result.returncode == 0


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_notebooks.py <prefix> [<prefix> ...]")
        print("  e.g.: python run_notebooks.py part0")
        print("  e.g.: python run_notebooks.py part2_01 part2_02")
        sys.exit(1)

    args = sys.argv[1:]
    keep_output = "--keep-output" in args
    run_all = "--all" in args
    prefixes = [a for a in args if not a.startswith("--")]
    if not prefixes:
        print("Usage: python run_notebooks.py [--keep-output] [--all] <prefix> ...")
        sys.exit(1)
    if keep_output:
        print("keep-output mode: real settings, results stay in the notebooks")
    if run_all:
        print("run-all mode: CI skip lists ignored")
    failures = []
    total = 0

    for prefix in prefixes:
        # If a broad prefix like "part1", sort sections in defined order
        if prefix == "part1":
            all_sections = []
            for p in PART1_ORDER:
                all_sections.extend(get_sections(p, run_all))
        else:
            all_sections = get_sections(prefix, run_all)

        if not all_sections:
            print(f"WARNING: no sections found matching '{prefix}'")
            continue

        for section in all_sections:
            notebooks = get_notebooks(section, run_all)
            for nb in notebooks:
                total += 1
                if not run_notebook(nb, keep_output=keep_output):
                    failures.append(str(nb.relative_to(TUTORIALS.parent)))

    print(f"\n{'='*60}")
    print(f"Results: {total - len(failures)}/{total} passed")
    if failures:
        print("Failures:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All notebooks passed!")


if __name__ == "__main__":
    main()
