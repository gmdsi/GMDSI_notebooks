"""Run tutorial notebooks by part/section prefix.

Usage:
    python run_notebooks.py part0          # run all part0 notebooks
    python run_notebooks.py part2_01 part2_02  # run specific sections in order
    python run_notebooks.py part1          # run all part1 notebooks

Notebooks within each section are sorted and run sequentially.
Sections are run in the order given on the command line.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

TUTORIALS = Path(__file__).resolve().parent.parent / "tutorials"
TIMEOUT = 1800  # 30 minutes per notebook

# Sections and individual notebooks to skip during testing.
SKIP_SECTIONS = {
    "part2_07_da",
    "part2_09_mou",
}
SKIP_NOTEBOOKS = {
    "freyberg_ies_2_localization.ipynb",
    "freyberg_ies_3_restarting.ipynb",
}

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

# Part1 sections that must run before part1_10 (pilotpoints -> fosm dependency)
PART1_ORDER = [
    "part1_01", "part1_02", "part1_03", "part1_04", "part1_05",
    "part1_06", "part1_07", "part1_08", "part1_09", "part1_10",
    "part1_11", "part1_12",
]


def get_sections(prefix):
    """Find tutorial section directories matching a prefix."""
    sections = sorted(
        d for d in TUTORIALS.iterdir()
        if d.is_dir() and d.name.startswith(prefix)
        and d.name not in SKIP_SECTIONS
    )
    return sections


def get_notebooks(section_dir):
    """Get ordered list of notebooks for a section directory."""
    dirname = section_dir.name
    if dirname in SECTION_ORDER:
        nbs = [
            section_dir / nb for nb in SECTION_ORDER[dirname]
            if (section_dir / nb).exists() and nb not in SKIP_NOTEBOOKS
        ]
    else:
        nbs = sorted(
            nb for nb in section_dir.glob("*.ipynb")
            if nb.name not in SKIP_NOTEBOOKS
        )
    return nbs


def patch_ies_notebook(nb_path):
    """Patch IES notebooks to use fewer realizations and iterations for CI."""
    import json
    with open(nb_path, "r") as f:
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
                    r'\g<1>1', line
                )
            if line != orig:
                changed = True
            new_source.append(line)
        cell["source"] = new_source
    if changed:
        with open(nb_path, "w") as f:
            json.dump(nb, f, indent=1)
        print(f"  Patched IES settings in {nb_path.name}")
    return changed


def run_notebook(nb_path):
    """Execute a notebook in place and clear output. Returns True on success."""
    print(f"\n{'='*60}")
    print(f"Running: {nb_path.relative_to(TUTORIALS.parent)}")
    print(f"{'='*60}")
    t0 = time.time()

    # Patch IES notebooks for faster CI runs, keeping a backup to restore after
    patched = False
    backup = None
    if "part2_06_ies" in str(nb_path):
        backup = nb_path.read_bytes()
        patched = patch_ies_notebook(nb_path)

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
    )

    elapsed = time.time() - t0
    status = "PASS" if result.returncode == 0 else "FAIL"
    print(f"{status}: {nb_path.name} ({elapsed:.0f}s)")

    # Restore original notebook content if it was patched
    if backup is not None:
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

    prefixes = sys.argv[1:]
    failures = []
    total = 0

    for prefix in prefixes:
        # If a broad prefix like "part1", sort sections in defined order
        if prefix == "part1":
            all_sections = []
            for p in PART1_ORDER:
                all_sections.extend(get_sections(p))
        else:
            all_sections = get_sections(prefix)

        if not all_sections:
            print(f"WARNING: no sections found matching '{prefix}'")
            continue

        for section in all_sections:
            notebooks = get_notebooks(section)
            for nb in notebooks:
                total += 1
                if not run_notebook(nb):
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
