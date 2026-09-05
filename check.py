"""Runnable numerical checks and a reproducible capture of the original CPU driver."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ORIGINAL_COMMIT = "5885c1d8c23c59ec5d55eb2f850da50ade28bb1d"
CASE = dict(depth=5.0, diameter=1.5, spacing=0.2, cutoff=1.0)


def capture_original(output):
    """Run the pinned driver with smaller grids and one worker; leave its math intact."""
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Use a new directory for each original reference run.")
    repo = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        for filename in ("functions.py", "generate_lattice.py", "potential_functions.py",
                         "cont_schrod.py", "spread_minimisation.py", "quasi_hubbard.py"):
            source = subprocess.check_output(
                ["git", "show", f"{ORIGINAL_COMMIT}:{filename}"], cwd=repo, text=True)
            (directory / filename).write_text(source)
        driver = (directory / "quasi_hubbard.py").read_text()
        driver = driver.replace("['cut_off'] = 4.0", f"['cut_off'] = {CASE['cutoff']}")
        driver = driver.replace("['global_step'] = 0.1", f"['global_step'] = {CASE['spacing']}")
        driver = driver.replace("n_jobs=-1", "n_jobs=1")
        driver = driver.replace(
            'dirname = "size_%.2f/fine/results_depth_%.4f" % (lattice_params[\'length\'], lattice_params["depth"])',
            f"dirname = {str(output)!r}")
        # Observe stages without changing the original numerical calls.
        driver = driver.replace("runtic = time.perf_counter()",
                                "runtic = time.perf_counter(); stage_times = {}; previous = runtic")
        for marker, stage in (("sites loaded: ", "sites"),
                              ("Wannier functions generated", "wannier"),
                              ("H wannier generated", "hamiltonian"),
                              ("S matrix generated", "overlap"),
                              ("Hamiltonian saved", "lowdin_transform"),
                              ("U_iijj computed", "lowdin_and_interactions"),
                              ("U_iiij computed", "density_assisted")):
            lines = driver.splitlines()
            for i, line in enumerate(lines):
                if "print(" in line and marker in line:
                    indent = line[:len(line) - len(line.lstrip())]
                    lines[i] += (f"\n{indent}now = time.perf_counter(); stage_times[{stage!r}] = "
                                 "now - previous; previous = now")
            driver = "\n".join(lines) + "\n"
        driver += '''
    import json, platform, importlib.metadata
    np.save(dirname + '/wannier_functions.npy', wannier_functions_vec)
    np.save(dirname + '/lowdin_basis_vec.npy', wannier_functions)
    np.save(dirname + '/lowdin_transform.npy', symm_orthog)
    metadata = dict(parameters=lattice_params, stage_seconds=stage_times,
                    source_commit="''' + ORIGINAL_COMMIT + '''", python=platform.python_version(),
                    versions={m: importlib.metadata.version(m) for m in
                              ('numpy', 'scipy', 'matplotlib', 'joblib')})
    Path(dirname + '/reference.json').write_text(json.dumps(metadata,
        default=lambda value: value.tolist(), indent=2) + '\\n')
'''
        (directory / "quasi_hubbard.py").write_text(driver)
        env = dict(os.environ, MPLBACKEND="Agg", MPLCONFIGDIR=str(directory / "mpl"),
                   OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        subprocess.run([sys.executable, str(directory / "quasi_hubbard.py"),
                        str(CASE["depth"]), str(CASE["diameter"])],
                       cwd=directory, env=env, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-original", metavar="NEW_DIRECTORY")
    args = parser.parse_args()
    if args.capture_original:
        capture_original(args.capture_original)
