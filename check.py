"""Runnable numerical checks and a reproducible capture of the original CPU driver."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

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


def check_imports_and_grid():
    from cont_schrod import generate_grid, shift_to_global_grid
    from functions import dot_states

    for first in ("functions", "generate_lattice", "quasi_hubbard"):
        subprocess.run([sys.executable, "-c", f"import {first}; import functions; "
                        "assert callable(functions.generate_sites)"], check=True)
    x, y = generate_grid((0.23, -0.18), 0.5, 0.1)
    np.testing.assert_allclose(x, np.arange(-3, 7) * 0.1, atol=1e-15)
    np.testing.assert_allclose(y, np.arange(-7, 3) * 0.1, atol=1e-15)
    state = np.ones((10, 10))
    shifted, error = shift_to_global_grid(state, x, y, x, y)
    assert not error
    np.testing.assert_array_equal(shifted, state)
    assert np.isclose(dot_states(state, (0.23, -0.18), state, (0.23, -0.18), 0.1, 0.5) * 0.1 ** 2, 1)


def check_physical():
    from scipy.sparse import load_npz
    import quasi_hubbard as driver

    def arrays(directory):
        return {name: (load_npz(directory / name).toarray() if name.endswith(".npz")
                       else np.load(directory / name))
                for stage in driver.STAGES for name in stage}

    def compare(expected, actual):
        for name in expected:
            np.testing.assert_allclose(actual[name], expected[name], rtol=1e-8, atol=1e-10,
                                       err_msg=name)

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "serial"
        driver.run(**CASE, output_dir=output)
        reference = arrays(output)
        assert reference["lattice_sites.npy"].shape == (6, 2)
        mtimes = {name: (output / name).stat().st_mtime_ns for name in reference}
        driver.run(**CASE, output_dir=output)
        assert mtimes == {name: (output / name).stat().st_mtime_ns for name in reference}
        with np.testing.assert_raises(ValueError):
            driver.run(**dict(CASE, depth=5.00000001), output_dir=output)
        legacy = Path(directory) / "legacy"
        legacy.mkdir()
        np.save(legacy / "S_matrix.npy", np.eye(6))
        with np.testing.assert_raises(ValueError):
            driver.run(**CASE, output_dir=legacy)

        # The Hamiltonian checks its own file, even when the overlap still exists.
        (output / "hamiltonian_wannier.npz").unlink()
        driver.run(**CASE, output_dir=output)
        compare(reference, arrays(output))
        assert (output / "wannier_functions.npy").stat().st_mtime_ns == mtimes["wannier_functions.npy"]

        # Interrupt after new wavefunctions are saved: no old dependent result may survive.
        (output / "wannier_functions.npy").unlink()
        real_save = driver.save_atomic

        def interrupt(path, value):
            if Path(path).name == "hamiltonian_wannier.npz":
                raise InterruptedError("simulated interrupted calculation")
            real_save(path, value)

        with patch.object(driver, "save_atomic", interrupt), np.testing.assert_raises(InterruptedError):
            driver.run(**CASE, output_dir=output)
        assert (output / "wannier_functions.npy").exists()
        assert not any((output / name).exists() for stage in driver.STAGES[2:] for name in stage)
        driver.run(**CASE, output_dir=output)
        compare(reference, arrays(output))

        parallel = driver.run(**CASE, workers=2, output_dir=Path(directory) / "parallel")
        compare(reference, arrays(parallel))
        cli = Path(directory) / "cli"
        subprocess.run([sys.executable, "quasi_hubbard.py", "5", "1.5", "--spacing", "0.2",
                        "--cutoff", "1", "--output-dir", str(cli)], check=True)
        compare(reference, arrays(cli))
    print("Six-site CPU, serial/parallel, CLI and interrupted-restart checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-original", metavar="NEW_DIRECTORY")
    parser.add_argument("--physical", action="store_true", help="also run the six-site calculation and restart checks")
    args = parser.parse_args()
    if args.capture_original:
        capture_original(args.capture_original)
    else:
        check_imports_and_grid()
        if args.physical:
            check_physical()
        print("Checks passed.")
