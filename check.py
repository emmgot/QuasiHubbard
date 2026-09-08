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
    from cont_schrod import generate_grid, shift_to_global_grid, hamiltonian
    from functions import dot_states
    from generate_lattice import clean_lattice_sites

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
    rectangular = np.arange(6).reshape(2, 3) + 1j
    shifted, error = shift_to_global_grid(rectangular, np.arange(3), np.arange(2),
                                         np.arange(-1, 4), np.arange(-1, 3))
    assert not error and shifted.shape == (4, 5)
    np.testing.assert_array_equal(shifted[1:3, 1:4], rectangular)
    shifted, error = shift_to_global_grid(rectangular, np.arange(3), np.arange(2),
                                         np.arange(1, 4), np.arange(2))
    np.testing.assert_array_equal(shifted[:, :2], rectangular[:, 1:])
    assert not error and not shifted[:, -1].any()
    shifted, error = shift_to_global_grid(rectangular, np.arange(3), np.arange(2),
                                         np.arange(10, 13), np.arange(2))
    assert error and not shifted.any()
    with np.testing.assert_raises(ValueError):
        shift_to_global_grid(rectangular, np.arange(3) + 0.1, np.arange(2), np.arange(3), np.arange(2))
    with np.testing.assert_raises(ValueError):
        generate_grid((0, 0), 1, 0)
    assert clean_lattice_sites([]).shape == (0, 2)
    np.testing.assert_array_equal(clean_lattice_sites([[0, 1]]), [[0, 1]])
    H = hamiltonian(np.arange(3) * 0.2, np.arange(2) * 0.2, np.zeros((2, 3)), "cpu")
    # Independent separable DVR construction checks signs, flattening and boundary couplings.
    def kinetic(n):
        delta = np.arange(n)[:, None] - np.arange(n)[None, :]
        result = 2 * (-1.) ** delta / (4 * np.pi ** 2 * 0.2 ** 2 * np.where(delta == 0, 1, delta) ** 2)
        np.fill_diagonal(result, 1 / (12 * 0.2 ** 2))
        return result
    expected = np.kron(np.eye(2), kinetic(3)) + np.kron(kinetic(2), np.eye(3))
    np.testing.assert_allclose(H.to_dense().numpy(), expected, atol=1e-14)


def check_masks():
    import potential_functions as potentials
    from potential_functions import potential_mask_contour

    X, Y = np.meshgrid(np.linspace(-1, 1, 7), np.linspace(-1, 1, 7))
    original = np.arange(49.).reshape(7, 7)
    masked, mask = potential_mask_contour(X, Y, original, (0, 0), 2.5, 0.3, 0.4,
                                          0.0, 5., np.zeros((4, 2)), np.zeros(4))
    assert mask.dtype == bool
    assert mask.any() and (~mask).any()
    np.testing.assert_array_equal(masked[mask], 20)
    np.testing.assert_array_equal(masked[~mask], original[~mask])

    # The narrower ellipse axis follows the stiffer direction, for any orientation.
    # A zero potential keeps the contour part from obscuring the ellipse test.
    for angle in (0., 0.37, np.pi / 2):
        stiff = np.array([np.cos(angle), np.sin(angle)])
        soft = np.array([-np.sin(angle), np.cos(angle)])
        hessian = 9 * np.outer(stiff, stiff) + np.outer(soft, soft)
        points = 0.25 * np.array([stiff, soft, -stiff, -soft])
        with patch.object(potentials, "hessian_mat", return_value=hessian):
            _, mask = potentials.potential_mask_site(points[:, 0], points[:, 1], np.zeros(4),
                                                     (0, 0), 5., np.zeros((4, 2)), np.zeros(4))
        np.testing.assert_array_equal(mask, [True, False, True, False])
    with np.testing.assert_raises(ValueError):
        potentials.potential_mask_hull(X, Y, original, np.array([[0, 0], [1, 0], [2, 0]]),
                                       1., 5., np.zeros((4, 2)), np.zeros(4))


def embed(basis, sites, spacing, window):
    """Independent common-grid placement for checks; do not use the production shift helper."""
    from cont_schrod import generate_grid
    grids = [generate_grid(site, window, spacing) for site in sites]
    x0 = min(x[0] for x, _ in grids)
    y0 = min(y[0] for _, y in grids)
    nx = round((max(x[-1] for x, _ in grids) - x0) / spacing) + 1
    ny = round((max(y[-1] for _, y in grids) - y0) / spacing) + 1
    full = np.zeros((ny, nx, len(sites)), dtype=basis.dtype)
    for i, (x, y) in enumerate(grids):
        ix, iy = round((x[0] - x0) / spacing), round((y[0] - y0) / spacing)
        full[iy:iy + len(y), ix:ix + len(x), i] = basis[:, i].reshape(len(y), len(x))
    return full.reshape(ny * nx, len(sites)), x0 + np.arange(nx) * spacing, y0 + np.arange(ny) * spacing


def check_lowdin():
    from functions import compute_lowdin, symmetric_orthogonalization
    sites, basis = np.array([[0., 0.], [1., 0.]]), np.full((16, 2), 0.5)
    full, _, _ = embed(basis, sites, 0.5, 1.)
    S = full.T @ full * 0.5 ** 2
    A, _ = symmetric_orthogonalization(S)
    np.testing.assert_allclose(A.T @ S @ A, np.eye(2), atol=1e-14)
    results = [compute_lowdin(i, basis, sites, A, 1., 0.5, return_norm=True) for i in range(2)]
    cropped, _, _ = embed(np.hstack([state for state, _ in results]), sites, 0.5, 1.)
    assert np.max(np.abs(cropped.T @ cropped * 0.5 ** 2 - np.eye(2))) > 0.01
    assert all(0 < norm < 1 for _, norm in results)
    for S in (np.zeros((2, 2)), np.diag([1., -1.]), np.diag([1., 1e-18])):
        with np.testing.assert_raises(ValueError):
            symmetric_orthogonalization(S)


def check_torch():
    from cont_schrod import hamiltonian, lowest_eigenstates, resolve_device

    x, y = np.arange(4) * 0.2, np.arange(3) * 0.2
    potential = np.arange(12).reshape(3, 4) / 10
    def kinetic(n):
        delta = np.arange(n)[:, None] - np.arange(n)[None, :]
        result = 2 * (-1.) ** delta / (4 * np.pi ** 2 * 0.2 ** 2
                                       * np.where(delta == 0, 1, delta) ** 2)
        np.fill_diagonal(result, 1 / (12 * 0.2 ** 2))
        return result

    expected = (np.kron(np.eye(3), kinetic(4)) + np.kron(kinetic(3), np.eye(4))
                + np.diag(potential.reshape(-1)))
    actual = hamiltonian(x, y, potential, "cpu").to_dense().numpy()
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
    values, vectors = lowest_eigenstates(x, y, potential, 2, seed=7, device="cpu")
    np.testing.assert_allclose(values, np.linalg.eigvalsh(expected)[:2], rtol=1e-10, atol=1e-10)
    residual = np.linalg.norm(expected @ vectors - vectors * values, axis=0) / np.maximum(1, np.abs(values))
    assert residual.max() < 1e-8
    assert resolve_device("cpu") == "cpu"
    with np.testing.assert_raises(ValueError):
        resolve_device("mps")


def inspect_physical(output):
    """Compare local representations with an independently assembled common grid."""
    from scipy.sparse import load_npz
    from cont_schrod import hamiltonian
    from potential_functions import potential
    output = Path(output)
    params = json.loads((output / "metadata.json").read_text())["parameters"]
    spacing, window = params["spacing"], params["cutoff"] + 1.25
    sites = np.load(output / "lattice_sites.npy")
    local_basis = np.load(output / "wannier_functions.npy")
    basis, x, y = embed(local_basis, sites, spacing, window)
    lowdin, _, _ = embed(np.load(output / "lowdin_basis_vec.npy"), sites, spacing, window)
    A = np.load(output / "lowdin_transform.npy")
    np.testing.assert_allclose(basis.T @ basis * spacing ** 2, np.load(output / "S_matrix.npy"), atol=1e-13)
    np.testing.assert_allclose(lowdin.T @ lowdin * spacing ** 2, np.load(output / "S_lowdin.npy"), atol=1e-13)
    full_lowdin = basis @ A
    support, _, _ = embed(np.ones_like(local_basis), sites, spacing, window)
    retained = np.sum(full_lowdin ** 2 * support, axis=0) * spacing ** 2
    np.testing.assert_allclose(retained, np.load(output / "lowdin_retained_norms.npy"), atol=1e-13)
    np.testing.assert_allclose(lowdin, full_lowdin * support / np.sqrt(retained), atol=1e-13)
    np.testing.assert_allclose(full_lowdin.T @ full_lowdin * spacing ** 2, np.eye(len(sites)), atol=1e-12)
    np.testing.assert_allclose(np.sum(lowdin ** 4, axis=0) * spacing ** 2, np.load(output / "hubbard_U.npy"), atol=1e-13)
    pair_density = (lowdin ** 2).T @ (lowdin ** 2) * spacing ** 2
    np.fill_diagonal(pair_density, 0)
    np.testing.assert_allclose(pair_density, np.load(output / "U_iijj.npy"), atol=1e-13)
    np.testing.assert_allclose(lowdin.T @ (lowdin ** 3) * spacing ** 2, np.load(output / "U_iiij.npy"), atol=1e-13)
    X, Y = np.meshgrid(x, y)
    H = hamiltonian(x, y, potential(X, Y, params["depth"], np.array(params["k"]),
                                    np.array(params["phis"])), "cpu")
    import torch
    H_basis = torch.sparse.mm(H, torch.as_tensor(basis, dtype=torch.float64)).numpy()
    H_lowdin = torch.sparse.mm(H, torch.as_tensor(lowdin, dtype=torch.float64)).numpy()
    common_H = basis.T @ H_basis * spacing ** 2
    actual_H = lowdin.T @ H_lowdin * spacing ** 2
    np.testing.assert_allclose(common_H, common_H.T, atol=1e-13)
    metrics = dict(common_grid_H_max_difference=float(np.max(np.abs(common_H - load_npz(output / "hamiltonian_wannier.npz").toarray()))),
                   cropped_H_max_difference=float(np.max(np.abs(actual_H - load_npz(output / "hamiltonian_real.npz").toarray()))),
                   crop_onsite_max_difference=float(np.max(np.abs(np.sum(full_lowdin ** 4 - lowdin ** 4, axis=0) * spacing ** 2))))
    print(json.dumps(metrics, indent=2))
    return metrics


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
        # These are regression values for this coarse case, not converged predictions.
        np.testing.assert_allclose(reference["hubbard_U.npy"],
            [12.841331765205, 9.115502225828, 14.806413907975, 7.949288951335, 7.821457280734, 10.748228003191],
            rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(np.linalg.eigvalsh(reference["hamiltonian_real.npz"]),
            [7.396249972872, 7.941659745782, 8.156846116361, 8.451433014269, 8.533540062060, 8.610748046460],
            rtol=1e-7, atol=1e-9)
        metrics = inspect_physical(output)
        assert metrics["common_grid_H_max_difference"] < 1e-5
        assert metrics["cropped_H_max_difference"] < 1e-5
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
        driver.run(**CASE, output_dir=legacy, fresh=True)
        compare(reference, arrays(legacy))
        metadata = json.loads((output / "metadata.json").read_text())
        with patch.object(driver, "CALCULATION_VERSION", driver.CALCULATION_VERSION + 1), np.testing.assert_raises(ValueError):
            driver.run(**CASE, output_dir=output)
        assert json.loads((output / "metadata.json").read_text()) == metadata

        # A failed atomic replacement leaves the previous complete checkpoint intact.
        with patch.object(Path, "replace", side_effect=InterruptedError), np.testing.assert_raises(InterruptedError):
            driver.save_atomic(output / "hubbard_U.npy", np.zeros(6))
        np.testing.assert_array_equal(np.load(output / "hubbard_U.npy"), reference["hubbard_U.npy"])
        assert not list(output.glob("*.tmp"))

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
    parser.add_argument("--inspect", metavar="OUTPUT_DIRECTORY", help="independently check a small physical run on a common grid")
    args = parser.parse_args()
    if args.capture_original:
        capture_original(args.capture_original)
    elif args.inspect:
        inspect_physical(args.inspect)
    else:
        check_imports_and_grid()
        check_masks()
        check_lowdin()
        check_torch()
        if args.physical:
            check_physical()
        print("Checks passed.")
