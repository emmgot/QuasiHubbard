"""Construct Wannier functions and Hubbard matrix elements for the eightfold lattice."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import tempfile
import time

import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import csc_matrix, hstack, issparse, load_npz, save_npz

from cont_schrod import generate_grid
from functions import (generate_wannier_function, compute_H, compute_S, compute_lowdin,
                       compute_U_iijj, compute_U_iiij)
from generate_lattice import generate_sites, generate_octagon, clean_rings
from potential_functions import potential

# Bump when numerical conventions or checkpoint contents change.
CALCULATION_VERSION = 1
# A missing stage invalidates every later stage, before any computation starts.
# ponytail: one ordered sequence; some independent stages are recomputed after a gap.
STAGES = (
    ("candidate_sites.npy", "lattice_sites.npy", "rings_list.npy", "octagon.npy"),
    ("wannier_functions.npy",),
    ("hamiltonian_wannier.npz",),
    ("S_matrix.npy",),
    ("lowdin_transform.npy", "hamiltonian_real.npz"),
    ("lowdin_basis_vec.npy",),
    ("hubbard_U.npy",),
    ("U_iijj.npy",),
    ("U_iiij.npy",),
)


def save_atomic(path, value):
    """Replace a complete file in its own directory; incomplete writes are never loaded."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as tmp:
        temporary = Path(tmp.name)
        try:
            if path.suffix == ".json":
                tmp.write((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())
            elif issparse(value):
                save_npz(tmp, value)
            else:
                np.save(tmp, value, allow_pickle=False)
            tmp.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def invalidate_incomplete(output, fresh=False):
    first_missing = next((i for i, stage in enumerate(STAGES)
                          if fresh or not all((output / name).is_file() for name in stage)), len(STAGES))
    if first_missing < len(STAGES):
        # Delete downstream first: even an interruption during deletion leaves a safe prefix.
        for stage in reversed(STAGES[first_missing:]):
            for name in reversed(stage):
                (output / name).unlink(missing_ok=True)
        (output / "summary.json").unlink(missing_ok=True)
        (output / "lattice_sites_V.png").unlink(missing_ok=True)


def plot_sites(output, params, candidates, sites):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    x, y = params["x_global"], params["y_global"]
    X, Y = np.meshgrid(x, y)
    depth = params["depth"]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"aspect": "equal"})
    ax.pcolormesh(x, y, potential(X, Y, depth, params["k"], params["phis"]),
                  cmap="jet", norm=TwoSlopeNorm(vmin=0, vcenter=2 * depth, vmax=4 * depth), alpha=0.6)
    ax.scatter(*candidates.T, s=5, c="k", alpha=0.6)
    ax.scatter(*sites.T, s=6, c="r", alpha=0.6)
    ax.set(xlim=(candidates[:, 0].min() - 1, candidates[:, 0].max() + 1),
           ylim=(candidates[:, 1].min() - 1, candidates[:, 1].max() + 1),
           xlabel=r"$x/\lambda$", ylabel=r"$y/\lambda$")
    fig.savefig(output / "lattice_sites_V.png", dpi=300)
    plt.close(fig)


def run(depth, diameter, *, spacing=0.1, cutoff=4.0, workers=1, output_dir=None,
        plot=False, fresh=False):
    """Run or resume a real, equal-spacing CPU calculation; return its output directory.

    Lengths are in optical wavelengths and depth is in recoil energies. Only one
    process should write to a given output directory at a time.
    """
    values = dict(depth=depth, diameter=diameter, spacing=spacing, cutoff=cutoff)
    for name, value in values.items():
        if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if not isinstance(workers, int) or workers == 0 or workers < -1:
        raise ValueError("workers must be a positive integer, or -1 for all CPUs.")
    x_window, y_window = generate_grid((0, 0), cutoff + 1.25, spacing)
    if len(x_window) < 2:
        raise ValueError("spacing is too large for the local window.")

    k0 = 2 * np.pi
    kx, ky = np.array([1, 0]) * k0, np.array([0, 1]) * k0
    k = np.array([kx, ky, 1 / np.sqrt(2) * (kx + ky), 1 / np.sqrt(2) * (kx - ky)])
    delta_x, delta_y = 75.0, 121 + np.sqrt(5)
    phis = np.array([-2 * np.pi * delta_x, -2 * np.pi * delta_y,
                     -2 * np.pi * 1 / np.sqrt(2) * (delta_x + delta_y),
                     -2 * np.pi * 1 / np.sqrt(2) * (delta_x - delta_y)])
    parameters = dict(values, k=k.tolist(), phis=phis.tolist())
    versions = {name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "matplotlib", "joblib")}
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                           cwd=Path(__file__).resolve().parent,
                                           stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    metadata = dict(calculation_version=CALCULATION_VERSION, parameters=parameters,
                    source_commit=revision, python=platform.python_version(), versions=versions)
    output = Path(output_dir) if output_dir is not None else Path(f"size_{diameter:.2f}/fine/results_depth_{depth:.4f}")
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "metadata.json"
    if not fresh and metadata_path.exists():
        saved = json.loads(metadata_path.read_text())
        if (saved.get("calculation_version") != CALCULATION_VERSION
                or saved.get("parameters") != parameters
                or saved.get("versions") != versions):
            raise ValueError("Incompatible checkpoint parameters, calculation version or dependencies; "
                             "choose a new output directory or explicitly use fresh=True / --fresh.")
    elif not fresh and any(output.iterdir()):
        raise ValueError("Existing directory has no metadata; choose a new directory or use --fresh.")
    invalidate_incomplete(output, fresh=fresh)
    if fresh or not metadata_path.exists():
        save_atomic(metadata_path, metadata)

    x_global, y_global = generate_grid((0, 0), diameter / 2 + cutoff + 0.5, spacing)
    params = dict(length=diameter, depth=depth, global_step=spacing, cut_off=cutoff,
                  k=k, phis=phis, x_global=x_global, y_global=y_global, n_x=len(x_global))
    for name, value in (("phis", phis), ("x_window", x_window), ("y_window", y_window)):
        save_atomic(output / f"{name}.npy", value)
    times, reused = {}, []
    started = previous = time.perf_counter()

    def finished(stage, was_reused):
        nonlocal previous
        now = time.perf_counter()
        times[stage] = now - previous
        previous = now
        if was_reused:
            reused.append(stage)
        print(f"{stage}: {'loaded' if was_reused else 'computed'} ({times[stage]:.3f}s)", flush=True)

    cached = (output / "lattice_sites.npy").exists()
    if cached:
        candidates, sites, rings, octagon = [np.load(output / name, allow_pickle=False) for name in STAGES[0]]
    else:
        candidates = generate_sites((0, 0), params, mask_radius=diameter / 2)
        sites, rings = clean_rings(candidates, generate_octagon(candidates, phis))
        if len(sites) == 0:
            raise ValueError("No retained sites; increase the system diameter.")
        octagon = generate_octagon(sites, phis)
        for name, value in zip(STAGES[0], (candidates, sites, rings, octagon)):
            save_atomic(output / name, value)
    params.update(lattice_sites=sites, rings_list=rings)
    n_sites = len(sites)
    print(f"{n_sites} retained sites; local grid {len(x_window)} x {len(y_window)}", flush=True)
    finished("sites", cached)

    cached = (output / "wannier_functions.npy").exists()
    if cached:
        basis = np.load(output / "wannier_functions.npy", allow_pickle=False)
    else:
        basis = np.hstack(Parallel(n_jobs=workers)(
            delayed(generate_wannier_function)(i, params) for i in range(n_sites)))
        save_atomic(output / "wannier_functions.npy", basis)
    if basis.shape != (len(x_window) * len(y_window), n_sites) or not np.isfinite(basis).all():
        raise ValueError("Invalid Wannier checkpoint shape or values; use a fresh run.")
    finished("wannier", cached)

    args = (basis, sites, cutoff, cutoff + 1.25, spacing)
    cached = (output / "hamiltonian_wannier.npz").exists()
    if cached:
        H = load_npz(output / "hamiltonian_wannier.npz")
    else:
        half = hstack(Parallel(n_jobs=workers)(delayed(compute_H)(i, *args, params) for i in range(n_sites)), format="csc")
        H = half + half.T
        save_atomic(output / "hamiltonian_wannier.npz", H)
    finished("hamiltonian", cached)

    cached = (output / "S_matrix.npy").exists()
    if cached:
        S = np.load(output / "S_matrix.npy", allow_pickle=False)
    else:
        half = hstack(Parallel(n_jobs=workers)(delayed(compute_S)(i, *args) for i in range(n_sites)), format="csc")
        S = (half + half.T).toarray()
        save_atomic(output / "S_matrix.npy", S)
    finished("overlap", cached)

    cached = (output / "lowdin_transform.npy").exists()
    if cached:
        transform = np.load(output / "lowdin_transform.npy", allow_pickle=False)
    else:
        eigenvalues, vectors = np.linalg.eigh(S)
        transform = (vectors * (1 / np.sqrt(eigenvalues))) @ vectors.T
        save_atomic(output / "lowdin_transform.npy", transform)
        save_atomic(output / "hamiltonian_real.npz", csc_matrix(transform @ (H @ transform.T)))
    finished("lowdin_transform", cached)

    cached = (output / "lowdin_basis_vec.npy").exists()
    if cached:
        lowdin = np.load(output / "lowdin_basis_vec.npy", allow_pickle=False)
    else:
        lowdin = np.hstack(Parallel(n_jobs=workers)(
            delayed(compute_lowdin)(i, basis, sites, transform, cutoff + 1.25, spacing) for i in range(n_sites)))
        save_atomic(output / "lowdin_basis_vec.npy", lowdin)
    finished("lowdin_basis", cached)

    cached = (output / "hubbard_U.npy").exists()
    if not cached:
        save_atomic(output / "hubbard_U.npy", np.sum(np.abs(lowdin) ** 4, axis=0) * spacing ** 2)
    finished("onsite", cached)
    for name, function in (("U_iijj", compute_U_iijj), ("U_iiij", compute_U_iiij)):
        cached = (output / f"{name}.npy").exists()
        if not cached:
            result = hstack(Parallel(n_jobs=workers)(
                delayed(function)(i, lowdin, sites, cutoff, cutoff + 1.25, spacing) for i in range(n_sites)), format="csc")
            if name == "U_iijj":
                result = result + result.T
            save_atomic(output / f"{name}.npy", result.toarray())
        finished(name, cached)
    save_atomic(output / "summary.json", dict(sites=n_sites, local_grid=list((len(x_window), len(y_window))),
                stage_seconds=times, total_seconds=time.perf_counter() - started, reused_stages=reused, workers=workers))
    if plot:
        plot_sites(output, params, candidates, sites)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("depth", type=float, help="lattice depth in recoil energies")
    parser.add_argument("diameter", type=float, help="system diameter in optical wavelengths")
    parser.add_argument("--spacing", type=float, default=0.1)
    parser.add_argument("--cutoff", type=float, default=4.0)
    parser.add_argument("--workers", type=int, default=1, help="worker count; -1 uses all CPUs")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plot", action="store_true", help="save a lattice potential plot")
    parser.add_argument("--fresh", action="store_true", help="discard this directory's calculation checkpoints")
    try:
        print(run(**vars(parser.parse_args())))
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
