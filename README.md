# QuasiHubbard

QuasiHubbard constructs localized Wannier functions and Hubbard matrix elements for the eightfold optical quasicrystal using the method of E. Gottlob and U. Schneider, *Hubbard models for quasicrystalline potentials*, Phys. Rev. B **107**, 144202. Please cite that paper when publishing results produced with this code.

The calculation finds potential minima, constructs nonorthogonal Wannier functions in local eigenspaces, applies a symmetric Löwdin transformation, and evaluates onsite and offsite interaction integrals. Energies are in recoil units, $E_{rec}=\hbar^2 k^2/(2m)$, and lengths in optical wavelengths $\lambda$, with $k=2\pi/\lambda$:

$$V(\mathbf r)=V_0\sum_{i=1}^{4}\cos^2(\mathbf k_i\cdot\mathbf r+\phi_i).$$

## Installation

The reference was verified on CPU with Python **3.9.6** and the versions pinned in `requirements.txt`. The numerical Hamiltonian and eigensolver use PyTorch in float64; SciPy remains a dependency for optimization, integration, and saved sparse matrices.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

The two positional arguments remain **depth and diameter**. The circular site selection uses half the diameter as its radius. This small example retains six sites:

```sh
python quasi_hubbard.py 5 1.5 --spacing 0.2 --cutoff 1 --device cpu --output-dir .reference/example
```

This is a coarse regression case, not a converged physical prediction. The default spacing is `0.1`, the default cutoff is `4.0`, and execution defaults to one worker. `--device auto` selects CUDA when it is available and otherwise uses CPU. Use `--device cuda` on the intended NVIDIA machine; CUDA execution requires one worker. On CPU, use `--workers 2` for two processes or `--workers -1` for all available CPUs. Each worker holds a local Hamiltonian and eigensolver workspace, so increase concurrency with memory use in mind.

Apple MPS is not used because this calculation preserves float64 numerical precision, which the tested MPS backend does not support. The MacBook therefore runs the PyTorch code on CPU; the same code selects CUDA on the future H100.

The eigensolver allows up to 2,000 iterations per site and stops early on convergence. For harder cases, increase `--eigensolver-maxiter` (or `eigensolver_maxiter` in Python). The limit is recorded in checkpoint metadata; changing it requires a new output directory or `--fresh`. The eigenpair residual check still rejects unconverged results.

Use `--plot` to save `lattice_sites_V.png`. Plotting is skipped by default. The same calculation is callable from Python:

```python
from quasi_hubbard import run

output = run(5.0, 1.5, spacing=0.2, cutoff=1.0, workers=1, device="cpu",
             output_dir=".reference/example", plot=False)
```

`run` returns a `pathlib.Path`. Importing the driver does not start a calculation. The four wavevectors and historical laser phases are defined in `run` and recorded exactly in metadata.

## Checkpoints

Running the same command again resumes completed stages and preserves the saved site ordering. `metadata.json` records exact parameters, the calculation version, source commit, hashes of all six calculation modules, Python version and dependency versions. Parameter, source or dependency mismatches are refused. Worker count and plotting can change on resume.

Every checkpoint is written to a temporary file and atomically replaced. A missing file invalidates its stage and every later stage **before** upstream recomputation starts. This intentionally recomputes some independent stages to keep the protocol simple. Completed Wannier generation is one checkpoint; an interruption within that stage recomputes all sites in the stage.

Use a new output directory for a different calculation. `--fresh` explicitly discards the known calculation checkpoints in the selected directory and recomputes them. Directories from older versions without metadata are not automatically migrated. Use only one writer per output directory, and do not edit checkpoint arrays by hand.

## Outputs

Columns of the wavefunction arrays follow `lattice_sites.npy`. Each column is a local `(ny, nx)` array flattened in C order, with index `ix + nx * iy`. Reconstruct its actual coordinates with `cont_schrod.generate_grid(site, cutoff + 1.25, spacing)`; the saved `x_window.npy` and `y_window.npy` describe a window centered at the origin. Grid centers snap to the nearest global grid point. The historical even-sized grid and half-open upper boundary are preserved.

| File | Contents |
| --- | --- |
| `candidate_sites.npy`, `lattice_sites.npy` | Minima before and after configuration-space filtering |
| `octagon.npy`, `rings_list.npy`, `phis.npy` | Configuration coordinates, ring list, laser phases; the current ring filter returns no rings |
| `wannier_functions.npy` | Normalized nonorthogonal Wannier columns |
| `hamiltonian_wannier.npz`, `S_matrix.npy` | Hamiltonian and overlap in the nonorthogonal basis |
| `lowdin_transform.npy`, `overlap_eigenvalues.npy` | $A=S^{-1/2}$ and the overlap spectrum |
| `hamiltonian_real.npz` | Algebraically transformed Hamiltonian $A^T H A$ |
| `lowdin_basis_vec.npy` | Transformed functions cropped to their local windows and renormalized |
| `lowdin_retained_norms.npy`, `S_lowdin.npy` | Squared norms before crop renormalization, and overlap of the saved cropped functions |
| `hubbard_U.npy` | Onsite integrals $\int |w_i|^4\,d^2r$ |
| `U_iijj.npy` | Offsite density integrals $\int |w_i|^2|w_j|^2\,d^2r$; diagonal is zero |
| `U_iiij.npy` | Entry `[j, i]` is $\int w_i^3 w_j\,d^2r$ for real functions; its diagonal equals `hubbard_U` |
| `summary.json` | Stage timings, reused stages, grid size, overlap conditioning and Löwdin crop diagnostics |

Load `.npy` files with `numpy.load` and the sparse Hamiltonians with `scipy.sparse.load_npz`. Interaction integrals do **not** include the coupling constant or transverse confinement needed to obtain physical interaction energies.

## Numerical scope and convergence

The active solver constructs the sinc DVR Hamiltonian as a PyTorch sparse COO tensor and finds its lowest states with `torch.lobpcg`. It uses real float64 states and equal spacing in both directions. The grid-placement helper supports rectangular and complex arrays; this does not extend complex-state support to the full calculation. The old finite-difference and SciPy eigensolver paths have been removed.

Minima must have positive Hessians and a scaled force residual $\|\nabla V\|/(V_0\max_i\|\mathbf k_i\|)\le10^{-5}$. A nominally successful minimization can fail that residual check; one tighter refinement is attempted. Conversely, a precision-loss status is accepted when the point passes the residual and curvature checks. Local masks require three non-collinear minima. Eigenpair residuals are checked against `1e-8` after scaling by `max(1, abs(eigenvalue))`. Each site has a deterministic eigensolver starting vector.

The Löwdin transform rejects nonpositive or numerically singular overlaps instead of clipping their eigenvalues, and checks algebraic orthogonality. Cropping and renormalization remain an approximation: `hamiltonian_real.npz` describes the algebraic basis, whereas interaction integrals use the saved cropped functions. Inspect both `cropped_overlap_max_error` and `max_crop_norm_loss` in `summary.json`, and check spacing and cutoff convergence for the observables you need. Small crop errors alone do not establish convergence. See [CPU verification notes](VERIFICATION.md) for measured differences.

## Verification

```sh
# Small synthetic checks; no testing framework required.
MPLBACKEND=Agg python check.py

# Six-site physical regression, common-grid matrix checks, serial/parallel,
# Python/CLI equivalence, interrupted checkpoints and a shallow-lattice solve.
# CPU execution; usually tens of seconds.
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python check.py --physical

# Independently inspect a small saved calculation on the union of local grids.
MPLBACKEND=Agg python check.py --inspect .reference/example

# Reproduce the untouched historical calculation twice in new directories.
# Requires the original commit to be present in this checkout's Git history.
python check.py --capture-original .reference/original-a
python check.py --capture-original .reference/original-b
```

The original capture uses commit `5885c1d8c23c59ec5d55eb2f850da50ade28bb1d`, modifies only run controls, and instruments the driver to save intermediate arrays and metadata. It does not use the refactored calculation modules. Original outputs preserve historical defects and should not be used as corrected scientific expectations.

## Code layout

The six modules remain: `quasi_hubbard.py` for orchestration and checkpoints, `functions.py` for Wannier construction and matrix elements, `generate_lattice.py` for minima and configuration filtering, `potential_functions.py` for potentials and masks, `cont_schrod.py` for grids and the PyTorch solver, and `spread_minimisation.py` for spread minimization.

## Authors and contributions

Emmanuel Gottlob and Ulrich Schneider. Questions, bug reports and contributions are welcome through GitHub issues and pull requests. Licensed under the [MIT License](LICENSE).
