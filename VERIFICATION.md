# CPU and PyTorch verification — 2026-09-08

Reference environment: Python 3.9.6, PyTorch 2.8.0, NumPy 2.0.2, SciPy 1.13.1, Matplotlib 3.9.4 and Joblib 1.5.3 on macOS arm64. The original source revision is `5885c1d8c23c59ec5d55eb2f850da50ade28bb1d`.

## Historical reference and structural changes

The small physical case uses depth 5, diameter 1.5, spacing 0.2 and cutoff 1, with the historical wavevectors and phases. It retains **six sites**, each with a 22 × 22 local wavefunction grid. Two independent executions of the original driver produced identical saved arrays in this environment. The capture includes site ordering, nonorthogonal wavefunctions, S, H, the Löwdin transformation, cropped wavefunctions, interaction integrals and parameter/version/timing metadata.

After removing the circular imports, moving grid helpers, extracting `run`, and replacing the checkpoint workflow, the largest absolute difference across those numerical outputs was **4.44e-15**. Site coordinates, nonorthogonal functions, S and the nonorthogonal Hamiltonian were identical. These are comparisons to historical behavior, not a validation of that behavior's physical accuracy.

Raw captures and subsequent runs are retained locally under `.reference/`, excluded from Git. `check.py --capture-original NEW_DIRECTORY` reproduces the historical capture from Git history. The corrected case's onsite integrals and Hamiltonian eigenvalues are recorded directly in the runnable physical check.

## Separately identified numerical changes

- **Boolean contour indexing:** replaced integer indexing with Boolean negation. The helper's returned potential is corrected; the six-site main-pipeline outputs did not change, consistent with the hull routine discarding that intermediate potential.
- **Hessian orientation:** paired symmetric eigenvectors now determine the ellipse axes, with the narrow axis aligned to the largest curvature. Axis-aligned and rotated examples pass. For the coarse case this changed the largest Hamiltonian element by **3.70e-4 E_rec**, the largest onsite-integral difference by **8.27e-3**, and the largest `U_iiij` difference by **2.38e-2**. These differences are recorded rather than hidden by updating a historical baseline.
- **Stationarity:** the cutoff-2 study found a nominally successful Newton-CG result with scaled force **3.41e-5**. One tighter refinement now handles such returns; points still failing the force/curvature checks are rejected. Precision-loss statuses with adequate stationarity are accepted. This prevents treating the optimizer's status alone as proof of a minimum.
- **Grid placement:** rectangular shapes and complex dtype are preserved; aligned grids are placed by integer offsets. Malformed/misaligned grids raise errors; disjoint supports still yield zero overlaps. The active solver remains real with equal x/y spacing.
- **Solver and Löwdin guards:** site/eigenspace dimensions, local geometry, eigenpair residuals, overlap positivity and numerical conditioning are checked. Site-specific starting vectors make execution order independent of the eigensolver seed. No overlap eigenvalues are clipped.

## Löwdin cropping and common-grid checks

The check embeds all local functions independently on the union of their grids. It verifies S and the actual final overlap, measures pre-renormalization retained norms, and evaluates interactions directly on that common grid. The uncropped algebraic basis is `W @ A`; comparing it to the cropped basis holds the original eigenspaces fixed.

| Spacing | Cutoff | Local grid | Maximum final overlap error | Maximum crop norm loss | Common-grid H vs saved nonorthogonal H, max absolute difference |
| --- | --- | --- | --- | --- | --- |
| 0.2 | 1 | 22 × 22 | 8.48e-9 | 1.48e-10 | 4.02e-6 E_rec |
| 0.1 | 1 | 44 × 44 | 1.81e-11 | 6.03e-13 | 1.32e-8 E_rec |
| 0.05 | 1 | 90 × 90 | 8.53e-16 | 1.55e-15 | 2.15e-13 E_rec |
| 0.1 | 2 | 64 × 64 | 3.84e-12 | 1.63e-13 | 3.10e-9 E_rec |

The algebraic overlap error stays around 1e-15. For the coarse case, the directly evaluated Hamiltonian of the saved cropped functions differs from `hamiltonian_real.npz` by at most **4.00e-6 E_rec**. Removing the Löwdin crop changes its onsite integrals by at most **2.70e-9**. Cropping is consequently retained and monitored for these cases; this does not establish its adequacy at other depths or larger diameters.

The Hamiltonian comparison is independent of the driver's explicit symmetrization: it constructs the DVR operator on the common grid and evaluates the full matrix. A separate synthetic check compares the rectangular DVR operator against a separable construction.

## Convergence limits

At cutoff 1, refining spacing from 0.1 to 0.05 changes onsite integrals by as much as **0.546**, and Hamiltonian entries by **0.00726 E_rec**. At spacing 0.1, increasing cutoff from 1 to 2 changes onsite integrals by up to **0.0392**, offsite density integrals by **0.00409**, and Hamiltonian entries by **0.000770 E_rec**. Site ordering and coordinates agree across these runs.

These are sensitivity measurements, not a completed production convergence study. In particular, small Löwdin crop errors do not imply converged interaction integrals. A larger representative system and the intended physical accuracy still need to determine production settings.

To reproduce the sensitivity runs, use new output directories:

```sh
python quasi_hubbard.py 5 1.5 --spacing 0.1 --cutoff 1 --output-dir .reference/h01
python quasi_hubbard.py 5 1.5 --spacing 0.05 --cutoff 1 --output-dir .reference/h005
python quasi_hubbard.py 5 1.5 --spacing 0.1 --cutoff 2 --output-dir .reference/c2
python check.py --inspect .reference/h01
python check.py --inspect .reference/h005
python check.py --inspect .reference/c2
```

## Restart and execution checks

`python check.py --physical` verifies the corrected six-site observables, direct common-grid overlaps/interactions, serial vs two-worker execution, Python vs CLI execution, unchanged checkpoint modification times on complete resume, exact-parameter and calculation-version rejection, explicit fresh handling of legacy directories, a missing Hamiltonian despite an existing S, and interruption after newly saved wavefunctions. A failed atomic replacement preserves the previous complete array. The plot path was also executed and visually checked.

## PyTorch solver replacement

The SciPy DVR assembly and `eigsh` path was replaced directly by a float64 PyTorch sparse COO Hamiltonian and `torch.lobpcg`; there is no backend abstraction or duplicate implementation. An independently constructed 12 by 12 DVR matrix verifies signs, flattening, potential placement and boundary couplings. Its two lowest eigenvalues and scaled residuals are checked directly.

On the six-site case with a 44 × 44 local grid, PyTorch CPU took approximately 1.97 s for Wannier construction and 0.05 s for Hamiltonian matrix elements. The prior SciPy profile took approximately 3.12 s and 2.92 s for those stages. The largest wavefunction difference was 1.97e-9, the largest real-Hamiltonian element difference was 7.03e-11, and its largest eigenvalue difference was 1.43e-11. The runnable coarse 22 × 22 regression retains the established onsite-integral and Hamiltonian-spectrum tolerances and passes serial, parallel, CLI, resume and interrupted-write checks.

The current MacBook runs this path on CPU. Apple MPS is excluded because the calculation uses float64. `--device auto` will select CUDA when the future NVIDIA H100 is available; CUDA execution is restricted to one worker to avoid duplicating GPU state. Actual CUDA correctness, memory use and end-to-end speed still need measurement on that hardware. Local minima searches still use a fine-grid extent depending on the full diameter, and repeated Hamiltonian construction remains a possible optimization target.

The iteration limit is now configurable with `--eigensolver-maxiter` (Python: `eigensolver_maxiter`) and defaults to 2,000. At depth 0.1, diameter 1.5, spacing 0.1 and cutoff 1, the former 500-iteration limit fails the residual check; the new default completes all six sites without relaxing the tolerance. `python check.py --physical` covers both outcomes, along with checkpoint rejection when the limit changes. The full suite passed on CPU after this change.

Physical serial/parallel and CLI regressions explicitly select CPU. The parallel check also simulates CUDA availability to guard against accidental automatic device selection. The small independent eigenpair check additionally runs on CUDA when available; actual CUDA execution has not been verified on this MacBook.
