"""Spatial grids and a sparse PyTorch sinc-DVR Hamiltonian."""
from math import pi
from typing import Tuple

import numpy as np
import torch


def resolve_device(device="auto"):
    """Resolve and validate the float64 CPU or CUDA calculation device."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available to PyTorch.")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("The solver supports float64 CPU or CUDA devices.")
    return str(device)


def _axis_laplacian(size, spacing, device):
    indices = torch.arange(size, device=device)
    delta = indices[:, None] - indices[None, :]
    parity = torch.where(delta.remainder(2) == 0, 1.0, -1.0)
    denominator = torch.where(delta == 0, 1, delta).to(torch.float64).square()
    values = -2 * parity / (spacing ** 2 * denominator)
    values.diagonal().fill_(-pi ** 2 / (3 * spacing ** 2))
    return values


def hamiltonian(x_array, y_array, potential, device="auto"):
    """Construct the float64 sinc-DVR Hamiltonian as a sparse COO tensor."""
    device = resolve_device(device)
    x_array, y_array, potential = map(np.asarray, (x_array, y_array, potential))
    if x_array.ndim != 1 or y_array.ndim != 1 or len(x_array) < 2 or len(y_array) < 2:
        raise ValueError("Hamiltonian grid axes need at least two points.")
    nx, ny = len(x_array), len(y_array)
    if potential.shape != (ny, nx) or not np.isfinite(potential).all() or np.iscomplexobj(potential):
        raise ValueError("Potential must be finite and real with shape (ny, nx).")
    dx, dy = x_array[1] - x_array[0], y_array[1] - y_array[0]
    if (not np.isfinite(x_array).all() or not np.isfinite(y_array).all()
            or dx <= 0 or dy <= 0
            or not np.allclose(np.diff(x_array), dx, rtol=1e-9, atol=0)
            or not np.allclose(np.diff(y_array), dy, rtol=1e-9, atol=0)
            or not np.isclose(dx, dy, rtol=1e-9, atol=0)):
        raise ValueError("The DVR requires matching uniform positive x/y spacing.")

    ix = torch.arange(nx, device=device)
    iy = torch.arange(ny, device=device)
    x_row = (iy[:, None, None] * nx + ix[None, :, None]).expand(ny, nx, nx).reshape(-1)
    x_col = (iy[:, None, None] * nx + ix[None, None, :]).expand(ny, nx, nx).reshape(-1)
    y_row = (iy[:, None, None] * nx + ix[None, None, :]).expand(ny, ny, nx).reshape(-1)
    y_col = (iy[None, :, None] * nx + ix[None, None, :]).expand(ny, ny, nx).reshape(-1)
    diagonal = torch.arange(nx * ny, device=device)

    scale = -1 / (4 * pi ** 2)
    x_values = (scale * _axis_laplacian(nx, dx, device)).repeat(ny, 1, 1).reshape(-1)
    y_values = (scale * _axis_laplacian(ny, dy, device))[:, :, None].expand(ny, ny, nx).reshape(-1)
    potential_values = torch.as_tensor(potential.reshape(-1), dtype=torch.float64, device=device)
    indices = torch.stack((torch.cat((x_row, y_row, diagonal)),
                           torch.cat((x_col, y_col, diagonal))))
    values = torch.cat((x_values, y_values, potential_values))
    return torch.sparse_coo_tensor(indices, values, (nx * ny, nx * ny),
                                   dtype=torch.float64, device=device).coalesce()


def lowest_eigenstates(x_array, y_array, potential, count, seed, device="auto"):
    """Return the lowest eigenpairs using sparse float64 LOBPCG."""
    matrix = hamiltonian(x_array, y_array, potential, device)
    if not 1 <= count <= matrix.shape[0] // 3:
        raise ValueError("LOBPCG requires at least three grid points per eigenpair.")
    generator = torch.Generator(device=matrix.device).manual_seed(seed)
    initial = torch.randn((matrix.shape[0], count), generator=generator,
                          dtype=torch.float64, device=matrix.device)
    values, vectors = torch.lobpcg(matrix, k=count, X=initial, niter=500,
                                   tol=1e-10, largest=False, method="ortho")
    residual = torch.linalg.vector_norm(
        torch.sparse.mm(matrix, vectors) - vectors * values, dim=0
    ) / torch.maximum(torch.ones_like(values), values.abs())
    if not torch.isfinite(residual).all() or residual.max().item() > 1e-8:
        raise ValueError(f"Eigensolver residual too large: {residual.max().item():.3g}.")
    return values.cpu().numpy(), vectors.cpu().numpy()


def apply_hamiltonian(x_array, y_array, potential, state, device="auto"):
    """Apply the sparse float64 DVR Hamiltonian to one state."""
    matrix = hamiltonian(x_array, y_array, potential, device)
    state = torch.as_tensor(state, dtype=torch.float64, device=matrix.device).reshape(-1, 1)
    return torch.sparse.mm(matrix, state).reshape(-1).cpu().numpy()


def closest_grid_point(point: Tuple[float, float], spacing: float) -> Tuple[float, float]:
    x, y = point
    return np.round(x / spacing) * spacing, np.round(y / spacing) * spacing


def generate_grid(site: Tuple[float, float], half_width: float, spacing: float) -> Tuple[np.ndarray, np.ndarray]:
    """Generate an even local grid aligned to the global grid."""
    if (np.shape(site) != (2,) or not np.isfinite(site).all()
            or not np.isfinite(half_width) or not np.isfinite(spacing)
            or half_width <= 0 or spacing <= 0):
        raise ValueError("Grid center must be finite; half-width and spacing must be finite and positive.")
    center = closest_grid_point(site, spacing)
    points = np.round(half_width / spacing)
    if not np.isfinite(points) or points < 1 or points > np.sqrt(np.iinfo(np.intp).max / 8) / 2:
        raise ValueError("Grid size is empty or exceeds the addressable float64 array size.")
    offsets = np.arange(-points, points) * spacing
    return offsets + center[0], offsets + center[1]


def shift_to_global_grid(local_matrix: np.ndarray, local_x: np.ndarray, local_y: np.ndarray,
                         global_x: np.ndarray, global_y: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Place a local matrix on an aligned target grid."""
    local_matrix = np.asarray(local_matrix)
    if local_matrix.shape != (len(local_y), len(local_x)):
        raise ValueError("local_matrix must have shape (len(local_y), len(local_x)).")
    global_matrix = np.zeros((len(global_y), len(global_x)), dtype=local_matrix.dtype)
    source, destination = [], []
    for local, target in ((local_y, global_y), (local_x, global_x)):
        local, target = np.asarray(local), np.asarray(target)
        if len(local) < 2 or len(target) < 2:
            raise ValueError("Grid axes need at least two points.")
        step = target[1] - target[0]
        if (not np.isfinite(local).all() or not np.isfinite(target).all() or step <= 0
                or not np.allclose(np.diff(local), step, rtol=1e-9, atol=0)
                or not np.allclose(np.diff(target), step, rtol=1e-9, atol=0)):
            raise ValueError("Local and target grids must have matching uniform positive spacing.")
        offset = (local[0] - target[0]) / step
        if not np.isclose(offset, np.rint(offset), rtol=0, atol=1e-7):
            raise ValueError("Local and target grids are not aligned.")
        offset = int(np.rint(offset))
        start, end = max(0, offset), min(len(target), offset + len(local))
        source.append(slice(start - offset, end - offset))
        destination.append(slice(start, end))
    if any(section.stop <= section.start for section in destination):
        return global_matrix, True
    global_matrix[tuple(destination)] = local_matrix[tuple(source)]
    return global_matrix, False
