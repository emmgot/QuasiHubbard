# Handles the Finite-Difference and Sinc discrete variable discretisation of the Schrodinger equation.
from math import pi
from typing import Tuple

import numpy as np
from scipy.sparse import lil_matrix, diags, csc_matrix


def index2_to_index1(i_x: int, j_y: int, nx: int) -> int:
    """
    Convert a 2D index (i_x, j_y) to a 1D index.

    Parameters:
    - i_x: The x-coordinate in the 2D grid.
    - j_y: The y-coordinate in the 2D grid.
    - nx: The number of grid points along the x-axis.

    Returns:
    - int: The 1D index corresponding to the given 2D index.
    """
    index = i_x + nx * j_y
    return index


def index1_to_index2(index1: int, nx: int) -> Tuple[int, int]:
    """
    Convert a 1D index to a 2D index (i_x, j_y).

    Parameters:
    - index1: The 1D index in the flattened array.
    - nx: The number of grid points along the x-axis in the 2D grid.

    Returns:
    - Tuple[int, int]: The 2D index (i_x, j_y) corresponding to the given 1D index.
    """
    j_y = np.floor_divide(index1, nx)
    i_x = np.remainder(index1, nx)
    return (i_x, j_y)


def laplacian(nx, ny, dx, dy):
    # Generate the discretized Laplacian Matrix using a 9 point stencil. dx must == dy.
    laplace = lil_matrix((nx * ny, nx * ny))
    n = nx * ny
    for j_y in range(ny):
        for i_x in range(nx):
            index = index2_to_index1(i_x, j_y, nx)
            indices = [(i_x + 1, j_y, nx), (i_x - 1, j_y, nx)]
            for ind in indices:
                if ind[0] < nx and ind[0] > 0 and ind[1] < ny and ind[1] > 0:
                    index2 = index2_to_index1(ind[0], ind[1], nx)
                    laplace[index, index2] = 0.5 / dx ** 2
            indices = [(i_x, j_y + 1, nx), (i_x, j_y - 1, nx)]
            for ind in indices:
                if ind[0] < nx and ind[0] > 0 and ind[1] < ny and ind[1] > 0:
                    index2 = index2_to_index1(ind[0], ind[1], nx)
                    laplace[index, index2] = 0.5 / dy ** 2
            indices = [(i_x + 1, j_y + 1, nx), (i_x - 1, j_y - 1, nx), (i_x - 1, j_y + 1, nx), (i_x + 1, j_y - 1, nx)]
            for ind in indices:
                if ind[0] < nx and ind[0] > 0 and ind[1] < ny and ind[1] > 0:
                    index2 = index2_to_index1(ind[0], ind[1], nx)
                    laplace[index, index2] = 0.25 / (dx * dy)
            laplace[index, index] = -3 / (dx * dy)
    return laplace


def laplacian_DVR(nx: int, ny: int, dx: float, dy: float) -> lil_matrix:
    """
    Generate the Laplacian matrix using the Sinc Discrete Variable Representation (DVR).

    Parameters:
    - nx (int): Number of grid points along the x-axis.
    - ny (int): Number of grid points along the y-axis.
    - dx (float): Grid spacing along the x-axis.
    - dy (float): Grid spacing along the y-axis.

    Returns:
    - laplace (lil_matrix): The discretized Laplacian matrix.

    The positions are indexed as follows: (x_i, y_j) --> m = i + j * nx
    """

    if not np.isfinite([dx, dy]).all() or dx <= 0 or not np.isclose(dx, dy, rtol=1e-9, atol=0):
        raise ValueError("The current DVR operator requires equal positive x/y spacing.")
    laplace = lil_matrix((nx * ny, nx * ny))

    for i_y in range(ny):
        for i_x in range(nx):
            index_i = index2_to_index1(i_x, i_y, nx)  # Convert 2D index to 1D
            laplace[index_i, index_i] = -np.pi ** 2 / 3 / (dx ** 2) * 2  # Diagonal element

            for j_x in range(nx):
                if j_x != i_x:
                    index_j = index2_to_index1(j_x, i_y, nx)
                    laplace[index_i, index_j] += -2 * (-1) ** (i_x - j_x) / (dx ** 2) / (
                            i_x - j_x) ** 2  # Off-diagonal elements (x direction)

            for j_y in range(ny):
                if j_y != i_y:
                    index_j = index2_to_index1(i_x, j_y, nx)
                    laplace[index_i, index_j] += -2 * (-1) ** (i_y - j_y) / (dy ** 2) / (
                            i_y - j_y) ** 2  # Off-diagonal elements (y direction)

    return laplace


def hamiltonian(x_array: np.ndarray, y_array: np.ndarray, V_mat: np.ndarray) -> (csc_matrix, csc_matrix, csc_matrix):
    """
    Generate the discretized Hamiltonian H = -1/(4pi^2) * p^2 + V(x, y),
    written in dimensionless units (E_rec=1, lambda=1).
    Note that dx must be equal to dy in this version.

    Parameters:
    - x_array (np.ndarray): 1D array of x positions.
    - y_array (np.ndarray): 1D array of y positions.
    - V_mat (np.ndarray): 2D array representing the potential energy V(x, y).

    Returns:
    - H (csc_matrix): The discretized Hamiltonian matrix.
    - laplace (csc_matrix): The discretized Laplacian matrix.
    - V (csc_matrix): The diagonal potential energy matrix.

    The positions are indexed as follows: (x_i, y_j) --> m = i + nx * j
    """

    x_array, y_array, V_mat = np.asarray(x_array), np.asarray(y_array), np.asarray(V_mat)
    if x_array.ndim != 1 or y_array.ndim != 1 or x_array.size < 2 or y_array.size < 2:
        raise ValueError("Hamiltonian grid axes need at least two points.")
    nx, ny = len(x_array), len(y_array)
    if V_mat.shape != (ny, nx) or not np.isfinite(V_mat).all() or np.iscomplexobj(V_mat):
        raise ValueError("Potential must be finite and real with shape (ny, nx).")
    dx = x_array[1] - x_array[0]
    dy = y_array[1] - y_array[0]

    if (not np.isfinite(x_array).all() or not np.isfinite(y_array).all()
            or dx <= 0 or dy <= 0
            or not np.allclose(np.diff(x_array), dx, rtol=1e-9, atol=0)
            or not np.allclose(np.diff(y_array), dy, rtol=1e-9, atol=0)):
        raise ValueError("Hamiltonian axes must have uniform positive spacing.")
    laplace = laplacian_DVR(nx, ny, dx, dy)  # Compute the Laplacian matrix using DVR

    V_diag = V_mat.flatten()
    V = diags(V_diag, offsets=0, shape=(nx * ny, nx * ny), format="csc", dtype=None)  # Diagonal potential energy matrix

    H = -1 / (4 * pi ** 2) * laplace + V  # Combine to form the Hamiltonian

    return H.tocsc(), laplace.tocsc(), V.tocsc()


def closest_grid_point(point: Tuple[float, float], global_step: float) -> Tuple[float, float]:
    """
    Finds the closest grid point to a given point based on the global step size.

    Parameters:
    - point (Tuple[float, float]): The x, y coordinates of the point to find the closest grid point for.
    - global_step (float): The step size for the global grid.

    Returns:
    - Tuple[float, float]: The x, y coordinates of the closest grid point.
    """
    x, y = point
    n_x = np.round(x / global_step)
    n_y = np.round(y / global_step)
    closest_x = n_x * global_step
    closest_y = n_y * global_step
    return (closest_x, closest_y)


def generate_grid(site: Tuple[float, float], half_width: float, global_step: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a grid synced with the global grid. The grid is centered around a given site and has an
    approximate half-width. The function returns x and y coordinates for the grid.

    Parameters:
    - site (Tuple[float, float]): The x, y coordinates of the center site.
    - half_width (float): The half-width of the grid.
    - global_step (float): The step size for the global grid.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: The x and y arrays defining the grid points.
    """

    if (np.shape(site) != (2,) or not np.isfinite(site).all()
            or not np.isfinite(half_width) or not np.isfinite(global_step)
            or half_width <= 0 or global_step <= 0):
        raise ValueError("Grid center must be finite; half-width and spacing must be finite and positive.")
    closest_point = closest_grid_point(site, global_step)
    n_points = np.round(half_width / global_step)
    if not np.isfinite(n_points) or n_points < 1 or n_points > np.sqrt(np.iinfo(np.intp).max / 8) / 2:
        raise ValueError("Grid size is empty or exceeds the addressable float64 array size.")
    actual_boundary = n_points * global_step

    grid_left = np.arange(n_points) * global_step - actual_boundary + closest_point[0]
    grid_right = np.arange(n_points) * global_step + closest_point[0]

    grid_down = np.arange(n_points) * global_step - actual_boundary + closest_point[1]
    grid_up = np.arange(n_points) * global_step + closest_point[1]

    x_array = np.concatenate([grid_left, grid_right])
    y_array = np.concatenate([grid_down, grid_up])

    return x_array, y_array


def shift_to_global_grid(local_matrix: np.ndarray, local_x: np.ndarray, local_y: np.ndarray,
                         global_x: np.ndarray, global_y: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Shifts a local matrix onto the global grid.

    Parameters:
    - local_matrix (np.ndarray): The local matrix to be shifted.
    - local_x (np.ndarray): The x-axis coordinates for the local grid.
    - local_y (np.ndarray): The y-axis coordinates for the local grid.
    - global_x (np.ndarray): The x-axis coordinates for the global grid.
    - global_y (np.ndarray): The y-axis coordinates for the global grid.

    Returns:
    - Tuple[np.ndarray, bool]: The shifted global matrix and a flag indicating if an error occurred.
    """

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
    if any(s.stop <= s.start for s in destination):
        return global_matrix, True  # Disjoint supports, not malformed coordinates.
    global_matrix[tuple(destination)] = local_matrix[tuple(source)]
    return global_matrix, False
