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

    laplace = lil_matrix((nx * ny, nx * ny))  # Initialize an empty sparse matrix
    n = nx * ny  # Total number of grid points

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

    nx = len(x_array)
    ny = len(y_array)
    dx = x_array[1] - x_array[0]
    dy = y_array[1] - y_array[0]

    H = lil_matrix((nx * ny, nx * ny))  # Initialize the Hamiltonian matrix
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

    closest_point = closest_grid_point(site, global_step)
    n_points = np.round(half_width / global_step)
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

    error_flag = False
    global_matrix = np.zeros((len(global_x), len(global_y)))
    global_step = np.diff(global_x)[0]

    try:
        # Calculate intersections
        left_intersection = np.round(np.max([local_x[0], global_x[0]]), 5)
        right_intersection = np.round(np.min([local_x[-1], global_x[-1]]), 5)
        down_intersection = np.round(np.max([local_y[0], global_y[0]]), 5)
        up_intersection = np.round(np.min([local_y[-1], global_y[-1]]), 5)

        # Find the indices in the local and global grids that correspond to the intersections
        left_local_index = np.argwhere(np.round(local_x, 5) == left_intersection)[0][0]
        right_local_index = np.argwhere(np.round(local_x, 5) == right_intersection)[0][0]
        down_local_index = np.argwhere(np.round(local_y, 5) == down_intersection)[0][0]
        up_local_index = np.argwhere(np.round(local_y, 5) == up_intersection)[0][0]

        left_global_index = np.argwhere(np.round(global_x, 5) == left_intersection)[0][0]
        right_global_index = np.argwhere(np.round(global_x, 5) == right_intersection)[0][0]
        down_global_index = np.argwhere(np.round(global_y, 5) == down_intersection)[0][0]
        up_global_index = np.argwhere(np.round(global_y, 5) == up_intersection)[0][0]

        # Update the global matrix with the local matrix data
        global_matrix[down_global_index:up_global_index + 1, left_global_index:right_global_index + 1] = \
            local_matrix[down_local_index:up_local_index + 1, left_local_index:right_local_index + 1]

    except IndexError:
        error_flag = True

    return global_matrix, error_flag

