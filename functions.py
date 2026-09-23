# This file contains miscellaneous functions about the generation of the TB-FDS Hamiltonian
# Imports

import numpy as np
from typing import Dict, List, Tuple, Union

import matplotlib.colors as colors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import integrate
from scipy.sparse import csc_matrix

from cont_schrod import (apply_hamiltonian, closest_grid_point, generate_grid,
                         lowest_eigenstates, shift_to_global_grid)
from generate_lattice import generate_sites, generate_octagon, clean_rings
from potential_functions import potential, potential_mask_hull
from spread_minimisation import min_spread


def dot_product(psi1, psi2, x_array, y_array):
    """
       Compute the dot product of two wave functions, psi1 and psi2, over a 2D grid.

       Args:
       psi1 (np.ndarray): First wave function array.
       psi2 (np.ndarray): Second wave function array.
       x_array (np.ndarray): Array of x coordinates.
       y_array (np.ndarray): Array of y coordinates.

       Returns:
       float: The dot product <psi1|psi2>.
       """

    # Multiply the complex conjugate of psi1 with psi2 element-wise
    integrand = np.multiply(np.conj(psi1), psi2)

    # Perform numerical integration along the x-axis
    res = integrate.trapz(integrand, x_array, axis=0)

    # Perform numerical integration along the y-axis
    res = integrate.trapz(res, y_array)

    return res


def is_neighbour(pos1: Union[List[float], np.ndarray], pos2: Union[List[float], np.ndarray], cut_off: float) -> bool:
    """
    Determines if two sites are neighbors based on a distance cutoff.

    Args:
    pos1 (Union[List[float], np.ndarray]): The position of the first site.
    pos2 (Union[List[float], np.ndarray]): The position of the second site.
    cut_off (float): The distance cutoff for defining neighbor sites.

    Returns:
    bool: True if the distance between pos1 and pos2 is less than cut_off, otherwise False.
    """

    # Calculate Euclidean distance between pos1 and pos2
    distance = np.linalg.norm(np.subtract(pos1, pos2))

    # Check if the distance is less than the cutoff
    if distance < cut_off:
        return True
    else:
        return False


def find_neighbours(i_site: int, site: Union[List[float], np.ndarray], cut_off: float,
                    sites: Union[List[List[float]], np.ndarray], x_sites: np.ndarray, y_sites: np.ndarray) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Finds the neighbouring sites of a given site within a cut-off distance.

    Parameters:
    - i_site (int): Index of the site in question.
    - site (Union[List[float], np.ndarray]): Coordinates of the site as a list or NumPy array.
    - cut_off (float): The cut-off distance for considering a site as a neighbour.
    - sites (Union[List[List[float]], np.ndarray]): List or array of all sites' coordinates.
    - x_sites (np.ndarray): x-coordinates of all sites.
    - y_sites (np.ndarray): y-coordinates of all sites.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: A tuple containing the list of neighbour sites and their indices.
    """

    x_distance = x_sites - site[0]
    y_distance = y_sites - site[1]
    radius_square = np.square(x_distance) + np.square(y_distance)
    neighbour_indices = np.where(radius_square <= cut_off ** 2)[0]
    neighbour_list = np.array(sites)[neighbour_indices]

    return neighbour_list, neighbour_indices


def generate_wannier_function(index_site: int, lattice_params: Dict[str, Union[np.ndarray, float]],
                              state_number: int = 1, plot: bool = False) -> np.ndarray:
    """
    Generates a Wannier function for a given site index and lattice parameters.

    Parameters:
    - index_site (int): Index of the site for which to generate the Wannier function.
    - lattice_params (dict): Dictionary containing lattice parameters.
    - state_number (int, optional): Number of states to include. Defaults to 1.
    - plot (bool, optional): Whether to plot the function. Defaults to False.

    Returns:
    - np.ndarray: Array containing the Wannier function.
    """
    minima = lattice_params['lattice_sites']
    rings_list = lattice_params['rings_list']
    cut_off = lattice_params['cut_off']
    if not 0 <= index_site < len(minima):
        raise ValueError("Wannier site index is outside the retained site list.")
    site = minima[index_site, :]

    depth = lattice_params['depth']
    global_step = lattice_params['global_step']
    k = lattice_params['k']
    phis = lattice_params['phis']
    window_radius = cut_off + 1.25
    x_window, y_window = generate_grid(site, window_radius, global_step)
    x_mean, y_mean = site
    neighbour_minima = generate_sites(site, lattice_params,
                                      mask_radius=cut_off)
    neighbour_octagon = generate_octagon(neighbour_minima, phis)
    neighbour_minima, neighbour_rings = clean_rings(neighbour_minima, neighbour_octagon)
    neighbour_rings = count_neighbour_rings(site, cut_off, rings_list)
    n_states = len(neighbour_minima)
    if not 1 <= state_number <= n_states < len(x_window) * len(y_window):
        raise ValueError(f"Need 1 <= state_number <= neighboring states ({n_states}) < grid dimension; "
                         "increase cutoff or refine spacing.")
    Xmesh, Ymesh = np.meshgrid(x_window, y_window)
    n_x_local = len(x_window)
    n_y_local = len(y_window)
    n_x_window = n_x_local
    dx = np.diff(x_window)[0]
    dy = np.diff(y_window)[0]
    V_window = potential(Xmesh, Ymesh, depth, k, phis)
    V_mask, _ = potential_mask_hull(Xmesh, Ymesh, V_window, neighbour_minima, cut_off,
                                    depth, k, phis, rings_list=neighbour_rings)
    val, vec = lowest_eigenstates(x_window, y_window, V_mask, n_states, index_site,
                                  lattice_params['device'], maxiter=lattice_params['eigensolver_maxiter'])
    norm = np.sum(np.abs(vec) ** 2, axis=0) * dx * dy
    normalized_states = vec / np.sqrt(norm)

    result_vec = np.zeros((n_x_local * n_y_local, state_number))
    for i_state in range(state_number):
        complex_coefs = np.squeeze(min_spread(normalized_states, x_mean, y_mean, x_window, y_window, i_state=i_state))

        wannier_function = complex_coefs * normalized_states
        wannier_function = np.sum(wannier_function, axis=1)
        global_phase = np.sign(wannier_function[np.argmax(
            np.abs(wannier_function))])
        wannier_function = (wannier_function * global_phase)
        norm = np.sum(np.square(np.abs(wannier_function))) * dx * dy
        wannier_function = wannier_function / np.sqrt(norm)
        result_vec[:, i_state] = wannier_function
        if plot:
            plt.figure(2)
            vec_plot = result_vec[:, i_state].reshape((n_x_window, n_x_window))
            cm = 1 / 2.54
            fig, ax = plt.subplots(1, 2, figsize=(7 * cm, 4 * cm))
            ax[0].axis('equal')
            norm = colors.SymLogNorm(vmin=-np.max(vec_plot), linthresh=1e-7,
                                     vmax=np.max(vec_plot))
            pcm = ax[0].pcolormesh(x_window, y_window, vec_plot, cmap='RdBu_r', norm=norm)

            the_divider = make_axes_locatable(ax[0])
            color_axis = the_divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(pcm, cax=color_axis)  # , extend='both')
            cbar.set_ticks([])
            ax[1].axis('equal')
            pcm1 = ax[1].pcolormesh(x_window, y_window, V_mask, cmap='jet')

            ax[1].scatter(neighbour_minima[:, 0], neighbour_minima[:, 1], c='k', edgecolor='None', s=1.5)
            ax[1].scatter(site[0], site[1], c='None', edgecolor='k', s=55.0, linewidth=0.8)

            the_divider = make_axes_locatable(ax[1])
            color_axis = the_divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(pcm1, cax=color_axis)  # , extend='both')
            cbar.set_ticks([])
            plt.savefig("wannier_site_" + str(index_site) + ".png", dpi=900)
            plt.clf()

    return result_vec


def dot_states(state_1: np.ndarray, site_1: Union[np.ndarray, Tuple[float, float]],
               state_2: np.ndarray, site_2: Union[np.ndarray, Tuple[float, float]],
               global_step: float, window_radius: float) -> Union[float, complex]:
    """
    Compute the dot product between two states defined over different local grids.

    Parameters:
    - state_1 (np.ndarray): The first state vector.
    - site_1 (Union[np.ndarray, Tuple[float, float]]): The coordinates for the first site.
    - state_2 (np.ndarray): The second state vector.
    - site_2 (Union[np.ndarray, Tuple[float, float]]): The coordinates for the second site.
    - global_step (float): The global step size for the grid.
    - window_radius (float): The radius of the local grid window around each site.

    Returns:
    - Union[float, complex]: The dot product of the two states or 0 if an error occurred.
    """

    # Generate the local grids for each site
    x_window_1, y_window_1 = generate_grid(site_1, window_radius, global_step)
    x_window_2, y_window_2 = generate_grid(site_2, window_radius, global_step)

    # Get the dimensions of the local grids
    n_x_window_2, n_y_window_2 = len(x_window_2), len(y_window_2)

    # Shift state_2 to the grid of state_1
    state_2_shift, error_flag = shift_to_global_grid(state_2.reshape((n_y_window_2, n_x_window_2)),
                                                     x_window_2, y_window_2,
                                                     x_window_1, y_window_1)

    # If no error occurred during the shifting, compute the dot product
    if error_flag == False:
        res = np.conj(state_1.ravel().T).dot(state_2_shift.ravel())
    else:
        res = 0.0

    return res


def compute_H(i_site: int, wannier_functions_vec: np.ndarray, minima_clean: np.ndarray,
              cut_off: float, window_radius: float, global_step: float,
              lattice_params: Dict[str, Union[float, np.ndarray]]) -> csc_matrix:
    """
    Compute a column of the Wannier Hamiltonian matrix for a given lattice site.

    Parameters:
    - i_site (int): The index of the lattice site under consideration.
    - wannier_functions_vec (np.ndarray): The array containing the Wannier functions.
    - minima_clean (np.ndarray): The coordinates of the lattice minima.
    - cut_off (float): The cutoff distance for considering interactions.
    - window_radius (float): The radius of the local grid window around each site.
    - global_step (float): The global step size for the grid.
    - lattice_params (Dict): Dictionary containing various lattice parameters.

    Returns:
    - csc_matrix: The computed column of the Wannier Hamiltonian matrix.
    """

    site_i = minima_clean[i_site]
    depth = lattice_params['depth']
    k = lattice_params['k']
    phis = lattice_params['phis']
    n_states = minima_clean.shape[0]

    H_wannier = np.zeros((n_states, 1))
    wannier_i = wannier_functions_vec[:, i_site]

    x_window_i, y_window_i = generate_grid(site_i, window_radius, global_step)
    X_window_i, Y_window_i = np.meshgrid(x_window_i, y_window_i)
    V_window_i = potential(X_window_i, Y_window_i, depth, k, phis)
    H_wannier_i = apply_hamiltonian(x_window_i, y_window_i, V_window_i, wannier_i,
                                    lattice_params['device'])

    for j_site in range(i_site + 1):
        site_j = minima_clean[j_site]
        distance = np.linalg.norm(np.subtract(site_i, site_j))

        if distance < 2 * np.sqrt(2) * window_radius - global_step:
            wannier_j = wannier_functions_vec[:, j_site]
            matrix_element = dot_states(H_wannier_i, site_i, wannier_j, site_j, global_step,
                                        window_radius) * global_step ** 2

            if j_site != i_site:
                H_wannier[j_site] = matrix_element
            else:
                H_wannier[j_site] = matrix_element / 2  # To help with symmetrization later on

    return csc_matrix(H_wannier)


def compute_S(i_site: int, wannier_functions_vec: np.ndarray, minima_clean: np.ndarray,
              cut_off: float, window_radius: float, global_step: float) -> csc_matrix:
    """
    Compute a column of the overlap matrix S for a given lattice site.

    Parameters:
    - i_site (int): The index of the lattice site under consideration.
    - wannier_functions_vec (np.ndarray): The array containing the Wannier functions.
    - minima_clean (np.ndarray): The coordinates of the lattice minima.
    - cut_off (float): The cutoff distance for considering interactions.
    - window_radius (float): The radius of the local grid window around each site.
    - global_step (float): The global step size for the grid.

    Returns:
    - csc_matrix: The computed column of the overlap matrix S.
    """
    site_i = minima_clean[i_site]
    n_states = minima_clean.shape[0]
    wannier_i = wannier_functions_vec[:, i_site]

    S_matrix = np.zeros((n_states, 1))

    for j_site in range(i_site + 1):
        site_j = minima_clean[j_site]
        distance = np.linalg.norm(np.subtract(site_i, site_j))

        if distance < 2 * np.sqrt(2) * window_radius + global_step:
            wannier_j = wannier_functions_vec[:, j_site]
            overlap = dot_states(wannier_i, site_i, wannier_j, site_j, global_step, window_radius) * global_step ** 2

            if j_site != i_site:
                S_matrix[j_site] = overlap
            else:
                S_matrix[j_site] = overlap / 2.0  # To help with symmetrization later on

    return csc_matrix(S_matrix)


def symmetric_orthogonalization(overlap):
    """Return S^(-1/2) and its spectrum, rejecting a numerically dependent basis."""
    overlap = np.asarray(overlap)
    if (overlap.ndim != 2 or overlap.shape[0] != overlap.shape[1] or not overlap.size
            or not np.isfinite(overlap).all() or np.iscomplexobj(overlap)
            or not np.allclose(overlap, overlap.T, rtol=1e-12, atol=1e-12)):
        raise ValueError("Overlap must be a finite real symmetric square matrix.")
    values, vectors = np.linalg.eigh(overlap)
    if values[0] <= len(values) * np.finfo(float).eps * max(1., values[-1]):
        raise ValueError(f"Overlap is nonpositive or numerically singular: smallest eigenvalue {values[0]:.3g}.")
    transform = (vectors * (1 / np.sqrt(values))) @ vectors.T
    if np.max(np.abs(transform.T @ overlap @ transform - np.eye(len(values)))) > 1e-8:
        raise ValueError("Löwdin transformation lost orthogonality; the overlap is poorly conditioned.")
    return transform, values


def compute_lowdin(i_site: int, wannier_functions_vec: np.ndarray, minima_clean: np.ndarray,
                   symm_orthog: np.ndarray, window_radius: float, global_step: float, plot: bool = False,
                   return_norm: bool = False):
    """
    Compute Löwdin orthonormalized wave function for a given lattice site.

    Parameters:
    - i_site (int): The index of the lattice site under consideration.
    - wannier_functions_vec (np.ndarray): The array containing the Wannier functions.
    - minima_clean (np.ndarray): The coordinates of the lattice minima.
    - symm_orthog (np.ndarray): Symmetric orthogonalization matrix.
    - window_radius (float): The radius of the local grid window around each site.
    - global_step (float): The global step size for the grid.
    - plot (bool): Whether to plot the result or not.
    - return_norm (bool): Also return the squared norm retained before renormalizing the crop.

    Returns:
    - np.ndarray: The Löwdin orthonormalized wave function at the site.
    """
    site_i = minima_clean[i_site]
    x_window_i, y_window_i = generate_grid(site_i, window_radius, global_step)
    n_x_window_i, n_y_window_i = len(x_window_i), len(y_window_i)
    res = np.zeros((n_x_window_i * n_y_window_i, 1))
    non_zero_indices = np.argwhere(symm_orthog[i_site, :] != 0.0).ravel()

    for j_site in non_zero_indices:
        site_j = minima_clean[j_site]
        state_j = wannier_functions_vec[:, j_site]
        x_window_j, y_window_j = generate_grid(site_j, window_radius, global_step)
        n_x_window_j, n_y_window_j = len(x_window_j), len(y_window_j)
        state_j_shift, error_flag = shift_to_global_grid(state_j.reshape((n_y_window_j, n_x_window_j)), x_window_j,
                                                         y_window_j,
                                                         x_window_i, y_window_i)
        if error_flag == False:
            res[:, 0] += symm_orthog[i_site, j_site] * state_j_shift.ravel()
    norm = float(np.sum(np.abs(res) ** 2) * global_step ** 2)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(f"Löwdin crop has zero or invalid norm at site {i_site}.")

    if plot:
        plt.figure(2)
        vec_plot = res.reshape((n_x_window_i, n_x_window_i))
        cm = 1 / 2.54
        fig, ax = plt.subplots(1, 1, figsize=(7 * cm, 4 * cm))
        ax.axis('equal')
        pcm = ax.pcolormesh(x_window_i, y_window_i, vec_plot, cmap='RdBu_r',
                            norm=colors.TwoSlopeNorm(vmin=-np.max(vec_plot),
                                                      vcenter=0., vmax=np.max(vec_plot)))

        ax.axis(xmin=site_i[0] - 2.3, xmax=site_i[0] + 2.3, ymin=site_i[1] - 2.3,
                ymax=site_i[1] + 2.3)  # ,option= 'equal')

        the_divider = make_axes_locatable(ax)
        color_axis = the_divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(pcm, cax=color_axis)  # , extend='both')
        cbar.set_ticks([])
        plt.savefig("lowdin_site_" + str(i_site) + ".png", dpi=900)
        plt.clf()
    result = res / np.sqrt(norm)
    return (result, norm) if return_norm else result


def count_neighbour_rings(site: Union[List[float], np.ndarray],
                          cut_off: float,
                          rings_list: np.ndarray) -> np.ndarray:
    """
    Count the neighboring ring sites for a given lattice site within a cutoff radius.

    Parameters:
    - site (Union[List[float], np.ndarray]): The coordinates of the lattice site.
    - cut_off (float): The cutoff distance for counting a ring as a neighbor.
    - rings_list (np.ndarray): The list of all ring sites in the lattice.

    Returns:
    - np.ndarray: The list of neighboring ring sites.
    """
    ring_radius = 0.35  # Defined radius of a ring
    neighbour_rings_list = np.zeros((0, 2))  # Initialize array to hold neighboring ring sites

    for i_ring in range(len(rings_list)):
        # Check if the ring site is a neighbor to the given lattice site
        if is_neighbour(site, rings_list[i_ring], cut_off + ring_radius):
            neighbour_rings_list = np.vstack([neighbour_rings_list, rings_list[i_ring, :]])

    return neighbour_rings_list


def in_octagon(X: Union[Tuple[np.ndarray, np.ndarray], np.ndarray]) -> np.ndarray:
    """
    Determine if points are inside the configuration space octagon.
    If they lie outside, these lattice sites do not host a lowest band Wannier function
    Parameters:
    - X (Union[Tuple[np.ndarray, np.ndarray], np.ndarray]): Tuple of x and y coordinates,
      or an array where the first row is x coordinates and the second row is y coordinates.

    Returns:
    - np.ndarray: Boolean array indicating whether each point is inside the octagon.
    """
    x, y = X
    check = np.ones((len(x), 8), dtype=bool)
    check[:, :] = False
    for i_theta in range(8):
        theta = i_theta * np.pi / 4
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        rotated_points = R.dot(np.reshape([x, y], (2, len(x)))).T
        x_rot, y_rot = zip(*rotated_points)
        check[:, i_theta] = np.abs(x_rot) <= 0.25
    return np.all(check, axis=1)


def compute_U_iijj(i_site: int,
                   wannier_functions_vec: np.ndarray,
                   minima_clean: np.ndarray,
                   cut_off: float,
                   window_radius: float,
                   global_step: float) -> csc_matrix:
    """
    Compute a column of the matrix U_iijj for specific i_site, considering only sites within a certain range.

    Parameters:
        - i_site (int): Index for site i.
        - wannier_functions_vec (np.ndarray): Wannier function vector.
        - minima_clean (np.ndarray): The coordinates of lattice sites.
        - cut_off (float): Cut-off distance.
        - window_radius (float): Radius of the window for local grid.
        - global_step (float): Step size for the grid.

    Returns:
        - csc_matrix: Sparse matrix with calculated U_iijj values.
    """
    site_i = minima_clean[i_site]
    n_states = minima_clean.shape[0]
    wannier_i = wannier_functions_vec[:, i_site]
    U_iijj = np.zeros((n_states, 1))
    for j_site in range(i_site):
        site_j = minima_clean[j_site]
        distance = np.linalg.norm(np.subtract(site_i, site_j))
        if (distance < 2 * np.sqrt(2) * window_radius + global_step):
            wannier_j = wannier_functions_vec[:, j_site]
            U_iijj[j_site] = dot_states(np.square(np.abs(wannier_i)), site_i, np.square(np.abs(wannier_j)), site_j,
                                        global_step,
                                        window_radius) * global_step ** 2
    return csc_matrix(U_iijj)


def compute_U_iiij(i_site: int,
                   wannier_functions_vec: np.ndarray,
                   minima_clean: np.ndarray,
                   cut_off: float,
                   window_radius: float,
                   global_step: float) -> csc_matrix:
    """
    Compute a column of the matrix U_iiij for specific i_site, considering only sites within a certain range.

    Parameters:
        - i_site (int): Index for site i.
        - wannier_functions_vec (np.ndarray): Wannier function vector.
        - minima_clean (np.ndarray): The coordinates of lattice sites.
        - cut_off (float): Cut-off distance.
        - window_radius (float): Radius of the window for local grid.
        - global_step (float): Step size for the grid.

    Returns:
        - csc_matrix: Sparse matrix with calculated U_iiij values.
    """
    site_i = minima_clean[i_site]
    n_states = minima_clean.shape[0]
    wannier_i = wannier_functions_vec[:, i_site]
    U_iiij = np.zeros((n_states, 1))
    for j_site in range(n_states):
        site_j = minima_clean[j_site]
        distance = np.linalg.norm(np.subtract(site_i, site_j))
        if (distance < 2 * np.sqrt(2) * window_radius + global_step):
            wannier_j = wannier_functions_vec[:, j_site]
            U_iiij[j_site] = dot_states(np.power(wannier_i, 3), site_i, wannier_j, site_j, global_step,
                                        window_radius) * global_step ** 2

    return csc_matrix(U_iiij)
