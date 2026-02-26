#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seismic hazard deaggregation plot with tectonic region type (TRT) coloring.

New function: deagg_plot_trt
    Same as deagg_plot, but accepts deaggregation files for individual
    tectonic regions and colors each bar according to the dominant TRT
    at that (M, R) bin.

Helper functions (unchanged from original):
    sph2cart, sphview, ravzip
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from os.path import join

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams.update({'font.size': 14})


# -----------------------------------------------------------------------------
# Helper functions (unchanged from original)
# -----------------------------------------------------------------------------

def sph2cart(r, theta, phi):
    """Spherical to Cartesian transformation."""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def sphview(ax):
    """Returns the camera position for 3D axes in spherical coordinates."""
    r = np.square(np.max([ax.get_xlim(), ax.get_ylim()], 1)).sum()
    theta, phi = np.radians((90 - ax.elev, ax.azim))
    return r, theta, phi


def ravzip(*itr):
    """Flatten and zip arrays."""
    return zip(*map(np.ravel, itr))


# -----------------------------------------------------------------------------
# Core computation: load a deagg file and return P(m | X>x) for each bin
# -----------------------------------------------------------------------------

def _load_deagg(deagg_file, prob_col_name, investigation_time):
    """
    Load an OpenQuake Mag/Dist deagg CSV and compute the conditional
    contribution P(M=m, R=r | X>x) for every bin.

    Returns the DataFrame with an added column 'P(m | X>x)'.
    """
    df = pd.read_csv(deagg_file, header=1)

    poe = df[prob_col_name].to_numpy()
    tot_prob_ex = 1 - np.prod(1 - poe)
    nu = -np.log(1 - tot_prob_ex) / investigation_time
    nu_m = -np.log(1 - poe) / investigation_time + 0.0

    df['P(m | X>x)'] = nu_m / nu
    return df


# -----------------------------------------------------------------------------
# Main plotting function
# -----------------------------------------------------------------------------

def deagg_plot_trt(deagg_file_all,
                   trt_files,
                   trt_weights,
                   prob_col_name,
                   delta_M,
                   delta_R,
                   investigation_time,
                   deagg_name,
                   plot_lim_distance=None,
                   cam_elev=35,
                   cam_azim=315,
                   trt_colors=None,
                   trt_labels=None,
                   min_contribution=0.05,
                   fig_save_name=None):
    """
    Plot a Mag/Dist seismic hazard deaggregation with bars colored by the
    dominant tectonic region type (TRT).

    Parameters
    ----------
    deagg_file_all : str
        Path to the OpenQuake Mag_Dist deagg CSV for ALL tectonic regions
        combined (the ``_for_plot.csv`` file).  This file drives the bar
        heights; the per-TRT files only drive the coloring.

    trt_files : list of str
        Paths to the per-TRT deagg CSV files in the same order as
        ``trt_weights`` and ``trt_labels``.  Each file must cover the same
        (mag, dist) grid as ``deagg_file_all``.

    trt_weights : list of float
        Fractional contribution of each TRT to the total hazard at the
        intensity level of interest (must sum to 1).  These are the
        percentages you can read off the TRT deagg output (e.g. from
        ``TRT-mean.csv``), divided by 100.
        Example: [0.7374, 0.0, 0.0684, 0.2072]  (crustal, stable, inslab, interface)

    prob_col_name : str
        Column name that contains P(X>x | M, R) in the CSV files
        (typically ``'mean'``).

    delta_M : float
        Magnitude bin width used in the OpenQuake calculation.

    delta_R : float
        Distance bin width [km] used in the OpenQuake calculation.

    investigation_time : float
        Investigation time [years] used in the OpenQuake calculation.

    deagg_name : str
        Title string for the plot.

    plot_lim_distance : float or None
        Maximum distance [km] to include in the plot.  Bins beyond this
        are silently dropped.  ``None`` means no limit.

    cam_elev : float
        Camera elevation angle for the 3-D view (degrees).

    cam_azim : float
        Camera azimuth angle for the 3-D view (degrees).

    trt_colors : list of str or None
        Matplotlib color strings, one per TRT.  If ``None``, a default
        colorblind-friendly palette is used.

    trt_labels : list of str or None
        Legend labels for each TRT.  If ``None``, generic labels
        ``['TRT 1', 'TRT 2', ...]`` are used.

    min_contribution : float
        Bars with a total % contribution below this threshold are hidden
        (rendered in white with no edge) to reduce visual clutter.

    Returns
    -------
    df_all : pandas.DataFrame
        The combined deagg DataFrame with columns ``mag``, ``dist``,
        ``P(m | X>x)``, and ``dominant_trt`` (integer index into
        ``trt_labels``).

    Notes
    -----
    Coloring logic
        For every (mag, dist) bin the function computes the *weighted*
        contribution from each individual TRT deagg file::

            contribution_trt_i(m, r) = trt_weights[i] * P_i(m, r | X>x)

        The bar is then colored with the color of the TRT that has the
        largest weighted contribution at that bin.  When a (mag, dist)
        bin exists in ``deagg_file_all`` but is absent from a per-TRT
        file (some TRTs may not produce earthquakes in that bin), the
        contribution for that TRT is taken as zero.
    """

    # ------------------------------------------------------------------
    # 1. Load the combined ("all TRTs") deagg file
    # ------------------------------------------------------------------
    df_all = _load_deagg(deagg_file_all, prob_col_name, investigation_time)

    # Mean magnitude and distance (weighted by contribution)
    mu_M = np.sum(df_all['P(m | X>x)'].to_numpy() * df_all['mag'].to_numpy())
    mu_R = np.sum(df_all['P(m | X>x)'].to_numpy() * df_all['dist'].to_numpy())

    # Optional distance filter
    if plot_lim_distance is not None:
        df_all = df_all[df_all['dist'] <= plot_lim_distance].copy()

    n_bins = len(df_all)

    # ------------------------------------------------------------------
    # 2. Validate inputs
    # ------------------------------------------------------------------
    n_trt = len(trt_files)
    if len(trt_weights) != n_trt:
        raise ValueError("trt_files and trt_weights must have the same length.")

    # Default colors (colorblind-friendly)
    default_colors = ['#E69F00',   # orange  – crustal
                      '#56B4E9',   # sky blue – stable
                      '#009E73',   # green   – inslab
                      '#CC79A7',   # pink    – interface
                      '#0072B2',   # blue
                      '#D55E00',   # vermillion
                      '#F0E442']   # yellow
    if trt_colors is None:
        trt_colors = default_colors[:n_trt]
    if trt_labels is None:
        trt_labels = [f'TRT {i+1}' for i in range(n_trt)]

    # ------------------------------------------------------------------
    # 3. Load each per-TRT deagg file and compute weighted bin contributions
    # ------------------------------------------------------------------
    # Build a contribution matrix: shape (n_bins, n_trt)
    # Key: (mag, dist) tuple -> row index in df_all
    all_keys = list(zip(df_all['mag'].to_numpy(), df_all['dist'].to_numpy()))
    key_to_idx = {k: i for i, k in enumerate(all_keys)}

    contrib_matrix = np.zeros((n_bins, n_trt))

    for trt_idx, (trt_file, weight) in enumerate(zip(trt_files, trt_weights)):
        df_trt = _load_deagg(trt_file, prob_col_name, investigation_time)

        # Filter to same distance limit
        if plot_lim_distance is not None:
            df_trt = df_trt[df_trt['dist'] <= plot_lim_distance]

        for _, row in df_trt.iterrows():
            key = (row['mag'], row['dist'])
            if key in key_to_idx:
                bin_idx = key_to_idx[key]
                contrib_matrix[bin_idx, trt_idx] = weight * row['P(m | X>x)']

    # Dominant TRT for each bin: argmax across TRT axis
    dominant_trt = np.argmax(contrib_matrix, axis=1)  # shape (n_bins,)

    df_all = df_all.copy()
    df_all['dominant_trt'] = dominant_trt

    # ------------------------------------------------------------------
    # 4. Set up bar geometry
    # ------------------------------------------------------------------
    x = df_all['dist'].to_numpy() - delta_R / 2
    y = df_all['mag'].to_numpy() - delta_M / 2
    z = np.zeros(n_bins)

    dx = np.full(n_bins, delta_R / 2)
    dy = np.full(n_bins, delta_M / 2)
    dz = 100 * df_all['P(m | X>x)'].to_numpy()   # % contribution

    # ------------------------------------------------------------------
    # 5. Plot
    # ------------------------------------------------------------------
    plt.rcParams.update({'font.size': 16})
    plt.figure(dpi=500, figsize=(14, 24))
    ax = plt.axes(projection='3d')
    ax.view_init(elev=cam_elev, azim=cam_azim)

    # Determine back-to-front plotting order to fix clipping artifacts
    x_c, y_c, z_c = sph2cart(*sphview(ax))
    zo = x * x_c + y * y_c + z * z_c
    idx_plot_order = zo.argsort()

    for idx in idx_plot_order:
        dz_val = dz[idx]

        if dz_val < min_contribution:
            color_curr = 'white'
            edge_col_curr = 'white'
        else:
            trt_idx = dominant_trt[idx]
            color_curr = trt_colors[trt_idx]
            edge_col_curr = 'black'

        pl = ax.bar3d(x[idx], y[idx], z[idx],
                      dx[idx], dy[idx], dz_val,
                      color=color_curr, alpha=0.9,
                      edgecolor=edge_col_curr,
                      shade=False)
        pl._sort_zpos = zo[idx]

    # ------------------------------------------------------------------
    # 6. Formatting
    # ------------------------------------------------------------------
    # Transparent pane backgrounds
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    ax.set_xlim(0, 200)
    ax.set_ylim(4.0, 10.0)
    ax.set_zlim(0.0, 20.0)

    ax.set_xlabel('R [km]')
    ax.set_ylabel('M')
    ax.set_zlabel('% contribution')
    
    # uncomment for plot title
    # plt.title(
    #     deagg_name +
    #     '\n mean: M = {}, R = {} km'.format(np.round(mu_M, 2), np.round(mu_R, 2))
    # )

    # Legend patches – only show TRTs that actually have non-zero weight
    legend_patches = []
    for i, (label, color, weight) in enumerate(zip(trt_labels, trt_colors, trt_weights)):
        if weight > 0:
            legend_patches.append(
                mpatches.Patch(facecolor=color, edgecolor='black',
                               label=f'{label} ({100*weight:.1f}%)')
            )
    if legend_patches:
        ax.legend(handles=legend_patches,
                  loc='upper left',
                  bbox_to_anchor=(0.45, 0.75), #(0.0, 1.0),
                  fontsize=20,
                  framealpha=0.8)

    ax.grid(True, linestyle='--')
    plt.tight_layout()
    ax.set_xlabel('R [km]',  fontsize=20, labelpad=10)
    ax.set_ylabel('M',       fontsize=20, labelpad=10)
    ax.set_zlabel('% contribution', fontsize=20, labelpad=3)
    
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='z', labelsize=16)
    if fig_save_name is not None:
        plt.savefig(fig_save_name+'.pdf', dpi=200, bbox_inches='tight')
    plt.show()

    return df_all


# =============================================================================
# Example usage
# =============================================================================
if __name__ == '__main__':

# =============================================================================
# Example usage – 4-story MRF, Seattle, SaAvg, 2% in 50 years
# =============================================================================    

    # Paths to the deagg files (update these to match your directory layout)
    
    haz_deagg_path = join('.','hazard_data','4-story_mrf')
    deagg_file_all      = join(haz_deagg_path, 'saAvg_all_trt_4_story_mrf_2in50','Mag_Dist-mean_for_plot.csv')       # all TRTs
    deagg_file_crustal  = join(haz_deagg_path, 'saAvg_crustal_trt_4_story_mrf_2in50', 'Mag_Dist-mean_for_plot.csv')  # active shallow crust
    deagg_file_inslab   = join(haz_deagg_path,'saAvg_inslab_trt_4_story_mrf_2in50','Mag_Dist-mean_for_plot.csv')   # subduction inslab
    deagg_file_interface = join(haz_deagg_path, 'saAvg_interface_trt_4_story_mrf_2in50','Mag_Dist-mean_for_plot.csv')  # subduction interface

    # Fractional TRT contributions to total hazard (from TRT-mean.csv)
    # Active Shallow Crust = 73.74 %, Stable Shallow Crust = 0 %,
    # Subduction Inslab = 6.84 %, Subduction Interface = 20.72 %
    trt_weights = [0.735, 0.0, 0.065, 0.20]

    trt_files = [
        deagg_file_crustal,
        None,               # Stable Shallow Crust – no file needed (weight = 0)
        deagg_file_inslab,
        deagg_file_interface,
    ]

    # For TRTs with weight = 0 we still need a placeholder; filter them out:
    active_files   = [f for f, w in zip(trt_files, trt_weights) if w > 0]
    active_weights = [w for w in trt_weights if w > 0]
    active_labels  = ['Active Shallow Crust', 'Subduction Inslab', 'Subduction Interface']
    active_colors  = ['#1b2a6b', '#d4a017', '#008b8b']#['#E69F00', '#009E73', '#CC79A7']
    
    fig_save_name = 'deagg_4_story_2in50'
    
    deagg_plot_trt(
        deagg_file_all      = deagg_file_all,
        trt_files           = active_files,
        trt_weights         = active_weights,
        prob_col_name       = 'mean',
        delta_M             = 0.2,
        delta_R             = 5,
        investigation_time  = 1,
        deagg_name          = 'Seattle, 4-story MRF, SaAvg, 2% in 50 years',
        plot_lim_distance   = 200,
        trt_labels          = active_labels,
        trt_colors          = active_colors, 
        fig_save_name       = fig_save_name
    )

# =============================================================================
# Example usage – 8-story MRF, Seattle, SaAvg, 2% in 50 years
# =============================================================================    

    # Paths to the deagg files (update these to match your directory layout)
    
    haz_deagg_path = join('.','hazard_data','8-story_mrf')
    deagg_file_all      = join(haz_deagg_path, 'saAvg_all_trt_8_story_mrf_2in50','Mag_Dist-mean_for_plot.csv')       # all TRTs
    deagg_file_crustal  = join(haz_deagg_path, 'saAvg_crustal_trt_8_story_mrf_2in50', 'Mag_Dist-mean_for_plot.csv')  # active shallow crust
    deagg_file_inslab   = join(haz_deagg_path,'saAvg_inslab_trt_8_story_mrf_2in50','Mag_Dist-mean_for_plot.csv')   # subduction inslab
    deagg_file_interface = join(haz_deagg_path, 'saAvg_interface_trt_8_story_mrf_2in50','Mag_Dist-mean_for_plot.csv')  # subduction interface

    # Fractional TRT contributions to total hazard (from TRT-mean.csv)
    # Active Shallow Crust = 76.5 %, Stable Shallow Crust = 0 %,
    # Subduction Inslab = 2.7 %, Subduction Interface = 20.8 %
    trt_weights = [0.765, 0.0, 0.027, 0.208]

    trt_files = [
        deagg_file_crustal,
        None,               # Stable Shallow Crust – no file needed (weight = 0)
        deagg_file_inslab,
        deagg_file_interface,
    ]

    # For TRTs with weight = 0 we still need a placeholder; filter them out:
    active_files   = [f for f, w in zip(trt_files, trt_weights) if w > 0]
    active_weights = [w for w in trt_weights if w > 0]
    active_labels  = ['Active Shallow Crust', 'Subduction Inslab', 'Subduction Interface']
    active_colors  = ['#1b2a6b', '#d4a017', '#008b8b']#['#E69F00', '#009E73', '#CC79A7']
    
    fig_save_name = 'deagg_8_story_2in50'
    
    deagg_plot_trt(
        deagg_file_all      = deagg_file_all,
        trt_files           = active_files,
        trt_weights         = active_weights,
        prob_col_name       = 'mean',
        delta_M             = 0.2,
        delta_R             = 5,
        investigation_time  = 1,
        deagg_name          = 'Seattle, 8-story MRF, SaAvg, 2% in 50 years',
        plot_lim_distance   = 200,
        trt_labels          = active_labels,
        trt_colors          = active_colors, 
        fig_save_name       = fig_save_name
    )
