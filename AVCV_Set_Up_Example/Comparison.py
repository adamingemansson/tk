"""
Comparison utilities for pairing two tracking tables by time (t) and nearest 3D neighbor.

Column assumptions (for both CSVs)
- index 0: ID
- index 1: t0
- index 2: t
- index 3: x
- index 4: y
- index 5: z
- index 6: FI
- index 7: Track Length

Functions
- comp(base_path, base_name, sec_path, sec_name, threshold) -> pandas.DataFrame
- cov(comparison, base_name, sec_name, threshold) -> pandas.DataFrame
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Union

import numpy as np
import pandas as pd

__all__ = ["comp", "cov"]


def comp(base_path: str, base_name: str, sec_path: str, sec_name: str, threshold: float) -> pd.DataFrame:
    """
    Compare two tracking datasets frame-by-frame and select the nearest secondary point in 3D.

    For each base row at time t, the closest secondary row at the same t is selected.
    If no secondary rows exist for that t, secondary fields are left empty.

    Parameters
    ----------
    base_path : str
        Path to the base CSV.
    base_name : str
        Label used for base columns (e.g., "CME").
    sec_path : str
        Path to the secondary CSV.
    sec_name : str
        Label used for secondary columns (e.g., "Dino").
    threshold : float
        Distance threshold used only for listing multiple candidate IDs/distances (diagnostics).

    Returns
    -------
    pandas.DataFrame
        Comparison table with base and secondary info, nearest distance, and multi-ID diagnostics,
        sorted by base ID and t.
    """

    print("Creating Comparison file...")
    # Light validation
    assert isinstance(base_name, str) and base_name.strip(), "base_name must be a non-empty string"
    assert isinstance(sec_name, str) and sec_name.strip(), "sec_name must be a non-empty string"
    assert isinstance(base_path, str) and os.path.isfile(base_path), f"Base CSV not found: {base_path}"
    assert isinstance(sec_path, str) and os.path.isfile(sec_path), f"Secondary CSV not found: {sec_path}"
    threshold = float(threshold)

    # Load datasets
    df_base = pd.read_csv(base_path)
    df_sec = pd.read_csv(sec_path)

    # Minimal shape checks
    assert df_base.shape[1] >= 8, "Base CSV must have at least 8 columns in the expected order"
    assert df_sec.shape[1] >= 8, "Secondary CSV must have at least 8 columns in the expected order"

    M_base = df_base.to_numpy()
    M_sec = df_sec.to_numpy()

    # Output: base(0:8), sec(8:16), distance(16), multi_id(17), multi_dist(18)
    comp_list = np.zeros((len(M_base), 19), dtype=object)

    # Group secondary entries by time t (column index 2)
    t_vec = defaultdict(list)
    for vec_sec in M_sec:
        t_vec[vec_sec[2]].append(vec_sec)
    for t_val in list(t_vec.keys()):
        t_vec[t_val] = np.array(t_vec[t_val])

    # Match each base row to the nearest secondary row at the same t
    for i, vec_base in enumerate(M_base):
        t_val = vec_base[2]
        sec_group = t_vec.get(t_val)

        # If no secondary entries at this time, leave secondary fields empty
        if sec_group is None or len(sec_group) == 0:
            comp_list[i, 0:8] = vec_base
            comp_list[i, 8:16] = [None] * 8
            comp_list[i, 16] = np.nan
            comp_list[i, 17] = ""
            comp_list[i, 18] = ""
            continue

        # Compute Euclidean distances in (x, y, z); indices 3:6 are x, y, z
        diffs = sec_group[:, 3:6] - vec_base[3:6]
        dists = np.linalg.norm(diffs, axis=1)
        min_idx = int(np.argmin(dists))
        min_dist = float(dists[min_idx])
        best_vec_sec = sec_group[min_idx]

        # Fill output row
        comp_list[i, 0:8] = vec_base
        comp_list[i, 8:16] = best_vec_sec
        comp_list[i, 16] = min_dist

        # Diagnostics: list all secondary IDs within threshold
        below = np.where(dists < threshold)[0]
        if len(below) > 0:
            sec_ids = ",".join(str(int(sec_group[j, 0])) for j in below)
            dists_str = ",".join(f"{dists[j]:.2f}" for j in below)
        else:
            sec_ids = ""
            dists_str = ""
        comp_list[i, 17] = sec_ids
        comp_list[i, 18] = dists_str

    # Build DataFrame using t0 instead of t_start
    df_comp = pd.DataFrame(
        comp_list,
        columns=[
            f"ID ({base_name})",
            f"t0 ({base_name})",
            "t",
            f"x ({base_name})",
            f"y ({base_name})",
            f"z ({base_name})",
            f"FI ({base_name})",
            f"Track Length ({base_name})",
            f"ID ({sec_name})",
            f"t0 ({sec_name})",
            "t_ig",  # ignored placeholder for original structure alignment
            f"x ({sec_name})",
            f"y ({sec_name})",
            f"z ({sec_name})",
            f"FI ({sec_name})",
            f"Track Length ({sec_name})",
            "Distance",
            f"Multi ID ({sec_name})",
            f"Multi Distance ({sec_name})",
        ],
    )
    # Drop placeholder and reorder for downstream stability
    df_comp = df_comp.drop("t_ig", axis=1)
    df_comp = df_comp[
        [
            f"ID ({base_name})",
            f"ID ({sec_name})",
            f"x ({base_name})",
            f"y ({base_name})",
            f"z ({base_name})",
            f"x ({sec_name})",
            f"y ({sec_name})",
            f"z ({sec_name})",
            "t",
            f"t0 ({base_name})",
            f"t0 ({sec_name})",
            f"FI ({base_name})",
            f"FI ({sec_name})",
            f"Track Length ({base_name})",
            f"Track Length ({sec_name})",
            "Distance",
            f"Multi ID ({sec_name})",
            f"Multi Distance ({sec_name})",
        ]
    ]
    # Sort for consistent usage
    df_sorted = df_comp.sort_values(by=[f"ID ({base_name})", "t"], ascending=[True, True])
    return df_sorted


def cov(comparison: Union[str, pd.DataFrame], base_name: str, sec_name: str, threshold: float) -> pd.DataFrame:
    """
    Compute simple coverage stats from a comparison table produced by comp().

    For each base ID:
    - Track Coverage = number of rows with Distance < threshold.
    - Missing Time Points = list of t where Distance > threshold.
    - Secondary segments summarized as "sec_id,start_index,track_length" joined by ';'.

    Parameters
    ----------
    comparison : str | pandas.DataFrame
        Path to a comparison CSV or the comparison DataFrame itself.
    base_name : str
        Label used for base columns (must match the label passed to comp()).
    sec_name : str
        Label used for secondary columns (must match the label passed to comp()).
    threshold : float
        Distance threshold used to decide coverage and missing time points.

    Returns
    -------
    pandas.DataFrame
        One row per base ID with basic coverage stats and a compact secondary summary.
    """

    print("Creating Coverage file...")

    # Load comparison input
    if isinstance(comparison, str):
        assert os.path.isfile(comparison), f"Comparison CSV not found: {comparison}"
        df_comp = pd.read_csv(comparison)
    elif isinstance(comparison, pd.DataFrame):
        df_comp = comparison
    else:
        raise TypeError("comparison must be a CSV file path or a pandas.DataFrame")

    threshold = float(threshold)
    M_comp = df_comp.to_numpy()

    # Group by base ID (column 0 after reordering in comp())
    ID_vec = defaultdict(list)
    for row in M_comp:
        ID_vec[row[0]].append(row)
    for key in list(ID_vec.keys()):
        ID_vec[key] = np.array(ID_vec[key])

    # Output: [ID(base), Track Length(base), Track Coverage, Missing t list, Secondary summary]
    ID_list = np.zeros((len(ID_vec), 5), dtype=object)

    for i, key in enumerate(ID_vec):
        ID_val = ID_vec[key]

        # Coverage and missing t use Distance column at index 15 (after reordering in comp())
        track_coverage = int(np.sum(ID_val[:, 15] < threshold))
        missing_t = ID_val[ID_val[:, 15] > threshold][:, 8]

        # Fill base fields
        ID_list[i][0] = ID_val[0][0]       # ID (base)
        ID_list[i][1] = ID_val[0][13]      # Track Length (base)
        ID_list[i][2] = track_coverage
        ID_list[i][3] = missing_t.tolist()

        # Build compact secondary segment summary across contiguous blocks of secondary IDs
        change_points = np.where(np.diff(ID_val[:, 1]) != 0)[0] + 1  # index where ID(sec) changes
        block_starts = np.insert(change_points, 0, 0)
        track_index = np.insert(change_points, 0, 0)
        y_ID = ID_val[:, 1][block_starts]         # IDs of secondary segments
        track_lengths = ID_val[:, 14][track_index]  # Track Length (sec)
        y = np.stack((y_ID, block_starts + 1, track_lengths), axis=1)
        y_all = ';'.join([','.join(map(str, row)) for row in y])
        ID_list[i][4] = y_all

    df_ID = pd.DataFrame(
        ID_list,
        columns=[
            f"ID ({base_name})",
            f"Track Length ({base_name})",
            "Track Coverage",
            "Missing Time Points",
            f"{sec_name} Data",
        ],
    )
    return df_ID