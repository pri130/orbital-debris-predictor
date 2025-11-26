"""
KD-tree screening: find candidate pairs within radius (meters)
"""
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

def find_candidate_pairs(positions_csv, screen_radius_m=200000.0):
    pos = pd.read_csv(positions_csv)
    coords = pos[['rx_km','ry_km','rz_km']].values
    r_km = screen_radius_m / 1000.0
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=r_km)
    rows = []
    for i,j in pairs:
        a = pos.iloc[i]; b = pos.iloc[j]
        rows.append({
            'i': int(i), 'j': int(j),
            'name_i': a['name'], 'name_j': b['name'],
            'rx_i_km': a['rx_km'], 'ry_i_km': a['ry_km'], 'rz_i_km': a['rz_km'],
            'rx_j_km': b['rx_km'], 'ry_j_km': b['ry_km'], 'rz_j_km': b['rz_km'],
            'vx_i_km_s': a['vx_km_s'], 'vy_i_km_s': a['vy_km_s'], 'vz_i_km_s': a['vz_km_s'],
            'vx_j_km_s': b['vx_km_s'], 'vy_j_km_s': b['vy_km_s'], 'vz_j_km_s': b['vz_km_s'],
        })
    return pd.DataFrame(rows)
