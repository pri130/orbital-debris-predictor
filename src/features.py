"""

Compute analytic closest approach features for candidate pairs and label them.
"""
import numpy as np
import pandas as pd

def analytic_closest_approach(r_rel_km, v_rel_km_s, window_s=600.0):
    v_dot_v = np.dot(v_rel_km_s, v_rel_km_s)
    if v_dot_v <= 1e-12:
        t_star = 0.0
    else:
        t_star = - np.dot(r_rel_km, v_rel_km_s) / v_dot_v
    t_star = max(-window_s, min(window_s, t_star))
    r_at_t = r_rel_km + v_rel_km_s * t_star
    min_dist_km = np.linalg.norm(r_at_t)
    return t_star, min_dist_km

def build_features(candidate_df, ca_window_s=600.0, label_thresh_m=1000.0):
    rows = []
    for _, row in candidate_df.iterrows():
        r_rel_km = np.array([row['rx_j_km'] - row['rx_i_km'],
                             row['ry_j_km'] - row['ry_i_km'],
                             row['rz_j_km'] - row['rz_i_km']])
        v_rel_km_s = np.array([row['vx_j_km_s'] - row['vx_i_km_s'],
                               row['vy_j_km_s'] - row['vy_i_km_s'],
                               row['vz_j_km_s'] - row['vz_i_km_s']])
        t_star, min_dist_km = analytic_closest_approach(r_rel_km, v_rel_km_s, ca_window_s)
        min_dist_m = float(min_dist_km*1000.0)
        rel_speed_m_s = float(np.linalg.norm(v_rel_km_s)*1000.0)
        r_norm_m = float(np.linalg.norm(r_rel_km)*1000.0)
        if r_norm_m < 1e-9:
            closing_rate_m_s = 0.0
        else:
            closing_rate_m_s = float(np.dot(r_rel_km*1000.0, v_rel_km_s*1000.0) / r_norm_m)
        label = 1 if min_dist_m < label_thresh_m else 0
        rows.append({
            'i': row['i'], 'j': row['j'],
            'name_i': row['name_i'], 'name_j': row['name_j'],
            'min_dist_m': min_dist_m,
            'time_to_CA_s': float(t_star),
            'rel_speed_m_s': rel_speed_m_s,
            'closing_rate_m_s': closing_rate_m_s,
            'label': label
        })
    return pd.DataFrame(rows)
