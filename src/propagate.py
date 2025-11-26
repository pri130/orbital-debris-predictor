"""

Propagate a set of TLEs to an epoch using sgp4 and save positions.csv
"""
import json
from sgp4.api import Satrec, jday
from datetime import datetime
import csv

def propagate_triples(triples, epoch=None):
    if epoch is None:
        epoch = datetime.utcnow()
    jd, fr = jday(epoch.year, epoch.month, epoch.day, epoch.hour, epoch.minute, epoch.second + epoch.microsecond*1e-6)
    rows = []
    for name, l1, l2 in triples:
        try:
            sat = Satrec.twoline2rv(l1, l2)
            err, r, v = sat.sgp4(jd, fr)
            if err == 0:
                rows.append({
                    "name": name,
                    "datetime_utc": epoch.isoformat(),
                    "rx_km": r[0], "ry_km": r[1], "rz_km": r[2],
                    "vx_km_s": v[0], "vy_km_s": v[1], "vz_km_s": v[2]
                })
        except Exception:
            pass
    return rows

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tles", required=True, help="JSON file with parsed TLE triples")
    parser.add_argument("--out", default="positions.csv")
    args = parser.parse_args()
    triples = json.load(open(args.tles))
    rows = propagate_triples(triples)
    import pandas as pd
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Saved {len(rows)} positions to {args.out}")
