"""
Simple TLE parser: collects triples (name, line1, line2) from raw text files.
"""
import os

def parse_tle_file(path):
    triples = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n\r") for ln in f.readlines() if ln.strip()]
    i = 0
    while i < len(lines):
        if lines[i].startswith("1 ") and i+1 < len(lines) and lines[i+1].startswith("2 "):
            # no name line, form NORAD_<id>
            l1 = lines[i]; l2 = lines[i+1]
            norad = l1.split()[1] if len(l1.split()) > 1 else "UNK"
            name = f"NORAD_{norad}"
            triples.append((name, l1, l2))
            i += 2
        elif i+2 < len(lines) and lines[i+1].startswith("1 ") and lines[i+2].startswith("2 "):
            name = lines[i]; l1 = lines[i+1]; l2 = lines[i+2]
            triples.append((name, l1, l2))
            i += 3
        else:
            i += 1
    return triples

if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="TLE text file")
    parser.add_argument("--out", default="parsed_tles.json")
    args = parser.parse_args()
    triples = parse_tle_file(args.input)
    with open(args.out, "w") as fw:
        json.dump(triples, fw, indent=2)
    print(f"Saved {len(triples)} TLE triples to {args.out}")
