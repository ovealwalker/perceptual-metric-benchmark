"""Build one tidy table: one row per rated audio clip, with the listeners'
MUSHRA mean and every metric's score, from the ODAQ benchmark export."""
import json, glob, os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

def decode(block):
    cols, dicts, rows = block["columns"], block["dictionaries"], block["rows"]
    out = []
    for r in rows:
        rec = {}
        for c, v in zip(cols, r):
            rec[c] = dicts[c][v] if c in dicts and isinstance(v, int) and not isinstance(v, bool) else v
        out.append(rec)
    return pd.DataFrame(out)

def main():
    meta = json.load(open(os.path.join(DATA, "analysis.json")))
    key = ["subset", "method_group", "item", "condition"]
    table = None
    for path in sorted(glob.glob(os.path.join(DATA, "detail", "*.json"))):
        name = os.path.basename(path)[:-5]
        if name == "rho-perfect":
            continue
        pairs = decode(json.load(open(path))["pairs"])
        if table is None:
            table = pairs[key + ["content_type", "spatial", "is_anchor", "mean_score"]].copy()
        table = table.merge(pairs[key + ["obj_score"]].rename(columns={"obj_score": name}), on=key, how="outer")
    table.to_csv(os.path.join(DATA, "odaq_table.csv"), index=False)
    print(table.shape)
    print(table[["subset", "method_group", "content_type"]].value_counts().head(30))
    print("negated (lower = better):", meta["negated_metrics"])

if __name__ == "__main__":
    main()
