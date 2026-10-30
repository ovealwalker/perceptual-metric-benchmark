"""Download the per-clip listening-test means and metric scores published by
the ODAQ benchmark (Fraunhofer IIS) into ./data. No audio is downloaded."""
import json
import os
import urllib.request

BASE = "https://raw.githubusercontent.com/Fraunhofer-IIS/ODAQ/main/benchmark/docs/"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def get(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with urllib.request.urlopen(url) as r, open(path, "wb") as f:
        f.write(r.read())


def main():
    get(BASE + "analysis.json", os.path.join(DATA, "analysis.json"))
    meta = json.load(open(os.path.join(DATA, "analysis.json")))
    for metric, rel in meta["detail_files"].items():
        name = rel.split("/")[-1]
        get(BASE + "data/" + name, os.path.join(DATA, "detail", name))
        print("downloaded", metric)


if __name__ == "__main__":
    main()
