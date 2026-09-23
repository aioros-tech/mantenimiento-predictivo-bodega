# -*- coding: utf-8 -*-
"""Download the weights listed in edge/models.yaml to <dest>/<repo>@<revision>/
unless already there (deploy-edge.sh runs this on the Jetson). Prints the paths."""
import shutil, sys
from pathlib import Path
import yaml
from huggingface_hub import snapshot_download


def model_dir(dest, m):
    return Path(dest) / f"{m['repo'].replace('/', '--')}@{m['revision']}"


if __name__ == "__main__":
    models_yaml, dest = sys.argv[1], sys.argv[2]
    for key, m in yaml.safe_load(open(models_yaml)).items():
        d = model_dir(dest, m)
        done = d / ".download-complete"
        if not done.exists():
            print(f"  downloading {m['repo']}@{m['revision'][:12]} ...", flush=True)
            tmp = d.with_name(d.name + ".partial")
            snapshot_download(m["repo"], revision=m["revision"], local_dir=tmp)
            shutil.rmtree(d, ignore_errors=True)   # interrupted between rename and touch
            tmp.rename(d)
            done.touch()
        size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 2**30
        print(f"  {key}: {d} ({size:.2f} GB)")
