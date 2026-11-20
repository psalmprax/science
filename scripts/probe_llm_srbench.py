#!/usr/bin/env python3
"""Print what the LLM-SRBench dataset on the Hugging Face hub contains, so that a loader can be written against it.

Read-only reconnaissance: lists the files, shows each table's columns and first row, and walks the first
few groups of any HDF5 file. Nothing is written to the repository. The dataset is treated as data: nothing
in it is executed.
"""
from __future__ import annotations

import sys

REPO = "nnheui/llm-srbench"


def main() -> int:
    from huggingface_hub import hf_hub_download, list_repo_files

    files = sorted(list_repo_files(REPO, repo_type="dataset"))
    print("FILES", len(files))
    for name in files:
        print("  FILE", name)
    for name in files:
        if name.endswith(".parquet"):
            import pandas as pd

            frame = pd.read_parquet(hf_hub_download(REPO, name, repo_type="dataset"))
            print("TABLE", name, "rows", len(frame), "columns", list(frame.columns))
            for column in frame.columns:
                print("   ", column, "=", repr(frame[column].iloc[0])[:300])
        elif name.endswith((".hdf5", ".h5")):
            import h5py

            path = hf_hub_download(REPO, name, repo_type="dataset")
            with h5py.File(path, "r") as handle:
                shown = 0

                def walk(key: str, item: object) -> None:
                    nonlocal shown
                    if shown < 60:
                        shape = getattr(item, "shape", None)
                        print("   H5", key, shape if shape is not None else "(group)", getattr(item, "dtype", ""))
                        shown += 1

                handle.visititems(walk)
                print("  H5 top-level", list(handle.keys())[:20])
    return 0


if __name__ == "__main__":
    sys.exit(main())
