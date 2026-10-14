import io
import zipfile
import numpy as np
import pandas as pd

def parse_french(path):
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
        lines = archive.read(name).decode("utf-8-sig").splitlines()
    # First daily table is value weighted; never concatenate the second table.
    start = next(i for i, line in enumerate(lines) if line.split(",")[0].strip().isdigit() and len(line.split(",")[0].strip()) == 8)
    end = start
    while end < len(lines) and lines[end].split(",")[0].strip().isdigit() and len(lines[end].split(",")[0].strip()) == 8:
        end += 1
    frame = pd.read_csv(io.StringIO("\n".join([lines[start-1], *lines[start:end]])), index_col=0)
    frame.columns = frame.columns.str.strip()
    frame.index = pd.to_datetime(frame.index.astype(str).str.strip(), format="%Y%m%d")
    frame = frame.apply(pd.to_numeric, errors="coerce").replace([-99.99, -999], np.nan).dropna()/100
    return frame.loc[~frame.index.duplicated()].sort_index()

def split_asof(frame, train_end, test_start):
    if pd.Timestamp(train_end) >= pd.Timestamp(test_start):
        raise ValueError("Training must end strictly before test begins")
    train, test = frame.loc[:train_end], frame.loc[test_start:]
    if train.empty or test.empty:
        raise ValueError("Empty train or test window")
    return train, test

