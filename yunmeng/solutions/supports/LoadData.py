# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Load data from different standard formats.
"""
import os
import json
import numpy as np
import pandas as pd
import netCDF4 as nc


def load_data(file_path: str) -> pd.DataFrame | dict | nc.Dataset:
    """
    Load data from a file.
    """
    if os.path.isfile(file_path):
        file_name, file_ext = os.path.splitext(file_path)
        if file_ext == ".csv":
            return load_csv(file_path)
        elif file_ext == ".json":
            return load_json(file_path)
        elif file_ext == ".h5" or file_ext == ".hdf5":
            return load_hdf5(file_path)
        elif file_ext == ".nc":
            return load_nc(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {file_ext}")
    else:
        raise FileNotFoundError(f"File not found: {file_path}")


def load_csv(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    """
    return pd.read_csv(file_path)


def load_json(file_path: str) -> dict:
    """
    Load data from a JSON file.
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def load_hdf5(file_path: str) -> pd.DataFrame:
    """
    Load data from a HDF5 file.
    """
    return pd.read_hdf(file_path)


def load_nc(file_path: str) -> nc.Dataset:
    """
    Load data from a NetCDF file.
    """
    return nc.Dataset(file_path, "r")
