# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Parameter tables for management of the configures.
"""


class ParamTable:
    """Flatten parameter table."""

    def __init__(self, configs: dict):
        self._flatt = flatten(configs)

    def get(self, key: str):
        """Get the value of a parameter."""
        if key in self._flatt:
            return self._flatt[key]
        else:
            raise KeyError(f"Key {key} not found in parameter table.")

    def count(self, key: str):
        """Count the parameter occurrences."""
        count = 0
        if key in self._flatt:
            count += 1
        return count

    def copy(self):
        return ParamTable(unflatten(self._flatt))


def flatten(configs: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten a nested dictionary."""
    items = {}
    for k, v in configs.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def unflatten(flattened: dict, sep: str = ".") -> dict:
    """Unflatten a flattened dictionary."""
    out = {}
    for k, v in flattened.items():
        parts = k.split(sep)
        cur = out
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v
    return out
