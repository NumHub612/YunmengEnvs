# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Parameter tables for management of the configures.
"""


class ParamTable:
    """Flatten parameter table."""

    def __init__(self, configs: dict, consts: dict):
        """Flatten parameter table.

        Args:
            configs: The parameter table.
            consts: The constant table.
        """
        self._flatt = flatten(configs)
        self._const = consts

    def get(self, key: str):
        val = self._flatt.get(key)
        return val

    def clone(self):
        return ParamTable(unflatten(self._flatt), self._const)


def flatten(configs, parent_key="", sep="."):
    items = {}
    for k, v in configs.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def unflatten(flattened, sep="."):
    out = {}
    for k, v in flattened.items():
        parts = k.split(sep)
        cur = out
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v
    return out
