try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("yunmeng")
except PackageNotFoundError:
    import os

    cur_path = os.path.abspath(os.path.dirname(__file__))
    ver_path = os.path.dirname(cur_path)
    ver_file = os.path.join(ver_path, "VERSION")

    if not os.path.exists(ver_file):
        __version__ = "dev"
    else:
        with open(ver_file, "r") as f:
            __version__ = f.read().strip()
