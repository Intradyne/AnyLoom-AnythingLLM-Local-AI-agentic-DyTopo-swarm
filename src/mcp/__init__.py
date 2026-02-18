# Make this local mcp package coexist with the pip-installed `mcp` package.
# We import everything from the pip mcp package and extend __path__ so that
# submodules like mcp.types, mcp.server, etc. resolve to the pip package,
# while mcp.system_status resolves to our local module.
import importlib
import sys
import os

# Temporarily remove our parent from sys.path so we can import the real mcp
_this_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_this_dir)

# Save and remove any sys.path entries that would resolve to this package
_removed = []
for _i in range(len(sys.path) - 1, -1, -1):
    try:
        _p = os.path.abspath(sys.path[_i])
    except (ValueError, OSError):
        continue
    if _p == _src_dir:
        _removed.append((_i, sys.path.pop(_i)))

# Remove ourselves from sys.modules so the real mcp can load
_self_module = sys.modules.pop(__name__, None)
_child_modules = {k: sys.modules.pop(k) for k in list(sys.modules) if k.startswith(__name__ + ".")}

# Import the real mcp package
_real_mcp = importlib.import_module("mcp")

# Restore sys.path
for _i, _p in sorted(_removed):
    sys.path.insert(_i, _p)

# Restore ourselves into sys.modules
sys.modules[__name__] = _self_module
for k, v in _child_modules.items():
    sys.modules[k] = v

# Copy the real mcp's namespace into ours
_real_attrs = {k: v for k, v in vars(_real_mcp).items() if not k.startswith("_")}
globals().update(_real_attrs)

# Extend __path__ to include the real mcp package path so submodule imports work
__path__ = [_this_dir] + list(_real_mcp.__path__)
