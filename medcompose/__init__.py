"""Public exports, loaded only when the requested backend is used."""
from importlib import import_module

_EXPORTS = {'visionzip': ('.visionzip', None), 'unicomp': ('.unicomp', None), 'volucompose': ('.volucompose', None), 'holitom': ('.holitom', None)}

__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = module if attribute is None else getattr(module, attribute)
    globals()[name] = value
    return value
