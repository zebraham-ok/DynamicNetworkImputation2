"""
Shared utility functions: deep merge, dynamic import, auto-argument resolution, device resolution.
"""
import importlib
import torch
import yaml
import os


def deep_merge(base, override):
    """Deep-merge two dicts; override takes precedence over base for keys with the same name"""
    result = {}
    all_keys = set(base.keys()) | set(override.keys())
    for key in all_keys:
        if key in override and key in base and isinstance(base[key], dict) and isinstance(override[key], dict):
            result[key] = deep_merge(base[key], override[key])
        elif key in override:
            result[key] = override[key]
        else:
            result[key] = base[key]
    return result


def import_attr(module_path, attr_name):
    """Dynamically import module.attr"""
    mod = importlib.import_module(module_path)
    return getattr(mod, attr_name)


def resolve_auto_kwargs(kwargs, context):
    """
    Resolve the auto values in kwargs:
    - "auto" string -> look up the same-named key in context
    - ["auto", 128, 64] -> the first list item is replaced by the corresponding value
    - "torch.nn.ReLU()" etc. -> evaluate
    """
    resolved = {}
    for k, v in kwargs.items():
        if v == "auto":
            resolved[k] = context.get(k, None)
        elif isinstance(v, list) and len(v) > 0 and v[0] == "auto":
            resolved[k] = [context.get(k, None)] + v[1:]
        elif isinstance(v, str) and v.startswith("torch."):
            resolved[k] = eval(v)
        else:
            resolved[k] = v
    return resolved


def resolve_device(device_str):
    """Resolve the device string; auto means automatic selection"""
    if device_str in (None, 'auto'):
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device_str
