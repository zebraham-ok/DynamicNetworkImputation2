"""
Training config loader utility
Usage: from Training.config_loader import load_config
       cfg = load_config('gatgru_vec')

Load order: Training/common_config.yaml (shared) <- Models/configs/{model}.yaml (model-specific)
"""

import os
import yaml
from utils import deep_merge


# Keys that used to steer a run but are no longer read by anything. They are reported on every load
# so a stale config cannot silently keep a setting that does nothing (earlier failure mode: the same
# quantity written under two key paths, with only one of them consumed).
DEPRECATED_KEYS = {
    ('trainer', 'patience'):
        'no longer read',
    ('trainer', 'kwargs', 'patience'):
        'no longer read',
}
DEPRECATED_KEY_HINT = 'the early-stopping budget is trainer.selection.patience only'


def _warn_on_deprecated_keys(cfg):
    """Print one warning per deprecated key path still present in the merged config."""
    for path, reason in DEPRECATED_KEYS.items():
        node = cfg
        for key in path[:-1]:
            node = node.get(key) if isinstance(node, dict) else None
            if not isinstance(node, dict):
                node = None
                break
        if isinstance(node, dict) and path[-1] in node:
            dotted = '.'.join(path)
            print(f"  [warn] config key '{dotted}' is {reason} "
                  f"(value {node[path[-1]]!r} ignored): {DEPRECATED_KEY_HINT}.")


def load_config(config_name):
    """Load model config: Training/common_config.yaml (shared) <- Models/configs/{model}.yaml (model-specific)"""
    training_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(training_dir)

    common_path = os.path.join(training_dir, 'common_config.yaml')
    with open(common_path, 'r', encoding='utf-8') as f:
        common_cfg = yaml.safe_load(f)

    # Model-specific config (Models/configs/)
    model_config_dir = os.path.join(root_dir, 'Models', 'configs')
    yaml_path = os.path.join(model_config_dir, f"{config_name}.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r', encoding='utf-8') as f:
            model_cfg = yaml.safe_load(f)
    elif os.path.exists(config_name):
        with open(config_name, 'r', encoding='utf-8') as f:
            model_cfg = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Model config not found: {config_name} (looked in {model_config_dir})")

    merged = deep_merge(common_cfg, model_cfg)
    _warn_on_deprecated_keys(merged)
    return merged
