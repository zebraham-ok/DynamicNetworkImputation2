"""
Training config loader utility
Usage: from Training.config_loader import load_config
       cfg = load_config('gatgru_vec')

Load order: Training/common_config.yaml (shared) <- Models/configs/{model}.yaml (model-specific)
"""

import os
import yaml
from utils import deep_merge


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

    return deep_merge(common_cfg, model_cfg)
