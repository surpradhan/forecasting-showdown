from pathlib import Path

import yaml

_CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def load_config(model_name: str) -> dict:
    """Return merged config: _base.yaml values overridden by <model_name>.yaml."""
    base = _load_yaml(_CONFIGS_DIR / "_base.yaml")
    model = _load_yaml(_CONFIGS_DIR / f"{model_name}.yaml")
    return {**base, **model}


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}
