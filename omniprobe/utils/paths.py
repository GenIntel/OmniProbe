
import os
from typing import Any


def require_env_path(env_var: str, description: str) -> str:
    value = os.environ.get(env_var)
    if value:
        return value
    raise RuntimeError(f"Set {env_var} to the {description}.")


def cfg_or_env_path(
    cfg: Any, key: str, env_var: str, description: str
) -> str:
    value = None
    if cfg is not None and hasattr(cfg, "get"):
        value = cfg.get(key)
    if value:
        return str(value)
    return require_env_path(env_var, description)
