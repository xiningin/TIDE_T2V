from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

# Matches ``${VAR}`` and ``${VAR:-default}``. Only the braced form is expanded so
# that a lone ``$`` inside prompt text or negative terms is never touched.
_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env_vars(value: Any) -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:-default}`` in strings.

    Unset variables without a default are left verbatim so a misconfiguration is
    visible (for example an unresolved ``${SANA_CKPT_DIR}`` path) rather than
    silently collapsing to an empty string.
    """

    if isinstance(value, str):
        def _replace(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            env_value = os.environ.get(name)
            if env_value is not None:
                return env_value
            if default is not None:
                return default
            return match.group(0)

        return _ENV_VAR_PATTERN.sub(_replace, value)
    if isinstance(value, dict):
        return {key: expand_env_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return expand_env_vars(data)


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    package_root = Path(__file__).resolve().parents[2]
    default_path = package_root / "config" / "default.yaml"
    config = load_yaml(default_path)
    if path:
        config = deep_update(config, load_yaml(path))
    return config


def load_prompt(name: str) -> str:
    package_root = Path(__file__).resolve().parents[2]
    prompt_path = package_root / "prompts" / name
    return prompt_path.read_text(encoding="utf-8")


def render_template(template: str, **variables: Any) -> str:
    rendered = template
    for key, value in variables.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    return rendered


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out
