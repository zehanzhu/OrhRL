from __future__ import annotations

from pathlib import Path
from typing import Any

from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf


def prepare_training_output_dirs(
    config: DictConfig,
    *,
    base_dir: str | Path | None = None,
) -> dict[str, Path]:
    OmegaConf.resolve(config)

    training_cfg = config.get("training")
    if training_cfg is None:
        return {}

    root_dir = _resolve_base_dir(base_dir)
    mate_cfg = training_cfg.get("mate") or {}
    export_cfg = mate_cfg.get("trajectory_export") or {}

    paths = {
        "run_dir": _resolve_output_path(root_dir, training_cfg.get("run_dir")),
        "model_checkpoints_dir": _resolve_output_path(
            root_dir, training_cfg.get("model_checkpoints_dir")
        ),
        "trajectory_output_dir": _resolve_output_path(
            root_dir, export_cfg.get("output_dir")
        ),
    }

    for path in paths.values():
        if path is not None:
            path.mkdir(parents=True, exist_ok=True)

    return {key: value for key, value in paths.items() if value is not None}


def _resolve_base_dir(base_dir: str | Path | None) -> Path:
    if base_dir is not None:
        return Path(base_dir).expanduser().resolve()
    try:
        return Path(get_original_cwd()).expanduser().resolve()
    except Exception:
        return Path.cwd().expanduser().resolve()


def _resolve_output_path(base_dir: Path, raw_value: Any) -> Path | None:
    if raw_value is None:
        return None

    value = str(raw_value).strip()
    if not value:
        return None

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()
