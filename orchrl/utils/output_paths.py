from __future__ import annotations

from pathlib import Path
from typing import Any

from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf, open_dict


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
        "mas_work_dir": _resolve_output_path(root_dir, mate_cfg.get("mas_work_dir")),
        "mas_log_dir": _resolve_output_path(root_dir, mate_cfg.get("mas_log_dir")),
        "config_template_path": _resolve_output_path(
            root_dir, mate_cfg.get("config_template_path")
        ),
        "trajectory_output_dir": _resolve_output_path(
            root_dir, export_cfg.get("output_dir")
        ),
        "train_data_path": _resolve_output_path(root_dir, training_cfg.get("train_data_path")),
        "val_data_path": _resolve_output_path(root_dir, training_cfg.get("val_data_path")),
    }

    mkdir_keys = {
        "run_dir",
        "model_checkpoints_dir",
        "mas_work_dir",
        "mas_log_dir",
        "trajectory_output_dir",
    }
    for key, path in paths.items():
        if path is not None and key in mkdir_keys:
            path.mkdir(parents=True, exist_ok=True)

    _apply_resolved_paths_to_config(config, paths)
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


def _apply_resolved_paths_to_config(config: DictConfig, paths: dict[str, Path | None]) -> None:
    training_cfg = config.get("training")
    if training_cfg is None:
        return

    mate_cfg = training_cfg.get("mate") or {}
    export_cfg = mate_cfg.get("trajectory_export") or {}

    resolved_values = {
        "training.run_dir": paths.get("run_dir"),
        "training.model_checkpoints_dir": paths.get("model_checkpoints_dir"),
        "training.train_data_path": paths.get("train_data_path"),
        "training.val_data_path": paths.get("val_data_path"),
        "training.mate.mas_work_dir": paths.get("mas_work_dir"),
        "training.mate.mas_log_dir": paths.get("mas_log_dir"),
        "training.mate.config_template_path": paths.get("config_template_path"),
        "training.mate.trajectory_export.output_dir": paths.get("trajectory_output_dir"),
    }

    OmegaConf.set_struct(config, False)
    try:
        with open_dict(config):
            for key, value in resolved_values.items():
                if value is not None:
                    OmegaConf.update(config, key, str(value), merge=False)
    finally:
        OmegaConf.set_struct(config, True)
