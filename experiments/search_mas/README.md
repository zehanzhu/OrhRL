# Search MAS Training Experiment

This directory contains the canonical training assets for the OrchRL Search MAS experiment.

Scope:
- `train.yaml`: Hydra config for the Search MAS training run
- `run_train_e2e.sh`: preflight + train launcher
- `mas_apps/search/`: external inference-only MAS app invoked as a subprocess during rollout
- runtime outputs: `outputs/training_runs/<experiment_name>/<run_id>/checkpoints` and `outputs/training_runs/<experiment_name>/<run_id>/trajectories`

The active training config supports two specialization modes:

- `role_sharing`: multiple roles share one policy, so only one policy is trained
- `role_specific`: each role uses its own policy, so multiple policies are trained

For multi-policy `role_specific` training, `training.model_checkpoints_dir` is the shared checkpoint root, while each PPO trainer saves into its own subdirectory under that root, for example `outputs/training_runs/<experiment_name>/<run_id>/checkpoints/verifier_model/global_step_<n>/actor`.

The Search MAS application itself stays under `mas_apps/search/` because it is a first-class external MAS app and validation target. Training-specific assets live here to keep framework code, external app code, and experiment code separated.

At training startup, OrchRL resolves `training.run_dir`, `training.model_checkpoints_dir`, and `training.mate.trajectory_export.output_dir`, then creates those directories automatically. The launcher log file is also written under `outputs/logs/`.

Dataset paths are also centralized in `train.yaml`:

- `training.data_root_dir`
- `training.train_data_path`
- `training.val_data_path`
- `training.validate_batch_size`

By default, `training.train_data_path` resolves to `${training.data_root_dir}/train.parquet`, and `training.val_data_path` resolves to `${training.data_root_dir}/test.parquet`.

The MATE rollout path reads `training.mate.prompt_loader.path` for training. Validation reads `training.val_data_path` directly, forces parallel single-trajectory rollout, iterates over the full validation set in batches of `training.validate_batch_size`, and only reports `validation/sample_avg_reward`. It updates cumulative reward batch by batch and does not retain the full validation trajectory set in memory.
