# OrchRL 适配 VERL Megatron MoE 训练实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 OrchRL 在保持多智能体训练编排不变的前提下，支持 `GRPO + critic disabled + actor/ref Megatron + rollout vLLM` 的最小 Megatron MoE 训练闭环。

**Architecture:** 复用 VERL 的 `RayPPOTrainer` 与 `megatron_workers`，只在 OrchRL 的 trainer 入口层增加 backend-aware wiring。核心改动集中在 `orchrl/trainer/train.py` 的 worker 选择与 resource-pool role mapping，并补充对应测试与一个 Megatron 实验配置。

**Tech Stack:** Python, Hydra, OmegaConf, Ray, VERL, Megatron, unittest, pytest

---

## 文件结构

本次实现预计涉及以下文件。

**创建**
- `experiments/search_mas/train_megatron_moe_smoke.yaml`
- `docs/superpowers/specs/2026-04-28-orchrl-megatron-moe-design.md`（仅在需要同步术语时更新，默认不动）

**修改**
- `orchrl/trainer/train.py`
- `tests/test_multi_agents_trainer_refactor.py`
- `tests/test_training_output_layout.py`

**可能不改但需要参考**
- `orchrl/config/ppo_trainer/base.yaml`
- `experiments/search_mas/train.yaml`

**职责划分**
- `orchrl/trainer/train.py`：后端感知的 worker 选择、role 选择、resource-pool role mapping
- `tests/test_multi_agents_trainer_refactor.py`：新增 train 入口层的 backend wiring 单元测试
- `tests/test_training_output_layout.py`：新增 Megatron smoke 配置存在性与关键字段检查
- `experiments/search_mas/train_megatron_moe_smoke.yaml`：Megatron/MoE 最小实验配置，不污染现有 FSDP 默认路径

### Task 1: 为 Train 入口补充 backend-aware wiring 的失败测试

**Files:**
- Modify: `tests/test_multi_agents_trainer_refactor.py`
- Test: `tests/test_multi_agents_trainer_refactor.py`

- [ ] **Step 1: 写失败测试，锁定 worker 选择与 role 选择接口**

在 `tests/test_multi_agents_trainer_refactor.py` 的 `TrainConfigNormalizationTests` 类后追加以下测试：

```python
    def test_select_policy_worker_backend_uses_fsdp_actor_rollout_ref_role(self):
        from orchrl.trainer import train as train_module

        config = OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "actor": {"strategy": "fsdp"},
                    "model": {},
                    "ref": {},
                    "rollout": {},
                },
                "algorithm": {
                    "use_kl_in_reward": False,
                },
            }
        )

        with mock.patch.object(
            train_module,
            "_ray_remote_actor_worker",
            return_value="fsdp-worker",
        ):
            role_worker_mapping, actor_role = train_module._build_role_worker_mapping(config)

        self.assertEqual(actor_role, "actor_rollout_ref")
        self.assertEqual(role_worker_mapping["actor_rollout_ref"], "fsdp-worker")

    def test_select_policy_worker_backend_uses_megatron_actor_rollout_role(self):
        from orchrl.trainer import train as train_module

        config = OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "actor": {"strategy": "megatron"},
                    "model": {},
                    "ref": {},
                    "rollout": {},
                },
                "algorithm": {
                    "use_kl_in_reward": False,
                },
            }
        )

        with mock.patch.object(
            train_module,
            "_ray_remote_actor_worker",
            return_value="megatron-worker",
        ):
            role_worker_mapping, actor_role = train_module._build_role_worker_mapping(config)

        self.assertEqual(actor_role, "actor_rollout")
        self.assertEqual(role_worker_mapping["actor_rollout"], "megatron-worker")
```

- [ ] **Step 2: 再写失败测试，锁定 ref policy 的按需注册行为**

继续在同一测试文件追加：

```python
    def test_build_role_worker_mapping_registers_ref_policy_when_reference_needed(self):
        from orchrl.trainer import train as train_module

        config = OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "actor": {
                        "strategy": "megatron",
                        "use_kl_loss": True,
                    },
                    "model": {},
                    "ref": {},
                    "rollout": {},
                },
                "algorithm": {
                    "use_kl_in_reward": False,
                },
            }
        )

        with mock.patch.object(
            train_module,
            "_ray_remote_actor_worker",
            return_value="megatron-worker",
        ):
            role_worker_mapping, actor_role = train_module._build_role_worker_mapping(config)

        self.assertEqual(actor_role, "actor_rollout")
        self.assertEqual(role_worker_mapping["actor_rollout"], "megatron-worker")
        self.assertEqual(role_worker_mapping["ref"], "megatron-worker")

    def test_build_role_worker_mapping_skips_ref_policy_for_grpo_without_kl(self):
        from orchrl.trainer import train as train_module

        config = OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "actor": {
                        "strategy": "megatron",
                        "use_kl_loss": False,
                    },
                    "model": {},
                    "ref": {},
                    "rollout": {},
                },
                "algorithm": {
                    "use_kl_in_reward": False,
                },
            }
        )

        with mock.patch.object(
            train_module,
            "_ray_remote_actor_worker",
            return_value="megatron-worker",
        ):
            role_worker_mapping, actor_role = train_module._build_role_worker_mapping(config)

        self.assertEqual(actor_role, "actor_rollout")
        self.assertNotIn("ref", role_worker_mapping)
```

- [ ] **Step 3: 运行新增测试，确认当前实现失败**

Run:

```bash
pytest tests/test_multi_agents_trainer_refactor.py -k "build_role_worker_mapping or select_policy_worker_backend" -v
```

Expected:
- FAIL
- 报错类似 `AttributeError: module 'orchrl.trainer.train' has no attribute '_build_role_worker_mapping'`

- [ ] **Step 4: 提交失败测试**

```bash
git add tests/test_multi_agents_trainer_refactor.py
git commit -m "test: cover backend-aware policy worker wiring"
```

### Task 2: 在 train.py 实现 backend-aware worker 与 role 选择

**Files:**
- Modify: `orchrl/trainer/train.py`
- Test: `tests/test_multi_agents_trainer_refactor.py`

- [ ] **Step 1: 在 `orchrl/trainer/train.py` 增加 worker helper 导入与包装函数**

在文件顶部移除固定 FSDP worker 导入，并加入局部包装函数，代码形态如下：

```python
from verl.single_controller.ray import RayWorkerGroup
```

在 `install_cleanup_hooks()` 下方新增：

```python
def _ray_remote_actor_worker(worker_cls):
    return ray.remote(max_concurrency=2048)(worker_cls)


def _need_reference_policy(config: DictConfig) -> bool:
    actor_cfg = getattr(getattr(config, "actor_rollout_ref", None), "actor", None)
    algorithm_cfg = getattr(config, "algorithm", None)
    return bool(
        getattr(actor_cfg, "use_kl_loss", False)
        or getattr(algorithm_cfg, "use_kl_in_reward", False)
    )
```

- [ ] **Step 2: 新增 backend-aware role/worker 构造函数**

在 `train.py` 中新增：

```python
def _build_role_worker_mapping(config: DictConfig):
    from verl.trainer.ppo.ray_trainer import Role

    actor_cfg = getattr(getattr(config, "actor_rollout_ref", None), "actor", None)
    strategy = str(getattr(actor_cfg, "strategy", "fsdp"))

    if strategy in {"fsdp", "fsdp2"}:
        from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

        actor_role = Role.ActorRolloutRef
        worker_cls = AsyncActorRolloutRefWorker
    elif strategy == "megatron":
        from verl.workers.megatron_workers import AsyncActorRolloutRefWorker

        actor_role = Role.ActorRollout
        worker_cls = AsyncActorRolloutRefWorker
    else:
        raise ValueError(f"Unsupported actor strategy for OrchRL policy training: {strategy}")

    worker = _ray_remote_actor_worker(worker_cls)
    role_worker_mapping = {actor_role: worker}

    if _need_reference_policy(config):
        role_worker_mapping[Role.RefPolicy] = worker

    return role_worker_mapping, actor_role
```

- [ ] **Step 3: 在 `train_multi_agents()` 中改用新构造函数**

把现有：

```python
    role_worker_mapping = {
        Role.ActorRolloutRef: ray.remote(max_concurrency=2048)(AsyncActorRolloutRefWorker),
    }
```

替换为：

```python
    role_worker_mapping, actor_role = _build_role_worker_mapping(config)

    managers = _build_policy_resource_pool_managers(config, actor_role=actor_role)
```

同时删除旧的：

```python
    managers = _build_policy_resource_pool_managers(config)
```

- [ ] **Step 4: 运行 Task 1 的测试，确认通过**

Run:

```bash
pytest tests/test_multi_agents_trainer_refactor.py -k "build_role_worker_mapping or select_policy_worker_backend" -v
```

Expected:
- PASS

- [ ] **Step 5: 提交 backend-aware worker 选择实现**

```bash
git add orchrl/trainer/train.py tests/test_multi_agents_trainer_refactor.py
git commit -m "feat: select policy worker backend by actor strategy"
```

### Task 3: 为 resource-pool role mapping 补充失败测试

**Files:**
- Modify: `tests/test_multi_agents_trainer_refactor.py`
- Test: `tests/test_multi_agents_trainer_refactor.py`

- [ ] **Step 1: 写失败测试，覆盖 critic disabled 与 actor role 动态映射**

在 `tests/test_multi_agents_trainer_refactor.py` 中追加：

```python
    def test_build_policy_resource_pool_managers_uses_megatron_actor_rollout_role(self):
        from orchrl.trainer.train import _build_policy_resource_pool_managers
        from verl.trainer.ppo.ray_trainer import Role

        config = OmegaConf.create(
            {
                "models": {"model_0": {"name": "policy_a"}},
                "resource": {
                    "n_gpus_per_node": 8,
                    "policy_nnodes": 1,
                    "gpus_per_policy": 8,
                },
                "critic": {"enable": False},
                "actor_rollout_ref": {
                    "actor": {"use_kl_loss": False},
                },
                "algorithm": {"use_kl_in_reward": False},
            }
        )

        managers = _build_policy_resource_pool_managers(config, actor_role=Role.ActorRollout)

        self.assertEqual(len(managers), 1)
        self.assertIn(Role.ActorRollout, managers[0].mapping)
        self.assertNotIn(Role.Critic, managers[0].mapping)
        self.assertNotIn(Role.RefPolicy, managers[0].mapping)
```

- [ ] **Step 2: 再写失败测试，覆盖需要 ref policy 时的 role 注册**

继续追加：

```python
    def test_build_policy_resource_pool_managers_registers_ref_policy_when_needed(self):
        from orchrl.trainer.train import _build_policy_resource_pool_managers
        from verl.trainer.ppo.ray_trainer import Role

        config = OmegaConf.create(
            {
                "models": {"model_0": {"name": "policy_a"}},
                "resource": {
                    "n_gpus_per_node": 8,
                    "policy_nnodes": 1,
                    "gpus_per_policy": 8,
                },
                "critic": {"enable": False},
                "actor_rollout_ref": {
                    "actor": {"use_kl_loss": True},
                },
                "algorithm": {"use_kl_in_reward": False},
            }
        )

        managers = _build_policy_resource_pool_managers(config, actor_role=Role.ActorRollout)

        self.assertEqual(managers[0].mapping[Role.ActorRollout], "global_pool_model_0")
        self.assertEqual(managers[0].mapping[Role.RefPolicy], "global_pool_model_0")
```

- [ ] **Step 3: 运行新增测试，确认当前实现失败**

Run:

```bash
pytest tests/test_multi_agents_trainer_refactor.py -k "build_policy_resource_pool_managers" -v
```

Expected:
- FAIL
- 因为当前 `_build_policy_resource_pool_managers()` 还没有 `actor_role` 参数，也没有动态 mapping

- [ ] **Step 4: 提交失败测试**

```bash
git add tests/test_multi_agents_trainer_refactor.py
git commit -m "test: cover dynamic resource pool role mapping"
```

### Task 4: 实现 resource-pool 的动态 role mapping

**Files:**
- Modify: `orchrl/trainer/train.py`
- Test: `tests/test_multi_agents_trainer_refactor.py`

- [ ] **Step 1: 为 `train.py` 增加角色需求判断函数**

在 `train.py` 中新增：

```python
def _critic_enabled(config: DictConfig) -> bool:
    critic_cfg = getattr(config, "critic", None)
    return bool(getattr(critic_cfg, "enable", False))
```

- [ ] **Step 2: 改造 `_build_policy_resource_pool_managers()` 签名和 mapping 逻辑**

把函数签名改成：

```python
def _build_policy_resource_pool_managers(config, *, actor_role):
```

并把内部固定 mapping：

```python
        mapping = {
            Role.ActorRolloutRef: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }
```

替换为：

```python
        mapping = {
            actor_role: global_pool_id,
        }
        if _need_reference_policy(config):
            mapping[Role.RefPolicy] = global_pool_id
        if _critic_enabled(config):
            mapping[Role.Critic] = global_pool_id
```

- [ ] **Step 3: 运行 resource-pool 测试，确认通过**

Run:

```bash
pytest tests/test_multi_agents_trainer_refactor.py -k "build_policy_resource_pool_managers" -v
```

Expected:
- PASS

- [ ] **Step 4: 回归运行 train 入口相关测试**

Run:

```bash
pytest tests/test_multi_agents_trainer_refactor.py -k "train_multi_agents_only_expands_in_special_case or build_role_worker_mapping or build_policy_resource_pool_managers" -v
```

Expected:
- PASS

- [ ] **Step 5: 提交动态 resource-pool 实现**

```bash
git add orchrl/trainer/train.py tests/test_multi_agents_trainer_refactor.py
git commit -m "feat: build policy resource pools from active roles"
```

### Task 5: 为 Megatron/MoE smoke 配置补充失败测试

**Files:**
- Modify: `tests/test_training_output_layout.py`
- Test: `tests/test_training_output_layout.py`

- [ ] **Step 1: 写失败测试，要求存在 Megatron smoke 配置文件**

在 `tests/test_training_output_layout.py` 中追加：

```python
    def test_search_mas_megatron_smoke_config_exists(self):
        repo_root = Path(__file__).resolve().parents[1]
        config_path = repo_root / "experiments/search_mas/train_megatron_moe_smoke.yaml"
        self.assertTrue(config_path.is_file(), f"Missing config: {config_path}")
```

- [ ] **Step 2: 写失败测试，要求 Megatron smoke 配置具备关键字段**

继续追加：

```python
    def test_search_mas_megatron_smoke_config_sets_grpo_megatron_and_disables_critic(self):
        repo_root = Path(__file__).resolve().parents[1]
        config_text = (
            repo_root / "experiments/search_mas/train_megatron_moe_smoke.yaml"
        ).read_text(encoding="utf-8")

        self.assertIn("specialization: role_sharing", config_text)
        self.assertIn("strategy: megatron", config_text)
        self.assertIn("critic:", config_text)
        self.assertIn("enable: false", config_text)
        self.assertIn("expert_model_parallel_size:", config_text)
        self.assertIn("tensor_model_parallel_size:", config_text)
        self.assertIn("pipeline_model_parallel_size:", config_text)
```

- [ ] **Step 3: 运行测试，确认当前失败**

Run:

```bash
pytest tests/test_training_output_layout.py -k "megatron_smoke_config" -v
```

Expected:
- FAIL
- 因为配置文件尚不存在

- [ ] **Step 4: 提交失败测试**

```bash
git add tests/test_training_output_layout.py
git commit -m "test: require Megatron MoE smoke config"
```

### Task 6: 新增 Megatron/MoE smoke 配置

**Files:**
- Create: `experiments/search_mas/train_megatron_moe_smoke.yaml`
- Test: `tests/test_training_output_layout.py`

- [ ] **Step 1: 新建最小 Megatron/MoE smoke 配置**

创建 `experiments/search_mas/train_megatron_moe_smoke.yaml`，初版内容如下：

```yaml
defaults:
  - train
  - _self_

specialization: role_sharing

training:
  experiment_name: search_mas_megatron_moe_smoke

models:
  model_0:
    name: verifier_model
    ppo_trainer_config:
      actor_rollout_ref:
        actor:
          _target_: verl.workers.config.McoreActorConfig
          strategy: megatron
          use_kl_loss: false
          ppo_mini_batch_size: 16
          ppo_micro_batch_size_per_gpu: 1
          ppo_max_token_len_per_gpu: 4096
          ppo_epochs: 1
          megatron:
            _target_: verl.workers.config.McoreEngineConfig
            strategy: megatron
            dtype: bfloat16
            tensor_model_parallel_size: 2
            pipeline_model_parallel_size: 1
            virtual_pipeline_model_parallel_size: null
            context_parallel_size: 1
            expert_model_parallel_size: 2
            expert_tensor_parallel_size: null
            sequence_parallel: true
            use_distributed_optimizer: true
            use_dist_checkpointing: false
            dist_checkpointing_path: null
            dist_checkpointing_prefix: ""
            param_offload: false
            grad_offload: false
            optimizer_offload: false
            seed: 42
            override_ddp_config: {}
            override_transformer_config: {}
            override_mcore_model_config: {}
            use_mbridge: true
            vanilla_mbridge: true
            use_remove_padding: true
            forward_only: false
            router_replay:
              mode: disabled
        ref:
          _target_: verl.workers.config.McoreActorConfig
          strategy: megatron
          log_prob_micro_batch_size_per_gpu: 1
          log_prob_max_token_len_per_gpu: 4096
          log_prob_use_dynamic_bsz: false
          megatron:
            seed: 42
            tensor_model_parallel_size: 2
            pipeline_model_parallel_size: 1
            virtual_pipeline_model_parallel_size: null
            context_parallel_size: 1
            expert_model_parallel_size: 2
            expert_tensor_parallel_size: null
            param_offload: false
            override_transformer_config: {}
            use_mbridge: true
            vanilla_mbridge: true
            use_remove_padding: true
            forward_only: true
        rollout:
          name: vllm
        model:
          path: Qwen/Qwen3-8B

critic:
  enable: false

algorithm:
  adv_estimator: grpo
  use_kl_in_reward: false
```

- [ ] **Step 2: 运行配置测试，确认通过**

Run:

```bash
pytest tests/test_training_output_layout.py -k "megatron_smoke_config" -v
```

Expected:
- PASS

- [ ] **Step 3: 提交 Megatron smoke 配置**

```bash
git add experiments/search_mas/train_megatron_moe_smoke.yaml tests/test_training_output_layout.py
git commit -m "feat: add Megatron MoE smoke training config"
```

### Task 7: 补充 train 入口的集成式单元测试

**Files:**
- Modify: `tests/test_multi_agents_trainer_refactor.py`
- Test: `tests/test_multi_agents_trainer_refactor.py`

- [ ] **Step 1: 写失败测试，验证 `train_multi_agents()` 在 Megatron GRPO 配置下选择 `Role.ActorRollout`**

在 `tests/test_multi_agents_trainer_refactor.py` 中追加：

```python
    def test_train_multi_agents_uses_megatron_actor_rollout_role_for_grpo(self):
        from orchrl.trainer import train as train_module
        from verl.trainer.ppo.ray_trainer import Role

        config = OmegaConf.create(
            {
                "specialization": "role_sharing",
                "resource": {"n_gpus_per_node": 1, "nnodes": 1},
                "agent_policy_configs": {
                    "agent_configs": {
                        "agent_0": {"name": "verifier", "policy_name": "policy_a"}
                    }
                },
                "base_models": {
                    "policy_0": {"path": "/models/base", "name": "policy_a"}
                },
                "models": {
                    "model_0": {
                        "path": "/models/base",
                        "name": "policy_a",
                        "ppo_trainer_config": {
                            "actor_rollout_ref": {
                                "actor": {"strategy": "megatron", "use_kl_loss": False},
                                "ref": {},
                                "rollout": {},
                                "model": {"path": "/models/base"},
                            }
                        },
                    }
                },
                "critic": {"enable": False},
                "algorithm": {"use_kl_in_reward": False},
            }
        )

        captured = {}

        class _FakeTrainer:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def init_workers(self):
                return None

            def init_mate_rollout_runtime(self):
                return None

            def fit(self):
                return None

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(train_module, "_patch_verl_reward_loop_for_external_mas"))
            stack.enter_context(mock.patch.object(train_module, "_validate_unique_role_specific_served_model_names"))
            stack.enter_context(mock.patch.object(train_module, "_build_policy_resource_pool_managers", return_value=["pool-0"]))
            stack.enter_context(mock.patch.object(train_module, "_build_role_worker_mapping", return_value=({Role.ActorRollout: "megatron-worker"}, Role.ActorRollout)))
            stack.enter_context(mock.patch.object(train_module, "MultiAgentsPPOTrainer", _FakeTrainer))
            stack.enter_context(mock.patch("verl.utils.fs.copy_local_path_from_hdfs", side_effect=lambda path: path))
            stack.enter_context(mock.patch("verl.utils.hf_tokenizer", side_effect=lambda path, trust_remote_code=False: f"tok::{path}"))
            train_module.train_multi_agents(config)

        self.assertEqual(captured["role_worker_mapping"], {Role.ActorRollout: "megatron-worker"})
```

- [ ] **Step 2: 运行测试，确认失败**

Run:

```bash
pytest tests/test_multi_agents_trainer_refactor.py -k "uses_megatron_actor_rollout_role_for_grpo" -v
```

Expected:
- 先 FAIL，直到 `train_multi_agents()` 完成新 helper 接入且测试 patch 点对齐

- [ ] **Step 3: 调整测试或实现，直至通过**

本步骤不新增新代码块，直接用前面任务已实现的 helper 接口对齐当前测试。

Run:

```bash
pytest tests/test_multi_agents_trainer_refactor.py -k "uses_megatron_actor_rollout_role_for_grpo" -v
```

Expected:
- PASS

- [ ] **Step 4: 提交 train 入口集成测试**

```bash
git add tests/test_multi_agents_trainer_refactor.py
git commit -m "test: cover Megatron GRPO train entry wiring"
```

### Task 8: 运行完整回归测试并整理结果

**Files:**
- Modify: `orchrl/trainer/train.py`
- Modify: `tests/test_multi_agents_trainer_refactor.py`
- Modify: `tests/test_training_output_layout.py`
- Create: `experiments/search_mas/train_megatron_moe_smoke.yaml`

- [ ] **Step 1: 运行 train 相关单元测试集合**

Run:

```bash
pytest tests/test_multi_agents_trainer_refactor.py tests/test_training_output_layout.py -v
```

Expected:
- PASS

- [ ] **Step 2: 运行更聚焦的 smoke 配置文本校验**

Run:

```bash
pytest tests/test_training_output_layout.py -k "megatron_smoke_config or search_mas_training_config_defines_run_scoped_output_layout" -v
```

Expected:
- PASS

- [ ] **Step 3: 检查 git diff，确认未误改无关文件**

Run:

```bash
git status --short
```

Expected:
- 仅包含以下文件：
  - `orchrl/trainer/train.py`
  - `tests/test_multi_agents_trainer_refactor.py`
  - `tests/test_training_output_layout.py`
  - `experiments/search_mas/train_megatron_moe_smoke.yaml`
  - 可选的计划/文档文件

- [ ] **Step 4: 提交最终实现**

```bash
git add orchrl/trainer/train.py tests/test_multi_agents_trainer_refactor.py tests/test_training_output_layout.py experiments/search_mas/train_megatron_moe_smoke.yaml
git commit -m "feat: add Megatron MoE training backend wiring for OrchRL"
```

## 自检

### Spec coverage

本计划覆盖了 spec 中的所有实现要求：

- backend-aware worker 选择：Task 1-2
- 动态 ref 注册：Task 1-2
- 动态 resource-pool mapping：Task 3-4
- 保持 `PolicyTrainerRegistry` / `TrainingStepExecutor` 基本不变：通过计划范围控制，未安排重构任务
- Megatron smoke 配置：Task 5-6
- `critic disabled`：Task 3、Task 6、Task 7
- `role_sharing` 优先验证：Task 6-7
- 回归与验证：Task 8

### Placeholder scan

已检查并避免：

- 没有 `TBD`、`TODO`
- 没有“自行处理异常”这类空洞描述
- 所有代码修改步骤都提供了具体代码块
- 所有测试步骤都提供了明确命令和预期

### Type consistency

计划中使用的关键函数与名称保持一致：

- `_ray_remote_actor_worker`
- `_need_reference_policy`
- `_critic_enabled`
- `_build_role_worker_mapping`
- `_build_policy_resource_pool_managers(config, *, actor_role)`

这些名字在各任务中前后一致，没有重名漂移。
