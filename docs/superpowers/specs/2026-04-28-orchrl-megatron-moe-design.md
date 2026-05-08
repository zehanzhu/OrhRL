# OrchRL 适配 VERL Megatron MoE 训练设计

## 概述

本设计的目标，是在不改动 OrchRL 多智能体编排主流程的前提下，复用 VERL 现有的 Megatron PPO worker 栈，为 OrchRL 增加一条基于 Megatron 的 MoE 训练引擎路径。

本次范围刻意收窄为最小闭环：

- 算法：`GRPO`
- Critic：关闭
- 训练后端：`Megatron`
- Reference policy：通过 Megatron ref worker 或 actor-backed ref 路径支持
- Rollout 后端：保持不变，继续使用 `vLLM`
- 多智能体编排：保持 OrchRL 现有实现

这次工作是“后端接线适配”，不是重写 PPO、GRPO，也不是重写多智能体 rollout 流程。

## 目标

- 让 OrchRL 的 policy training 能够使用 VERL 现成的 Megatron worker 栈，以支持 MoE 模型训练。
- 保持 OrchRL 当前的多智能体控制流和 policy trainer registry 结构。
- 先在较小范围内验证单 policy 路径，再支持 `role_sharing` 和 `role_specific` 两种 specialization 模式。
- 首版实现仅覆盖 GRPO 下的 actor/ref 训练路径。

## 非目标

- 不实现 Megatron critic 训练。
- 不替换 VERL 的 PPO 或 GRPO 逻辑。
- 不替换 OrchRL 的 MATE rollout 编排。
- v1 不引入 Megatron 原生 rollout serving。
- 首版不解决所有 LoRA、router replay、定制 checkpoint 转换场景。

## 现状

### OrchRL 侧

OrchRL 当前将 policy backend 写死为 FSDP worker 实现：

- `orchrl/trainer/train.py` 直接导入 `verl.workers.fsdp_workers.AsyncActorRolloutRefWorker`
- `orchrl/trainer/train.py` 直接注册 `Role.ActorRolloutRef`
- resource-pool mapping 固定包含 `Role.ActorRolloutRef`、`Role.Critic`、`Role.RefPolicy`，没有后端感知能力

但 OrchRL 的多智能体外层本身其实是后端无关的：

- `PolicyTrainerRegistry` 负责实例化 `RayPPOTrainer`
- `TrainingStepExecutor` 通过抽象接口与底层 worker 交互，例如：
  - `compute_log_prob`
  - `compute_ref_log_prob`
  - `update_actor`

对于 `critic disabled` 的 GRPO 路径，OrchRL 已经通过 `ppo_trainer.use_critic` 对 critic 相关逻辑做了分支保护，因此现有训练步执行器本身可以工作在 actor-only 模式。

### VERL 侧

VERL 已经提供了完整的 Megatron 训练路径：

- `verl.trainer.main_ppo` 在 `actor.strategy == megatron` 时选择 Megatron workers
- `verl.workers.megatron_workers` 中包含：
  - `AsyncActorRolloutRefWorker`
  - `CriticWorker`
  - `MegatronWorker`
- `verl.workers.actor.megatron_actor.MegatronPPOActor` 实现 PPO actor update 逻辑
- `verl.utils.checkpoint.megatron_checkpoint_manager.MegatronCheckpointManager` 实现 Megatron 分布式 checkpoint 管理

VERL 的 Megatron worker 栈已覆盖：

- 分布式初始化
- TP/PP/EP/CP 模型并行初始化
- actor 模型构建
- reference policy 构建
- actor log-prob 计算
- ref log-prob 计算
- actor 更新
- Megatron 分布式 checkpoint 保存和恢复

## 问题定义

OrchRL 目前无法暴露 VERL 的 Megatron backend，根本原因不是配置里缺一个 `strategy: megatron`，而是 OrchRL 的 backend 注册路径带有固定的 FSDP 假设。

最关键的不匹配点是 role wiring：

- OrchRL 当前默认假设使用 `Role.ActorRolloutRef`
- VERL 的 legacy Megatron worker 流通常是通过 `Role.ActorRollout` 接入
- 独立的 reference policy worker 只在需要时注册，且仅在 ref 不与 actor 融合时才单独存在

因此，只改配置中的 strategy 字段不够，OrchRL 必须适配其 backend 选择和 role mapping 逻辑。

## 推荐方案

推荐采用“薄适配层”方案：

- 保留 OrchRL 的多智能体编排
- 保留 VERL 的 `RayPPOTrainer`
- 保留 VERL 的 Megatron actor/ref worker 实现
- 仅替换 OrchRL 中写死的 FSDP 假设，改为 backend-aware wiring

相比于复制 `verl.trainer.main_ppo` 的逻辑，或者新写一套 trainer，这种方案与上游 VERL 的行为偏差最小，后续维护成本也最低。

## 详细设计

### 1. Backend-Aware Policy Worker Selection

在 `orchrl/trainer/train.py` 中增加基于 backend 的 worker 与 role 选择逻辑。

规则如下：

- 如果 `actor.strategy in {fsdp, fsdp2}`：
  - worker class: `verl.workers.fsdp_workers.AsyncActorRolloutRefWorker`
  - actor role: `Role.ActorRolloutRef`
- 如果 `actor.strategy == megatron`：
  - worker class: `verl.workers.megatron_workers.AsyncActorRolloutRefWorker`
  - actor role: `Role.ActorRollout`

这样做的原因，是对齐 VERL 现有 legacy worker 的行为，避免强行把 Megatron backend 塞进 FSDP 的 role 结构中。

### 2. 动态 Reference Policy 注册

Reference policy 的注册必须具备 backend 感知能力，并且按需启用。

规则如下：

- 如果当前不需要 reference policy，则不注册 `Role.RefPolicy`
- 如果需要 reference policy 且 `ref_in_actor == false`，则注册 `Role.RefPolicy`
- 如果需要 reference policy 且 `ref_in_actor == true`，则允许 actor worker 直接提供 reference 行为

针对本次 GRPO 首版目标：

- `algorithm.use_kl_in_reward = false`
- `actor.use_kl_loss = false`

这意味着初始实验里 reference policy 很可能不是强依赖项，但 OrchRL 仍应把 Megatron ref 路径接好，因为后续 GRPO 变体可能会启用 KL 相关项。

### 3. 动态 Resource-Pool Mapping

`orchrl/trainer/train.py` 中的 resource-pool 注册逻辑，必须从“固定角色集合”改为“按当前启用角色集合”驱动。

规则如下：

- 始终注册当前解析出的 actor role
- 仅在需要时注册 `Role.RefPolicy`
- 当 `critic.enable == false` 时，不注册 `Role.Critic`

在当前设计下，最小激活角色集合为：

- actor role
- 可选的 ref policy role

这样可以避免创建无用 worker，也能避免与 `RayPPOTrainer.use_critic` 的状态不一致。

### 4. 保持 PolicyTrainerRegistry 和 TrainingStepExecutor 基本不变

`orchrl/trainer/policy_trainer_registry.py` 结构上不应做大改。

原因：

- 它已经把每个 policy trainer 视为 `RayPPOTrainer`
- 它本身并不依赖 FSDP 专有方法
- 只要 config 与 role wiring 合法，它既可以承载 FSDP-backed 的 VERL trainer，也可以承载 Megatron-backed 的 VERL trainer

`orchrl/trainer/training_step_executor.py` 在 v1 也应尽量保持结构不变。

原因：

- 它通过后端无关的抽象接口工作
- `critic disabled` 的 GRPO 路径会自然跳过 critic 逻辑
- actor/ref 所需方法已经在 VERL 的 Megatron worker 中实现

### 5. Megatron Actor/Ref 配置结构

OrchRL 必须支持基于 VERL `McoreActorConfig` 和 `McoreEngineConfig` 的 Megatron-compatible policy trainer config。

actor 配置至少应使用：

- `_target_: verl.workers.config.McoreActorConfig`
- `strategy: megatron`
- `megatron:`
  - `tensor_model_parallel_size`
  - `pipeline_model_parallel_size`
  - `virtual_pipeline_model_parallel_size`
  - `context_parallel_size`
  - `expert_model_parallel_size`
  - `expert_tensor_parallel_size`
  - `sequence_parallel`
  - `use_distributed_optimizer`
  - `use_dist_checkpointing`
  - `dist_checkpointing_path`
  - `dist_checkpointing_prefix`
  - `param_offload`
  - `grad_offload`
  - `optimizer_offload`
  - `seed`
  - `override_ddp_config`
  - `override_transformer_config`
  - `override_mcore_model_config`
  - `use_mbridge`
  - `vanilla_mbridge`
  - `use_remove_padding`
  - `forward_only`
  - `router_replay`

ref 配置也必须使用 `McoreActorConfig`，并满足：

- `strategy: megatron`
- `forward_only: true`
- TP/PP/EP 与 actor 保持镜像一致

ref 的并行布局必须与 actor 一致，否则 log-prob 计算可能落在不兼容的分布式权重切分上。

### 6. Critic 按设计关闭

本设计明确假设：

- `critic.enable: false`

原因是本次目标算法为 GRPO，且当前使用场景不需要 value model。

这一点必须在以下位置保持一致：

- OrchRL 配置
- resource-pool 注册逻辑
- smoke test 方案

因此，本设计不要求实现 Megatron critic 支持。

### 7. Rollout Backend 保持 vLLM

v1 中 rollout 引擎保持不变：

- rollout backend: `vllm`
- OrchRL 的 MATE rollout 路径保持不变

这样可以把适配范围收敛在“训练引擎”本身，避免首版引入过多变量。

### 8. Specialization Mode 的验证顺序

实现和验证应分两阶段推进：

1. `role_sharing` 单 policy 冒烟路径
2. `role_specific` 多 policy 路径

这样做更稳。

原因：

- 单 policy 先验证 worker 选择、actor/ref role wiring、Megatron update path
- 多 policy 会额外引入多个 policy trainer、资源隔离、checkpoint 目录布局等复杂度

## 配置策略建议

首版实现不应直接把 `orchrl/config/ppo_trainer/base.yaml` 的默认后端替换为 Megatron。

更稳的做法是：

- 保留 `base.yaml` 继续服务 FSDP 路径
- 新增一个 Megatron 专用 override 或实验配置，用于 MoE 训练

这样可以获得：

- 更安全的 rollout 方式
- 更清晰的 backend-specific 调参边界
- 更容易回退到当前已有的 FSDP 路径

## 验证计划

### 阶段 1：配置校验

需要验证：

- Megatron actor/ref 配置可以被 OmegaConf 正常解析
- 所有必须的 Megatron 字段都已提供
- `critic.enable == false`
- actor/ref 的 TP/PP/EP 配置一致

### 阶段 2：单 Policy 冒烟测试

使用 `specialization=role_sharing`。

成功标准：

- workers 初始化成功
- rollout collection 成功
- `compute_log_prob` 成功
- 若启用了 reference policy，`compute_ref_log_prob` 成功
- 至少完成 1 个 training step 的 `update_actor`

### 阶段 3：多 Policy 冒烟测试

使用 `specialization=role_specific`。

成功标准：

- 每个 policy trainer 都能成功初始化
- 某些 step 缺少部分 policy batch 时，不会导致全局训练崩溃
- 仍然存在 batch 的 policy 能完成 log-prob 计算与 actor update

### 阶段 4：Checkpoint 验证

成功标准：

- checkpoint 保存成功
- checkpoint 恢复后 global step 一致
- 多 policy 的 checkpoint 目录彼此隔离，不互相覆盖

### 阶段 5：并行配置验证

成功标准：

- 给定 TP/PP/EP 后，Megatron model parallel 初始化成功
- actor 和 ref 使用兼容的分布式布局
- `RayPPOTrainer.init_workers` 不因 role mapping 错误而失败

## 风险与约束

### 多 Policy 资源压力

`role_specific` 下接 Megatron 代价可能很高，因为每个 policy trainer 都可能需要自己的一套 model-parallel worker 组。

这不仅是代码问题，更是资源调度和集群容量约束问题。

### 权重加载兼容性

如果目标 MoE 模型不兼容 VERL 当前支持的 Hugging Face 到 Megatron bridge 路径，那么失败点会出现在模型加载阶段，而不是 OrchRL 的训练逻辑阶段。

### LoRA 与 Ref-in-Actor 变体

本设计首版不针对 LoRA-heavy 场景做优化。

这些路径应作为非 LoRA Megatron 闭环稳定之后的后续验证项。

### Checkpoint 格式差异

分布式 Megatron checkpoint 与 Hugging Face checkpoint 的处理路径可能不同，具体取决于模型类型和 bridge 路径。本设计默认以 VERL 当前的 Megatron checkpoint manager 为唯一权威实现。

## 最终确定的 v1 范围

本次批准的 v1 范围为：

- OrchRL 多智能体训练
- VERL Megatron actor backend
- 支持 MoE 的 Megatron engine 配置
- GRPO
- critic disabled
- rollout 保持 vLLM
- 先验证 `role_sharing`
- 再验证 `role_specific`

这是当前最小、也最有可能稳定跑通的 OrchRL + VERL Megatron MoE 训练路径。
