# OrchRL 仓库说明

OrchRL 是一个面向多 Agent 编排场景的非侵入式强化学习训练框架仓库。OrchRL 的核心优势不是“内置某个特定 MAS 应用”，而是可以对外部黑盒 MAS 系统做非侵入式的 RL 训练。

当前仓库的主路径包括：
- `orchrl/`: 框架层代码
- `experiments/search_mas/`: 当前主打的 Search MAS 训练实验入口
- `mas_apps/search/`: 被训练流程直接调用的外部 Search MAS 推理应用


这里的“非侵入式”主要体现在：

- 不要求把 MAS 重写成 OrchRL 内部模块或 trainer 子类
- MAS 应用可以继续作为独立工程保留在 `mas_apps/` 下
- 训练侧只需要提供工作目录、启动命令模板、配置模板、role 到 policy 的映射
- OrchRL 负责补齐 monitor、trajectory 采集、reward 计算、policy 更新、验证与输出管理

也就是说，OrchRL 的目标是把“黑盒 MAS 应用”接进 RL 训练闭环，而不是要求你先把 MAS 内部工作流改造成框架原生实现。



如果你第一次进入这个仓库，可以把它理解成：

1. `orchrl/` 负责训练、rollout、monitor 路由、reward bridge、验证和输出管理
2. `mas_apps/search/` 负责 Search MAS 应用本身
3. `experiments/search_mas/` 负责把两者装配成一条可运行的端到端训练路径



## 当前主路径

当前仓库最完整、最优先支持的路径是 Search MAS 端到端训练。

核心入口：

- 训练启动脚本：`bash experiments/search_mas/run_train_e2e.sh`
- 主训练配置：`experiments/search_mas/train.yaml`
- 外部 MAS 应用：`mas_apps/search/`
- 训练输出根目录：`outputs/`

当前训练配置支持两种 specialization 模式：

- `role_sharing`: 多个 agent 共用一个 policy，只训练一个 policy
- `role_specific`: 每个 agent 使用自己的 policy，训练多个 policy

当前 Search MAS 默认使用 `role_specific`。

## 仓库结构

```text
.
├── orchrl/                       # 框架层
│   ├── agent_trajectory_engine/  # monitor、rollout backend、trajectory datatypes
│   ├── trainer/                  # train entry、trainer orchestration、MATE runtime、validation
│   ├── utils/                    # cleanup、Ray init、输出路径、导入工具
│   └── config/                   # 基础 PPO trainer 配置片段
├── experiments/
│   └── search_mas/               # 当前 canonical 训练实验
├── mas_apps/
│   └── search/                   # Search MAS 推理应用与检索服务脚本
├── task_plugins/
│   └── search_mas/               # Search MAS reward / answer stats 插件
├── outputs/                      # 日志、checkpoints、trajectory 等运行输出
├── tests/                        # 回归测试
└── OrchRL_Training_Flow.md       # 更细粒度的训练链路说明
```

按职责看，可以分成 4 层：

- 框架层：`orchrl/`
- 应用层：`mas_apps/`
- 实验层：`experiments/`
- 任务插件层：`task_plugins/`

其中：

- 框架层不放具体 Search MAS 业务逻辑
- 应用层不放训练器框架逻辑
- 实验层只放当前实验需要的配置和启动脚本
- 任务插件层只放任务级 reward/统计逻辑

## 5 分钟快速定位

如果你只想尽快找到“该改哪里”：

- 改训练入口：`experiments/search_mas/run_train_e2e.sh`
- 改训练超参和路径：`experiments/search_mas/train.yaml`
- 改 reward 逻辑：`task_plugins/search_mas/reward.py`
- 改训练器主流程：`orchrl/trainer/`
- 改 monitor / rollout 路由：`orchrl/agent_trajectory_engine/`

## 环境与依赖边界

当前 Search MAS 训练路径依赖 3 类运行环境：

### 1. 主训练环境

主训练环境负责运行：

- `python -m orchrl.trainer.train`
- `experiments/search_mas/run_train_e2e.sh`
- 训练过程中拉起的 Search MAS 子进程

推荐：

```bash
conda create -n orchrl-train python=3.12 -y
conda activate orchrl-train
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

仓库根 `requirements.txt` 已经包含：

- OrchRL 框架自身依赖
- `mas_apps/search/requirements.txt`

所以准备主训练环境时，不需要额外单独安装 `mas_apps/search/requirements.txt`。

### 2. VERL / vLLM 依赖

当前仓库不内置 `verl`，但训练代码会直接 import `verl`。

需要在同一个 Python 环境中安装一份兼容的 VERL checkout：

```bash
python -m pip install -e /path/to/verl
```

同时，`vllm` 也应与这份 VERL 和当前 CUDA 环境兼容。

### 3. Retriever 环境

Search MAS 的本地检索服务建议放在独立 conda 环境里，也就是单独的 separate conda environment，尤其是 `faiss-gpu`、`datasets` 这类依赖不应混入主训练环境。

Retriever 的详细准备方式见：

- `mas_apps/search/README.md`

如果你已经有远程检索服务，只需要正确设置 `SEARCH_MAS_RETRIEVAL_SERVICE_URL`，不必本地部署 retriever。

## Search MAS 端到端训练

### 1. 训练前需要准备什么

在执行训练脚本前，请确认：

- `experiments/search_mas/train.yaml` 中引用的模型路径存在
- `training.train_data_path` 和 `training.val_data_path` 指向的数据文件存在
- `training.mate.config_template_path` 指向的 Search MAS 模板配置存在
- 检索服务可用

### 2. 运行训练

```bash
bash experiments/search_mas/run_train_e2e.sh
```

这个脚本会先做几件事：

1. 从 `experiments/search_mas/train.yaml` 解析关键运行路径
2. 检查 MAS 工作目录、配置模板、数据路径、模型路径是否存在
3. 设置若干运行环境变量
4. 调用 `python3 -m orchrl.trainer.train`

训练 launcher 解析训练数据路径时，使用的是 `cfg.training.train_data_path` 和 `cfg.training.val_data_path`，而不是旧文档里提过的 `cfg.training.mate.prompt_loader.path`。

### 3. 输出会落到哪里

当前输出统一落在 `outputs/` 下。

主要包括：

- 训练日志：`outputs/logs/search_mas_train_e2e_<timestamp>.log`
- 训练运行目录：`outputs/training_runs/<experiment_name>/<run_id>/`
- checkpoint 根目录：`outputs/training_runs/<experiment_name>/<run_id>/checkpoints`
- trajectory 导出目录：`outputs/training_runs/<experiment_name>/<run_id>/trajectories`
- MAS 子进程日志目录：`outputs/training_runs/<experiment_name>/<run_id>/mas_logs`

这些目录会在训练启动时 created automatically，不需要手动预建。MAS 子进程的 stdout/stderr 也会按 episode 持久化到 `mas_log_dir` 下，便于排查失败轨迹。

## 关键配置心智模型

当前 Search MAS 实验最重要的配置文件是：

- `experiments/search_mas/train.yaml`

你通常只需要先理解这几组字段：

### 1. 数据路径

- `training.data_root_dir`
- `training.train_data_path`
- `training.val_data_path`
- `training.validate_batch_size`

默认情况下：

- 训练集是 `${training.data_root_dir}/train.parquet`
- 验证集是 `${training.data_root_dir}/test.parquet`

### 2. 多 Agent 与 policy 绑定

- `agent_policy_configs.agent_configs`
- `training.mate.role_policy_mapping`
- `base_models`
- `models`

这里定义了：

- 有哪些 agent role
- 每个 role 用哪个 policy
- 每个 policy 对应哪个 base model
- 每个 policy 的 PPO / rollout 配置

### 3. MATE rollout 配置

- `training.mate`

这里控制：

- 训练时的 rollout source
- tree / parallel 模式
- prompt loader
- reward provider
- trajectory export
- monitor pool

当前实现里，`training.mate.prompt_loader` 还额外决定训练集采样语义：

- `train_repeat`
- `train_shuffle`
- `train_seed`

其中训练集会按 `train_repeat/train_shuffle` 做可重复的多轮采样；验证集始终走确定性的全量遍历。

另外还有两组直接影响训练/验证语义的字段：

- `training.mate.reward.match_mode`
- `training.mate.failure_policy`

`match_mode` 会同时作用于训练 reward、trajectory export answer stats，以及 standalone validation；`failure_policy` 则控制 rollout 失败在训练中是告警还是按阈值中断。

### 4. 输出路径

- `training.output_root_dir`
- `training.run_name`
- `training.run_id`
- `training.run_dir`
- `training.model_checkpoints_dir`
- `training.mate.trajectory_export.output_dir`

## 代码分层说明

### `orchrl/agent_trajectory_engine/`

这部分是轨迹采集与 runtime 路由层，负责：

- monitor actor
- lease / monitor pool 管理
- rollout backend
- token-id 路径下的请求封装与响应归档
- parallel/tree rollout

如果你在看“不同 agent 的请求是如何被路由到不同推理服务”的问题，主要看这里。

### `orchrl/trainer/`

这部分是训练 orchestration 层，核心模块包括：

- `train.py`
  - 仓库级训练入口
  - 初始化 Ray
  - 校验 specialization 与 policy topology
- `multi_agents_ppo_trainer.py`
  - 多 policy 训练总控
- `policy_trainer_registry.py`
  - 按 policy 创建和持有 PPO trainer
- `mate/runtime.py`
  - 装配 prompt loader、reward provider、rollout adapter、monitor pool
- `training_step_executor.py`
  - 单个训练 step 的 rollout 收集、batch 转换、参数更新
- `validation_runner.py`
  - 完整验证集遍历与 checkpoint 保存逻辑

### `task_plugins/search_mas/`

这部分是任务级插件层，不属于通用训练框架。当前主要包含：

- reward 计算
- answer stats 统计

### `mas_apps/search/`

这部分是外部 Search MAS 应用本体，不是训练框架的一部分。它负责：

- 多 Agent 编排
- LLM 调用
- 检索调用
- standalone 验证

更细的应用说明见：

- `mas_apps/search/README.md`

## 训练执行链路

当前主链路可以概括为：

1. `bash experiments/search_mas/run_train_e2e.sh`
2. `python -m orchrl.trainer.train`
3. `train.py` 初始化 Ray，构建 `MultiAgentsPPOTrainer`
4. `PolicyTrainerRegistry` 初始化每个 policy 对应的 PPO trainer
5. `MateRuntime` 初始化：
   - train/val prompt loader
   - reward provider
   - rollout adapter
   - monitor pool manager
6. `TrainingStepExecutor` 在每个 step：
   - 触发 rollout
   - 将 episode 转成按 policy 分组的 `DataProto`
   - 调用各 policy trainer 更新参数
7. `ValidationRunner` 按 `validate_batch_size` 遍历完整验证集，计算样本级平均 reward

更细的流程图与逐模块说明见：

- `OrchRL_Training_Flow.md`

## 当前验证逻辑

当前验证阶段的行为是：

- 遍历完整验证集，不是只抽一小部分 prompt
- 按 `training.validate_batch_size` 分 batch 验证
- 每个验证样本只 rollout 一条轨迹
- 验证阶段强制使用 `parallel` 模式，而不是 tree rollout
- 统计整个验证集上的 `validation/sample_avg_reward`
- 统计验证准确率 `validation/accuracy`
- 同时统计 `validation/failed_sample_count` 和 `validation/failed_sample_rate`
- 同时输出 MAS 系统级指标，例如 `mas/validation/accuracy`、`mas/validation/avg_turns`、`mas/validation/search_call_rate`、`mas/validation/answer_rate`
- 失败样本会按 expected sample count 计入分母，reward 视为 `0.0`
- 逐 batch 更新累计平均值，does not retain the full validation trajectory set in memory

## 与 `mas_apps/search/README.md` 的职责边界

根 README 只解释整个仓库。

以下内容不在这里展开，而是交给 `mas_apps/search/README.md`：

- Search MAS 推理应用内部目录结构
- Verifier / Searcher / Answerer 的应用逻辑细节
- 检索服务下载与部署细节
- standalone inference / validation 命令
- OpenAI / vLLM / retriever 的环境变量覆盖细节

换句话说：

- 想理解“整个仓库怎么组织、训练怎么跑”，看根 `README.md`
- 想理解“Search MAS 应用本身怎么单独运行”，看 `mas_apps/search/README.md`

## 文档导航

建议按下面顺序阅读：

1. `README.md`
2. `experiments/search_mas/README.md`
3. `mas_apps/search/README.md`
4. `OrchRL_Training_Flow.md`

其中：

- 根 `README.md` 建立仓库级心智模型
- `experiments/search_mas/README.md` 建立实验级心智模型
- `mas_apps/search/README.md` 建立应用级心智模型
- `OrchRL_Training_Flow.md` 建立更细的训练执行链路心智模型

## 仓库记忆与 Codex 使用

为了让后续在这个仓库里的 Codex 会话能更稳定地继承上下文，当前仓库已经把关键记忆落成了仓库级文档，而不是只依赖自动 memory 提炼。

建议优先阅读这些文件：

1. `AGENTS.md`
2. `REPOSITORY_FUNCTION_ANALYSIS.md`
3. `CODE_REVIEW_FINDINGS.md`
4. `OrchRL_Training_Flow.md`

其中：

- `AGENTS.md` 记录仓库身份、当前主链路、稳定偏好和仓库级工作约束
- `REPOSITORY_FUNCTION_ANALYSIS.md` 记录当前仓库实现的功能与细粒度训练链路
- `CODE_REVIEW_FINDINGS.md` 记录当前已确认的代码 review 问题与验证状态
- `OrchRL_Training_Flow.md` 记录更细的训练执行路径

如果你希望给这个仓库使用独立的 Codex memory，而不影响用户目录下的 `~/.codex/`，可以从仓库根目录这样启动：

```bash
CODEX_HOME=$(pwd)/.codex-home codex
```

这会使用仓库里的 `.codex-home/` 作为本仓库专属 Codex home：

- 不会删除或覆盖 `~/.codex/`
- 会让这个仓库的 memory/config 与全局 Codex 状态隔离
- 但自动 memories 仍然只是辅助机制，不能替代上面的仓库级记忆文档

## 常见问题

### 数据集路径应该改哪里

优先改：

- `experiments/search_mas/train.yaml` 中的
  - `training.data_root_dir`
  - `training.train_data_path`
  - `training.val_data_path`

### 模型路径应该改哪里

优先改：

- `experiments/search_mas/train.yaml` 中的 `base_models`

### Search MAS 的配置模板在哪里

当前训练子进程使用：

- `mas_apps/search/configs/inference.yaml`

### standalone 运行 Search MAS 用哪个配置

优先使用：

- `mas_apps/search/configs/search_mas_example.yaml`

### 当前仓库是不是只围绕 Search MAS

从当前代码、目录与文档成熟度看，是的。当前最完整、最稳定、最推荐使用的路径就是 Search MAS。
