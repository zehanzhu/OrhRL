# 当前 MAS 系统任务编排与上下文传递分析

## 1. 结论先行

基于当前仓库中实际接入 `agent_trajectory_engine` 的 Search MAS 实现，可以明确回答两个问题：

1. 当前 MAS 的任务编排是一个显式的、串行的 router-search-answer 循环，不是多个 agent 并发协商。
2. 每个 agent 在发起模型调用时，发送的 `messages` 会包含“截至当前轮次的全部历史上下文”，但这个“上下文”不是结构化 trace 对象，而是被压平成一段累计字符串 `team_context`，再嵌入到单条 `user` message 中。

换句话说：

- 是的，后续 agent 调用会带上此前所有 agent 输出和工具观测。
- 不是以多轮 chat history 的形式发送，而是以“单条 user prompt + 内嵌全文历史”的形式发送。
- 会产生明显的文本冗余；尤其是在 monitor/trajectory 导出里，后续调用会重复保存前面轮次的全部文本上下文。

## 2. 分析范围

本文分析的“当前 MAS 系统”，对应的是仓库里的 Search MAS 路径：

- `mas_apps/search/search_mas/apps/search/app.py`
- `mas_apps/search/search_mas/apps/search/orchestrator.py`
- `mas_apps/search/search_mas/core/agent.py`
- `mas_apps/search/search_mas/core/orchestration.py`
- `mas_apps/search/search_mas/core/llm.py`
- `orchrl/agent_trajectory_engine/_support/launcher.py`
- `orchrl/agent_trajectory_engine/monitor_actor.py`

其中：

- Search MAS 负责“应用侧编排”和“prompt 构造”。
- `agent_trajectory_engine` 负责把每个 agent 的模型调用代理到 monitor/back-end，并记录轨迹。

## 3. 核心组件与职责

### 3.1 应用入口

`SearchMASApplication.solve()` 是应用入口，它直接调用 orchestrator：

```python
def solve(self, question: str) -> MASRunResult:
    out = self.orchestrator.run(question)
    return MASRunResult(...)
```

对应代码见：

- `mas_apps/search/search_mas/apps/search/app.py:80-88`

### 3.2 三个 agent

Search MAS 中有三个 agent：

- `VerifierAgent`
  - 负责判断当前信息是否足够回答问题
  - 输出 `<verify>yes</verify>` 或 `<verify>no</verify>`
- `SearchAgent`
  - 负责生成检索 query
  - 输出 `<search>...</search>`
- `AnswerAgent`
  - 负责综合已有信息给出最终答案
  - 输出 `<answer>...</answer>`

这些解析逻辑在：

- `mas_apps/search/search_mas/apps/search/agents.py:19-46`

### 3.3 运行态记忆

当前实现没有独立的 Memory 对象，也没有按 message 列表维护“原始多轮对话历史”。

运行态真正起“记忆”作用的是一个字符串：

- `team_context: str`

它在 orchestrator 内部不断累积，作为后续每个 agent prompt 的一部分。

## 4. 任务编排主流程

### 4.1 总体流程

`SearchRouterOrchestrator.run(question)` 的核心循环见：

- `mas_apps/search/search_mas/apps/search/orchestrator.py:39-120`

主流程可以展开为：

```text
输入 question
  -> 初始化 team_context = ""
  -> for loop_i in range(max_turns):
       1. verifier.run(question, team_context, step=loop_i)
       2. parse_decision(verifier_output)
       3. 把 verifier_output 追加到 team_context
       4. 如果 approved:
            answerer.run(question, team_context, step=loop_i)
            提取 final_answer
            结束
          否则:
            searcher.run(question, team_context, step=loop_i)
            提取 search query
            把 searcher_output 追加到 team_context
            调用 search tool
            把 <information>...</information> 追加到 team_context
            进入下一轮 loop
  -> 如果达到 max_turns 且还没有 final_response:
       answerer.run(question, team_context, step=loop_count)
```

### 4.2 逐步方法调用链

单轮未通过 verifier 时的方法调用链是：

```text
SearchMASApplication.solve
  -> SearchRouterOrchestrator.run
    -> VerifierAgent.run
      -> BaseChatAgent.run
        -> prompt_template.format(...)
        -> build_messages(prompt)
        -> OpenAICompatibleLLM.chat(messages, gen_cfg)
    -> VerifierAgent.parse_decision
    -> append_team_context
    -> SearchAgent.run
      -> BaseChatAgent.run
        -> prompt_template.format(...)
        -> build_messages(prompt)
        -> OpenAICompatibleLLM.chat(...)
    -> SearchAgent.extract_search_query
    -> append_team_context
    -> _run_search
      -> search_client.search(query)
    -> append_tool_observation
    -> 下一轮 verifier.run(...)
```

若 verifier 通过，则会走：

```text
... -> VerifierAgent.parse_decision(approved=True)
    -> append_team_context
    -> AnswerAgent.run
      -> BaseChatAgent.run
        -> prompt_template.format(...)
        -> build_messages(prompt)
        -> OpenAICompatibleLLM.chat(...)
    -> AnswerAgent.extract_answer
    -> 结束
```

## 5. 数据流拆解

### 5.1 question 如何流动

输入问题 `question` 从 `solve(question)` 进入后，会原样传给每一次 agent 调用：

- `verifier.run(question, team_context, step=...)`
- `searcher.run(question, team_context, step=...)`
- `answerer.run(question, team_context, step=...)`

这个问题文本会被放入 prompt 模板的 `{env_prompt}`。

### 5.2 team_context 如何流动

`team_context` 初始为空串：

```python
team_context = ""
```

之后通过两个函数累计：

```python
def append_team_context(team_context: str, agent_id: str, response: str) -> str:
    return team_context + f'\nThe output of "{agent_id}": {response}\n'

def append_tool_observation(team_context: str, observation: str) -> str:
    return team_context + f"\n{observation}\n"
```

对应代码：

- `mas_apps/search/search_mas/core/orchestration.py:4-9`

因此，`team_context` 里会累积两类信息：

1. 每个 agent 的完整输出文本
2. 每次工具调用得到的 `<information>...</information>` 观测文本

### 5.3 prompt 如何构造

所有 agent 最终都经过 `BaseChatAgent.run()`：

```python
prompt = self.prompt_template.format(
    env_prompt=env_prompt,
    team_context=team_context,
    step=step,
)
messages = self.build_messages(prompt)
response = self.llm.chat(messages, self.generation_config)
```

而 `build_messages()` 的实现非常关键：

```python
return [{"role": "user", "content": prompt}]
```

对应代码：

- `mas_apps/search/search_mas/core/agent.py:21-34`

这说明：

- 没有 system message
- 没有把历史拆成多条 user/assistant message
- 没有把历史 trace 结构化传给模型
- 所有历史都被直接拼进当前这一条 `user` prompt 的 `content`

### 5.4 LLM 请求 payload 如何发送

最终发送给后端的 payload 由 `OpenAICompatibleLLM.chat()` 构造：

```python
payload = {
    "model": gen_cfg.model,
    "messages": messages,
    ...
}
resp = self.client.chat.completions.create(**payload)
```

对应代码：

- `mas_apps/search/search_mas/core/llm.py:35-63`

也就是说，真正发出的请求核心字段是：

```json
{
  "model": "<agent-role-or-model-name>",
  "messages": [
    {
      "role": "user",
      "content": "<完整 prompt，其中已经包含累计 team_context>"
    }
  ]
}
```

## 6. Prompt 模板决定了每个 agent 实际看到什么

模板定义见：

- `mas_apps/search/search_mas/apps/search/prompts.py:1-44`

三类 agent 的 prompt 结构分别是：

### 6.1 Verifier

```text
# Task Introduction
{env_prompt}

# Your Role
...

# Your Teammates' Outputs
{team_context}
```

### 6.2 Searcher

```text
# Task Introduction
{env_prompt}

# Your Teammates' Outputs at Step {step}
{team_context}

# Your Role
...
```

### 6.3 Answerer

```text
# Task Introduction
{env_prompt}

# Your Teammates' Outputs
{team_context}

# Your Role
...
```

因此，从 prompt 结构上就可以确认：每个 agent 都会直接看到累计到当前时刻的 `team_context` 全文。

## 7. 逐轮上下文累积图

下面用当前实现的真实控制流，画出 prompt 内容如何逐轮膨胀。

### 7.1 Turn 0: 第一次 verifier

初始状态：

```text
question = Q
team_context = ""
```

发送给 verifier 的 prompt 等价于：

```text
Task Introduction: Q
Teammates' Outputs:
  [空]
```

所以第一次 verifier 调用不包含任何历史 teammate 输出。

### 7.2 Turn 0: searcher

如果 verifier 判定 `<verify>no</verify>`，则：

1. 先把 verifier 输出写入 `team_context`
2. 再调用 searcher

此时 searcher 看到的是：

```text
Task Introduction: Q
Teammates' Outputs at Step 0:
  The output of "Verifier Agent": <verifier_output>
```

### 7.3 Turn 0: search tool 之后

searcher 输出 query 后：

1. `searcher_output` 追加到 `team_context`
2. 调用 search tool
3. search tool 返回的结果包装为：

```text
<information>{search_result.result_text}</information>
```

4. 再把这段 observation 追加到 `team_context`

此时 `team_context` 已包含：

```text
The output of "Verifier Agent": <verifier_output>
The output of "Search Agent": <searcher_output>
<information>...</information>
```

### 7.4 Turn 1: 下一次 verifier

下一轮 verifier 调用时，发送出去的 prompt 就包含上面全部累计内容：

```text
Task Introduction: Q
Teammates' Outputs:
  The output of "Verifier Agent": <上一轮 verifier_output>
  The output of "Search Agent": <上一轮 searcher_output>
  <information>...</information>
```

### 7.5 最终 answerer

一旦 verifier 批准，或者达到 `max_turns` 被强制回答，answerer 收到的也是“截至当前为止完整累计的 `team_context`”。

所以最后一次 answerer 调用通常会看到：

```text
Task Introduction: Q
Teammates' Outputs:
  The output of "Verifier Agent": ...
  The output of "Search Agent": ...
  <information>...</information>
  The output of "Verifier Agent": ...
  The output of "Search Agent": ...
  <information>...</information>
  ...
```

## 8. 真实轨迹证据

下面不是推断，而是来自导出的真实 trajectory。

### 8.1 第一次 verifier 没有历史上下文

在导出轨迹中，第一条 verifier 请求的 message 内容末尾就是空的 teammate outputs：

- `checkpoints/search_mas_app/trajectories/step_000000/prompt_prompt-0_0.json:67-97`

其中可以看到：

```text
# Your Teammates' Outputs

```

这说明首轮 verifier 的 prompt 中，历史为空。

### 8.2 searcher 请求中已经包含 verifier 的完整输出

导出轨迹中的 searcher 调用记录：

- `checkpoints/search_mas_app/trajectories/step_000000/prompt_prompt-0_0.json:9105-9135`

可以直接看到，它的 `messages[0].content` 内嵌了：

```text
The output of "Verifier Agent": <think>...</think>
<verify>no</verify> ...
```

这证明 searcher 并不是只收到原始问题，而是收到了上一位 agent 的完整文本输出。

### 8.3 后续 verifier 请求中包含 agent 输出和 tool observation

导出轨迹中的下一轮 verifier 调用记录：

- `checkpoints/search_mas_app/trajectories/step_000000/prompt_prompt-0_0.json:10459-10489`

其 `messages[0].content` 中同时出现了：

```text
The output of "Verifier Agent": ...
The output of "Search Agent": ...
<information>{"result": ...}</information>
```

这证明到了后续轮次，prompt 中确实包含：

1. 之前 verifier 的输出
2. 之前 searcher 的输出
3. 工具返回的 `<information>` 观测

### 8.4 answerer 请求中也会继承已有历史

导出轨迹中的 answerer 调用记录：

- `checkpoints/search_mas_app/trajectories/step_000000/prompt_prompt-0_0.json:1645-1675`

其中 `messages[0].content` 里已经包含：

```text
The output of "Verifier Agent": ...
```

这说明 answerer 也不是单独根据问题回答，而是根据累计历史作答。

## 9. 训练/监控侧如何接收这些请求

### 9.1 Launcher 会把每个 role 的模型路由改写到 monitor

`MASLauncher.prepare_config()` 会重写配置：

```python
llm_cfg["base_url"] = monitor_url
...
role_cfg["model"] = role
...
role_llm_cfg["base_url"] = monitor_url
```

对应代码：

- `orchrl/agent_trajectory_engine/_support/launcher.py:19-72`

这意味着训练/回放时，每个 agent 发出的 chat completion 请求会被送到 monitor，而不是直接打真实模型服务。

### 9.2 Monitor 直接读取请求里的 model 和 messages

`MonitorActor._handle_chat_completions()` 中：

```python
agent_role = body.get("model")
messages = body.get("messages", [])
...
model_request = ModelRequest(
    agent_role=agent_role,
    messages=messages,
    ...
)
```

对应代码：

- `orchrl/agent_trajectory_engine/monitor_actor.py:172-223`

这进一步说明：

1. monitor 看到的就是 agent 原样发出的 `messages`
2. 如果 agent 侧把完整 `team_context` 拼进了 prompt，那么 monitor 记录到的也是这份完整拼接后的 prompt

## 10. “是否包含所有上下文记录”的准确回答

这个问题需要分层回答。

### 10.1 对模型请求来说：是，包含截至当前 turn 的全部历史文本上下文

如果“所有上下文记录”指的是：

- 之前各 agent 的输出文本
- 工具观测文本 `<information>...</information>`

那么答案是：

**是。**

每次新 agent 调用都会把这些历史文本累加进 `team_context`，然后整体塞入当前 prompt。

### 10.2 但不是“所有结构化运行信息”都发给模型

如果“所有上下文记录”指的是更广义的运行时元数据，例如：

- `trace` 里的结构化 `metadata`
- `loop_index`
- `turn_index`
- monitor 记录的 `sampling_fingerprint`
- replay/branch 等控制字段

那么答案是：

**不是。**

这些元数据不会自动作为结构化对象传给模型；模型看到的是被模板化后的自然语言 prompt 文本。

### 10.3 它也不是传统的多轮 chat history

当前实现不是：

```json
[
  {"role": "system", ...},
  {"role": "user", ...},
  {"role": "assistant", ...},
  {"role": "user", ...}
]
```

而是：

```json
[
  {
    "role": "user",
    "content": "当前问题 + 全部累计历史文本"
  }
]
```

这一点非常关键，因为它决定了：

- prompt 累积完全由应用层字符串拼接控制
- 模型端看不到“谁说的是 assistant，谁说的是 tool”的原生角色边界
- 所有历史边界都退化成普通文本格式约定

## 11. 会不会导致存储冗余

会，而且是当前实现的自然结果。

### 11.1 运行态冗余

运行过程中，`team_context` 每轮都会把旧内容整体保留，再附加新内容。

因此第 `k` 次调用的 prompt 包含：

- 第 0 轮之前的所有文本
- 第 1 轮新增文本
- ...
- 第 k-1 轮新增文本

这意味着 prompt 长度会单调增长。

### 11.2 轨迹导出冗余

monitor/trajectory 导出保存的是“每次调用当时发出去的 `messages`”。

所以：

- 第一次调用保存一份短 prompt
- 第二次调用保存一份“第一次 prompt 内容 + 新增文本”的长 prompt
- 第三次调用再保存一份更长 prompt

因此在轨迹文件里，同一段早期历史会在多个后续调用中被重复保存。

### 11.3 但 `trace` 本身没有同等级别的全文重复

`SearchRouterOrchestrator` 返回的 `trace` 结构只有：

- `loop_index`
- `agent_id`
- `output`
- `metadata`

定义见：

- `mas_apps/search/search_mas/apps/base.py:7-27`

也就是说：

- `trace` 记录的是“每一步各自产生了什么”
- 真正高冗余的是“每次模型调用时携带的累计 prompt 文本”

### 11.4 一个重要推断

以下是根据代码行为得出的推断：

如果轮次增多、每轮输出较长，那么总的 prompt 传输量和轨迹存储量会呈现明显的累积增长，近似表现出“后续轮次重复携带前文”的二次型开销趋势。

这不是额外的隐藏机制，而是由“累计字符串 + 每次全量发送”这一设计直接导致的。

## 12. 为什么系统会这样设计

从代码实现看，这种设计的优点很直接：

1. 简单
   - 不需要维护复杂 memory object
   - 不需要把历史拆成结构化 chat messages
2. 可重放
   - 每次请求的 prompt 都是自洽闭包
   - monitor 直接拿到完整请求即可回放
3. 易训练
   - 轨迹导出后，每次模型调用样本都已经带有完整输入文本

代价也很直接：

1. 文本冗余高
2. token 开销大
3. 历史越长，后续每次调用越贵
4. 角色边界和工具边界都退化为普通文本，不如原生多消息格式清晰

## 13. 最终回答

针对你的问题，可以压缩成一句话：

**当前 MAS 系统里，每个 agent 发起模型调用时，发送的是“原始问题 + 截至当前轮次累计的全部 teammate/tool 文本上下文”，这些历史会被压平成单条 `user` message 的 `content`；因此是包含完整前文的，也确实会带来显著的 prompt/存储冗余。**
