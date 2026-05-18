# AstrBot vLLM Embedding Provider

为 AstrBot 提供一个独立的 `vllm_embedding` 模型提供商，用来直接对接 vLLM 的 OpenAI-compatible Embedding 接口。

这个插件的目标不是继续“伪装成 OpenAI Embedding”，而是把 vLLM 作为一个有自己兼容边界的 Embedding Provider 正式接入 AstrBot：单独的 provider 类型、单独的新增卡片、单独的运行日志，同时继续复用 AstrBot 原生 provider 管理页、知识库绑定方式和 core config 持久化机制。

---

## 它解决了什么问题

如果你直接把 vLLM 的 Embedding 服务地址填进 AstrBot 原生 `OpenAI Embedding` 提供商，常见痛点通常有这几类：

- AstrBot / OpenAI 侧习惯会主动传 `dimensions`，但 vLLM 的 OpenAI-compatible Embedding 路径并不保证接受这个参数。
- 你在配置里填的是 HuggingFace 模型名，例如 `BAAI/bge-m3`，但 vLLM 实际对外暴露的 `served-model-name` 往往只有 `bge-m3`。
- 本地或内网部署时，Python 环境变量里的代理可能会劫持请求，导致明明是 `127.0.0.1` 或局域网地址，却依然被错误地走了代理链路。
- 使用“伪装 OpenAI Embedding”的方案时，排障日志和使用语义都不直观，长期维护成本高。

本插件就是为了解决这些问题：

- 给 AstrBot 增加一个真正独立的 `vllm_embedding` provider 类型。
- 请求 embedding 时始终跳过 `dimensions`，把向量宽度决定权交还给 vLLM 和实际服务模型。
- 通过 `/v1/models` 自动把配置模型名对齐到 vLLM 实际接受的 `served-model-name`。
- 对本地 / 内网端点自动使用 `trust_env=False` 的直连 `httpx.AsyncClient`，避免环境代理干扰。
- 保留 AstrBot 原生 provider CRUD、测试、知识库绑定、Living Memory 调用路径，不改 AstrBot 本体文件。

---

## 核心特性

- 新增独立的 `vllm_embedding` provider 类型。
- 在“新增模型提供商 -> Embedding”中显示独立的 `vLLM Embedding` 卡片。
- 继续使用 AstrBot 原生 provider 管理页，不劫持 `OpenAI Embedding`。
- embedding 请求不主动向 vLLM 发送 `dimensions`。
- 自动调用 `/v1/models`，把配置模型名对齐到 vLLM `served-model-name`。
- 若 `/v1/models` 不可用，会优先回退到模型 basename，例如从 `BAAI/bge-m3` 回退到 `bge-m3`。
- 当端点是本地 / 私网地址且未显式配置代理时，自动切换为 `trust_env=False` 的直连 transport。
- 自动缓存检测到的向量维度；如果没有显式配置维度，会优先复用缓存或常见模型推断值。
- 为知识库检索、文档入库、批量嵌入请求输出明确的 provider 级日志。
- 以 AstrBot core config 作为主存储，同时在 `plugin_data` 中维护灾备镜像，并在启动时补种丢失的 `vllm_embedding` 条目。

---

## 架构说明

```text
AstrBot Provider Page / ProviderManager
   ^
   |  原生 provider CRUD / test / knowledge base binding
   v
astrbot_plugin_vllm_embedding_provider
   ^
   |  register_provider_adapter("vllm_embedding")
   |  template display alias
   |  backup + reseed
   v
vLLM OpenAI-compatible Embedding API
   ^
   |  GET /v1/models
   |  POST /v1/embeddings
   v
Embedding Model Runtime
```

插件本身只做三件事：

1. 注册独立的 `vllm_embedding` provider。
2. 让 AstrBot 的 provider 新增页把它显示为 `vLLM Embedding`。
3. 维护 `core config -> plugin_data backup -> 启动期缺失补种` 这条灾备链路。

Provider 的新增、编辑、删除、测试和启停，仍然全部走 AstrBot 自带的 provider 管理逻辑。

---

## 为什么 vLLM Embedding 不应主动传入 `dimensions`

这是本插件最核心的兼容点。

OpenAI Embedding 生态里，客户端主动传 `dimensions` 是一种常见用法，因为部分官方模型支持在请求时要求输出特定宽度的向量。但对 vLLM 来说，情况并不一样：

- vLLM 提供的是 OpenAI-compatible 接口，不代表每个 Embedding 参数都和 OpenAI 官方后端完全等价。
- 很多 vLLM Embedding 部署本质上是“按 served model 的原生维度返回向量”，并不接受客户端再主动要求降维或改维。
- 如果客户端仍然机械地把 `dimensions` 发过去，可能会遇到请求失败、兼容性异常，或者把原本正常的本地部署变成难以排查的“伪 OpenAI 错误”。

因此这个插件的策略非常明确：

- `embedding_dimensions` 字段保留在 AstrBot 侧，作为配置展示、向量库维度确认、自动检测结果回填和 `get_dim()` 的本地参考值。
- 但在真正向 vLLM 发送 `/v1/embeddings` 请求时，插件不会把 `embedding_dimensions` 转成 `dimensions` 参数发出去。

换句话说：

- `embedding_dimensions` 是 AstrBot 本地配置语义。
- 不是这个插件发给 vLLM 的主动控制参数。

这样做的结果是：vLLM 返回模型原生向量宽度，AstrBot 再根据检测结果或显式配置去维护自己的向量库一致性。

---

## 为什么还要做 `served-model-name` 对齐

在 AstrBot 表单里，用户更习惯填写 HuggingFace 模型名，例如：

- `BAAI/bge-m3`
- `BAAI/bge-large-zh-v1.5`

但 vLLM 实际对外暴露的模型 id 常常不是这个全名，而是启动服务时设置的 `served-model-name`，例如：

- `bge-m3`
- `bge-large-zh-v1.5`

如果插件不做处理，就会出现：

- WebUI 里填的是 `BAAI/bge-m3`
- 请求时发的也是 `BAAI/bge-m3`
- 但 vLLM 实际只接受 `bge-m3`

本插件会先尝试调用 `/v1/models`：

- 优先按 `id` 精确匹配
- 再按 `root` 匹配
- 若仍找不到，则回退到 basename，例如把 `BAAI/bge-m3` 回退成 `bge-m3`

这能覆盖大多数“用户填 HuggingFace 名称，vLLM 实际服务名不同”的场景。

---

## 为什么本地 / 内网端点要规避环境代理

很多 AstrBot 部署环境里会设置 `HTTP_PROXY` / `HTTPS_PROXY` 等环境变量。对外网 API 这通常没问题，但对本地 / 内网 vLLM 端点来说，经常会带来反效果：

- `http://127.0.0.1:8001/v1`
- `http://192.168.x.x:8001/v1`
- `http://host.docker.internal:8001/v1`

这些地址本来应该直连，但默认 transport 仍可能读取环境代理配置，导致请求绕出去，表现为超时、空响应或莫名其妙的 5xx。

本插件会在以下条件同时满足时自动切换为直连模式：

- 未显式填写 `proxy`
- `embedding_api_base` 指向本地回环地址、`host.docker.internal` 或私网 IP

此时会使用 `httpx.AsyncClient(trust_env=False)`，直接绕开环境代理。

---

## 功能范围

### 已覆盖

- 独立 provider 注册与新增卡片显示
- provider 测试
- 单条 embedding 请求
- 批量 embedding 请求
- 知识库文档入库 / 分块 / 检索
- Living Memory 等显式依赖 Embedding Provider 的调用路径
- provider 级请求日志
- core config 灾备镜像与启动期补种

### 不做的事

- 不修改 AstrBot 本体代码
- 不覆盖或劫持原生 `OpenAI Embedding` 提供商
- 不提供单独插件配置页表单
- 不把 `embedding_dimensions` 作为请求参数主动发给 vLLM
- 不处理 chat completion、rerank、speech 等其它 provider 能力

---

## 安装

### 方式一：手动安装

将插件目录放入 AstrBot 的插件目录：

```bash
cd data/plugins
git clone https://github.com/Creeper3222/astrbot_plugin_vllm_embedding_provider
```

安装依赖：

```bash
pip install -r data/plugins/astrbot_plugin_vllm_embedding_provider/requirements.txt
```

如果你的 AstrBot 环境通过 `uv` 管理，也可以使用：

```bash
uv pip install -r data/plugins/astrbot_plugin_vllm_embedding_provider/requirements.txt
```

### 方式二：通过 AstrBot WebUI 安装

如果后续已经接入插件市场，也可以直接通过 AstrBot WebUI 安装本插件。

---

## 快速开始

首次安装后的初始状态是：插件启用后会注册 `vllm_embedding` provider 类型，但 provider 实例本身仍然需要你在 AstrBot 的 provider 管理页里手动创建。也就是说，只有在你显式新增一个 `vLLM Embedding` provider 并填写连接信息后，它才会真正参与知识库、记忆检索或其它 Embedding 场景。

### 1. 先确认 vLLM Embedding 服务可用

你至少需要保证以下两个接口可访问：

- `GET /v1/models`
- `POST /v1/embeddings`

常见本地地址例如：

```text
http://127.0.0.1:8001/v1
```

### 2. 启用本插件

在 AstrBot 插件管理页启用 `astrbot_plugin_vllm_embedding_provider`。

启用后预期行为：

- 插件配置页显示“这个插件没有配置”
- Provider 页面新增 `vLLM Embedding` 卡片

### 3. 在 AstrBot 中创建 `vLLM Embedding` provider

进入 AstrBot WebUI：

1. 打开 `模型提供商`
2. 切换到 `Embedding`
3. 点击 `新增模型提供商`
4. 选择 `vLLM Embedding`

当前默认值大致如下：

| 设置项 | 默认值 | 说明 |
|------|------|------|
| `ID` | `vllm_embedding` | provider 唯一标识，可自行改成别的 id |
| `启用` | `关闭` | 新建后默认不启用，方便先填写再保存 |
| `API Key` | `空` | 若你的 vLLM 未启用鉴权，可以保持为空 |
| `API Base URL` | `空` | 建议填写完整 `/v1` 地址，如 `http://127.0.0.1:8001/v1` |
| `嵌入模型` | `空` | 可填 HuggingFace 全名或 vLLM served-model-name |
| `嵌入维度` | `空` | AstrBot 侧本地参考值，不会主动发给 vLLM |
| `超时时间` | `20` | 单位秒 |
| `代理地址` | `空` | 若留空且是本地/内网端点，插件会尽量直连 |

### 4. 推荐填写方式

最常见的一组填写示例如下：

| 设置项 | 示例值 |
|------|------|
| `API Base URL` | `http://127.0.0.1:8001/v1` |
| `嵌入模型` | `BAAI/bge-m3` |
| `API Key` | 留空 |
| `嵌入维度` | 先留空，或检测后再填 |
| `超时时间` | `20` |

建议顺序：

1. 先填 `API Base URL`
2. 再填 `嵌入模型`
3. 如无鉴权需求，`API Key` 保持空
4. 先保存，再用 AstrBot 原生“测试 provider”能力确认可连通
5. 需要时再用“自动检测嵌入维度”辅助确认本地维度配置

### 5. 在知识库中使用

AstrBot 的知识库会按显式的 `embedding_provider_id` 绑定 provider，而不是走某个全局“默认 embedding provider”。

因此推荐做法是：

1. 新建或确认 `vLLM Embedding` provider 可用
2. 创建新的知识库时选择它作为 Embedding 模型提供商
3. 上传文档并完成分块 / 入库
4. 用知识库检索测试确认召回和 provider 日志都正常

如果已有旧知识库仍绑定其他 provider，它不会被这个插件自动替换。

### 6. 在记忆插件中使用

只要对应插件最终也是通过 AstrBot 的 `EmbeddingProvider` 调用链拿向量，本插件就可以正常参与。换句话说，它并不是只给知识库用，而是给整个 AstrBot 的 Embedding Provider 生态用。

---

## 配置项说明

| 配置项 | 作用 | 说明 |
|------|------|------|
| `id` | provider 唯一 id | 知识库等模块最终按它绑定 provider |
| `embedding_api_key` | API Key | 本地无鉴权时通常可留空 |
| `embedding_api_base` | vLLM 地址 | 插件会自动规范化为 `/v1` 风格 |
| `embedding_model` | 模型名 | 可填 HuggingFace 全名，也可直接填 served-model-name |
| `embedding_dimensions` | 本地维度参考值 | 只用于 AstrBot 侧，不会主动传给 vLLM |
| `timeout` | 请求超时 | 单位秒 |
| `proxy` | 显式代理 | 若设置了它，插件会尊重该代理，不再强制直连 |

---

## 持久化与恢复策略

### 主存储

AstrBot core config 中的 `provider` 列表是主存储。

### 灾备镜像

插件会把当前所有 `type == vllm_embedding` 的 provider 镜像写入：

```text
data/plugin_data/astrbot_plugin_vllm_embedding_provider/providers_backup.json
```

### 恢复逻辑

启动时，插件会：

1. 读取 core config 中现有的 `vllm_embedding` provider
2. 读取 `plugin_data` 里的备份镜像
3. 只在“备份存在、core config 缺失同 id provider”时执行补种

这意味着：

- core config 是真源
- backup 只是灾备
- 如果你通过 AstrBot 原生 provider 页面正常删除一个 `vllm_embedding` provider，备份也会同步刷新，因此它不会在下次启动时被误恢复

---

## 日志与排障

本插件的 provider 日志前缀为：

```text
[vLLM Embedding]
```

你通常能从日志里直接看到：

- 是否成功把模型名对齐到 `served-model-name`
- 当前是单条还是批量 embedding 请求
- 请求使用的模型名
- 是否跳过了 `dimensions`
- 是否切换到了 `trust_env=False` 的直连 transport

如果你遇到问题，优先检查：

1. `embedding_api_base` 是否可访问
2. `/v1/models` 是否返回了实际服务模型
3. `embedding_model` 是否和服务模型名可对齐
4. 是否存在环境代理干扰
5. 日志里是否出现 `[vLLM Embedding]` 的 warning / info

---

## 已知限制

- 本插件只处理 Embedding Provider，不覆盖 chat completion、rerank、TTS、STT 等其它能力。
- 插件不会把 `embedding_dimensions` 当作请求参数主动发给 vLLM；如果你希望“按请求动态降维”，这不是当前设计目标。
- 对齐 `served-model-name` 的最佳路径依赖 `/v1/models`；如果该接口不可用，插件只能退回到配置模型名或 basename。
- 插件管理页显示“这个插件没有配置”是预期行为，因为 provider 实例配置本来就属于 AstrBot 原生 provider 管理页。
- provider 的持久化、测试、启停和删除仍由 AstrBot 原生逻辑负责；本插件只在这些原生操作之后维护备份镜像。

---

## 版本

- 当前版本：`v0.1.0`