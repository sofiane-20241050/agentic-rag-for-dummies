# Agentic RAG for Dummies — 技术点深度分析

> 基于 `agentic-rag-for-dummies` 项目（Notebook + Project 完整拆解）
> 用于秋招面试准备 — 聚焦架构设计决策，非完整代码复现

---

## 一、项目概览

一个**生产级 Agentic RAG 系统**，由 Notebook 演进为模块化项目。核心理念：**让 Agent 自主决定何时检索、检索什么、检索多少**，而不是简单的"检索→拼接→回答"单次调用。

| 维度 | 该项目的选择 |
|------|------------|
| LLM | Ollama 本地 `granite4.1:8b`（可替换为任何 LangChain 兼容模型） |
| 嵌入 | `Qwen3-Embedding-0.6B`（稠密 1024 维）+ BM25（稀疏）→ 混合检索 |
| 向量库 | Qdrant 本地文件存储（`qdrant_db/`） |
| 编排 | LangGraph 双层状态图 |
| UI | Gradio `ChatInterface` 流式聊天 |
| 可观测 | Langfuse（可选，环境变量开关） |

---

## 二、父子分块策略（Parent-Child Chunking）

**这是该项目最核心的检索优化技术**，区别于 HelloAgents 的单层固定窗口切片。

### 2.1 为什么需要父子分块？

单层切片的矛盾：

- 切片太小 → 语义碎片化，检索召回率高但信息不完整
- 切片太大 → 检索精度下降，无关信息混入，浪费 context window

父子分块**同时解决两个问题**：用小块做精确检索，用大块做完整上下文。

### 2.2 具体实现

```
原始文档（Markdown）
    │
    ├─ MarkdownHeaderTextSplitter 按 H1/H2/H3 标题切分
    │
    ├─ merge_small_parents()  → 小于 2000 字符的合并
    ├─ split_large_parents()  → 大于 4000 字符的切分
    ├─ clean_small_chunks()   → 剩余的碎片用 rebalance_pair 重新分配
    │
    └─ Parent Chunk（2000~4000 字符）
         │
         ├─ 存为 JSON 文件 → parent_store/{parent_id}.json
         │
         └─ RecursiveCharacterTextSplitter(chunk_size=500, overlap=100)
              │
              └─ Child Chunk（500 字符）→ 入 Qdrant 向量库
```

### 2.3 关键设计：rebalance_pair 再平衡

```python
def rebalance_pair(first, second, min_size, max_size):
    combined = first + second
    # 在两块之间找一个最优分割点
    # 优先在 \n\n → \n → 空格 处切分
    # 确保两边都不小于 min_size
```

不是简单截断——当小块落到阈值以下时，和邻居**重新协商边界**，避免产生语义断裂的碎片。

### 2.4 元数据传递链

每个 Child Chunk 的 metadata 里带有 `parent_id`：

```python
p_chunk.metadata.update({"source": doc_path.stem + ".pdf", "parent_id": parent_id})
# Child 继承 Parent 的 metadata → 检索到 Child 就能定位到 Parent
```

### 2.5 两阶段检索流程

```
用户 Query
    │
    ▼
① search_child_chunks(query)
    → Qdrant 混合检索（稠密 + 稀疏）
    → 返回前 7 个 Child Chunk + 它们的 parent_id
    │
    ▼
② Agent 判断：Child Chunk 信息是否足够？
    │
    ├─ 足够 → 直接基于 Child 内容回答
    │
    └─ 不够 → retrieve_parent_chunks(parent_id)
                → 从 JSON 文件加载完整 Parent Chunk（2000~4000 字符）
                → 获得完整上下文
```

**这是 Agentic 的关键**：Agent 自己决定何时需要更多上下文，而不是系统预设的固定逻辑。

---

## 三、混合检索（Hybrid Retrieval）

### 3.1 双路嵌入

```python
dense  = HuggingFaceEmbeddings("Qwen/Qwen3-Embedding-0.6B")  # 语义匹配
sparse = FastEmbedSparse("Qdrant/bm25")                        # 关键词匹配

# Qdrant 同时创建稠密和稀疏向量配置
client.create_collection(
    vectors_config=qmodels.VectorParams(size=1024, distance=COSINE),
    sparse_vectors_config={"sparse": qmodels.SparseVectorParams()},
)

# LangChain Qdrant 的 HYBRID 模式自动融合两路
QdrantVectorStore(
    retrieval_mode=RetrievalMode.HYBRID,
    sparse_vector_name="sparse"
)
```

### 3.2 稠密 vs 稀疏的互补

| | 稠密（Dense） | 稀疏（Sparse/BM25） |
|---|---|---|
| 原理 | 神经网络语义编码 | 词频-逆文档频率统计 |
| 优势 | 同义词语义匹配，"汽车"能找到"轿车" | 专有名词精确匹配，"Qdrant"就是"Qdrant" |
| 盲区 | 对冷门术语、代码、数字不敏感 | 不理解语义，找不到同义词 |
| 融合方式 | Qdrant 内部加权合并两路分数 | — |

**面试金句**："混合检索用稠密向量覆盖语义泛化，用稀疏向量覆盖精确关键词，两者的互补性解决了纯向量检索对专有名词和代码片段不敏感的问题。"

---

## 四、Agentic 循环设计

### 4.1 双层 LangGraph 架构

```
主图 (State)                          Agent 子图 (AgentState)
═══════════                           ══════════════════════════
                                      ┌─────────────────┐
summarize_history                     │  orchestrator   │←──────┐
    │                                 │  (LLM + Tools)  │       │
    ▼                                 └───────┬─────────┘       │
rewrite_query                                  │ 条件路由         │
    │                               ┌──────────┼──────────┐     │
    ├─ 清晰 → 并行 Agent 子图        │          │          │     │
    ├─ 模糊 → request_clarification  ▼          ▼          ▼     │
    │          (中断等人输入)       tools   fallback  collect   │
    │                              │          │          │      │
    ▼                              ▼          │          │      │
aggregate_answers             should_compress  │          │      │
    │                              │           │          │      │
    ▼                          ┌───┴───┐       │          │      │
   END                         │压缩后  │       │          │      │
                               │回编排器│       │          │      │
                               └───────┘       │          │      │
                                               ▼          │      │
                                          (达到上限时)     │      │
                                               │          │      │
                                               └──────────┘      │
                                                    │            │
                                                    ▼            │
                                              compress_context ─┘
```

### 4.2 Agent 子图循环决策

```
orchestrator (LLM 决策)
    │
    ├─ tool_calls 存在 + 未超限 → tools 执行 → should_compress_context
    │                                              │
    │                          token 超限? ──是──→ compress_context → orchestrator
    │                               │
    │                              否→ orchestrator（继续循环）
    │
    ├─ tool_calls 存在 + 已超限 (MAX_TOOL_CALLS=8, MAX_ITERATIONS=10)
    │         → fallback_response → collect_answer
    │
    └─ 无 tool_calls（LLM 认为证据充分）
              → collect_answer
```

### 4.3 强制首次检索

关键设计：Agent 子图启动时，`orchestrator` 注入一条 **YOU MUST CALL 'search_child_chunks' AS THE FIRST STEP** 指令。不是让 LLM 自由发挥——**先检索，后推理，禁止凭记忆回答**。

```python
if not state.get("messages"):
    force_search = HumanMessage(
        content="YOU MUST CALL 'search_child_chunks' AS THE FIRST STEP TO ANSWER THIS QUESTION."
    )
    response = llm_with_tools.invoke([sys_msg] + summary + [human_msg, force_search])
```

### 4.4 上下文压缩 + 去重追踪

Agent 循环多轮后会积累大量 Tool Message，撑爆 context window。`compress_context` 节点：

1. 用 LLM 把当前轮次的检索结果总结为结构化摘要（按源文件分组 + Gaps 部分）
2. 记录 `retrieval_keys` 集合：`{"search::已查过的query", "parent::已拉取的parent_id"}`
3. 摘要末尾追加：**Already executed (do NOT repeat):** 列表
4. 之后 `orchestrator` 看到这个摘要就知道**别再搜同样的东西**

```python
# 摘要末尾的防重复块
block = "\n\n---\n**Already executed (do NOT repeat):**\n"
block += "Parent chunks retrieved:\n" + "\n".join(f"- {p}" for p in parent_ids)
block += "Search queries already run:\n" + "\n".join(f"- {q}" for q in search_queries)
```

### 4.5 Token 感知的动态压缩阈值

不是固定值压缩，而是动态计算：

```python
max_allowed = BASE_TOKEN_THRESHOLD + int(current_token_summary * TOKEN_GROWTH_FACTOR)
# 第一次: 2000 + 0 = 2000
# 第二次（summary 500 tokens）: 2000 + 450 = 2450
# 第三次（summary 1000 tokens）: 2000 + 900 = 2900
```

越到后面允许越大的上下文窗口——因为摘要本身就是有价值的上下文，不该过早压缩。

---

## 五、查询改写与多子问题拆分

### 5.1 结构化输出

`rewrite_query` 用 `llm.with_structured_output(QueryAnalysis)` 强制 LLM 输出 JSON：

```python
class QueryAnalysis(BaseModel):
    is_clear: bool          # 问题是否明确
    questions: List[str]    # 改写后的子问题列表（最多 3 个）
    clarification_needed: str  # 不明确时需要什么信息
```

### 5.2 多点设计

1. **模糊检测边界**：只有模糊指代（"它"、"那个"、"上一个"）才算不清晰，不因为"之前没聊过这个专有名词"就拒绝搜索
2. **多子问题并行**：对复杂问题自动拆分，每个子问题用 LangGraph 的 `Send` API 并行起一个 Agent 子图
3. **人机回环**：主图配 `interrupt_before=["request_clarification"]`，用户输入后自动恢复

```python
# 并行启动子 Agent
return [
    Send("agent", {"question": q, "question_index": i})
    for i, q in enumerate(state["rewrittenQuestions"])
]
```

---

## 六、Project 模块化结构

从 Notebook 到工程项目的拆解映射：

| Notebook 章节 | Project 模块 |
|--------------|-------------|
| §1 环境配置 | `config.py` — 所有常量集中管理 |
| §2 LLM 初始化 | `core/rag_system.py` — LLM + Embedding + Qdrant 统一初始化 |
| §3 嵌入 | 同上 |
| §4 Qdrant 配置 | `db/vector_db_manager.py` — 集合创建、维度校验 |
| §5 PDF→Markdown | `core/document_manager.py` — 文档转换与加载 |
| §6 父子分块 | `document_chunker.py` — 完整的分块、合并、拆分、再平衡逻辑 |
| §7 工具定义 | `rag_agent/tools.py` — `ToolFactory` 封装两个检索工具 |
| §8 System Prompt | `rag_agent/prompts.py` — 6 组 Prompt 字符串 |
| §9 状态定义 | `rag_agent/graph_state.py` + `rag_agent/schemas.py` |
| §10 限制配置 | `config.py`（MAX_TOOL_CALLS, MAX_ITERATIONS 等） |
| §11 节点与路由 | `rag_agent/nodes.py` + `rag_agent/edges.py` |
| §12 图构建 | `rag_agent/graph.py` — `create_agent_graph()` 主入口 |
| §13 Gradio | `ui/gradio_app.py` + `core/chat_interface.py` |
| 可观测 | `core/observability.py` — Langfuse 集成 |

### 设计亮点

- **`ToolFactory` 模式**：把 Tool 创建和 Qdrant 集合解耦，注入依赖而非全局变量
- **`logged_node` 装饰器**：`core/execution_logger.py` 统一给所有节点加日志和耗时统计
- **`partial` 注入**：`graph.py` 用 `functools.partial` 把 LLM 实例注入节点，而非依赖全局状态

---

## 七、面试速记表

| 技术点 | 一句话总结 |
|--------|----------|
| **父子分块** | 小 chunk（500 字符）入向量库做精确检索，大 chunk（2000~4000 字符）存 JSON 做完整上下文；两阶段检索，Agent 自决何时拉父块 |
| **混合检索** | 稠密向量（Qwen3-0.6B）做语义泛化 + BM25 稀疏向量做关键词精确匹配，Qdrant HYBRID 模式自动融合 |
| **Agent 循环** | 8 次 tool call / 10 轮迭代硬限制，设置后即强制首次检索，避免 Agent "凭记忆回答" |
| **上下文压缩** | Token 超限时 LLM 按源文件结构化压缩，附带"已检索列表"防重复，阈值随摘要长度动态增长 |
| **查询改写** | `with_structured_output` 强制 JSON 结构化输出，自动拆多子问题 + 只对模糊指代标记不清晰 + 人机回环中断点 |
| **并行 Agent** | LangGraph `Send` API 对拆分后的子问题并行起 Agent 子图，每个有独立状态隔离 |
| **兜底机制** | `fallback_response` 在达到工具/迭代上限后基于已有检索证据生成最佳回答，不丢弃已获取信息 |
| **模块化工程** | `ToolFactory` 依赖注入 + `logged_node` 统一日志 + `functools.partial` LLM 注入，比 Notebook 版更符合工程规范 |
