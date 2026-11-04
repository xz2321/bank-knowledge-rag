# 本地知识库问答助手

这个目录提供一条本地入库管线：使用 MinerU 解析 PDF，保留图片并提取图片/图表中的文字，生成 Markdown 并建立本地检索索引，供自带的问答页面使用。

## 环境配置（`.env`）

复制项目根目录的 `.env.example` 为 `.env` 并填入自己的凭据。入口脚本会自动载入该文件（见 `knowledge_assistant/config.py`），不需要再手动 `$env:...`。**系统环境变量优先于 `.env`**。`.env` 已加入忽略规则，不应提交。

示例使用 OpenAI 兼容的 DeepSeek 配置。具体模型能力和可用名称取决于服务商当前提供的接口；请按自己的服务商配置问答与图片理解模型：

```dotenv
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.deepseek.com/v1
CHAT_MODEL=deepseek-flash
IMAGE_MODEL=deepseek-flash
```

- `CHAT_MODEL`：问答生成模型。
- `IMAGE_MODEL`：图片/图表理解模型，审核工作台每次上传 PDF 后会用它为提取出的图片追加“图片大模型理解”段落。
- 换服务商时只改这三行即可（任何 OpenAI 兼容端点）。注意 `.env` 里有密钥，不要提交到代码仓库或发给他人。

## 使用

应用脚本统一使用项目根目录的 `.lc` Python 环境（上传工作台、PyMuPDF/MinerU 路由、索引、问答和评测）。MinerU CLI 保留在 `.mineru` 环境，由应用子进程调用；不要用 `.mineru\Scripts\python.exe` 运行应用脚本。

首次创建应用环境：

```powershell
py -3.12 -m venv .lc
.\.lc\Scripts\python.exe -m pip install -r .\knowledge_assistant\requirements-app.txt
```

在 PowerShell 中运行：

```powershell
.\.lc\Scripts\python.exe .\knowledge_assistant\build_knowledge_base.py
```

默认读取当前目录的所有 `*.pdf` 和 `*.pdf.pdf`，输出到 `knowledge_base_output`。

参数：

```powershell
.\.lc\Scripts\python.exe .\knowledge_assistant\build_knowledge_base.py --tier basic
.\.lc\Scripts\python.exe .\knowledge_assistant\build_knowledge_base.py --files "财务会计-财务管理(基础版).pdf.pdf"
```

`standard` 对图片、表格和公式更好，但首次运行可能下载模型且速度较慢；4GB 显存建议先用 `basic` 验证，再用 `standard` 处理重点资料。

每个文件只包含 MinerU 解析出的正文（不额外补标题、学科/来源元信息和“## 正文”之类的样板段落），避免这些文字占用检索分块；图片文件保存在同目录的 `assets` 子目录中，视觉模型识别结果直接写回原来的图片位置。

## MinerU WebUI

当前 WebUI 地址：`http://127.0.0.1:7860/`。
如需重新启动：

```powershell
.\.mineru\Scripts\mineru-kit.exe webui --host 127.0.0.1 --port 7860
```

## 建立本地索引

```powershell
.\.lc\Scripts\python.exe .\knowledge_assistant\build_local_index.py
```

输出 `knowledge_base_output/local_index.json`（当前 14 篇、约 808 个分块），问答页和向量混合检索都读这个文件。索引重建约 2 秒。

## 图片大模型理解

在建立索引时设置 OpenAI 兼容的大模型环境变量，服务会读取 `assets` 图片并把视觉描述追加到文本分块：

```powershell
$env:OPENAI_API_KEY="你的API密钥"
$env:OPENAI_BASE_URL="https://api.openai.com/v1"   # 国内兼容服务可替换
$env:IMAGE_MODEL="gpt-4o-mini"
.\.lc\Scripts\python.exe .\knowledge_assistant\build_local_index.py
```

没有配置密钥时仍会建立索引，只使用 MinerU 的 OCR 和版面解析文本。索引只依赖本地文件，不需要额外的向量数据库或检索服务。

### 给已有 Markdown 补图片描述

MinerU 会把内嵌图片换成占位符 `[图片已解析，原图未嵌入文本]`。新生成的文档会自动用视觉模型把图中内容写回原位置；历史文档可以补跑：

```powershell
.\.lc\Scripts\python.exe .\knowledge_assistant\apply_vision.py            # 知识库 + 审核任务全部补齐
.\.lc\Scripts\python.exe .\knowledge_assistant\apply_vision.py --only kb  # 只补知识库 markdown
.\.lc\Scripts\python.exe .\knowledge_assistant\apply_vision.py --file "路径.md"
```

补完后重建索引才会生效：`python .\knowledge_assistant\build_local_index.py`。识别失败的图片会保留占位符但位置不动，不会让后面的图文错位。

## 问答助手（LangChain LCEL 版）

建立索引后启动问答页面（`.lc` 同时包含应用运行、PyMuPDF、评测及上传所需依赖）：

```powershell
.\.lc\Scripts\python.exe .\knowledge_assistant\lc_qa_app.py
```

浏览器打开 `http://127.0.0.1:8090/`。如需补齐依赖：

```powershell
.\.lc\Scripts\python.exe -m pip install -r .\knowledge_assistant\requirements-app.txt
```

### LangChain 组件与链路

| 环节 | 实现 |
|---|---|
| 多轮改写 | `ChatPromptTemplate \| ChatOpenAI(temperature=0) \| StrOutputParser` |
| 向量召回 | 自定义 `BaseRetriever`（Qdrant + 千问 embedding，失败自动返回空） |
| 词法召回 | 自定义 `BaseRetriever`（包一层现有 jieba + BM25） |
| 融合 | 自定义 `FusionRetriever`：两路各自排名 → RRF（`k=60`，权重 0.5/0.5） |
| 重排 | `RunnableLambda` 包住千问 rerank，失败保留融合顺序 |
| 生成 | `ChatPromptTemplate`（system + 历史消息 + 资料） `\|` `ChatOpenAI` `\|` `StrOutputParser` |
| 拒答/引用校验 | 复用 `qa_assistant` 中已验证的实现（阈值、罕见词证据、编号校验） |

注意：

- 对外分数仍是 0~1 可比分（向量余弦 / BM25 饱和分），RRF 分只用于排序，避免破坏 `MIN_RELEVANCE` 语义。
- 向量或重排不可用时会自动降级（向量 → 纯 BM25；重排 → 保留融合顺序），日志里会打印降级原因。
- 需要回滚到旧实现：停掉 8090 后改跑 `.\.lc\Scripts\python.exe .\knowledge_assistant\qa_assistant.py`（旧版加权融合）。

### 问答链路细节

- **意图分流**：寒暄（你好/谢谢/再见）、自我介绍类问题（“你是谁”“怎么用”）走固定短回复，不检索、不挂引用，返回 `mode="chat"`；避免给“你好”配上三条无关出处。
- **拒答阈值**：`MIN_RELEVANCE`（默认 `0.35`）。最高重排分低于阈值时直接返回“知识库中没有找到依据”，不调用大模型、不列来源。实测域外问题（天气 0.0465）会被拦下，约 4 秒返回。
- **引用到片段**：答案中用 `[1] [2]` 标注依据，接口返回 `citations`（标题、章节、相关度、原文片段、`source_path`、`parent_id`、`page/page_end`）。片段不是简单截前 N 字，而是从父块里按句子用 BM25 挑出与问题最相关的一句，再取它前后窗口——避免答案句落在块末尾时被截掉；完全没命中“罕见查询词”的引用会被丢弃。
- **表格清洗**：MinerU 会把表格转成 HTML，片段里先按行转成“含义：…；公式：…”的可读文本，再删掉残余标签，避免引用里出现 `<td>` 或孤立的“含义/公式”单元格。
- **页码溯源**：建索引时用 PyMuPDF 抽取 PDF 每页文本，按多个 16 字窗口定位每个子块，写入 `page/page_end`；表格块匹配不到时用同一父块其他子块或整节文本兜底。当前 780 个子块中 **772 个有页码（99%）**，引用卡片显示“第 57 页”“第 94–95 页”。
- **引用校验**：生成后强制校验编号——删掉答案里不存在的 `[n]`、丢弃没被引用的引用、把编号重排为连续；接口返回 `citation_check`（`mode` = ok/repaired、被删编号、是否重排），前端展示来源文件路径。
- **流式输出**：问答页面通过 `/chat/stream` 接收检索/生成状态，并在模型生成时逐段显示答案；生成完成后仅做本地引用编号和主张覆盖检查，不额外调用模型复核。原 `/chat` JSON 接口保留给评测与旧客户端使用。
- **多轮对话**：会话存 SQLite（刷新/重启不丢）。追问会先被改写成可独立检索的问句（`question_used` 字段可看到），生成时也会带上最近 4 条对话。
- **父子分块**：按 Markdown 标题切小节，小节过短（`MIN_SECTION`，默认 200 字）自动合并成父块；子块仍保留**自己那一节的标题**（避免引用里出现“标题是上一节、内容是下一节”）。索引规模随本地导入的资料变化；父块（`PARENT_SIZE`，默认 2000）整体喂给大模型。
- **引用展示**：前端把答案按 Markdown 渲染（加粗/列表/标题），`[n]` 显示为可点击样式的引用标记；引用卡片分三行——文件名+相关度、章节+页码、清洗后的片段，最后一行小字是源文件路径。

### 评测（`eval/`）

改了分块、权重、阈值之后，靠“问几句看看”判断不出好坏，所以有一套基线评测：

```powershell
.\.lc\Scripts\python.exe .\knowledge_assistant\eval\eval_rag.py --validate  # 校验金标短语是否都在索引里
.\.lc\Scripts\python.exe .\knowledge_assistant\eval\eval_rag.py --run       # 跑全量
.\.lc\Scripts\python.exe .\knowledge_assistant\eval\eval_rag.py --run --limit 5
.\.lc\Scripts\python.exe .\knowledge_assistant\eval\eval_rag.py --run --judge  # 可选：额外调用LLM评审答案质量
```

- 金标：本地 `eval/gold.json` 共 200 题。该文件含有从本地资料摘取的内容，公共仓库默认不附带；确认有再分发权后可自行提供给评测脚本。
- 指标：`recall@5`、作答率、误拒率、拒答准确率、引用准确率、引用支持检查通过率、未覆盖主张数、编造引用编号数、引用自动修正数，以及 **答案正确性**（LLM 对照参考答案打分：完全正确 ≥0.8 / 部分正确 0.4~0.8 / 明显错误 <0.4）。平均耗时按端到端统计。
- 报告：本地生成到 `eval/reports/<时间>.json` 和 `latest.json`，默认被 Git 忽略，可能含资料摘录和回答内容。
- 对比：`--compare 旧报告 新报告`，输出指标 delta 和逐题变化。
- 评测接口参数加 `debug: true` 时会额外返回 `retrieved`（命中块内容），仅用于评测。

答案质量 LLM 评审默认关闭；需要离线抽查时显式添加 `--judge`。线上问答也不再调用模型做第二次依据复核，只执行本地引用编号和词语重合检查；这类检查只能提示需人工核对，不能证明语义正确。当前本地 v5 评测集由 40 道原有回归题和 160 道经规则过滤的来源题组成，覆盖 14 份资料。生成规则会剔除不完整/OCR 残片问题，并要求期望短语同时出现在参考摘录中。该题集、完整报告和逐题结果不随公共仓库发布；克隆后运行评测前，需要自行准备有权使用的 `eval/gold.json`，并先构建本地知识库索引。

## PDF 审核工作台

你可以上传新的 PDF，下载 MinerU/视觉大模型处理后的 Markdown，人工修改后再上传入库：

```powershell
.\.lc\Scripts\python.exe .\knowledge_assistant\review_app.py
```

打开 `http://127.0.0.1:8091/`：

1. 选择新的 PDF，点击“开始处理”。
2. 处理完成后，可点“直接入库”将解析结果加入知识库；也可下载 Markdown，人工核对 OCR、表格、公式和图片说明后，在第二栏上传审核后的版本。
3. 入库会复制 Markdown 到 `knowledge_base_output/markdown`、重建 `local_index.json`，并同步启用中的向量库。
4. 回到 `http://127.0.0.1:8090/` 使用更新后的知识库问答。

上传后的 Markdown 不会直接覆盖原 PDF；每次处理都会创建独立审核任务目录，便于回溯。

## Qdrant 混合检索 + 千问重排

当前已启用（配置在 `.env`）：**向量用千问、重排用千问、生成答案用 DeepSeek**。

流程：`问题` → 千问 `qwen3.7-text-embedding` 向量化 → 本地 Qdrant 召回 → 与关键词分数融合（`0.7/0.3`）→ 取一批候选交给千问 `qwen3.7-text-rerank` 重排 → DeepSeek Flash 生成答案。Qdrant 数据在 `knowledge_base_output/qdrant`，不需要单独起数据库服务。

```dotenv
QWEN_API_KEY=sk-...
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=qwen3.7-text-embedding      # 1024 维
RERANK_MODEL=qwen3.7-text-rerank
RERANK_URL=https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank
USE_QDRANT=1
```

几个实现细节：

- 千问 embedding 单次最多 **20 条**，`hybrid_retriever.py` 用 `EMBEDDING_BATCH`（默认 20）分批。
- 重排接口不在 OpenAI 兼容层，走 DashScope 原生地址；`rerank.py` 负责调用，失败自动回退到原顺序（`RERANK_ENABLED=0` 可临时关闭）。
- 重建向量库：`python .\knowledge_assistant\hybrid_retriever.py`（808 个分块约 1 分钟）。
- 换 embedding 模型（维度变化）必须重建集合，否则检索报维度不匹配。

### 分词与词法检索

词法部分使用 **jieba 搜索引擎模式分词 + BM25**（`retrieval.py`）：

- 分词：`jieba.lcut_for_search`，去掉停用词和纯标点，输出如 `['杜邦', '分析', '分析法', '核心', '公式', '净利润', '净资产']`。
- 打分：BM25（`k1=1.2`、`b=0.75`），带 IDF 和文档长度归一化；`score()` 仍返回 0~1（`raw/(raw+4)`）供 `0.7/0.3` 融合使用。
- 索引：问答检索使用预建倒排索引，单次打分约 **0.2 ms**（原来全量扫描约 1.2 s）。
- 缓存：倒排索引存 `knowledge_base_output/bm25_cache.pkl`，按 `local_index.json` 的 mtime/size 失效；首次建表约 4 秒，之后加载约 0.08 秒。
- `GET /health` 会返回 `tokenizer`、`terms`、`avgdl`，便于确认当前使用的是哪套检索。

没有配置千问密钥时，`USE_QDRANT` 不生效，助手回退到本地关键词召回。

### 用 PDF 文字层重建 Markdown（OCR 质量差时）

有的 PDF 本身有完整文字层，但 MinerU 走 OCR 后出现缺字、`<sub>` 标签（`银行招聘笔试·法律基础知识一本通` 就是典型）。这时不需要重做 OCR，直接按文字层重建：

```powershell
.\.lc\Scripts\python.exe .\knowledge_assistant\repair_from_pdf.py `
  --pdf ".\银行招聘笔试·法律基础知识一本通.pdf.pdf" `
  --output ".\knowledge_base_output\markdown\银行招聘笔试·法律基础知识一本通.pdf.md" `
  --title "银行招聘笔试·法律基础知识一本通"
```

做法：PyMuPDF 逐行取字 → 字号（≥14 一级标题、≥12 二级标题）判层级 → 行尾标点判段落 → 相邻重复行只留更长的一份。改完记得重建索引和向量库；原 OCR 版本已备份为 `*.md.ocr.bak`。
