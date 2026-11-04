# 本地知识库问答助手（RAG）

一个面向银行考试资料的本地知识库问答应用：按页解析 PDF、建立 BM25 + 向量混合索引，并在回答中展示可追溯引用。问答页面支持多轮对话和流式生成；系统提供本地引用格式与词语重合提示，帮助用户核对来源。

## 快速开始

当前应用以 Windows 和 Python 3.12 为主。知识 PDF、生成的索引、会话数据库和 API 密钥都由使用者本地提供，仓库不包含这些个人/授权资料。

```powershell
py -3.12 -m venv .lc
.\.lc\Scripts\python.exe -m pip install -r .\knowledge_assistant\requirements-app.txt
Copy-Item .env.example .env
```

编辑 `.env`，填入 OpenAI 兼容问答服务和所需的 Qwen API 凭据。之后依照[详细使用说明](knowledge_assistant/README.md)安装/配置 PDF 解析环境、导入有使用权的 PDF 并建立索引，再启动问答和审核工作台。

- 问答页面：`http://127.0.0.1:8090/`
- PDF 审核工作台：`http://127.0.0.1:8091/`

## 仓库内容

- `knowledge_assistant/`：问答服务、PDF 解析与索引工具、审核工作台和评测代码。
- `.env.example`：不含密钥的配置模板。
- `.gitignore`：排除本地密钥、PDF、知识库产物、虚拟环境和运行数据。

评测题集和报告可能含有从本地资料摘取的文字，默认保留在本地；只有在确认拥有相应再分发权时才应公开。

## 许可证

本仓库不附带开源许可证。代码的公开展示不代表授予复制、修改、分发或商用许可；原始 PDF、模型服务和第三方依赖分别受其各自许可约束。
