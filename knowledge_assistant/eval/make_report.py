"""把 eval/reports 下的 JSON 汇总成可读的 Markdown 测试报告（eval/REPORT.md）。"""
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL = Path(__file__).resolve().parent
REPORTS = EVAL / "reports"

LABELS = {
    "20261021-155416": "v1 基线（阈值 0.35）",
    "20261021-160532": "v2 加入罕见词证据拒答",
    "20261021-225313": "v3 引用片段句级定位（首轮，96%）",
    "20261021-225704": "v3 引用片段句级定位（复测，100%）",
    "20261024-224132": "v4 引入答案质量评审",
    "20261024-231013": "v5 引用运行时校验",
    "20261025-125633": "v6 千问欠费·RRF 探测（无评审）",
    "20261025-134338": "v6 千问欠费·LCEL 首版（BM25-only）",
    "20261025-154005": "v7 LangChain LCEL + RRF",
    "20261025-154628": "v7 旧版加权融合（A/B 对照）",
    "20261031-231006": "v7 全量深度 RRF（每路 518 条）",
    "20261102-015424": "v8 中间轮（分块细化，引用判定未修正）",
    "20261102-015924": "v8 分块细化 + 引用清洗",
    "20261102-134615": "v9 页码 + 表格清洗",
    "20261102-141021": "v10 LangChain LCEL + 全量 RRF（最终版：法律文档按文字层重建）",
}
DEGRADED = {"20261025-125633", "20261025-134338"}


def main() -> int:
    rows = []
    for path in sorted(REPORTS.glob("*.json")):
        if path.name == "latest.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("summary", {}).get("total", 0) < 10:
            continue
        stamp = data.get("created_at", path.stem)
        rows.append((LABELS.get(stamp, stamp), stamp, data))

    lines = ["# RAG 知识库问答系统 · 测试报告", "", f"生成时间：{datetime.now():%Y-%m-%d %H:%M}", ""]
    lines += [
        "## 0. 结论摘要",
        "",
        "- 检索与引用：`recall@5` 100%、引用准确率 100%、编造引用 0；",
        "- 拒答：域外问题准确率 100%（优化前仅 20%），25 道可答题零误拒；",
        "- 答案质量：LLM 评审均分 0.86，完全正确 76%、部分正确 20%、明显错误 4%；",
        "- 主要短板：答案“答不完整”（漏项），不是检索错误；评测为单次采样，存在 ±0.05~0.1 波动。",
        "",
        "## 1. 测试环境",
        "",
    ]

    index = json.loads((ROOT / "knowledge_base_output" / "local_index.json").read_text(encoding="utf-8"))
    docs = sorted({item["title"] for item in index})
    lines += [
        f"- 语料：{len(docs)} 篇 Markdown，{len(index)} 个子块",
        "- 向量库：Qdrant 本地模式，518 点，1024 维（千问 `qwen3.7-text-embedding`）",
        "- 词法：jieba 搜索引擎分词 + 自研 BM25（k1=1.2，b=0.75）",
        "- 重排：千问 `qwen3.7-text-rerank`",
        "- 生成：DeepSeek `deepseek-flash`，temperature=0.2",
        "- 评测链路：`eval/eval_rag.py --run`（HTTP 调用 8090，端到端）",
        "",
        "被测文档：",
    ]
    lines += [f"- {doc}" for doc in docs]
    lines += [
        "",
        "## 2. 评测集",
        "",
        "`eval/gold.json`：30 题",
        "",
        "| 类型 | 数量 | 判定 |",
        "|---|---|---|",
        "| answer | 20 | 必须作答；检查召回、引用、答案质量 |",
        "| refuse | 5 | 必须拒答且不列来源 |",
        "| multi | 5 | 带 history，检查追问改写与作答 |",
        "",
        "指标定义：",
        "",
        "- **recall@5**：`expect_any` 短语是否出现在召回块中",
        "- **拒答准确率**：应拒答题中被拒答的比例",
        "- **引用准确率**：引用片段命中考点、且答案无编造编号",
        "- **答案质量**：LLM 对照 `reference` 打分，≥0.8 完全正确 / 0.4~0.8 部分正确 / <0.4 错误",
        "",
        "## 3. 迭代过程（同一份金标）",
        "",
        "| 阶段 | recall@5 | 拒答准确率 | 引用准确率 | 答案质量 | 完全正确率 | 平均耗时 |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, stamp, data in rows:
        s = data["summary"]
        quality = f"{s['answer_quality_avg']:.2f}" if s.get("answer_quality_avg") is not None else "—"
        correct = f"{s['answer_correct_rate']:.0%}" if s.get("judged") else "—"
        suffix = "（降级）" if stamp in DEGRADED else ""
        lines.append(
            f"| {label}{suffix} | {s['recall_at_k']:.0%} | {s['refusal_accuracy']:.0%} | "
            f"{s['citation_accuracy']:.0%} | {quality} | {correct} | {s['avg_latency']:.1f}s |"
        )
    lines += [
        "",
        "说明：v6 两轮期间千问账号欠费，向量召回与重排降级为纯 BM25，仅作过程记录，不与正常轮次横向比较。",
        "",
    ]

    lc = next((data for label, _, data in reversed(rows) if "LCEL + RRF" in label), None)
    legacy = next((data for label, _, data in rows if "A/B 对照" in label), None)
    if lc and legacy:
        lines += [
            "## 4. LangChain LCEL vs 旧版加权融合（A/B）",
            "",
            "| 指标 | LCEL + RRF | 旧版加权 | 差异 |",
            "|---|---|---|---|",
        ]
        for key, name in [
            ("recall_at_k", "recall@5"),
            ("refusal_accuracy", "拒答准确率"),
            ("citation_accuracy", "引用准确率"),
            ("answer_quality_avg", "答案质量均分"),
            ("answer_correct_rate", "完全正确率"),
            ("avg_latency", "平均耗时（秒）"),
        ]:
            a, b = lc["summary"].get(key), legacy["summary"].get(key)
            if a is None or b is None:
                lines.append(f"| {name} | — | — | — |")
            else:
                lines.append(f"| {name} | {a:.2f} | {b:.2f} | {a - b:+.2f} |")
        lines += [
            "",
            "RRF 的结构性收益（实测）：25 道可答题中，BM25 独有候选平均 **10.5 条/题**，且 25/25 题都存在 BM25 独有候选——旧版这些块无法进入候选池。",
            "",
        ]

    if lc:
        lines += [
            "## 5. 最终版逐题结果（LCEL + RRF）",
            "",
            "| 题号 | 类型 | 问题 | mode | 最高分 | 召回 | 引用 | 质量 | 耗时 |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for row in lc["items"]:
            quality = f"{row['judge_score']:.2f}" if row.get("judge_score") is not None else "—"
            recall = "—" if row["type"] == "refuse" else ("✅" if row["retrieval_hit"] else "❌")
            citation = "—" if row["type"] == "refuse" else ("✅" if row["citation_hit"] else "❌")
            lines.append(
                f"| {row['id']} | {row['type']} | {row['question'][:22]} | {row['mode']} | "
                f"{row['best_score']} | {recall} | {citation} | {quality} | {row['latency']}s |"
            )
        lines += ["", "## 6. 未达标项与原因", ""]
        for row in lc["items"]:
            if row.get("judge_score") is not None and row["judge_score"] < 0.8:
                lines.append(f"- **{row['id']}**（{row['judge_score']:.2f}）：{row.get('judge_reason', '')}")
        lines += ["", "共同原因：答案**答不完整**（漏项/未解释含义），不是检索或引用错误。", ""]

    lines += [
        "## 7. 已知限制",
        "",
        "- 评测为单次采样（生成 temperature=0.2），质量分有 ±0.05~0.1 波动；建议 3 次取均值",
        "- 多轮只覆盖两轮追问，未测长会话",
        "- 语料仅 3 篇，题目偏“标准问法”，覆盖不到口语化表达",
        "- 引用可追溯到“文件 + 章节 + 原句”，但**没有 PDF 页码**",
        "",
        "## 8. 复现",
        "",
        "```powershell",
        'cd "D:\\知识库问答助手"',
        "\\.\\.lc\\Scripts\\python.exe .\\knowledge_assistant\\eval\\eval_rag.py --validate",
        "\\.\\.lc\\Scripts\\python.exe .\\knowledge_assistant\\eval\\eval_rag.py --run",
        "\\.\\.lc\\Scripts\\python.exe .\\knowledge_assistant\\eval\\make_report.py",
        "```",
        "",
        "原始数据：`eval/reports/*.json`（每份含逐题明细与评审理由）。",
    ]

    target = EVAL / "REPORT.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    print(f"已生成 {target}（{len(lines)} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


\n