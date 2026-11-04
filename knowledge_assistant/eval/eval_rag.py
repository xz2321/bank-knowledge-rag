"""RAG 评测：金标集 → 调用 /chat → recall@k / 拒答准确率 / 引用准确率。

用法：
    python eval_rag.py --validate     # 只校验金标里的期望短语是否存在于索引
    python eval_rag.py --run          # 跑全量评测并输出报告
    python eval_rag.py --run --limit 5
    python eval_rag.py --run --judge  # 显式开启额外的 LLM 答案评审
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import load_env  # noqa: E402

load_env()

ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = ROOT / "knowledge_base_output" / "local_index.json"
MANIFEST_PATH = ROOT / "knowledge_base_output" / "manifest.json"
GOLD_PATH = Path(__file__).resolve().parent / "gold.json"
REPORT_DIR = Path(__file__).resolve().parent / "reports"

JUDGE_PROMPT = (
    "你是严格的阅卷人。请对照参考答案，给模型答案打分（0~1，可以是小数）：\n"
    "- 1.0：核心事实与参考答案一致，无编造、无矛盾；\n"
    "- 0.7~0.9：主要知识点正确，缺少少量细节；\n"
    "- 0.4~0.6：部分正确，有明显遗漏或偏差；\n"
    "- 0.1~0.3：大部分错误或答非所问；\n"
    "- 0.0：完全错误、拒答或空回答。\n"
    "只输出一行 JSON：{{\"score\": 0.x, \"verdict\": \"correct|partial|incorrect\", \"reason\": \"一句话理由\"}}\n\n"
    "问题：{question}\n参考答案：{reference}\n模型答案：{answer}"
)


def judge_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    from openai import OpenAI

    return OpenAI(api_key=key, base_url=os.getenv("OPENAI_BASE_URL") or None)


def configured_judge(enabled: bool):
    """Avoid even constructing an evaluator client unless explicitly requested."""
    return judge_client() if enabled else None


def judge_answer(client, question: str, reference: str, answer: str) -> dict:
    """用 LLM 对照参考答案打分；失败返回 None 分数，不影响其他指标。"""
    if client is None or not reference or not answer:
        return {"score": None, "verdict": "skipped", "reason": "无参考答案或未作答"}
    prompt = JUDGE_PROMPT.format(question=question, reference=reference, answer=answer[:2000])
    try:
        response = client.chat.completions.create(
            model=os.getenv("JUDGE_MODEL", os.getenv("CHAT_MODEL", "gpt-4o-mini")),
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
            timeout=120,
        )
        text = (response.choices[0].message.content or "").strip()
        match = re.search(r"\{.*\}", text, re.S)
        data = json.loads(match.group(0) if match else text)
        score = float(data.get("score", 0))
        return {
            "score": round(max(0.0, min(1.0, score)), 2),
            "verdict": str(data.get("verdict", "")),
            "reason": str(data.get("reason", ""))[:120],
        }
    except Exception as exc:
        return {"score": None, "verdict": "error", "reason": f"评审失败：{exc}"[:120]}


def load_gold(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_index() -> list[dict]:
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


def source_page_counts() -> dict[str, int]:
    """Map indexed document titles to the source PDF's physical page count."""
    if not MANIFEST_PATH.exists():
        return {}
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    counts = {}
    for item in manifest:
        if item.get("status") != "ok":
            continue
        # Uploaded names end in .pdf.pdf; index titles omit only the last suffix.
        title = Path(item.get("file", "")).stem
        counts[title] = int(item.get("pymupdf_pages", 0)) + int(item.get("mineru_pages", 0))
    return counts


def validate(gold: dict, chunks: list[dict]) -> list[str]:
    """检查每条期望短语是否真的能在索引里找到，避免评测集本身写错。"""
    problems: list[str] = []
    for item in gold["items"]:
        expected_source = item.get("expect_source", "")
        source_chunks = [
            chunk for chunk in chunks
            if not expected_source or expected_source in str(chunk.get("title", ""))
        ]
        if expected_source and not source_chunks:
            problems.append(f"{item['id']}: 期望来源在索引中不存在 -> {expected_source}")
        for phrase in item.get("expect_any", []):
            if item.get("generated_from") and phrase not in item.get("reference", ""):
                problems.append(f"{item['id']}: 自动生成题的期望短语未出现在参考摘录中 -> {phrase}")
            hit = any(
                phrase in chunk.get("content", "") or phrase in (chunk.get("metadata") or {}).get("parent", "")
                for chunk in source_chunks
            )
            if not hit:
                suffix = f"（来源：{expected_source}）" if expected_source else ""
                problems.append(f"{item['id']}: 期望短语未出现在指定来源索引中 -> {phrase}{suffix}")
    return problems


def call_chat(base_url: str, item: dict, timeout: float) -> tuple[dict, float]:
    payload = {
        "question": item["question"],
        "top_k": 5,
        "debug": True,
        "history": item.get("history", []),
    }
    started = time.perf_counter()
    response = requests.post(base_url.rstrip("/") + "/chat", json=payload, timeout=timeout)
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    return response.json(), elapsed


def evaluate_item(item: dict, reply: dict, elapsed: float, page_counts: dict[str, int] | None = None) -> dict:
    mode = reply.get("mode", "answer")
    answer = reply.get("answer", "")
    citations = reply.get("citations") or []
    retrieved = reply.get("retrieved") or []
    citation_check = reply.get("citation_check") or {}
    support_review = str(citation_check.get("support_review", ""))
    failed_closed = support_review.startswith("failed_closed")
    phrases = item.get("expect_any", [])

    retrieval_hit = any(phrase in (chunk.get("content") or "") for phrase in phrases for chunk in retrieved)
    citation_hit = any(
        phrase in (citation.get("snippet") or "")
        or phrase in (citation.get("heading") or "")
        or phrase in (citation.get("context") or "")
        for phrase in phrases
        for citation in citations
    )
    cited_ids = {int(value) for value in re.findall(r"\[(\d+)\]", answer)}
    citation_ids = {int(citation.get("id", 0)) for citation in citations}
    fabricated = sorted(cited_ids - citation_ids)
    expected_source = item.get("expect_source", "")
    cited_titles = [str(citation.get("title", "")) for citation in citations]
    retrieved_titles = [str(chunk.get("title", "")) for chunk in retrieved]
    source_hit = not expected_source or any(expected_source in title for title in cited_titles)
    page_counts = page_counts or {}
    invalid_pages = []
    for citation in citations:
        title, page = str(citation.get("title", "")), citation.get("page")
        if page is None:
            invalid_pages.append({"title": title, "page": None, "reason": "missing"})
        elif title in page_counts and not (1 <= int(page) <= page_counts[title]):
            invalid_pages.append({"title": title, "page": page, "reason": "out_of_range"})

    return {
        "id": item["id"],
        "type": item.get("type", "answer"),
        "question": item["question"],
        "mode": mode,
        "answered": mode == "answer" and not failed_closed,
        "refused": mode == "refuse" or failed_closed,
        "failed_closed": failed_closed,
        "best_score": reply.get("best_score"),
        "question_used": reply.get("question_used"),
        "retrieval_hit": retrieval_hit,
        "citation_hit": citation_hit,
        "expected_source": expected_source,
        "source_hit": source_hit,
        "retrieved_source_hit": not expected_source or any(expected_source in title for title in retrieved_titles),
        "citation_count": len(citations),
        "invalid_citation_pages": invalid_pages,
        "citation_mode": (reply.get("citation_check") or {}).get("mode", ""),
        "support_review": support_review,
        "support_check_status": (citation_check.get("support_check") or {}).get("status", ""),
        "uncited_claims": (citation_check.get("support_check") or {}).get("uncited_claims", []),
        "no_overlap_claims": (citation_check.get("support_check") or {}).get("no_overlap_claims", []),
        "fabricated_ids": fabricated,
        "latency": round(elapsed, 2),
        "answer_head": answer[:120],
    }


def summarize(rows: list[dict]) -> dict:
    answerable = [row for row in rows if row["type"] in {"answer", "multi"}]
    refusable = [row for row in rows if row["type"] == "refuse"]
    answered = [row for row in rows if row["answered"]]
    answer_attempts = [row for row in rows if row["mode"] == "answer"]
    judged = [row for row in answerable if row.get("judge_score") is not None]

    def ratio(numerator: int, denominator: int) -> float:
        return round(numerator / denominator, 4) if denominator else 0.0

    return {
        "total": len(rows),
        "answerable": len(answerable),
        "should_refuse": len(refusable),
        "recall_at_k": ratio(sum(1 for row in answerable if row["retrieval_hit"]), len(answerable)),
        "answer_rate": ratio(sum(1 for row in answerable if row["answered"]), len(answerable)),
        "false_refusal_rate": ratio(sum(1 for row in answerable if row["refused"]), len(answerable)),
        "refusal_accuracy": ratio(sum(1 for row in refusable if row["refused"]), len(refusable)),
        "citation_accuracy": ratio(
            sum(1 for row in answer_attempts if row["citation_hit"] and not row["fabricated_ids"] and row["citation_count"] > 0),
            len(answer_attempts),
        ),
        "expected_source_hit_rate": ratio(sum(1 for row in answerable if row.get("source_hit")), len(answerable)),
        "expected_source_retrieval_rate": ratio(sum(1 for row in answerable if row.get("retrieved_source_hit")), len(answerable)),
        "citation_page_validity": ratio(sum(1 for row in answered if not row.get("invalid_citation_pages")), len(answered)),
        "missing_or_invalid_citation_page_count": sum(len(row.get("invalid_citation_pages") or []) for row in answered),
        "fabricated_citation_count": sum(len(row["fabricated_ids"]) for row in answered),
        "citation_repaired_count": sum(1 for row in answered if row.get("citation_mode") == "repaired"),
        "support_review_verified": sum(1 for row in answer_attempts if row.get("support_review") == "verified"),
        "support_review_failed_closed": sum(1 for row in answer_attempts if str(row.get("support_review", "")).startswith("failed_closed")),
        "support_review_skipped": sum(1 for row in answer_attempts if row.get("support_review") == "skipped_no_model"),
        "support_check_pass_rate": ratio(sum(1 for row in answer_attempts if row.get("support_check_status") == "checked"), len([row for row in answer_attempts if row.get("support_check_status") in {"checked", "review"}])),
        "support_check_uncited_claim_count": sum(len(row.get("uncited_claims") or []) for row in answer_attempts),
        "support_check_no_overlap_claim_count": sum(len(row.get("no_overlap_claims") or []) for row in answer_attempts),
        "answer_quality_avg": round(sum(row["judge_score"] for row in judged) / len(judged), 3) if judged else None,
        "answer_correct_rate": ratio(sum(1 for row in judged if row["judge_score"] >= 0.8), len(judged)),
        "answer_partial_rate": ratio(sum(1 for row in judged if 0.4 <= row["judge_score"] < 0.8), len(judged)),
        "answer_wrong_rate": ratio(sum(1 for row in judged if row["judge_score"] < 0.4), len(judged)),
        "judged": len(judged),
        "avg_latency": round(sum(row["latency"] for row in rows) / len(rows), 2) if rows else 0,
        "max_latency": max((row["latency"] for row in rows), default=0),
    }


def print_report(summary: dict, rows: list[dict]) -> None:
    print("\n== 汇总 ==")
    print(f"题目数 {summary['total']}（可答 {summary['answerable']} / 应拒答 {summary['should_refuse']}）")
    print(f"recall@5            : {summary['recall_at_k']:.2%}")
    print(f"作答率（可答题）     : {summary['answer_rate']:.2%}")
    print(f"误拒率（可答题）     : {summary['false_refusal_rate']:.2%}")
    print(f"拒答准确率（应拒答） : {summary['refusal_accuracy']:.2%}")
    print(f"引用准确率（已作答） : {summary['citation_accuracy']:.2%}")
    print(f"期望来源引用命中率   : {summary.get('expected_source_hit_rate', 0):.2%}")
    print(f"期望来源召回命中率   : {summary.get('expected_source_retrieval_rate', 0):.2%}")
    print(f"引用页码有效率       : {summary.get('citation_page_validity', 0):.2%}（缺失/越界 {summary.get('missing_or_invalid_citation_page_count', 0)}）")
    print(f"编造引用编号         : {summary['fabricated_citation_count']}")
    print(f"引用自动修正         : {summary.get('citation_repaired_count', 0)} 题")
    print(f"来源支持复核         : 通过 {summary.get('support_review_verified', 0)} / 失败关闭 {summary.get('support_review_failed_closed', 0)} / 跳过 {summary.get('support_review_skipped', 0)}")
    print(f"引用支持检查         : 通过率 {summary.get('support_check_pass_rate', 0):.2%} / 未覆盖主张 {summary.get('support_check_uncited_claim_count', 0)} / 无词面重合 {summary.get('support_check_no_overlap_claim_count', 0)}")
    if summary.get("judged", 0) == 0:
        print("答案质量评审         : 未运行（默认关闭；需要时加 --judge）")
    if summary.get("answer_quality_avg") is not None:
        print(f"答案质量均分         : {summary['answer_quality_avg']:.2f}（评审 {summary['judged']} 题）")
        print(f"  完全正确(≥0.8)     : {summary['answer_correct_rate']:.2%}")
        print(f"  部分正确(0.4~0.8)  : {summary['answer_partial_rate']:.2%}")
        print(f"  明显错误(<0.4)     : {summary['answer_wrong_rate']:.2%}")
    print(f"平均耗时            : {summary['avg_latency']}s（最大 {summary['max_latency']}s）")

    problems = [
        row for row in rows
        if (row["type"] in {"answer", "multi"} and (not row["answered"] or not row["retrieval_hit"] or not row["citation_hit"]))
        or (row["type"] == "refuse" and not row["refused"])
        or (row["type"] in {"answer", "multi"} and (not row.get("source_hit") or row.get("invalid_citation_pages")))
        or row["fabricated_ids"]
        or (row.get("judge_score") is not None and row["judge_score"] < 0.8)
    ]
    if problems:
        print("\n== 需要关注 ==")
        for row in problems:
            flags = []
            if row["type"] in {"answer", "multi"} and not row["answered"]:
                flags.append("被拒答")
            if row["type"] in {"answer", "multi"} and not row["retrieval_hit"]:
                flags.append("召回未命中")
            if row["type"] in {"answer", "multi"} and not row["citation_hit"]:
                flags.append("引用未命中")
            if row["type"] in {"answer", "multi"} and not row.get("source_hit"):
                flags.append("期望来源未引用")
            if row.get("invalid_citation_pages"):
                flags.append("引用页码缺失/越界")
            if row["type"] == "refuse" and not row["refused"]:
                flags.append("应拒未拒")
            if row["fabricated_ids"]:
                flags.append("编造编号" + str(row["fabricated_ids"]))
            if row.get("judge_score") is not None and row["judge_score"] < 0.8:
                flags.append(f"答案{row['judge_score']:.2f}：{row.get('judge_reason', '')[:30]}")
            print(f"  {row['id']} [{row['type']}] {row['question'][:28]} -> {', '.join(flags)} | best={row['best_score']}")


def main() -> int:
    # Windows consoles may use a legacy code page that cannot print some
    # Unicode symbols in per-item status lines; keep the evaluation running.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="RAG 基线评测")
    parser.add_argument("--gold", type=Path, default=GOLD_PATH)
    parser.add_argument("--base-url", default="http://127.0.0.1:8090")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条，调试用")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--judge", action="store_true", help="显式启用 LLM 答案质量评审（会增加耗时和模型调用）")
    parser.add_argument("--no-judge", action="store_true", help=argparse.SUPPRESS)  # 兼容旧命令；现在默认已关闭
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("OLD", "NEW"), help="对比两份报告")
    parser.add_argument("--keep-sessions", action="store_true")
    args = parser.parse_args()

    if args.compare:
        old = json.loads(args.compare[0].read_text(encoding="utf-8"))
        new = json.loads(args.compare[1].read_text(encoding="utf-8"))
        keys = [
            "recall_at_k", "answer_rate", "false_refusal_rate", "refusal_accuracy",
            "citation_accuracy", "citation_repaired_count", "answer_quality_avg",
            "answer_correct_rate", "support_review_verified", "support_review_failed_closed",
            "support_check_pass_rate", "support_check_uncited_claim_count", "avg_latency",
        ]
        print(f"{'指标':<20}{'旧':>10}{'新':>10}{'变化':>10}")
        for key in keys:
            before = old["summary"].get(key)
            after = new["summary"].get(key)
            if before is None and after is None:
                print(f"{key:<32}{'-':>10}{'-':>10}{'-':>10}")
                continue
            before_text = f"{before:.4f}" if before is not None else "—"
            after_text = f"{after:.4f}" if after is not None else "—"
            delta_text = f"{after - before:+.4f}" if before is not None and after is not None else "—"
            print(f"{key:<32}{before_text:>10}{after_text:>10}{delta_text:>10}")
        old_rows = {row["id"]: row for row in old["items"]}
        print("\n逐题变化：")
        for row in new["items"]:
            previous = old_rows.get(row["id"])
            if not previous:
                continue
            changed = []
            if previous["mode"] != row["mode"]:
                changed.append(f"mode {previous['mode']}→{row['mode']}")
            if previous["retrieval_hit"] != row["retrieval_hit"]:
                changed.append(f"召回 {previous['retrieval_hit']}→{row['retrieval_hit']}")
            if previous["citation_hit"] != row["citation_hit"]:
                changed.append(f"引用 {previous['citation_hit']}→{row['citation_hit']}")
            if changed:
                print(f"  {row['id']}: " + "，".join(changed))
        return 0

    gold = load_gold(args.gold)
    items = gold["items"][: args.limit] if args.limit else gold["items"]

    if args.validate:
        problems = validate(gold, load_index())
        print("金标校验：", "全部通过" if not problems else f"{len(problems)} 条问题")
        for problem in problems:
            print("  -", problem)
        if not args.run:
            return 1 if problems else 0

    if not args.run:
        print("加 --run 执行评测")
        return 0

    rows: list[dict] = []
    sessions: list[str] = []
    page_counts = source_page_counts()
    judge_enabled = bool(args.judge and not args.no_judge)
    judge = configured_judge(judge_enabled)
    print(f"开始运行 {len(items)} 题；答案评审模型：{'开启' if judge_enabled else '关闭'}")
    for index, item in enumerate(items, 1):
        try:
            reply, elapsed = call_chat(args.base_url, item, args.timeout)
        except Exception as exc:
            rows.append({
                "id": item["id"], "type": item.get("type", "answer"), "question": item["question"],
                "mode": "error", "answered": False, "refused": False, "best_score": None,
                "question_used": None, "retrieval_hit": False, "citation_hit": False,
                "citation_count": 0, "fabricated_ids": [], "latency": 0,
                "judge_score": None, "judge_verdict": "error", "judge_reason": "请求失败",
                "answer_head": f"请求失败：{exc}",
            })
            print(f"[{index}/{len(items)}] {item['id']} 请求失败：{exc}")
            continue
        row = evaluate_item(item, reply, elapsed, page_counts)
        row["gold_generated"] = bool(item.get("generated_from"))

        # 答案正确性：可答题才评审；被拒答的直接记 0 分
        if row["type"] in {"answer", "multi"}:
            if row["answered"]:
                judged = judge_answer(judge, item["question"], item.get("reference", ""), reply.get("answer", ""))
            elif judge is not None:
                judged = {"score": 0.0, "verdict": "incorrect", "reason": "未作答"}
            else:
                judged = {"score": None, "verdict": "skipped", "reason": "答案评审已关闭"}
            row["judge_score"] = judged["score"]
            row["judge_verdict"] = judged["verdict"]
            row["judge_reason"] = judged["reason"]
        else:
            row["judge_score"] = None
            row["judge_verdict"] = "n/a"
            row["judge_reason"] = ""

        rows.append(row)
        if reply.get("session_id"):
            sessions.append(reply["session_id"])
        mark = "OK " if (row["retrieval_hit"] if row["type"] != "refuse" else row["refused"]) else "!! "
        score_text = f" 质量{row['judge_score']:.2f}" if row["judge_score"] is not None else ""
        print(f"[{index}/{len(items)}] {mark}{item['id']} {item['question'][:26]} | mode={row['mode']} best={row['best_score']} 引用{row['citation_count']}{score_text} {row['latency']}s")

    summary = summarize(rows)
    print_report(summary, rows)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report = {
        "created_at": stamp,
        "gold": str(args.gold),
        "gold_version": gold.get("version"),
        "gold_item_count": len(items),
        "judge_enabled": judge_enabled,
        "gold_quality_audit": {
            "generated_item_count": sum(1 for item in items if item.get("generated_from")),
            "generated_retrieval_miss_ids": [row["id"] for row in rows if row.get("gold_generated") and not row.get("retrieval_hit")],
            "generated_citation_miss_ids": [row["id"] for row in rows if row.get("gold_generated") and not row.get("citation_hit")],
            "generated_expected_source_miss_ids": [row["id"] for row in rows if row.get("gold_generated") and not row.get("source_hit")],
        },
        "summary": summary,
        "items": rows,
    }
    (REPORT_DIR / f"{stamp}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / "latest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已保存：{REPORT_DIR / (stamp + '.json')}")

    if not args.keep_sessions:
        for session_id in sessions:
            try:
                requests.delete(args.base_url.rstrip("/") + "/sessions/" + session_id, timeout=20)
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
