"""Expand the smoke-test gold set with source-grounded questions from the active index.

The added answers are short verbatim excerpts from indexed source text. Questions
are generated from explicit definitions, classifications, formula labels, and
section headings; no model is called. Review the generated entries before treating
them as a permanent benchmark.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GOLD = Path(__file__).with_name("gold.json")
INDEX = ROOT / "knowledge_base_output" / "local_index.json"
TARGET_ADDITIONS = 160
PER_SOURCE = 11
GENERIC_TOPICS = {
    "内容", "特点", "原则", "公式", "方法", "定义", "概念", "因素", "问题",
    "主要", "大体上", "比较", "微观", "宏观", "信息", "用途", "内涵",
    "效果", "领导", "管理", "决策", "信用", "资产", "利润", "应当",
    "可以", "反映", "包括", "正确", "直接", "前者", "后者", "后者是",
    "一种商品", "种商品", "变化程度", "复杂程度", "资金来源", "常见方法",
}

SPACE = re.compile(r"\s+")
TAGS = re.compile(r"<[^>]+>")
FACT_CUE = re.compile(r"是指|指的是|包括|分为|可分为|反映|遵循")
TOPIC_CUE = re.compile(r"(?:是指|指的是|包括|分为|可分为|反映|遵循)")
HEADING = re.compile(r"^(?:#{1,6}\s*)?((?:第[一二三四五六七八九十\d]+[章节部分]|[一二三四五六七八九十]+[、：:]|考点\s*\d+|\d+(?:\.\d+){1,3})?.{2,36})$")


def clean(value: str) -> str:
    value = TAGS.sub(" ", value)
    value = value.replace("\u200b", "").replace("\xa0", " ")
    return SPACE.sub(" ", value).strip(" #*|\t\r\n")


def topic_and_question(sentence: str, headings: list[str]) -> tuple[str, str] | None:
    sentence = clean(sentence).strip(" 。；;，,")
    if not (16 <= len(sentence) <= 180):
        return None
    cue = TOPIC_CUE.search(sentence)
    topic = ""
    question = ""
    if cue:
        full_prefix = sentence[:cue.start()].strip()
        pieces = re.split(r"[。；，,]", full_prefix)
        prefix = pieces[-1]
        colon_parts = re.split(r"[：:]", prefix)
        generic = re.compile(r"^(?:主要|一般|通常|大体上|还|也|则|可|能|应该|包括|是)?$")
        if len(colon_parts) > 1 and generic.fullmatch(colon_parts[-1].strip()):
            prefix = colon_parts[-2]
        elif len(colon_parts) > 1:
            prefix = colon_parts[-1]
        prefix = re.sub(r"[（(][^（）()]*[）)]", "", prefix)
        prefix = re.sub(r"^(?:其中|而|并且|此外|同时|因此|在.+?中|所谓的?|主要|大体上|观点\S{0,8}认为)", "", prefix)
        prefix = prefix.strip(" —–-·_（）()【】[]{}：:，,。；;、")
        prefix = re.sub(r"^[（(）)\d一二三四五六七八九十、.\s-]+", "", prefix).strip()
        if 2 <= len(prefix) <= 22:
            topic = prefix
            verb = cue.group(0)
            if verb in {"包括", "分为", "可分为"}:
                question = f"{topic}{'包括哪些内容' if verb == '包括' else '分为哪几类'}？"
            elif verb == "反映":
                question = f"{topic}反映什么？"
            else:
                question = f"{topic}是什么？"
    if not topic and "=" in sentence:
        formula = re.match(r"^([^=：:]{2,18})\s*=", sentence)
        if formula:
            topic = clean(formula.group(1)).strip()
            question = f"{topic}的计算公式是什么？"
    if not topic:
        colon = re.match(r"^([\u4e00-\u9fffA-Za-z0-9（）()·]{2,18})[：:]\s*(.+)$", sentence)
        if colon and re.search(r"有|包括|分为|是|指|由", colon.group(2)):
            topic = clean(colon.group(1)).strip()
            question = f"{topic}是什么？"
    topic = topic.strip(" ：:，,。；;（）()—–-•·【】[]{}<>")
    if not (2 <= len(topic) <= 22 and question and len(question) <= 45):
        return None
    if topic in {"内容", "特点", "原则", "公式", "方法", "定义", "概念", "因素", "问题", "主要", "大体上", "比较", "微观", "宏观"}:
        return None
    if re.search(r"(?:因此|大体上|主要|观点|不应|被|应该|的)$", topic) or re.search(r"[\d：:（）(){}]", topic) or topic.startswith(("、", "）", "(")):
        return None
    return topic, question


def clean_candidate(topic: str, question: str, reference: str) -> bool:
    """Reject fragments, pronouns, formula variables, and OCR/HTML residue."""
    if not (2 <= len(topic) <= 18 and 8 <= len(question) <= 36 and 20 <= len(reference) <= 150):
        return False
    if topic not in reference or topic in GENERIC_TOPICS:
        return False
    if not re.search(r"[\u4e00-\u9fff]", topic):
        return False
    if re.search(r"[\d$∆Δ∑√=+*/#【】{}<>•·|：:（）()]", topic):
        return False
    if re.search(r"<\w+|colspan|rowspan|\uFFFD|\?{2,}|\*{2,}", reference, re.I):
        return False
    if re.match(r"^(?:因为|所以|因此|后者|前者|其中|并且|而|将|应当|主要|大体|以便|对|从|根据此)", topic):
        return False
    if re.search(r"(?:的|是|应当|应该|可以|包括|并|与|和|的情况下)$", topic):
        return False
    if re.search(r"(?:前者|后者|这类|该类|它|其中|上述|本身)$", topic):
        return False
    return True


def candidates_by_source(chunks: list[dict]) -> dict[str, list[dict]]:
    found: dict[str, list[dict]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    source_text: dict[str, str] = defaultdict(str)
    for chunk in chunks:
        source_text[str(chunk.get("title", ""))] += "\n" + str(chunk.get("content", ""))
    for chunk in chunks:
        title = str(chunk.get("title", ""))
        metadata = chunk.get("metadata") or {}
        if not title or int(metadata.get("page") or 0) <= 1:
            continue
        text = str(chunk.get("content", ""))
        lines = [clean(line) for line in text.splitlines()]
        headings = [line for line in lines if 3 <= len(line) <= 42 and HEADING.match(line)]
        sentences = []
        for line in lines:
            if not line or line.startswith(("答案", "解析", "[A]", "[B]", "[C]", "[D]")):
                continue
            sentences.extend(re.split(r"(?<=[。！？；])", line))
        # Keep facts first; the heading fallback supplies a prompt for fact-rich
        # paragraphs that use a colon or table row instead of a definition verb.
        for sentence in sentences:
            sentence = clean(sentence).strip(" 。；;，,")
            if not (18 <= len(sentence) <= 180):
                continue
            if not FACT_CUE.search(sentence) and "=" not in sentence:
                continue
            parsed = topic_and_question(sentence, headings)
            if not parsed:
                continue
            topic, question = parsed
            # Repair a few known mid-word page/chunk cuts when the intact term
            # occurs elsewhere in the same source document.
            if topic.startswith("争") and "竞" + topic in source_text[title]:
                old_topic = topic
                topic = "竞" + topic
                question = question.replace(old_topic, topic)
            if not clean_candidate(topic, question, sentence):
                continue
            key = (title, question)
            if key in seen:
                continue
            # The evaluation phrase is taken from the topic named in the query.
            phrase = topic if topic in text else topic[-min(8, len(topic)):]
            if len(phrase) < 2 or phrase not in text or phrase not in sentence:
                continue
            seen.add(key)
            found[title].append({
                "question": question,
                "expect_any": [phrase],
                "expect_source": title,
                "reference": sentence[:240],
                "source_page": metadata.get("page"),
                "generated_from": "indexed_excerpt",
            })
    return found


def main() -> int:
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    gold["items"] = [item for item in gold["items"] if not item.get("generated_from")]
    chunks = json.loads(INDEX.read_text(encoding="utf-8"))
    candidates = candidates_by_source(chunks)
    existing_keys = {(str(item.get("expect_source", "")), item["question"]) for item in gold["items"]}
    additions: list[dict] = []
    selected_by_source: dict[str, list[dict]] = defaultdict(list)
    for title in sorted(candidates):
        source_items = [item for item in candidates[title] if (title, item["question"]) not in existing_keys]
        for item in source_items[:PER_SOURCE]:
            additions.append(item)
            selected_by_source[title].append(item)
    if len(additions) < TARGET_ADDITIONS:
        def prior_coverage(title: str) -> int:
            return sum(
                1 for item in gold["items"]
                if item.get("expect_source") and str(item["expect_source"]) in title
            )

        extras = {
            title: [item for item in candidates[title][PER_SOURCE:] if (title, item["question"]) not in existing_keys]
            for title in candidates
        }
        while len(additions) < TARGET_ADDITIONS:
            choices = [title for title in extras if extras[title]]
            if not choices:
                break
            title = min(choices, key=lambda name: (prior_coverage(name) + len(selected_by_source[name]), name))
            item = extras[title].pop(0)
            additions.append(item)
            selected_by_source[title].append(item)
    if len(additions) < TARGET_ADDITIONS:
        raise SystemExit(f"Only generated {len(additions)} viable source-grounded questions; refusing to pad the set.")
    for i, item in enumerate(additions[:TARGET_ADDITIONS], 1):
        item["id"] = f"s{i:03d}"
        item["type"] = "answer"
    gold["version"] = 5
    gold["description"] = "200题RAG分层评测集：原40题快速回归用例 + 160道经规则过滤、与参考摘录短语对齐的来源绑定题。新增题由当前知识库索引生成，仍建议抽样人工复核。"
    gold["items"].extend(additions[:TARGET_ADDITIONS])
    GOLD.write_text(json.dumps(gold, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(gold['items'])} questions ({len(additions[:TARGET_ADDITIONS])} added).")
    for title in sorted(candidates):
        count = sum(1 for item in additions[:TARGET_ADDITIONS] if item["expect_source"] == title)
        print(f"{count:>3}  {title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
