"""用 PDF 文字层重建 Markdown（替代 OCR 结果）。

适用：PDF 有完整文字层、但 MinerU 的 OCR 识别质量差（缺字、<sub> 标签）。
做法：PyMuPDF 逐行取字 + 字号判标题 + 相邻行合并 + 重复行去重。
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
SKIP_HEADERS = {"综合知识", "银行招聘笔试 · 学习资料"}
PARAGRAPH_END = ("。", "！", "？", "；", "：", "）", "】")
BULLET_PREFIX = ("•", "💡", "【", "★", "注：")


def clean(text: str) -> str:
    return text.replace("\u200b", "").replace("\xa0", " ").strip()


def extract_pages(pdf: Path) -> list[list[dict]]:
    pages: list[list[dict]] = []
    with pymupdf.open(pdf) as document:
        for page in document:
            lines: list[dict] = []
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    spans = [span for span in line["spans"] if span["text"].strip()]
                    if not spans:
                        continue
                    text = clean("".join(span["text"] for span in spans))
                    if not text or text in SKIP_HEADERS:
                        continue
                    size = round(max(span["size"] for span in spans), 1)
                    if size < 8.0:
                        continue
                    lines.append({"text": text, "size": size})
            pages.append(lines)
    return pages


def dedupe(lines: list[dict]) -> list[dict]:
    """去掉重复绘制的行：只比较相邻行，短行是长行前缀时保留更长的一份。

    只比相邻可以避免误删正文中合法重复的短语，同时能处理这份 PDF
    “先画一遍截断的灰字、再画一遍完整正文”的排版。
    """
    result: list[dict] = []
    for line in lines:
        norm = re.sub(r"\s+", "", line["text"])
        if result:
            previous_norm = re.sub(r"\s+", "", result[-1]["text"])
            if previous_norm and norm:
                if norm.startswith(previous_norm) and len(norm) > len(previous_norm):
                    result[-1] = line
                    continue
                if previous_norm.startswith(norm) and len(previous_norm) > len(norm):
                    continue
                if norm == previous_norm:
                    continue
        result.append(line)
    return result


def build_markdown(pages: list[list[dict]], title: str) -> str:
    blocks: list[tuple[str, str]] = []
    seen_headings: set[str] = set()
    buffer = ""

    def heading_text(text: str) -> str:
        text = re.sub(r"^(模块[一二三四五六七八九十百]+)(?!\s)", r"\1 ", text)
        return re.sub(r"\s*([★☆]+)\s*$", r" \1", text)

    def flush() -> None:
        nonlocal buffer
        if buffer.strip():
            blocks.append(("body", buffer.strip()))
        buffer = ""

    for lines in pages:
        for line in dedupe(lines):
            text = line["text"].strip("•·- ")
            if not text or text in {"•", "·", "-"}:
                continue
            normalized = re.sub(r"\s+", "", text)
            if line["size"] >= 14.0:
                if normalized in seen_headings:
                    continue
                seen_headings.add(normalized)
                flush()
                blocks.append(("h1", heading_text(text)))
            elif line["size"] >= 12.0:
                if normalized in seen_headings:
                    continue
                seen_headings.add(normalized)
                flush()
                blocks.append(("h2", heading_text(text)))
            elif text.startswith(BULLET_PREFIX):
                flush()
                buffer = text
            elif not buffer:
                buffer = text
            elif buffer.endswith(PARAGRAPH_END):
                flush()
                buffer = text
            else:
                buffer += text
        flush()

    out = [f"# {title}", ""]
    for kind, text in blocks:
        if kind == "h1":
            out += ["", f"## {text}", ""]
        elif kind == "h2":
            out += ["", f"### {text}", ""]
        else:
            out += [text, ""]
    return "\n".join(out).strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    title = args.title or args.output.name.removesuffix(".md")
    markdown = build_markdown(extract_pages(args.pdf), title)
    args.output.write_text(markdown, encoding="utf-8")
    print(f"已生成 {args.output}（{len(markdown)} 字符，{markdown.count(chr(10)) + 1} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

\n