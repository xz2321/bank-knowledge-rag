"""Build a local searchable index from MinerU Markdown output.

Set OPENAI_API_KEY and IMAGE_MODEL to enable optional image understanding for
images found below knowledge_base_output. Without a key, MinerU's OCR text is
still indexed normally.
"""
from __future__ import annotations

import argparse
import base64
import bisect
import json
import os
import re
from collections import Counter
from pathlib import Path

from config import load_env

ROOT = Path(__file__).resolve().parents[1]
load_env()

# MinerU 把内嵌图片（data URI）换成的占位符，视觉描述会写回这些位置。
IMAGE_PLACEHOLDER = "[图片已解析，原图未嵌入文本]"


def _image_sort_key(path: Path) -> tuple[int, str]:
    """按文件名里的编号排序，避免 embedded_10 排在 embedded_2 前面。"""
    match = re.search(r"(\d+)", path.stem)
    return (int(match.group(1)) if match else 10**9, path.stem)


def extract_embedded_images(markdown_path: Path, output_dir: Path, prefix: str = "embedded") -> int:
    """Extract MinerU data-URI images before Markdown cleanup removes them."""
    text = markdown_path.read_text(encoding="utf-8", errors="replace")
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r"data:image/(png|jpe?g|webp);base64,([A-Za-z0-9+/=]+)", re.IGNORECASE)
    count = 0
    for index, match in enumerate(pattern.finditer(text), 1):
        ext = match.group(1).lower().replace("jpeg", "jpg")
        target = output_dir / f"{prefix}_{index}.{ext}"
        target.write_bytes(base64.b64decode(match.group(2)))
        count += 1
    return count


def inline_image_notes(target: Path, notes: dict[str, str]) -> int:
    """把视觉模型描述按顺序写回 Markdown 的图片占位符。

    第 N 个占位符对应第 N 张图片（``image_notes`` 会为识别失败/不支持的图片保留
    空字符串占位），所以某张图失败也不会让后面的描述整体错位。
    占位符独占一行时保留换行（公式/表格不被压成一行），行内时压成一行，
    避免破坏表格结构。
    """
    if not notes:
        if target.exists():
            text = target.read_text(encoding="utf-8")
            if IMAGE_PLACEHOLDER in text:
                target.write_text(text.replace(IMAGE_PLACEHOLDER, ""), encoding="utf-8")
        return 0
    text = target.read_text(encoding="utf-8")
    parts = text.split(IMAGE_PLACEHOLDER)
    if len(parts) < 2:
        return 0
    values = list(notes.values())
    count = min(len(parts) - 1, len(values))
    out: list[str] = [parts[0]]
    written = 0
    for index in range(count):
        body = (values[index] or "").strip()
        before = parts[index]
        if not body:
            # 无可转写文字时不把系统占位说明写进正文。
            out.append("")
        elif before.endswith("\n") or not before.strip():
            out.append(f"{body}\n")
            written += 1
        else:
            flat = " ".join(body.split())
            out.append(flat)
            written += 1
        out.append(parts[index + 1])
    for index in range(count, len(parts) - 1):
        out.append("")
        out.append(parts[index + 1])
    target.write_text("".join(out), encoding="utf-8")
    return written


def chunks(text: str, size: int = 1200, overlap: int = 160) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    result: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= size:
            current = f"{current}\n\n{paragraph}".strip()
        else:
            if current:
                result.append(current)
            tail = current[-overlap:] if current else ""
            current = f"{tail}\n\n{paragraph}".strip()
    if current:
        result.append(current)
    return result


IMAGE_PROMPT = (
    "只还原图片本身的信息，不要任何解释、总结或背景知识。"
    "图中有公式/换算关系时，按图中的关系写成等式（例如：A = B × C）。"
    "其余文字、标注、表格数据按从上到下的顺序逐行原样列出，表格用 | 分隔单元格。"
    "不要写“这张图片展示”“以下是”之类的句子，也不要加标题。"
    "图中没有可辨识的文字、公式或表格数据时，只输出空内容，不要描述图形。"
)


def image_notes(image_dir: Path) -> dict[str, str]:
    """Describe images with an OpenAI-compatible vision endpoint when enabled."""
    key = os.getenv("OPENAI_API_KEY")
    if not key or not image_dir.exists():
        return {}
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=os.getenv("OPENAI_BASE_URL") or None)
    except Exception as exc:
        print(f"跳过图片大模型：{exc}")
        return {}
    model = os.getenv("IMAGE_MODEL", "gpt-4o-mini")
    notes: dict[str, str] = {}
    raster = {".png", ".jpg", ".jpeg", ".webp"}
    extra = {".gif", ".bmp", ".svg", ".tif", ".tiff"}
    images = [
        item
        for item in image_dir.rglob("*")
        if item.is_file() and item.suffix.lower() in raster | extra
    ]
    for image in sorted(images, key=_image_sort_key):
        if image.suffix.lower() not in raster:
            # 矢量图等不支持的格式：占住位置，但不送给视觉模型。
            notes[image.name] = ""
            continue
        data = base64.b64encode(image.read_bytes()).decode("ascii")
        mime = "image/png" if image.suffix.lower() == ".png" else "image/jpeg"
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": IMAGE_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
                ]}],
            )
            notes[image.name] = response.choices[0].message.content or ""
        except Exception as exc:
            print(f"图片理解失败 {image.name}: {exc}")
            notes[image.name] = ""
    return notes


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def normalize_for_match(text: str) -> str:
    """页码匹配用：去掉标题标记/HTML/标点空白，只保留中英文数字。"""
    text = re.sub(r"^#{1,6}.*$", " ", text or "", flags=re.MULTILINE)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"[^\u4e00-\u9fff0-9A-Za-z]", "", text)


def find_pdf(source: str) -> Path | None:
    """按知识库文件名找原始 PDF（兼容 xxx.pdf 和 xxx.pdf.pdf 两种命名）。"""
    review_root = ROOT / "knowledge_base_output" / "review_workspace"
    candidates = [
        ROOT / source,
        ROOT / f"{source}.pdf",
        ROOT / f"{source}.pdf.pdf",
        ROOT / "knowledge_base_output" / source,
        ROOT / "knowledge_base_output" / f"{source}.pdf",
        ROOT / "knowledge_base_output" / f"{source}.pdf.pdf",
    ]
    if review_root.is_dir():
        for job in review_root.iterdir():
            if job.is_dir():
                candidates.extend((job / source, job / f"{source}.pdf", job / f"{source}.pdf.pdf"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def page_index(pdf: Path) -> tuple[str, list[int], int]:
    """把 PDF 每页文本拼成一整串，并记录每页起始偏移，用于反查页码。"""
    import pymupdf

    parts: list[str] = []
    offsets: list[int] = []
    position = 0
    with pymupdf.open(pdf) as document:
        for page in document:
            text = normalize_for_match(page.get_text("text"))
            offsets.append(position)
            parts.append(text)
            position += len(text)
        total = document.page_count
    return "".join(parts), offsets, total


def locate_page(full: str, offsets: list[int], content: str) -> tuple[int | None, int | None]:
    """用多个 16 字窗口在 PDF 全文里定位子块，返回 (起始页, 结束页)，1-based。"""
    text = normalize_for_match(content)
    if not full or len(text) < 32:
        return None, None
    hits: list[int] = []
    for ratio in (0.0, 0.25, 0.5, 0.75):
        start = int(len(text) * ratio)
        piece = text[start:start + 16]
        index = full.find(piece)
        if index >= 0:
            hits.append(bisect.bisect_right(offsets, index))
    if not hits:
        return None, None
    page = Counter(hits).most_common(1)[0][0]
    return page, max(hits)


PAGE_MARKER_RE = re.compile(
    r"<!-- PyMuPDF page (\d+) -->|<!-- MinerU pages (\d+)-(\d+) -->"
)


def split_page_marked_text(text: str) -> list[tuple[int, int, str]]:
    """Return parser-marked page/range blocks, with the marker removed."""
    markers = list(PAGE_MARKER_RE.finditer(text))
    blocks: list[tuple[int, int, str]] = []
    for index, marker in enumerate(markers):
        start = int(marker.group(1) or marker.group(2))
        end = int(marker.group(1) or marker.group(3))
        stop = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        body = text[marker.end():stop].strip()
        if body:
            blocks.append((start, end, body))
    return blocks


def split_sidecar_page_text(text: str, sidecar: Path) -> list[tuple[int, int, str]]:
    """Read page ranges from metadata kept outside the clean Markdown text."""
    try:
        entries = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    blocks: list[tuple[int, int, str]] = []
    for item in entries if isinstance(entries, list) else []:
        try:
            start, end = int(item["start"]), int(item["end"])
            page_start, page_end = int(item["page_start"]), int(item["page_end"])
        except (KeyError, TypeError, ValueError):
            continue
        body = text[start:end].strip()
        if body:
            blocks.append((page_start, page_end, body))
    return blocks


def split_sections(text: str) -> list[tuple[str, str]]:
    """按 Markdown 标题切成小节，返回 [(标题, 正文)]；标题行保留在正文里。"""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    sections: list[tuple[str, str]] = []
    heading = ""
    buffer: list[str] = []
    for line in text.splitlines():
        if HEADING_RE.match(line):
            body = "\n".join(buffer).strip()
            if body:
                sections.append((heading, body))
            buffer = [line]
            heading = line.strip()
        else:
            buffer.append(line)
    body = "\n".join(buffer).strip()
    if body:
        sections.append((heading, body))
    return sections


def group_sections(sections: list[tuple[str, str]], min_size: int) -> list[list[tuple[str, str]]]:
    """把过短的小节合并成“父块组”，组内保留每一节自己的标题。

    这样父块可以整段喂给大模型，而子块仍然带着自己那一节的标题，
    避免引用里出现“标题写着上一节、内容却是下一节”的错位。
    """
    groups: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    total = 0
    for heading, body in sections:
        current.append((heading, body))
        total += len(body)
        if total >= min_size:
            groups.append(current)
            current, total = [], 0
    if current:
        if groups:
            groups[-1].extend(current)
        else:
            groups.append(current)
    return groups


def merge_small_sections(sections: list[tuple[str, str]], min_size: int) -> list[tuple[str, str]]:
    """兼容旧接口：合并后只保留组内第一个标题。"""
    result: list[tuple[str, str]] = []
    for group in group_sections(sections, min_size):
        head = group[0][0] if group else ""
        body = "\n\n".join(item for _, item in group if item)
        result.append((head, body))
    return result


def split_children(text: str, size: int, overlap: int) -> list[str]:
    """把一节内容切成用于召回的“子块”，段落边界优先，带少量重叠。"""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    children: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > size:
            # 超长段落（常见于表格/OCR 长行）按字符硬切
            if current:
                children.append(current)
                current = ""
            for start in range(0, len(paragraph), size - overlap):
                piece = paragraph[start:start + size]
                if piece.strip():
                    children.append(piece.strip())
            continue
        if len(current) + len(paragraph) + 2 <= size:
            current = f"{current}\n\n{paragraph}".strip()
        else:
            if current:
                children.append(current)
            tail = current[-overlap:] if current else ""
            current = f"{tail}\n\n{paragraph}".strip()
    if current:
        children.append(current)
    return children


def fill_estimated_pages(records: list[dict]) -> int:
    """给定位失败的子块补页码：用同文档里最近的已知页（文档顺序即页顺序）。

    补出来的页码会带 page_estimated=True，前端显示“约第 N 页”，与精确页码区分。
    """
    filled = 0
    index = 0
    while index < len(records):
        end = index
        while end < len(records) and records[end]["title"] == records[index]["title"]:
            end += 1
        group = records[index:end]
        known = [position for position, item in enumerate(group) if item["metadata"].get("page")]
        if known:
            for position, item in enumerate(group):
                if item["metadata"].get("page"):
                    continue
                before = max((k for k in known if k < position), default=None)
                after = min((k for k in known if k > position), default=None)
                source = before if before is not None else after
                item["metadata"]["page"] = group[source]["metadata"]["page"]
                item["metadata"]["page_end"] = group[source]["metadata"].get("page_end")
                item["metadata"]["page_estimated"] = True
                filled += 1
        index = end
    return filled


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "knowledge_base_output" / "markdown")
    parser.add_argument("--output", type=Path, default=ROOT / "knowledge_base_output" / "local_index.json")
    parser.add_argument("--chunk-size", type=int, default=int(os.getenv("CHUNK_SIZE", "600")), help="子块大小（召回单元）")
    parser.add_argument("--overlap", type=int, default=int(os.getenv("CHUNK_OVERLAP", "80")), help="子块重叠字符数")
    parser.add_argument("--parent-size", type=int, default=int(os.getenv("PARENT_SIZE", "2000")), help="父块（喂给大模型）最大字符数")
    parser.add_argument("--min-section", type=int, default=int(os.getenv("MIN_SECTION", "200")), help="小节短于该值时与相邻小节合并")
    args = parser.parse_args()
    records: list[dict[str, object]] = []
    for path in sorted(args.source.glob("*.md")):
        # 图片描述已经在生成阶段写进正文的【图 N】段，这里不再调用视觉模型，
        # 否则每次重建索引都会把所有图片重新识别一遍（几分钟 + 额外费用）。
        text = path.read_text(encoding="utf-8", errors="replace")
        source = path.name.removesuffix(".md")
        subject = source.split("-")[0]
        pdf = find_pdf(source)
        page_full, page_offsets, page_total = page_index(pdf) if pdf else ("", [], 0)
        print(f"{source}: PDF={pdf.name if pdf else '未找到'}，{page_total} 页")
        marked_blocks = split_sidecar_page_text(text, path.with_suffix(".pages.json"))
        if not marked_blocks:
            marked_blocks = split_page_marked_text(text)
        blocks = marked_blocks or [(None, None, text)]
        current_heading = ""
        for block_no, (marked_start, marked_end, block_text) in enumerate(blocks, 1):
            for section_no, group in enumerate(
                group_sections(split_sections(block_text), args.min_section), 1
            ):
                parent_id = f"{source}#p{marked_start or 'auto'}-{marked_end or ''}#{block_no}-{section_no}"
                parent_text = "\n\n".join(body for _, body in group)[: args.parent_size]
                fallback_heading = next((heading for heading, _ in group if heading), current_heading)
                pending: list[tuple[str, str, int | None, int | None]] = []
                for section_heading, section_body in group:
                    if section_heading:
                        current_heading = section_heading.lstrip("# ").strip()
                    heading_label = (section_heading or fallback_heading).lstrip("# ").strip()
                    for child in split_children(section_body, args.chunk_size, args.overlap):
                        if marked_start is not None:
                            page, page_end = marked_start, marked_end
                            if marked_start != marked_end:
                                matched_page, matched_end = locate_page(page_full, page_offsets, child)
                                if (matched_page == matched_end and matched_page is not None
                                        and marked_start <= matched_page <= marked_end):
                                    page = page_end = matched_page
                        else:
                            page, page_end = locate_page(page_full, page_offsets, child)
                        pending.append((heading_label, child, page, page_end))

                # Legacy Markdown without parser page markers still uses text matching.
                known = [page for _, _, page, _ in pending if page]
                fallback_page = Counter(known).most_common(1)[0][0] if known else None
                if fallback_page is None and marked_start is None:
                    fallback_page, _ = locate_page(page_full, page_offsets, parent_text)

                for heading_label, child, page, page_end in pending:
                    page = page or fallback_page
                    page_end = page_end or page
                    records.append({
                        "title": source,
                        "content": child,
                        "metadata": {
                            "source": source,
                            "subject": subject,
                            "heading": heading_label,
                            "parent_id": parent_id,
                            "parent": parent_text,
                            "page": page,
                            "page_end": page_end,
                            "page_estimated": False if marked_start is not None else None,
                            "page_source": "parser" if marked_start is not None else "text_match",
                        },
                    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    estimated = fill_estimated_pages(records)
    args.output.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    print(f"已建立本地索引：{len(records)} 个子块 -> {args.output}（其中 {estimated} 个页码为估算）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
