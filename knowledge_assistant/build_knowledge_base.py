from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from build_local_index import (
    IMAGE_PLACEHOLDER,
    extract_embedded_images,
    image_notes,
    inline_image_notes,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "knowledge_base_output"
DEFAULT_MINERU_MAX_PAGES_PER_RUN = 6
PAGE_MARKER_RE = re.compile(
    r"<!-- PyMuPDF page (\d+) -->|<!-- MinerU pages (\d+)-(\d+) -->"
)
FOOTER_LINE_RE = re.compile(
    r"^\s*重点可视化\s*[·•・]\s*内容保真\s*[·•・]\s*非学习信息已清理"
    r"(?:\s+\d{1,4}\s*/\s*\d{1,4})?\s*$"
)
PAGE_COUNTER_LINE_RE = re.compile(r"^\s*\d{1,4}\s*/\s*\d{1,4}\s*$")


def split_page_runs(page_numbers: list[int], max_pages: int = DEFAULT_MINERU_MAX_PAGES_PER_RUN) -> list[list[int]]:
    """Group selected pages into contiguous, bounded MinerU input ranges."""
    if max_pages < 1:
        raise ValueError("max_pages must be positive")
    runs: list[list[int]] = []
    for page_no in sorted(set(page_numbers)):
        if not runs or page_no != runs[-1][-1] + 1 or len(runs[-1]) >= max_pages:
            runs.append([])
        runs[-1].append(page_no)
    return runs


def classify_page(text: str, words: list, image_area_ratio: float = 0.0) -> tuple[bool, set[str]]:
    """Conservative page router based on text, strong math glyphs, and layout rows.

    Table rows are inferred from separate text blocks sharing a baseline, avoiding
    PyMuPDF's expensive table finder and avoiding ordinary justified paragraphs.
    """
    reasons: set[str] = set()
    if len(text.strip()) < 80:
        reasons.add("sparse_text")
    strong_math = len(re.findall(r"[∑∫√≤≥≈÷∞±]|\\(?:frac|sum|int|sqrt)", text))
    if strong_math >= 2:
        reasons.add("formula")
    if image_area_ratio >= 0.02:
        reasons.add("image")

    rows: dict[int, dict[int, list[float]]] = {}
    for word in words:
        # PyMuPDF word tuple: x0,y0,x1,y1,text,block_no,line_no,word_no
        row = round(word[1] / 3)
        block = int(word[5]) if len(word) > 5 else round(word[0] / 10)
        rows.setdefault(row, {}).setdefault(block, []).append(float(word[0]))
    table_rows = []
    for blocks in rows.values():
        starts = sorted(min(xs) for xs in blocks.values())
        if len(starts) >= 2 and max(starts[i + 1] - starts[i] for i in range(len(starts) - 1)) >= 10:
            table_rows.append(starts)
    if len(table_rows) >= 3:
        positions = [round(start / 12) for row in table_rows for start in row]
        recurring = {position for position in set(positions) if positions.count(position) >= 3}
        if len(recurring) >= 2:
            reasons.add("table")
    return bool(reasons), reasons


def clean_routed_markdown(text: str) -> tuple[str, list[dict[str, int]]]:
    """Remove parser markers and repeated PDF footer lines, preserving page spans separately."""
    markers = list(PAGE_MARKER_RE.finditer(text))
    blocks: list[tuple[int | None, int | None, str]] = []
    if markers:
        for index, marker in enumerate(markers):
            page_start = int(marker.group(1) or marker.group(2))
            page_end = int(marker.group(1) or marker.group(3))
            stop = markers[index + 1].start() if index + 1 < len(markers) else len(text)
            blocks.append((page_start, page_end, text[marker.end():stop]))
    else:
        blocks.append((None, None, text))

    output: list[str] = []
    page_map: list[dict[str, int]] = []
    cursor = 0
    for page_start, page_end, block in blocks:
        block = block.replace(IMAGE_PLACEHOLDER, "")
        lines = [line for line in block.splitlines()
                 if not FOOTER_LINE_RE.match(line) and not PAGE_COUNTER_LINE_RE.match(line)]
        body = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
        if not body:
            continue
        if output:
            cursor += 2  # separator: two newlines between output blocks
        start = cursor
        output.append(body)
        cursor += len(body)
        if page_start is not None and page_end is not None:
            page_map.append({"start": start, "end": cursor, "page_start": page_start, "page_end": page_end})
    return ("\n\n".join(output).strip() + "\n" if output else "\n"), page_map


def route_pdf_pages(pdf: Path, raw_dir: Path, tier: str) -> tuple[Path, dict[str, int]]:
    """Extract straightforward text pages with PyMuPDF; send complex pages to MinerU.

    MinerU is run on contiguous runs of complex pages, preserving page order and
    avoiding one subprocess per page. The returned Markdown is a merged document.
    """
    import pymupdf

    raw_dir.mkdir(parents=True, exist_ok=True)
    merged = raw_dir / f"{pdf.stem}.md"
    simple_pages: list[tuple[int, str]] = []
    complex_pages: list[int] = []
    reason_counts: dict[str, int] = {}
    with pymupdf.open(pdf) as doc:
        for page_no, page in enumerate(doc):
            text = page.get_text("text", sort=True).strip()
            words = page.get_text("words")
            page_area = max(page.rect.width * page.rect.height, 1)
            image_area = sum(
                max(0, min(info["bbox"][2], page.rect.width) - max(info["bbox"][0], 0))
                * max(0, min(info["bbox"][3], page.rect.height) - max(info["bbox"][1], 0))
                for info in page.get_image_info()
            )
            complex_page, reasons = classify_page(text, words, image_area / page_area)
            for reason in reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            (complex_pages if complex_page else simple_pages).append(
                page_no if complex_page else (page_no, text)
            )

    # Parse each contiguous complex run as a small PDF, so mixed PDFs retain
    # page-level routing without losing MinerU's layout/table/formula handling.
    page_markdown: dict[int, str] = {n: f"<!-- PyMuPDF page {n + 1} -->\n{text}" for n, text in simple_pages}
    max_run_pages = int(os.getenv("MINERU_MAX_PAGES_PER_RUN", DEFAULT_MINERU_MAX_PAGES_PER_RUN))
    runs = split_page_runs(complex_pages, max_run_pages)
    with pymupdf.open(pdf) as original:
        for run_no, pages in enumerate(runs, 1):
            part_pdf = raw_dir / f"{pdf.stem}_complex_{run_no}.pdf"
            part = pymupdf.open()
            for page_no in pages:
                part.insert_pdf(original, from_page=page_no, to_page=page_no)
            part.save(part_pdf)
            part.close()
            run_dir = raw_dir / f"complex_{run_no}"
            parsed = run_mineru(part_pdf, run_dir, tier)
            content = parsed.read_text(encoding="utf-8", errors="replace").strip()
            source_images = parsed.parent / "images"
            if source_images.exists():
                image_dir = raw_dir / "images"
                image_dir.mkdir(parents=True, exist_ok=True)
                for image_no, image in enumerate(source_images.rglob("*"), 1):
                    if image.is_file():
                        renamed = f"run{run_no}_{image_no}_{image.name}"
                        shutil.copy2(image, image_dir / renamed)
                        content = content.replace(image.name, renamed)
            # MinerU output is retained as a grouped run; page comments identify
            # its original page span for reviewers and downstream citations.
            page_markdown[pages[0]] = f"<!-- MinerU pages {pages[0] + 1}-{pages[-1] + 1} -->\n{content}"
    merged.write_text("\n\n".join(page_markdown[n] for n in sorted(page_markdown)) + "\n", encoding="utf-8")
    return merged, {
        "pymupdf_pages": len(simple_pages), "mineru_pages": len(complex_pages),
        "mineru_runs": len(runs), "max_mineru_run_pages": max((len(run) for run in runs), default=0),
        **{f"route_{key}_pages": value for key, value in sorted(reason_counts.items())},
    }


def clean_text(value: str) -> str:
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按页路由 PyMuPDF/MinerU 批量生成本地知识库 Markdown")
    parser.add_argument("--tier", choices=("flash", "basic", "standard", "advanced"), default="basic")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--files", nargs="*", help="只处理指定文件名")
    return parser.parse_args()


def run_mineru(pdf: Path, raw_dir: Path, tier: str) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ROOT / ".mineru" / "Scripts" / "mineru-kit.exe"),
        "parse",
        str(pdf),
        "--output",
        str(raw_dir / f"{pdf.stem}.md"),
        "--tier",
        tier,
        "--ocr-mode",
        "auto",
    ]
    completed = subprocess.run(cmd, cwd=ROOT, capture_output=True)
    stdout = completed.stdout.decode("utf-8", errors="replace")
    stderr = completed.stderr.decode("utf-8", errors="replace")
    if completed.returncode != 0:
        raise RuntimeError(stderr[-3000:] or stdout[-3000:])
    target = raw_dir / f"{pdf.stem}.md"
    if not target.exists() or target.stat().st_size == 0:
        detail = (stderr or stdout)[-1500:]
        raise FileNotFoundError(f"MinerU 未生成 Markdown: {pdf.name}\n{detail}")
    return target


def make_markdown(
    pdf: Path,
    source_md: Path,
    target: Path,
    asset_dir: Path,
    describe_images: bool = True,
) -> None:
    text = source_md.read_text(encoding="utf-8", errors="replace")
    target.parent.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)
    source_assets = source_md.parent / "images"
    if source_assets.exists():
        for asset in source_assets.rglob("*"):
            if asset.is_file():
                shutil.copy2(asset, asset_dir / asset.name)
    # 保留图片引用，同时把相对路径统一为本地入库目录中的 assets 路径。
    # 正文里不放超大的 base64 图片，图里的文字由视觉模型写回原位置。
    text = re.sub(
        r"!\[([^\]]*)\]\(data:image/[^;]+;base64,[^\)]+\)",
        IMAGE_PLACEHOLDER,
        text,
    )
    # MinerU may place base64 images inside HTML tables instead of Markdown
    # image syntax. Remove those payloads as well so the document stays text-sized
    # while retaining the surrounding OCR/table text.
    text = re.sub(
        r"<img\b[^>]*src=[\"']data:image/[^;]+;base64,[^\"']+[\"'][^>]*>",
        IMAGE_PLACEHOLDER,
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"!\[([^\]]*)\]\((?:images/)?([^\)]+)\)", r"![\1](assets/\2)", text)
    # 只输出 MinerU 解析出的正文：不额外加标题、学科/来源元信息、
    # 图片说明和“## 正文”分节标题，避免入库内容被样板文字稀释。
    # Apply optional visual notes before removing page markers so their insertions
    # are included in the offsets saved for exact page-to-text mapping.
    target.write_text(text, encoding="utf-8")
    if describe_images:
        # 图片描述直接写回正文，图片本身不需要长期保留在 assets 里。
        with tempfile.TemporaryDirectory() as tmp:
            extract_embedded_images(source_md, Path(tmp), prefix="img")
            inline_image_notes(target, image_notes(Path(tmp)))
    text, page_map = clean_routed_markdown(target.read_text(encoding="utf-8", errors="replace"))
    target.write_text(clean_text(text), encoding="utf-8")
    page_map_path = target.with_suffix(".pages.json")
    if page_map:
        page_map_path.write_text(json.dumps(page_map, ensure_ascii=False, indent=2), encoding="utf-8")
    elif page_map_path.exists():
        page_map_path.unlink()


def main() -> int:
    args = parse_args()
    out = args.output.resolve()
    raw_dir = out / "mineru_raw"
    md_dir = out / "markdown"
    asset_dir = md_dir / "assets"
    out.mkdir(parents=True, exist_ok=True)
    pdfs = sorted({p.resolve() for p in ROOT.iterdir() if p.is_file() and p.name.lower().endswith(".pdf")})
    if args.files:
        wanted = set(args.files)
        pdfs = [p for p in pdfs if p.name in wanted]
    if not pdfs:
        print("没有找到待处理 PDF", file=sys.stderr)
        return 2
    log = []
    for index, pdf in enumerate(pdfs, 1):
        print(f"[{index}/{len(pdfs)}] 处理 {pdf.name}", flush=True)
        try:
            source, route_stats = route_pdf_pages(pdf, raw_dir, args.tier)
            target = md_dir / f"{pdf.stem}.md"
            make_markdown(pdf, source, target, asset_dir)
            log.append({"file": pdf.name, "status": "ok", "markdown": str(target), **route_stats})
            print(
                f"  完成: {target}（PyMuPDF {route_stats['pymupdf_pages']} 页，"
                f"MinerU {route_stats['mineru_pages']} 页，{route_stats['mineru_runs']} 段，"
                f"最长 {route_stats['max_mineru_run_pages']} 页）",
                flush=True,
            )
        except Exception as exc:
            log.append({"file": pdf.name, "status": "failed", "error": str(exc)})
            print(f"  失败: {exc}", file=sys.stderr, flush=True)
    (out / "manifest.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = sum(item["status"] != "ok" for item in log)
    print(f"完成 {len(log) - failed}/{len(log)} 个文件；结果目录：{out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
