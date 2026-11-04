"""给已有 Markdown 补上图片描述：抽图 -> 视觉理解 -> 写回原占位符位置。

用法：
    python apply_vision.py                # 知识库 + 审核任务，全部补齐
    python apply_vision.py --only kb      # 只补 knowledge_base_output/markdown
    python apply_vision.py --only review  # 只补审核任务
    python apply_vision.py --file <md>    # 只补一个文件
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from build_local_index import (
    IMAGE_PLACEHOLDER,
    extract_embedded_images,
    image_notes,
    inline_image_notes,
)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "knowledge_base_output" / "mineru_raw"
KB_DIR = ROOT / "knowledge_base_output" / "markdown"
WORK_DIR = ROOT / "knowledge_base_output" / "review_workspace"


def has_placeholder(path: Path) -> bool:
    return IMAGE_PLACEHOLDER in path.read_text(encoding="utf-8")


def knowledge_docs() -> list[Path]:
    return [path for path in sorted(KB_DIR.glob("*.md")) if has_placeholder(path)]


def review_docs() -> list[Path]:
    return [path for path in sorted(WORK_DIR.glob("*/markdown/*.md")) if has_placeholder(path)]


def process(doc: Path) -> int:
    if doc.parent.name == "markdown":
        source = RAW_DIR / doc.name
        if not source.exists():
            print(f"跳过（找不到原始解析）：{doc.name}")
            return 0
        with tempfile.TemporaryDirectory() as tmp:
            extract_embedded_images(source, Path(tmp), prefix="img")
            return inline_image_notes(doc, image_notes(Path(tmp)))
    assets = doc.parent / "assets"
    if assets.exists() and any(assets.iterdir()):
        return inline_image_notes(doc, image_notes(assets))
    # 审核任务目录里没有落盘图片时，回退到该任务的 mineru_raw 里抽图。
    source = doc.parent.parent / "mineru_raw" / doc.name
    if source.exists():
        with tempfile.TemporaryDirectory() as tmp:
            extract_embedded_images(source, Path(tmp), prefix="img")
            return inline_image_notes(doc, image_notes(Path(tmp)))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="给已有 Markdown 补图片描述")
    parser.add_argument("--only", choices=("all", "kb", "review"), default="all")
    parser.add_argument("--file", type=Path, help="只处理指定文件")
    args = parser.parse_args()

    if args.file:
        docs = [args.file.resolve()]
    elif args.only == "kb":
        docs = knowledge_docs()
    elif args.only == "review":
        docs = review_docs()
    else:
        docs = knowledge_docs() + review_docs()

    total = 0
    for index, doc in enumerate(docs, 1):
        written = process(doc)
        total += written
        print(f"[{index}/{len(docs)}] {doc.relative_to(ROOT)}  写入 {written} 处", flush=True)
    print(f"完成：{len(docs)} 个文件，共写入 {total} 处图片描述")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
