"""Prepare downloadable Markdown files for human review before ingestion."""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "knowledge_base_output" / "markdown"
DEFAULT_OUTPUT = ROOT / "knowledge_base_output" / "review_package"


def main() -> int:
    parser = argparse.ArgumentParser(description="生成可人工核对的 MinerU/大模型 Markdown 审核包")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.exists():
        raise SystemExit(f"找不到 Markdown 目录：{source}")
    output.mkdir(parents=True, exist_ok=True)
    files = sorted(source.glob("*.md"))
    if not files:
        raise SystemExit("没有找到 Markdown 文件")
    copied = []
    for path in files:
        target = output / path.name
        shutil.copy2(path, target)
        copied.append({"file": path.name, "status": "待人工核对", "path": str(target)})
    # Keep a concise audit note next to the files so reviewers know what to check.
    readme = output / "审核说明.md"
    readme.write_text(
        "# 知识库 Markdown 审核包\n\n"
        "这些 Markdown 已经过 MinerU 解析；图片位置已由视觉大模型写成 `【图 N】` 段，内容就是图中文字/公式/表格。\n\n"
        "审核重点：\n\n"
        "- 检查标题、段落顺序、表格、公式和 OCR 错字。\n"
        "- 检查 `【图 N】` 段的内容是否与图片一致（识别失败的图会保留 `[图片已解析，原图未嵌入文本]` 标记，可手动补）。\n"
        "- 修正后只把审核通过的 `.md` 上传到知识库，不要上传 `审核说明.md`。\n"
        "- `manifest.json` 是审核清单，不是知识库文档。\n",
        encoding="utf-8",
    )
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps({"created_at": datetime.now().isoformat(timespec="seconds"), "files": copied}, ensure_ascii=False, indent=2), encoding="utf-8")
    archive = output.parent / "knowledge_base_review_package.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(output.iterdir()):
            if path.is_file():
                zf.write(path, arcname=f"review_package/{path.name}")
    print(f"已生成 {len(files)} 份待审核 Markdown：{output}")
    print(f"可下载压缩包：{archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
