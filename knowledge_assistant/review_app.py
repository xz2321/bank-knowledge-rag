"""Upload -> MinerU/vision processing -> review Markdown -> ingest workflow."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import uuid
from datetime import datetime
from urllib.parse import quote
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from html import escape

from build_knowledge_base import make_markdown, route_pdf_pages
from build_local_index import extract_embedded_images, image_notes, inline_image_notes
from config import load_env

ROOT = Path(__file__).resolve().parents[1]
load_env()
WORK = ROOT / "knowledge_base_output" / "review_workspace"
EXPORT_DIR = ROOT / "exports"
MARKDOWN_DIR = ROOT / "knowledge_base_output" / "markdown"
INDEX = ROOT / "knowledge_base_output" / "local_index.json"
app = FastAPI(title="知识库文档审核工作台")


def safe_name(name: str) -> str:
    return Path(name or "uploaded.pdf").name.replace(".pdf.pdf", ".pdf")


def rebuild_index() -> tuple[bool, str, int | None]:
    """重建本地索引，返回 (是否成功, 错误信息, 分块数)。"""
    python = ROOT / ".lc" / "Scripts" / "python.exe"
    script = ROOT / "knowledge_assistant" / "build_local_index.py"
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        [str(python), str(script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    chunks = None
    match = re.search(r"(\d+)\s*个分块", result.stdout or "")
    if match:
        chunks = int(match.group(1))
    return result.returncode == 0, (result.stderr or result.stdout or "")[-1500:], chunks


def sync_vectors() -> tuple[bool, str]:
    """知识库增删后同步向量库。

    本地 Qdrant 同一时间只能被一个进程打开，而 8090 问答服务持有它，
    所以这里让 8090 自己重建（POST /reindex），8091 不直接碰存储目录。
    """
    if os.getenv("USE_QDRANT", "0") != "1":
        return True, "未启用向量检索"
    import requests

    url = os.getenv("QA_BASE_URL", "http://127.0.0.1:8090").rstrip("/") + "/reindex"
    try:
        response = requests.post(url, timeout=900)
        response.raise_for_status()
        return True, f"向量库已重建（{response.json().get('points', 0)} 个点）"
    except Exception as exc:
        return False, f"向量库未同步：{exc}（重启 8090 会按新索引重建，或手动 POST /reindex）"


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return (Path(__file__).with_name("review.html")).read_text(encoding="utf-8")


@app.post("/process")
async def process_pdf(file: UploadFile = File(...)) -> dict[str, str | int | bool]:
    # 注意：FastAPI 会按返回注解校验响应体，``generated_at``/``size_bytes`` 这类
    # 数值字段不能再声明成 str，否则处理成功后仍会抛 ResponseValidationError（500）。
    filename = safe_name(file.filename or "uploaded.pdf")
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(400, "只支持 PDF 文件")
    job = WORK / uuid.uuid4().hex[:12]
    raw = job / "mineru_raw"
    source = job / filename
    output = job / "markdown"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(await file.read())
    try:
        parsed, route_stats = route_pdf_pages(source, raw, os.getenv("MINERU_TIER", "basic"))
        output.mkdir(parents=True, exist_ok=True)
        assets = output / "assets"
        # MinerU 4 may embed figures as data URIs in Markdown rather than
        # creating an images directory. Extract them before cleanup.
        extract_embedded_images(parsed, assets)
        target = output / f"{Path(filename).stem}.md"
        # 审核流程自己抽图并描述（见下），这里不再重复抽图。
        make_markdown(source, parsed, target, assets, describe_images=False)
        # 视觉模型描述直接写回原图片位置；识别失败的图保留占位符，位置不会错乱。
        inline_image_notes(target, image_notes(assets))
        return {
            "job_id": job.name,
            "filename": target.name,
            "download": f"/download/{job.name}/{quote(target.name)}?v={target.stat().st_mtime_ns}",
            "view": f"/view/{job.name}/{quote(target.name)}?v={target.stat().st_mtime_ns}",
            "generated_at": datetime.fromtimestamp(target.stat().st_mtime).isoformat(timespec="seconds"),
            "size_bytes": target.stat().st_size,
            "images_extracted": len(list(assets.glob("*"))),
            "vision_enabled": bool(os.getenv("OPENAI_API_KEY")),
            **route_stats,
        }
    except Exception as exc:
        shutil.rmtree(job, ignore_errors=True)
        raise HTTPException(500, f"处理失败：{exc}") from exc


@app.get("/download/{job_id}/{filename}")
def download(job_id: str, filename: str) -> Response:
    path = WORK / job_id / "markdown" / Path(filename).name
    if not path.exists():
        raise HTTPException(404, "审核文件不存在")
    # 同时给出 ASCII 文件名（filename）和中文文件名（filename*）：
    # 只给中文名时，个别浏览器的下载会被静默丢弃。
    chinese_name = f"{path.stem}_{job_id}.md"
    ascii_name = f"{job_id}.md"
    disposition = (
        f'attachment; filename="{ascii_name}"; '
        f"filename*=utf-8''{quote(chinese_name)}"
    )
    return Response(
        content=path.read_bytes(),
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": disposition,
            "Content-Length": str(path.stat().st_size),
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )


@app.post("/export/{job_id}/{filename}")
def export_file(job_id: str, filename: str) -> dict[str, str]:
    """把结果直接复制到本机 exports 目录，绕开浏览器的下载行为。"""
    path = WORK / job_id / "markdown" / Path(filename).name
    if not path.exists():
        raise HTTPException(404, "审核文件不存在")
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    target = EXPORT_DIR / f"{path.stem}_{job_id}.md"
    shutil.copy2(path, target)
    return {"path": str(target)}


@app.post("/kb/export/{filename}")
def export_kb_doc(filename: str) -> dict[str, str]:
    """把知识库里的 Markdown 复制到本机 exports 目录。"""
    path = MARKDOWN_DIR / Path(filename).name
    if not path.exists():
        raise HTTPException(404, "知识库文档不存在")
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    target = EXPORT_DIR / path.name
    shutil.copy2(path, target)
    return {"path": str(target)}


@app.post("/kb/delete/{filename}")
def delete_kb_doc(filename: str) -> dict[str, str | int]:
    """从知识库删除一个 Markdown 并重建索引。"""
    name = Path(filename).name
    if not name.lower().endswith(".md"):
        raise HTTPException(400, "只支持 .md 文件")
    target = (MARKDOWN_DIR / name).resolve()
    if target.parent != MARKDOWN_DIR.resolve() or not target.exists():
        raise HTTPException(404, "知识库文档不存在")
    target.unlink()
    ok, detail, chunks = rebuild_index()
    if not ok:
        raise HTTPException(500, f"文档已删除，但索引重建失败：{detail}")
    vectors_ok, vectors_detail = sync_vectors()
    return {
        "deleted": name,
        "chunks": chunks or 0,
        "vectors_ok": vectors_ok,
        "vectors": vectors_detail,
    }


SAVE_SCRIPT = """
<script>
async function exportTo(url, box) {
  if (box) box.textContent = '正在保存…';
  try {
    const response = await fetch(url, {method: 'POST'});
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || ('HTTP ' + response.status));
    if (box) box.textContent = '已保存到：' + data.path;
  } catch (error) {
    if (box) box.textContent = '保存失败：' + error.message;
  }
}
function armDelete(button, box, message) {
  document.querySelectorAll('button.btn.danger[data-armed]').forEach(function (other) {
    if (other !== button) {
      delete other.dataset.armed;
      other.textContent = '删除';
    }
  });
  button.dataset.armed = '1';
  button.textContent = '确认删除';
  if (box) box.textContent = message;
  setTimeout(function () {
    if (button.isConnected) {
      delete button.dataset.armed;
      button.textContent = '删除';
    }
  }, 6000);
}
async function requestDelete(url, button, box, label) {
  if (box) box.textContent = '正在删除…';
  try {
    const response = await fetch(url, {method: 'POST'});
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || ('HTTP ' + response.status));
    const row = button.closest('tr');
    if (row) row.remove();
    let text = '已删除 ' + label;
    if (data.exports_removed) text += '（含 ' + data.exports_removed + ' 个导出副本）';
    if (data.chunks) text += '，索引已重建：' + data.chunks + ' 个分块';
    if (data.vectors) text += '；' + data.vectors;
    if (box) box.textContent = text;
  } catch (error) {
    if (box) box.textContent = '删除失败：' + error.message;
  }
}
function removeJob(jobId, button, box) {
  if (!button.dataset.armed) {
    armDelete(button, box, '再点一次「确认删除」即可删除任务 ' + jobId + '（Markdown、图片、导出副本一起删除）');
    return;
  }
  requestDelete('/delete/' + encodeURIComponent(jobId), button, box, '任务 ' + jobId);
}
function removeKbDoc(name, button, box) {
  if (!button.dataset.armed) {
    armDelete(button, box, '再点一次「确认删除」即可从知识库移除该文档（会重建索引）');
    return;
  }
  // name 已经是 URL 编码过的路径段，不能再编码一次。
  requestDelete('/kb/delete/' + name, button, box, '知识库文档');
}
</script>
"""


@app.get("/view/{job_id}/{filename}", response_class=HTMLResponse)
def view(job_id: str, filename: str) -> HTMLResponse:
    path = WORK / job_id / "markdown" / Path(filename).name
    if not path.exists():
        raise HTTPException(404, "审核文件不存在")
    content = escape(path.read_text(encoding="utf-8", errors="replace"))
    generated = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    download_url = f"/download/{quote(job_id)}/{quote(path.name)}"
    export_url = f"/export/{quote(job_id)}/{quote(path.name)}"
    html = f'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(path.name)}</title>
<style>body{{margin:0;background:#f6f7f9;color:#202124;font-family:system-ui,"Microsoft YaHei",sans-serif}}header{{position:sticky;top:0;background:#fff;border-bottom:1px solid #e1e4e8;padding:14px 5vw;line-height:1.6}}main{{max-width:1100px;margin:20px auto;padding:0 20px}}pre{{background:#fff;border:1px solid #e1e4e8;border-radius:8px;padding:24px;white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.65;box-shadow:0 1px 2px #0000000d}}.btn{{display:inline-block;margin-top:10px;padding:8px 16px;background:#1769e0;color:#fff;border:0;border-radius:5px;text-decoration:none;font-size:14px;cursor:pointer}}.btn.ghost{{background:#fff;color:#1769e0;border:1px solid #1769e0;margin-left:8px}}.status{{margin-top:8px;font-size:13px;color:#0b6b2f}}</style>
<header><strong>{escape(path.name)}</strong><br><small>任务号：{escape(job_id)}　生成时间：{generated}　文件大小：{path.stat().st_size:,} 字节</small><br>
<button class="btn" onclick="exportTo('{export_url}', document.getElementById('st'))">下载 Markdown</button>
<a class="btn ghost" href="{download_url}" download>浏览器直链</a>
<a class="btn ghost" href="/jobs">全部任务</a>
<div id="st" class="status"></div></header>
<main><pre>{content}</pre></main>%SCRIPT%</html>'''
    return HTMLResponse(html.replace("%SCRIPT%", SAVE_SCRIPT), headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0", "Pragma": "no-cache"})


@app.get("/jobs", response_class=HTMLResponse)
def jobs() -> HTMLResponse:
    rows: list[str] = []
    entries = sorted(
        (item for item in WORK.glob("*") if (item / "markdown").is_dir()),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for job in entries:
        for markdown in sorted((job / "markdown").glob("*.md")):
            stamp = markdown.stat().st_mtime
            when = datetime.fromtimestamp(stamp).strftime("%Y-%m-%d %H:%M")
            download_url = f"/download/{quote(job.name)}/{quote(markdown.name)}"
            export_url = f"/export/{quote(job.name)}/{quote(markdown.name)}"
            view_url = f"/view/{quote(job.name)}/{quote(markdown.name)}"
            rows.append(
                "<tr>"
                f"<td>{escape(markdown.stem)}</td>"
                f"<td>{when}</td>"
                f"<td>{markdown.stat().st_size:,}</td>"
                f"<td>{escape(job.name)}</td>"
                f'<td><button class="btn" onclick="exportTo(\'{export_url}\', document.getElementById(\'st\'))">下载</button>'
                f'<a class="btn ghost" href="{download_url}" download>直链</a>'
                f'<a class="btn ghost" href="{view_url}">查看</a>'
                f'<button class="btn danger" onclick="removeJob(\'{escape(job.name)}\', this, document.getElementById(\'st\'))">删除</button></td>'
                "</tr>"
            )
    body = "".join(rows) or '<tr><td colspan="5">还没有处理记录，回到工作台上传 PDF 即可。</td></tr>'
    kb_rows: list[str] = []
    for markdown in sorted(MARKDOWN_DIR.glob("*.md")):
        kb_rows.append(
            "<tr>"
            f"<td>{escape(markdown.stem)}</td>"
            f"<td>{datetime.fromtimestamp(markdown.stat().st_mtime):%Y-%m-%d %H:%M}</td>"
            f"<td>{markdown.stat().st_size:,}</td>"
            f'<td><button class="btn" onclick="exportTo(\'/kb/export/{quote(markdown.name)}\', document.getElementById(\'st\'))">下载</button>'
            f'<button class="btn danger" onclick="removeKbDoc(\'{quote(markdown.name)}\', this, document.getElementById(\'st\'))">删除</button></td>'
            "</tr>"
        )
    kb_body = "".join(kb_rows) or '<tr><td colspan="4">知识库还是空的</td></tr>'
    html = f'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>历史任务 - 知识库文档审核工作台</title>
<style>body{{margin:0;background:#f6f7f9;color:#202124;font-family:system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1000px;margin:32px auto;padding:0 20px}}h1{{font-size:22px}}h2{{font-size:17px;margin-top:34px}}table{{width:100%;border-collapse:collapse;background:#fff;border:1px solid #e1e4e8;border-radius:8px;overflow:hidden}}th,td{{padding:10px 12px;border-bottom:1px solid #eef0f2;text-align:left;font-size:14px}}th{{background:#fafbfc;font-weight:600}}a.btn,button.btn{{display:inline-block;padding:5px 12px;background:#1769e0;color:#fff;border:0;border-radius:4px;text-decoration:none;font-size:13px;cursor:pointer}}a.btn.ghost,button.btn.ghost{{background:#fff;color:#1769e0;border:1px solid #1769e0;margin-left:6px}}button.btn.danger{{background:#d1322f;margin-left:6px}}.status{{margin:12px 0;font-size:13px;color:#0b6b2f}}</style>
<main><h1>历史处理任务</h1><p><a class="btn ghost" href="/">返回工作台</a></p>
<div id="st" class="status"></div>
<table><thead><tr><th>文件</th><th>生成时间</th><th>大小（字节）</th><th>任务号</th><th>操作</th></tr></thead><tbody>{body}</tbody></table>
<h2>知识库文档（已入库，删除后自动重建索引）</h2>
<table><thead><tr><th>文件</th><th>修改时间</th><th>大小（字节）</th><th>操作</th></tr></thead><tbody>{kb_body}</tbody></table></main>%SCRIPT%</html>'''
    return HTMLResponse(html.replace("%SCRIPT%", SAVE_SCRIPT), headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0", "Pragma": "no-cache"})


@app.post("/ingest")
async def ingest_markdown(file: UploadFile = File(...)) -> dict[str, str | int]:
    filename = Path(file.filename or "reviewed.md").name
    if not filename.lower().endswith(".md"):
        raise HTTPException(400, "这里只接受人工审核后的 .md 文件")
    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
    target = MARKDOWN_DIR / filename
    target.write_bytes(await file.read())
    ok, detail, chunks = rebuild_index()
    if not ok:
        raise HTTPException(500, f"索引更新失败：{detail}")
    vectors_ok, vectors_detail = sync_vectors()
    message = f"审核 Markdown 已入库，索引已重建（{chunks or 0} 个分块）；{vectors_detail}"
    return {
        "message": message,
        "file": filename,
        "index": str(INDEX),
        "vectors_ok": vectors_ok,
    }


@app.post("/ingest-job/{job_id}/{filename}")
def ingest_processed_markdown(job_id: str, filename: str) -> dict[str, str | int | bool]:
    """直接把本次解析生成的 Markdown 入库，无需先下载再上传。"""
    if not re.fullmatch(r"[0-9a-f]{8,32}", job_id):
        raise HTTPException(400, "任务号不合法")
    source = (WORK / job_id / "markdown" / Path(filename).name).resolve()
    job_root = (WORK / job_id).resolve()
    if source.parent != (job_root / "markdown").resolve() or not source.is_file():
        raise HTTPException(404, "解析后的 Markdown 不存在")
    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
    target = MARKDOWN_DIR / source.name
    shutil.copy2(source, target)
    ok, detail, chunks = rebuild_index()
    if not ok:
        raise HTTPException(500, f"文档已复制到知识库目录，但索引更新失败：{detail}")
    vectors_ok, vectors_detail = sync_vectors()
    return {
        "message": f"解析结果已直接入库，索引已重建（{chunks or 0} 个分块）；{vectors_detail}",
        "file": target.name,
        "vectors_ok": vectors_ok,
    }


@app.post("/delete/{job_id}")
def delete_job(job_id: str) -> dict[str, str | int]:
    """删除一个审核任务：任务目录（Markdown + 图片）和对应的导出副本。"""
    if not re.fullmatch(r"[0-9a-f]{8,32}", job_id):
        raise HTTPException(400, "任务号不合法")
    job = (WORK / job_id).resolve()
    if job.parent != WORK.resolve() or not job.is_dir():
        raise HTTPException(404, "任务不存在")
    removed_exports = 0
    if EXPORT_DIR.exists():
        for path in EXPORT_DIR.glob(f"*_{job_id}.md"):
            path.unlink(missing_ok=True)
            removed_exports += 1
    shutil.rmtree(job)
    return {"deleted": job_id, "exports_removed": removed_exports}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("review_app:app", host="127.0.0.1", port=int(os.getenv("REVIEW_PORT", "8091")), reload=False)
