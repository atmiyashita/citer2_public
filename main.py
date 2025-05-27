import asyncio
import functools
import json
import logging
import os
import tempfile
import uuid
from typing import Optional, Dict   

import janus
from fastapi import (
    FastAPI, UploadFile, File, Form, Query, Request, Depends, HTTPException
)
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request as StarRequest
from sse_starlette.sse import EventSourceResponse
from fastapi.security import HTTPBearer
from fastapi_users import FastAPIUsers

from agent.pipeline import run_pipeline
from agent.routes.billing import router as billing_router
from agent.auth import google_router, auth_router, current_active_user, jwt_backend
from agent.models import User
from datetime import datetime, timedelta


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)



strategy = jwt_backend.get_strategy()
app = FastAPI(title="Literature Search Agent")
JOB_TTL      = int(os.getenv("JOB_TTL_SEC", "600"))
jobs: Dict[str, dict] = {}    

SESSION_SECRET = os.getenv("SESSION_SECRET")
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET)

app.include_router(billing_router)
app.include_router(google_router)
app.include_router(auth_router)

# ===== UTILITY (SSE formatting) ===================================
def sse(event: str, data):
    """EventSourceResponse が受け取れる dict 形式を返す"""
    if not isinstance(data, str):
        data = json.dumps(data, ensure_ascii=False)
    if os.getenv("DEBUG_PROGRESS") == "1":
        logging.debug("[SSE-OUT] %s %s", event, data)
    return {"event": event, "data": data}

def role_required(min_role: int):
    def dep(user: User = Depends(current_active_user)):
        if user.role < min_role:
            raise HTTPException(403, "upgrade required")
        return user
    return Depends(dep)

# ===== REDIRECT ===========================================
@app.get("/")
async def index(request: StarRequest):
    qs = request.url.query
    url = f"/index.html?{qs}" if qs else "/index.html"
    return RedirectResponse(url)

# ===== Pipeline実行＆ファイル生成ユーティリティ ===========================
async def _run_pipeline_and_savefiles(text, hits, threshold, max_refs, user, progress=None):
    tmpdir = tempfile.gettempdir()
    cited, ris_str, rep_base, det_base, references = await asyncio.get_event_loop().run_in_executor(
        None,
        functools.partial(
            run_pipeline, text, hits, threshold, max_refs,
            False, None, progress,
        )
    )
    ris_path = os.path.join(tmpdir, f"{uuid.uuid4()}.ris")
    with open(ris_path, "w", encoding="utf-8") as f:
        f.write(ris_str)

    return cited, ris_str, rep_base, det_base, ris_path, det_base, references

# ====== Job管理付き パイプライン実行 ===========================
async def run_pipeline_with_progress(
    job_id: str, text: str, user: User, hits: int, threshold: int, max_refs: int,) -> None: 
    """
    擬似的にジョブの進捗を管理しつつパイプラインを実行
    """
    # 進捗キューを用意
    q: asyncio.Queue = asyncio.Queue()
    jobs[job_id] = {"queue": q, "done": False, "result": None, "stamp":  datetime.utcnow(),}

    def progress_callback(progress_obj: dict):
        # Syncでないのでqueueに入れるには工夫必要（ここはあくまで例）
        loop = asyncio.get_running_loop()
        loop.call_soon_threadsafe(q.put_nowait, progress_obj)

    # バックグラウンドでパイプ実行
    loop = asyncio.get_running_loop()
    fut = loop.run_in_executor(
        None,
        functools.partial(
            run_pipeline, text, hits, threshold, max_refs, False, None, progress_callback
        ),
    )

    try:
        # fut も 5要素返すよう変更した run_pipeline
        cited, ris_str, rep_base, det_base, references = await asyncio.wrap_future(fut)
    except Exception as e:
        jobs[job_id].update(done=True, result={"error": str(e)})
        return

    # 結果保存
    _, _, _, _, ris_path, det_base, _ = await _run_pipeline_and_savefiles(
        text, hits, threshold, max_refs, user, progress=None
    )

    jobs[job_id]["done"] = True
    jobs[job_id]["result"] = {
        "cited": cited,
        "ris_url": f"/download/{os.path.basename(ris_path)}",
        "detail_json": f"/download/{det_base}_detail.json",
        "references": references,
    }

# ===== SSE用 Progress Stream ====================================
@app.get("/api/run_stream_sse")
async def run_agent_stream_sse(
    text: str = Query(...),
    hits: int = Query(10),
    threshold: int = Query(4),
    max_refs: int = Query(3),
    user: User = Depends(current_active_user),
):
    q: janus.Queue[dict] = janus.Queue()
    def progress(obj: dict):
        q.sync_q.put(obj)
        if os.getenv("DEBUG_PROGRESS") == "1":
            logging.debug("[PROGRESS] %s", obj)

    async def event_generator():
        yield sse("init", {})
        loop = asyncio.get_running_loop()
        fut = loop.create_task(_run_pipeline_and_savefiles(
            text, hits, threshold, max_refs, user, progress
        ))

        while True:
            try:
                obj = await asyncio.wait_for(q.async_q.get(), 0.4)
                yield sse("progress", obj)
            except asyncio.TimeoutError:
                pass
            if fut.done() and q.async_q.empty():
                break

        try:
            cited, ris_str, rep_base, det_base, ris_path, det_base, references = await fut
        except Exception:
            logging.exception("pipeline error")
            yield sse("error", {"error": "internal"})
            return

        yield sse(
            "done",
            {
                "cited": cited,
                "ris_url": f"/download/{os.path.basename(ris_path)}",
                "detail_json": f"/download/{det_base}_detail.json",
                "references": references,
            },
        )

    return EventSourceResponse(event_generator())

# -------- SSE 進捗購読 ----------------
@app.get("/api/run_stream_events")
async def run_stream_events(
    job_id: str = Query(...),
    user: User = Depends(current_active_user),
):
    if job_id not in jobs:
        raise HTTPException(404, "job not found")

    q: asyncio.Queue = jobs[job_id]["queue"]

    async def event_generator():
        yield sse("init", {})
        # 進捗通知
        while not jobs[job_id]["done"]:
            try:
                obj = await asyncio.wait_for(q.get(), timeout=0.5)
                yield sse("progress", obj)
            except asyncio.TimeoutError:
                continue
        # 終了
        result = jobs[job_id]["result"] or {}
        if "error" in result:
            yield sse("error", {"error": result["error"]})
        else:
            yield sse("done", result)

    return EventSourceResponse(event_generator())



# ===== 最終結果only(Fetch/JSON向け) ======================
@app.get("/api/run_stream_get")
async def run_agent_stream_get(
    text: str = Query(...),
    hits: int = Query(10),
    threshold: int = Query(4),
    max_refs: int = Query(3),
    user: User = Depends(current_active_user),
):
    cited, ris_str, rep_base, det_base, ris_path, det_base, references = await _run_pipeline_and_savefiles(
        text, hits, threshold, max_refs, user, progress=None
    )
    result = {
        "cited": cited,
        "ris_url": f"/download/{os.path.basename(ris_path)}",
        "detail_json": f"/download/{det_base}_detail.json",
        "references": references,
    }
    return JSONResponse(result)

# ===== POST/UPLOADED FILE用エンドポイント(省略 or fetch/putにも分岐応用可) =====
@app.post("/api/run_stream")
async def run_agent_stream_post(
    text: str = Form(None),
    file: UploadFile = File(None),
    hits: int = Form(10),
    threshold: int = Form(4),
    max_refs: int = Form(3),
    user: User = role_required(1),
):
    if not text and not file:
        raise HTTPException(400, "text or file is required")
    if file:
        text = (await file.read()).decode("utf-8", errors="ignore")

    return await run_agent_stream_get(
        text=text,
        hits=hits,
        threshold=threshold,
        max_refs=max_refs,
        user=user
    )

# -------- 新規ジョブ開始（POST/FETCH 用） ----------------
@app.post("/api/run_stream_start")
async def run_stream_start(
    text: str = Form(...),
    hits: int = Form(10),
    threshold: int = Form(4),
    max_refs: int = Form(3),
    user: User = Depends(current_active_user),
):
    job_id = str(uuid.uuid4())
    # バックグラウンドでパイプライン開始
    asyncio.create_task(
        run_pipeline_with_progress(job_id, text, user, hits, threshold, max_refs)
    )
    return {"job_id": job_id}


#  ========= 大量ジョブでメモリが膨れるのを防ぐ簡易クリーナー

@app.on_event("startup")
async def _job_gc_task():
    async def gc_loop():
        while True:
            now = datetime.utcnow()
            for jid in list(jobs.keys()):
                if (now - jobs[jid]["stamp"]) > timedelta(seconds=JOB_TTL):
                    jobs.pop(jid, None)
            await asyncio.sleep(JOB_TTL // 2)

    asyncio.create_task(gc_loop())


# ===== FILE DOWNLOAD & HEALTH CHECK =====================================
@app.get("/download/{fname}")
def download(fname: str):
    path = os.path.join(tempfile.gettempdir(), fname)
    if not os.path.exists(path):
        return JSONResponse({"error": "not found"}, 404)
    mime = (
        "application/json" if fname.endswith(".json")
        else "text/csv" if fname.endswith(".csv")
        else "application/x-research-info-systems"
    )
    return FileResponse(path, media_type=mime, filename=fname)

@app.get("/health")
def health():
    return {"status": "ok"}

# ===== STATIC FILES =====================================
app.mount("/", StaticFiles(directory="static", html=True), name="static")