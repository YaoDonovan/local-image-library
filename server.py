"""Offline-first local image library MVP. Run with: python server.py"""
from __future__ import annotations

import hashlib
import heapq
import importlib.util
import io
import json
import mimetypes
import os
import queue
import re
import secrets
import shutil
import sqlite3
import threading
import time
import uuid
import webbrowser
from contextlib import contextmanager
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

try:
    from PIL import Image, ImageOps, UnidentifiedImageError
except ImportError:  # Browsing and metadata still work without Pillow.
    Image = ImageOps = None
    UnidentifiedImageError = Exception

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIF_AVAILABLE = True
except ImportError:
    HEIF_AVAILABLE = False

try:
    import rawpy
    RAW_AVAILABLE = True
except ImportError:
    rawpy = None
    RAW_AVAILABLE = False

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("LOCAL_IMAGE_LIBRARY_DATA", Path.home() / "AppData/Local/LocalImageLibrary"))
DB_PATH = DATA_DIR / "catalog.db"
THUMB_DIR = DATA_DIR / "thumbnails"
TRASH_DIR = DATA_DIR / "trash"
HOST, PORT = "127.0.0.1", int(os.environ.get("LOCAL_IMAGE_LIBRARY_PORT", "8765"))
BASE_IMAGE_EXTS = {".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".webp", ".bmp", ".dib", ".gif", ".tif", ".tiff", ".avif", ".heic", ".heif", ".ico", ".icns", ".tga", ".pcx", ".ppm", ".pgm", ".pbm", ".pnm", ".xbm", ".dds", ".sgi"}
RAW_EXTS = {".3fr", ".ari", ".arw", ".bay", ".cap", ".cr2", ".cr3", ".dcs", ".dcr", ".dng", ".drf", ".eip", ".erf", ".fff", ".gpr", ".iiq", ".k25", ".kdc", ".mef", ".mos", ".mrw", ".nef", ".nrw", ".orf", ".pef", ".ptx", ".pxn", ".raf", ".raw", ".rw2", ".rwl", ".rwz", ".sr2", ".srf", ".srw", ".x3f"}
PIL_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "GIF", "TIFF", "AVIF", "ICO", "ICNS", "TGA", "PCX", "PPM", "PGM", "PBM", "XBM", "DDS", "SGI", "MPO"}
PIL_IMAGE_EXTS = {ext.lower() for ext, fmt in (Image.registered_extensions().items() if Image else []) if fmt in PIL_IMAGE_FORMATS}
IMAGE_EXTS = BASE_IMAGE_EXTS | PIL_IMAGE_EXTS
if not HEIF_AVAILABLE: IMAGE_EXTS -= {".heic", ".heif"}
if RAW_AVAILABLE: IMAGE_EXTS |= RAW_EXTS
TOKEN = secrets.token_urlsafe(32)
MUTEX = threading.RLock()
CONTROL: dict[str, threading.Event] = {}
SCAN_QUEUE: queue.Queue[str] = queue.Queue()
SEMANTIC_ENGINE = None
SEMANTIC_MODEL_VERSION = "siglip2-base-patch16-224@75de2d55"


class ClosingConnection(sqlite3.Connection):
    def __exit__(self, exc_type, exc, tb):
        try:
            return super().__exit__(exc_type, exc, tb)
        finally:
            self.close()


def connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=30, factory=ClosingConnection)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    return db


def init_db() -> None:
    with connect() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS roots (
          id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, online INTEGER NOT NULL DEFAULT 1,
          added_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS assets (
          id INTEGER PRIMARY KEY, root_id INTEGER NOT NULL REFERENCES roots(id) ON DELETE CASCADE,
          path TEXT UNIQUE NOT NULL, file_size INTEGER NOT NULL, mtime REAL NOT NULL,
          sha256 TEXT, dhash TEXT, captured_at TEXT, width INTEGER, height INTEGER,
          rating INTEGER NOT NULL DEFAULT 0, missing INTEGER NOT NULL DEFAULT 0, scan_token TEXT,
          indexed_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS assets_capture ON assets(captured_at);
        CREATE INDEX IF NOT EXISTS assets_root ON assets(root_id, missing);
        CREATE TABLE IF NOT EXISTS tags (id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL);
        CREATE TABLE IF NOT EXISTS asset_tags (
          asset_id INTEGER NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
          tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
          PRIMARY KEY(asset_id, tag_id)
        );
        CREATE TABLE IF NOT EXISTS ocr_text (
          asset_id INTEGER PRIMARY KEY REFERENCES assets(id) ON DELETE CASCADE,
          text TEXT NOT NULL DEFAULT '', language TEXT, model_version TEXT
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS ocr_fts USING fts5(text, content='ocr_text', content_rowid='asset_id');
        CREATE TRIGGER IF NOT EXISTS ocr_ai AFTER INSERT ON ocr_text BEGIN
          INSERT INTO ocr_fts(rowid,text) VALUES(new.asset_id,new.text);
        END;
        CREATE TRIGGER IF NOT EXISTS ocr_au AFTER UPDATE ON ocr_text BEGIN
          INSERT INTO ocr_fts(ocr_fts,rowid,text) VALUES('delete',old.asset_id,old.text);
          INSERT INTO ocr_fts(rowid,text) VALUES(new.asset_id,new.text);
        END;
        CREATE TRIGGER IF NOT EXISTS ocr_ad AFTER DELETE ON ocr_text BEGIN
          INSERT INTO ocr_fts(ocr_fts,rowid,text) VALUES('delete',old.asset_id,old.text);
        END;
        CREATE TABLE IF NOT EXISTS jobs (
          id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL,
          root_id INTEGER, progress INTEGER NOT NULL DEFAULT 0, total INTEGER,
          current_path TEXT, message TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS operations (
          id TEXT PRIMARY KEY, kind TEXT NOT NULL, source TEXT NOT NULL, target TEXT,
          status TEXT NOT NULL, undo_json TEXT, details_json TEXT,
          created_at TEXT NOT NULL, confirmed_at TEXT
        );
        CREATE TABLE IF NOT EXISTS edits (
          asset_id INTEGER PRIMARY KEY REFERENCES assets(id) ON DELETE CASCADE,
          recipe_json TEXT NOT NULL, version INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS semantic_embeddings (
          asset_id INTEGER PRIMARY KEY REFERENCES assets(id) ON DELETE CASCADE,
          model_version TEXT NOT NULL, vector BLOB NOT NULL, dimensions INTEGER NOT NULL,
          indexed_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS semantic_embeddings_model ON semantic_embeddings(model_version, asset_id);
        """)
        columns = {row[1] for row in db.execute("PRAGMA table_info(assets)")}
        if "scan_token" not in columns:
            db.execute("ALTER TABLE assets ADD COLUMN scan_token TEXT")
        db.execute("UPDATE jobs SET status='paused', message='应用关闭时已暂停，可继续扫描', updated_at=? WHERE status='running'", (now(),))


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


def authorized(path: Path, db: sqlite3.Connection) -> bool:
    return any(inside(path, Path(row[0])) for row in db.execute("SELECT path FROM roots"))


def is_authorized(path: Path) -> bool:
    with connect() as db:
        return authorized(path, db)


def calc_dhash(path: Path) -> tuple[str | None, int | None, int | None, str | None]:
    if Image is None:
        return None, None, None, None
    try:
        with open_image(path) as im:
            im = ImageOps.exif_transpose(im)
            width, height = im.size
            thumb = im.convert("L").resize((9, 8))
            pixels = thumb.load()
            px = [pixels[x, y] for y in range(8) for x in range(9)]
            bits = sum((1 << i) for i in range(64) if px[(i // 8) * 9 + (i % 8)] > px[(i // 8) * 9 + (i % 8) + 1])
            captured = None
            exif = im.getexif()
            raw = exif.get(36867) or exif.get(306)
            if raw:
                try:
                    captured = datetime.strptime(str(raw), "%Y:%m:%d %H:%M:%S").isoformat(sep=" ")
                except ValueError:
                    pass
            return f"{bits:016x}", width, height, captured
    except Exception:
        return None, None, None, None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def open_image(path: Path):
    if Image is None:
        raise RuntimeError("图像解码需要安装 Pillow")
    if path.suffix.lower() in RAW_EXTS and RAW_AVAILABLE:
        with rawpy.imread(str(path)) as raw:
            image = Image.fromarray(raw.postprocess())
            try: yield image
            finally: image.close()
    else:
        with Image.open(path) as image:
            yield image


def scan_worker() -> None:
    while True:
        job_id = SCAN_QUEUE.get()
        try:
            with connect() as db:
                job = db.execute("SELECT kind FROM jobs WHERE id=?", (job_id,)).fetchone()
            if job and job["kind"] == "semantic_index":
                run_semantic_index(job_id)
            else:
                run_scan(job_id)
        except Exception as exc:
            update_job(job_id, status="failed", message=f"扫描失败：{type(exc).__name__}")
        finally:
            SCAN_QUEUE.task_done()


def semantic_model_dir() -> Path:
    from semantic import model_directory
    return model_directory(APP_DIR, DATA_DIR)


def semantic_engine():
    global SEMANTIC_ENGINE
    if SEMANTIC_ENGINE is None:
        from semantic import SiglipEngine
        SEMANTIC_ENGINE = SiglipEngine(semantic_model_dir())
    return SEMANTIC_ENGINE


def semantic_status() -> dict:
    path = semantic_model_dir()
    required = ("config.json", "model.safetensors", "preprocessor_config.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json")
    model_present = path.is_dir() and all((path / name).is_file() for name in required)
    runtime_available = all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "numpy"))
    with connect() as db:
        indexed = db.execute("SELECT count(*) FROM semantic_embeddings e JOIN assets a ON a.id=e.asset_id WHERE a.missing=0 AND e.model_version=?", (SEMANTIC_MODEL_VERSION,)).fetchone()[0]
        pending = db.execute("SELECT count(*) FROM assets a LEFT JOIN semantic_embeddings e ON e.asset_id=a.id AND e.model_version=? WHERE a.missing=0 AND e.asset_id IS NULL", (SEMANTIC_MODEL_VERSION,)).fetchone()[0]
    return {"model": "SigLIP2 Base", "model_id": "google/siglip2-base-patch16-224", "model_present": model_present,
            "runtime_available": runtime_available, "ready": model_present and runtime_available,
            "loaded": bool(SEMANTIC_ENGINE and SEMANTIC_ENGINE._model is not None),
            "device": SEMANTIC_ENGINE.device if SEMANTIC_ENGINE and SEMANTIC_ENGINE._model is not None else "按需选择 GPU/CPU",
            "indexed": indexed, "pending": pending}


def run_semantic_index(job_id: str) -> None:
    engine = semantic_engine()
    if not engine.available:
        update_job(job_id, status="failed", message="未找到完整的本地 SigLIP2 模型")
        return
    try:
        with connect() as db:
            total = db.execute("SELECT count(*) FROM assets a LEFT JOIN semantic_embeddings e ON e.asset_id=a.id AND e.model_version=? WHERE a.missing=0 AND e.asset_id IS NULL", (SEMANTIC_MODEL_VERSION,)).fetchone()[0]
        update_job(job_id, status="running", progress=0, total=total, message=f"正在加载 SigLIP2 模型 · 待处理 {total} 张图片")
        engine.load()
        device = engine.device.upper()
        batch_size = max(1, min(8, int(os.environ.get("LOCAL_IMAGE_LIBRARY_SEMANTIC_BATCH", "4"))))
        started = time.monotonic()
        update_job(job_id, message=f"SigLIP2 已在 {device} 就绪 · 批量 {batch_size} · 待处理 {total} 张")
        processed = 0
        failed = 0
        last_id = 0
        stop = CONTROL.setdefault(job_id, threading.Event())
        while True:
            with connect() as db:
                rows = db.execute("""SELECT a.id,a.path FROM assets a LEFT JOIN semantic_embeddings e
                    ON e.asset_id=a.id AND e.model_version=? WHERE a.missing=0 AND e.asset_id IS NULL AND a.id>?
                    ORDER BY a.id LIMIT ?""", (SEMANTIC_MODEL_VERSION, last_id, batch_size)).fetchall()
            if not rows:
                break
            if stop.is_set():
                with connect() as db: current = db.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
                if current and current["status"] == "cancelled": return
                update_job(job_id, status="paused", progress=processed, total=total, message=f"语义索引已暂停 · 已处理 {processed}/{total} 张")
                return
            last_id = rows[-1]["id"]
            images, valid_rows = [], []
            for row in rows:
                path = Path(row["path"])
                try:
                    if not path.is_file() or not is_authorized(path):
                        raise OSError("文件当前不可访问")
                    with open_image(path) as source:
                        image = ImageOps.exif_transpose(source).convert("RGB")
                    image.thumbnail((2048, 2048))
                    images.append(image)
                    valid_rows.append(row)
                except Exception:
                    failed += 1
                    processed += 1
                    update_job(job_id, progress=processed, total=total, current_path=str(path), message=f"跳过无法解码的图片 · {processed}/{total}")
            vectors = []
            if images:
                try:
                    vectors = engine.encode_images(images)
                except Exception:
                    # A single problematic image or transient GPU memory pressure should not lose the whole batch.
                    for image in images:
                        try:
                            vectors.extend(engine.encode_images([image]))
                        except Exception:
                            vectors.append(None)
                finally:
                    for image in images: image.close()
                with connect() as db:
                    for row, vector in zip(valid_rows, vectors):
                        if vector is None:
                            failed += 1
                        else:
                            db.execute("""INSERT INTO semantic_embeddings(asset_id,model_version,vector,dimensions,indexed_at)
                                VALUES(?,?,?,?,?) ON CONFLICT(asset_id) DO UPDATE SET model_version=excluded.model_version,
                                vector=excluded.vector,dimensions=excluded.dimensions,indexed_at=excluded.indexed_at""",
                                (row["id"], SEMANTIC_MODEL_VERSION, vector.tobytes(), len(vector), now()))
                        processed += 1
            elapsed = max(time.monotonic() - started, 0.001)
            update_job(job_id, progress=processed, total=total, current_path=rows[-1]["path"],
                       message=f"SigLIP2 · {device} · {processed}/{total} 张 · {processed / elapsed:.1f} 张/秒")
        elapsed = max(time.monotonic() - started, 0.001)
        update_job(job_id, status="completed", progress=processed, total=total,
                   message=f"语义索引完成：新增 {processed - failed} 张，跳过 {failed} 张 · {device} · {processed / elapsed:.1f} 张/秒")
    except Exception as exc:
        update_job(job_id, status="failed", message=f"SigLIP2 索引失败：{type(exc).__name__}: {str(exc)[:160]}")
    finally:
        CONTROL.pop(job_id, None)


def update_job(job_id: str, **fields: object) -> None:
    fields["updated_at"] = now()
    sql = ",".join(f"{k}=?" for k in fields)
    with connect() as db:
        db.execute(f"UPDATE jobs SET {sql} WHERE id=?", (*fields.values(), job_id))


def run_scan(job_id: str) -> None:
    with connect() as db:
        job = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not job:
            return
        if job["status"] == "cancelled": return
        root = db.execute("SELECT * FROM roots WHERE id=?", (job["root_id"],)).fetchone()
    if not root:
        update_job(job_id, status="failed", message="图库目录已移除")
        return
    base = Path(root["path"])
    stop = CONTROL.setdefault(job_id, threading.Event())
    if stop.is_set():
        with connect() as db: current = db.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
        if current and current["status"] == "cancelled": return
        update_job(job_id, status="paused", message="扫描已暂停")
        return
    if not base.is_dir():
        update_job(job_id, status="failed", message="目录当前不可用")
        return
    update_job(job_id, status="running", progress=0, total=None, message="正在遍历目录并解析图片（已处理 0 张）")
    total = 0
    for current, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in {".git", "$RECYCLE.BIN", "System Volume Information", "@eaDir"}]
        if stop.is_set():
            with connect() as db: current_job = db.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            if current_job and current_job["status"] == "cancelled": return
            update_job(job_id, status="paused", progress=total, current_path=current, message="扫描已暂停；继续时会增量复核")
            return
        for name in files:
            path = Path(current) / name
            if path.suffix.lower() not in IMAGE_EXTS or path.is_symlink() or not inside(path, base): continue
            if stop.is_set():
                with connect() as db: current_job = db.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
                if current_job and current_job["status"] == "cancelled": return
                update_job(job_id, status="paused", progress=total, current_path=str(path), message="扫描已暂停；继续时会增量复核")
                return
            try:
                stat = path.stat()
                with connect() as db:
                    old = db.execute("SELECT id,file_size,mtime FROM assets WHERE path=?", (str(path),)).fetchone()
                if old and old["file_size"] == stat.st_size and old["mtime"] == stat.st_mtime:
                    with connect() as db:
                        db.execute("UPDATE assets SET missing=0,scan_token=? WHERE id=?", (job_id, old["id"]))
                else:
                    dhash, width, height, captured = calc_dhash(path)
                    digest = sha256(path)
                    with connect() as db:
                        if old:
                            db.execute("DELETE FROM semantic_embeddings WHERE asset_id=?", (old["id"],))
                        db.execute("""INSERT INTO assets(root_id,path,file_size,mtime,sha256,dhash,captured_at,width,height,missing,indexed_at,scan_token)
                            VALUES(?,?,?,?,?,?,?,?,?,0,?,?) ON CONFLICT(path) DO UPDATE SET
                            root_id=excluded.root_id,file_size=excluded.file_size,mtime=excluded.mtime,
                            sha256=excluded.sha256,dhash=excluded.dhash,
                            captured_at=COALESCE(excluded.captured_at,assets.captured_at),width=excluded.width,height=excluded.height,
                            missing=0,indexed_at=excluded.indexed_at,scan_token=excluded.scan_token""",
                            (root["id"], str(path), stat.st_size, stat.st_mtime, digest, dhash, captured, width, height, now(), job_id))
            except (OSError, PermissionError) as exc:
                update_job(job_id, current_path=str(path), message=f"跳过无法读取的文件：{type(exc).__name__}")
            total += 1
            if total % 10 == 0:
                update_job(job_id, progress=total, current_path=str(path), message=f"正在扫描与更新索引，已处理 {total} 张")
    with connect() as db:
        db.execute("UPDATE assets SET missing=1 WHERE root_id=? AND COALESCE(scan_token,'')<>?", (root["id"], job_id))
        db.execute("UPDATE roots SET online=1 WHERE id=?", (root["id"],))
    update_job(job_id, status="completed", progress=total, total=total, message=f"扫描完成，共索引 {total} 张图片")
    CONTROL.pop(job_id, None)


def semantic_candidates(query: str, filter_clauses: list[str], filter_args: list[object], min_score: float = 0.1) -> list[tuple[int, float]] | None:
    status = semantic_status()
    if not status["ready"] or not status["indexed"]:
        return None
    try:
        import numpy as np
        query_vector = semantic_engine().encode_text(query)
        where = " AND ".join(filter_clauses)
        heap: list[tuple[float, int]] = []
        with connect() as db:
            cursor = db.execute(f"""SELECT a.id,e.vector FROM assets a JOIN semantic_embeddings e ON e.asset_id=a.id
                WHERE {where} AND e.model_version=?""", [*filter_args, SEMANTIC_MODEL_VERSION])
            while True:
                batch = cursor.fetchmany(2048)
                if not batch: break
                vectors = np.stack([np.frombuffer(row["vector"], dtype=np.float16) for row in batch]).astype(np.float32)
                scores = vectors @ query_vector
                for row, score in zip(batch, scores):
                    score = float(score)
                    if score >= min_score:
                        heapq.heappush(heap, (score, int(row["id"])))
        return [(asset_id, score) for score, asset_id in sorted(heap, reverse=True)]
    except Exception as exc:
        print(f"[{now()}] semantic query unavailable: {type(exc).__name__}")
        return None


def list_assets(q: str = "", root_id: str = "", date_from: str = "", date_to: str = "", tag: str = "", rating: str = "", page: int = 1, limit: int = 60, min_score: float = 0.1) -> dict:
    page, limit = max(1, page), min(max(1, limit), 120)
    min_score = max(0.0, min(0.6, min_score))
    clauses = ["a.missing=0"]
    args: list[object] = []
    query = q.strip()
    year = re.search(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", query)
    if year:
        if not date_from: date_from = f"{year.group(1)}-01-01"
        if not date_to: date_to = f"{year.group(1)}-12-31"
        query = query[:year.start()] + " " + query[year.end():]
    if root_id:
        clauses.append("a.root_id=?"); args.append(int(root_id))
    if date_from:
        clauses.append("substr(COALESCE(a.captured_at,datetime(a.mtime,'unixepoch','localtime')),1,10)>=?"); args.append(date_from)
    if date_to:
        clauses.append("substr(COALESCE(a.captured_at,datetime(a.mtime,'unixepoch','localtime')),1,10)<=?"); args.append(date_to)
    if rating:
        clauses.append("a.rating>=?"); args.append(int(rating))
    if tag:
        clauses.append("EXISTS(SELECT 1 FROM asset_tags at JOIN tags t ON t.id=at.tag_id WHERE at.asset_id=a.id AND t.name LIKE ?)"); args.append(f"%{tag}%")
    filter_clauses, filter_args = list(clauses), list(args)
    relevance = "0.0"
    order_args: list[object] = []
    if query:
        # Pull year filters from free text and use local text bigrams as a fallback
        # where Chinese language embeddings are not configured.
        reduced = re.sub(r"照片|图片|图像|拍摄|一张|有|的", " ", query, flags=re.IGNORECASE)
        pieces = re.findall(r"[A-Za-z0-9_-]+|[\u4e00-\u9fff]+", reduced.lower())
        terms: list[str] = []
        for piece in pieces:
            if re.fullmatch(r"[\u4e00-\u9fff]+", piece) and len(piece) > 2:
                terms.extend(piece[i:i+2] for i in range(len(piece)-1))
            else:
                terms.append(piece)
        terms = list(dict.fromkeys(term for term in terms if term))[:12] or [query]
        groups, score_terms = [], []
        for term in terms:
            pattern = f"%{term}%"
            fts_term = '"' + term.replace('"', '""') + '"'
            fields = "a.path LIKE ? OR EXISTS(SELECT 1 FROM ocr_text o WHERE o.asset_id=a.id AND o.text LIKE ?) OR EXISTS(SELECT 1 FROM ocr_fts WHERE rowid=a.id AND ocr_fts MATCH ?) OR EXISTS(SELECT 1 FROM asset_tags at JOIN tags t ON t.id=at.tag_id WHERE at.asset_id=a.id AND t.name LIKE ?)"
            groups.append("(" + fields + ")")
            args.extend([pattern, pattern, fts_term, pattern])
            score_terms.append("(CASE WHEN (" + fields + ") THEN 1 ELSE 0 END)")
        clauses.append("(" + " OR ".join(groups) + ")")
        relevance = " + ".join(score_terms)
        for term in terms:
            pattern = f"%{term}%"
            fts_term = '"' + term.replace('"', '""') + '"'
            order_args.extend([pattern, pattern, fts_term, pattern])
    where = " AND ".join(clauses)
    semantic_hits = semantic_candidates(query, filter_clauses, filter_args, min_score) if query else None
    with connect() as db:
        total = db.execute(f"SELECT count(*) FROM assets a WHERE {where}", args).fetchone()[0]
        sql_args = list(args) + order_args
        # In hybrid mode, rank every semantic match above the selected cutoff and
        # every lexical match. The previous hard cap silently truncated all searches to 300.
        sql_limit = -1 if semantic_hits is not None else limit
        sql_offset = 0 if semantic_hits is not None else (page - 1) * limit
        sql_args += [sql_limit, sql_offset]
        rows = db.execute(f"""SELECT a.*, (SELECT group_concat(t.name, ', ') FROM asset_tags at JOIN tags t ON t.id=at.tag_id WHERE at.asset_id=a.id) tags,
                    (SELECT substr(text,1,240) FROM ocr_text o WHERE o.asset_id=a.id) ocr_snippet
                    FROM assets a WHERE {where} ORDER BY {relevance} DESC, COALESCE(a.captured_at,datetime(a.mtime,'unixepoch','localtime')) DESC, a.id DESC LIMIT ? OFFSET ?""", sql_args).fetchall()
        if semantic_hits is not None:
            by_id: dict[int, tuple[float, dict]] = {}
            hit_ids = [asset_id for asset_id, _ in semantic_hits]
            if hit_ids:
                item_by_id = {}
                # Keep the query below SQLite's parameter limit for large result sets.
                for start in range(0, len(hit_ids), 500):
                    batch_ids = hit_ids[start:start + 500]
                    marks = ",".join("?" for _ in batch_ids)
                    semantic_rows = db.execute(f"""SELECT a.*, (SELECT group_concat(t.name, ', ') FROM asset_tags at JOIN tags t ON t.id=at.tag_id WHERE at.asset_id=a.id) tags,
                        (SELECT substr(text,1,240) FROM ocr_text o WHERE o.asset_id=a.id) ocr_snippet
                        FROM assets a WHERE a.id IN ({marks})""", batch_ids).fetchall()
                    item_by_id.update({int(row["id"]): dict(row) for row in semantic_rows})
                for asset_id, score in semantic_hits:
                    if asset_id in item_by_id:
                        item_by_id[asset_id]["semantic_score"] = round(score, 4)
                        by_id[asset_id] = (score, item_by_id[asset_id])
            for row in rows:
                item = dict(row)
                item["semantic_score"] = by_id[item["id"]][0] if item["id"] in by_id else None
                # Exact filename, tag, and OCR matches remain easy to find alongside visual matches.
                lexical_priority = 1.0 + (item["semantic_score"] or 0.0)
                old = by_id.get(item["id"])
                if old is None or lexical_priority > old[0]: by_id[item["id"]] = (lexical_priority, item)
            ranked = sorted(by_id.values(), key=lambda x: (x[0], x[1].get("captured_at") or "", x[1]["id"]), reverse=True)
            total = len(ranked)
            result_items = [item for _, item in ranked[(page - 1) * limit:page * limit]]
            return {"items": result_items, "total": total, "page": page, "pages": (total + limit - 1) // limit, "semantic": True}
    return {"items": [dict(x) for x in rows], "total": total, "page": page, "pages": (total + limit - 1) // limit, "semantic": False}


def asset_detail(asset_id: int) -> dict | None:
    with connect() as db:
        row = db.execute("""SELECT a.*,r.online FROM assets a JOIN roots r ON r.id=a.root_id WHERE a.id=?""", (asset_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        result["tags"] = [r[0] for r in db.execute("SELECT t.name FROM tags t JOIN asset_tags at ON t.id=at.tag_id WHERE at.asset_id=? ORDER BY t.name", (asset_id,))]
        result["ocr_text"] = (db.execute("SELECT text FROM ocr_text WHERE asset_id=?", (asset_id,)).fetchone() or [""])[0]
        recipe = db.execute("SELECT recipe_json FROM edits WHERE asset_id=?", (asset_id,)).fetchone()
        result["recipe"] = json.loads(recipe[0]) if recipe else {"rotate": 0, "flip": False}
        return result


def make_thumbnail(asset_id: int) -> bytes | None:
    detail = asset_detail(asset_id)
    if not detail or not Path(detail["path"]).is_file() or not is_authorized(Path(detail["path"])): return None
    if Image is None: return None
    target = THUMB_DIR / f"{asset_id}.jpg"
    if target.exists() and target.stat().st_mtime >= Path(detail["path"]).stat().st_mtime:
        target.touch()
        return target.read_bytes()
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    with open_image(Path(detail["path"])) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((480, 360))
        im.save(target, "JPEG", quality=82, optimize=True)
    trim_thumbnail_cache()
    return target.read_bytes()


def cache_limit_bytes() -> int:
    with connect() as db:
        row = db.execute("SELECT value FROM settings WHERE key='thumbnail_cache_limit_gb'").fetchone()
    return int(float(row[0]) * 1024**3) if row else 100 * 1024**3


def trim_thumbnail_cache() -> None:
    if not THUMB_DIR.is_dir(): return
    files = sorted((p for p in THUMB_DIR.glob("*.jpg") if p.is_file()), key=lambda p: p.stat().st_mtime)
    total = sum(p.stat().st_size for p in files)
    limit = cache_limit_bytes()
    for item in files:
        if total <= limit: break
        try:
            size = item.stat().st_size
            item.unlink()
            total -= size
        except OSError:
            continue


def plan_operation(kind: str, source: Path, target: Path | None = None, recipe: dict | None = None) -> dict:
    with connect() as db:
        if not authorized(source, db) or not source.is_file():
            raise ValueError("源文件不在已授权图库中，或文件当前不可用")
        if kind in {"move", "rename"}:
            if target is None: raise ValueError("缺少目标路径")
            if not authorized(target.parent, db): raise ValueError("目标目录必须已添加到图库根目录")
            if target.exists(): raise ValueError("目标文件已存在，禁止覆盖")
            if kind == "rename" and target.parent != source.parent: raise ValueError("改名操作只能在原目录内执行")
        elif kind == "export":
            if target is None: raise ValueError("缺少导出路径")
            if target.exists(): raise ValueError("导出目标已存在，禁止覆盖")
        elif kind != "delete":
            raise ValueError("不支持的操作")
        op_id = uuid.uuid4().hex
        detail = {"size": source.stat().st_size, "name": source.name}
        db.execute("INSERT INTO operations(id,kind,source,target,status,details_json,created_at) VALUES(?,?,?,?,?,?,?)",
                   (op_id, kind, str(source), str(target) if target else None, "planned", json.dumps({**detail, "recipe": recipe or {}}), now()))
        return {"id": op_id, "kind": kind, "source": str(source), "target": str(target) if target else "回收站", "bytes": detail["size"], "status": "planned"}


def copy_verified(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + f".{uuid.uuid4().hex}.tmp")
    shutil.copy2(source, temp)
    if sha256(source) != sha256(temp):
        temp.unlink(missing_ok=True)
        raise OSError("复制后校验失败，源文件已保留")
    os.replace(temp, target)


def render_export(source: Path, target: Path, recipe: dict) -> None:
    if Image is None:
        raise RuntimeError("导出图像编辑需要 Pillow")
    with open_image(source) as im:
        im = ImageOps.exif_transpose(im)
        rotate = int(recipe.get("rotate", 0)) % 360
        if rotate: im = im.rotate(-rotate, expand=True)
        if recipe.get("flip"): im = ImageOps.mirror(im)
        crop = recipe.get("crop")
        if crop and len(crop) == 4:
            w, h = im.size
            box = tuple(int(v * edge) for v, edge in zip(crop, (w, h, w, h)))
            if box[2] > box[0] and box[3] > box[1]: im = im.crop(box)
        if recipe.get("width"):
            w = max(1, int(recipe["width"]))
            im.thumbnail((w, w * 10000), Image.Resampling.LANCZOS)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name + f".{uuid.uuid4().hex}.tmp")
        fmt = target.suffix.lstrip(".").upper()
        if fmt == "JPG": fmt = "JPEG"
        if fmt not in {"JPEG", "PNG", "WEBP", "TIFF", "BMP"}: fmt = "JPEG"
        if fmt == "JPEG" and im.mode not in {"RGB", "L"}: im = im.convert("RGB")
        im.save(temp, format=fmt)
        os.replace(temp, target)


def confirm_operation(op_id: str) -> dict:
    with connect() as db:
        op = db.execute("SELECT * FROM operations WHERE id=?", (op_id,)).fetchone()
        if not op: raise ValueError("操作计划不存在")
        if op["status"] != "planned": raise ValueError("操作计划已处理")
        source = Path(op["source"]); target = Path(op["target"]) if op["target"] else None
        details = json.loads(op["details_json"] or "{}")
        if not source.is_file() or not authorized(source, db): raise ValueError("源文件已变化，或授权已失效")
        if target and target.exists(): raise ValueError("目标文件已存在，未执行任何更改")
    undo: dict[str, object] = {"source": str(source), "target": str(target) if target else None}
    try:
        if op["kind"] in {"move", "rename"}:
            if not target or not is_authorized(target.parent): raise ValueError("目标目录授权已失效")
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.stat().st_dev == target.parent.stat().st_dev:
                source.rename(target)
            else:
                copy_verified(source, target); source.unlink()
        elif op["kind"] == "delete":
            TRASH_DIR.mkdir(parents=True, exist_ok=True)
            target = TRASH_DIR / f"{op_id}_{source.name}"
            copy_verified(source, target); source.unlink()
            undo["target"] = str(target)
        elif op["kind"] == "export":
            render_export(source, target, details.get("recipe", {}))
            undo["export"] = True
        with connect() as db:
            db.execute("UPDATE operations SET status='completed',undo_json=?,confirmed_at=? WHERE id=?", (json.dumps(undo), now(), op_id))
            if op["kind"] in {"move", "rename"}:
                db.execute("UPDATE assets SET path=?,missing=0 WHERE path=?", (str(target), str(source)))
            elif op["kind"] == "delete":
                db.execute("UPDATE assets SET path=?,missing=1 WHERE path=?", (str(target), str(source)))
        return {"id": op_id, "status": "completed", "message": "操作已完成，可在操作记录中撤销"}
    except Exception as exc:
        with connect() as db:
            db.execute("UPDATE operations SET status='failed' WHERE id=?", (op_id,))
        raise


def undo_operation(op_id: str) -> dict:
    with connect() as db:
        op = db.execute("SELECT * FROM operations WHERE id=?", (op_id,)).fetchone()
        if not op or op["status"] != "completed": raise ValueError("没有可撤销的已完成操作")
        info = json.loads(op["undo_json"] or "{}")
    source = Path(info["source"]); target = Path(info["target"]) if info.get("target") else None
    if op["kind"] == "export":
        if target and target.exists(): target.unlink()
    else:
        if not target or not target.exists(): raise ValueError("操作结果文件已不存在，无法撤销")
        if source.exists(): raise ValueError("原路径已被占用，无法安全撤销")
        source.parent.mkdir(parents=True, exist_ok=True)
        if source.parent.stat().st_dev == target.stat().st_dev: target.rename(source)
        else: copy_verified(target, source); target.unlink()
        with connect() as db:
            db.execute("UPDATE assets SET path=?,missing=0 WHERE path=?", (str(source), str(target)))
    with connect() as db: db.execute("UPDATE operations SET status='undone' WHERE id=?", (op_id,))
    return {"id": op_id, "status": "undone", "message": "已撤销"}


class Handler(BaseHTTPRequestHandler):
    server_version = "LocalImageLibrary/0.1"
    def log_message(self, fmt: str, *args: object) -> None:
        # Do not put filenames, OCR text, or request bodies in default logs.
        print(f"[{now()}] {self.command} {self.path.split('?')[0]} {args[1] if len(args)>1 else ''}")
    def send_json(self, data: object, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)
    def body(self) -> dict:
        length = min(int(self.headers.get("Content-Length", "0")), 2_000_000)
        return json.loads(self.rfile.read(length) or b"{}")
    def do_GET(self) -> None:
        u = urlparse(self.path); path = unquote(u.path)
        if path == "/":
            content = (APP_DIR / "index.html").read_text(encoding="utf-8").replace("__APP_TOKEN__", TOKEN).encode()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(content))); self.end_headers(); self.wfile.write(content); return
        if path in {"/app.js", "/style.css"}:
            file = APP_DIR / path.lstrip("/")
            content = file.read_bytes(); self.send_response(200); self.send_header("Content-Type", "text/javascript; charset=utf-8" if path.endswith("js") else "text/css; charset=utf-8"); self.send_header("Content-Length", str(len(content))); self.end_headers(); self.wfile.write(content); return
        try:
            if path == "/api/health":
                with connect() as db: total = db.execute("SELECT count(*) FROM assets WHERE missing=0").fetchone()[0]
                semantic = semantic_status()
                self.send_json({"ok": True, "images": total, "data_dir": str(DATA_DIR), "pillow": Image is not None, "heif": HEIF_AVAILABLE, "raw": RAW_AVAILABLE, "scan_extensions": sorted(IMAGE_EXTS), "preview_extensions": sorted((PIL_IMAGE_EXTS | ({".heic", ".heif"} if HEIF_AVAILABLE else set()) | (RAW_EXTS if RAW_AVAILABLE else set())) & IMAGE_EXTS), "ocr": False, "semantic_model": semantic["ready"], "semantic_indexed": semantic["indexed"], "semantic_pending": semantic["pending"], "semantic_device": semantic["device"], "thumbnail_cache_limit_gb": round(cache_limit_bytes()/1024**3, 1)}); return
            if path == "/api/semantic/status":
                self.send_json(semantic_status()); return
            if path == "/api/roots":
                with connect() as db: rows = [dict(r) for r in db.execute("SELECT r.*,count(a.id) image_count FROM roots r LEFT JOIN assets a ON a.root_id=r.id AND a.missing=0 GROUP BY r.id ORDER BY r.id")]
                for row in rows: row["online"] = Path(row["path"]).is_dir()
                self.send_json({"items": rows}); return
            if path == "/api/settings":
                self.send_json({"thumbnail_cache_limit_gb": round(cache_limit_bytes()/1024**3, 1)}); return
            if path == "/api/backup":
                backup_path = DATA_DIR / f"catalog-backup-{uuid.uuid4().hex}.db"
                source = connect(); destination = sqlite3.connect(backup_path)
                try: source.backup(destination)
                finally: destination.close(); source.close()
                size = backup_path.stat().st_size
                self.send_response(200); self.send_header("Content-Type", "application/vnd.sqlite3"); self.send_header("Content-Disposition", "attachment; filename=local-image-library-backup.db"); self.send_header("Content-Length", str(size)); self.send_header("Cache-Control", "no-store"); self.end_headers()
                try:
                    with backup_path.open("rb") as f: shutil.copyfileobj(f, self.wfile)
                finally: backup_path.unlink(missing_ok=True)
                return
            if path == "/api/jobs":
                with connect() as db: rows = [dict(r) for r in db.execute("SELECT j.*,r.path root_path FROM jobs j LEFT JOIN roots r ON r.id=j.root_id ORDER BY j.created_at DESC LIMIT 100")]
                self.send_json({"items": rows}); return
            if path == "/api/operations":
                with connect() as db: rows = [dict(r) for r in db.execute("SELECT * FROM operations ORDER BY created_at DESC LIMIT 100")]
                self.send_json({"items": rows}); return
            if path == "/api/duplicates":
                with connect() as db: rows = [dict(r) for r in db.execute("SELECT id,path,dhash FROM assets WHERE missing=0 AND dhash IS NOT NULL")]
                buckets: dict[str, list[dict]] = {}
                for row in rows:
                    value = int(row["dhash"], 16)
                    for shift in (0, 16, 32, 48):
                        buckets.setdefault(f"{shift}:{(value >> shift) & 0xffff}", []).append(row)
                groups: list[list[dict]] = []
                assigned: set[int] = set()
                for row in rows:
                    if row["id"] in assigned: continue
                    value = int(row["dhash"], 16)
                    candidates: dict[int, dict] = {}
                    for shift in (0, 16, 32, 48):
                        candidates.update((c["id"], c) for c in buckets[f"{shift}:{(value >> shift) & 0xffff}"] if c["id"] != row["id"])
                    near = [c for c in candidates.values() if (value ^ int(c["dhash"], 16)).bit_count() <= 5 and c["id"] not in assigned]
                    if near:
                        group = [row, *near]
                        groups.append(group)
                        assigned.update(x["id"] for x in group)
                result = [{"items": [{**x, "thumbnail": f"/api/assets/{x['id']}/thumbnail"} for x in g]} for g in groups[:200]]
                self.send_json({"items": result}); return
            if path.startswith("/api/assets/search"):
                params = json.loads(self.headers.get("X-Search-Body", "{}")) if False else {}
                from urllib.parse import parse_qs
                qs = parse_qs(u.query)
                result = list_assets(qs.get("q", [""])[0], qs.get("root", [""])[0], qs.get("from", [""])[0], qs.get("to", [""])[0], qs.get("tag", [""])[0], qs.get("rating", [""])[0], int(qs.get("page", ["1"])[0]), int(qs.get("limit", ["60"])[0]), float(qs.get("min_score", ["0.1"])[0]))
                self.send_json(result); return
            if path.startswith("/api/assets/") and path.endswith("/thumbnail"):
                aid = int(path.split("/")[3]); data = make_thumbnail(aid)
                if data is None: self.send_error(404); return
                self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.send_header("Cache-Control", "private, max-age=3600"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
            if path.startswith("/api/assets/") and path.endswith("/original"):
                aid = int(path.split("/")[3]); detail = asset_detail(aid)
                if not detail or not Path(detail["path"]).is_file() or not is_authorized(Path(detail["path"])): self.send_error(404); return
                source = Path(detail["path"]); size = source.stat().st_size
                self.send_response(200); self.send_header("Content-Type", mimetypes.guess_type(source.name)[0] or "application/octet-stream"); self.send_header("Content-Length", str(size)); self.send_header("Cache-Control", "private, no-store"); self.end_headers()
                with source.open("rb") as f: shutil.copyfileobj(f, self.wfile)
                return
            if path.startswith("/api/assets/"):
                aid = int(path.rsplit("/", 1)[1]); detail = asset_detail(aid)
                if detail is None: self.send_error(404); return
                self.send_json(detail); return
            self.send_error(404)
        except (ValueError, sqlite3.Error, OSError) as exc: self.send_json({"error": str(exc)}, 400)
    def do_DELETE(self) -> None:
        if self.headers.get("X-App-Token") != TOKEN:
            self.send_json({"error": "本地请求令牌无效"}, 403); return
        path = unquote(urlparse(self.path).path)
        try:
            if path.startswith("/api/operations/"):
                op_id = path.rsplit("/", 1)[1]
                with connect() as db:
                    op = db.execute("SELECT id,status FROM operations WHERE id=?", (op_id,)).fetchone()
                    if not op: raise ValueError("操作记录不存在")
                    db.execute("DELETE FROM operations WHERE id=?", (op_id,))
                self.send_json({"id": op_id, "deleted": True, "status": op["status"]}); return
            if not path.startswith("/api/roots/"):
                self.send_error(404); return
            root_id = int(path.rsplit("/", 1)[1])
            with connect() as db:
                root = db.execute("SELECT * FROM roots WHERE id=?", (root_id,)).fetchone()
                if not root: raise ValueError("图库目录不存在")
                active = db.execute("SELECT count(*) FROM jobs WHERE root_id=? AND status IN ('queued','running')", (root_id,)).fetchone()[0]
                if active: raise ValueError("该图库正在扫描，请先在任务中心暂停或取消扫描")
                asset_ids = [row[0] for row in db.execute("SELECT id FROM assets WHERE root_id=?", (root_id,))]
                db.execute("UPDATE jobs SET status='cancelled',message='图库目录已从索引中移除',updated_at=? WHERE root_id=? AND status='paused'", (now(), root_id))
                for operation in db.execute("SELECT id,source FROM operations WHERE status='planned'").fetchall():
                    if inside(Path(operation["source"]), Path(root["path"])):
                        db.execute("UPDATE operations SET status='cancelled' WHERE id=?", (operation["id"],))
                db.execute("DELETE FROM roots WHERE id=?", (root_id,))
            for asset_id in asset_ids:
                try: (THUMB_DIR / f"{asset_id}.jpg").unlink(missing_ok=True)
                except OSError: pass
            self.send_json({"id": root_id, "removed": True, "originals_deleted": False, "message": "已从本机索引移除；图库文件未删除"})
        except (ValueError, sqlite3.Error, OSError) as exc: self.send_json({"error": str(exc)}, 400)
    def do_POST(self) -> None:
        if self.headers.get("X-App-Token") != TOKEN:
            self.send_json({"error": "本地请求令牌无效"}, 403); return
        path = unquote(urlparse(self.path).path)
        try:
            if path.startswith("/api/assets/") and path.endswith("/edited-export"):
                aid = int(path.split("/")[3])
                if self.headers.get("Content-Type", "").split(";")[0].strip().lower() != "image/png": raise ValueError("编辑结果必须是 PNG 图片")
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 8 or length > 100 * 1024 * 1024: raise ValueError("图片文件为空或超过 100 MB")
                target_text = unquote(self.headers.get("X-Export-Path", "")).strip()
                if not target_text: raise ValueError("请指定保存路径")
                target = Path(target_text).expanduser().resolve()
                if target.suffix.lower() != ".png": raise ValueError("保存文件扩展名必须为 .png")
                if not target.parent.is_dir(): raise ValueError("保存目录不存在，请先选择或创建目录")
                if target.exists(): raise ValueError("目标文件已存在，禁止覆盖")
                payload = self.rfile.read(length)
                if not payload.startswith(b"\x89PNG\r\n\x1a\n"): raise ValueError("文件内容不是有效 PNG")
                if Image is None: raise ValueError("当前环境缺少 Pillow，无法验证 PNG 文件")
                with Image.open(io.BytesIO(payload)) as preview:
                    if preview.format != "PNG": raise ValueError("文件内容不是有效 PNG")
                    preview.verify()
                with connect() as db:
                    asset = db.execute("SELECT path FROM assets WHERE id=? AND missing=0", (aid,)).fetchone()
                    if not asset or not authorized(Path(asset["path"]), db) or not Path(asset["path"]).is_file(): raise ValueError("源图片不在已授权图库或当前不可访问")
                    source = Path(asset["path"])
                op_id = uuid.uuid4().hex
                try:
                    with target.open("xb") as output: output.write(payload)
                    details = json.dumps({"size": len(payload), "name": target.name, "recipe": {"editor": "canvas"}})
                    undo = json.dumps({"source": str(source), "target": str(target), "export": True})
                    with connect() as db:
                        db.execute("INSERT INTO operations(id,kind,source,target,status,undo_json,details_json,created_at,confirmed_at) VALUES(?,?,?,?,?,?,?,?,?)", (op_id, "export", str(source), str(target), "completed", undo, details, now(), now()))
                except Exception:
                    target.unlink(missing_ok=True)
                    raise
                self.send_json({"id": op_id, "saved": True, "target": str(target)}); return
            data = self.body()
            if path == "/api/settings":
                limit = max(1, min(500, int(data.get("thumbnail_cache_limit_gb", 100))))
                with connect() as db: db.execute("INSERT INTO settings(key,value) VALUES('thumbnail_cache_limit_gb',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str(limit),))
                trim_thumbnail_cache()
                self.send_json({"thumbnail_cache_limit_gb": limit}); return
            if path == "/api/semantic/index":
                status = semantic_status()
                if not status["ready"]:
                    raise ValueError("SigLIP2 模型或本机推理依赖尚未就绪")
                with connect() as db:
                    active = db.execute("SELECT id,status FROM jobs WHERE kind='semantic_index' AND status IN ('queued','running') ORDER BY created_at DESC LIMIT 1").fetchone()
                    if active:
                        self.send_json({"id": active["id"], "status": active["status"], "existing": True}, 202); return
                    jid = uuid.uuid4().hex
                    db.execute("INSERT INTO jobs(id,kind,status,created_at,updated_at,message) VALUES(?,?,?,?,?,?)", (jid, "semantic_index", "queued", now(), now(), "等待 SigLIP2 图片内容索引"))
                CONTROL[jid] = threading.Event(); SCAN_QUEUE.put(jid)
                self.send_json({"id": jid, "status": "queued"}, 202); return
            if path == "/api/assets/search":
                result = list_assets(str(data.get("query", data.get("q", ""))), str(data.get("root", "")), str(data.get("date_from", "")), str(data.get("date_to", "")), str(data.get("tag", "")), str(data.get("rating", "")), int(data.get("page", 1)), int(data.get("limit", 60)))
                self.send_json(result); return
            if path == "/api/roots":
                root = Path(str(data.get("path", ""))).expanduser().resolve()
                if not root.is_dir(): raise ValueError("请选择已存在的本地目录")
                with connect() as db:
                    cur = db.execute("INSERT OR IGNORE INTO roots(path,online,added_at) VALUES(?,1,?)", (str(root), now()))
                    row = db.execute("SELECT * FROM roots WHERE path=?", (str(root),)).fetchone()
                self.send_json(dict(row), 201); return
            if path.startswith("/api/roots/") and path.endswith("/scan"):
                root_id = int(path.split("/")[3])
                with connect() as db:
                    if not db.execute("SELECT 1 FROM roots WHERE id=?", (root_id,)).fetchone(): raise ValueError("图库不存在")
                    jid = uuid.uuid4().hex
                    db.execute("INSERT INTO jobs(id,kind,status,root_id,created_at,updated_at,message) VALUES(?,?,?,?,?,?,?)", (jid, "scan", "queued", root_id, now(), now(), "等待扫描"))
                CONTROL[jid] = threading.Event(); SCAN_QUEUE.put(jid); self.send_json({"id": jid, "status": "queued"}, 202); return
            if path.startswith("/api/jobs/") and path.endswith("/action"):
                jid = path.split("/")[3]; action = data.get("action")
                with connect() as db: job = db.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
                if not job: raise ValueError("任务不存在")
                if action == "pause":
                    CONTROL.setdefault(jid, threading.Event()).set()
                elif action == "resume":
                    if job["status"] not in {"paused", "failed"}: raise ValueError("该任务当前不可恢复")
                    update_job(jid, status="queued", message="等待恢复"); CONTROL[jid] = threading.Event(); SCAN_QUEUE.put(jid)
                elif action == "cancel":
                    CONTROL.setdefault(jid, threading.Event()).set(); update_job(jid, status="cancelled", message="任务已取消")
                else: raise ValueError("未知任务操作")
                self.send_json({"id": jid, "status": action}); return
            if path.startswith("/api/assets/") and path.endswith("/tag"):
                aid = int(path.split("/")[3]); name = str(data.get("name", "")).strip()[:80]
                if not name: raise ValueError("标签不能为空")
                with connect() as db:
                    db.execute("INSERT OR IGNORE INTO tags(name) VALUES(?)", (name,)); db.execute("INSERT OR IGNORE INTO asset_tags(asset_id,tag_id) SELECT ?,id FROM tags WHERE name=?", (aid, name))
                self.send_json(asset_detail(aid)); return
            if path.startswith("/api/assets/") and path.endswith("/rating"):
                aid = int(path.split("/")[3]); rating = max(0, min(5, int(data.get("rating", 0))))
                with connect() as db: db.execute("UPDATE assets SET rating=? WHERE id=?", (rating, aid))
                self.send_json(asset_detail(aid)); return
            if path.startswith("/api/assets/") and path.endswith("/ocr"):
                aid = int(path.split("/")[3]); text = str(data.get("text", ""))[:100000]
                with connect() as db: db.execute("INSERT INTO ocr_text(asset_id,text,language,model_version) VALUES(?,?,?,?) ON CONFLICT(asset_id) DO UPDATE SET text=excluded.text,language=excluded.language,model_version=excluded.model_version", (aid, text, "manual", "manual-entry"))
                self.send_json({"ok": True}); return
            if path.startswith("/api/assets/") and path.endswith("/edit"):
                aid = int(path.split("/")[3]); recipe = {"rotate": int(data.get("rotate", 0)) % 360, "flip": bool(data.get("flip", False)), "width": data.get("width")}
                with connect() as db: db.execute("INSERT INTO edits(asset_id,recipe_json) VALUES(?,?) ON CONFLICT(asset_id) DO UPDATE SET recipe_json=excluded.recipe_json,version=edits.version+1", (aid, json.dumps(recipe)))
                self.send_json(asset_detail(aid)); return
            if path == "/api/operations/plan":
                kind = str(data.get("kind", "")); source = Path(str(data.get("source", ""))).resolve(); target = Path(str(data["target"])).expanduser().resolve() if data.get("target") else None
                op = plan_operation(kind, source, target, data.get("recipe") or {}); self.send_json(op, 201); return
            if path.startswith("/api/operations/") and path.endswith("/confirm"):
                op_id = path.split("/")[3]; self.send_json(confirm_operation(op_id)); return
            if path.startswith("/api/operations/") and path.endswith("/undo"):
                op_id = path.split("/")[3]; self.send_json(undo_operation(op_id)); return
            self.send_error(404)
        except (ValueError, sqlite3.Error, OSError, RuntimeError) as exc: self.send_json({"error": str(exc)}, 400)


def main() -> None:
    init_db()
    threading.Thread(target=scan_worker, daemon=True, name="scan-worker").start()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Local Image Library ready: http://{HOST}:{PORT}")
    print(f"Data directory: {DATA_DIR}")
    if os.environ.get("LOCAL_IMAGE_LIBRARY_OPEN_BROWSER", "1") == "1":
        webbrowser.open_new_tab(f"http://{HOST}:{PORT}")
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__ == "__main__":
    main()
