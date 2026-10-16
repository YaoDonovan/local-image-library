import json
import io
import os
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from urllib.parse import quote, quote_plus

import server

try:
    from PIL import Image
except ImportError:
    Image = None


@unittest.skipIf(Image is None, "Pillow is needed for end-to-end image checks")
class AppFlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.old_data, cls.old_db, cls.old_thumb, cls.old_trash = server.DATA_DIR, server.DB_PATH, server.THUMB_DIR, server.TRASH_DIR
        server.DATA_DIR = Path(cls.temp.name) / "appdata"
        server.DB_PATH = server.DATA_DIR / "catalog.db"
        server.THUMB_DIR = server.DATA_DIR / "thumbnails"
        server.TRASH_DIR = server.DATA_DIR / "trash"
        cls.library = Path(cls.temp.name) / "library"
        cls.other = Path(cls.temp.name) / "other"
        cls.library.mkdir(); cls.other.mkdir()
        cls.image_path = cls.library / "sunset.jpg"
        Image.new("RGB", (32, 20), (240, 80, 30)).save(cls.image_path)
        cls.original_bytes = cls.image_path.read_bytes()
        (cls.library / "sunset-copy.jpg").write_bytes(cls.original_bytes)
        nested = cls.library / "subfolder"
        nested.mkdir()
        Image.new("RGB", (24, 18), (20, 160, 220)).save(nested / "nested.png")
        server.init_db()
        threading.Thread(target=server.scan_worker, daemon=True).start()
        cls.http = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        cls.server_thread = threading.Thread(target=cls.http.serve_forever, daemon=True)
        cls.server_thread.start()
        cls.base = f"http://127.0.0.1:{cls.http.server_port}"
        response = cls.call("POST", "/api/roots", {"path": str(cls.library)})
        cls.root_id = response["id"]

    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown(); cls.server_thread.join(timeout=2); cls.http.server_close(); cls.temp.cleanup()
        server.DATA_DIR, server.DB_PATH, server.THUMB_DIR, server.TRASH_DIR = cls.old_data, cls.old_db, cls.old_thumb, cls.old_trash

    @classmethod
    def call(cls, method, path, payload=None):
        body = json.dumps(payload or {}).encode() if method != "GET" else None
        headers = {"X-App-Token": server.TOKEN}
        if body is not None: headers["Content-Type"] = "application/json"
        req = Request(cls.base + path, data=body, headers=headers, method=method)
        with urlopen(req, timeout=5) as res:
            data = res.read()
            if res.headers.get_content_type() == "application/json": return json.loads(data)
            return data

    def test_scan_search_tag_thumbnail_and_safe_operations(self):
        job = self.call("POST", f"/api/roots/{self.root_id}/scan", {})
        deadline = time.time() + 10
        while time.time() < deadline:
            jobs = self.call("GET", "/api/jobs")["items"]
            match = next(j for j in jobs if j["id"] == job["id"])
            if match["status"] in {"completed", "failed"}: break
            time.sleep(.05)
        self.assertEqual(match["status"], "completed", match)
        self.assertEqual(self.call("GET", "/api/assets/search")["total"], 3)
        rows = self.call("GET", "/api/assets/search?q=sunset")
        self.assertEqual(rows["total"], 2)
        aid = rows["items"][0]["id"]
        self.assertEqual(self.call("GET", f"/api/assets/{aid}/original"), self.original_bytes)
        thumb = self.call("GET", f"/api/assets/{aid}/thumbnail")
        self.assertTrue(thumb.startswith(b"\xff\xd8"))
        duplicates = self.call("GET", "/api/duplicates")["items"]
        self.assertTrue(any({Path(x["path"]).name for x in group["items"]} >= {"sunset.jpg", "sunset-copy.jpg"} for group in duplicates))
        self.call("POST", f"/api/assets/{aid}/tag", {"name": "海边"})
        self.call("POST", f"/api/assets/{aid}/ocr", {"text": "订单号 1234 海边日落"})
        self.assertEqual(self.call("GET", "/api/assets/search?q=1234")["total"], 1)
        self.assertEqual(self.call("GET", f"/api/assets/search?tag={quote_plus('海边')}")["total"], 1)
        with server.connect() as db: db.execute("UPDATE assets SET captured_at='2024-07-04 18:00:00' WHERE id=?", (aid,))
        self.assertEqual(self.call("GET", f"/api/assets/search?q={quote_plus('2024年海边的日落')}")["total"], 1)
        bad = json.dumps({"kind":"move", "source":str(self.image_path), "target":str(self.other / "sunset.jpg")}).encode()
        req = Request(self.base + "/api/operations/plan", data=bad, headers={"X-App-Token":server.TOKEN,"Content-Type":"application/json"}, method="POST")
        with self.assertRaises(HTTPError) as error: urlopen(req, timeout=5)
        self.assertEqual(error.exception.code, 400)
        destination = self.library / "renamed.jpg"
        plan = self.call("POST", "/api/operations/plan", {"kind":"rename", "source":str(self.image_path), "target":str(destination)})
        self.call("POST", f"/api/operations/{plan['id']}/confirm", {})
        self.assertTrue(destination.exists()); self.assertFalse(self.image_path.exists())
        self.call("POST", f"/api/operations/{plan['id']}/undo", {})
        self.assertTrue(self.image_path.exists()); self.assertEqual(self.image_path.read_bytes(), self.original_bytes)
        plan = self.call("POST", "/api/operations/plan", {"kind":"delete", "source":str(self.image_path)})
        self.call("POST", f"/api/operations/{plan['id']}/confirm", {})
        self.assertFalse(self.image_path.exists())
        self.call("POST", f"/api/operations/{plan['id']}/undo", {})
        self.assertTrue(self.image_path.exists()); self.assertEqual(self.image_path.read_bytes(), self.original_bytes)
        edited = self.library / "sunset-export.png"
        plan = self.call("POST", "/api/operations/plan", {"kind":"export", "source":str(self.image_path), "target":str(edited), "recipe":{"rotate":90}})
        self.call("POST", f"/api/operations/{plan['id']}/confirm", {})
        with Image.open(edited) as exported: self.assertEqual(exported.size, (20, 32))
        self.assertEqual(self.image_path.read_bytes(), self.original_bytes)
        self.call("POST", f"/api/operations/{plan['id']}/undo", {})
        self.assertFalse(edited.exists())
        plan = self.call("POST", "/api/operations/plan", {"kind":"delete", "source":str(self.image_path)})
        self.call("DELETE", f"/api/operations/{plan['id']}")
        self.assertNotIn(plan['id'], [item['id'] for item in self.call("GET", "/api/operations")["items"]])
        self.assertTrue(self.image_path.exists())
        canvas_png = io.BytesIO(); Image.new("RGB", (7, 9), (20, 40, 60)).save(canvas_png, format="PNG")
        output = Path(self.temp.name) / "canvas-edited.png"
        req = Request(self.base + f"/api/assets/{aid}/edited-export", data=canvas_png.getvalue(), headers={"X-App-Token":server.TOKEN,"Content-Type":"image/png","X-Export-Path":quote(str(output), safe="")}, method="POST")
        with urlopen(req, timeout=5) as response: export = json.loads(response.read())
        self.assertTrue(export["saved"]); self.assertTrue(output.is_file())
        self.call("POST", f"/api/operations/{export['id']}/undo", {})
        self.assertFalse(output.exists())
        removed = self.call("DELETE", f"/api/roots/{self.root_id}")
        self.assertFalse(removed["originals_deleted"])
        self.assertTrue(self.image_path.exists())
        self.assertTrue((self.library / "subfolder" / "nested.png").exists())
        self.assertEqual(self.call("GET", "/api/assets/search")["total"], 0)

    def test_siglip_local_index_and_content_search(self):
        if not server.semantic_status()["ready"]:
            self.skipTest("本机未安装完整的 SigLIP2 模型和推理依赖")
        root = self.call("POST", "/api/roots", {"path": str(self.library)})
        scan = self.call("POST", f"/api/roots/{root['id']}/scan", {})
        deadline = time.time() + 20
        while time.time() < deadline:
            job = next(j for j in self.call("GET", "/api/jobs")["items"] if j["id"] == scan["id"])
            if job["status"] in {"completed", "failed"}: break
            time.sleep(.05)
        self.assertEqual(job["status"], "completed", job)
        index = self.call("POST", "/api/semantic/index", {})
        deadline = time.time() + 120
        while time.time() < deadline:
            job = next(j for j in self.call("GET", "/api/jobs")["items"] if j["id"] == index["id"])
            if job["status"] in {"completed", "failed"}: break
            time.sleep(.25)
        self.assertEqual(job["status"], "completed", job)
        self.assertEqual(job["progress"], 3)
        results = self.call("GET", f"/api/assets/search?q={quote_plus('蓝色的照片')}&min_score=0")
        self.assertTrue(results["semantic"])
        self.assertEqual(results["total"], 3)
        self.assertTrue(all(isinstance(item["semantic_score"], float) for item in results["items"]))
        strict = self.call("GET", f"/api/assets/search?q={quote_plus('蓝色的照片')}&min_score=1")
        self.assertTrue(strict["semantic"])
        self.assertEqual(strict["total"], 0)

    def test_write_routes_require_local_token(self):
        req = Request(self.base + "/api/roots", data=json.dumps({"path":str(self.library)}).encode(), headers={"Content-Type":"application/json"}, method="POST")
        with self.assertRaises(HTTPError) as error:
            urlopen(req, timeout=5)
        self.assertEqual(error.exception.code, 403)

    def test_local_settings_and_sqlite_backup(self):
        settings = self.call("POST", "/api/settings", {"thumbnail_cache_limit_gb": 2})
        self.assertEqual(settings["thumbnail_cache_limit_gb"], 2)
        self.assertEqual(self.call("GET", "/api/settings")["thumbnail_cache_limit_gb"], 2)
        backup = self.call("GET", "/api/backup")
        self.assertTrue(backup.startswith(b"SQLite format 3\x00"))

    def test_local_ui_shell_is_served_with_private_token(self):
        page = self.call("GET", "/")
        self.assertIn(b"window.APP_TOKEN", page)
        self.assertIn(server.TOKEN.encode(), page)
        self.assertIn(b"/style.css", page)
        self.assertIn(b"scan-progress-banner", page)
        self.assertIn(b'id="undo-edit"', page)
        self.assertIn(b'id="crop-apply"', page)
        self.assertIn(b'id="text-editor"', page)
        self.assertIn(b'class="brush-size-control"', page)
        script = self.call("GET", "/app.js")
        self.assertIn(b"function loadAssets", script)
        self.assertIn(b"function updateScanBanner", script)
        self.assertIn(b"async function pushUndoSnapshot", script)
        self.assertIn(b"function mosaicDab", script)
        self.assertIn(b"function addTextToCanvas", script)
        self.assertIn("授权并扫描".encode(), page)


if __name__ == "__main__":
    unittest.main()
