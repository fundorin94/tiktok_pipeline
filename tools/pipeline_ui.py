"""Local web UI for running the pipeline.

The CLI needs a case id, a stage name and the right environment variables in
the right order, which is a lot to remember for something you run twenty
times a day while tuning models. This serves the same seven stages as
buttons, streams the log while a stage runs, and shows what each case
already has on disk.

Run:  venv/Scripts/python.exe tools/pipeline_ui.py
Then open http://localhost:8730/
"""
import html
import json
import os
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import CASES_DIR, DB_PATH  # noqa: E402
from orchestrator.cost import estimate_cost  # noqa: E402
from orchestrator.db import Database  # noqa: E402

PORT = 8730
PYTHON = ROOT / "venv" / "Scripts" / "python.exe"
STAGES = ["story", "script", "archive", "voiceover", "video", "metadata", "publish"]

# One run at a time: every heavy stage wants the same GPU, and two archive
# stages at once would fight over it and thrash VRAM.
_run = {"proc": None, "case": "", "stage": "", "log": deque(maxlen=400), "started": 0.0}
_lock = threading.Lock()


def _running() -> bool:
    proc = _run["proc"]
    return proc is not None and proc.poll() is None


def _cases() -> list:
    if not CASES_DIR.exists():
        return []
    return sorted(d.name for d in CASES_DIR.iterdir() if d.is_dir())


def _case_state(case_id: str) -> dict:
    """What this case already has on disk, so it's obvious which stage is
    next without opening the folder."""
    d = CASES_DIR / case_id
    videos = sorted((d / "video").glob("part*.mp4")) if (d / "video").exists() else []
    ai = list((d / "media" / "ai_generated").glob("*.png")) if (d / "media" / "ai_generated").exists() else []
    real = list((d / "media" / "accepted").glob("*")) if (d / "media" / "accepted").exists() else []
    return {
        "brief": (d / "brief.json").exists(),
        "script": (d / "script.json").exists(),
        "media": (d / "media_manifest.json").exists(),
        "audio": (d / "audio").exists(),
        "videos": len(videos),
        "ai_frames": len(ai),
        "real_photos": len(real),
        "metadata": (d / "metadata.json").exists(),
    }


def _start(case_id: str, stage: str, topic: str, fast: bool, max_parts: str) -> str:
    if _running():
        return "A stage is already running."
    if not case_id:
        return "Case id is required."

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    if fast:
        env["PIPELINE_FAST_IMAGES"] = "1"
    if max_parts.strip().isdigit() and int(max_parts) > 0:
        env["PIPELINE_MAX_PARTS"] = max_parts.strip()

    cmd = [str(PYTHON), "-u", "run_pipeline.py", "--case-id", case_id, "--stage", stage]
    if topic.strip():
        cmd += ["--topic", topic.strip()]

    with _lock:
        _run["log"].clear()
        _run["case"], _run["stage"], _run["started"] = case_id, stage, time.time()
        _run["proc"] = subprocess.Popen(
            cmd, cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        )
    threading.Thread(target=_pump, args=(_run["proc"],), daemon=True).start()
    return ""


def _pump(proc) -> None:
    """Collect output lines. Progress bars from diffusers repeat the same
    line hundreds of times, so those are collapsed instead of flooding."""
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        if "it/s" in line or "%|" in line:
            with _lock:
                if _run["log"] and _run["log"][-1].startswith("  ...generating"):
                    _run["log"][-1] = "  ...generating (progress)"
                else:
                    _run["log"].append("  ...generating (progress)")
            continue
        with _lock:
            _run["log"].append(line)


def _topics() -> dict:
    """Topic stored per case. Shown in the table because it decides what the
    research stage looks up, and it isn't visible anywhere else."""
    try:
        db = Database(DB_PATH)
        with db._connect() as conn:
            return {r["id"]: (r["topic"] or "") for r in conn.execute("SELECT id, topic FROM cases")}
    except Exception:
        return {}


def _costs() -> tuple:
    try:
        db = Database(DB_PATH)
        return estimate_cost(db.get_usage(_run["case"])) if _run["case"] else 0.0, estimate_cost(db.get_usage())
    except Exception:
        return 0.0, 0.0


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Pipeline</title>
<style>
 :root {{ color-scheme: light dark; }}
 body {{ font-family: system-ui, sans-serif; max-width: 1000px; margin: 32px auto;
        padding: 0 20px; line-height: 1.5; }}
 h1 {{ font-size: 22px; margin-bottom: 4px; }}
 .row {{ display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-start; }}
 .card {{ border: 1px solid #8884; border-radius: 10px; padding: 16px; margin: 14px 0; flex: 1; min-width: 300px; }}
 .btn {{ background: #fe2c55; color: #fff; border: 0; border-radius: 7px;
        padding: 8px 14px; font-size: 14px; cursor: pointer; }}
 .btn.sec {{ background: #6b7280; }}
 .btn:disabled {{ opacity: .45; cursor: not-allowed; }}
 input, select {{ padding: 7px; font-size: 14px; border-radius: 6px;
        border: 1px solid #8886; background: transparent; color: inherit; }}
 pre {{ background: #0b0d10; color: #d6e2f0; padding: 12px; border-radius: 8px;
        max-height: 420px; overflow: auto; font-size: 12.5px; white-space: pre-wrap; }}
 table {{ border-collapse: collapse; width: 100%; font-size: 13.5px; }}
 td, th {{ text-align: left; padding: 5px 8px; border-bottom: 1px solid #8883; }}
 .muted {{ color: #8a8f98; font-size: 13px; }}
 .live {{ color: #0a7d28; font-weight: 600; }}
 .idle {{ color: #8a8f98; }}
</style></head><body>
<h1>True Crime Pipeline</h1>
<p class="muted">Local control panel. One stage runs at a time — they share the GPU.</p>
{body}
<script>
setInterval(async () => {{
  const r = await fetch('/status');
  const s = await r.json();
  document.getElementById('log').textContent = s.log;
  document.getElementById('state').innerHTML = s.state;
  document.querySelectorAll('.needs-idle').forEach(b => b.disabled = s.running);
  if (!s.running && window._wasRunning) location.reload();
  window._wasRunning = s.running;
}}, 1500);
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, payload: bytes, ctype="text/html; charset=utf-8", status=200):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/":
            self._send(PAGE.format(body=self._body()).encode("utf-8"))
        elif path == "/status":
            with _lock:
                log = "\n".join(_run["log"]) or "(no output yet)"
            self._send(json.dumps({
                "running": _running(), "log": log, "state": self._state_line(),
            }).encode("utf-8"), "application/json")
        elif path == "/stop":
            proc = _run["proc"]
            if proc and proc.poll() is None:
                proc.terminate()
            self.send_response(302); self.send_header("Location", "/"); self.end_headers()
        else:
            self._send(b"not found", status=404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        form = urllib.parse.parse_qs(self.rfile.read(length).decode("utf-8"))
        # The "full pipeline" button posts its own flag rather than a second
        # field named "stage", which would collide with the dropdown.
        stage = "all" if form.get("full") else form.get("stage", ["story"])[0]
        error = _start(
            form.get("case_id", [""])[0].strip(),
            stage,
            form.get("topic", [""])[0],
            form.get("fast", [""])[0] == "on",
            form.get("max_parts", [""])[0],
        )
        if error:
            with _lock:
                _run["log"].append(f"!! {error}")
        self.send_response(302); self.send_header("Location", "/"); self.end_headers()

    def _state_line(self) -> str:
        if _running():
            mins = (time.time() - _run["started"]) / 60
            return (f'<span class="live">running: {html.escape(_run["stage"])} '
                    f'on {html.escape(_run["case"])} &mdash; {mins:.0f} min</span>')
        if _run["stage"]:
            return f'<span class="idle">last run: {html.escape(_run["stage"])} on {html.escape(_run["case"])} (finished)</span>'
        return '<span class="idle">idle</span>'

    def _body(self) -> str:
        cases = _cases()
        opts = "".join(f'<option value="{html.escape(c)}">{html.escape(c)}</option>' for c in cases)
        stage_opts = "".join(f'<option value="{s}">{s}</option>' for s in STAGES)

        topics = _topics()
        rows = ""
        for c in cases:
            st = _case_state(c)
            def mark(ok): return "yes" if ok else "&mdash;"
            topic = topics.get(c) or '<span style="color:#b3261e">not set</span>'
            rows += (
                f"<tr><td><b>{html.escape(c)}</b></td>"
                f"<td>{html.escape(topic) if topics.get(c) else topic}</td>"
                f"<td>{mark(st['brief'])}</td><td>{mark(st['script'])}</td>"
                f"<td>{st['real_photos']} real / {st['ai_frames']} ai</td>"
                f"<td>{mark(st['audio'])}</td><td>{st['videos']}</td>"
                f"<td>{mark(st['metadata'])}</td></tr>"
            )
        table = (
            "<table><tr><th>case</th><th>topic</th><th>brief</th><th>script</th><th>frames</th>"
            f"<th>audio</th><th>videos</th><th>meta</th></tr>{rows}</table>"
            if rows else '<p class="muted">No cases yet — enter a new id and topic below.</p>'
        )

        case_cost, total_cost = _costs()
        return f"""
<div class="card"><h2 style="margin-top:0;font-size:17px">Run a stage</h2>
<form method="post" class="row" style="gap:10px;align-items:center">
  <input name="case_id" list="cases" placeholder="case id" required style="width:150px">
  <datalist id="cases">{opts}</datalist>
  <select name="stage">{stage_opts}</select>
  <input name="topic" placeholder="topic (first run only)" style="width:200px">
  <input name="max_parts" placeholder="max parts" style="width:90px">
  <label class="muted"><input type="checkbox" name="fast"> fast images (SD 1.5)</label>
  <button class="btn needs-idle" type="submit">Run stage</button>
  <button class="btn needs-idle" type="submit" name="full" value="1"
          style="background:#1f6feb">Run full pipeline</button>
  <a class="btn sec" href="/stop" style="text-decoration:none">Stop</a>
</form>
<p class="muted" style="margin-bottom:0"><b>Run full pipeline</b> goes story &rarr; script &rarr;
archive &rarr; voiceover &rarr; video &rarr; metadata &rarr; publish in one go, ignoring the
stage dropdown. Publishing stays a dry run unless <code>PUBLISH_DRY_RUN=false</code> is set.
Topic matters only on the first <code>story</code> run — without it a random case is
researched. Expect a few hours on SDXL, well under one with fast images.</p></div>

<div class="card"><div id="state">{self._state_line()}</div>
<p class="muted">Cost — this case: ${case_cost:.2f} &middot; all cases: ${total_cost:.2f}</p>
<pre id="log">(no output yet)</pre></div>

<div class="card"><h2 style="margin-top:0;font-size:17px">Cases on disk</h2>{table}</div>
"""


def main():
    if not PYTHON.exists():
        raise SystemExit(f"python not found at {PYTHON}")
    server = HTTPServer(("localhost", PORT), Handler)
    print(f"Pipeline UI on http://localhost:{PORT}/  (Ctrl+C to stop)")
    threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{PORT}/")).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
