from __future__ import annotations

import http.server
import json
import os
import threading
import webbrowser


# ---------------------------------------------------------------------------
# Embedded viewer HTML
# ---------------------------------------------------------------------------

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ClimbAnalyze Viewer</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:#111;color:#eee;height:100vh;display:flex;flex-direction:column;overflow:hidden}

#header{padding:10px 18px;background:#1a1a1a;border-bottom:1px solid #2e2e2e;display:flex;align-items:center;gap:10px;flex-shrink:0}
#header h1{font-size:15px;font-weight:600;color:#fff;letter-spacing:-.01em}
#status-badge{font-size:11px;padding:2px 8px;border-radius:10px;background:#1e5c38;color:#5dd;font-weight:500}
#status-badge.low_quality{background:#5c3d0a;color:#fbba}
#status-badge.no_person_detected{background:#5c1010;color:#faa}
#meta-info{font-size:11px;color:#666;margin-left:auto}

#main{display:flex;flex:1;overflow:hidden}

/* --- player side --- */
#player-section{flex:1;display:flex;flex-direction:column;padding:14px 14px 10px;gap:10px;min-width:0;overflow:hidden}

video{width:100%;flex:1;min-height:0;background:#000;border-radius:6px;display:block;object-fit:contain}

#tl-wrap{position:relative;height:36px;background:#1e1e1e;border-radius:6px;cursor:pointer;user-select:none;flex-shrink:0;border:1px solid #2e2e2e}
#tl-progress{position:absolute;left:0;top:0;height:100%;background:#333;border-radius:6px;pointer-events:none;transition:width .1s linear}
#tl-thumb{position:absolute;top:50%;transform:translate(-50%,-50%);width:11px;height:11px;background:#fff;border-radius:50%;pointer-events:none;box-shadow:0 0 4px rgba(0,0,0,.6)}
.tl-marker{position:absolute;top:4px;bottom:4px;width:4px;border-radius:2px;cursor:pointer;transform:translateX(-50%)}
.tl-marker:hover{width:6px;top:2px;bottom:2px}
.tl-marker.low{background:#2a7a5a}
.tl-marker.medium{background:#c88000}
.tl-marker.high{background:#c03030}
.tl-marker.active{outline:2px solid #fff;outline-offset:1px}

#tl-label{font-size:11px;color:#555;margin-top:2px;min-height:16px;padding-left:2px}

/* --- sidebar --- */
#sidebar{width:310px;display:flex;flex-direction:column;border-left:1px solid #2e2e2e;overflow:hidden;flex-shrink:0}
#sidebar-hdr{padding:10px 14px;background:#1a1a1a;font-size:12px;font-weight:600;border-bottom:1px solid #2e2e2e;color:#ccc}
#issue-list{flex:1;overflow-y:auto;padding:8px;display:flex;flex-direction:column;gap:5px}
#issue-list::-webkit-scrollbar{width:6px}
#issue-list::-webkit-scrollbar-thumb{background:#333;border-radius:3px}

.issue-card{padding:9px 11px;background:#1a1a1a;border:1px solid #2e2e2e;border-radius:6px;cursor:pointer;transition:background .12s,border-color .12s}
.issue-card:hover{background:#222;border-color:#444}
.issue-card.active{border-color:#888;background:#222}
.ic-head{display:flex;align-items:center;gap:7px;margin-bottom:3px}
.ic-label{font-size:13px;font-weight:500;flex:1}
.sev{font-size:10px;padding:1px 6px;border-radius:8px;text-transform:uppercase;letter-spacing:.04em;font-weight:600}
.sev.low{background:#0d2e22;color:#3ca}
.sev.medium{background:#2e1e05;color:#d90}
.sev.high{background:#2e0d0d;color:#d44}
.ic-time{font-size:11px;color:#666}
.ic-msg{font-size:12px;color:#aaa;margin-top:3px;line-height:1.4}
.ic-rec{font-size:12px;color:#6af;margin-top:2px;line-height:1.4}

/* --- annotated overlay --- */
#overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.88);z-index:100;align-items:center;justify-content:center;flex-direction:column;gap:10px}
#overlay.show{display:flex}
#overlay-panel{display:flex;gap:16px;align-items:flex-start;max-width:96vw;max-height:86vh}
#overlay-left,#overlay-right{display:flex;flex-direction:column;align-items:center;gap:5px;min-width:0}
.overlay-panel-label{font-size:11px;color:#666;letter-spacing:.06em;text-transform:uppercase;flex-shrink:0}
#overlay-img{max-height:80vh;max-width:46vw;border-radius:8px;box-shadow:0 8px 40px rgba(0,0,0,.8)}
#overlay-correction{max-height:80vh;max-width:46vw;border-radius:8px;box-shadow:0 8px 40px rgba(0,0,0,.8)}
#overlay-close{position:absolute;top:14px;right:18px;font-size:20px;cursor:pointer;color:#bbb;background:rgba(0,0,0,.55);width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;line-height:1;z-index:1}
#overlay-close:hover{color:#fff}
#overlay-countdown{font-size:12px;color:#555;padding:3px 12px;background:rgba(0,0,0,.5);border-radius:12px;flex-shrink:0}
</style>
</head>
<body>
<div id="header">
  <h1>ClimbAnalyze</h1>
  <span id="status-badge"></span>
  <span id="meta-info"></span>
</div>
<div id="main">
  <div id="player-section">
    <video id="video" controls preload="metadata"></video>
    <div id="tl-wrap" title="Click to seek">
      <div id="tl-progress"></div>
      <div id="tl-thumb"></div>
    </div>
    <div id="tl-label">Loading…</div>
  </div>
  <div id="sidebar">
    <div id="sidebar-hdr">Issues <span id="issue-count" style="color:#555;font-weight:400"></span></div>
    <div id="issue-list"></div>
  </div>
</div>
<div id="overlay">
  <span id="overlay-close">&#x2715;</span>
  <div id="overlay-panel">
    <div id="overlay-left">
      <span class="overlay-panel-label">Detected issue</span>
      <img id="overlay-img" src="" alt="">
    </div>
    <div id="overlay-right" style="display:none">
      <span class="overlay-panel-label">Correct form</span>
      <img id="overlay-correction" src="" alt="">
    </div>
  </div>
  <span id="overlay-countdown"></span>
</div>

<script>
'use strict';
const video      = document.getElementById('video');
const tlWrap     = document.getElementById('tl-wrap');
const tlProg     = document.getElementById('tl-progress');
const tlThumb    = document.getElementById('tl-thumb');
const tlLabel    = document.getElementById('tl-label');
const overlay    = document.getElementById('overlay');
const overlayImg = document.getElementById('overlay-img');
const overlayCorr = document.getElementById('overlay-correction');
const overlayRight = document.getElementById('overlay-right');
const countdownEl  = document.getElementById('overlay-countdown');

let data = null;
let prevTime = 0;
let activeId = null;

// Auto-resume state — only set when overlay was opened by auto-pause during playback
let autopaused = false;
let resumeInterval = null;

function fmt(ms) {
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s / 60);
  return m > 0 ? m + ':' + String(s % 60).padStart(2,'0') : s + 's';
}

async function init() {
  const resp = await fetch('/api/analysis');
  data = await resp.json();

  const badge = document.getElementById('status-badge');
  badge.textContent = data.status;
  badge.className = data.status;

  const src = data.source;
  document.getElementById('meta-info').textContent =
    src.width + '\\xd7' + src.height + '\\u2002' + src.fps + 'fps\\u2002' +
    (src.durationMs / 1000).toFixed(1) + 's\\u2002\\u00b7\\u2002' + data.issues.length + ' issues';

  video.src = '/video';
  document.getElementById('issue-count').textContent = '(' + data.issues.length + ')';

  buildIssueList();
  buildTimeline();
  setupEvents();

  tlLabel.textContent = data.issues.length
    ? 'Colored markers show issue start times — click to jump'
    : 'No issues detected';
}

function buildIssueList() {
  const list = document.getElementById('issue-list');
  list.innerHTML = '';
  data.issues.forEach(iss => {
    const card = document.createElement('div');
    card.className = 'issue-card';
    card.id = 'card-' + iss.id;
    card.innerHTML =
      '<div class="ic-head">' +
        '<span class="ic-label">' + esc(iss.label) + '</span>' +
        '<span class="sev ' + iss.severity + '">' + iss.severity + '</span>' +
      '</div>' +
      '<div class="ic-time">' + fmt(iss.startMs) + ' – ' + fmt(iss.endMs) + '</div>' +
      '<div class="ic-msg">' + esc(iss.message) + '</div>' +
      '<div class="ic-rec">\\u2192 ' + esc(iss.recommendation) + '</div>';
    card.addEventListener('click', () => jumpTo(iss));
    list.appendChild(card);
  });
}

function buildTimeline() {
  document.querySelectorAll('.tl-marker').forEach(el => el.remove());
  const dur = data.source.durationMs;
  data.issues.forEach(iss => {
    const m = document.createElement('div');
    m.className = 'tl-marker ' + iss.severity;
    m.id = 'marker-' + iss.id;
    m.style.left = (iss.startMs / dur * 100) + '%';
    m.title = iss.label + '  ' + fmt(iss.startMs);
    m.addEventListener('click', e => { e.stopPropagation(); jumpTo(iss); });
    tlWrap.appendChild(m);
  });
}

// Find the annotated frame path for an issue + role
function findFrame(issId, role) {
  const af = (data.artifacts.annotatedFrames || [])
    .find(f => f.issueId === issId && f.frameRole === role);
  return af ? '/' + af.path.replace(/^\\//, '') : null;
}

function jumpTo(iss) {
  // Manual jump — no auto-resume
  clearResumeTimer();
  autopaused = false;
  video.currentTime = iss.startMs / 1000;
  prevTime = iss.startMs / 1000;
  video.pause();
  setActive(iss.id);

  const peakSrc = findFrame(iss.id, 'peak');
  const corrSrc = findFrame(iss.id, 'correction');
  if (peakSrc) showOverlay(peakSrc, corrSrc, false);
}

function setActive(id) {
  activeId = id;
  document.querySelectorAll('.issue-card').forEach(c => c.classList.remove('active'));
  document.querySelectorAll('.tl-marker').forEach(m => m.classList.remove('active'));
  if (id) {
    document.getElementById('card-' + id)?.classList.add('active');
    document.getElementById('card-' + id)?.scrollIntoView({block:'nearest', behavior:'smooth'});
    document.getElementById('marker-' + id)?.classList.add('active');
  }
}

// showOverlay(peakSrc, corrSrc, startTimer)
// startTimer=true only when triggered by auto-pause during playback
function showOverlay(peakSrc, corrSrc, startTimer) {
  overlayImg.src = peakSrc;
  if (corrSrc) {
    overlayCorr.src = corrSrc;
    overlayRight.style.display = 'flex';
  } else {
    overlayRight.style.display = 'none';
  }
  overlay.classList.add('show');

  if (startTimer) {
    autopaused = true;
    startResumeTimer();
  }
}

function hideOverlay() {
  clearResumeTimer();
  overlay.classList.remove('show');
  // If the overlay was opened by auto-pause, close also resumes playback
  if (autopaused) {
    autopaused = false;
    video.play();
  }
}

function startResumeTimer() {
  clearResumeTimer();
  let secs = 10;
  countdownEl.textContent = 'Resuming in ' + secs + 's';
  resumeInterval = setInterval(() => {
    secs -= 1;
    if (secs > 0) {
      countdownEl.textContent = 'Resuming in ' + secs + 's';
    } else {
      clearResumeTimer();
      hideOverlay();  // hideOverlay checks autopaused and calls video.play()
    }
  }, 1000);
}

function clearResumeTimer() {
  if (resumeInterval !== null) {
    clearInterval(resumeInterval);
    resumeInterval = null;
  }
  countdownEl.textContent = '';
}

function setupEvents() {
  document.getElementById('overlay-close').addEventListener('click', hideOverlay);
  overlay.addEventListener('click', e => { if (e.target === overlay) hideOverlay(); });

  video.addEventListener('seeked', () => { prevTime = video.currentTime; });

  video.addEventListener('play', () => {
    clearResumeTimer();
    autopaused = false;
    prevTime = video.currentTime;
    overlay.classList.remove('show');
    setActive(null);
  });

  video.addEventListener('timeupdate', () => {
    const now = video.currentTime;
    const dur = data.source.durationMs / 1000;
    const pct = dur > 0 ? now / dur * 100 : 0;
    tlProg.style.width = pct + '%';
    tlThumb.style.left = pct + '%';

    // Auto-pause when playback crosses an issue start (forward only)
    if (!video.paused) {
      for (const iss of data.issues) {
        const t = iss.startMs / 1000;
        if (prevTime < t && t <= now) {
          video.currentTime = t;
          video.pause();
          setActive(iss.id);
          const peakSrc = findFrame(iss.id, 'peak');
          const corrSrc = findFrame(iss.id, 'correction');
          if (peakSrc) showOverlay(peakSrc, corrSrc, true);  // true = start 10s timer
          prevTime = t;
          return;
        }
      }
    }
    prevTime = now;
  });

  tlWrap.addEventListener('click', e => {
    if (e.target.classList.contains('tl-marker')) return;
    const rect = tlWrap.getBoundingClientRect();
    const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    video.currentTime = pct * data.source.durationMs / 1000;
  });
}

function esc(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

init().catch(err => {
  document.getElementById('tl-label').textContent = 'Error: ' + err.message;
});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

def _make_handler(output_dir: str, video_path: str, root_dir: str):
    class ViewerHandler(http.server.SimpleHTTPRequestHandler):
        _output_dir = output_dir
        _video_path = video_path

        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=root_dir, **kwargs)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._serve_html()
            elif self.path == "/api/analysis":
                self._serve_json()
            elif self.path == "/video":
                self._serve_video()
            else:
                super().do_GET()

        def do_HEAD(self):
            if self.path == "/video":
                path = self._video_path
                try:
                    file_size = os.path.getsize(path)
                except OSError:
                    self.send_error(404, "Video not found")
                    return
                self.send_response(200)
                self.send_header("Content-Type", self.guess_type(path))
                self.send_header("Content-Length", str(file_size))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
            else:
                super().do_HEAD()

        def _serve_video(self):
            path = self._video_path
            try:
                file_size = os.path.getsize(path)
            except OSError:
                self.send_error(404, "Video not found")
                return

            ctype = self.guess_type(path)
            range_header = self.headers.get("Range", "")

            if range_header.startswith("bytes="):
                try:
                    spec = range_header[6:].split(",")[0].strip()
                    s, e = spec.split("-")
                    start = int(s) if s else 0
                    end = int(e) if e else file_size - 1
                except ValueError:
                    self.send_error(400, "Bad Range header")
                    return
                end = min(end, file_size - 1)
                length = end - start + 1
                self.send_response(206)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(length))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                with open(path, "rb") as fh:
                    fh.seek(start)
                    remaining = length
                    while remaining > 0:
                        chunk = fh.read(min(65536, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            else:
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(file_size))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                with open(path, "rb") as fh:
                    while True:
                        chunk = fh.read(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)

        def _serve_html(self):
            body = _HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_json(self):
            path = os.path.join(self._output_dir, "analysis.json")
            try:
                with open(path, "rb") as f:
                    body = f.read()
            except OSError as exc:
                self.send_error(404, str(exc))
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):  # noqa: A002
            pass  # suppress per-request logs; errors still go to stderr

    return ViewerHandler


def serve(
    output_dir: str = "outputs",
    port: int = 8742,
    open_browser: bool = True,
) -> None:
    analysis_path = os.path.join(output_dir, "analysis.json")
    with open(analysis_path) as f:
        analysis = json.load(f)

    video_path: str = analysis["source"]["videoPath"]
    if not os.path.isabs(video_path):
        video_path = os.path.join(os.getcwd(), video_path)

    handler_cls = _make_handler(
        output_dir=os.path.abspath(output_dir),
        video_path=os.path.abspath(video_path),
        root_dir=os.getcwd(),
    )

    with http.server.ThreadingHTTPServer(("", port), handler_cls) as httpd:
        url = f"http://localhost:{port}"
        print(f"Viewer: {url}  (Ctrl+C to stop)")
        if open_browser:
            threading.Timer(0.5, webbrowser.open, args=[url]).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nViewer stopped.")
