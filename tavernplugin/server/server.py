"""TavernPlugin bridge server: TavernGym behind a small localhost HTTP API.

    python tavernplugin/server/server.py [--port 8765] [--host 127.0.0.1] [-v]

Endpoints (all JSON, POST bodies are the snapshot form documented in
``tavernplugin/README.md``):

    GET  /health    is the server up, is TavernGym built, which game version
    POST /combat    win / tie / loss probabilities for the two boards
    POST /align     compare the logged boards with the boards TavernGym rebuilds
    POST /advise    rank candidate placements of the current warband
    POST /step      apply one action to a rebuilt state (verification loop)

Only the Python standard library is used, so the server starts even when the
Rust extension is missing: ``/health`` then reports ``"taverngym": false`` and
the other endpoints answer 503 with the reason.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bridge  # noqa: E402

VERSION = "0.1.0"
STARTED = time.time()
VERBOSE = False


def watch_parent(pid, httpd):
    """Exit when the game process that started us is gone.

    The plugin launches this server as a child of Hearthstone; without this a
    crash would leave python running forever. On Windows the wait is exact
    (OpenProcess + WaitForSingleObject), elsewhere it degrades to polling.
    """
    try:
        if sys.platform == "win32":
            import ctypes  # noqa: PLC0415

            SYNCHRONIZE = 0x00100000
            handle = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, int(pid))
            if handle:
                ctypes.windll.kernel32.WaitForSingleObject(handle, 0xFFFFFFFF)
                print(f"parent {pid} exited, shutting down", flush=True)
                httpd.shutdown()
                return
    except Exception as exc:  # pragma: no cover - defensive
        print(f"parent watch unavailable ({exc}); polling", flush=True)

    while True:
        time.sleep(2)
        try:
            os.kill(int(pid), 0)
        except OSError:
            print(f"parent {pid} exited, shutting down", flush=True)
            httpd.shutdown()
            return
        except Exception:
            return



def health():
    info = {"ok": True, "service": "tavernplugin", "version": VERSION, "uptime_s": int(time.time() - STARTED)}
    try:
        core = bridge.load_core()
        info.update(
            taverngym=True,
            game_version=core.game_version(),
            server_rules=core.server_rules_version(),
            cards=core.N_CARDS,
            actions=core.N_ACTIONS,
        )
    except bridge.TavernGymUnavailable as exc:
        info.update(taverngym=False, error=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        info.update(taverngym=False, error=f"{type(exc).__name__}: {exc}")
    return info


ROUTES = {
    "/health": lambda body: health(),
    "/combat": lambda body: bridge.combat(body, n=int(body.get("n", 512))),
    "/align": lambda body: bridge.align(body),
    "/advise": lambda body: bridge.advise(body, n=int(body.get("n", 128))),
}


class Handler(BaseHTTPRequestHandler):
    server_version = f"TavernPlugin/{VERSION}"

    def log_message(self, fmt, *args):  # noqa: A003 - stdlib signature
        if VERBOSE:
            sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def _send(self, status, payload):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802 - stdlib signature
        parsed = urlparse(self.path)
        handler = ROUTES.get(parsed.path)
        if handler is None:
            self._send(404, {"ok": False, "error": f"no route {parsed.path}"})
            return
        # the in-game client falls back to GET when HttpWebRequest's POST path is
        # unavailable (stripped mscorlib), carrying the snapshot in ?body=
        body = {}
        query = parse_qs(parsed.query)
        if "body" in query:
            try:
                body = json.loads(query["body"][0])
            except json.JSONDecodeError as exc:
                self._send(400, {"ok": False, "error": f"bad body parameter: {exc}"})
                return
        self._dispatch(handler, body)

    def do_POST(self):  # noqa: N802 - stdlib signature
        path = urlparse(self.path).path
        handler = ROUTES.get(path)
        if handler is None:
            self._send(404, {"ok": False, "error": f"no route {path}"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError as exc:
            self._send(400, {"ok": False, "error": f"bad json: {exc}"})
            return
        self._dispatch(handler, body)

    def _dispatch(self, handler, body):
        started = time.time()
        try:
            payload = handler(body)
            status = 200
        except bridge.TavernGymUnavailable as exc:
            payload, status = {"ok": False, "error": f"TavernGym is not built: {exc}"}, 503
        except Exception as exc:
            payload, status = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 500
            if VERBOSE:
                traceback.print_exc()
        payload.setdefault("elapsed_ms", int((time.time() - started) * 1000))
        self._send(status, payload)


def main(argv=None):
    global VERBOSE
    ap = argparse.ArgumentParser(description="TavernPlugin bridge server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--parent-pid", type=int, default=0,
                    help="exit when this process id disappears (the game sets it)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    VERBOSE = args.verbose

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    info = health()
    print(f"tavernplugin server {VERSION} on http://{args.host}:{args.port}", flush=True)
    print("  taverngym:", "ready" if info.get("taverngym") else f"NOT AVAILABLE ({info.get('error')})", flush=True)
    if info.get("taverngym"):
        print(f"  game {info['game_version']}  cards {info['cards']}  actions {info['actions']}", flush=True)
    print("  ctrl-c to stop", flush=True)
    if args.parent_pid:
        threading.Thread(target=watch_parent, args=(args.parent_pid, httpd), daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

