#!/usr/bin/env python3
"""Prompt-cache relay for Codex on OpenRouter (make_task.py --provider openrouter).

OpenRouter caches Claude prompts on its Responses API only when a request has a top-level `cache_control`,
and Codex cannot add request fields. Without it every Claude request pays the full input price; Codex trials
read millions of input tokens, almost all of them a repeated prefix. The relay sits on the task's internal
network between Codex and the egress proxy: Codex's base URL is http://relay:8080/api/v1, and the relay
forwards each request to https://openrouter.ai through the egress proxy and streams the response back.

The only change it makes: a POST to .../responses whose JSON body names a model starting with
RELAY_CACHE_MODELS (default "anthropic/") and has no cache_control gets `"cache_control":{"type":"ephemeral"}`
appended at the top level. Nothing the model sees changes.

WebSocket upgrades get 501, so Codex uses HTTPS from the start. The log (/var/log/relay/relay.jsonl) has one
line per request: time, method, path, model, whether cache_control was added, status, sizes and seconds.
Headers, which hold the API key, and bodies are never logged.
"""
import http.client
import http.server
import json
import os
import ssl
import threading
import time

UPSTREAM = os.environ.get("RELAY_UPSTREAM", "openrouter.ai")
PROXY = os.environ.get("RELAY_PROXY", "egress:3128")  # host:port of the egress proxy; empty = connect directly
PORT = int(os.environ.get("RELAY_PORT", "8080"))
PREFIXES = tuple(p for p in os.environ.get("RELAY_CACHE_MODELS", "anthropic/").split(",") if p)
LOG = os.environ.get("RELAY_LOG", "/var/log/relay/relay.jsonl")
TIMEOUT = float(os.environ.get("RELAY_TIMEOUT", "900"))
CACHE = b'"cache_control":{"type":"ephemeral"}'
HOP = {"connection", "keep-alive", "proxy-connection", "proxy-authorization", "te", "trailer", "transfer-encoding",
       "upgrade", "host", "content-length"}
_lock = threading.Lock()
_local = threading.local()
_ctx = ssl.create_default_context()


def log(rec):
    with _lock, open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


def read_body(rfile, headers):
    n = headers.get("Content-Length")
    if n is not None:
        return rfile.read(int(n))
    if "chunked" in (headers.get("Transfer-Encoding") or "").lower():
        out = bytearray()
        while True:
            size = int(rfile.readline().split(b";")[0].strip() or b"0", 16)
            if size == 0:
                while rfile.readline() not in (b"\r\n", b"\n", b""):
                    pass
                return bytes(out)
            out += rfile.read(size)
            rfile.readline()
    return b""


def add_cache_control(body):
    """(new body, model, added): append cache_control to a Responses request for a matching model."""
    try:
        d = json.loads(body)
    except ValueError:
        return body, None, False
    if not isinstance(d, dict):
        return body, None, False
    model = d.get("model")
    if not (isinstance(model, str) and model.startswith(PREFIXES)) or "cache_control" in d:
        return body, model, False
    s = body.rstrip()
    if not s.endswith(b"}"):
        return body, model, False
    head = s[:-1].rstrip()
    new = head + (b"" if head.endswith(b"{") else b",") + CACHE + b"}"
    try:
        ok = json.loads(new).get("cache_control") == {"type": "ephemeral"}
    except ValueError:
        ok = False
    return (new, model, True) if ok else (body, model, False)


def connect():
    if PROXY:
        host, port = PROXY.rsplit(":", 1)
        c = http.client.HTTPSConnection(host, int(port), context=_ctx, timeout=TIMEOUT)
        c.set_tunnel(UPSTREAM, 443)
    else:
        c = http.client.HTTPSConnection(UPSTREAM, 443, context=_ctx, timeout=TIMEOUT)
    return c


class Relay(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.relay()

    def do_POST(self):
        self.relay()

    def do_DELETE(self):
        self.relay()

    def plain(self, status, text, rec, close=False):
        b = text.encode()
        self.send_response_only(status)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(b)))
        if close:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        self.wfile.write(b)
        rec["status"] = status
        if rec.get("path") != "/relay-health":
            log(rec)

    def relay(self):
        t0 = time.time()
        path = self.path.split("?")[0]
        rec = {"t": round(t0, 3), "method": self.command, "path": path}
        if path == "/relay-health":
            return self.plain(200, "ok", rec)
        if (self.headers.get("Upgrade") or "").lower() == "websocket":
            rec["websocket"] = True
            return self.plain(501, "websocket not relayed; use HTTPS", rec, close=True)
        if not self.path.startswith("/api/"):
            return self.plain(404, "not relayed", rec)
        body = read_body(self.rfile, self.headers)
        rec["req_bytes"] = len(body)
        added = False
        if self.headers.get("Content-Encoding"):
            rec["req_encoding"] = self.headers.get("Content-Encoding")
        elif self.command == "POST" and path.endswith("/responses") and body:
            body, rec["model"], added = add_cache_control(body)
        rec["cache_control_added"] = added
        hdrs = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        if body or self.command == "POST":
            hdrs["Content-Length"] = str(len(body))
        resp = None
        for attempt in (0, 1):  # a kept-alive upstream connection may have been closed by the far end; retry once
            c = getattr(_local, "conn", None)
            if c is None:
                c = _local.conn = connect()
            try:
                c.request(self.command, self.path, body=body or None, headers=hdrs)
                resp = c.getresponse()
                break
            except (OSError, http.client.HTTPException) as e:
                c.close()
                _local.conn = None
                if attempt:
                    rec["error"] = type(e).__name__
                    rec["secs"] = round(time.time() - t0, 3)
                    return self.plain(502, f"relay: upstream {type(e).__name__}", rec)
        self.send_response_only(resp.status, resp.reason)
        for k, v in resp.getheaders():
            if k.lower() not in HOP:
                self.send_header(k, v)
        length = resp.getheader("Content-Length")
        chunked = length is None and self.command != "HEAD" and resp.status not in (204, 304)
        if chunked:
            self.send_header("Transfer-Encoding", "chunked")
        elif length is not None:
            self.send_header("Content-Length", length)
        self.end_headers()
        n = 0
        try:
            while True:
                b = resp.read1(65536)
                if not b:
                    break
                n += len(b)
                self.wfile.write(b"%x\r\n%s\r\n" % (len(b), b) if chunked else b)
            if chunked:
                self.wfile.write(b"0\r\n\r\n")
        except OSError as e:  # Codex went away mid-stream: drop the upstream connection
            rec["error"] = "client " + type(e).__name__
            c.close()
            _local.conn = None
            self.close_connection = True
        if resp.will_close:
            c.close()
            _local.conn = None
        rec.update(status=resp.status, resp_bytes=n, secs=round(time.time() - t0, 3))
        log(rec)


def main():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    srv = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Relay)
    srv.daemon_threads = True
    log({"t": round(time.time(), 3), "start": True, "upstream": UPSTREAM, "proxy": PROXY or None, "cache_models": PREFIXES})
    srv.serve_forever()


if __name__ == "__main__":
    main()
