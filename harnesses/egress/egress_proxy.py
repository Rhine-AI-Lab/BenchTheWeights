"""Egress proxy for the solver's container: HTTP CONNECT to an allowlist of host:port, nothing else.

The solver's container is only on the task's internal network. This proxy is the one service on both that
network and a network with internet access. It relays a TLS connection only when a client asks for an
allowed `host:port` with CONNECT, as HTTPS_PROXY clients do. Plain HTTP requests and every other destination
get 403. Every request, allowed or refused, is one JSON line on stdout and in /var/log/egress/egress.jsonl,
which Harbor collects with the trial.

    EGRESS_ALLOW=api.anthropic.com:443 EGRESS_PORT=3128 python egress_proxy.py
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

ALLOW = {h.strip().lower() for h in os.environ.get("EGRESS_ALLOW", "api.anthropic.com:443").split(",") if h.strip()}
PORT = int(os.environ.get("EGRESS_PORT", "3128"))
LOG = Path(os.environ.get("EGRESS_LOG", "/var/log/egress/egress.jsonl"))
DENY = b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
BAD_GATEWAY = b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"


def log(**kw) -> None:
    line = json.dumps({"t": round(time.time(), 3), **kw})
    print(line, flush=True)
    try:
        with LOG.open("a") as f:
            f.write(line + "\n")
    except OSError:
        pass


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while chunk := await reader.read(65536):
            writer.write(chunk)
            await writer.drain()
    except (ConnectionError, asyncio.IncompleteReadError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def refuse(writer: asyncio.StreamWriter, reply: bytes) -> None:
    try:
        writer.write(reply)
        await writer.drain()
    finally:
        writer.close()


async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    peer = str(writer.get_extra_info("peername"))
    try:
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 30)
    except (asyncio.TimeoutError, asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError):
        writer.close()
        return
    request = head.split(b"\r\n", 1)[0].decode("latin-1")
    parts = request.split()
    if len(parts) != 3 or parts[0].upper() != "CONNECT":
        log(action="refused", reason="not a CONNECT request", request=request[:200], peer=peer)
        return await refuse(writer, DENY)
    target = parts[1].lower()
    if target not in ALLOW:
        log(action="refused", target=target, peer=peer)
        return await refuse(writer, DENY)
    host, port = target.rsplit(":", 1)
    try:
        up_r, up_w = await asyncio.wait_for(asyncio.open_connection(host, int(port)), 30)
    except (OSError, asyncio.TimeoutError) as exc:
        log(action="error", target=target, error=str(exc)[:200], peer=peer)
        return await refuse(writer, BAD_GATEWAY)
    log(action="allowed", target=target, peer=peer)
    writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
    await writer.drain()
    await asyncio.gather(pipe(reader, up_w), pipe(up_r, writer))


async def main() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    server = await asyncio.start_server(handle, "0.0.0.0", PORT)
    log(action="ready", port=PORT, allow=sorted(ALLOW))
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
