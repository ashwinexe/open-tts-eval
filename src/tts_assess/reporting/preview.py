from __future__ import annotations

import base64
import hashlib
import http.server
import os
import socket
import threading
import time
import webbrowser
from pathlib import Path


def serve_report(html_path: Path, port: int = 8042, open_browser: bool = True) -> None:
    html_path = html_path.resolve()
    ws_port = port + 1
    inject = _inject_script(ws_port)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/", f"/{html_path.name}"):
                content = html_path.read_text(encoding="utf-8").replace(
                    "</body>", inject + "</body>"
                )
                data = content.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                super().do_GET()

        def log_message(self, fmt, *args):
            pass

    os.chdir(html_path.parent)
    threading.Thread(target=_ws_server, args=(html_path, ws_port), daemon=True).start()
    server = http.server.HTTPServer(("localhost", port), Handler)
    url = f"http://localhost:{port}/"
    print(f"Serving {html_path.name} on {url}")
    print("Watching for changes... (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


def _inject_script(ws_port: int) -> str:
    return f"""<script>
(function(){{
  var ws = new WebSocket("ws://localhost:{ws_port}");
  ws.onmessage = function(e) {{
    if (e.data === "reload") {{
      var y = window.scrollY;
      sessionStorage.setItem("_tts_assess_scroll", y);
      location.reload();
    }}
  }};
  window.addEventListener("load", function() {{
    var y = sessionStorage.getItem("_tts_assess_scroll");
    if (y) {{ window.scrollTo(0, parseInt(y)); sessionStorage.removeItem("_tts_assess_scroll"); }}
  }});
}})();
</script>"""


def _ws_server(html_path: Path, ws_port: int) -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("localhost", ws_port))
    srv.listen(5)
    clients = []

    def accept_loop() -> None:
        while True:
            conn, _ = srv.accept()
            data = conn.recv(4096).decode()
            key = ""
            for line in data.split("\r\n"):
                if line.startswith("Sec-WebSocket-Key:"):
                    key = line.split(": ", 1)[1].strip()
            accept = base64.b64encode(
                hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-5AB5DC11650A").encode()).digest()
            ).decode()
            conn.send(
                (
                    "HTTP/1.1 101 Switching Protocols\r\n"
                    "Upgrade: websocket\r\n"
                    "Connection: Upgrade\r\n"
                    f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
                ).encode()
            )
            clients.append(conn)

    threading.Thread(target=accept_loop, daemon=True).start()
    last = hashlib.md5(html_path.read_bytes()).hexdigest()
    while True:
        time.sleep(0.5)
        try:
            cur = hashlib.md5(html_path.read_bytes()).hexdigest()
        except FileNotFoundError:
            continue
        if cur == last:
            continue
        last = cur
        frame = b"\x81" + bytes([len(b"reload")]) + b"reload"
        dead = []
        for client in clients:
            try:
                client.send(frame)
            except Exception:
                dead.append(client)
        for client in dead:
            clients.remove(client)
        print(f"reloaded ({time.strftime('%H:%M:%S')})")
