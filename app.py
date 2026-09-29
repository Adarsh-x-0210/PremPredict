"""
Premier League Transfer Value Predictor - Local Web Application
Starts a lightweight local HTTP server and opens your web browser.
"""

import http.server
import socketserver
import webbrowser
import os
import sys

PORT = 8080
DIRECTORY = os.path.dirname(os.path.abspath(__file__))


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)


def start_server():
    os.chdir(DIRECTORY)
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        url = f"http://localhost:{PORT}/index.html"
        print("=" * 60)
        print(f"  PREMIER LEAGUE TRANSFER VALUE PREDICTOR (WEB APP) ")
        print("=" * 60)
        print(f"\n[INFO] Serving Web App at: {url}")
        print("[INFO] Opening your default web browser...")
        print("[INFO] Press Ctrl+C in this terminal to stop the server.\n")
        
        webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[INFO] Server stopped gracefully.")


if __name__ == "__main__":
    start_server()
