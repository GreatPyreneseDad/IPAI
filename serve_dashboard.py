#!/usr/bin/env python3
"""
Simple Dashboard Server

Basic HTTP server to serve the IPAI dashboard without complex dependencies.
"""

import http.server
import socketserver
import os
import webbrowser
from pathlib import Path

# Change to dashboard directory
dashboard_dir = Path(__file__).parent / "dashboard"
os.chdir(dashboard_dir)

PORT = 8090

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()

    def do_GET(self):
        if self.path == '/':
            self.path = '/index.html'
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

if __name__ == "__main__":
    handler = MyHTTPRequestHandler
    
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"🎛️  IPAI Dashboard Server running at:")
        print(f"🌐 http://localhost:{PORT}")
        print(f"📱 http://127.0.0.1:{PORT}")
        print(f"⏹️  Press Ctrl+C to stop")
        print()
        
        # Try to open browser
        try:
            webbrowser.open(f'http://localhost:{PORT}')
            print("🚀 Opening dashboard in your default browser...")
        except:
            print("💡 Please manually open the URL in your browser")
        
        httpd.serve_forever()