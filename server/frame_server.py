#!/usr/bin/env python3
"""
Lightweight frame server for stream.html
Serves only screenshot frames, separate from main game server
"""

import os
import sys
import time
import json
import base64
import threading
from pathlib import Path
from PIL import Image
import numpy as np
import argparse

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from fastapi import FastAPI, Response
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError:
    print("❌ FastAPI not available. Install with: pip install fastapi uvicorn")
    sys.exit(1)

app = FastAPI(title="Pokemon Frame Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
current_frame = None
frame_lock = threading.Lock()
frame_counter = 0
last_update = time.time()

# Frame cache for shared memory communication
# Use cache directory instead of /tmp
CACHE_DIR = ".pokeagent_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
FRAME_CACHE_FILE = os.path.join(CACHE_DIR, "frame_cache.json")
FRAME_UPDATE_INTERVAL = 0.025  # 40 FPS

def load_frame_from_cache():
    """Load the latest frame from shared cache file"""
    global current_frame, frame_counter, last_update
    
    try:
        if os.path.exists(FRAME_CACHE_FILE):
            with open(FRAME_CACHE_FILE, 'r') as f:
                data = json.load(f)
                
            # Check if frame is newer
            cache_counter = data.get('frame_counter', 0)
            if cache_counter > frame_counter:
                with frame_lock:
                    current_frame = data.get('frame_data')
                    frame_counter = cache_counter
                    last_update = time.time()
                    
    except Exception as e:
        pass  # Silently handle cache read errors

def frame_updater():
    """Background thread to periodically check for new frames"""
    while True:
        try:
            load_frame_from_cache()
            time.sleep(FRAME_UPDATE_INTERVAL)
        except Exception:
            time.sleep(0.1)

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "ok", "server": "frame_server"}

@app.get("/frame")
async def get_frame():
    """Get the current game frame"""
    global current_frame, frame_counter, last_update
    
    try:
        load_frame_from_cache()  # Try to get latest frame
        
        with frame_lock:
            if current_frame:
                # Frame is already base64 encoded
                return Response(
                    content=json.dumps({
                        "frame": current_frame,
                        "frame_count": frame_counter,
                        "timestamp": last_update,
                        "status": "ok"
                    }),
                    media_type="application/json"
                )
            else:
                # No frame available
                return Response(
                    content=json.dumps({
                        "frame": None,
                        "frame_count": 0,
                        "timestamp": time.time(),
                        "status": "no_frame"
                    }),
                    media_type="application/json"
                )
                
    except Exception as e:
        return Response(
            content=json.dumps({
                "frame": None,
                "error": str(e),
                "status": "error"
            }),
            media_type="application/json",
            status_code=500
        )

@app.get("/status")
async def get_status():
    """Get frame server status"""
    return {
        "frame_count": frame_counter,
        "last_update": last_update,
        "cache_file": FRAME_CACHE_FILE,
        "cache_exists": os.path.exists(FRAME_CACHE_FILE)
    }

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Pokemon Frame Server")
    parser.add_argument("--port", type=int, default=8001, help="Port to run on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    args = parser.parse_args()
    
    print(f"🖼️ Starting Pokemon Frame Server on {args.host}:{args.port}")
    print(f"📁 Frame cache: {FRAME_CACHE_FILE}")
    
    # Start background frame updater
    frame_thread = threading.Thread(target=frame_updater, daemon=True)
    frame_thread.start()
    
    # Start server
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")