#!/usr/bin/env python3
"""
Get local IP address for network interface display
"""

import socket

def get_local_ip():
    """Get the local IP address of this machine"""
    try:
        # Connect to a remote address to determine local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # Use Google DNS as target (doesn't actually connect)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            return local_ip
    except Exception:
        # Fallback to localhost
        return "127.0.0.1"

if __name__ == "__main__":
    print(f"Local IP: {get_local_ip()}")