#!/usr/bin/env python3
"""
Error handling and recovery utilities for the Pokemon agent.
"""

import os
import sys
import time
import signal
import traceback
import subprocess
from typing import Optional, Callable


class ErrorHandler:
    """Manages error recovery and graceful shutdowns"""
    
    def __init__(self, max_consecutive_errors=3, recovery_delay=2.0):
        """
        Initialize error handler.
        
        Args:
            max_consecutive_errors: Max errors before triggering recovery
            recovery_delay: Delay in seconds before retrying after errors
        """
        self.consecutive_errors = 0
        self.max_consecutive_errors = max_consecutive_errors
        self.recovery_delay = recovery_delay
        self.total_errors = 0
        self.shutdown_requested = False
        
        # Callbacks
        self.on_shutdown = None
        self.on_recovery = None
        
        # Install signal handlers
        self._install_signal_handlers()
    
    def _install_signal_handlers(self):
        """Install signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n🛑 Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
        
        if self.on_shutdown:
            try:
                self.on_shutdown()
            except:
                pass
        
        # Give a moment for cleanup
        time.sleep(0.5)
        sys.exit(0)
    
    def handle_error(self, error: Exception, context: str = "") -> bool:
        """
        Handle an error with potential recovery.
        
        Args:
            error: The exception that occurred
            context: Additional context about where the error occurred
        
        Returns:
            bool: True if recovery should be attempted, False if fatal
        """
        self.consecutive_errors += 1
        self.total_errors += 1
        
        # Log the error
        print(f"❌ Error in {context}: {str(error)}")
        if self.consecutive_errors > 1:
            print(f"   (Error {self.consecutive_errors}/{self.max_consecutive_errors})")
        
        # Check if we should attempt recovery
        if self.consecutive_errors >= self.max_consecutive_errors:
            print(f"⚠️ Too many consecutive errors, initiating recovery...")
            return self.attempt_recovery()
        
        # Simple delay before continuing
        time.sleep(self.recovery_delay)
        return True
    
    def attempt_recovery(self) -> bool:
        """
        Attempt to recover from errors.
        
        Returns:
            bool: True if recovery successful, False otherwise
        """
        try:
            print("🔄 Attempting recovery...")
            
            # Call recovery callback if set
            if self.on_recovery:
                success = self.on_recovery()
                if success:
                    self.reset_error_counter()
                    print("✅ Recovery successful")
                    return True
            
            # Default recovery: just reset and continue
            self.reset_error_counter()
            time.sleep(self.recovery_delay * 2)  # Longer delay for recovery
            return True
            
        except Exception as e:
            print(f"❌ Recovery failed: {e}")
            return False
    
    def reset_error_counter(self):
        """Reset the consecutive error counter"""
        if self.consecutive_errors > 0:
            print(f"✅ Resetting error counter (was {self.consecutive_errors})")
        self.consecutive_errors = 0
    
    def track_success(self):
        """Track a successful operation to reset error counter"""
        if self.consecutive_errors > 0:
            self.consecutive_errors = 0
    
    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested"""
        return self.shutdown_requested
    
    def get_error_stats(self) -> dict:
        """Get error statistics"""
        return {
            'consecutive_errors': self.consecutive_errors,
            'total_errors': self.total_errors,
            'max_consecutive': self.max_consecutive_errors
        }


class ServerRestartHandler:
    """Handles server process restart on failures"""
    
    def __init__(self, restart_threshold=5):
        """
        Initialize server restart handler.
        
        Args:
            restart_threshold: Number of failures before restart
        """
        self.restart_threshold = restart_threshold
        self.failure_count = 0
        self.server_process = None
        self.server_cmd = []
        self.server_env = {}
    
    def set_server_process(self, process: subprocess.Popen, cmd: list, env: dict = None):
        """
        Set the server process to manage.
        
        Args:
            process: The subprocess.Popen instance
            cmd: Command used to start the server
            env: Environment variables for the server
        """
        self.server_process = process
        self.server_cmd = cmd
        self.server_env = env or {}
    
    def check_and_restart(self) -> bool:
        """
        Check if restart is needed and perform it.
        
        Returns:
            bool: True if restarted successfully
        """
        self.failure_count += 1
        
        if self.failure_count >= self.restart_threshold:
            return self.restart_server()
        
        return False
    
    def restart_server(self) -> bool:
        """
        Restart the server process.
        
        Returns:
            bool: True if restarted successfully
        """
        if not self.server_cmd:
            print("❌ No server command configured for restart")
            return False
        
        try:
            print("🔄 Restarting server process...")
            
            # Kill existing process
            if self.server_process:
                try:
                    self.server_process.terminate()
                    time.sleep(1)
                    if self.server_process.poll() is None:
                        self.server_process.kill()
                except:
                    pass
            
            # Start new process
            self.server_process = subprocess.Popen(
                self.server_cmd,
                env={**os.environ, **self.server_env},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            print(f"✅ Server restarted with PID {self.server_process.pid}")
            
            # Give it time to initialize
            time.sleep(3)
            
            # Reset failure count
            self.failure_count = 0
            return True
            
        except Exception as e:
            print(f"❌ Failed to restart server: {e}")
            return False
    
    def reset_failure_count(self):
        """Reset the failure counter"""
        self.failure_count = 0


# Global error handler instance
_error_handler = None


def get_error_handler() -> ErrorHandler:
    """Get or create the global error handler"""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


def handle_agent_error(error: Exception, context: str = "") -> bool:
    """
    Handle an agent error using the global error handler.
    
    Args:
        error: The exception that occurred
        context: Context about where the error occurred
    
    Returns:
        bool: True if recovery should be attempted
    """
    handler = get_error_handler()
    return handler.handle_error(error, context)


def reset_error_counter():
    """Reset the global error counter"""
    handler = get_error_handler()
    handler.reset_error_counter()


def install_shutdown_handler(callback: Optional[Callable] = None):
    """
    Install a shutdown handler.
    
    Args:
        callback: Optional callback to run on shutdown
    """
    handler = get_error_handler()
    if callback:
        handler.on_shutdown = callback