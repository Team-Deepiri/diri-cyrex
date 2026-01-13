#!/usr/bin/env python3
"""
Cyrex Custom File Watcher
Intelligent file watcher that only reloads on actual code changes,
ignoring cache files, logs, and other non-code files.
"""
import os
import sys
import time
import subprocess
import signal
from pathlib import Path
from typing import Set, Optional
import logging

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("cyrex_watcher")

# Try to use watchdog if available, fallback to polling
# IMPORTANT: Define FileSystemEventHandler BEFORE it's used in class definition
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent
    HAS_WATCHDOG = True
    logger.info("watchdog library available")
except ImportError:
    HAS_WATCHDOG = False
    # Create dummy classes for when watchdog is not available
    # These MUST be defined before CyrexFileHandler class
    class FileSystemEventHandler:
        """Dummy base class when watchdog is not available"""
        pass
    
    class FileSystemEvent:
        """Dummy event class when watchdog is not available"""
        def __init__(self, src_path=""):
            self.src_path = src_path
            self.is_directory = False
    
    Observer = None
    logger.warning("watchdog not available, file watching will be disabled")


# Define handler class - works with or without watchdog
class CyrexFileHandler(FileSystemEventHandler):
    """Handler for file system events with intelligent filtering"""
    
    # Patterns to ignore (won't trigger reload)
    IGNORE_PATTERNS = {
        '__pycache__',
        '*.pyc',
        '*.pyo',
        '*.pyd',
        '*.log',
        '.git',
        '.cache',
        '.pytest_cache',
        '.mypy_cache',
        '.ruff_cache',
        '*.swp',
        '*.swo',
        '*~',
        '.DS_Store',
        'node_modules',
        'dist',
        'build',
        '*.egg-info',
        '.venv',
        'venv',
        '*.tmp',
        '*.bak',
    }
    
    # File extensions that should trigger reload
    CODE_EXTENSIONS = {'.py', '.pyx', '.pyi'}
    
    def __init__(self, reload_callback, debounce_seconds: float = 0.5):
        super().__init__()
        self.reload_callback = reload_callback
        self.debounce_seconds = debounce_seconds
        self.last_reload_time = 0
        self.pending_reload = False
    
    def should_ignore(self, file_path: str) -> bool:
        """Check if file should be ignored"""
        path = Path(file_path)
        
        # Check if any part of the path matches ignore patterns
        for part in path.parts:
            if any(
                part.startswith(ignore.replace('*', '')) or 
                part.endswith(ignore.replace('*', '')) or
                ignore.replace('*', '') in part
                for ignore in self.IGNORE_PATTERNS
                if '*' not in ignore
            ):
                return True
            
            # Check wildcard patterns
            for pattern in self.IGNORE_PATTERNS:
                if '*' in pattern:
                    if part.endswith(pattern.replace('*', '')):
                        return True
        
        # Check file extension
        if path.suffix and path.suffix not in self.CODE_EXTENSIONS:
            return True
        
        return False
    
    def on_modified(self, event: FileSystemEvent):
        """Handle file modification events"""
        if event.is_directory:
            return
        
        file_path = event.src_path
        
        if self.should_ignore(file_path):
            logger.debug(f"Ignoring change: {file_path}")
            return
        
        logger.info(f"Code change detected: {file_path}")
        logger.info("Scheduling application reload...")
        self._schedule_reload()
    
    def on_created(self, event: FileSystemEvent):
        """Handle file creation events"""
        if event.is_directory:
            return
        
        file_path = event.src_path
        
        if self.should_ignore(file_path):
            return
        
        # Only reload if it's a code file
        if Path(file_path).suffix in self.CODE_EXTENSIONS:
            logger.info(f"New code file created: {file_path}")
            self._schedule_reload()
    
    def _schedule_reload(self):
        """Schedule a reload with debouncing"""
        current_time = time.time()
        
        # Debounce: only reload if enough time has passed since last reload
        if current_time - self.last_reload_time < self.debounce_seconds:
            self.pending_reload = True
            return
        
        self._trigger_reload()
    
    def _trigger_reload(self):
        """Trigger the reload callback"""
        self.last_reload_time = time.time()
        self.pending_reload = False
        logger.info("=" * 60)
        logger.info("TRIGGERING APPLICATION RELOAD")
        logger.info("=" * 60)
        self.reload_callback()


class CyrexWatcher:
    """Custom file watcher for Cyrex development"""
    
    def __init__(
        self,
        watch_dir: str = "/app/app",
        app_module: str = "app.main:app",
        host: str = "0.0.0.0",
        port: int = 8000,
        log_level: str = "info"
    ):
        self.watch_dir = Path(watch_dir)
        self.app_module = app_module
        self.host = host
        self.port = port
        self.log_level = log_level
        self.process: Optional[subprocess.Popen] = None
        self.observer = None  # Type will be Observer if watchdog available, None otherwise
        self.running = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def _reload_app(self):
        """Reload the application by restarting the uvicorn process"""
        if self.process:
            logger.info("Stopping current process...")
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Process didn't terminate, killing...")
                self.process.kill()
                self.process.wait()
            except Exception as e:
                logger.error(f"Error stopping process: {e}")
        
        # Start new process
        self._start_app()
    
    def _start_app(self):
        """Start the uvicorn application"""
        cmd = [
            sys.executable, "-m", "uvicorn",
            self.app_module,
            "--host", self.host,
            "--port", str(self.port),
            "--log-level", self.log_level,
            # Don't use uvicorn's reload - we handle it ourselves
        ]
        
        logger.info(f"Starting application: {' '.join(cmd)}")
        # Don't capture output - let it go to stdout/stderr directly
        # This prevents buffering issues and allows uvicorn's output to display properly
        self.process = subprocess.Popen(
            cmd,
            # stdout and stderr go to parent process (docker logs)
        )
    
    def start(self):
        """Start watching and running the application"""
        if not self.watch_dir.exists():
            logger.error(f"Watch directory does not exist: {self.watch_dir}")
            sys.exit(1)
        
        logger.info(f"Starting Cyrex watcher")
        logger.info(f"Watching directory: {self.watch_dir}")
        logger.info(f"Application: {self.app_module}")
        
        # Set running flag before starting watcher
        self.running = True
        
        # Start the application
        self._start_app()
        
        # Start file watcher
        if HAS_WATCHDOG:
            self._start_watchdog()
        else:
            logger.warning("watchdog not available, file watching disabled")
            logger.info("Application running without file watching")
            # Just keep the process running
            try:
                if self.process:
                    self.process.wait()
            except KeyboardInterrupt:
                self.stop()
    
    def _start_watchdog(self):
        """Start watchdog-based file watching"""
        if not HAS_WATCHDOG:
            logger.error("Cannot start watchdog - library not available")
            return
        
        try:
            event_handler = CyrexFileHandler(self._reload_app, debounce_seconds=0.5)
            # Use PollingObserver for better compatibility with Docker on Windows
            # Regular Observer uses inotify which may not work in Docker volumes
            try:
                from watchdog.observers.polling import PollingObserver
                self.observer = PollingObserver(timeout=1)
                logger.info("Using PollingObserver (better for Docker/Windows)")
            except ImportError:
                self.observer = Observer()
                logger.info("Using standard Observer")
            
            self.observer.schedule(event_handler, str(self.watch_dir), recursive=True)
            self.observer.start()
            
            logger.info("File watcher started successfully")
            logger.info(f"Watching for changes in: {self.watch_dir}")
            logger.info("File watcher is active - changes to .py files will trigger reload")
            
            # Keep the watcher running and monitor the application process
            try:
                while self.running:
                    # Check if the process is still alive
                    if self.process and self.process.poll() is not None:
                        logger.warning("Application process died, restarting...")
                        self._start_app()
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, shutting down...")
                self.stop()
        except Exception as e:
            logger.error(f"Failed to start file watcher: {e}", exc_info=True)
            logger.info("Application will run without file watching")
            # If watchdog fails, still keep the process running
            try:
                if self.process:
                    self.process.wait()
            except KeyboardInterrupt:
                self.stop()
    
    def stop(self):
        """Stop watching and the application"""
        self.running = False
        
        if self.observer:
            self.observer.stop()
            self.observer.join()
            logger.info("File watcher stopped")
        
        if self.process:
            logger.info("Stopping application...")
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            except Exception as e:
                logger.error(f"Error stopping process: {e}")
        
        logger.info("Cyrex watcher stopped")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cyrex Custom File Watcher")
    parser.add_argument(
        "--watch-dir",
        default=os.getenv("WATCH_DIR", "/app/app"),
        help="Directory to watch for changes (default: /app/app)"
    )
    parser.add_argument(
        "--app-module",
        default=os.getenv("APP_MODULE", "app.main:app"),
        help="Application module (default: app.main:app)"
    )
    parser.add_argument(
        "--host",
        default=os.getenv("HOST", "0.0.0.0"),
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8000")),
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "info"),
        help="Log level (default: info)"
    )
    
    args = parser.parse_args()
    
    watcher = CyrexWatcher(
        watch_dir=args.watch_dir,
        app_module=args.app_module,
        host=args.host,
        port=args.port,
        log_level=args.log_level
    )
    
    watcher.start()


if __name__ == "__main__":
    main()

