#!/usr/bin/env python3
"""
Enhanced main entry point for I20 XES Tools with comprehensive logging,
crash detection, memory monitoring, and production-ready error handling.

Features:
- Timestamped log files with rotation
- Memory usage monitoring (when psutil available)
- Qt exception handling
- Signal handlers for crashes
- Automatic crash report generation
- Environment information logging
- Graceful shutdown handling
"""

import os
import sys
import json
import time
import signal
import traceback
import warnings
import faulthandler
import threading
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import platform
import gc

# Optional imports with graceful fallbacks
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

try:
    import resource
    RESOURCE_AVAILABLE = True
except ImportError:
    resource = None
    RESOURCE_AVAILABLE = False

# Enable fault handler for segfaults
faulthandler.enable(all_threads=True)

# Global variables for tracking
APP_START_TIME = time.time()
MEMORY_MONITOR_THREAD = None
# LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR = Path(os.environ.get("LOG_DIR", "/tmp/xestools_logs"))
# os.makedirs(LOG_DIR, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

CRASH_REPORT_DIR = LOG_DIR / "crash_reports"
LOGGER = None


class MemoryMonitor:
    """Monitor memory usage in a separate thread."""
    
    def __init__(self, interval: int = 60, threshold_mb: int = 1024):
        self.interval = interval
        self.threshold_mb = threshold_mb
        self.running = False
        self.thread = None
        self.peak_memory = 0
        
        if PSUTIL_AVAILABLE and psutil:
            self.process = psutil.Process()
        else:
            self.process = None
        
    def start(self):
        """Start monitoring memory usage."""
        if not PSUTIL_AVAILABLE:
            if LOGGER:
                LOGGER.warning("Memory monitoring disabled: psutil not available")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        if LOGGER:
            LOGGER.info(f"Memory monitoring started (interval={self.interval}s, threshold={self.threshold_mb}MB)")
        
    def stop(self):
        """Stop monitoring memory usage."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        if LOGGER:
            LOGGER.info(f"Memory monitoring stopped. Peak memory usage: {self.peak_memory:.1f} MB")
        
    def _monitor_loop(self):
        """Main monitoring loop."""
        if not self.process:
            return
            
        while self.running:
            try:
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                
                if memory_mb > self.peak_memory:
                    self.peak_memory = memory_mb
                
                if memory_mb > self.threshold_mb and LOGGER:
                    LOGGER.warning(f"High memory usage: {memory_mb:.1f} MB (threshold: {self.threshold_mb} MB)")
                    
                    # Log additional memory details
                    try:
                        gc.collect()  # Force garbage collection
                        if PSUTIL_AVAILABLE and psutil:
                            virtual_mem = psutil.virtual_memory()
                            LOGGER.warning(f"System memory: {virtual_mem.percent}% used, "
                                         f"{virtual_mem.available / (1024**3):.1f} GB available")
                    except Exception as e:
                        if LOGGER:
                            LOGGER.error(f"Failed to get detailed memory info: {e}")
                        
                time.sleep(self.interval)
                
            except Exception as e:
                if LOGGER:
                    LOGGER.error(f"Memory monitoring error: {e}")
                time.sleep(self.interval)


class CrashReporter:
    """Generate comprehensive crash reports with system information."""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Collect system information for crash reports."""
        try:
            # Basic system info
            info = {
                "timestamp": datetime.now().isoformat(),
                "python_version": sys.version,
                "platform": platform.platform(),
                "architecture": platform.architecture(),
                "processor": platform.processor(),
                "hostname": platform.node(),
            }
            
            # Process info (if psutil available)
            if PSUTIL_AVAILABLE and psutil:
                try:
                    process = psutil.Process()
                    info["process"] = {
                        "pid": process.pid,
                        "memory_mb": process.memory_info().rss / (1024 * 1024),
                        "cpu_percent": process.cpu_percent(),
                        "num_threads": process.num_threads(),
                        "create_time": datetime.fromtimestamp(process.create_time()).isoformat(),
                    }
                except Exception as e:
                    info["process_error"] = str(e)
            else:
                info["process"] = "psutil not available"
                
            # System resources (if psutil available)
            if PSUTIL_AVAILABLE and psutil:
                try:
                    info["system"] = {
                        "cpu_count": psutil.cpu_count(),
                        "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                        "memory_percent": psutil.virtual_memory().percent,
                        "disk_usage_percent": psutil.disk_usage('/').percent,
                    }
                except Exception as e:
                    info["system_error"] = str(e)
            else:
                info["system"] = "psutil not available"
                
            # Resource usage (if resource module available)
            if RESOURCE_AVAILABLE and resource:
                try:
                    usage = resource.getrusage(resource.RUSAGE_SELF)
                    info["resource_usage"] = {
                        "user_time": usage.ru_utime,
                        "system_time": usage.ru_stime,
                        "max_rss": usage.ru_maxrss,
                        "page_faults": usage.ru_majflt,
                    }
                except Exception as e:
                    info["resource_error"] = str(e)
                
            # Environment variables (relevant ones)
            env_vars = {k: v for k, v in os.environ.items() 
                       if any(keyword in k.upper() for keyword in 
                             ['PYTHON', 'QT', 'DISPLAY', 'HOME', 'PATH'])}
            info["environment"] = env_vars
            
            # Loaded modules (first 50 to avoid huge reports)
            info["loaded_modules"] = sorted(list(sys.modules.keys()))[:50]
            
            return info
            
        except Exception as e:
            return {"error": f"Failed to collect system info: {e}"}
    
    @staticmethod
    def generate_crash_report(exception_info: Optional[tuple] = None, 
                            signal_info: Optional[Dict[str, Any]] = None) -> Path:
        """Generate a comprehensive crash report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        crash_file = CRASH_REPORT_DIR / f"crash_report_{timestamp}.json"
        
        # Ensure crash report directory exists
        CRASH_REPORT_DIR.mkdir(parents=True, exist_ok=True)
        
        report = {
            "crash_info": {
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": time.time() - APP_START_TIME,
            },
            "system_info": CrashReporter.get_system_info(),
        }
        
        # Add exception information
        if exception_info:
            exc_type, exc_value, exc_tb = exception_info
            report["exception"] = {
                "type": exc_type.__name__ if exc_type else "Unknown",
                "message": str(exc_value) if exc_value else "No message",
                "traceback": traceback.format_exception(exc_type, exc_value, exc_tb) if exc_tb else []
            }
            
        # Add signal information
        if signal_info:
            report["signal"] = signal_info
            
        # Add memory information (if available)
        try:
            if MEMORY_MONITOR_THREAD and PSUTIL_AVAILABLE and psutil:
                report["memory"] = {
                    "peak_memory_mb": MEMORY_MONITOR_THREAD.peak_memory,
                    "current_memory_mb": psutil.Process().memory_info().rss / (1024 * 1024)
                }
        except Exception:
            pass
            
        # Write crash report
        try:
            with open(crash_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            if LOGGER:
                LOGGER.error(f"Crash report saved: {crash_file}")
            else:
                print(f"ERROR: Crash report saved: {crash_file}", file=sys.stderr)
            return crash_file
        except Exception as e:
            if LOGGER:
                LOGGER.error(f"Failed to write crash report: {e}")
            else:
                print(f"ERROR: Failed to write crash report: {e}", file=sys.stderr)
            return crash_file


def setup_logging() -> logging.Logger:
    """Setup comprehensive logging with file rotation and console output."""
    # Create logs directory
    LOG_DIR.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("xestools")
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"xestools_{timestamp}.log"
    
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)-20s [%(filename)s:%(lineno)d] %(message)s"
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    # Error file handler (errors only)
    error_file = LOG_DIR / f"xestools_errors_{timestamp}.log"
    error_handler = logging.handlers.RotatingFileHandler(
        error_file, maxBytes=5*1024*1024, backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_format)
    logger.addHandler(error_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    logger.info(f"Error log file: {error_file}")
    
    return logger


def qt_exception_handler(exc_type, exc_value, exc_tb):
    """Handle Qt exceptions and generate crash reports."""
    if LOGGER:
        LOGGER.error("Qt exception occurred:", exc_info=(exc_type, exc_value, exc_tb))
    else:
        print(f"ERROR: Qt exception: {exc_type.__name__}: {exc_value}", file=sys.stderr)
    
    # Generate crash report
    CrashReporter.generate_crash_report((exc_type, exc_value, exc_tb))
    
    # Call the default exception handler
    sys.__excepthook__(exc_type, exc_value, exc_tb)


def signal_handler(signum, frame):
    """Handle system signals and generate crash reports."""
    signal_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else f"Signal{signum}"
    if LOGGER:
        LOGGER.error(f"Received signal {signum} ({signal_name})")
    else:
        print(f"ERROR: Received signal {signum} ({signal_name})", file=sys.stderr)
    
    signal_info = {
        "signal_number": signum,
        "signal_name": signal_name,
        "frame_info": {
            "filename": frame.f_code.co_filename if frame else "Unknown",
            "function": frame.f_code.co_name if frame else "Unknown", 
            "line_number": frame.f_lineno if frame else "Unknown",
        }
    }
    
    # Generate crash report
    CrashReporter.generate_crash_report(signal_info=signal_info)
    
    # Graceful shutdown
    graceful_shutdown(exit_code=128 + signum)


def graceful_shutdown(exit_code: int = 0):
    """Perform graceful shutdown with cleanup."""
    if LOGGER:
        LOGGER.info("Initiating graceful shutdown...")
    else:
        print("INFO: Initiating graceful shutdown...", file=sys.stderr)
    
    try:
        # Stop memory monitoring
        if MEMORY_MONITOR_THREAD:
            MEMORY_MONITOR_THREAD.stop()
            
        # Log final statistics
        uptime = time.time() - APP_START_TIME
        if LOGGER:
            LOGGER.info(f"Application uptime: {uptime:.1f} seconds")
        
        # Qt cleanup
        try:
            # Try importing Qt modules for cleanup
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance()
            if app:
                if LOGGER:
                    LOGGER.info("Closing Qt application...")
                app.quit()
        except ImportError:
            if LOGGER:
                LOGGER.warning("Qt modules not available for cleanup")
        except Exception as e:
            if LOGGER:
                LOGGER.error(f"Error during Qt cleanup: {e}")
            
    except Exception as e:
        if LOGGER:
            LOGGER.error(f"Error during graceful shutdown: {e}")
        else:
            print(f"ERROR: Error during graceful shutdown: {e}", file=sys.stderr)
    finally:
        if LOGGER:
            LOGGER.info("Shutdown complete")
        sys.exit(exit_code)


def setup_signal_handlers():
    """Setup signal handlers for crash detection."""
    # Handle common crash signals
    crash_signals = [signal.SIGTERM, signal.SIGINT]
    
    # Add SIGHUP and SIGQUIT on Unix systems
    if hasattr(signal, 'SIGHUP'):
        crash_signals.append(signal.SIGHUP)
    if hasattr(signal, 'SIGQUIT'):
        crash_signals.append(signal.SIGQUIT)
        
    for sig in crash_signals:
        try:
            signal.signal(sig, signal_handler)
            if LOGGER:
                LOGGER.debug(f"Registered handler for signal {sig}")
        except (OSError, ValueError) as e:
            if LOGGER:
                LOGGER.warning(f"Could not register handler for signal {sig}: {e}")


def log_startup_info():
    """Log comprehensive startup information."""
    if not LOGGER:
        return
        
    LOGGER.info("=" * 60)
    LOGGER.info("XES Tools Enhanced Startup")
    LOGGER.info("=" * 60)
    
    # System information
    LOGGER.info(f"Python version: {sys.version}")
    LOGGER.info(f"Platform: {platform.platform()}")
    LOGGER.info(f"Architecture: {platform.architecture()[0]}")
    LOGGER.info(f"Processor: {platform.processor()}")
    LOGGER.info(f"Working directory: {os.getcwd()}")
    
    # Memory information (if psutil available)
    if PSUTIL_AVAILABLE and psutil:
        try:
            memory = psutil.virtual_memory()
            LOGGER.info(f"System memory: {memory.total / (1024**3):.1f} GB total, "
                       f"{memory.percent}% used")
        except Exception as e:
            LOGGER.warning(f"Could not get memory info: {e}")
    else:
        LOGGER.info("Memory info: psutil not available")
        
    # Environment
    LOGGER.info(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    LOGGER.info(f"Qt platform: {os.environ.get('QT_QPA_PLATFORM', 'default')}")
    
    # Check for required modules
    required_modules = ['PySide6', 'numpy', 'h5py', 'matplotlib', 'lmfit']
    for module in required_modules:
        try:
            __import__(module)
            LOGGER.info(f"✓ {module} available")
        except ImportError:
            LOGGER.warning(f"✗ {module} not available")
            
    # Optional modules
    optional_modules = [('psutil', PSUTIL_AVAILABLE), ('resource', RESOURCE_AVAILABLE)]
    for module, available in optional_modules:
        if available:
            LOGGER.info(f"✓ {module} available (optional)")
        else:
            LOGGER.info(f"○ {module} not available (optional)")
            
    LOGGER.info("=" * 60)


def main():
    """Enhanced main entry point with comprehensive error handling."""
    global LOGGER, MEMORY_MONITOR_THREAD
    
    try:
        # Setup logging first
        LOGGER = setup_logging()
        
        # Log startup information
        log_startup_info()
        
        # Setup signal handlers
        setup_signal_handlers()
        
        # Setup Qt exception handling
        sys.excepthook = qt_exception_handler
        
        # Configure warnings
        if sys.flags.dev_mode == 0:
            os.environ.setdefault("PYTHONWARNINGS", "default")
        warnings.simplefilter("default")
        
        # Start memory monitoring
        MEMORY_MONITOR_THREAD = MemoryMonitor(interval=30, threshold_mb=512)
        MEMORY_MONITOR_THREAD.start()
        
        LOGGER.info("Importing Qt modules...")
        try:
            from PySide6.QtWidgets import QApplication
            from PySide6.QtCore import qInstallMessageHandler, QtMsgType
        except ImportError as e:
            LOGGER.error(f"Failed to import Qt modules: {e}")
            LOGGER.error("Please install PySide6: pip install PySide6")
            sys.exit(1)
        
        # Qt message handler
        def qt_message_handler(msg_type, context, message):
            if not LOGGER:
                return
                
            if msg_type == QtMsgType.QtDebugMsg:
                LOGGER.debug(f"Qt: {message}")
            elif msg_type == QtMsgType.QtInfoMsg:
                LOGGER.info(f"Qt: {message}")
            elif msg_type == QtMsgType.QtWarningMsg:
                LOGGER.warning(f"Qt: {message}")
            elif msg_type == QtMsgType.QtCriticalMsg:
                LOGGER.error(f"Qt Critical: {message}")
            elif msg_type == QtMsgType.QtFatalMsg:
                LOGGER.critical(f"Qt Fatal: {message}")
                
        qInstallMessageHandler(qt_message_handler)
        
        # Reduce matplotlib logging noise
        try:
            logging.getLogger("matplotlib").setLevel(logging.INFO)
        except:
            pass
        
        LOGGER.info("Creating Qt application...")
        app = QApplication(sys.argv)
        
        # Set application properties
        app.setApplicationName("XES Tools")
        app.setApplicationVersion("1.0")
        app.setOrganizationName("Diamond Light Source")
        
        LOGGER.info("Importing main window...")
        try:
            from i20_xes.main_gui import MainWindow
        except ImportError as e:
            LOGGER.error(f"Failed to import MainWindow: {e}")
            LOGGER.error("Check that the i20_xes package is properly installed")
            sys.exit(1)
        
        LOGGER.info("Creating main window...")
        window = MainWindow()
        
        LOGGER.info("Showing main window...")
        window.show()
        
        LOGGER.info("Starting Qt event loop...")
        exit_code = app.exec()
        
        LOGGER.info(f"Qt application exited with code: {exit_code}")
        graceful_shutdown(exit_code)
        
    except ImportError as e:
        error_msg = f"Missing required module: {e}"
        if LOGGER:
            LOGGER.error(error_msg)
        else:
            print(f"ERROR: {error_msg}", file=sys.stderr)
        
        # Try to generate crash report even without full logging
        try:
            CrashReporter.generate_crash_report(sys.exc_info())
        except:
            pass
            
        sys.exit(1)
        
    except Exception as e:
        error_msg = f"Fatal error during startup: {e}"
        if LOGGER:
            LOGGER.critical(error_msg, exc_info=True)
        else:
            print(f"CRITICAL: {error_msg}", file=sys.stderr)
            traceback.print_exc()
        
        # Generate crash report
        try:
            CrashReporter.generate_crash_report(sys.exc_info())
        except:
            pass
            
        sys.exit(1)


if __name__ == "__main__":
    main()