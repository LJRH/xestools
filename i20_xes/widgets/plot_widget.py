import os, gc, logging, traceback, weakref
from typing import List, Tuple, Optional
import numpy as np
from PySide6 import QtCore
from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
try:
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
except ImportError:
    from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.artist import Artist
import matplotlib.pyplot as plt

from i20_xes.modules.dataset import DataSet

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)

# Create console handler if not exists
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(name)s:%(lineno)d - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

_LINE_COLORS = ["#ffffff", "#ffcc00", "#00ccff"]
_BAND_ALPHA = 0.15

class PlotWidget(QWidget):
    """Enhanced PlotWidget with comprehensive error handling and logging."""
    
    lines_changed = QtCore.Signal()

    def __init__(self, parent=None):
        """Initialize the enhanced plot widget with comprehensive error handling."""
        logger.debug("Initializing PlotWidget")
        
        try:
            super().__init__(parent)
            
            # Store parent as weak reference to avoid circular references
            self._parent_ref = weakref.ref(parent) if parent else None
            
            # Initialize matplotlib components with error handling
            logger.debug("Creating matplotlib figure")
            self.figure = Figure(constrained_layout=True, facecolor='white')
            
            logger.debug("Creating canvas")
            self.canvas = FigureCanvas(self.figure)
            
            logger.debug("Creating toolbar")
            self.toolbar = NavigationToolbar(self.canvas, self)

            # Create subplots with error handling
            logger.debug("Creating subplots")
            gs = self.figure.add_gridspec(nrows=2, ncols=1, height_ratios=[4, 1])
            self.ax_img = self.figure.add_subplot(gs[0, 0])
            self.ax_prof = self.figure.add_subplot(gs[1, 0])

            # Initialize state variables
            self._cbar = None
            self._lines: List[Line2D] = []
            self._bands: List[Artist] = []  # keep handles to band artists (axvspan/axhspan)
            self._drag_idx = None
            self._line_orientation = "vertical"
            self._bandwidth = 2.0
            self._suppress_emit = False
            self._dataset: Optional[DataSet] = None
            
            # Track matplotlib event connections for cleanup
            self._event_connections = []
            
            # Flag to track if widget is being destroyed
            self._is_closing = False

            # Create layout with error handling
            logger.debug("Setting up layout")
            layout = QVBoxLayout(self)
            layout.addWidget(self.toolbar)
            layout.addWidget(self.canvas)

            # Connect matplotlib events with error handling
            logger.debug("Connecting matplotlib events")
            self._connect_events()
            
            logger.debug("PlotWidget initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize PlotWidget: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _connect_events(self):
        """Connect matplotlib events with error handling and tracking."""
        try:
            if self.canvas and not self._is_closing:
                logger.debug("Connecting matplotlib events")
                
                # Connect events and store connection IDs
                cid1 = self.canvas.mpl_connect("button_press_event", self._safe_on_press)
                cid2 = self.canvas.mpl_connect("button_release_event", self._safe_on_release)
                cid3 = self.canvas.mpl_connect("motion_notify_event", self._safe_on_motion)
                
                self._event_connections.extend([cid1, cid2, cid3])
                logger.debug(f"Connected {len(self._event_connections)} matplotlib events")
                
        except Exception as e:
            logger.error(f"Failed to connect matplotlib events: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    def _disconnect_events(self):
        """Disconnect all matplotlib events safely."""
        logger.debug("Disconnecting matplotlib events")
        
        try:
            if self.canvas and self._event_connections:
                for cid in self._event_connections:
                    try:
                        self.canvas.mpl_disconnect(cid)
                        logger.debug(f"Disconnected event {cid}")
                    except Exception as e:
                        logger.warning(f"Failed to disconnect event {cid}: {e}")
                
                self._event_connections.clear()
                logger.debug("All matplotlib events disconnected")
                
        except Exception as e:
            logger.error(f"Error during event disconnection: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    def closeEvent(self, event):
        """Handle widget closure with proper cleanup."""
        logger.debug("PlotWidget closeEvent triggered")
        
        try:
            self._is_closing = True
            
            # Disconnect matplotlib events first
            self._disconnect_events()
            
            # Clear all plot data
            self._safe_clear_all()
            
            # Force garbage collection
            self._force_cleanup()
            
            logger.debug("PlotWidget cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during closeEvent: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        finally:
            # Always accept the close event
            if hasattr(event, 'accept'):
                event.accept()
            
            # Call parent closeEvent
            try:
                super().closeEvent(event)
            except Exception as e:
                logger.warning(f"Error calling parent closeEvent: {e}")

    def _force_cleanup(self):
        """Force cleanup of matplotlib objects and memory."""
        logger.debug("Forcing cleanup of matplotlib objects")
        
        try:
            # Clear figure completely
            if hasattr(self, 'figure') and self.figure:
                logger.debug("Clearing matplotlib figure")
                try:
                    self.figure.clear()
                    plt.close(self.figure)
                except Exception as e:
                    logger.warning(f"Error closing matplotlib figure: {e}")
            
            # Force garbage collection
            logger.debug("Running garbage collection")
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error during force cleanup: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    # ------------ Event wrapper methods with error handling ------------
    
    def _safe_on_press(self, event):
        """Safely handle button press events."""
        try:
            if not self._is_closing:
                self._on_press(event)
        except Exception as e:
            logger.error(f"Error in _on_press: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    def _safe_on_release(self, event):
        """Safely handle button release events."""
        try:
            if not self._is_closing:
                self._on_release(event)
        except Exception as e:
            logger.error(f"Error in _on_release: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    def _safe_on_motion(self, event):
        """Safely handle motion events."""
        try:
            if not self._is_closing:
                self._on_motion(event)
        except Exception as e:
            logger.error(f"Error in _on_motion: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    # ------------ helpers to suppress signals ------------
    
    def set_signal_suppressed(self, state: bool):
        """Set signal suppression state."""
        logger.debug(f"Setting signal suppressed: {state}")
        self._suppress_emit = bool(state)

    def _emit_lines_changed(self):
        """Emit lines changed signal if not suppressed."""
        try:
            if not self._suppress_emit and not self._is_closing:
                logger.debug("Emitting lines_changed signal")
                self.lines_changed.emit()
        except Exception as e:
            logger.error(f"Error emitting lines_changed signal: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    # ------------ core clear/plot with enhanced error handling ------------
    
    def _safe_clear_all(self):
        """Safely clear all plot elements."""
        logger.debug("Starting safe clear all")
        
        try:
            # Remove colorbar first with enhanced error handling
            self._safe_remove_colorbar()
            
            # Remove bands with enhanced error handling
            self._safe_remove_bands()
            
            # Clear lines with enhanced error handling
            self._safe_clear_lines()
            
            # Clear axes safely
            self._safe_clear_axes()
            
            # Reset state
            self._reset_state()
            
            logger.debug("Safe clear all completed")
            
        except Exception as e:
            logger.error(f"Error during safe clear all: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    def clear(self):
        """Enhanced clear method with comprehensive error handling."""
        logger.debug("Starting plot clear")
        
        try:
            if self._is_closing:
                logger.debug("Widget is closing, skipping clear")
                return
            
            self._safe_clear_all()
            
        except Exception as e:
            logger.error(f"Error in clear method: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    def _safe_remove_colorbar(self):
        """Safely remove colorbar with comprehensive error handling."""
        if self._cbar is not None:
            logger.debug("Removing colorbar")
            try:
                # Try multiple removal methods
                if hasattr(self._cbar, 'remove'):
                    self._cbar.remove()
                elif hasattr(self._cbar, 'ax') and hasattr(self._cbar.ax, 'remove'):
                    self._cbar.ax.remove()
                
                logger.debug("Colorbar removed successfully")
                
            except Exception as e:
                logger.warning(f"Error removing colorbar (expected): {e}")
                
                # Try alternative removal methods
                try:
                    if hasattr(self._cbar, 'ax'):
                        if hasattr(self._cbar.ax, 'clear'):
                            self._cbar.ax.clear()
                        if hasattr(self._cbar.ax, 'set_visible'):
                            self._cbar.ax.set_visible(False)
                except Exception as e2:
                    logger.warning(f"Alternative colorbar removal failed: {e2}")
            
            finally:
                self._cbar = None
                logger.debug("Colorbar reference cleared")

    def _safe_remove_bands(self):
        """Enhanced band removal with better error handling."""
        if not self._bands:
            return
            
        logger.debug(f"Removing {len(self._bands)} band artists")
        
        # Create a copy of the list to iterate over
        bands_to_remove = list(self._bands)
        
        for i, band in enumerate(bands_to_remove):
            try:
                if band is not None:
                    logger.debug(f"Removing band {i}")
                    
                    # Check if the artist is still valid
                    if hasattr(band, 'axes') and band.axes is not None:
                        band.remove()
                        logger.debug(f"Band {i} removed successfully")
                    else:
                        logger.debug(f"Band {i} already detached from axes")
                        
            except Exception as e:
                logger.warning(f"Error removing band {i}: {e}")
                
                # Try alternative removal methods
                try:
                    if hasattr(band, 'set_visible'):
                        band.set_visible(False)
                        logger.debug(f"Band {i} hidden instead of removed")
                except Exception as e2:
                    logger.warning(f"Failed to hide band {i}: {e2}")
        
        # Clear the bands list
        self._bands.clear()
        logger.debug("All bands removed and list cleared")

    def _safe_clear_lines(self):
        """Safely clear all lines."""
        if not self._lines:
            return
            
        logger.debug(f"Clearing {len(self._lines)} lines")
        
        # Create a copy of the list to iterate over
        lines_to_remove = list(self._lines)
        
        for i, line in enumerate(lines_to_remove):
            try:
                if line is not None and hasattr(line, 'remove'):
                    logger.debug(f"Removing line {i}")
                    line.remove()
                    logger.debug(f"Line {i} removed successfully")
            except Exception as e:
                logger.warning(f"Error removing line {i}: {e}")
        
        # Clear the lines list
        self._lines.clear()
        logger.debug("All lines cleared")

    def _safe_clear_axes(self):
        """Safely clear axes."""
        logger.debug("Clearing axes")
        
        try:
            if hasattr(self, 'ax_img') and self.ax_img is not None:
                logger.debug("Clearing image axis")
                self.ax_img.cla()
                
            if hasattr(self, 'ax_prof') and self.ax_prof is not None:
                logger.debug("Clearing profile axis")
                self.ax_prof.cla()
                
            logger.debug("Axes cleared successfully")
            
        except Exception as e:
            logger.error(f"Error clearing axes: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    def _reset_state(self):
        """Reset internal state variables."""
        logger.debug("Resetting internal state")
        
        try:
            self._lines = []
            self._bands = []
            self._drag_idx = None
            self._cbar = None
            
            logger.debug("Internal state reset complete")
            
        except Exception as e:
            logger.error(f"Error resetting state: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    def plot(self, dataset: Optional[DataSet]):
        """Enhanced plot method with comprehensive error handling."""
        logger.debug(f"Starting plot with dataset: {type(dataset)}")
        
        try:
            if self._is_closing:
                logger.debug("Widget is closing, skipping plot")
                return
            
            self._dataset = dataset
            self.clear()

            if dataset is None:
                logger.debug("Dataset is None, showing 'No data loaded'")
                self._plot_no_data()
                return

            logger.debug(f"Plotting dataset: kind={dataset.kind}")
            
            if dataset.kind == "1D":
                self._plot_1d_dataset(dataset)
            elif dataset.kind == "2D":
                self._plot_2d_dataset(dataset)
            else:
                logger.warning(f"Unknown dataset kind: {dataset.kind}")
                self._plot_no_data()

            # Safe canvas draw
            self._safe_canvas_draw()
            
            logger.debug("Plot completed successfully")
            
        except Exception as e:
            logger.error(f"Error in plot method: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Attempt to show error state
            try:
                self._plot_error_state(str(e))
            except Exception as e2:
                logger.error(f"Failed to show error state: {e2}")

    def _plot_no_data(self):
        """Plot no data state."""
        try:
            if hasattr(self, 'ax_img') and self.ax_img is not None:
                self.ax_img.set_title("No data loaded")
                
            if hasattr(self, 'ax_prof') and self.ax_prof is not None:
                self.ax_prof.set_visible(True)  # keep ROI subplot visible for future RXES
                
        except Exception as e:
            logger.error(f"Error in _plot_no_data: {e}")

    def _plot_error_state(self, error_msg: str):
        """Plot error state."""
        try:
            if hasattr(self, 'ax_img') and self.ax_img is not None:
                self.ax_img.set_title(f"Error: {error_msg[:50]}...")
                
        except Exception as e:
            logger.error(f"Error in _plot_error_state: {e}")

    def _plot_1d_dataset(self, dataset: DataSet):
        """Plot 1D dataset with error handling."""
        logger.debug("Plotting 1D dataset")
        
        try:
            # Hide profile axis for simple 1D
            if hasattr(self, 'ax_prof') and self.ax_prof is not None:
                self.ax_prof.set_visible(False)
            
            # Prepare data with validation
            x = dataset.x if dataset.x is not None else np.arange(len(dataset.y))
            y = dataset.y
            
            if x is None or y is None:
                raise ValueError("Invalid x or y data for 1D plot")
            
            # Validate data shapes
            if len(x) != len(y):
                logger.warning(f"X and Y data length mismatch: {len(x)} vs {len(y)}")
                min_len = min(len(x), len(y))
                x = x[:min_len]
                y = y[:min_len]
            
            # Plot with error handling
            if hasattr(self, 'ax_img') and self.ax_img is not None:
                self.ax_img.plot(x, y, lw=1.5)
                self.ax_img.set_xlabel(dataset.xlabel or "Energy")
                self.ax_img.set_ylabel(dataset.ylabel or "Intensity")
                
                title = os.path.basename(dataset.source) if dataset.source else "Data"
                self.ax_img.set_title(title)
            
            logger.debug("1D dataset plotted successfully")
            
        except Exception as e:
            logger.error(f"Error plotting 1D dataset: {e}")
            raise

    def _plot_2d_dataset(self, dataset: DataSet):
        """Plot 2D dataset with error handling."""
        logger.debug("Plotting 2D dataset")
        
        try:
            # Show profile axis for RXES
            if hasattr(self, 'ax_prof') and self.ax_prof is not None:
                self.ax_prof.set_visible(True)

            Z = dataset.z
            if Z is None:
                raise ValueError("No Z data for 2D plot")
            
            logger.debug(f"2D data shape: {Z.shape}")
            
            # Try pcolormesh first if we have 2D coordinate arrays
            if dataset.x2d is not None and dataset.y2d is not None:
                logger.debug("Using pcolormesh for 2D plot")
                self._plot_pcolormesh(dataset, Z)
            else:
                logger.debug("Using imshow for 2D plot")
                self._plot_imshow(dataset, Z)

            # Set up colorbar
            if self._cbar is not None and hasattr(self._cbar, 'set_label'):
                self._cbar.set_label(dataset.zlabel or "Intensity")
            
            # Set labels and title
            if hasattr(self, 'ax_img') and self.ax_img is not None:
                self.ax_img.set_xlabel(dataset.xlabel or "X")
                self.ax_img.set_ylabel(dataset.ylabel or "Y")
                
                title = os.path.basename(dataset.source) if dataset.source else "Data"
                self.ax_img.set_title(title)

            # Prepare profile axis
            self._setup_profile_axis()

            # Ensure at least one ROI line for RXES, and draw initial bands
            if len(self._lines) == 0:
                self._create_initial_line()
            self._update_bands()
            
            logger.debug("2D dataset plotted successfully")
            
        except Exception as e:
            logger.error(f"Error plotting 2D dataset: {e}")
            raise

    def _plot_pcolormesh(self, dataset: DataSet, Z: np.ndarray):
        """Plot using pcolormesh with error handling."""
        try:
            if hasattr(self, 'ax_img') and self.ax_img is not None:
                pc = self.ax_img.pcolormesh(dataset.x2d, dataset.y2d, Z, shading="auto", cmap="viridis")
                self._cbar = self.figure.colorbar(pc, ax=self.ax_img)
                logger.debug("Pcolormesh plot created successfully")
                
        except Exception as e:
            logger.error(f"Error creating pcolormesh: {e}")
            raise

    def _plot_imshow(self, dataset: DataSet, Z: np.ndarray):
        """Plot using imshow with error handling."""
        try:
            extent = None
            if dataset.x is not None and dataset.y is not None:
                try:
                    if np.all(np.diff(dataset.x) > 0) and np.all(np.diff(dataset.y) > 0):
                        extent = [float(dataset.x.min()), float(dataset.x.max()),
                                 float(dataset.y.min()), float(dataset.y.max())]
                        logger.debug(f"Calculated extent: {extent}")
                except Exception as e:
                    logger.warning(f"Error calculating extent: {e}")
                    extent = None
            
            if hasattr(self, 'ax_img') and self.ax_img is not None:
                im = self.ax_img.imshow(Z, aspect='auto', origin='lower', extent=extent, cmap='viridis')
                self._cbar = self.figure.colorbar(im, ax=self.ax_img)
                logger.debug("Imshow plot created successfully")
                
        except Exception as e:
            logger.error(f"Error creating imshow: {e}")
            raise

    def _setup_profile_axis(self):
        """Set up the profile axis for 2D plots."""
        try:
            if hasattr(self, 'ax_prof') and self.ax_prof is not None:
                self.ax_prof.set_ylabel("Integrated counts")
                self.ax_prof.grid(True, alpha=0.3)
                logger.debug("Profile axis set up successfully")
                
        except Exception as e:
            logger.error(f"Error setting up profile axis: {e}")

    def _safe_canvas_draw(self):
        """Safely draw the canvas."""
        try:
            if hasattr(self, 'canvas') and self.canvas is not None and not self._is_closing:
                logger.debug("Drawing canvas")
                self.canvas.draw_idle()
                logger.debug("Canvas drawn successfully")
                
        except Exception as e:
            logger.error(f"Error drawing canvas: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    # ------------ XES overlays (hide ROI subplot) with enhanced error handling ------------
    
    def plot_xes_bundle(self, curves, avg=None, title="XES bundle"):
        """Plot XES bundle with enhanced error handling."""
        logger.debug(f"Plotting XES bundle with {len(curves) if curves else 0} curves")
        
        try:
            if self._is_closing:
                logger.debug("Widget is closing, skipping XES bundle plot")
                return
            
            self.clear()
            
            if hasattr(self, 'ax_prof') and self.ax_prof is not None:
                self.ax_prof.set_visible(False)
                self.ax_prof.cla()

            # Plot curves with error handling
            for i, c in enumerate(curves or []):
                try:
                    x = c.get("x")
                    y = c.get("y")
                    if x is None or y is None:
                        logger.warning(f"Curve {i} has invalid data, skipping")
                        continue
                    
                    if hasattr(self, 'ax_img') and self.ax_img is not None:
                        self.ax_img.plot(x, y, lw=1.0, alpha=c.get("alpha", 0.7),
                                       color=c.get("color", None), label=c.get("label", ""))
                    
                except Exception as e:
                    logger.warning(f"Error plotting curve {i}: {e}")

            # Plot average with error handling
            if avg is not None and avg.get("x") is not None:
                try:
                    if hasattr(self, 'ax_img') and self.ax_img is not None:
                        self.ax_img.plot(avg["x"], avg["y"], lw=2.2, color="black", 
                                       label=avg.get("label", "Average"))
                except Exception as e:
                    logger.warning(f"Error plotting average: {e}")

            # Set labels and title
            if hasattr(self, 'ax_img') and self.ax_img is not None:
                self.ax_img.set_xlabel("ω (eV)")
                self.ax_img.set_ylabel("Intensity (XES)")
                self.ax_img.set_title(title)
                
                # Add legend with error handling
                try:
                    if curves and len(curves) <= 12:
                        self.ax_img.legend(loc="best", fontsize=9)
                except Exception as e:
                    logger.warning(f"Error adding legend: {e}")

            self._safe_canvas_draw()
            logger.debug("XES bundle plotted successfully")
            
        except Exception as e:
            logger.error(f"Error in plot_xes_bundle: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    # ------------ ROI line/band API with enhanced error handling ------------
    
    def set_line_orientation(self, orientation: str):
        """Set line orientation with enhanced error handling."""
        logger.debug(f"Setting line orientation to: {orientation}")
        
        try:
            assert orientation in ("vertical", "horizontal")
            
            # If orientation unchanged, just refresh bands and notify
            if orientation == self._line_orientation:
                if len(self._lines) == 0:
                    self._create_initial_line()
                self._update_bands()
                self._emit_lines_changed()
                return

            # Orientation changed: rebuild lines in the new orientation
            old_n = max(1, len(self._lines))
            
            # Remove all existing lines safely
            self._safe_clear_lines()

            self._line_orientation = orientation
            
            # Create same number of lines in the new orientation at sensible positions
            self._create_initial_line()
            while len(self._lines) < old_n:
                self._add_line_internal()

            self._update_bands()
            self._safe_canvas_draw()
            self._emit_lines_changed()
            
            logger.debug(f"Line orientation changed to: {orientation}")
            
        except Exception as e:
            logger.error(f"Error setting line orientation: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    def set_bandwidth(self, width: float):
        """Set bandwidth with enhanced error handling."""
        logger.debug(f"Setting bandwidth to: {width}")
        
        try:
            self._bandwidth = max(0.0, float(width))
            self._update_bands()
            self._emit_lines_changed()
            
        except Exception as e:
            logger.error(f"Error setting bandwidth: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    def ensure_line_count(self, n: int):
        """Ensure specific line count with enhanced error handling."""
        logger.debug(f"Ensuring line count: {n}")
        
        try:
            n = max(0, min(3, n))
            
            if len(self._lines) == 0 and n > 0:
                self._create_initial_line()
                
            while len(self._lines) < n:
                self._add_line_internal()
                
            while len(self._lines) > n:
                ln = self._lines.pop()
                try:
                    if ln is not None and hasattr(ln, 'remove'):
                        ln.remove()
                except Exception as e:
                    logger.warning(f"Error removing line: {e}")
                    
            self._update_bands()
            self._safe_canvas_draw()
            self._emit_lines_changed()
            
            logger.debug(f"Line count set to: {len(self._lines)}")
            
        except Exception as e:
            logger.error(f"Error ensuring line count: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    def add_line(self):
        """Add line with enhanced error handling."""
        logger.debug("Adding line")
        
        try:
            if len(self._lines) == 0:
                self._create_initial_line()
            elif len(self._lines) < 3:
                self._add_line_internal()
                
            self._update_bands()
            self._safe_canvas_draw()
            self._emit_lines_changed()
            
            logger.debug(f"Line added, total lines: {len(self._lines)}")
            
        except Exception as e:
            logger.error(f"Error adding line: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    def remove_line(self):
        """Remove line with enhanced error handling."""
        logger.debug("Removing line")
        
        try:
            if len(self._lines) == 0:
                return
                
            ln = self._lines.pop()
            try:
                if ln is not None and hasattr(ln, 'remove'):
                    ln.remove()
            except Exception as e:
                logger.warning(f"Error removing line: {e}")
                
            self._update_bands()
            self._safe_canvas_draw()
            self._emit_lines_changed()
            
            logger.debug(f"Line removed, remaining lines: {len(self._lines)}")
            
        except Exception as e:
            logger.error(f"Error removing line: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    def get_line_positions(self) -> List[float]:
        """Get line positions with enhanced error handling."""
        logger.debug("Getting line positions")
        
        try:
            pos = []
            for i, ln in enumerate(self._lines):
                try:
                    if self._line_orientation == "vertical":
                        xdata = ln.get_xdata(orig=False)
                        if len(xdata) >= 2 and xdata[0] == xdata[1]:
                            pos.append(xdata[0])
                    else:
                        ydata = ln.get_ydata(orig=False)
                        if len(ydata) >= 2 and ydata[0] == ydata[1]:
                            pos.append(ydata[0])
                except Exception as e:
                    logger.warning(f"Error getting position for line {i}: {e}")
                    
            logger.debug(f"Got {len(pos)} line positions")
            return pos
            
        except Exception as e:
            logger.error(f"Error getting line positions: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []

    def set_line_positions(self, positions: List[float]):
        """Set line positions with enhanced error handling."""
        logger.debug(f"Setting line positions: {positions}")
        
        try:
            self.ensure_line_count(len(positions))
            
            for i, (ln, val) in enumerate(zip(self._lines, positions)):
                try:
                    if self._line_orientation == "vertical":
                        ln.set_xdata([val, val])
                    else:
                        ln.set_ydata([val, val])
                except Exception as e:
                    logger.warning(f"Error setting position for line {i}: {e}")
                    
            self._update_bands()
            self._safe_canvas_draw()
            self._emit_lines_changed()
            
            logger.debug("Line positions set successfully")
            
        except Exception as e:
            logger.error(f"Error setting line positions: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    def _create_initial_line(self):
        """Create initial line with enhanced error handling."""
        logger.debug(f"Creating initial line, orientation: {self._line_orientation}")
        
        try:
            if not hasattr(self, 'ax_img') or self.ax_img is None:
                logger.error("No ax_img available for creating line")
                return
            
            if self._line_orientation == "vertical":
                xmin, xmax = self.ax_img.get_xlim()
                x = xmin + 0.5 * (xmax - xmin)
                ln = self.ax_img.axvline(x, color=_LINE_COLORS[0], lw=1.8, alpha=0.95)
            else:
                ymin, ymax = self.ax_img.get_ylim()
                y = ymin + 0.5 * (ymax - ymin)
                ln = self.ax_img.axhline(y, color=_LINE_COLORS[0], lw=1.8, alpha=0.95)
                
            self._lines = [ln]
            logger.debug("Initial line created successfully")
            
        except Exception as e:
            logger.error(f"Error creating initial line: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    def _add_line_internal(self):
        """Add line internally with enhanced error handling."""
        logger.debug("Adding line internally")
        
        try:
            if not hasattr(self, 'ax_img') or self.ax_img is None:
                logger.error("No ax_img available for adding line")
                return
            
            idx = len(self._lines)
            color = _LINE_COLORS[min(idx, len(_LINE_COLORS) - 1)]
            
            if self._line_orientation == "vertical":
                xmin, xmax = self.ax_img.get_xlim()
                x = xmin + (0.33 if idx == 1 else 0.66) * (xmax - xmin)
                ln = self.ax_img.axvline(x, color=color, lw=1.6, alpha=0.9, ls="--")
            else:
                ymin, ymax = self.ax_img.get_ylim()
                y = ymin + (0.33 if idx == 1 else 0.66) * (ymax - ymin)
                ln = self.ax_img.axhline(y, color=color, lw=1.6, alpha=0.9, ls="--")
                
            self._lines.append(ln)
            logger.debug(f"Line added internally, total lines: {len(self._lines)}")
            
        except Exception as e:
            logger.error(f"Error adding line internally: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    # ------------ band management with enhanced error handling ------------
    
    def _update_bands(self):
        """Update bands with enhanced error handling."""
        logger.debug("Updating bands")
        
        try:
            self._safe_remove_bands()
            
            if len(self._lines) == 0 or self._bandwidth <= 0:
                self._safe_canvas_draw()
                return
                
            half = self._bandwidth / 2.0
            
            for i, ln in enumerate(self._lines):
                try:
                    color = _LINE_COLORS[min(i, len(_LINE_COLORS) - 1)]
                    
                    if self._line_orientation == "vertical":
                        xdata = ln.get_xdata(orig=False)
                        if len(xdata) < 2 or xdata[0] != xdata[1]:
                            continue  # not a vertical line
                        x0 = xdata[0]
                        band = self.ax_img.axvspan(x0 - half, x0 + half, color=color, alpha=_BAND_ALPHA, lw=0)
                    else:
                        ydata = ln.get_ydata(orig=False)
                        if len(ydata) < 2 or ydata[0] != ydata[1]:
                            continue  # not a horizontal line
                        y0 = ydata[0]
                        band = self.ax_img.axhspan(y0 - half, y0 + half, color=color, alpha=_BAND_ALPHA, lw=0)
                        
                    self._bands.append(band)
                    logger.debug(f"Band {i} created successfully")
                    
                except Exception as e:
                    logger.warning(f"Error creating band {i}: {e}")
            
            self._safe_canvas_draw()
            logger.debug(f"Bands updated successfully, total bands: {len(self._bands)}")
            
        except Exception as e:
            logger.error(f"Error updating bands: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    # ------------ mouse interaction with enhanced error handling ------------
    
    def _on_press(self, event):
        """Handle mouse press with enhanced error handling."""
        logger.debug("Mouse press event")
        
        try:
            if event.inaxes != self.ax_img or len(self._lines) == 0:
                return
                
            tol = 10.0
            
            for idx, ln in enumerate(self._lines):
                try:
                    if self._line_orientation == "vertical":
                        xdata = ln.get_xdata()
                        if len(xdata) > 0:
                            x0 = xdata[0]
                            dpx = abs(self.ax_img.transData.transform((x0, 0))[0] - event.x)
                            if dpx <= tol:
                                self._drag_idx = idx
                                logger.debug(f"Started dragging line {idx}")
                                break
                    else:
                        ydata = ln.get_ydata()
                        if len(ydata) > 0:
                            y0 = ydata[0]
                            dpy = abs(self.ax_img.transData.transform((0, y0))[1] - event.y)
                            if dpy <= tol:
                                self._drag_idx = idx
                                logger.debug(f"Started dragging line {idx}")
                                break
                except Exception as e:
                    logger.warning(f"Error checking line {idx} for drag: {e}")
                    
        except Exception as e:
            logger.error(f"Error in _on_press: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    def _on_motion(self, event):
        """Handle mouse motion with enhanced error handling."""
        try:
            if self._drag_idx is None or event.inaxes != self.ax_img:
                return
                
            if self._drag_idx >= len(self._lines):
                logger.warning(f"Invalid drag index: {self._drag_idx}")
                self._drag_idx = None
                return
            
            line = self._lines[self._drag_idx]
            
            if self._line_orientation == "vertical":
                if event.xdata is None:
                    return
                x = event.xdata
                line.set_xdata([x, x])
            else:
                if event.ydata is None:
                    return
                y = event.ydata
                line.set_ydata([y, y])
                
            self._update_bands()
            self._safe_canvas_draw()
            self._emit_lines_changed()
            
        except Exception as e:
            logger.error(f"Error in _on_motion: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    def _on_release(self, event):
        """Handle mouse release with enhanced error handling."""
        logger.debug("Mouse release event")
        
        try:
            if self._drag_idx is not None:
                logger.debug(f"Finished dragging line {self._drag_idx}")
                
            self._drag_idx = None
            self._emit_lines_changed()
            self._update_bands()
            
        except Exception as e:
            logger.error(f"Error in _on_release: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    # Profile panel with enhanced error handling
    
    def plot_profiles(self, x_label: str, curves: List[Tuple[np.ndarray, np.ndarray, str]]):
        """Plot profiles with enhanced error handling."""
        logger.debug(f"Plotting {len(curves)} profiles")
        
        try:
            if not hasattr(self, 'ax_prof') or self.ax_prof is None:
                logger.error("No profile axis available")
                return
            
            self.ax_prof.clear()
            
            for i, (xv, yv, lbl) in enumerate(curves):
                try:
                    if xv is not None and yv is not None:
                        self.ax_prof.plot(xv, yv, label=lbl)
                        logger.debug(f"Plotted profile {i}: {lbl}")
                except Exception as e:
                    logger.warning(f"Error plotting profile {i}: {e}")
            
            self.ax_prof.set_xlabel(x_label)
            self.ax_prof.set_ylabel("Integrated counts")
            
            if curves:
                try:
                    self.ax_prof.legend(loc="best", fontsize=9)
                except Exception as e:
                    logger.warning(f"Error adding profile legend: {e}")
                    
            try:
                self.ax_prof.grid(True, alpha=0.3)
            except Exception as e:
                logger.warning(f"Error adding profile grid: {e}")
                
            self._safe_canvas_draw()
            logger.debug("Profiles plotted successfully")
            
        except Exception as e:
            logger.error(f"Error in plot_profiles: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")