import os, gc, logging, traceback, weakref
from typing import Optional, List, Tuple

import numpy as np

# Set up logging for enhanced error handling and debugging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(name)s:%(lineno)d - %(message)s')
    handler.setFormatter(formatter)
    logger.setLevel(logging.WARN)
from PySide6.QtCore import Qt, QSignalBlocker
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QTabWidget,
    QFileDialog, QMessageBox, QInputDialog, QDialog, QListWidgetItem,
    QGroupBox
)

from xestools.modules.dataset import DataSet
from xestools.widgets.io_panel import IOPanel, RXESPanel
from xestools.widgets.xes_panel import XESPanel, SPECIAL_ROLE # SPECIAL ROLE to help known which boxes are ticked at a given time.
from xestools.widgets.normalise_dialog import NormaliseDialog
from xestools.widgets.background_dialog import BackgroundDialog
from xestools.modules import io as io_mod
from xestools.modules.scan import Scan
from xestools.modules import i20_loader

# ==================== FEATURE FLAG: Silx Integration ====================
# Set to True to use silx-based plotting (professional synchrotron tools)
# Set to False to use matplotlib-based plotting (legacy)
USE_SILX = True

# Set to True to use new RIXS-style Plot2D with built-in profile extraction
# Set to False to use wrapped ImageView with manual ROI lines
USE_RIXS_PLOT = True
# =========================================================================

# Conditional import based on feature flag
if USE_SILX:
    if USE_RIXS_PLOT:
        try:
            from xestools.widgets.rixs_widget import RIXSWidget as PlotWidget
            logger.info("Using RIXS-style Plot2D with profile toolbar (xraylarch style)")
        except ImportError as e:
            logger.warning(f"Failed to import RIXSWidget, falling back to SilxPlotWidget: {e}")
            try:
                from xestools.widgets.silx_plot_widget import SilxPlotWidget as PlotWidget
                logger.info("Using silx-based plotting (professional mode)")
                USE_RIXS_PLOT = False
            except ImportError as e2:
                logger.warning(f"Failed to import silx, falling back to matplotlib: {e2}")
                from xestools.widgets.plot_widget import PlotWidget
                USE_SILX = False
                USE_RIXS_PLOT = False
    else:
        try:
            from xestools.widgets.silx_plot_widget import SilxPlotWidget as PlotWidget
            logger.info("Using silx-based plotting (professional mode)")
        except ImportError as e:
            logger.warning(f"Failed to import silx, falling back to matplotlib: {e}")
            from xestools.widgets.plot_widget import PlotWidget
            USE_SILX = False
else:
    from xestools.widgets.plot_widget import PlotWidget
    logger.info("Using matplotlib-based plotting (legacy mode)")

# special keys (use these exact strings everywhere)
AVG_KEY = "average"
BKSUB_KEY = "average_bksub"  # note: 'bksub' (no 'g')

# Prefix for multiple averages (e.g., "average_279496+279517")
MULTI_AVG_PREFIX = "average_"

class MainWindow(QMainWindow):
    """Enhanced Main Window with comprehensive error handling, logging, and memory management."""
    def __init__(self):

        super().__init__()
        self.setWindowTitle("Luke's Handy I20 XES/RXES Explorer")
        self.resize(1300, 900)
        
        # Enhanced state tracking for cleanup
        self._is_closing = False
        self._plot_widget_ref: Optional[weakref.ReferenceType] = None

        # Session containers
        self.scan = Scan()
        self.current_scan_number: Optional[int] = None
        self.dataset: Optional[DataSet] = None
        self._last_profiles: List[Tuple[np.ndarray, np.ndarray, str]] = []
        self._last_bkg: tuple[np.ndarray, np.ndarray] | None = None    # (x, background)
        self._last_resid: tuple[np.ndarray, np.ndarray] | None = None  # (x, residual)
        self._last_bkg_report: str = ""
        self._xes_wide: tuple[np.ndarray, np.ndarray, str] | None = None  # (x, y, path)

        self._xes_avg_bksub: Optional[tuple[np.ndarray, np.ndarray]] = None

        # XES multi-scan workflow
        self._xes_items: List[dict] = []  # each: dict(path, channel, x, y, label)
        self._xes_norm_target = None  # ('avg', None) or ('single', index)
        self._xes_avg: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._xes_avg_norm_factor: Optional[float] = None
        
        # Multiple averages support: dict of key -> (x, y) tuples
        # Keys are like "average_279496+279517" indicating which scans were averaged
        self._xes_multi_avgs: dict = {}  # key -> (x_array, y_array)

        # Layout
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)

        # Left: I/O + tabs (RXES | XES)
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)

        self.io_panel = IOPanel()
        self.rxes_panel = RXESPanel()
        self.xes_panel = XESPanel()

        self.tabs = QTabWidget()
        self.tabs.addTab(self.rxes_panel, "RXES")
        self.tabs.addTab(self.xes_panel, "XES")

        left_layout.addWidget(self.io_panel)
        left_layout.addWidget(self.tabs)
        left_layout.addStretch(1)

        # Right: Plot area with stacked widgets for RXES (2D) and XES (1D)
        from PySide6.QtWidgets import QStackedWidget
        from silx.gui.plot import Plot1D
        
        self.plot_stack = QStackedWidget()
        
        # RXES plot (2D) - index 0
        self.plot = PlotWidget()  # This is RIXSWidget for 2D RXES maps
        self.plot_stack.addWidget(self.plot)
        
        # XES plot (1D) - index 1  
        self.xes_plot = Plot1D()
        self.xes_plot.setGraphTitle("XES Spectra")
        self.xes_plot.setGraphXLabel("Energy (eV)")
        self.xes_plot.setGraphYLabel("Intensity")
        self.xes_plot.setDataMargins(yMinMargin=0.1, yMaxMargin=0.1)
        # Enable position info snapping to curves
        positionInfo = self.xes_plot.getPositionInfoWidget()
        if positionInfo is not None:
            positionInfo.setSnappingMode(positionInfo.SNAPPING_CURVE)
        self.plot_stack.addWidget(self.xes_plot)
        
        # Store weak reference for cleanup tracking
        self._plot_widget_ref = weakref.ref(self.plot)
        
        try:
            # Connect plot signals based on widget type
            if hasattr(self.plot, 'lines_changed'):
                # Old SilxPlotWidget/PlotWidget style
                self.plot.lines_changed.connect(self.update_profiles)
            if hasattr(self.plot, 'profile_extracted'):
                # New RIXSWidget style - profiles are extracted via toolbar
                self.plot.profile_extracted.connect(self._on_profile_extracted)
        except Exception as e:
            logger.warning(f"Failed to connect plot signals: {e}")

        root_layout.addWidget(left_container, 0)
        root_layout.addWidget(self.plot_stack, 1)

        # I/O actions
        self.io_panel.btn_load.clicked.connect(self.on_load)
        self.io_panel.btn_save_ascii.clicked.connect(self.on_save_ascii)
        self.io_panel.btn_save_nexus.clicked.connect(self.on_save_nexus)

        # RXES controls
        self.rxes_panel.rb_upper.toggled.connect(self.on_rxes_channel_changed)
        self.rxes_panel.rb_lower.toggled.connect(self.on_rxes_channel_changed)
        self.rxes_panel.rb_mode_incident.toggled.connect(self.on_mode_changed)
        self.rxes_panel.rb_mode_transfer.toggled.connect(self.on_mode_changed)
        self.rxes_panel.chk_contours.toggled.connect(self.on_contours_toggled)
        self.rxes_panel.spn_contour_levels.valueChanged.connect(self.on_contours_toggled)
        self.rxes_panel.cmb_contour_color.currentTextChanged.connect(self.on_contours_toggled)
        self.rxes_panel.spn_contour_width.valueChanged.connect(self.on_contours_toggled)
        self.rxes_panel.chk_contour_fill.toggled.connect(self.on_contours_toggled)
        self.rxes_panel.chk_contour_gravity.toggled.connect(self.on_contours_toggled)
        self.rxes_panel.chk_contour_labels.toggled.connect(self.on_contours_toggled)
        self.rxes_panel.btn_load_xes.clicked.connect(self.on_rxes_normalise)
        
        # ROI extraction controls - only needed for old plot widget
        if not USE_RIXS_PLOT:
            # Old style: manual ROI line management
            self.rxes_panel.btn_add_line.clicked.connect(self.on_add_line)
            self.rxes_panel.btn_remove_line.clicked.connect(self.on_remove_line)
            self.rxes_panel.btn_update_spectrum.clicked.connect(self.on_update_spectrum)
            self.rxes_panel.btn_save_spectrum.clicked.connect(self.on_save_spectrum)
            self.rxes_panel.sl_width.valueChanged.connect(self.on_bandwidth_changed)
            self.rxes_panel.rb_extr_incident.toggled.connect(self.on_extraction_changed)
            self.rxes_panel.rb_extr_emission.toggled.connect(self.on_extraction_changed)
            self.rxes_panel.rb_extr_transfer.toggled.connect(self.on_extraction_changed)
        else:
            # New RIXS style: hide obsolete ROI controls (profile toolbar handles this)
            # Hide the entire ROI Extraction groupbox
            for i in range(self.rxes_panel.layout().count()):
                widget = self.rxes_panel.layout().itemAt(i).widget()
                if widget and isinstance(widget, QGroupBox) and widget.title() == "ROI Extraction":
                    widget.setVisible(False)
                    logger.info("ROI Extraction panel hidden (using silx profile toolbar instead)")
                    break

        # XES controls
        self.xes_panel.btn_load_scans.clicked.connect(self.on_xes_load_scans)
        self.xes_panel.btn_remove_selected.clicked.connect(self.on_xes_remove_selected)
        self.xes_panel.btn_clear_all.clicked.connect(self.on_xes_clear_all)
        # Refresh overlays when ticking/unticking scans
        # self.xes_panel.list.itemChanged.connect(self.on_xes_list_item_changed)
        self.xes_panel.list.itemChanged.connect(lambda : self._refresh_xes_plot())
        self._xes_list_bulk_updating = False # Guard to suppress itemChanged during bulk updates
        # Buffer for background-subtracted average
        self._xes_avg_bkgsub: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.xes_panel.btn_average.clicked.connect(self.on_xes_average_selected)
        self.xes_panel.btn_save_norm.clicked.connect(self.on_xes_save_normalised)
        self.xes_panel.btn_save_average.clicked.connect(self.on_xes_save_average)
        self.xes_panel.btn_save_avg_norm.clicked.connect(self.on_xes_save_normalised_average)
        self.xes_panel.btn_load_xes.clicked.connect(self.on_xes_normalise_average)  # Normalisation
        # Background
        self.xes_panel.btn_load_wide.clicked.connect(self.on_xes_load_wide)
        self.xes_panel.btn_bkg_extract.clicked.connect(self.on_xes_background_extract)
        self.xes_panel.btn_save_fit_log.clicked.connect(self.on_xes_save_fit_log)
        self.xes_panel.btn_save_bkg_data.clicked.connect(self.on_xes_save_bkg_data)
        self.xes_panel.rb_upper.toggled.connect(self.on_xes_channel_changed)
        self.xes_panel.rb_lower.toggled.connect(self.on_xes_channel_changed)

        # XES background extraction state
        self._xes_fit_bounds: Optional[Tuple[float, float, float, float]] = None
        self._xes_background: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._xes_residual: Optional[Tuple[np.ndarray, np.ndarray]] = None

        # Tabs change
        self.tabs.currentChanged.connect(self.on_tab_changed)

        # Status bar
        self.status = self.statusBar()
        self.status.showMessage("Ready")

        self.update_ui_state()
        self._refresh_xes_plot()
        
        logger.info("MainWindow initialized")

    def closeEvent(self, event):
        """Handle window closure with comprehensive cleanup."""
        logger.info("Enhanced MainWindow closeEvent triggered")
        
        try:
            self._is_closing = True
            
            # Cleanup plot widget if it has cleanup methods
            if hasattr(self.plot, 'closeEvent'):
                try:
                    self.plot.closeEvent(event)
                    logger.debug("Plot widget cleanup triggered")
                except Exception as e:
                    logger.warning(f"Error during plot widget cleanup: {e}")
            
            # Clear data structures to free memory
            self._cleanup_data_structures()
            
            # Force garbage collection
            gc.collect()
            
            logger.info("Enhanced MainWindow cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during closeEvent: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        finally:
            # Always accept the close event
            event.accept()
            
            # Call parent closeEvent
            try:
                super().closeEvent(event)
            except Exception as e:
                logger.warning(f"Error calling parent closeEvent: {e}")

    def _cleanup_data_structures(self):
        """Clean up data structures to free memory."""
        logger.debug("Cleaning up data structures")
        
        try:
            # Clear large data structures
            if hasattr(self, '_xes_items'):
                self._xes_items.clear()
            if hasattr(self, '_last_profiles'):
                self._last_profiles.clear()
            
            # Clear arrays
            self._last_bkg = None
            self._last_resid = None
            self._xes_wide = None
            self._xes_avg = None
            self._xes_avg_bksub = None
            
            # Clear dataset
            self.dataset = None
            
            logger.debug("Data structures cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up data structures: {e}")

    # ---------------- UI state and Dataset ----------------
    def update_ui_state(self):
        has_data = self.dataset is not None
        self.io_panel.btn_save_ascii.setEnabled(has_data)
        try:
            self.io_panel.btn_save_nexus.setEnabled(has_data and io_mod.H5_AVAILABLE)
            if not io_mod.H5_AVAILABLE:
                self.io_panel.btn_save_nexus.setToolTip("Install h5py to enable NeXus saving: pip install h5py")
            else:
                self.io_panel.btn_save_nexus.setToolTip("")
        except Exception:
            pass

        if self.dataset:
            self.io_panel.path_edit.setText(self.dataset.source or "")
            if self.dataset.kind == "1D":
                n = len(self.dataset.y) if self.dataset.y is not None else 0
                self.io_panel.info_label.setText(f"1D spectrum, N={n}")
            elif self.dataset.kind == "2D":
                ny, nx = self.dataset.z.shape
                self.io_panel.info_label.setText(f"2D map, shape={ny}×{nx}")
            else:
                self.io_panel.info_label.setText("Unknown dataset")
        else:
            self.io_panel.path_edit.setText("")
            self.io_panel.info_label.setText("No data loaded")

        incident_mode = self.rxes_panel.rb_mode_incident.isChecked()
        
        # Only update extraction mode controls if not using new RIXS widget
        if not USE_RIXS_PLOT:
            self.rxes_panel.rb_extr_incident.setEnabled(True)
            self.rxes_panel.rb_extr_emission.setEnabled(incident_mode)
            self.rxes_panel.rb_extr_transfer.setEnabled(not incident_mode)

        if self.tabs.currentIndex() == 1 or (self.dataset and self.dataset.kind == "1D"):
            chan = "Upper" if self.xes_panel.rb_upper.isChecked() else "Lower"
            mode = "XES (1D)"
        else:
            chan = "Upper" if self.rxes_panel.rb_upper.isChecked() else "Lower"
            mode = "Incident Energy (Ω vs ω)" if incident_mode else "Energy Transfer (Ω vs Ω−ω)"
        base = os.path.basename(self.dataset.source) if self.dataset and self.dataset.source else ""
        self.io_panel.status_label.setText(f"Channel: {chan} | Mode: {mode} | File: {base}")

        if self.current_scan_number is not None and self.current_scan_number in self.scan:
            nf = self.scan[self.current_scan_number].get("norm_factor", None)
            if nf and np.isfinite(nf) and nf > 0:
                self.rxes_panel.lbl_norm.setText(f"Normalised by XES area: {nf:.6g}")
            else:
                self.rxes_panel.lbl_norm.setText("No normalisation applied")
        else:
            self.rxes_panel.lbl_norm.setText("No normalisation applied")
    def set_dataset(self, dataset: Optional[DataSet]):
        self.dataset = dataset
        self.plot.plot(self.dataset)
        if self.dataset and self.dataset.kind == "2D":
            try:
                self.plot.set_signal_suppressed(True)
                self.set_line_orientation_for_current_mode()
                self.plot.ensure_line_count(1)
                self.plot.set_bandwidth(self._get_bandwidth_ev())
                self.plot.set_signal_suppressed(False)
                self.update_profiles()
            except Exception:
                pass
        self.update_ui_state()

    # ---------------- Load dispatcher ----------------
    def on_load(self):
        """Load I20 data with automatic scan type detection."""
        filters = [
            "I20 data (*.nxs *.h5 *.hdf *.hdf5 *.dat *.txt *.csv)",
            "NeXus (*.nxs *.h5 *.hdf *.hdf5)",
            "ASCII (*.dat *.txt *.csv)",
            "All files (*)"
        ]
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Load I20 Data", "", ";;".join(filters)
        )
        if not paths:
            return
        
        # Multiple files → XES workflow (batch loading)
        if len(paths) > 1:
            self._load_xes_batch(paths)
            return
        
        # Single file → auto-detect
        self._load_auto_detect(paths[0])

    def _load_auto_detect(self, path):
        """Load single file with automatic scan type detection."""
        try:
            # Auto-detect scan type from file
            scan_type = i20_loader.detect_scan_type_from_file(path)
            
            # Route to appropriate loader
            if scan_type == 'RXES':
                self._load_rxes_single(path)
                self.status.showMessage(
                    f"Auto-detected RXES (2D map): {os.path.basename(path)}", 
                    5000
                )
            elif scan_type == 'XES':
                self._load_xes_batch([path])
                self.status.showMessage(
                    f"Auto-detected XES (1D spectrum): {os.path.basename(path)}", 
                    5000
                )
            else:
                raise ValueError(f"Unknown scan type: {scan_type}")
                
        except ValueError as e:
            # XANES or detection failure
            QMessageBox.critical(
                self, "Cannot load file",
                f"Auto-detection failed for:\n{path}\n\n{e}\n\n"
                f"Tip: Use XES panel 'Load Scans...' to force XES loading."
            )
    def _load_rxes_single(self, path):
        """Load single file as RXES (2D map). Path already selected."""
        try:
            ext = os.path.splitext(path)[1].lower()
            
            # Handle NeXus files
            if ext in (".nxs", ".h5", ".hdf", ".hdf5"):
                if i20_loader.is_probably_detector_hdf(path):
                    QMessageBox.warning(self, "Detector HDF selected",
                                        "This looks like a raw detector file. Please pick the scan .nxs/.h5.")
                    return
                scan_number = i20_loader.add_scan_from_nxs(self.scan, path)
            
            # Handle ASCII files
            elif ext in (".dat", ".txt", ".csv"):
                scan_number = i20_loader.add_scan_from_i20_ascii(self.scan, path)
            
            # Unknown extension - try NeXus as default
            else:
                scan_number = i20_loader.add_scan_from_nxs(self.scan, path)
            
            self.current_scan_number = scan_number
            self.status.showMessage(f"Loaded RXES: {path}", 5000)
            self.tabs.setCurrentIndex(0)
            self.refresh_rxes_view()
            
        except Exception as e:
            QMessageBox.critical(self, "Load error", f"Failed to load RXES:\n{path}\n\n{e}")

    # ---------------- RXES: Views and Controls ----------------
    def refresh_rxes_view(self):
        if self.current_scan_number is None:
            return
        entry = self.scan.get(self.current_scan_number)
        if not entry:
            return

        use_upper = self.rxes_panel.rb_upper.isChecked()
        mode_transfer = self.rxes_panel.rb_mode_transfer.isChecked()

        # Select emission/intensity by channel
        if use_upper:
            emission_2d = entry.get("energy_upper_2d")
            Z = entry.get("intensity_upper")
            zlabel = "FFI1_medipix1 (counts)"
        else:
            emission_2d = entry.get("energy_lower_2d")
            Z = entry.get("intensity_lower")
            zlabel = "FFI1_medipix2 (counts)"

        if emission_2d is None or Z is None:
            QMessageBox.warning(self, "Missing channel", "Selected channel not present in this scan.")
            return

        bragg_off_2d = entry.get("braggOffset_2d")
        if bragg_off_2d is None:
            QMessageBox.critical(self, "Missing axis", "bragg1WithOffset grid not found.")
            return

        # Axes: Y = ω (rows), X = Ω (cols)
        y_omega, x_Omega, transposed = i20_loader.reduce_axes_for(emission_2d, bragg_off_2d)
        Z = np.asarray(Z, dtype=float)
        if transposed:
            Z = Z.T

        # Apply RXES normalisation factor (by XES area) if present
        nf = entry.get("norm_factor", None)
        if nf and np.isfinite(nf) and nf > 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                Z = Z / nf
            if "area" not in zlabel:
                zlabel = zlabel + " / area"

        # Build DataSet for plotting
        if mode_transfer:
            # Energy transfer view: X=Ω, Y=Ω−ω
            X2D = np.broadcast_to(x_Omega[None, :], Z.shape)
            Y2D = X2D - np.broadcast_to(y_omega[:, None], Z.shape)
            ds = DataSet("2D",
                         x2d=X2D, y2d=Y2D, z=Z,
                         xlabel="Ω (eV)", ylabel="Ω − ω (eV)", zlabel=zlabel,
                         source=entry.get("path", ""))
        else:
            ds = DataSet("2D",
                         x=x_Omega, y=y_omega, z=Z,
                         xlabel="Ω (eV)", ylabel="ω (eV)", zlabel=zlabel,
                         source=entry.get("path", ""))

        self.set_dataset(ds)
        try:
            self.plot.autoscale_current()
        except Exception:
            pass
    def on_mode_changed(self):
        incident_mode = self.rxes_panel.rb_mode_incident.isChecked()
        
        # Only manage extraction mode controls for old plot widget
        if not USE_RIXS_PLOT:
            if incident_mode and self.rxes_panel.rb_extr_transfer.isChecked():
                self.rxes_panel.rb_extr_incident.setChecked(True)
            if not incident_mode and self.rxes_panel.rb_extr_emission.isChecked():
                self.rxes_panel.rb_extr_incident.setChecked(True)
        
        self.refresh_rxes_view()
        try:
            self.plot.autoscale_current()
        except Exception:
            pass
    
    def on_contours_toggled(self, _=None):
        """Handle contour overlay toggle and level changes."""
        if not USE_RIXS_PLOT:
            return
        
        show_contours = self.rxes_panel.chk_contours.isChecked()
        n_levels = self.rxes_panel.spn_contour_levels.value()
        color = self.rxes_panel.cmb_contour_color.currentText()
        line_width = self.rxes_panel.spn_contour_width.value()
        fill_islands = self.rxes_panel.chk_contour_fill.isChecked()
        show_gravity = self.rxes_panel.chk_contour_gravity.isChecked()
        show_labels = self.rxes_panel.chk_contour_labels.isChecked()
        
        if hasattr(self.plot, 'rxesPlot'):
            if show_contours:
                self._add_contours_to_plot(n_levels, color, line_width, fill_islands, show_gravity, show_labels)
            else:
                self._remove_contours_from_plot()
    
    def _add_contours_to_plot(self, n_levels: int = 10, color: str = 'white', line_width: float = 0.5,
                               fill_islands: bool = False, show_gravity: bool = False, show_labels: bool = False):
        """Add contour lines to the current RXES map using silx marchingsquares.
        
        Args:
            n_levels: Number of contour levels
            color: Color of contour lines (e.g., 'white', 'black', 'red')
            line_width: Width of contour lines
            fill_islands: Fill closed contour regions with semi-transparent color
            show_gravity: Show gravity centers (centroids) of closed contours
            show_labels: Label contours with their intensity values
        """
        if not hasattr(self.plot, 'rxesPlot'):
            return
        
        # Get the current image
        image = self.plot.rxesPlot.getImage("RXES")
        if image is None:
            return
        
        from silx.image import marchingsquares
        
        data = image.getData(copy=False)
        origin = image.getOrigin()
        scale = image.getScale()
        
        # Compute contour levels
        vmin, vmax = np.nanmin(data), np.nanmax(data)
        levels = np.linspace(vmin, vmax, n_levels + 2)[1:-1]  # Exclude min/max
        
        # Remove old contours first
        self._remove_contours_from_plot()
        
        # Create contour lines using silx marchingsquares
        try:
            contour_count = 0
            gravity_centers = []
            label_positions = []
            
            for i, level in enumerate(levels):
                # find_contours returns polygons in pixel coordinates
                polygons = marchingsquares.find_contours(data, level)
                
                for j, polygon in enumerate(polygons):
                    if len(polygon) > 1:
                        # Convert pixel coordinates to data coordinates
                        # polygon is (N, 2) array with (row, col) format
                        pixel_y = polygon[:, 0]  # rows
                        pixel_x = polygon[:, 1]  # columns
                        
                        # Apply scale and origin transformation
                        data_x = origin[0] + pixel_x * scale[0]
                        data_y = origin[1] + pixel_y * scale[1]
                        
                        # Check if polygon is closed (for island filling and gravity)
                        is_closed = np.allclose(polygon[0], polygon[-1], atol=1.0)
                        
                        # Fill islands (closed polygons) with semi-transparent color
                        if fill_islands and is_closed and len(data_x) > 2:
                            # Create a fill using scatter with low alpha
                            # Note: silx Plot2D doesn't have native polygon fill,
                            # so we approximate with the contour line itself
                            pass  # Would need custom Shape item for proper fill
                        
                        # Add contour line
                        self.plot.rxesPlot.addCurve(
                            data_x, data_y,
                            legend=f"_contour_{i}_{j}",
                            color=color,
                            linewidth=line_width,
                            linestyle='-',
                            selectable=False
                        )
                        contour_count += 1
                        
                        # Calculate gravity center (centroid) for closed polygons
                        if show_gravity and is_closed and len(data_x) > 2:
                            centroid_x = np.mean(data_x)
                            centroid_y = np.mean(data_y)
                            gravity_centers.append((centroid_x, centroid_y, level))
                        
                        # Store label position (mid-point of contour)
                        if show_labels and len(data_x) > 2:
                            mid_idx = len(data_x) // 2
                            label_positions.append((data_x[mid_idx], data_y[mid_idx], level))
            
            # Add gravity center markers
            if show_gravity and gravity_centers:
                gx = [g[0] for g in gravity_centers]
                gy = [g[1] for g in gravity_centers]
                self.plot.rxesPlot.addScatter(
                    gx, gy,
                    legend="_contour_gravity",
                    symbol='o',
                    colormap=None,
                    value=[g[2] for g in gravity_centers]
                )
                logger.info(f"Added {len(gravity_centers)} gravity centers")
            
            # Add contour value labels
            if show_labels and label_positions:
                # Track marker names for cleanup
                if not hasattr(self, '_contour_label_markers'):
                    self._contour_label_markers = []
                
                # silx doesn't have native text annotation, but we can use markers
                # For now, log the positions - would need custom text items
                for lx, ly, lv in label_positions[:n_levels]:  # Limit to one per level
                    # Add a small marker at label position
                    marker_name = f"_contour_label_{lv:.2f}"
                    self.plot.rxesPlot.addMarker(
                        lx, ly,
                        legend=marker_name,
                        text=f"{lv:.3g}",
                        color=color,
                        symbol=None
                    )
                    self._contour_label_markers.append(marker_name)
                logger.info(f"Added {min(len(label_positions), n_levels)} contour labels")
            
            logger.info(f"Added {contour_count} contour segments ({n_levels} levels, color={color}, width={line_width})")
        except Exception as e:
            logger.warning(f"Could not add contours: {e}")
            import traceback
            traceback.print_exc()
    
    def _remove_contours_from_plot(self):
        """Remove all contour lines, markers, and scatter plots from the plot."""
        if not hasattr(self.plot, 'rxesPlot'):
            return
        
        # Remove all curves that start with _contour_
        curves = self.plot.rxesPlot.getAllCurves()
        for curve in curves:
            name = curve.getName()
            if name.startswith("_contour_"):
                self.plot.rxesPlot.remove(name, kind='curve')
        
        # Remove gravity center scatter plot
        try:
            self.plot.rxesPlot.remove("_contour_gravity", kind='scatter')
        except Exception:
            pass
        
        # Remove contour label markers (stored in _contour_label_markers list)
        if hasattr(self, '_contour_label_markers'):
            for marker_name in self._contour_label_markers:
                try:
                    self.plot.rxesPlot.removeMarker(marker_name)
                except Exception:
                    pass
            self._contour_label_markers = []
        
        logger.debug("Removed contour overlays")
    
    def on_extraction_changed(self, checked: bool):
        # Only react when a radio becomes checked; ignore the False uncheck event
        if not checked:
            return

        # Prevent illegal combinations by switching modes when needed
        # 1) If user selects Constant Transfer while in Incident mode -> switch to Transfer
        if self.rxes_panel.rb_extr_transfer.isChecked() and self.rxes_panel.rb_mode_incident.isChecked():
            self.rxes_panel.rb_mode_transfer.setChecked(True)
            return

        # 2) If user selects Constant Emission while in Transfer mode -> switch to Incident
        if self.rxes_panel.rb_extr_emission.isChecked() and self.rxes_panel.rb_mode_transfer.isChecked():
            self.rxes_panel.rb_mode_incident.setChecked(True)
            return

        # Update line orientation only (don’t autoscale the image)
        try:
            self.plot.set_signal_suppressed(True)
            self.set_line_orientation_for_current_mode()
        finally:
            try:
                self.plot.set_signal_suppressed(False)
            except Exception:
                pass

        # Recompute profiles for current extraction
        self.update_profiles()
    def set_line_orientation_for_current_mode(self):
        incident_mode = self.rxes_panel.rb_mode_incident.isChecked()
        if incident_mode:
            # Incident view (Ω vs ω)
            if self.rxes_panel.rb_extr_emission.isChecked():
                self.plot.set_line_orientation("horizontal")
            else:
                self.plot.set_line_orientation("vertical")
        else:
            # Transfer view (Ω vs Ω−ω)
            if self.rxes_panel.rb_extr_transfer.isChecked():
                self.plot.set_line_orientation("horizontal")
            else:
                self.plot.set_line_orientation("vertical")
    def on_rxes_channel_changed(self):
        # Rebuild RXES view if a scan is loaded
        if self.current_scan_number is not None:
            self.refresh_rxes_view()
        self.update_ui_state()
    def on_tab_changed(self, idx: int):
        """Handle tab changes between RXES (index 0) and XES (index 1)."""
        self.update_ui_state()
        
        # Switch plot stack based on active tab
        if hasattr(self, 'plot_stack'):
            if idx == 0:
                # RXES tab - show 2D plot
                self.plot_stack.setCurrentIndex(0)
            else:
                # XES tab - show 1D plot
                self.plot_stack.setCurrentIndex(1)
                # Refresh XES plot when switching to XES tab
                self._refresh_xes_plot()

    # ---------------- XES: multi-scan workflow - Load, Plot and Average ----------------
    def _load_xes_batch(self, paths):
        """Load files as XES (1D spectra, batch mode). Paths already selected."""
        
        ui_channel = "upper" if self.xes_panel.rb_upper.isChecked() else "lower"

        added = 0
        self._xes_list_bulk_updating = True
        blocker = QSignalBlocker(self.xes_panel.list)
        try:
            for path in paths:
                try:
                    ext = os.path.splitext(path)[1].lower()
                    if ext in (".nxs", ".h5", ".hdf", ".hdf5"):
                        # Skip raw detector files
                        if i20_loader.is_probably_detector_hdf(path):
                            QMessageBox.warning(self, "Load XES", f"Skipped detector HDF (not a scan): {path}")
                            continue
                        # Load into Scan once
                        sn = i20_loader.add_scan_from_nxs(self.scan, path)
                        entry = self.scan.get(sn, {})
                        avail = i20_loader.available_channels(entry)
                        if not avail:
                            raise ValueError("No usable detector channel in this file (missing emission/intensity).")

                        # Choose initial channel: UI choice if available, else the other
                        if ui_channel in avail:
                            ch0 = ui_channel
                        else:
                            ch0 = avail[0]

                        x, y = i20_loader.xes_from_scan_entry(entry, channel=ch0)
                        order = np.argsort(x)
                        x = np.asarray(x)[order]; y = np.asarray(y)[order]
                        ok = np.isfinite(x) & np.isfinite(y)
                        x, y = x[ok], y[ok]
                        if x.size == 0:
                            raise ValueError("Empty/invalid data after sanitizing")

                        item = {
                            "path": path,
                            "kind": "nxs",
                            "scan_number": sn,
                            "available_channels": avail,    # e.g. ['upper'] or ['upper','lower']
                            "channel": ch0,                 # currently shown channel
                            "x": x,
                            "y": y,
                            "label": os.path.basename(path),
                        }
                    else:
                        # ASCII: load once; channel notion not applicable
                        x, y = i20_loader.xes_from_ascii(path)
                        order = np.argsort(x)
                        x = np.asarray(x)[order]; y = np.asarray(y)[order]
                        ok = np.isfinite(x) & np.isfinite(y)
                        x, y = x[ok], y[ok]
                        if x.size == 0:
                            raise ValueError("Empty/invalid data after sanitizing")

                        item = {
                            "path": path,
                            "kind": "ascii",
                            "available_channels": [],       # none; not channel-switchable
                            "channel": None,
                            "x": x,
                            "y": y,
                            "label": os.path.basename(path),
                        }

                    self._xes_items.append(item)
                    self.xes_panel.add_item(item["label"], checked=True)
                    added += 1
                except Exception as e:
                    QMessageBox.warning(self, "Load XES", f"Failed to load {path}:\n{e}")
        finally:
            del blocker
            self._xes_list_bulk_updating = False

        self._update_xes_channel_controls() # Grey out unavailable channel buttons.
        if added:
            # Reset derived products
            self._xes_avg = None
            self._xes_avg_bkgsub = None
            self._xes_avg_norm_factor = None
            self.xes_panel.lbl_norm.setText("Average: no normalisation")
            # Refresh
            self._refresh_xes_plot()
            self.update_status_label()
            if hasattr(self, "_update_xes_buttons"):
                self._update_xes_buttons()
            self.status.showMessage(f"Loaded {added} XES scan(s)", 5000)

    def on_xes_load_scans(self):
        """XES panel: Load multiple scans (explicit XES loading)."""
        filters = [
            "XES spectrum (*.nxs *.h5 *.hdf *.hdf5 *.txt *.dat *.csv)",
            "NeXus/HDF5 (*.nxs *.h5 *.hdf *.hdf5)",
            "ASCII (*.txt *.dat *.csv)",
            "All files (*)"
        ]
        paths, _ = QFileDialog.getOpenFileNames(self, "Load XES scans", "", ";;".join(filters))
        if not paths:
            return
        
        self._load_xes_batch(paths)
    def on_xes_remove_selected(self):
        selected_rows = sorted({i.row() for i in self.xes_panel.list.selectedIndexes()}, reverse=True)
        for r in selected_rows:
            if 0 <= r < len(self._xes_items):
                self._xes_items.pop(r)
            try:
                self.xes_panel.list.takeItem(r)
            except Exception:
                pass
        self._xes_avg = None
        self._xes_avg_bkgsub = None
        self._xes_norm_target = None
        self._xes_avg_norm_factor = None
        self.xes_panel.lbl_norm.setText("Average: no normalisation")
        self._refresh_xes_plot()
        self.update_status_label()
        self._update_xes_buttons()
    def on_xes_list_item_changed(self, item):
        if getattr(self, "_xes_list_bulk_updating", False):
            return
        self._refresh_xes_plot()
    def on_xes_channel_changed(self, checked: bool = False):
        if not checked:
            return
        if not self._xes_items and not getattr(self, "_xes_wide", None):
            self.update_status_label()
            return

        new_channel = "upper" if self.xes_panel.rb_upper.isChecked() else "lower"
        changed = False
        errors = []

        for idx, it in enumerate(self._xes_items):
            if it.get("kind") != "nxs":
                continue  # ASCII: nothing to switch
            avail = it.get("available_channels", [])
            if new_channel not in avail:
                continue  # this file doesn't have that channel
            try:
                entry = self.scan.get(it["scan_number"], None)
                if entry is None:
                    raise RuntimeError("Scan entry missing")
                x, y = i20_loader.xes_from_scan_entry(entry, channel=new_channel)
                order = np.argsort(x)
                x = np.asarray(x)[order]; y = np.asarray(y)[order]
                ok = np.isfinite(x) & np.isfinite(y)
                x, y = x[ok], y[ok]
                self._xes_items[idx] = {**it, "x": x, "y": y, "channel": new_channel}
                changed = True
            except Exception as e:
                errors.append(f"{os.path.basename(it.get('path',''))}: {e}")

        if changed:
            # Invalidate derived products
            self._xes_avg = None
            self._xes_avg_bkgsub = None
            self._xes_avg_norm_factor = None
            self._last_bkg = None
            self._last_resid = None
            self._last_bkg_report = ""
            self.xes_panel.lbl_norm.setText("Average: no normalisation")
            # Remove specials if your panel supports it
            self._xes_list_bulk_updating = True
            blk = QSignalBlocker(self.xes_panel.list)
            try:
                if hasattr(self.xes_panel, "remove_special"):
                    self.xes_panel.remove_special(AVG_KEY)
                    self.xes_panel.remove_special(BKSUB_KEY)
            finally:
                del blk
                self._xes_list_bulk_updating = False

            self._refresh_xes_plot()
            self.update_status_label()
            if hasattr(self, "_update_xes_buttons"):
                self._update_xes_buttons()
                self._update_xes_channel_controls()
        if errors:
            try:
                QMessageBox.warning(self, "XES channel switch",
                                    "Some scans could not switch channel:\n" + "\n".join(errors))
            except Exception:
                pass  
    def on_xes_clear_all(self):
        self._xes_list_bulk_updating = True
        blocker = QSignalBlocker(self.xes_panel.list)
        try:
            self._xes_items.clear()
            self._xes_avg = None
            self._xes_avg_bkgsub = None
            self._xes_norm_target = None
            self._xes_avg_norm_factor = None
            self.xes_panel.clear_items()
            self.xes_panel.lbl_norm.setText("Average: no normalisation")
        finally:
            del blocker
            self._xes_list_bulk_updating = False
        self._refresh_xes_plot()
        self.update_status_label()
        self._update_xes_buttons()
        self._update_xes_channel_controls()
    def on_xes_average_selected(self):
        """
        Average the selected (ticked) XES scans.
        
        Creates a named average with a key like 'average_279496+279517' that
        includes the scan numbers of the averaged scans. Multiple averages
        can coexist in the list.
        """
        idxs = self.xes_panel.checked_indices()
        idxs = [i for i in idxs if 0 <= i < len(self._xes_items)]
        if not idxs:
            QMessageBox.information(self, "Average XES", "No scans ticked for averaging.")
            return
        xs = [self._xes_items[i]["x"] for i in idxs]
        ys = [self._xes_items[i]["y"] for i in idxs]
        xt, yt = self._regrid_and_average(xs, ys)
        if xt.size == 0:
            QMessageBox.warning(self, "Average", "No overlapping domain found to average.")
            return
        
        # Generate a unique key based on scan numbers
        scan_nums = []
        for i in idxs:
            item = self._xes_items[i]
            # Extract scan number from path or label
            label = item.get("label", "")
            # Try to extract scan number from label (e.g., "279496_upper")
            parts = label.split("_")
            if parts and parts[0].isdigit():
                scan_nums.append(parts[0])
            else:
                # Fallback: use index
                scan_nums.append(str(i))
        
        # Create key like "average_279496+279517"
        avg_key = MULTI_AVG_PREFIX + "+".join(scan_nums)
        avg_label = f"average ({'+'.join(scan_nums)})"
        
        # Store in multi-averages dict
        self._xes_multi_avgs[avg_key] = (xt, yt)
        
        # Also update legacy single average for backward compatibility
        self._xes_avg = (xt, yt)
        self._xes_avg_norm_factor = None
        self.xes_panel.lbl_norm.setText(f"Average [{'+'.join(scan_nums)}]: no normalisation")

        # Insert/replace special rows - use the new named average
        self._upsert_xes_special_row(avg_key, avg_label, checked=True)

        # Remove stale background-subtracted row
        self._xes_avg_bkgsub = None
        self._xes_list_bulk_updating = True
        blk = QSignalBlocker(self.xes_panel.list)
        try:
            self.xes_panel.remove_special(BKSUB_KEY)
        finally:
            del blk
            self._xes_list_bulk_updating = False

        # Sync dataset and plot
        self.dataset = DataSet("1D", x=xt, y=yt, xlabel="ω (eV)", ylabel="Intensity (XES)", source="")
        self._refresh_xes_plot()
        self.update_status_label()
        if hasattr(self, "_update_xes_buttons"):
            self._update_xes_buttons()
        self.status.showMessage(f"Created average: {avg_label}", 4000)
        logger.info(f"Created multi-scan average: {avg_key}")
    def _regrid_and_average(self, x_list: List[np.ndarray], y_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        if len(x_list) == 1:
            return x_list[0], y_list[0]
        try:
            lo = max([np.nanmin(x) for x in x_list if x.size])
            hi = min([np.nanmax(x) for x in x_list if x.size])
        except Exception:
            return np.array([]), np.array([])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            x0 = x_list[0]
            ys = [np.interp(x0, x, y, left=np.nan, right=np.nan) for x, y in zip(x_list, y_list)]
            Y = np.nanmean(np.vstack(ys), axis=0)
            ok = np.isfinite(x0) & np.isfinite(Y)
            return x0[ok], Y[ok]
        x0 = x_list[0]
        dx = np.diff(x0)
        step = float(np.nanmedian(dx[dx > 0])) if dx.size else max(0.1, (hi - lo) / 1000.0)
        xt = np.arange(lo, hi + 0.5 * step, step)
        Ys = [np.interp(xt, x, y, left=np.nan, right=np.nan) for x, y in zip(x_list, y_list)]
        Yt = np.nanmean(np.vstack(Ys), axis=0)
        ok = np.isfinite(xt) & np.isfinite(Yt)
        return xt[ok], Yt[ok]
    def on_xes_save_normalised(self):
        if not self._xes_avg_norm_factor or not self._xes_norm_target:
            QMessageBox.information(self, "Save normalised", "No normalised data to save.")
            return

        kind, idx = self._xes_norm_target
        if kind == "avg":
            if self._xes_avg is None:
                QMessageBox.information(self, "Save normalised", "No normalised average available.")
                return
            x, y = self._xes_avg
            default_name = "xes_normalised_avg.csv"
        else:
            if not (0 <= idx < len(self._xes_items)):
                QMessageBox.information(self, "Save normalised", "Normalised scan not available.")
                return
            it = self._xes_items[idx]
            x, y = it["x"], it["y"]
            base = os.path.basename(it.get("path", "")) or "scan"
            default_name = f"{os.path.splitext(base)[0]}_normalised.csv"

        path, _ = QFileDialog.getSaveFileName(self, "Save normalised XES", default_name,
                                            "CSV (*.csv);;All files (*)")
        if not path:
            return
        try:
            arr = np.column_stack([x, y])
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write("omega_eV,intensity_normalised\n")
                np.savetxt(fh, arr, delimiter=",", fmt="%.10g", comments="")
                # Commented last row with the normalisation factor
                fh.write(f"# normalisation_factor={self._xes_avg_norm_factor:.10g}\n")
            self.status.showMessage(f"Saved normalised XES: {path}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Save error", f"Failed to save normalised data:\n{e}")
    def on_xes_save_average(self):
        if self._xes_avg is None:
            QMessageBox.information(self, "Save average", "No averaged spectrum to save.")
            return
        x, y = self._xes_avg
        if x is None or y is None or len(x) == 0:
            QMessageBox.information(self, "Save average", "Average is empty.")
            return
        default_name = "xes_average.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Save XES average", default_name, "CSV (*.csv);;All files (*)")
        if not path:
            return
        try:
            out = np.column_stack([x, y])
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write("omega_eV,intensity\n")
                np.savetxt(fh, out, delimiter=",", fmt="%.10g", comments="")
            self.status.showMessage(f"Saved XES average: {path}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Save error", f"Failed to save average:\n{e}")
    def on_xes_save_overlays(self):
        if not self._xes_items:
            QMessageBox.information(self, "Save overlays", "No XES scans loaded.")
            return
        maxlen = max(len(it["x"]) for it in self._xes_items)
        cols = []
        headers = []
        for it in self._xes_items:
            x = it["x"]; y = it["y"]
            pad = maxlen - len(x)
            if pad > 0:
                x = np.pad(x, (0, pad), constant_values=np.nan)
                y = np.pad(y, (0, pad), constant_values=np.nan)
            cols.extend([x, y])
            headers.extend([f"X {it['label']}", f"Y {it['label']}"])
        out = np.column_stack(cols)
        default_name = "xes_overlays.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Save XES overlays", default_name, "CSV (*.csv);;All files (*)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(", ".join(headers) + "\n")
                np.savetxt(fh, out, delimiter=",", fmt="%.10g", comments="")
            self.status.showMessage(f"Saved XES overlays: {path}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Save error", f"Failed to save overlays:\n{e}")
  
    # ---------------- XES: multi-scan workflow - Normalisation ----------------
    def on_xes_normalise_average(self):
        """
        Use an external XES spectrum to select an area and normalise:
        - If an average exists, normalise the average.
        - Else if only one scan is loaded, normalise that scan.
        - Else prompt to average first.
        """
        # Decide what to normalise (average if present, otherwise single)
        target = None
        if self._xes_avg is not None and self._xes_avg[0].size and self._xes_avg[1].size:
            target = ("avg", (self._xes_avg[0], self._xes_avg[1]))
        elif len(self._xes_items) == 1:
            target = ("single", 0)

        if target is None:
            QMessageBox.information(self, "Normalise XES", "Average first (or load a single scan).")
            return

        # 1) Pick an external XES file
        filters = [
            "XES spectrum (*.nxs *.txt *.dat *.csv)",
            "NeXus (*.nxs)",
            "ASCII (*.txt *.dat *.csv)",
            "All files (*)"
        ]
        path, _ = QFileDialog.getOpenFileName(self, "Load XES spectrum for normalisation", "", ";;".join(filters))
        if not path:
            return

        try:
            # 2) Load it using the selected channel (Upper/Lower)
            # Check if cross-channel normalisation is enabled
            use_upper = self.xes_panel.rb_upper.isChecked()
            cross_channel = False
            if hasattr(self.xes_panel, 'chk_cross_channel'):
                cross_channel = self.xes_panel.chk_cross_channel.isChecked()
            
            # If cross-channel, use opposite channel for loading normalisation spectrum
            if cross_channel:
                # Use opposite channel
                norm_channel = "lower" if use_upper else "upper"
                norm_label = "Lower" if use_upper else "Upper"
                logger.info(f"Cross-channel normalisation: loading {norm_label} channel for normalisation")
            else:
                norm_channel = "upper" if use_upper else "lower"
                norm_label = "Upper" if use_upper else "Lower"
            
            channel = norm_channel
            x_ext, y_ext = i20_loader.xes_from_path(path, channel=channel, type="XES")

            # 3) Let the user select an area on that external spectrum
            title_suffix = f"{norm_label} channel"
            if cross_channel:
                title_suffix += " (cross-channel)"
            dlg = NormaliseDialog(x_ext, y_ext, parent=self, title=f"Select area (XES {title_suffix})")
            if dlg.exec() != QDialog.Accepted:
                return
            area = dlg.selected_area()
            if not np.isfinite(area) or area <= 0:
                QMessageBox.warning(self, "Normalise XES", "Selected area is invalid or non‑positive.")
                return

            # 4) Apply normalisation to the target
            if target[0] == "avg":
                x, y = target[1]
                y_norm = y / area
                self._xes_avg = (x, y_norm)
                self._xes_avg_norm_factor = float(area)
                self.dataset = DataSet("1D", x=x, y=y_norm, xlabel="ω (eV)", ylabel="Intensity / area", source="")
                self.xes_panel.lbl_norm.setText(f"Average normalised by area: {area:.6g}")
                self._xes_norm_target = ("avg", None)
            else:
                idx = target[1]
                item = self._xes_items[idx]
                y_norm = item["y"] / area
                self._xes_items[idx] = {**item, "y": y_norm}
                self._xes_avg = None
                self._xes_avg_norm_factor = float(area)
                self.dataset = DataSet("1D", x=item["x"], y=y_norm, xlabel="ω (eV)", ylabel="Intensity / area", source=item["path"])
                self.xes_panel.lbl_norm.setText(f"Scan normalised by area: {area:.6g}")
                self._xes_norm_target = ("single", idx)

            # 5) Refresh the overlays/average plot
            self._refresh_xes_plot()
            self.update_status_label()
            self.status.showMessage("XES normalisation applied", 4000)
            self._update_xes_buttons()

        except Exception as e:
            QMessageBox.critical(self, "XES normalise error", f"Failed to normalise XES:\n{path}\n\n{e}") 
    def on_xes_save_normalised_average(self):
        """
        Save the normalised average (requires that a normalisation factor has been applied).
        """
        if self._xes_avg is None or not self._xes_avg_norm_factor:
            QMessageBox.information(self, "Save normalised average", "No normalised average to save.")
            return
        x, y = self._xes_avg
        default_name = "xes_average_normalised.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Save XES normalised average", default_name, "CSV (*.csv);;All files (*)")
        if not path:
            return
        try:
            arr = np.column_stack([x, y])
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write("omega_eV,intensity_normalised\n")
                np.savetxt(fh, arr, delimiter=",", fmt="%.10g", comments="")
            self.status.showMessage(f"Saved XES normalised average: {path}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Save error", f"Failed to save normalised average:\n{e}")

    # ---------------- XES: multi-scan workflow - Background Subtraction ----------------
    def on_xes_load_wide(self):
        filters = [
            "XES spectrum (*.nxs *.txt *.dat *.csv)",
            "NeXus (*.nxs)",
            "ASCII (*.txt *.dat *.csv)",
            "All files (*)"
        ]
        path, _ = QFileDialog.getOpenFileName(self, "Load wide XES scan", "", ";;".join(filters))
        if not path:
            return
        try:
            use_upper = self.xes_panel.rb_upper.isChecked()
            channel = "upper" if use_upper else "lower"
            xw, yw = i20_loader.xes_from_path(path, channel=channel, type="XES")
            # Clean
            order = np.argsort(xw)
            xw = np.asarray(xw)[order]; yw = np.asarray(yw)[order]
            ok = np.isfinite(xw) & np.isfinite(yw)
            xw, yw = xw[ok], yw[ok]
            self._xes_wide = (xw, yw, path)
            self.xes_panel.lbl_wide.setText(f"Wide scan: {os.path.basename(path)} ({xw.size} pts)")
            self.status.showMessage("Loaded wide XES", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Load wide XES", f"Failed to load wide scan:\n{e}")
    def on_xes_background_extract(self):
        x_main = None; y_main = None
        if self._xes_avg is not None and self._xes_avg[0] is not None and self._xes_avg[0].size:
            x_main, y_main = self._xes_avg
        elif len(self._xes_items) == 1:
            x_main = self._xes_items[0]["x"]; y_main = self._xes_items[0]["y"]
        if x_main is None or y_main is None:
            QMessageBox.information(self, "Background Extraction", "Average first (or load a single scan).")
            return

        xw, yw = (None, None)
        if getattr(self, "_xes_wide", None):
            xw, yw, _ = self._xes_wide

        dlg = BackgroundDialog(x_main, y_main, x_wide=xw, y_wide=yw, parent=self,
                            title="Background Extraction (XES)")
        if dlg.exec() != QDialog.Accepted:
            return

        xfit, y_bg, y_res = dlg.result_arrays()
        self._last_bkg_report = ""
        try:
            self._last_bkg_report = dlg.get_fit_report()
        except Exception:
            try:
                self._last_bkg_report = dlg.report.toPlainText()
            except Exception:
                self._last_bkg_report = ""

        self._last_bkg = (xfit.copy(), y_bg.copy())
        self._last_resid = (xfit.copy(), y_res.copy())
        self._xes_avg_bkgsub = (xfit.copy(), y_res.copy())

        self._upsert_xes_special_row(BKSUB_KEY, "average_bksub", checked=True)

        self.dataset = DataSet("1D", x=xfit, y=y_res, xlabel="ω (eV)", ylabel="Residual (XES)", source="")
        self.plot.plot(self.dataset)
        self._update_bkg_buttons()
        self._refresh_xes_plot()
        self.status.showMessage("Background extracted (residual shown).", 4000)
    def on_xes_save_fit_log(self):
        if not self._last_bkg_report:
            QMessageBox.information(self, "Save fit log", "No fit log to save. Run Background Extraction first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save fit log", "bkg_fit_log.txt",
                                              "Text (*.txt);;All files (*)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(self._last_bkg_report)
            self.status.showMessage(f"Saved fit log: {path}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Save error", f"Failed to save fit log:\n{e}")
    def on_xes_save_bkg_data(self):
        if self._last_bkg is None or self._last_resid is None:
            QMessageBox.information(self, "Save background", "No background/residual to save. Run Background Extraction first.")
            return
        x_bg, y_bg = self._last_bkg
        x_rs, y_rs = self._last_resid
        if not np.array_equal(x_bg, x_rs):
            xs = np.intersect1d(x_bg, x_rs)
            if xs.size == 0:
                QMessageBox.warning(self, "Save background", "Could not align background and residual on a common grid.")
                return
            y_bg = np.interp(xs, x_bg, y_bg, left=np.nan, right=np.nan)
            y_rs = np.interp(xs, x_rs, y_rs, left=np.nan, right=np.nan)
            x_out = xs
        else:
            x_out = x_bg
        out = np.column_stack([x_out, y_bg, y_rs])
        path, _ = QFileDialog.getSaveFileName(self, "Save background/residual", "bkg_extracted.csv",
                                            "CSV (*.csv);;All files (*)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write("omega_eV,background,residual\n")
                np.savetxt(fh, out, delimiter=",", fmt="%.10g", comments="")
            self.status.showMessage(f"Saved background/residual: {path}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Save error", f"Failed to save background/residual:\n{e}")

    def _update_bkg_buttons(self):
        has_fit = bool(self._last_bkg_report) and (self._last_bkg is not None) and (self._last_resid is not None)
        try:
            self.xes_panel.btn_save_fit_log.setEnabled(bool(self._last_bkg_report))
            self.xes_panel.btn_save_bkg_data.setEnabled(has_fit)
        except Exception:
            pass

    # ---------------- XES: Plotting and Status Helpers ----------------
    def _upsert_xes_special_row(self, key: str, label: str, checked: bool = True):
        self._xes_list_bulk_updating = True
        blocker = QSignalBlocker(self.xes_panel.list)
        try:
            pos = -1
            for i in range(self.xes_panel.list.count()):
                it = self.xes_panel.list.item(i)
                if it.data(SPECIAL_ROLE) == key:
                    pos = i
                    break
            it_new = QListWidgetItem(label)
            it_new.setData(SPECIAL_ROLE, key)
            it_new.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            it_new.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
            f = it_new.font(); f.setItalic(True); it_new.setFont(f)
            if pos >= 0:
                self.xes_panel.list.takeItem(pos)
                self.xes_panel.list.insertItem(pos, it_new)
            else:
                self.xes_panel.list.addItem(it_new)
        finally:
            del blocker
            self._xes_list_bulk_updating = False
    def _set_special_row(self, key: str, label: str, checked: bool, line=None):
        # find existing row with this key
        pos = -1
        for i in range(self.xes_panel.list.count()):
            it = self.xes_panel.list.item(i)
            if it.data(SPECIAL_ROLE) == key:
                pos = i
                break

        it_new = QListWidgetItem(label)  # use the class you imported
        it_new.setData(SPECIAL_ROLE, key)
        # it_new.setData(LINE_ROLE, line)
        it_new.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        it_new.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        f = it_new.font(); f.setItalic(True); it_new.setFont(f)

        if pos >= 0:
            # replace in-place so row order is stable
            self.xes_panel.list.takeItem(pos)
            self.xes_panel.list.insertItem(pos, it_new)
        else:
            self.xes_panel.list.addItem(it_new)
    def _refresh_xes_plot(self):
        """
        Refresh the XES plot with all checked scans and averages.
        
        Supports multiple named averages (e.g., average_279496+279517).
        """
        curves = []
        show_average = False
        show_bkgsub = False
        checked_averages = []  # List of (key, label) for checked multi-averages
        real_idx = 0

        # Determine which specials are checked and collect real curves
        for row in range(self.xes_panel.list.count()):
            lit = self.xes_panel.list.item(row)
            key = lit.data(SPECIAL_ROLE)
            
            # Handle legacy single average
            if key == AVG_KEY:
                show_average = (lit.checkState() == Qt.CheckState.Checked)
                continue
            
            # Handle background-subtracted average
            if key == BKSUB_KEY:
                show_bkgsub = (lit.checkState() == Qt.CheckState.Checked)
                continue
            
            # Handle multi-averages (keys starting with "average_")
            if key and isinstance(key, str) and key.startswith(MULTI_AVG_PREFIX) and key != AVG_KEY:
                if lit.checkState() == Qt.CheckState.Checked:
                    label = lit.text()
                    checked_averages.append((key, label))
                continue
            
            # Regular scan item
            if real_idx >= len(self._xes_items):
                break
            if lit.checkState() == Qt.CheckState.Checked:
                item = self._xes_items[real_idx]
                curves.append({
                    "x": item["x"],
                    "y": item["y"],
                    "label": item["label"],
                    "alpha": 0.9,
                    "color": None
                })
            real_idx += 1

        # Special overlays - legacy single average
        avg = None
        if self._xes_avg is not None and show_average:
            avg = {"x": self._xes_avg[0], "y": self._xes_avg[1], "label": "Average (XES)"}
        
        # Add multiple averages as curves (with emphasis)
        for avg_key, avg_label in checked_averages:
            if avg_key in self._xes_multi_avgs:
                x, y = self._xes_multi_avgs[avg_key]
                curves.append({
                    "x": x, 
                    "y": y, 
                    "label": avg_label,
                    "alpha": 1.0,
                    "color": "black",  # Emphasize averages
                })
                # Use the last checked average as the "main" average for dataset
                if avg is None:
                    avg = {"x": x, "y": y, "label": avg_label}
        
        if self._xes_avg_bkgsub is not None and show_bkgsub:
            bx, by = self._xes_avg_bkgsub
            curves.append({"x": bx, "y": by, "label": "Average bkg-sub", "alpha": 1.0, "color": "tab:purple"})

        # Draw using the dedicated XES plot (Plot1D)
        if hasattr(self, 'xes_plot'):
            self.xes_plot.clear()
            self.xes_plot.setGraphTitle("XES scans (overlays)")
            
            # Plot individual curves
            for curve in curves:
                x = curve.get('x')
                y = curve.get('y')
                if x is None or y is None:
                    continue
                label = curve.get('label', 'Scan')
                color = curve.get('color', None)
                alpha = curve.get('alpha', 0.9)
                
                # Determine line width based on curve type
                linewidth = 2.5 if color == 'black' else 1.0
                
                self.xes_plot.addCurve(
                    x, y,
                    legend=label,
                    color=color,
                    linewidth=linewidth,
                )
            
            # Plot average with emphasis (if legacy single average)
            if avg is not None and not any(c.get('label') == avg.get('label') for c in curves):
                self.xes_plot.addCurve(
                    avg['x'], avg['y'],
                    legend=avg.get('label', 'Average'),
                    color='black',
                    linewidth=2.5
                )
            
            self.xes_plot.setGraphXLabel("Energy (eV)")
            self.xes_plot.setGraphYLabel("Intensity")
            self.xes_plot.resetZoom()
        else:
            # Fallback to old method
            try:
                self.plot.plot_xes_bundle(curves, avg=avg, title="XES scans (overlays)")
            except Exception:
                if avg is not None:
                    self.plot.plot(DataSet("1D", x=avg["x"], y=avg["y"], xlabel="ω (eV)", ylabel="Intensity (XES)"))
                else:
                    self.plot.plot(None)

        if avg is not None:
            self.dataset = DataSet("1D", x=avg["x"], y=avg["y"], xlabel="ω (eV)", ylabel="Intensity (XES)")

        self.update_status_label()
        if hasattr(self, "_update_xes_buttons"):
            self._update_xes_buttons()
    def update_status_label(self):
        if self.tabs.currentIndex() == 1 or (self.dataset and self.dataset.kind == "1D"):
            chan = "Upper" if self.xes_panel.rb_upper.isChecked() else "Lower"
            base = ""
            self.io_panel.status_label.setText(f"Channel: {chan} | Mode: XES (1D bundle) | File: {base}")
        else:
            self.update_ui_state()

    # ---------------- RXES: ROI profiles and Line Extraction ----------------
    def _build_coordinate_grids(self, ds: DataSet) -> Tuple[np.ndarray, np.ndarray]:
        Z = ds.z
        if ds.x2d is not None and ds.y2d is not None:
            return ds.x2d, ds.y2d
        elif ds.x is not None and ds.y is not None:
            X, Y = np.meshgrid(ds.x, ds.y)  # (ny, nx)
            return X, Y
        else:
            ny, nx = Z.shape
            X = np.arange(nx, dtype=float)[None, :].repeat(ny, axis=0)
            Y = np.arange(ny, dtype=float)[:, None].repeat(nx, axis=1)
            return X, Y
    def on_add_line(self):
        try:
            self.plot.add_line()
        except Exception:
            pass
        self.update_profiles()
    def on_remove_line(self):
        try:
            self.plot.remove_line()
        except Exception:
            pass
        self.update_profiles()
    def on_update_spectrum(self):
        self.update_profiles()
    def on_save_spectrum(self):
        # Ensure latest profiles
        self.update_profiles()
        if not self._last_profiles:
            QMessageBox.information(self, "No data", "No spectra to save. Add/move lines to generate profiles.")
            return
        # Build combined array as [X1,Y1,X2,Y2,...]
        maxlen = max(len(p[0]) for p in self._last_profiles)
        cols = []
        headers = []
        for (xv, yv, lbl) in self._last_profiles:
            pad = maxlen - len(xv)
            if pad > 0:
                xv = np.pad(xv, (0, pad), constant_values=np.nan)
                yv = np.pad(yv, (0, pad), constant_values=np.nan)
            cols.extend([xv, yv])
            headers.extend([f"X {lbl}", f"Y {lbl}"])
        out = np.column_stack(cols)
        default_name = "profiles.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Save spectra as CSV", default_name, "CSV (*.csv);;All files (*)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(", ".join(headers) + "\n")
                np.savetxt(fh, out, delimiter=",", comments="", fmt="%.10g")
            self.status.showMessage(f"Saved spectra: {path}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Save error", f"Failed to save spectra:\n{path}\n\n{e}")
    def on_bandwidth_changed(self, v: int):
        width_ev = self._get_bandwidth_ev()
        try:
            self.plot.set_bandwidth(width_ev)
        except Exception:
            pass
        self.update_profiles()
    def _get_bandwidth_ev(self) -> float:
        # Preferred path: let the panel do the scaling
        if hasattr(self.rxes_panel, "bandwidth_ev"):
            try:
                return float(self.rxes_panel.bandwidth_ev())
            except Exception:
                pass
        # Fallback: infer scaling from slider range
        try:
            sl = self.rxes_panel.sl_width
            v = float(sl.value())
            # If the slider max is > 3, it's the scaled 0.2 eV-step slider (1..15) -> divide by 5
            return v / 5.0 if sl.maximum() > 3 else v
        except Exception:
            return 1.0

    def _on_profile_extracted(self):
        """
        Handler for profile extraction in new RIXSWidget.
        
        In the new architecture, profiles are extracted via the silx profile
        toolbar directly. This handler is called when a profile is extracted
        and can be used for additional processing or logging.
        """
        logger.info("Profile extracted via silx toolbar")
        
        # Get the last extracted profile if needed
        if hasattr(self.plot, 'getLastProfile'):
            profile = self.plot.getLastProfile()
            if profile:
                x, y, label = profile
                logger.debug(f"Profile: {label}, {len(x)} points")
                
                # Store for potential saving
                self._last_profiles = [(x, y, label)]
        
        self.update_ui_state()

    def update_profiles(self):
        """
        Update profile extraction for old-style plot widget.
        
        In the new RIXSWidget architecture, profile extraction is handled
        automatically by the silx profile toolbar, so this method becomes
        a no-op when USE_RIXS_PLOT is True.
        """
        # Skip for new RIXS widget - profiles handled by silx toolbar
        if USE_RIXS_PLOT:
            logger.debug("update_profiles skipped (using silx profile toolbar)")
            return
        
        self._last_profiles = []
        if self.dataset is None or self.dataset.kind != "2D":
            return

        ds = self.dataset
        Z = np.asarray(ds.z, dtype=float)
        X2D, Y2D = self._build_coordinate_grids(ds)

        incident_view = (ds.x2d is None or ds.y2d is None)
        width_ev = self._get_bandwidth_ev()

        line_vals = []
        try:
            line_vals = self.plot.get_line_positions()
        except Exception:
            pass

        if len(line_vals) == 0:
            # No ROI lines: leave the image and its limits intact
            self._last_profiles = []
            try:
                if hasattr(self.plot, "clear_profiles"):
                    self.plot.clear_profiles()
            except Exception:
                pass
            self.update_ui_state()
            return

        curves: List[Tuple[np.ndarray, np.ndarray, str]] = []
        if self.rxes_panel.rb_extr_incident.isChecked():
            if incident_view:
                x_profile = ds.y if ds.y is not None else Y2D[:, 0]
                for val in line_vals:
                    mask = np.abs(X2D - val) <= (width_ev / 2.0)
                    y_profile = np.nansum(np.where(mask, Z, 0.0), axis=1)
                    curves.append((x_profile, y_profile, f"Ω={val:.2f} eV"))
                xlab = "ω (eV)"
            else:
                x_profile = np.nanmedian(Y2D, axis=1)  # Δ = Ω−ω
                for val in line_vals:
                    mask = np.abs(X2D - val) <= (width_ev / 2.0)
                    y_profile = np.nansum(np.where(mask, Z, 0.0), axis=1)
                    curves.append((x_profile, y_profile, f"Ω={val:.2f} eV"))
                xlab = "Ω − ω (eV)"
            try:
                self.plot.plot_profiles(xlab, curves)
            except Exception:
                pass
            self._last_profiles = curves

        elif self.rxes_panel.rb_extr_emission.isChecked():
            if not incident_view:
                return
            x_profile = ds.x if ds.x is not None else X2D[0, :]
            for val in line_vals:
                mask = np.abs(Y2D - val) <= (width_ev / 2.0)
                y_profile = np.nansum(np.where(mask, Z, 0.0), axis=0)
                curves.append((x_profile, y_profile, f"ω={val:.2f} eV"))
            try:
                self.plot.plot_profiles("Ω (eV)", curves)
            except Exception:
                pass
            self._last_profiles = curves

        else:
            # Constant Transfer (Ω−ω): transfer view only
            if incident_view:
                self.rxes_panel.rb_mode_transfer.setChecked(True)
                return
            x_profile = ds.x2d[0, :] if ds.x2d is not None else X2D[0, :]
            for val in line_vals:
                mask = np.abs(Y2D - val) <= (width_ev / 2.0)
                y_profile = np.nansum(np.where(mask, Z, 0.0), axis=0)
                curves.append((x_profile, y_profile, f"Δ={val:.2f} eV"))
            try:
                self.plot.plot_profiles("Ω (eV)", curves)
            except Exception:
                pass
            self._last_profiles = curves

        self.update_ui_state()

    # ---------------- RXES: Normalisation ----------------
    def on_rxes_normalise(self):
        """
        Load a 1D XES from file and normalise current RXES map by selected area.
        """
        if self.current_scan_number is None:
            QMessageBox.information(self, "RXES", "Load an RXES .nxs scan first.")
            return
        filters = ["XES spectrum (*.nxs *.txt *.dat *.csv)", "NeXus (*.nxs)", "ASCII (*.txt *.dat *.csv)", "All files (*)"]
        path, _ = QFileDialog.getOpenFileName(self, "Load XES spectrum", "", ";;".join(filters))
        if not path:
            return
        try:
            use_upper = self.rxes_panel.rb_upper.isChecked()
            channel = "upper" if use_upper else "lower"
            x, y = i20_loader.xes_from_path(path, channel=channel, type="XES")
            dlg = NormaliseDialog(x, y, parent=self, title=f"XES ({'Upper' if use_upper else 'Lower'})")
            if dlg.exec() != QDialog.Accepted:
                return
            area = dlg.selected_area()
            if not np.isfinite(area) or area <= 0:
                QMessageBox.warning(self, "Invalid area", "Selected XES area is not valid or non‑positive.")
                return
            self.scan[self.current_scan_number]["norm_factor"] = float(area)
            self.rxes_panel.lbl_norm.setText(f"Normalised by XES area: {area:.6g}")
            self.refresh_rxes_view()
        except Exception as e:
            QMessageBox.critical(self, "XES load error", f"Failed to load/normalise from XES:\n{path}\n\n{e}")

        # ---------------- Generic: Buttons & Toggles ----------------
    def _update_xes_buttons(self):
        has_avg = self._xes_avg is not None and len(self._xes_avg[0]) > 0
        self.xes_panel.btn_save_average.setEnabled(has_avg)
        has_norm_factor = bool(self._xes_avg_norm_factor)
        # Save Normalised (any target: avg or single)
        self.xes_panel.btn_save_norm.setEnabled(has_norm_factor and (self._xes_norm_target is not None))
        # Save Normalised Avg. only when the average exists and was normalised
        self.xes_panel.btn_save_avg_norm.setEnabled(has_avg and has_norm_factor)
    def _update_xes_channel_controls(self):
        """
        Inspect loaded XES items and update the channel radios:
        - If only 'upper' is available across all NeXus items, select Upper and disable Lower.
        - If only 'lower' is available, select Lower and disable Upper.
        - If both (or none, e.g. only ASCII items), enable both and leave selection as-is.
        """
        # Consider only NeXus items (ASCII has no channel concept)
        has_upper = any(
            it.get("kind") == "nxs" and "upper" in it.get("available_channels", [])
            for it in self._xes_items
        )
        has_lower = any(
            it.get("kind") == "nxs" and "lower" in it.get("available_channels", [])
            for it in self._xes_items
        )

        # Default: enable both (e.g. only ASCII loaded or both channels present)
        enable_upper = True
        enable_lower = True

        # If exactly one channel is available across all NeXus items, lock to it
        if has_upper and not has_lower:
            enable_lower = False
            if not self.xes_panel.rb_upper.isChecked():
                # Will trigger on_xes_channel_changed(checked=True)
                self.xes_panel.rb_upper.setChecked(True)
        elif has_lower and not has_upper:
            enable_upper = False
            if not self.xes_panel.rb_lower.isChecked():
                self.xes_panel.rb_lower.setChecked(True)

        # Apply enable states (greys out unavailable channel)
        self.xes_panel.rb_upper.setEnabled(enable_upper)
        self.xes_panel.rb_lower.setEnabled(enable_lower)

    # ---------------- Generic saves ----------------
    def on_save_ascii(self):
        if self.dataset is None:
            QMessageBox.information(self, "No data", "Nothing to save. Load/compute a dataset first.")
            return
        base = "spectrum.csv" if self.dataset.kind == "1D" else "data.csv"
        start = os.path.join(os.path.dirname(self.dataset.source), base) if self.dataset.source else base
        path, _ = QFileDialog.getSaveFileName(self, "Save as ASCII", start, "CSV (*.csv);;All files (*)")
        if not path:
            return
        try:
            io_mod.save_ascii(path, self.dataset)
            self.status.showMessage(f"Saved ASCII: {path}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Save error", f"Failed to save ASCII:\n{path}\n\n{e}")
    def on_save_nexus(self):
        if self.dataset is None:
            QMessageBox.information(self, "No data", "Nothing to save.")
            return
        try:
            if not io_mod.H5_AVAILABLE:
                QMessageBox.warning(self, "h5py missing", "Install h5py to save NeXus files: pip install h5py")
                return
        except Exception:
            pass
        base = "data.nxs"
        start = os.path.join(os.path.dirname(self.dataset.source), base) if self.dataset.source else base
        path, _ = QFileDialog.getSaveFileName(self, "Save as NeXus (HDF5)", start, "NeXus/HDF5 (*.nxs *.h5 *.hdf5);;All files (*)")
        if not path:
            return
        try:
            io_mod.save_nexus(path, self.dataset)
            self.status.showMessage(f"Saved NeXus: {path}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Save error", f"Failed to save NeXus:\n{path}\n\n{e}")

