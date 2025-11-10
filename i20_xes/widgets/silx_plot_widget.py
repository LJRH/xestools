"""
Silx-based plot widget for I20 XES/RXES visualization.

This module provides a professional visualization widget using the silx toolkit,
specifically designed for synchrotron data. It replaces the matplotlib-based
PlotWidget with enhanced features including:
- Built-in colorbar and colormap management
- Interactive zoom, pan, and aspect ratio controls
- Professional ROI management system
- Contour analysis capabilities
- Better memory management and performance

Author: OpenCode Assistant
Date: November 10, 2025
"""

import os
import logging
import weakref
from typing import List, Tuple, Optional

import numpy as np
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QSplitter, QMessageBox

from silx.gui.plot import ImageView, Plot1D
from silx.gui.plot.tools.roi import RegionOfInterestManager
from silx.gui.plot.items.roi import (
    HorizontalLineROI, VerticalLineROI, RectangleROI, PolygonROI
)

from i20_xes.modules.dataset import DataSet

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(name)s:%(lineno)d - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Color scheme for ROI lines (matching matplotlib version)
_LINE_COLORS = ["#ffffff", "#ffcc00", "#00ccff"]
_BAND_ALPHA = 0.15


class SilxPlotWidget(QWidget):
    """
    Enhanced plot widget using silx for 2D RXES and 1D XES visualization.
    
    Maintains API compatibility with matplotlib-based PlotWidget while providing
    professional synchrotron visualization tools.
    
    Signals:
        lines_changed: Emitted when ROI lines are modified (compatible with old API)
    """
    
    # Maintain signal name for compatibility
    lines_changed = Signal()
    
    def __init__(self, parent=None):
        """
        Initialize the silx plot widget.
        
        Args:
            parent: Parent QWidget (typically MainWindow)
        """
        super().__init__(parent)
        logger.info("Initializing SilxPlotWidget")
        
        # Store weak reference to parent
        self._parent_ref = weakref.ref(parent) if parent else None
        
        # Initialize state
        self._dataset: Optional[DataSet] = None
        self._line_orientation = "vertical"
        self._bandwidth = 1.0
        self._suppress_emit = False
        self._is_closing = False
        
        # Setup UI
        self._setup_ui()
        
        logger.info("SilxPlotWidget initialized successfully")
    
    def _setup_ui(self):
        """Create the UI layout with ImageView and Plot1D."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Vertical splitter for 2D image (top) and 1D profiles (bottom)
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Main 2D image viewer
        self.imageView = ImageView()
        self.imageView.setKeepDataAspectRatio(True)
        self.imageView.setColormap("viridis")
        
        # Enable interactive tools
        self.imageView.setGraphTitle("RXES Map")
        self.imageView.setGraphXLabel("Energy (eV)")
        self.imageView.setGraphYLabel("Energy (eV)")
        
        # 1D profile plot
        self.profilePlot = Plot1D()
        self.profilePlot.setGraphTitle("ROI Profiles")
        self.profilePlot.setGraphXLabel("Energy (eV)")
        self.profilePlot.setGraphYLabel("Integrated Intensity")
        self.profilePlot.setActiveCurveHandling(False)  # Don't highlight on click
        
        # Add to splitter with 3:1 ratio
        self.splitter.addWidget(self.imageView)
        self.splitter.addWidget(self.profilePlot)
        self.splitter.setStretchFactor(0, 3)  # 75% for image
        self.splitter.setStretchFactor(1, 1)  # 25% for profiles
        
        layout.addWidget(self.splitter)
        
        # ROI manager for the image view
        self.roiManager = RegionOfInterestManager(self.imageView)
        self.roiManager.sigRoiChanged.connect(self._on_roi_changed)
        
        logger.debug("UI setup complete")
    
    # ==================== API Compatibility Methods ====================
    # These methods maintain compatibility with the old PlotWidget API
    
    def plot(self, dataset: Optional[DataSet]):
        """
        Plot a dataset (API compatible with matplotlib version).
        
        Args:
            dataset: DataSet object containing 1D or 2D data
        """
        logger.info(f"Plotting dataset: {type(dataset)}")
        
        if self._is_closing:
            logger.debug("Widget is closing, skipping plot")
            return
        
        self._dataset = dataset
        self.clear()
        
        if dataset is None:
            logger.debug("Dataset is None, showing 'No data loaded'")
            self.imageView.setGraphTitle("No data loaded")
            return
        
        logger.info(f"Dataset kind: {dataset.kind}")
        
        if dataset.kind == "2D":
            self._plot_2d_dataset(dataset)
        elif dataset.kind == "1D":
            self._plot_1d_dataset(dataset)
        else:
            logger.warning(f"Unknown dataset kind: {dataset.kind}")
            self.imageView.setGraphTitle(f"Error: Unknown data type '{dataset.kind}'")
    
    def clear(self):
        """Clear all plots and ROIs."""
        logger.debug("Clearing plot widget")
        
        if self._is_closing:
            return
        
        # Clear image
        self.imageView.clear()
        
        # Clear profiles
        self.profilePlot.clear()
        
        # Clear ROIs
        self.roiManager.clear()
        
        logger.debug("Plot cleared")
    
    def set_signal_suppressed(self, state: bool):
        """Suppress or enable signal emission (for batch updates)."""
        self._suppress_emit = bool(state)
        logger.debug(f"Signal suppression: {self._suppress_emit}")
    
    def set_line_orientation(self, orientation: str):
        """
        Set ROI line orientation.
        
        Args:
            orientation: "vertical" or "horizontal"
        """
        assert orientation in ("vertical", "horizontal"), \
            f"Invalid orientation: {orientation}"
        
        logger.info(f"Setting line orientation: {orientation}")
        
        # If unchanged, just refresh
        if orientation == self._line_orientation:
            if len(self.roiManager.getRois()) == 0:
                self._create_initial_line()
            self._emit_lines_changed()
            return
        
        # Orientation changed: rebuild ROIs
        old_count = len(self.roiManager.getRois())
        self.roiManager.clear()
        self._line_orientation = orientation
        
        # Recreate same number of lines in new orientation
        for i in range(max(1, old_count)):
            if i == 0:
                self._create_initial_line()
            else:
                self._add_line_internal()
        
        self._emit_lines_changed()
        logger.info(f"Line orientation changed to: {orientation}")
    
    def set_bandwidth(self, width: float):
        """
        Set integration bandwidth for ROI lines.
        
        Args:
            width: Bandwidth in data units (eV)
        """
        self._bandwidth = max(0.0, float(width))
        logger.debug(f"Bandwidth set to: {self._bandwidth} eV")
        self._update_bandwidth_visuals()
        self._emit_lines_changed()
    
    def add_line(self):
        """Add a new ROI line."""
        rois = self.roiManager.getRois()
        # Filter to actual line ROIs (exclude bandwidth visuals)
        line_rois = [r for r in rois if isinstance(r, (VerticalLineROI, HorizontalLineROI))
                     and not r.getName().endswith("_band")]
        
        if len(line_rois) >= 3:
            logger.warning("Maximum 3 ROI lines reached")
            return
        
        if len(line_rois) == 0:
            self._create_initial_line()
        else:
            self._add_line_internal()
        
        self._update_bandwidth_visuals()
        self._emit_lines_changed()
        logger.info(f"ROI line added, total: {len(line_rois) + 1}")
    
    def remove_line(self):
        """Remove the last ROI line."""
        rois = self.roiManager.getRois()
        line_rois = [r for r in rois if isinstance(r, (VerticalLineROI, HorizontalLineROI))
                     and not r.getName().endswith("_band")]
        
        if len(line_rois) == 0:
            return
        
        # Remove last line and its bandwidth visual
        last_roi = line_rois[-1]
        self.roiManager.removeRoi(last_roi)
        
        # Remove associated bandwidth visual
        band_name = f"{last_roi.getName()}_band"
        band_roi = self._get_roi_by_name(band_name)
        if band_roi:
            self.roiManager.removeRoi(band_roi)
        
        self._emit_lines_changed()
        logger.info(f"ROI line removed, remaining: {len(line_rois) - 1}")
    
    def ensure_line_count(self, n: int):
        """
        Ensure exactly n ROI lines exist.
        
        Args:
            n: Desired number of lines (0-3)
        """
        n = max(0, min(3, n))
        
        rois = self.roiManager.getRois()
        line_rois = [r for r in rois if isinstance(r, (VerticalLineROI, HorizontalLineROI))
                     and not r.getName().endswith("_band")]
        
        current = len(line_rois)
        
        # Add lines if needed
        while current < n:
            if current == 0:
                self._create_initial_line()
            else:
                self._add_line_internal()
            current += 1
        
        # Remove lines if needed
        while current > n:
            self.remove_line()
            current -= 1
        
        self._update_bandwidth_visuals()
        self._emit_lines_changed()
        logger.debug(f"Line count set to: {n}")
    
    def get_line_positions(self) -> List[float]:
        """
        Get positions of all ROI lines.
        
        Returns:
            List of line positions in data coordinates
        """
        positions = []
        rois = self.roiManager.getRois()
        line_rois = [r for r in rois if isinstance(r, (VerticalLineROI, HorizontalLineROI))
                     and not r.getName().endswith("_band")]
        
        for roi in line_rois:
            pos = roi.getPosition()
            positions.append(pos)
        
        logger.debug(f"Retrieved {len(positions)} line positions")
        return positions
    
    def set_line_positions(self, positions: List[float]):
        """
        Set positions of ROI lines.
        
        Args:
            positions: List of positions in data coordinates
        """
        self.ensure_line_count(len(positions))
        
        rois = self.roiManager.getRois()
        line_rois = [r for r in rois if isinstance(r, (VerticalLineROI, HorizontalLineROI))
                     and not r.getName().endswith("_band")]
        
        for roi, pos in zip(line_rois, positions):
            roi.setPosition(pos)
        
        self._update_bandwidth_visuals()
        self._emit_lines_changed()
        logger.debug(f"Set {len(positions)} line positions")
    
    def plot_profiles(self, x_label: str, curves: List[Tuple[np.ndarray, np.ndarray, str]]):
        """
        Plot extracted profiles in the bottom panel.
        
        Args:
            x_label: Label for X axis
            curves: List of (x, y, label) tuples for each profile
        """
        logger.debug(f"Plotting {len(curves)} profiles")
        
        self.profilePlot.clear()
        
        for i, (x, y, label) in enumerate(curves):
            if x is not None and y is not None and len(x) > 0:
                # Use ROI colors
                color_idx = min(i, len(_LINE_COLORS) - 1)
                color = _LINE_COLORS[color_idx]
                
                self.profilePlot.addCurve(
                    x, y,
                    legend=label,
                    color=color,
                    linewidth=2.0
                )
        
        self.profilePlot.setGraphXLabel(x_label)
        self.profilePlot.setGraphYLabel("Integrated Intensity")
        self.profilePlot.replot()
        
        logger.debug(f"Profiles plotted: {len(curves)}")
    
    def plot_xes_bundle(self, curves, avg=None, title="XES bundle"):
        """
        Plot XES overlay (1D spectra bundle).
        
        Args:
            curves: List of curve dicts with keys: x, y, label, alpha, color
            avg: Average curve dict (optional)
            title: Plot title
        """
        logger.info(f"Plotting XES bundle with {len(curves) if curves else 0} curves")
        
        # Hide image view, show only profiles in full height
        self.imageView.setVisible(False)
        self.profilePlot.setVisible(True)
        
        self.profilePlot.clear()
        self.profilePlot.setGraphTitle(title)
        
        # Add individual curves
        for i, curve in enumerate(curves or []):
            x = curve.get("x")
            y = curve.get("y")
            if x is None or y is None:
                continue
            
            self.profilePlot.addCurve(
                x, y,
                legend=curve.get("label", f"Curve {i}"),
                color=curve.get("color"),
                linewidth=1.0,
                linestyle='-'
            )
        
        # Add average with emphasis
        if avg is not None and avg.get("x") is not None:
            self.profilePlot.addCurve(
                avg["x"], avg["y"],
                legend=avg.get("label", "Average"),
                color="black",
                linewidth=2.5
            )
        
        self.profilePlot.setGraphXLabel("ω (eV)")
        self.profilePlot.setGraphYLabel("Intensity (XES)")
        self.profilePlot.replot()
        
        logger.info("XES bundle plotted")
    
    # ==================== Internal Methods ====================
    
    def _plot_2d_dataset(self, dataset: DataSet):
        """Plot 2D RXES map."""
        logger.info("Plotting 2D RXES dataset")
        
        # Show both panels for RXES
        self.imageView.setVisible(True)
        self.profilePlot.setVisible(True)
        
        Z = dataset.z
        if Z is None:
            raise ValueError("No Z data for 2D plot")
        
        logger.debug(f"2D data shape: {Z.shape}")
        
        # Set image data with proper axis scaling
        if dataset.x2d is not None and dataset.y2d is not None:
            # Curvilinear grid (energy transfer mode)
            # For now, use simple origin/scale approximation
            # TODO: Full curvilinear support
            origin = (float(dataset.x2d[0, 0]), float(dataset.y2d[0, 0]))
            x_range = float(dataset.x2d[0, -1] - dataset.x2d[0, 0])
            y_range = float(dataset.y2d[-1, 0] - dataset.y2d[0, 0])
            scale = (x_range / Z.shape[1], y_range / Z.shape[0])
            
            logger.debug(f"Curvilinear grid: origin={origin}, scale={scale}")
            self.imageView.setImage(Z, origin=origin, scale=scale, copy=False)
        else:
            # Regular grid (incident energy mode)
            if dataset.x is not None and dataset.y is not None:
                x_min, x_max = float(dataset.x.min()), float(dataset.x.max())
                y_min, y_max = float(dataset.y.min()), float(dataset.y.max())
                origin = (x_min, y_min)
                scale = ((x_max - x_min) / Z.shape[1], (y_max - y_min) / Z.shape[0])
                
                logger.debug(f"Regular grid: origin={origin}, scale={scale}")
                self.imageView.setImage(Z, origin=origin, scale=scale, copy=False)
            else:
                # No axis info, use pixel coordinates
                logger.debug("No axis information, using pixel coordinates")
                self.imageView.setImage(Z, copy=False)
        
        # Set labels
        self.imageView.setGraphXLabel(dataset.xlabel or "X")
        self.imageView.setGraphYLabel(dataset.ylabel or "Y")
        
        title = os.path.basename(dataset.source) if dataset.source else "RXES Map"
        self.imageView.setGraphTitle(title)
        
        # Colorbar label (silx manages it automatically!)
        colorbar = self.imageView.getColorBarWidget()
        if colorbar and dataset.zlabel:
            # Note: silx 2.x doesn't have direct colorbar label setting
            # The label is part of the plot title/info
            pass
        
        # Ensure at least one ROI line exists
        if len(self.roiManager.getRois()) == 0:
            self._create_initial_line()
        
        logger.info("2D RXES dataset plotted successfully")
    
    def _plot_1d_dataset(self, dataset: DataSet):
        """Plot 1D spectrum (XES mode)."""
        logger.info("Plotting 1D dataset")
        
        # Hide image view, show only profiles
        self.imageView.setVisible(False)
        self.profilePlot.setVisible(True)
        
        y = dataset.y
        if y is None:
            raise ValueError("Invalid y data for 1D plot")
        
        x = dataset.x if dataset.x is not None else np.arange(len(y))
        
        # Validate shapes
        if len(x) != len(y):
            logger.warning(f"X and Y length mismatch: {len(x)} vs {len(y)}")
            min_len = min(len(x), len(y))
            x = x[:min_len]
            y = y[:min_len]
        
        self.profilePlot.clear()
        self.profilePlot.addCurve(x, y, legend="Spectrum", linewidth=1.5)
        self.profilePlot.setGraphXLabel(dataset.xlabel or "Energy (eV)")
        self.profilePlot.setGraphYLabel(dataset.ylabel or "Intensity")
        
        title = os.path.basename(dataset.source) if dataset.source else "Spectrum"
        self.profilePlot.setGraphTitle(title)
        self.profilePlot.replot()
        
        logger.info("1D dataset plotted successfully")
    
    def _create_initial_line(self):
        """Create the first ROI line at center of image."""
        logger.debug(f"Creating initial ROI line, orientation: {self._line_orientation}")
        
        # Get current image bounds
        bounds = self._get_image_bounds()
        if bounds is None:
            logger.warning("No image bounds available, cannot create ROI")
            return
        
        x_min, x_max, y_min, y_max = bounds
        
        if self._line_orientation == "vertical":
            pos = x_min + 0.5 * (x_max - x_min)
            roi = VerticalLineROI()
        else:
            pos = y_min + 0.5 * (y_max - y_min)
            roi = HorizontalLineROI()
        
        roi.setName("ROI 1")
        roi.setPosition(pos)
        roi.setColor(_LINE_COLORS[0])
        roi.setLineWidth(2)
        roi.setSelectable(True)
        roi.setEditable(True)
        
        self.roiManager.addRoi(roi)
        roi.sigRegionChanged.connect(self._on_roi_changed)
        
        logger.debug("Initial ROI line created")
    
    def _add_line_internal(self):
        """Add another ROI line (internal, no signal emit)."""
        rois = self.roiManager.getRois()
        line_rois = [r for r in rois if isinstance(r, (VerticalLineROI, HorizontalLineROI))
                     and not r.getName().endswith("_band")]
        
        idx = len(line_rois)
        if idx >= 3:
            return
        
        color_idx = min(idx, len(_LINE_COLORS) - 1)
        color = _LINE_COLORS[color_idx]
        
        # Get image bounds
        bounds = self._get_image_bounds()
        if bounds is None:
            return
        
        x_min, x_max, y_min, y_max = bounds
        
        if self._line_orientation == "vertical":
            # Position at 1/3 or 2/3
            frac = 0.33 if idx == 1 else 0.66
            pos = x_min + frac * (x_max - x_min)
            roi = VerticalLineROI()
        else:
            frac = 0.33 if idx == 1 else 0.66
            pos = y_min + frac * (y_max - y_min)
            roi = HorizontalLineROI()
        
        roi.setName(f"ROI {idx + 1}")
        roi.setPosition(pos)
        roi.setColor(color)
        roi.setLineWidth(2)
        roi.setLinestyle('--')  # Dashed for subsequent lines
        roi.setSelectable(True)
        roi.setEditable(True)
        
        self.roiManager.addRoi(roi)
        roi.sigRegionChanged.connect(self._on_roi_changed)
        
        logger.debug(f"Added ROI line {idx + 1}")
    
    def _update_bandwidth_visuals(self):
        """
        Update bandwidth visualization rectangles around ROI lines.
        
        Creates semi-transparent rectangles showing the integration region.
        """
        # Remove old bandwidth visuals
        rois = list(self.roiManager.getRois())
        for roi in rois:
            if roi.getName().endswith("_band"):
                self.roiManager.removeRoi(roi)
        
        if self._bandwidth <= 0:
            return
        
        # Get line ROIs
        line_rois = [r for r in self.roiManager.getRois() 
                     if isinstance(r, (VerticalLineROI, HorizontalLineROI))]
        
        bounds = self._get_image_bounds()
        if bounds is None:
            return
        
        x_min, x_max, y_min, y_max = bounds
        half = self._bandwidth / 2.0
        
        for i, line_roi in enumerate(line_rois):
            color_idx = min(i, len(_LINE_COLORS) - 1)
            color = _LINE_COLORS[color_idx]
            
            pos = line_roi.getPosition()
            
            # Create rectangle ROI for bandwidth visualization
            band_roi = RectangleROI()
            band_roi.setName(f"{line_roi.getName()}_band")
            band_roi.setEditable(False)
            band_roi.setSelectable(False)
            
            if isinstance(line_roi, VerticalLineROI):
                # Vertical line: band spans Y, centered on X
                origin = (pos - half, y_min)
                size = (self._bandwidth, y_max - y_min)
            else:
                # Horizontal line: band spans X, centered on Y
                origin = (x_min, pos - half)
                size = (x_max - x_min, self._bandwidth)
            
            band_roi.setGeometry(origin=origin, size=size)
            band_roi.setColor(color)
            band_roi.setLineWidth(0)  # No outline
            
            # Set transparency (silx uses alpha in color)
            # Note: This may vary by silx version
            try:
                band_roi.setAlpha(_BAND_ALPHA)
            except AttributeError:
                # Fallback for older silx versions
                pass
            
            self.roiManager.addRoi(band_roi)
        
        logger.debug(f"Updated {len(line_rois)} bandwidth visuals")
    
    def _get_image_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """
        Get current image bounds in data coordinates.
        
        Returns:
            Tuple of (x_min, x_max, y_min, y_max) or None if no image
        """
        plot = self.imageView.getPlot()
        if plot is None:
            return None
        
        image = self.imageView.getImage()
        if image is None:
            return None
        
        # Get axis limits
        x_limits = plot.getGraphXLimits()
        y_limits = plot.getGraphYLimits()
        
        return (x_limits[0], x_limits[1], y_limits[0], y_limits[1])
    
    def _get_roi_by_name(self, name: str):
        """Get ROI by name."""
        for roi in self.roiManager.getRois():
            if roi.getName() == name:
                return roi
        return None
    
    def _on_roi_changed(self):
        """Handle ROI changed signal from silx."""
        logger.debug("ROI changed")
        
        # Update bandwidth visuals
        self._update_bandwidth_visuals()
        
        # Emit compatible signal
        self._emit_lines_changed()
    
    def _emit_lines_changed(self):
        """Emit lines_changed signal if not suppressed."""
        if not self._suppress_emit and not self._is_closing:
            logger.debug("Emitting lines_changed signal")
            self.lines_changed.emit()
    
    def closeEvent(self, event):
        """Handle widget closure."""
        logger.info("SilxPlotWidget closeEvent")
        
        try:
            self._is_closing = True
            self.roiManager.clear()
            self.imageView.clear()
            self.profilePlot.clear()
            logger.info("SilxPlotWidget cleanup complete")
        except Exception as e:
            logger.error(f"Error during closeEvent: {e}")
        finally:
            if hasattr(event, 'accept'):
                event.accept()
            super().closeEvent(event)
