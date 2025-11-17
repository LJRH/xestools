#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RXES Widget - Main Visualization Widget
========================================

A complete RXES/RIXS visualization widget that integrates:
- XESRxesPlot2D for 2D map display
- Profile window for extracted line cuts
- Built-in silx profile extraction tools
- Energy transfer mode switching

This widget provides a drop-in replacement for the old SilxPlotWidget/PlotWidget
with enhanced capabilities based on the xraylarch architecture.

Author: Luke Higgins / OpenCode Assistant
Date: November 2025
"""

import logging
import os
from typing import Optional, List, Tuple, Any

import numpy as np
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QSplitter, QStatusBar

from silx.gui.plot import Plot1D
from silx.gui.plot.tools.profile.manager import ProfileWindow

from .rxes_plot import XESRxesPlot2D
from xestools.modules.dataset import DataSet

logger = logging.getLogger(__name__)


class RXESWidget(QWidget):
    """
    Comprehensive RXES/RIXS visualization widget.
    
    This widget provides:
    - 2D RXES map display with proper energy axis scaling
    - Built-in profile extraction toolbar (horizontal, vertical, diagonal cuts)
    - 1D profile display with multiple overlays
    - Energy transfer mode switching
    - Integration with xestools DataSet objects
    
    The widget uses a vertical splitter layout:
    - Top: 2D RXES map (XESRxesPlot2D)
    - Bottom: 1D profile plot (Plot1D)
    
    Signals:
        data_loaded: Emitted when new data is loaded
        profile_extracted: Emitted when a profile is extracted
    
    This replaces the old SilxPlotWidget while maintaining some API compatibility.
    """
    
    # Signals for integration with main window
    data_loaded = Signal()
    profile_extracted = Signal()
    
    def __init__(self, parent: Any = None):
        """
        Initialize the RXES widget.
        
        Args:
            parent: Parent QWidget (typically MainWindow)
        """
        super().__init__(parent)
        
        self._dataset: Optional[DataSet] = None
        self._is_closing = False
        
        self._setup_ui()
        
        logger.info("RXESWidget initialized")
    
    def _setup_ui(self):
        """
        Create the UI layout with RXES plot.
        
        Layout:
        - Single XESRxesPlot2D widget that fills the entire area
        - Profile extraction is handled by the built-in silx profile toolbar
        - Profiles appear in a separate window managed by silx
        
        This simplified layout removes the manual profile subplot since
        silx's ProfileToolBar handles profile display automatically.
        """
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create 2D RXES plot (profile window will be created internally by silx)
        # The profile toolbar will manage its own profile display window
        self.rxesPlot = XESRxesPlot2D(
            parent=self,
            profileWindow=None,  # Let silx manage the profile window
            title="RXES Map"
        )
        
        # Add plot directly to layout (no splitter needed)
        layout.addWidget(self.rxesPlot)
        
        # Keep profilePlot reference for backward compatibility (points to silx-managed window)
        self.profilePlot = self.rxesPlot.getProfileWindow()
        
        logger.debug("RXES widget UI setup complete (single plot, no splitter)")
    
    def plot(self, dataset: Optional[DataSet]):
        """
        Plot a dataset (2D RXES map or 1D spectrum).
        
        This is the main method for displaying data. It handles both
        2D RXES maps and 1D XES spectra.
        
        Args:
            dataset: DataSet object containing the data to display
        """
        if self._is_closing:
            logger.debug("Widget closing, skipping plot")
            return
        
        self._dataset = dataset
        
        if dataset is None:
            self.clear()
            self.rxesPlot.setGraphTitle("No data loaded")
            return
        
        logger.info(f"Plotting dataset: kind={dataset.kind}")
        
        if dataset.kind == "2D":
            self._plot_rxes_map(dataset)
        elif dataset.kind == "1D":
            self._plot_1d_spectrum(dataset)
        else:
            logger.warning(f"Unknown dataset kind: {dataset.kind}")
            self.rxesPlot.setGraphTitle(f"Error: Unknown data type '{dataset.kind}'")
        
        self.data_loaded.emit()
    
    def _plot_rxes_map(self, dataset: DataSet):
        """
        Display a 2D RXES map.
        
        Args:
            dataset: DataSet with 2D z-data
        """
        logger.info("Plotting 2D RXES map")
        
        # Show the 2D plot (profile window managed by silx toolbar)
        self.rxesPlot.setVisible(True)
        
        Z = dataset.z
        if Z is None:
            logger.error("No Z data for 2D plot")
            return
        
        # Don't force aspect ratio - let data fill the window
        # This is important when X and Y ranges are very different
        self.rxesPlot.setKeepDataAspectRatio(False)
        
        # Determine axis coordinates
        if dataset.x2d is not None and dataset.y2d is not None:
            # Curvilinear grid (energy transfer mode)
            # Use first row for X, first column for Y
            incident = dataset.x2d[0, :]  # Incident energies (Ω)
            emission = dataset.y2d[:, 0]   # Emission energies (ω)
        else:
            # Regular grid
            incident = dataset.x if dataset.x is not None else np.arange(Z.shape[1])
            emission = dataset.y if dataset.y is not None else np.arange(Z.shape[0])
        
        # Get title from source file
        title = os.path.basename(dataset.source) if dataset.source else "RXES Map"
        
        # Use custom method for proper scaling
        self.rxesPlot.addRXESImage(
            data=Z,
            incident_energy=incident,
            emission_energy=emission,
            title=title,
            xlabel=dataset.xlabel or "Incident Energy Ω (eV)",
            ylabel=dataset.ylabel or "Emission Energy ω (eV)"
        )
        
        logger.info(f"RXES map plotted: {Z.shape}")
    
    def _plot_1d_spectrum(self, dataset: DataSet):
        """
        Display a 1D spectrum (XES mode).
        
        In 1D mode, the spectrum is displayed as a curve on the profile plot.
        The rxesPlot is cleared but stays visible with the 1D data shown.
        
        Args:
            dataset: DataSet with 1D y-data
        """
        logger.info("Plotting 1D spectrum")
        
        y = dataset.y
        if y is None:
            logger.error("No Y data for 1D plot")
            return
        
        x = dataset.x if dataset.x is not None else np.arange(len(y))
        
        # Clear any existing image from 2D plot
        self.rxesPlot.clear()
        
        # Disable aspect ratio lock for 1D data (it doesn't make sense for curves)
        self.rxesPlot.setKeepDataAspectRatio(False)
        
        # Add spectrum as a curve to the 2D plot (which can display curves too)
        title = os.path.basename(dataset.source) if dataset.source else "Spectrum"
        
        self.rxesPlot.addCurve(
            x, y,
            legend=title,
            linewidth=1.5,
            color='#1F77B4'
        )
        
        self.rxesPlot.setGraphTitle(title)
        self.rxesPlot.setGraphXLabel(dataset.xlabel or "Energy (eV)")
        self.rxesPlot.setGraphYLabel(dataset.ylabel or "Intensity")
        
        # Reset zoom to show all data
        self.rxesPlot.resetZoom()
        
        logger.info(f"1D spectrum plotted: {len(y)} points")
    
    def clear(self):
        """
        Clear all plots.
        
        Removes images, curves, and profiles from both the 2D and 1D plots.
        """
        if self._is_closing:
            return
        
        logger.debug("Clearing RXES widget")
        
        # Clear 2D plot
        self.rxesPlot.clear()
        
        # Clear profile window (silx ProfileWindow wraps a Plot1D internally)
        if self.profilePlot is not None:
            # ProfileWindow has getPlot1D() method to access internal plot
            plot1d = self.profilePlot.getPlot1D(init=False)
            if plot1d is not None:
                plot1d.clear()
        
        # Reset profile toolbar history
        if hasattr(self.rxesPlot, 'profile'):
            toolbar = self.rxesPlot.profile
            if hasattr(toolbar, '_clearAllProfiles'):
                toolbar._clearAllProfiles()
    
    def setEnergyTransferMode(self, enabled: bool):
        """
        Switch between incident energy and energy transfer display modes.
        
        In energy transfer mode, the Y-axis shows Ω - ω instead of ω.
        
        Args:
            enabled: True for energy transfer mode, False for incident energy mode
        """
        self.rxesPlot.setEnergyTransferMode(enabled)
        
        if self._dataset is not None and self._dataset.kind == "2D":
            # Re-plot with new coordinate system
            # This would require data transformation
            logger.info(f"Energy transfer mode: {enabled}")
    
    def getProfileToolbar(self):
        """
        Get the profile extraction toolbar.
        
        Returns:
            ProfileToolBar instance (silx standard toolbar)
        """
        return self.rxesPlot.profile
    
    def get2DPlot(self) -> XESRxesPlot2D:
        """
        Get the 2D RXES plot widget.
        
        Returns:
            XESRxesPlot2D instance
        """
        return self.rxesPlot
    
    def getProfilePlot(self) -> ProfileWindow:
        """
        Get the profile window widget.
        
        Returns:
            ProfileWindow instance (silx profile display window)
        """
        return self.profilePlot
    
    def getLastProfile(self) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
        """
        Get the most recently extracted profile.
        
        Returns:
            Tuple of (x, y, label) or None if no profiles
        """
        toolbar = self.getProfileToolbar()
        if hasattr(toolbar, 'getProfileHistory'):
            history = toolbar.getProfileHistory()
            if history:
                return history[-1]
        return None
    
    def getAllProfiles(self) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        """
        Get all extracted profiles.
        
        Returns:
            List of (x, y, label) tuples
        """
        toolbar = self.getProfileToolbar()
        if hasattr(toolbar, 'getProfileHistory'):
            return toolbar.getProfileHistory()
        return []
    
    def plot_xes_bundle(self, curves: List[dict], avg: Optional[dict] = None, 
                       title: str = "XES Overlay"):
        """
        Plot multiple XES spectra overlaid (for XES averaging workflow).
        
        This method displays multiple 1D curves on the rxesPlot,
        useful for comparing scans before averaging.
        
        Args:
            curves: List of dicts with keys: x, y, label, color (optional)
            avg: Optional average curve dict
            title: Plot title
        """
        logger.info(f"Plotting XES bundle: {len(curves)} curves")
        
        # Clear and prepare plot for 1D curves
        self.rxesPlot.clear()
        # Disable aspect ratio for 1D curve bundles
        self.rxesPlot.setKeepDataAspectRatio(False)
        self.rxesPlot.setGraphTitle(title)
        
        # Plot individual curves
        for i, curve in enumerate(curves or []):
            x = curve.get('x')
            y = curve.get('y')
            if x is None or y is None:
                continue
            
            label = curve.get('label', f'Scan {i+1}')
            color = curve.get('color', None)
            
            self.rxesPlot.addCurve(
                x, y,
                legend=label,
                color=color,
                linewidth=1.0
            )
        
        # Plot average with emphasis
        if avg is not None:
            x = avg.get('x')
            y = avg.get('y')
            if x is not None and y is not None:
                self.rxesPlot.addCurve(
                    x, y,
                    legend=avg.get('label', 'Average'),
                    color='black',
                    linewidth=2.5
                )
        
        self.rxesPlot.setGraphXLabel("Energy (eV)")
        self.rxesPlot.setGraphYLabel("Intensity")
        self.rxesPlot.resetZoom()
    
    def autoscale_current(self):
        """
        Reset zoom to fit all data.
        
        This resets the view to show the full data extent in both plots.
        """
        self.rxesPlot.resetZoom()
        if self.profilePlot is not None:
            # ProfileWindow wraps Plot1D internally
            plot1d = self.profilePlot.getPlot1D(init=False)
            if plot1d is not None:
                plot1d.resetZoom()
        logger.debug("Autoscale applied")
    
    def closeEvent(self, event):
        """
        Handle widget closure with cleanup.
        
        Args:
            event: QCloseEvent
        """
        logger.info("RXESWidget closing")
        
        try:
            self._is_closing = True
            self._dataset = None
            self.rxesPlot.clear()
            if self.profilePlot is not None:
                self.profilePlot.clear()
        except Exception as e:
            logger.error(f"Error during closeEvent: {e}")
        finally:
            event.accept()
            super().closeEvent(event)
    
    # ================= Compatibility Methods =================
    # These methods provide backward compatibility with old PlotWidget API
    
    def set_signal_suppressed(self, state: bool):
        """Compatibility: suppress signals during batch operations."""
        # Not needed for new architecture
        pass
    
    def set_line_orientation(self, orientation: str):
        """
        Compatibility: set profile line orientation.
        
        In the new architecture, orientation is controlled via the profile
        toolbar buttons. This method is kept for API compatibility.
        """
        logger.debug(f"set_line_orientation called: {orientation} (handled by toolbar)")
    
    def set_bandwidth(self, width: float):
        """
        Compatibility: set integration bandwidth.
        
        In the new architecture, line width is controlled via the profile
        toolbar's line width spinner.
        """
        logger.debug(f"set_bandwidth called: {width} (handled by toolbar)")
    
    def add_line(self):
        """Compatibility: add ROI line (not needed, toolbar handles this)."""
        logger.debug("add_line called (handled by profile toolbar)")
    
    def remove_line(self):
        """Compatibility: remove ROI line (not needed, toolbar handles this)."""
        logger.debug("remove_line called (handled by profile toolbar)")
    
    def ensure_line_count(self, n: int):
        """Compatibility: ensure specific number of lines."""
        logger.debug(f"ensure_line_count called: {n} (not applicable in new architecture)")
    
    def get_line_positions(self) -> List[float]:
        """Compatibility: get ROI line positions."""
        logger.debug("get_line_positions called (use profile toolbar instead)")
        return []
    
    def set_line_positions(self, positions: List[float]):
        """Compatibility: set ROI line positions."""
        logger.debug(f"set_line_positions called: {positions} (use profile toolbar instead)")
    
    def plot_profiles(self, x_label: str, curves: List[Tuple[np.ndarray, np.ndarray, str]]):
        """
        Compatibility: plot profiles in bottom panel.
        
        In the new architecture, profiles are automatically added by the
        profile toolbar. This method is kept for manual profile addition.
        """
        self.profilePlot.clear()
        
        for i, (x, y, label) in enumerate(curves):
            if x is not None and y is not None:
                self.profilePlot.addCurve(x, y, legend=label, linewidth=2.0)
        
        self.profilePlot.setGraphXLabel(x_label)
        logger.debug(f"Plotted {len(curves)} profiles manually")
    
    def clear_profiles(self):
        """Compatibility: clear profile plot."""
        self.profilePlot.clear()
        logger.debug("Profiles cleared")
