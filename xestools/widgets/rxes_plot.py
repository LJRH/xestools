#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RXES/RXES Plot Widgets
=======================

Professional silx-based plot widgets for RXES (Resonant X-ray Emission Spectroscopy)
and RXES (Resonant Inelastic X-ray Scattering) visualization.

This module provides:
- XESRxesPlot2D: Enhanced 2D plot for RXES maps with profile extraction
- XESPlotArea: MDI area for multiple plots with shared profile window
- XESROIDockWidget: ROI management panel

Based on the xraylarch/qtrxes implementation by Newville et al., adapted for
I20 beamline XES/RXES data at Diamond Light Source.

Author: Luke Higgins / OpenCode Assistant
Date: November 2025
"""

import logging
from typing import Optional, Any, List

import numpy as np
from silx.gui import qt
from silx.gui.plot import Plot2D, Plot1D
from silx.gui.plot.tools.roi import RegionOfInterestManager, RegionOfInterestTableWidget
from silx.gui.plot.tools.profile.manager import ProfileWindow
from silx.gui.plot.actions import PlotAction

from .profile_toolbar import XESProfileToolBar, DEFAULT_OVERLAY_COLORS, XESROIManager

logger = logging.getLogger(__name__)


class XESRxesPlot2D(Plot2D):
    """
    RXES/RXES-specific 2D plot widget with integrated profile extraction.
    
    This widget extends silx's Plot2D to provide specialized features for
    synchrotron RXES data:
    
    - Incident energy (Ω) vs Emission energy (ω) maps
    - Energy transfer (Ω - ω) coordinate transformation
    - Integrated profile extraction along arbitrary directions
    - Contour overlay for intensity features
    - Professional colormap management
    
    The plot automatically connects to a profile window (Plot1D) where
    extracted line profiles are displayed. Multiple profiles can be overlaid
    with different colors for comparison.
    
    Attributes:
        _title: Plot title string
        _logger: Logger instance
        _profileWindow: Associated Plot1D for displaying profiles
        _overlayColors: List of colors for profile overlays
        _index: Index for multiple plot windows in MDI setup
        profile: XESProfileToolBar instance for line cut tools
    """
    
    def __init__(self, parent: Any = None, backend: str = None, 
                 logger_instance: Any = None, profileWindow: Any = None, 
                 overlayColors: Optional[List[str]] = None, 
                 title: str = "RXES Map"):
        """
        Initialize the RXES 2D plot widget.
        
        Args:
            parent: Parent QWidget
            backend: silx plot backend ('gl', 'mpl', None for auto)
            logger_instance: Custom logger (uses module logger if None)
            profileWindow: Plot1D instance for profile display
            overlayColors: List of hex color strings for profiles
            title: Default plot title
        """
        super().__init__(parent=parent, backend=backend)
        
        self._title = title
        self._logger = logger_instance or logger
        self._overlayColors = overlayColors or DEFAULT_OVERLAY_COLORS.copy()
        self._index = 0  # For MDI window tracking
        
        # Set up profile window (create if not provided)
        if profileWindow is None:
            # Use silx's ProfileWindow which has proper sigClose and setRoiProfile
            self._profileWindow = ProfileWindow()
            self._profileWindow.setWindowTitle("RXES Profiles")
        else:
            self._profileWindow = profileWindow
        
        # Configure plot appearance
        self._configureToolbar()
        self._configurePlotDefaults()
        
        # Replace default profile toolbar with our enhanced version
        self._setupProfileToolbar()
        
        self._logger.info(f"XESRxesPlot2D initialized: {title}")
    
    def _configureToolbar(self):
        """
        Configure the plot toolbar by hiding unnecessary actions.
        
        Removes tools that are not relevant for RXES analysis to provide
        a cleaner interface focused on the task at hand.
        """
        # Hide mask tool (not needed for RXES viewing)
        mask_action = self.getMaskAction()
        if mask_action:
            mask_action.setVisible(False)
        
        # Hide Y-axis inversion (RXES should always have proper orientation)
        y_invert = self.getYAxisInvertedAction()
        if y_invert:
            y_invert.setVisible(False)
        
        # Keep aspect ratio control but make it less prominent
        aspect_action = self.getKeepDataAspectRatioAction()
        if aspect_action:
            aspect_action.setToolTip("Toggle fixed energy aspect ratio")
        
        # Colorbar is essential for RXES - keep it visible
        colorbar_action = self.getColorBarAction()
        if colorbar_action:
            colorbar_action.setVisible(True)
        
        self._logger.debug("Toolbar configured for RXES workflow")
    
    def _configurePlotDefaults(self):
        """
        Set default plot configuration for RXES data.
        
        Applies sensible defaults for synchrotron XES/RXES visualization.
        """
        # Don't force 1:1 aspect ratio - allow data to fill the plot window
        # This is better for RXES maps where X and Y ranges may differ significantly
        self.setKeepDataAspectRatio(False)
        
        # Use a perceptually uniform colormap suitable for intensity data
        # YlOrBr is good for emission intensity (similar to xraylarch)
        colormap = self.getDefaultColormap()
        colormap.setName('viridis')  # or 'YlOrBr' for thermal style
        
        # Set axis labels for RXES
        self.setGraphTitle(self._title)
        self.setGraphXLabel("Incident Energy Ω (eV)")
        self.setGraphYLabel("Emission Energy ω (eV)")
        
        # Enable data cursor by default
        self.setActiveCurveHandling(False)
        
        self._logger.debug("Plot defaults configured")
    
    def _setupProfileToolbar(self):
        """
        Configure the profile toolbar for XES/RXES analysis.
        
        Instead of replacing the toolbar (which causes ROI manager conflicts),
        we configure the existing ProfileToolBar that Plot2D creates automatically.
        We just update its profile window and add our custom actions.
        """
        # Use the existing profile toolbar created by Plot2D
        if hasattr(self, 'profile') and self.profile is not None:
            # Update the profile window to our custom one with sigClose
            profile_manager = self.profile.getProfileManager()
            if profile_manager is not None:
                profile_manager.setProfileWindow(self._profileWindow)
            
            # Add our custom actions to the existing toolbar
            # These provide XES-specific functionality
            clear_action = qt.QAction("Clear All", self.profile)
            clear_action.setToolTip("Remove all profile overlays")
            clear_action.triggered.connect(self._clearAllProfiles)
            self.profile.addAction(clear_action)
            
            self._logger.debug("Configured existing ProfileToolBar for XES workflow")
        else:
            self._logger.warning("No profile toolbar found in Plot2D - creating one")
            # Fallback: create our own if none exists
            self.profile = XESProfileToolBar(
                plot=self,
                profileWindow=self._profileWindow,
                overlayColors=self._overlayColors
            )
            self.addToolBar(self.profile)
        
        self._logger.debug("Profile toolbar setup complete")
    
    def _clearAllProfiles(self):
        """Clear all profiles from the profile window."""
        if self._profileWindow is not None:
            self._profileWindow.clear()
        self._logger.debug("All profiles cleared")
    
    def setIndex(self, index: int):
        """
        Set the plot window index (for MDI management).
        
        Args:
            index: Integer index for this plot window
        """
        self._index = index
        self._logger.debug(f"Plot index set to {index}")
    
    def getIndex(self) -> int:
        """Get the plot window index."""
        return self._index
    
    def getProfileWindow(self) -> ProfileWindow:
        """
        Get the associated profile display window.
        
        Returns:
            ProfileWindow instance where profiles are displayed
        """
        return self._profileWindow
    
    def setProfileWindow(self, window: ProfileWindow):
        """
        Set a new profile display window.
        
        Args:
            window: ProfileWindow instance for profile display
        """
        self._profileWindow = window
        # Update the profile manager to use this window
        profile_manager = self.profile.getProfileManager()
        if profile_manager is not None:
            profile_manager.setProfileWindow(window)
        self._logger.debug("Profile window updated")
    
    def addRXESImage(self, data: np.ndarray, 
                     incident_energy: Optional[np.ndarray] = None,
                     emission_energy: Optional[np.ndarray] = None,
                     title: Optional[str] = None,
                     xlabel: Optional[str] = None,
                     ylabel: Optional[str] = None,
                     replace: bool = True):
        """
        Add an RXES map to the plot with proper axis scaling.
        
        This is the primary method for displaying RXES data. It handles
        the coordinate transformation from pixel indices to energy values.
        
        Args:
            data: 2D numpy array of RXES intensity data
            incident_energy: 1D array of incident energies (X axis, Ω)
            emission_energy: 1D array of emission energies (Y axis, ω)
            title: Optional plot title
            xlabel: Optional X axis label
            ylabel: Optional Y axis label
            replace: If True, replace existing image; if False, add as overlay
        """
        if data is None or data.ndim != 2:
            self._logger.error("Invalid RXES data: must be 2D array")
            return
        
        self._logger.info(f"Adding RXES image: shape={data.shape}")
        
        # Calculate origin and scale from energy axes
        if incident_energy is not None and emission_energy is not None:
            # Energy axes provided - scale image accordingly
            origin = (float(incident_energy[0]), float(emission_energy[0]))
            
            # Scale = (dx per pixel, dy per pixel)
            if len(incident_energy) > 1:
                x_scale = (float(incident_energy[-1]) - float(incident_energy[0])) / data.shape[1]
            else:
                x_scale = 1.0
            
            if len(emission_energy) > 1:
                y_scale = (float(emission_energy[-1]) - float(emission_energy[0])) / data.shape[0]
            else:
                y_scale = 1.0
            
            scale = (x_scale, y_scale)
            
            self._logger.debug(f"Image origin: {origin}, scale: {scale}")
        else:
            # No energy axes - use pixel coordinates
            origin = (0, 0)
            scale = (1.0, 1.0)
            self._logger.warning("No energy axes provided, using pixel coordinates")
        
        # Add the image
        self.addImage(
            data,
            origin=origin,
            scale=scale,
            legend="RXES",
            replace=replace,
            copy=False
        )
        
        # Update labels and title
        if title:
            self.setGraphTitle(title)
        if xlabel:
            self.setGraphXLabel(xlabel)
        if ylabel:
            self.setGraphYLabel(ylabel)
        
        self._logger.info("RXES image added successfully")
    
    def addContours(self, nlevels: int = 10, colormap: str = None):
        """
        Add contour lines to the current RXES image.
        
        Contours help visualize intensity features and are useful for
        identifying emission lines and resonance features.
        
        Args:
            nlevels: Number of contour levels
            colormap: Colormap for contour colors (None = use plot default)
        """
        # Get current image data
        image = self.getActiveImage()
        if image is None:
            self._logger.warning("No image to add contours to")
            return
        
        data = image.getData(copy=False)
        origin = image.getOrigin()
        scale = image.getScale()
        
        # Compute contour levels
        vmin, vmax = np.nanmin(data), np.nanmax(data)
        levels = np.linspace(vmin, vmax, nlevels + 2)[1:-1]  # Exclude min/max
        
        # Add contour lines
        # Note: silx Plot2D doesn't have built-in contour support like matplotlib
        # This would need a custom implementation using addCurve for each contour
        # For now, log that this is a planned feature
        self._logger.info(f"Contour overlay planned for {nlevels} levels (not yet implemented)")
        
        # TODO: Implement contour extraction using matplotlib.contour or scipy
        # and overlay as curves on the plot
    
    def setEnergyTransferMode(self, enable: bool = True):
        """
        Switch between incident energy and energy transfer coordinate systems.
        
        In energy transfer mode, the Y-axis shows ΔE = Ω - ω instead of ω.
        This is useful for analyzing inelastic scattering features.
        
        Args:
            enable: If True, use energy transfer coordinates
        """
        if enable:
            self.setGraphYLabel("Energy Transfer ΔE = Ω - ω (eV)")
            self._logger.info("Switched to energy transfer mode")
        else:
            self.setGraphYLabel("Emission Energy ω (eV)")
            self._logger.info("Switched to incident energy mode")
        
        # TODO: Actual coordinate transformation of the image data
        # This requires resampling the image onto the new grid


class XESPlotArea(qt.QMdiArea):
    """
    MDI (Multiple Document Interface) area for RXES plot windows.
    
    This container manages multiple XESRxesPlot2D windows with a shared
    profile display. It allows users to compare different RXES maps
    side-by-side while viewing extracted profiles in a common window.
    
    Features:
    - Multiple 2D RXES plots in sub-windows
    - Shared Profile window for all extractions
    - Cascade/tile window arrangement
    - Synchronized colormap settings
    
    Signals:
        changed: Emitted when plot windows are added/removed
    """
    
    # Signal emitted when plot configuration changes
    changed = qt.Signal()
    
    def __init__(self, parent: Any = None, profileWindow: Any = None,
                 overlayColors: Optional[List[str]] = None,
                 logger_instance: Any = None):
        """
        Initialize the MDI plot area.
        
        Args:
            parent: Parent QWidget
            profileWindow: Shared Plot1D for profiles (created if None)
            overlayColors: List of hex color strings for profiles
            logger_instance: Custom logger instance
        """
        super().__init__(parent=parent)
        
        self._logger = logger_instance or logger
        self._overlayColors = overlayColors or DEFAULT_OVERLAY_COLORS.copy()
        
        # Create shared profile window if not provided
        if profileWindow is None:
            self._profileWindow = self._createProfileWindow()
        else:
            self._profileWindow = profileWindow
        
        # Enable context menu for window management
        self.setContextMenuPolicy(qt.Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._showContextMenu)
        
        # Add initial RXES plot
        self.addRixsPlot2D()
        
        self.setMinimumSize(400, 400)
        self.setWindowTitle('RXES Plot Area')
        
        self._logger.info("XESPlotArea initialized")
    
    def _createProfileWindow(self) -> ProfileWindow:
        """
        Create a new ProfileWindow for profile display.
        
        Returns:
            ProfileWindow instance configured for RXES profiles
        """
        # Create as MDI sub-window
        subWindow = qt.QMdiSubWindow(parent=self)
        # Use silx's ProfileWindow which has proper sigClose and setRoiProfile
        plotWindow = ProfileWindow(parent=subWindow)
        plotWindow.setWindowTitle("Profiles")
        subWindow.setWidget(plotWindow)
        subWindow.setWindowTitle("Profiles")
        subWindow.show()
        
        self._logger.debug("Profile window created")
        return plotWindow
    
    def _showContextMenu(self, position: qt.QPoint):
        """
        Show context menu for plot area management.
        
        Args:
            position: Mouse position for menu display
        """
        menu = qt.QMenu('RXES Plot Area', self)
        
        # Add new plot window
        add_action = qt.QAction('Add RXES Plot Window', self,
                                triggered=self.addRixsPlot2D)
        menu.addAction(add_action)
        
        menu.addSeparator()
        
        # Window arrangement options
        cascade_action = qt.QAction('Cascade Windows', self,
                                    triggered=self.cascadeSubWindows)
        menu.addAction(cascade_action)
        
        tile_action = qt.QAction('Tile Windows', self,
                                 triggered=self.tileSubWindows)
        menu.addAction(tile_action)
        
        menu.exec_(self.mapToGlobal(position))
    
    def addRixsPlot2D(self, profileWindow: Any = None) -> XESRxesPlot2D:
        """
        Add a new RXES 2D plot window to the MDI area.
        
        Args:
            profileWindow: Custom profile window (uses shared if None)
            
        Returns:
            The newly created XESRxesPlot2D instance
        """
        subWindow = qt.QMdiSubWindow(parent=self)
        
        # Use shared profile window unless specified
        pw = profileWindow or self._profileWindow
        
        plotWindow = XESRxesPlot2D(
            parent=subWindow,
            profileWindow=pw,
            overlayColors=self._overlayColors
        )
        
        # Set index for identification
        plotWindow.setIndex(len(self.plotWindows()))
        
        subWindow.setWidget(plotWindow)
        subWindow.setWindowTitle(f"RXES Plot {plotWindow.getIndex() + 1}")
        subWindow.show()
        
        self.changed.emit()
        self._logger.info(f"Added RXES plot window {plotWindow.getIndex() + 1}")
        
        return plotWindow
    
    def plotWindows(self) -> List[XESRxesPlot2D]:
        """
        Get list of all RXES plot windows.
        
        Returns:
            List of XESRxesPlot2D instances
        """
        windows = []
        for subWindow in self.subWindowList():
            widget = subWindow.widget()
            if isinstance(widget, XESRxesPlot2D):
                windows.append(widget)
        return windows
    
    def getPlotWindow(self, index: int) -> Optional[XESRxesPlot2D]:
        """
        Get a specific plot window by index.
        
        Args:
            index: Window index (0-based)
            
        Returns:
            XESRxesPlot2D instance or None if index out of range
        """
        windows = self.plotWindows()
        if 0 <= index < len(windows):
            return windows[index]
        return None
    
    def getProfileWindow(self) -> ProfileWindow:
        """
        Get the shared profile display window.
        
        Returns:
            ProfileWindow instance for profile display
        """
        return self._profileWindow


class XESROIDockWidget(qt.QDockWidget):
    """
    Dock widget for ROI (Region of Interest) management.
    
    This widget provides a table view of all ROIs drawn on an RXES plot,
    along with tools for adding, removing, and editing ROIs.
    
    Features:
    - Table showing ROI labels, positions, and types
    - Toolbar with buttons for all ROI drawing modes
    - Integration with silx's ROI management system
    - Automatic cleanup when hidden
    """
    
    def __init__(self, plot: XESRxesPlot2D, parent: Any = None):
        """
        Initialize the ROI dock widget.
        
        Args:
            plot: XESRxesPlot2D instance to manage ROIs for
            parent: Parent QMainWindow
        """
        if not isinstance(plot, XESRxesPlot2D):
            raise TypeError("'plot' must be an instance of XESRxesPlot2D")
        
        title = f"ROI Manager - Plot {plot.getIndex() + 1}"
        super().__init__(title, parent=parent)
        
        self._plot = plot
        
        # Create ROI manager for the plot
        self._roiManager = RegionOfInterestManager(plot)
        self._roiManager.setColor('#FF69B4')  # Hot pink for visibility
        self._roiManager.sigRoiAdded.connect(self._onRoiAdded)
        
        # Table widget showing ROI information
        self._roiTable = RegionOfInterestTableWidget()
        self._roiTable.setRegionOfInterestManager(self._roiManager)
        
        # Toolbar with ROI drawing mode buttons
        self._roiToolbar = qt.QToolBar()
        self._roiToolbar.setIconSize(qt.QSize(20, 20))
        
        # Add action for each supported ROI type
        for roiClass in self._roiManager.getSupportedRoiClasses():
            action = self._roiManager.getInteractionModeAction(roiClass)
            self._roiToolbar.addAction(action)
        
        # Layout
        widget = qt.QWidget()
        layout = qt.QVBoxLayout()
        widget.setLayout(layout)
        layout.addWidget(self._roiToolbar)
        layout.addWidget(self._roiTable)
        
        self.setWidget(widget)
        
        # Stop ROI interaction when dock is hidden
        self.visibilityChanged.connect(self._onVisibilityChanged)
        
        logger.info(f"XESROIDockWidget created for plot {plot.getIndex() + 1}")
    
    def _onRoiAdded(self, roi: Any):
        """
        Callback when a new ROI is added.
        
        Args:
            roi: The newly added ROI object
        """
        if hasattr(roi, 'getLabel') and roi.getLabel() == '':
            roi.setLabel(f'{len(self._roiManager.getRois())}')
        
        logger.debug(f"ROI added to manager")
    
    def _onVisibilityChanged(self, visible: bool):
        """
        Handle visibility changes.
        
        Stops ROI interaction mode when the dock is hidden to prevent
        confusion about the current interaction state.
        
        Args:
            visible: True if dock became visible, False if hidden
        """
        if not visible:
            self._roiManager.stop()
            logger.debug("ROI interaction stopped (dock hidden)")
    
    def getROIManager(self) -> RegionOfInterestManager:
        """
        Get the underlying silx ROI manager.
        
        Returns:
            RegionOfInterestManager instance
        """
        return self._roiManager
    
    def clearROIs(self):
        """Remove all ROIs from the plot."""
        self._roiManager.clear()
        logger.debug("All ROIs cleared")
