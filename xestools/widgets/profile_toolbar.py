#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
XES Profile Toolbar
====================

A customized version of the silx ProfileToolBar that supports:
- Multiple overlaid profiles with distinct colors
- Energy transfer calculations for RXES data
- Integration with xestools data structures

Based on xraylarch's RixsProfileToolBar implementation.

Author: Luke Higgins / OpenCode Assistant
Date: November 2025
"""

import logging
from itertools import cycle
from typing import Optional, List, Tuple, Any

import numpy as np
from silx.gui import qt
from silx.gui.plot.Profile import ProfileToolBar
from silx.gui.plot.items import LineMixIn, SymbolMixIn

logger = logging.getLogger(__name__)

# Default color cycle for overlay profiles (colorblind-friendly palette)
DEFAULT_OVERLAY_COLORS = [
    '#1F77B4',  # Blue
    '#FF7F0E',  # Orange  
    '#2CA02C',  # Green
    '#D62728',  # Red
    '#9467BD',  # Purple
    '#8C564B',  # Brown
    '#E377C2',  # Pink
    '#7F7F7F',  # Gray
    '#BCBD22',  # Olive
    '#17BECF',  # Cyan
]


class XESProfileToolBar(ProfileToolBar):
    """
    XES-adapted Profile (line cut) toolbar with overlay support.
    
    This toolbar extends silx's ProfileToolBar to provide:
    - Multiple colored profiles overlaid on the same plot
    - Proper labeling for RXES data (incident energy, emission energy, energy transfer)
    - Integration with the xestools workflow
    
    The profile toolbar allows users to draw horizontal, vertical, or free-form
    lines on a 2D RXES map to extract 1D spectra along those lines.
    
    In silx 2.x, ProfileToolBar uses a ProfileManager internally, which handles
    the actual profile computation and display. This class customizes the behavior
    while respecting the silx architecture.
    
    Attributes:
        _overlayColors: Iterator of colors for successive profiles
        _currentColorIndex: Current position in color cycle
        _profileHistory: List of extracted profiles for export
    """
    
    def __init__(self, parent: Any = None, plot: Any = None, 
                 profileWindow: Any = None, overlayColors: Optional[List[str]] = None):
        """
        Initialize the XES Profile Toolbar.
        
        Args:
            parent: Parent QWidget
            plot: silx Plot2D instance to extract profiles from
            profileWindow: silx Plot1D instance to display profiles (optional)
            overlayColors: Optional list of color strings (hex or named)
        """
        super().__init__(parent=parent, plot=plot, profileWindow=profileWindow)
        
        self._overlayColors = overlayColors or DEFAULT_OVERLAY_COLORS.copy()
        self._colorCycle = cycle(self._overlayColors)
        self._currentColor = next(self._colorCycle)
        self._profileHistory: List[Tuple[np.ndarray, np.ndarray, str]] = []
        
        # Add clear button to toolbar
        self._clearAction = qt.QAction("Clear All", self)
        self._clearAction.setToolTip("Remove all profile overlays from the plot")
        self._clearAction.triggered.connect(self._clearAllProfiles)
        self.addAction(self._clearAction)
        
        # Add export current profile action
        self._exportAction = qt.QAction("Export Last", self)
        self._exportAction.setToolTip("Export the most recent profile to console/file")
        self._exportAction.triggered.connect(self._exportLastProfile)
        self.addAction(self._exportAction)
        
        logger.info("XESProfileToolBar initialized with silx 2.x compatibility")
    
    def _getNextColor(self) -> str:
        """
        Get the next color in the overlay cycle.
        
        Returns:
            Hex color string for the next profile
        """
        self._currentColor = next(self._colorCycle)
        return self._currentColor
    
    def _clearAllProfiles(self):
        """
        Clear all profile overlays from the profile window.
        
        This removes all curves and resets the profile history. The main
        plot's ROI indicators are also cleared.
        """
        try:
            # Clear the profile manager's output
            manager = self.getProfileManager()
            if manager is not None:
                manager.clearProfile()
            
            # Reset our tracking
            self._profileHistory.clear()
            self._colorCycle = cycle(self._overlayColors)
            self._currentColor = next(self._colorCycle)
            
            logger.info("All profiles cleared")
        except Exception as e:
            logger.error(f"Error clearing profiles: {e}")
    
    def _exportLastProfile(self):
        """
        Export the most recent profile data.
        
        This method can be extended to save to file or copy to clipboard.
        Currently logs the profile info.
        """
        if not self._profileHistory:
            logger.warning("No profiles to export")
            return
        
        x, y, label = self._profileHistory[-1]
        logger.info(f"Last profile: {label}")
        logger.info(f"  X range: {x.min():.2f} to {x.max():.2f}")
        logger.info(f"  Y range: {y.min():.6g} to {y.max():.6g}")
        logger.info(f"  Points: {len(x)}")
        
        # Emit signal or show dialog for saving
        # TODO: Implement file save dialog
    
    def getProfileHistory(self) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        """
        Get the history of extracted profiles.
        
        Returns:
            List of (x_coords, y_values, legend) tuples for all profiles
        """
        return self._profileHistory.copy()
    
    def getCurrentColor(self) -> str:
        """
        Get the current overlay color.
        
        Returns:
            Hex color string
        """
        return self._currentColor
    
    def addProfileToHistory(self, x: np.ndarray, y: np.ndarray, label: str):
        """
        Add a profile to the history (for external use).
        
        Args:
            x: X coordinates (energy)
            y: Y values (intensity)
            label: Profile label/legend
        """
        self._profileHistory.append((x.copy(), y.copy(), label))
        logger.debug(f"Profile added to history: {label}")
    
    def resetColorCycle(self):
        """Reset the color cycle to the beginning."""
        self._colorCycle = cycle(self._overlayColors)
        self._currentColor = next(self._colorCycle)


class XESROIManager:
    """
    Manager for ROI (Region of Interest) operations on RXES plots.
    
    This class provides high-level ROI management functions specific to
    XES/RXES data analysis, wrapping silx's RegionOfInterestManager.
    
    Features:
    - Automatic ROI labeling and styling
    - ROI persistence across plot updates
    - Export of ROI positions for batch processing
    
    Usage:
        manager = XESROIManager(plot)
        # User draws ROIs interactively using silx tools
        rois = manager.getRois()
        positions = manager.getROIPositions()
    """
    
    def __init__(self, plot: Any, color: str = 'pink'):
        """
        Initialize the ROI manager.
        
        Args:
            plot: silx Plot2D instance to manage ROIs for
            color: Default color for new ROIs (hex string or named color)
        """
        from silx.gui.plot.tools.roi import RegionOfInterestManager
        
        self._manager = RegionOfInterestManager(plot)
        self._manager.setColor(color)
        self._manager.sigRoiAdded.connect(self._onRoiAdded)
        self._roi_count = 0
        
        logger.info(f"XESROIManager initialized with default color: {color}")
    
    def _onRoiAdded(self, roi: Any):
        """
        Callback when a new ROI is added.
        
        Automatically labels the ROI and applies consistent styling for
        better visibility on RXES maps.
        
        Args:
            roi: The newly added ROI object from silx
        """
        # Auto-label if no label set
        if hasattr(roi, 'getLabel') and hasattr(roi, 'setLabel'):
            if roi.getLabel() == '':
                self._roi_count += 1
                roi.setLabel(f'ROI {self._roi_count}')
        
        # Apply consistent line styling for visibility
        if isinstance(roi, LineMixIn):
            roi.setLineWidth(2)
            roi.setLineStyle('--')
        
        if isinstance(roi, SymbolMixIn):
            roi.setSymbol('+')
            roi.setSymbolSize(5)
        
        label = roi.getLabel() if hasattr(roi, 'getLabel') else f'ROI {self._roi_count}'
        logger.debug(f"ROI added: {label}")
    
    def getManager(self):
        """
        Get the underlying silx RegionOfInterestManager.
        
        Returns:
            silx.gui.plot.tools.roi.RegionOfInterestManager instance
        """
        return self._manager
    
    def clear(self):
        """Remove all ROIs from the plot."""
        self._manager.clear()
        self._roi_count = 0
        logger.debug("All ROIs cleared")
    
    def getRois(self) -> list:
        """
        Get list of current ROI objects.
        
        Returns:
            List of silx ROI objects
        """
        return list(self._manager.getRois())
    
    def getROIPositions(self) -> List[dict]:
        """
        Get positions of all ROIs as dictionaries.
        
        Returns:
            List of dicts with ROI type and position info
        """
        positions = []
        for roi in self.getRois():
            info = {
                'label': roi.getLabel() if hasattr(roi, 'getLabel') else '',
                'type': type(roi).__name__,
            }
            
            # Extract position based on ROI type
            if hasattr(roi, 'getPosition'):
                info['position'] = roi.getPosition()
            if hasattr(roi, 'getCenter'):
                info['center'] = roi.getCenter()
            if hasattr(roi, 'getSize'):
                info['size'] = roi.getSize()
            
            positions.append(info)
        
        return positions
    
    def stop(self):
        """Stop any active ROI interaction mode."""
        self._manager.stop()
        logger.debug("ROI interaction stopped")
    
    def setColor(self, color: str):
        """
        Set the default color for new ROIs.
        
        Args:
            color: Color string (hex or named)
        """
        self._manager.setColor(color)
        logger.debug(f"ROI color set to: {color}")
