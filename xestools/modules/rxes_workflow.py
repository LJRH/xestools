"""
RXES (2D) Workflow Module

Handles all 2D RXES (Resonant X-ray Emission Spectroscopy) operations including:
- Loading RXES scans from NeXus/ASCII files
- Channel selection (for multi-channel systems)
- Energy mode transformations (incident vs energy transfer)
- Area-based normalization
- Profile extraction support

This module provides the business logic separate from UI concerns.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Tuple, Dict, List

import numpy as np

from .scan import Scan
from .base_loader import (
    validate_rxes_data,
    reduce_axes_to_1d,
    reduce_to_1d,
)

logger = logging.getLogger(__name__)


class RXESWorkflow:
    """
    Workflow manager for 2D RXES operations.
    
    This class encapsulates the business logic for RXES data processing,
    separating it from UI concerns.
    
    Args:
        scan: Scan container for storing loaded data
        loader_module: Facility-specific loader module (e.g., i20_loader)
    """
    
    def __init__(self, scan: Scan, loader_module: Any = None):
        """
        Initialize RXES workflow.
        
        Args:
            scan: Scan container instance
            loader_module: Module with add_scan_from_nxs, available_channels functions
        """
        self.scan = scan
        self.loader = loader_module
        self._current_scan_number: Optional[int] = None
        self._current_channel: str = "upper"
        self._display_mode: str = "incident"  # 'incident' or 'transfer'
        self._normalization_factor: Optional[float] = None
        
    @property
    def current_scan_number(self) -> Optional[int]:
        """Get the current active scan number."""
        return self._current_scan_number
    
    @property
    def current_channel(self) -> str:
        """Get current detector channel."""
        return self._current_channel
    
    @current_channel.setter
    def current_channel(self, value: str):
        """Set current detector channel."""
        self._current_channel = value.lower()
    
    @property
    def display_mode(self) -> str:
        """Get current display mode ('incident' or 'transfer')."""
        return self._display_mode
    
    @display_mode.setter
    def display_mode(self, value: str):
        """Set display mode."""
        if value not in ('incident', 'transfer'):
            raise ValueError(f"Invalid display mode: {value}")
        self._display_mode = value
    
    def load_rxes(self, path: str, scan_number: Optional[int] = None) -> int:
        """
        Load an RXES scan from file.
        
        Args:
            path: Path to NeXus or ASCII file
            scan_number: Optional scan number (auto-assigned if None)
        
        Returns:
            Scan number used
        
        Raises:
            RuntimeError: If loader module not configured
            ValueError: If file cannot be loaded
        """
        if self.loader is None:
            raise RuntimeError("No loader module configured")
        
        snum = self.loader.add_scan_from_nxs(self.scan, path, scan_number)
        self._current_scan_number = snum
        logger.info(f"Loaded RXES scan {snum} from {path}")
        return snum
    
    def get_available_channels(self, scan_number: Optional[int] = None) -> List[str]:
        """
        Get list of available detector channels for a scan.
        
        Args:
            scan_number: Scan number to check (uses current if None)
        
        Returns:
            List of channel names (e.g., ['upper', 'lower'])
        """
        if scan_number is None:
            scan_number = self._current_scan_number
        if scan_number is None:
            return []
        
        entry = self.scan.get(scan_number)
        if entry is None:
            return []
        
        if self.loader and hasattr(self.loader, 'available_channels'):
            return self.loader.available_channels(entry)
        
        # Fallback: check for standard keys
        channels = []
        if entry.get('emission_2d') is not None and entry.get('intensity') is not None:
            channels.append('default')
        # I20-specific keys
        if entry.get('energy_upper_2d') is not None and entry.get('intensity_upper') is not None:
            channels.append('upper')
        if entry.get('energy_lower_2d') is not None and entry.get('intensity_lower') is not None:
            channels.append('lower')
        return channels
    
    def get_rxes_data(
        self,
        scan_number: Optional[int] = None,
        channel: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Get RXES data arrays for plotting.
        
        Args:
            scan_number: Scan number (uses current if None)
            channel: Channel to use (uses current if None)
        
        Returns:
            Dict with keys: incident, emission, intensity, monitor (optional)
            Arrays are 2D with shape (n_emission, n_incident)
        """
        if scan_number is None:
            scan_number = self._current_scan_number
        if channel is None:
            channel = self._current_channel
        
        if scan_number is None:
            raise ValueError("No scan loaded")
        
        entry = self.scan.get(scan_number)
        if entry is None:
            raise ValueError(f"Scan {scan_number} not found")
        
        # Try generic keys first
        if 'incident_2d' in entry:
            result = {
                'incident': np.asarray(entry['incident_2d']),
                'emission': np.asarray(entry['emission_2d']),
                'intensity': np.asarray(entry['intensity']),
            }
            if 'monitor' in entry:
                result['monitor'] = np.asarray(entry['monitor'])
            return result
        
        # I20-specific keys
        use_upper = channel.lower().startswith('u')
        emission_key = 'energy_upper_2d' if use_upper else 'energy_lower_2d'
        intensity_key = 'intensity_upper' if use_upper else 'intensity_lower'
        
        emission = entry.get(emission_key)
        intensity = entry.get(intensity_key)
        incident = entry.get('braggOffset_2d')
        
        if emission is None or intensity is None or incident is None:
            raise ValueError(f"Channel {channel} data not available in scan {scan_number}")
        
        result = {
            'incident': np.asarray(incident),
            'emission': np.asarray(emission),
            'intensity': np.asarray(intensity),
        }
        if 'I1' in entry:
            result['monitor'] = np.asarray(entry['I1'])
        
        return result
    
    def compute_display_coordinates(
        self,
        scan_number: Optional[int] = None,
        channel: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute coordinates for display based on current mode.
        
        Args:
            scan_number: Scan number (uses current if None)
            channel: Channel to use (uses current if None)
            mode: Display mode ('incident' or 'transfer', uses current if None)
        
        Returns:
            Dict with keys: x_2d, y_2d, z, x_label, y_label
            - For 'incident' mode: X=Omega (incident), Y=omega (emission)
            - For 'transfer' mode: X=Omega (incident), Y=Omega-omega (energy transfer)
        """
        if mode is None:
            mode = self._display_mode
        
        data = self.get_rxes_data(scan_number, channel)
        incident = data['incident']
        emission = data['emission']
        intensity = data['intensity']
        
        # Ensure proper 2D orientation
        y_emission, x_incident, transposed = reduce_axes_to_1d(emission, incident)
        
        if transposed:
            incident = incident.T
            emission = emission.T
            intensity = intensity.T
        
        # Apply normalization if set
        z = intensity.astype(float)
        if self._normalization_factor is not None:
            z = z / self._normalization_factor
        
        if mode == 'incident':
            # Standard mode: X=incident, Y=emission
            x_2d = np.broadcast_to(x_incident[None, :], z.shape)
            y_2d = np.broadcast_to(y_emission[:, None], z.shape)
            x_label = "Incident Energy Ω (eV)"
            y_label = "Emission Energy ω (eV)"
        else:
            # Energy transfer mode: X=incident, Y=incident-emission
            x_2d = np.broadcast_to(x_incident[None, :], z.shape)
            # Energy transfer: Delta = Omega - omega
            y_2d = x_2d - np.broadcast_to(y_emission[:, None], z.shape)
            x_label = "Incident Energy Ω (eV)"
            y_label = "Energy Transfer Ω−ω (eV)"
        
        return {
            'x_2d': x_2d,
            'y_2d': y_2d,
            'z': z,
            'x_label': x_label,
            'y_label': y_label,
            'x_1d': x_incident,
            'y_1d': y_emission,
        }
    
    def normalize_by_area(
        self,
        xes_energy: np.ndarray,
        xes_intensity: np.ndarray,
        energy_range: Tuple[float, float],
    ) -> float:
        """
        Compute normalization factor from XES spectrum area.
        
        Args:
            xes_energy: 1D energy array
            xes_intensity: 1D intensity array
            energy_range: (min, max) energy range for integration
        
        Returns:
            Normalization factor (area under curve in range)
        """
        mask = (xes_energy >= energy_range[0]) & (xes_energy <= energy_range[1])
        if not mask.any():
            raise ValueError(f"No data points in range {energy_range}")
        
        # Integrate using trapezoidal rule
        area = np.trapz(xes_intensity[mask], xes_energy[mask])
        if area <= 0:
            raise ValueError(f"Invalid area: {area}")
        
        self._normalization_factor = area
        logger.info(f"Set normalization factor to {area:.6g} from range {energy_range}")
        return area
    
    def clear_normalization(self):
        """Clear any applied normalization."""
        self._normalization_factor = None
        logger.info("Cleared RXES normalization")
    
    def extract_profile(
        self,
        scan_number: Optional[int] = None,
        channel: Optional[str] = None,
        axis: str = 'incident',
        position: float = 0.0,
        bandwidth: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract 1D profile from 2D RXES map.
        
        Args:
            scan_number: Scan number (uses current if None)
            channel: Channel to use (uses current if None)
            axis: 'incident' (constant incident energy) or 'emission' (constant emission)
            position: Position in eV for the cut
            bandwidth: Integration bandwidth in eV
        
        Returns:
            (energy, intensity) 1D arrays
        """
        coords = self.compute_display_coordinates(scan_number, channel)
        x_1d = coords['x_1d']
        y_1d = coords['y_1d']
        z = coords['z']
        
        if axis == 'incident':
            # Constant incident energy: integrate along emission axis
            # Find columns within bandwidth of position
            mask = np.abs(x_1d - position) <= bandwidth / 2
            if not mask.any():
                raise ValueError(f"No data at incident energy {position} +/- {bandwidth/2}")
            profile = np.mean(z[:, mask], axis=1)
            energy = y_1d
        else:
            # Constant emission energy: integrate along incident axis
            mask = np.abs(y_1d - position) <= bandwidth / 2
            if not mask.any():
                raise ValueError(f"No data at emission energy {position} +/- {bandwidth/2}")
            profile = np.mean(z[mask, :], axis=0)
            energy = x_1d
        
        return energy, profile
    
    def get_scan_metadata(self, scan_number: Optional[int] = None) -> Dict[str, Any]:
        """
        Get metadata for a scan.
        
        Args:
            scan_number: Scan number (uses current if None)
        
        Returns:
            Dict with metadata (path, channels, shape, ranges, etc.)
        """
        if scan_number is None:
            scan_number = self._current_scan_number
        if scan_number is None:
            return {}
        
        entry = self.scan.get(scan_number)
        if entry is None:
            return {}
        
        metadata = {
            'scan_number': scan_number,
            'path': entry.get('path', ''),
            'source': entry.get('source', 'unknown'),
            'channels': self.get_available_channels(scan_number),
            'averaged': entry.get('averaged', False),
            'normalised': entry.get('normalised', False),
        }
        
        # Add data shape info
        try:
            data = self.get_rxes_data(scan_number)
            metadata['shape'] = data['intensity'].shape
            metadata['incident_range'] = (float(np.nanmin(data['incident'])), 
                                          float(np.nanmax(data['incident'])))
            metadata['emission_range'] = (float(np.nanmin(data['emission'])), 
                                          float(np.nanmax(data['emission'])))
        except Exception:
            pass
        
        return metadata
    
    def clear(self):
        """Clear all loaded data and reset state."""
        self._current_scan_number = None
        self._normalization_factor = None
        logger.info("RXES workflow cleared")
