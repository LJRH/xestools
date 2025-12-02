"""
XES (1D) Workflow Module

Handles all 1D XES (X-ray Emission Spectroscopy) operations including:
- Loading multiple XES spectra
- Averaging selected spectra
- Area-based normalization
- Background subtraction (LMFIT-based)
- Export results

This module provides the business logic separate from UI concerns.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional, Tuple, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class XESWorkflow:
    """
    Workflow manager for 1D XES operations.
    
    This class encapsulates the business logic for XES data processing,
    separating it from UI concerns.
    """
    
    def __init__(self, loader_module: Any = None):
        """
        Initialize XES workflow.
        
        Args:
            loader_module: Facility-specific loader module (e.g., i20_loader)
        """
        self.loader = loader_module
        self._items: List[Dict[str, Any]] = []
        self._current_channel: str = "upper"
        
        # Derived data
        self._average: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._average_name: Optional[str] = None
        self._background_subtracted: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._background: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._residual: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._background_report: str = ""
        
        # Normalization
        self._norm_target: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._norm_factor: Optional[float] = None
        
        # Wide scan for background fitting
        self._wide_scan: Optional[Tuple[np.ndarray, np.ndarray]] = None
    
    @property
    def current_channel(self) -> str:
        """Get current detector channel."""
        return self._current_channel
    
    @current_channel.setter
    def current_channel(self, value: str):
        """Set current detector channel and reload data if needed."""
        self._current_channel = value.lower()
    
    @property
    def items(self) -> List[Dict[str, Any]]:
        """Get list of loaded XES items."""
        return self._items
    
    @property
    def has_average(self) -> bool:
        """Check if average has been computed."""
        return self._average is not None
    
    @property
    def has_background(self) -> bool:
        """Check if background has been extracted."""
        return self._background is not None
    
    @property
    def has_normalization(self) -> bool:
        """Check if normalization target is set."""
        return self._norm_target is not None
    
    def load_spectrum(
        self,
        path: str,
        channel: Optional[str] = None,
        scan_type: str = "XES",
    ) -> Dict[str, Any]:
        """
        Load a single XES spectrum from file.
        
        Args:
            path: Path to NeXus or ASCII file
            channel: Channel to use (uses current if None)
            scan_type: 'XES' or 'RXES' (for NeXus files)
        
        Returns:
            Dict with keys: path, x, y, channel, kind, name
        """
        if channel is None:
            channel = self._current_channel
        
        ext = os.path.splitext(path)[1].lower()
        
        if ext in (".nxs", ".h5", ".hdf", ".hdf5"):
            if self.loader is None:
                raise RuntimeError("No loader module configured for NeXus files")
            
            x, y = self.loader.xes_from_path(path, channel=channel, type=scan_type)
            
            # Check available channels
            from .scan import Scan
            temp_scan = Scan()
            snum = self.loader.add_scan_from_nxs(temp_scan, path)
            avail_channels = self.loader.available_channels(temp_scan[snum])
            
            item = {
                'path': path,
                'x': np.asarray(x),
                'y': np.asarray(y),
                'channel': channel,
                'kind': 'nxs',
                'name': os.path.basename(path),
                'scan_number': snum,
                'available_channels': avail_channels,
            }
        else:
            # ASCII file
            if self.loader and hasattr(self.loader, 'xes_from_ascii'):
                x, y = self.loader.xes_from_ascii(path)
            else:
                # Simple 2-column fallback
                data = np.genfromtxt(path, comments="#", delimiter=None, dtype=float)
                data = np.atleast_2d(data)
                if data.shape[1] < 2:
                    raise ValueError("ASCII XES must have at least two columns")
                x, y = data[:, 0], data[:, 1]
            
            # Sort and clean
            order = np.argsort(x)
            x, y = x[order], y[order]
            ok = np.isfinite(x) & np.isfinite(y)
            x, y = x[ok], y[ok]
            
            item = {
                'path': path,
                'x': x,
                'y': y,
                'channel': channel,
                'kind': 'ascii',
                'name': os.path.basename(path),
                'available_channels': [],  # ASCII doesn't have channel concept
            }
        
        self._items.append(item)
        logger.info(f"Loaded XES spectrum from {path}")
        return item
    
    def load_spectra(
        self,
        paths: List[str],
        channel: Optional[str] = None,
        scan_type: str = "XES",
    ) -> List[Dict[str, Any]]:
        """
        Load multiple XES spectra.
        
        Args:
            paths: List of file paths
            channel: Channel to use
            scan_type: 'XES' or 'RXES'
        
        Returns:
            List of loaded items
        """
        items = []
        errors = []
        
        for path in paths:
            try:
                item = self.load_spectrum(path, channel, scan_type)
                items.append(item)
            except Exception as e:
                errors.append(f"{os.path.basename(path)}: {e}")
                logger.warning(f"Failed to load {path}: {e}")
        
        if errors:
            logger.warning(f"Errors loading spectra: {errors}")
        
        return items
    
    def remove_item(self, index: int) -> None:
        """Remove an item by index."""
        if 0 <= index < len(self._items):
            del self._items[index]
            self._invalidate_derived()
            logger.info(f"Removed XES item at index {index}")
    
    def clear_all(self) -> None:
        """Clear all loaded spectra and derived data."""
        self._items.clear()
        self._invalidate_derived()
        self._wide_scan = None
        logger.info("Cleared all XES data")
    
    def _invalidate_derived(self) -> None:
        """Invalidate all derived data products."""
        self._average = None
        self._average_name = None
        self._background_subtracted = None
        self._background = None
        self._residual = None
        self._background_report = ""
        self._norm_factor = None
    
    def switch_channel(self, new_channel: str) -> List[str]:
        """
        Switch detector channel for all loaded NeXus spectra.
        
        Args:
            new_channel: New channel name ('upper' or 'lower')
        
        Returns:
            List of error messages for failed reloads
        """
        errors = []
        changed = False
        
        for i, item in enumerate(self._items):
            if item.get('kind') != 'nxs':
                continue
            
            avail = item.get('available_channels', [])
            if new_channel not in avail:
                continue
            
            try:
                x, y = self.loader.xes_from_path(
                    item['path'], 
                    channel=new_channel, 
                    type="XES"
                )
                order = np.argsort(x)
                x, y = np.asarray(x)[order], np.asarray(y)[order]
                ok = np.isfinite(x) & np.isfinite(y)
                x, y = x[ok], y[ok]
                
                self._items[i] = {**item, 'x': x, 'y': y, 'channel': new_channel}
                changed = True
            except Exception as e:
                errors.append(f"{os.path.basename(item.get('path', ''))}: {e}")
        
        if changed:
            self._current_channel = new_channel
            self._invalidate_derived()
        
        return errors
    
    def average_selected(self, indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute average of selected spectra.
        
        Args:
            indices: List of item indices to average
        
        Returns:
            (x, y) averaged spectrum
        
        Raises:
            ValueError: If no valid spectra selected
        """
        if not indices:
            raise ValueError("No spectra selected for averaging")
        
        # Collect selected spectra
        spectra = []
        names = []
        for i in indices:
            if 0 <= i < len(self._items):
                item = self._items[i]
                spectra.append((item['x'], item['y']))
                # Extract scan number from filename
                name = os.path.splitext(item['name'])[0]
                if '_' in name:
                    # Try to extract just the number part
                    parts = name.split('_')
                    names.append(parts[0])
                else:
                    names.append(name)
        
        if not spectra:
            raise ValueError("No valid spectra found at selected indices")
        
        if len(spectra) == 1:
            x_avg, y_avg = spectra[0]
            self._average_name = f"single_{names[0]}"
        else:
            # Regrid and average
            x_avg, y_avg = self._regrid_and_average(spectra)
            self._average_name = f"average_{'+'.join(names)}"
        
        self._average = (x_avg, y_avg)
        logger.info(f"Computed average of {len(spectra)} spectra: {self._average_name}")
        return x_avg, y_avg
    
    def _regrid_and_average(
        self,
        spectra: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Regrid spectra to common grid and compute average.
        
        Uses linear interpolation to common energy grid.
        """
        if not spectra:
            raise ValueError("No spectra to average")
        
        # Find common energy range
        x_min = max(np.min(x) for x, _ in spectra)
        x_max = min(np.max(x) for x, _ in spectra)
        
        if x_min >= x_max:
            raise ValueError("No overlapping energy range")
        
        # Use finest grid spacing
        n_points = max(len(x) for x, _ in spectra)
        x_common = np.linspace(x_min, x_max, n_points)
        
        # Interpolate all spectra to common grid
        y_interp = []
        for x, y in spectra:
            y_new = np.interp(x_common, x, y)
            y_interp.append(y_new)
        
        # Average
        y_avg = np.mean(y_interp, axis=0)
        
        return x_common, y_avg
    
    def set_normalization_target(
        self,
        energy: np.ndarray,
        intensity: np.ndarray,
    ) -> None:
        """
        Set the reference XES spectrum for normalization.
        
        Args:
            energy: 1D energy array
            intensity: 1D intensity array
        """
        self._norm_target = (np.asarray(energy), np.asarray(intensity))
        self._norm_factor = None  # Will be computed when needed
        logger.info("Set normalization target spectrum")
    
    def normalize_by_area(
        self,
        energy: np.ndarray,
        intensity: np.ndarray,
        energy_range: Tuple[float, float],
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Normalize spectrum to reference using area in energy range.
        
        Args:
            energy: 1D energy array to normalize
            intensity: 1D intensity array to normalize
            energy_range: (min, max) energy range for area calculation
        
        Returns:
            (energy, normalized_intensity, factor)
        """
        if self._norm_target is None:
            raise ValueError("No normalization target set")
        
        ref_x, ref_y = self._norm_target
        
        # Compute area of reference in range
        ref_mask = (ref_x >= energy_range[0]) & (ref_x <= energy_range[1])
        if not ref_mask.any():
            raise ValueError("No reference data in energy range")
        ref_area = np.trapz(ref_y[ref_mask], ref_x[ref_mask])
        
        # Compute area of spectrum in range
        spec_mask = (energy >= energy_range[0]) & (energy <= energy_range[1])
        if not spec_mask.any():
            raise ValueError("No spectrum data in energy range")
        spec_area = np.trapz(intensity[spec_mask], energy[spec_mask])
        
        if spec_area <= 0:
            raise ValueError("Invalid spectrum area")
        
        factor = ref_area / spec_area
        normalized = intensity * factor
        
        logger.info(f"Normalized by factor {factor:.6g} (ref_area={ref_area:.6g}, spec_area={spec_area:.6g})")
        return energy, normalized, factor
    
    def extract_background(
        self,
        energy: np.ndarray,
        intensity: np.ndarray,
        model: str = "polynomial",
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Extract background using LMFIT model.
        
        Args:
            energy: 1D energy array
            intensity: 1D intensity array
            model: Model type ('polynomial', 'exponential', 'linear', etc.)
            params: Model-specific parameters
        
        Returns:
            Dict with: background, residual, report, fit_result
        """
        try:
            from lmfit import models as lm_models
        except ImportError:
            raise RuntimeError("lmfit is required for background extraction. pip install lmfit")
        
        if params is None:
            params = {}
        
        # Build model
        if model == "polynomial":
            degree = params.get('degree', 3)
            if degree == 1:
                fit_model = lm_models.LinearModel()
            elif degree == 2:
                fit_model = lm_models.QuadraticModel()
            else:
                fit_model = lm_models.PolynomialModel(degree=degree)
        elif model == "exponential":
            fit_model = lm_models.ExponentialModel()
        elif model == "linear":
            fit_model = lm_models.LinearModel()
        elif model == "gaussian":
            fit_model = lm_models.GaussianModel()
        else:
            raise ValueError(f"Unknown model: {model}")
        
        # Initial guess
        pars = fit_model.guess(intensity, x=energy)
        
        # Fit
        result = fit_model.fit(intensity, pars, x=energy)
        
        background = result.best_fit
        residual = intensity - background
        
        # Store results
        self._background = (energy, background)
        self._residual = (energy, residual)
        self._background_report = result.fit_report()
        
        if self._average is not None:
            avg_x, _ = self._average
            if np.allclose(avg_x, energy):
                self._background_subtracted = (energy, residual)
        
        logger.info(f"Extracted background using {model} model")
        
        return {
            'background': (energy, background),
            'residual': (energy, residual),
            'report': result.fit_report(),
            'fit_result': result,
        }
    
    def set_wide_scan(self, energy: np.ndarray, intensity: np.ndarray) -> None:
        """
        Set wide scan data for background fitting reference.
        
        Args:
            energy: 1D energy array
            intensity: 1D intensity array
        """
        self._wide_scan = (np.asarray(energy), np.asarray(intensity))
        logger.info("Set wide scan reference")
    
    def get_wide_scan(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get wide scan data."""
        return self._wide_scan
    
    def get_average(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get averaged spectrum."""
        return self._average
    
    def get_average_name(self) -> str:
        """Get name of averaged spectrum."""
        return self._average_name or "average"
    
    def get_background(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get background curve."""
        return self._background
    
    def get_residual(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get residual (spectrum - background)."""
        return self._residual
    
    def get_background_report(self) -> str:
        """Get LMFIT fit report."""
        return self._background_report
    
    def get_background_subtracted_average(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get background-subtracted average."""
        return self._background_subtracted
    
    def export_average_csv(self, path: str) -> None:
        """Export average to CSV file."""
        if self._average is None:
            raise ValueError("No average computed")
        
        x, y = self._average
        data = np.column_stack([x, y])
        header = "energy_eV,intensity"
        np.savetxt(path, data, delimiter=',', header=header, comments='', fmt='%.10g')
        logger.info(f"Exported average to {path}")
    
    def export_background_csv(self, path: str) -> None:
        """Export background and residual to CSV file."""
        if self._background is None or self._residual is None:
            raise ValueError("No background extracted")
        
        x, bg = self._background
        _, resid = self._residual
        data = np.column_stack([x, bg, resid])
        header = "energy_eV,background,residual"
        np.savetxt(path, data, delimiter=',', header=header, comments='', fmt='%.10g')
        logger.info(f"Exported background to {path}")
    
    def export_fit_report(self, path: str) -> None:
        """Export LMFIT fit report to text file."""
        if not self._background_report:
            raise ValueError("No fit report available")
        
        with open(path, 'w') as f:
            f.write(self._background_report)
        logger.info(f"Exported fit report to {path}")
