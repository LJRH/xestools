"""
I/O Workflow Module

High-level file operations orchestration for XES/RXES data.
Coordinates between different file formats and loaders.

This module provides:
- Auto-detection of file types
- Format validation
- High-level save/load orchestration
- Metadata extraction
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional, Dict, Tuple

import numpy as np

from .dataset import DataSet
from .io import (
    save_ascii,
    save_nexus,
    load_path,
    H5_AVAILABLE,
)

logger = logging.getLogger(__name__)


class IOWorkflow:
    """
    I/O workflow manager for file operations.
    
    Provides high-level orchestration of file loading and saving,
    with auto-detection and validation.
    """
    
    def __init__(self, loader_module: Any = None):
        """
        Initialize I/O workflow.
        
        Args:
            loader_module: Facility-specific loader module (optional)
        """
        self.loader = loader_module
    
    @staticmethod
    def detect_file_type(path: str) -> str:
        """
        Detect file type from extension and content.
        
        Args:
            path: File path
        
        Returns:
            File type: 'nexus', 'ascii', 'unknown'
        """
        ext = os.path.splitext(path)[1].lower()
        
        if ext in ('.nxs', '.h5', '.hdf', '.hdf5', '.nx5'):
            return 'nexus'
        elif ext in ('.dat', '.txt', '.csv', '.asc'):
            return 'ascii'
        else:
            return 'unknown'
    
    @staticmethod
    def is_nexus_available() -> bool:
        """Check if NeXus/HDF5 support is available."""
        return H5_AVAILABLE
    
    def validate_file(self, path: str) -> Dict[str, Any]:
        """
        Validate file and extract basic metadata.
        
        Args:
            path: File path
        
        Returns:
            Dict with: valid, type, size, mtime, errors, warnings
        """
        result = {
            'valid': False,
            'type': 'unknown',
            'path': path,
            'exists': False,
            'size': 0,
            'mtime': None,
            'errors': [],
            'warnings': [],
        }
        
        # Check existence
        if not os.path.exists(path):
            result['errors'].append(f"File does not exist: {path}")
            return result
        
        result['exists'] = True
        result['size'] = os.path.getsize(path)
        result['mtime'] = os.path.getmtime(path)
        
        # Detect type
        result['type'] = self.detect_file_type(path)
        
        if result['type'] == 'unknown':
            result['warnings'].append(f"Unknown file type: {os.path.splitext(path)[1]}")
        
        # Check for raw detector HDF
        if result['type'] == 'nexus':
            if not H5_AVAILABLE:
                result['errors'].append("h5py not installed for NeXus support")
                return result
            
            if self.loader and hasattr(self.loader, 'is_probably_detector_hdf'):
                if self.loader.is_probably_detector_hdf(path):
                    result['errors'].append("File appears to be raw detector HDF, not processed scan")
                    return result
        
        result['valid'] = True
        return result
    
    def detect_scan_type(self, path: str) -> str:
        """
        Detect whether file contains RXES or XES data.
        
        Args:
            path: File path
        
        Returns:
            'RXES', 'XES', or 'unknown'
        """
        if self.loader and hasattr(self.loader, 'detect_scan_type_from_file'):
            try:
                return self.loader.detect_scan_type_from_file(path)
            except Exception as e:
                logger.warning(f"Could not detect scan type: {e}")
        
        return 'unknown'
    
    def load_generic_dataset(self, path: str) -> DataSet:
        """
        Load file as generic DataSet (1D or 2D).
        
        Uses io.load_path which handles both ASCII and NeXus.
        
        Args:
            path: File path
        
        Returns:
            DataSet object
        """
        return load_path(path)
    
    def save_dataset_ascii(self, path: str, dataset: DataSet) -> None:
        """
        Save DataSet to ASCII CSV file.
        
        Args:
            path: Output file path
            dataset: DataSet to save
        """
        save_ascii(path, dataset)
        logger.info(f"Saved dataset to ASCII: {path}")
    
    def save_dataset_nexus(self, path: str, dataset: DataSet) -> None:
        """
        Save DataSet to NeXus/HDF5 file.
        
        Args:
            path: Output file path
            dataset: DataSet to save
        
        Raises:
            RuntimeError: If h5py not available
        """
        save_nexus(path, dataset)
        logger.info(f"Saved dataset to NeXus: {path}")
    
    def save_1d_spectrum(
        self,
        path: str,
        energy: np.ndarray,
        intensity: np.ndarray,
        xlabel: str = "Energy (eV)",
        ylabel: str = "Intensity",
    ) -> None:
        """
        Save 1D XES spectrum.
        
        Args:
            path: Output file path
            energy: Energy array
            intensity: Intensity array
            xlabel: X-axis label
            ylabel: Y-axis label
        """
        ext = os.path.splitext(path)[1].lower()
        
        ds = DataSet(
            "1D",
            x=np.asarray(energy),
            y=np.asarray(intensity),
            xlabel=xlabel,
            ylabel=ylabel,
            source=path,
        )
        
        if ext in ('.nxs', '.h5', '.hdf5'):
            save_nexus(path, ds)
        else:
            save_ascii(path, ds)
        
        logger.info(f"Saved 1D spectrum to {path}")
    
    def save_2d_map(
        self,
        path: str,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        xlabel: str = "Incident Energy (eV)",
        ylabel: str = "Emission Energy (eV)",
        zlabel: str = "Intensity",
    ) -> None:
        """
        Save 2D RXES map.
        
        Args:
            path: Output file path
            x: X-axis (incident energy) 1D array
            y: Y-axis (emission energy) 1D array
            z: 2D intensity array with shape (len(y), len(x))
            xlabel: X-axis label
            ylabel: Y-axis label
            zlabel: Z (intensity) label
        """
        ext = os.path.splitext(path)[1].lower()
        
        ds = DataSet(
            "2D",
            x=np.asarray(x),
            y=np.asarray(y),
            z=np.asarray(z),
            xlabel=xlabel,
            ylabel=ylabel,
            zlabel=zlabel,
            source=path,
        )
        
        if ext in ('.nxs', '.h5', '.hdf5'):
            save_nexus(path, ds)
        else:
            save_ascii(path, ds)
        
        logger.info(f"Saved 2D map to {path}")
    
    def get_file_metadata(self, path: str) -> Dict[str, Any]:
        """
        Extract metadata from file.
        
        Args:
            path: File path
        
        Returns:
            Dict with file metadata
        """
        metadata = {
            'path': path,
            'filename': os.path.basename(path),
            'directory': os.path.dirname(path),
            'extension': os.path.splitext(path)[1],
            'type': self.detect_file_type(path),
        }
        
        if os.path.exists(path):
            stat = os.stat(path)
            metadata['size_bytes'] = stat.st_size
            metadata['mtime'] = stat.st_mtime
            
            # Human-readable size
            size = stat.st_size
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size < 1024:
                    metadata['size_human'] = f"{size:.1f} {unit}"
                    break
                size /= 1024
        
        # Try to get scan type
        if self.loader:
            try:
                metadata['scan_type'] = self.detect_scan_type(path)
            except Exception:
                pass
        
        return metadata
    
    def export_profiles_csv(
        self,
        path: str,
        profiles: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """
        Export multiple profiles to CSV file.
        
        Args:
            path: Output file path
            profiles: Dict mapping profile names to (x, y) tuples
        """
        if not profiles:
            raise ValueError("No profiles to export")
        
        # Build header
        columns = ['energy_eV']
        for name in profiles:
            columns.append(name)
        
        # Find common length (or use longest)
        max_len = max(len(x) for x, _ in profiles.values())
        
        # Build data array
        data = np.full((max_len, len(columns)), np.nan)
        
        for i, (name, (x, y)) in enumerate(profiles.items()):
            if i == 0:
                data[:len(x), 0] = x
            data[:len(y), i + 1] = y
        
        header = ','.join(columns)
        np.savetxt(path, data, delimiter=',', header=header, comments='', fmt='%.10g')
        logger.info(f"Exported {len(profiles)} profiles to {path}")
    
    @staticmethod
    def suggest_save_path(
        original_path: str,
        suffix: str = "",
        new_extension: Optional[str] = None,
    ) -> str:
        """
        Suggest a save path based on original file path.
        
        Args:
            original_path: Original file path
            suffix: Suffix to add before extension (e.g., '_processed')
            new_extension: New extension (e.g., '.csv'), or None to keep original
        
        Returns:
            Suggested save path
        """
        base, ext = os.path.splitext(original_path)
        if new_extension is not None:
            ext = new_extension
        return f"{base}{suffix}{ext}"
