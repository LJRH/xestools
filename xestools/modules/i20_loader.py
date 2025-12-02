"""
Diamond Light Source I20 beamline data loader.

This module provides I20-specific data loading functionality built on top of
the generic base_loader functions. Other facilities can use this as a template
by defining their own schema and reusing the base functions.

I20 Schema:
- Multi-channel spectrometer (Upper/Lower detectors)
- NeXus paths: /entry1/I1/... for energy, /entry1/instrument/... for detectors
- ASCII format: Tab-delimited with metadata headers
"""
from __future__ import annotations

import os
import re
from typing import Any, Optional, Tuple

import numpy as np

try:
    import h5py
    H5_AVAILABLE = True
except Exception:
    H5_AVAILABLE = False

from .scan import Scan
from .base_loader import (
    create_multi_channel_schema,
    validate_rxes_data,
    validate_scan_type_from_data,
    analyze_grid_structure,
    normalize_grid_to_2d,
    reduce_to_1d,
    reduce_axes_to_1d,
    load_from_nexus,
    get_available_channels,
    is_probably_detector_hdf as _base_is_probably_detector_hdf,
)


# ----------------------------- I20 Schema Definition -----------------------------

I20_SCHEMA = create_multi_channel_schema(
    name='Diamond_I20',
    incident_path='/entry1/I1/bragg1WithOffset',
    channels={
        'upper': {
            'emission_path': '/entry1/I1/XESEnergyUpper',
            'intensity_path': '/entry1/instrument/medipix1/FFI1_medipix1',
            'intensity_fallback': '/entry1/instrument/medipix1/medipix1_roi_total',
        },
        'lower': {
            'emission_path': '/entry1/I1/XESEnergyLower',
            'intensity_path': '/entry1/instrument/medipix2/FFI1_medipix2',
            'intensity_fallback': '/entry1/instrument/medipix2/medipix2_roi_total',
        },
    },
    monitor_path='/entry1/I1/I1',
)

# Legacy aliases for backward compatibility
def reduce_axes_for(emission_2d: np.ndarray, bragg_offset_2d: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Legacy wrapper for reduce_axes_to_1d."""
    return reduce_axes_to_1d(emission_2d, bragg_offset_2d)


def _reduce_to_1d(energy: np.ndarray, intensity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Legacy wrapper for reduce_to_1d."""
    return reduce_to_1d(energy, intensity)


# ----------------------------- I20 Heuristics -----------------------------

def is_probably_detector_hdf(path: str) -> bool:
    """
    Heuristic to detect a raw detector HDF (not the RXES scan .nxs).
    
    Returns True if the file contains /entry/data/data (typical for detector
    frames) and lacks /entry1/I1/I1 (typical for scan .nxs).
    """
    return _base_is_probably_detector_hdf(
        path, 
        detector_path="/entry/data/data",
        scan_path="/entry1/I1/I1"
    )


# ----------------------------- RXES Loader -----------------------------

def add_scan_from_nxs(scan: Scan, path: str, scan_number: Optional[Any] = None) -> Any:
    """
    Load an I20 RXES scan (.nxs) and append into the Scan container.
    
    Reads both Upper and Lower channels if available.
    Uses I20-specific NeXus schema.
    """
    if not H5_AVAILABLE:
        raise RuntimeError("h5py is required to load NeXus files. pip install h5py")

    with h5py.File(path, "r") as fh:
        # Load common data
        i1 = fh[I20_SCHEMA['nexus']['incident_energy_path']][...]
        bragg_off = i1  # bragg1WithOffset
        i1_monitor = fh[I20_SCHEMA['nexus']['monitor_path']][...]

        # Load channel-specific data
        energy_upper = None
        energy_lower = None
        upper_int = None
        lower_int = None
        
        # Try Upper channel
        try:
            upper_config = I20_SCHEMA['nexus']['channels']['upper']
            energy_upper = fh[upper_config['emission_path']][...]
            try:
                upper_int = fh[upper_config['intensity_path']][...]
            except KeyError:
                upper_int = fh[upper_config['intensity_fallback']][...]
        except Exception:
            pass
        
        # Try Lower channel
        try:
            lower_config = I20_SCHEMA['nexus']['channels']['lower']
            energy_lower = fh[lower_config['emission_path']][...]
            try:
                lower_int = fh[lower_config['intensity_path']][...]
            except KeyError:
                lower_int = fh[lower_config['intensity_fallback']][...]
        except Exception:
            pass

    if scan_number is None:
        scan_number = scan.next_index()

    # Create entry with I20-specific structure (both channels)
    scan.add_scan(
        scan_number,
        {
            "path": path,
            "I1": i1_monitor,
            "braggOffset_2d": bragg_off,
            "energy_upper_2d": energy_upper,
            "energy_lower_2d": energy_lower,
            "intensity_upper": upper_int,
            "intensity_lower": lower_int,
            "averaged": False,
            "normalised": False,
        },
    )
    return scan_number


# ----------------------------- XES Loaders (1D) -----------------------------

def xes_from_nxs(
    path: str,
    channel: str = "upper",
    *,
    type: str = "RXES",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a 1D spectrum from an I20 .nxs.

    type='RXES' (default):
      X = Omega = incident energy
      Y = integrated intensity
    
    type='XES':
      X = omega = emission energy
      Y = intensity
    """
    if not H5_AVAILABLE:
        raise RuntimeError("h5py is required to load NeXus files. pip install h5py")

    use_upper = channel.lower().startswith("u")
    ch_key = 'upper' if use_upper else 'lower'
    ch_config = I20_SCHEMA['nexus']['channels'][ch_key]

    with h5py.File(path, "r") as fh:
        if str(type).strip().upper() == "XES":
            # Emission energy (omega) on X-axis
            energy = fh[ch_config['emission_path']][...]
        else:
            # Incident energy (Omega) on X-axis
            energy = fh[I20_SCHEMA['nexus']['incident_energy_path']][...]
        
        # Load intensity with fallback
        try:
            inten = fh[ch_config['intensity_path']][...]
        except KeyError:
            inten = fh[ch_config['intensity_fallback']][...]
        
        return reduce_to_1d(energy, inten)


def xes_from_ascii(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load 1D XES from ASCII file.
    
    Supports both:
    - I20 beamline format (.dat) - multi-column with header
    - Simple 2-column format - energy, intensity
    """
    # Try I20 format first
    try:
        metadata = parse_i20_ascii_metadata(path)
        if metadata['scan_type'] == 'XES':
            scan = Scan()
            snum = add_scan_from_i20_ascii(scan, path)
            entry = scan[snum]
            return entry['energy'], entry['intensity']
    except Exception:
        pass
    
    # Fallback: simple 2-column format
    data = np.genfromtxt(path, comments="#", delimiter=None, dtype=float)
    data = np.atleast_2d(data)
    if data.shape[1] < 2:
        raise ValueError("ASCII XES must have at least two columns (energy, intensity)")
    x = data[:, 0]
    y = data[:, 1]
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


def xes_from_scan_entry(entry: dict, channel: str = "upper") -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a 1D XES curve from a Scan entry produced by add_scan_from_nxs.
    """
    use_upper = channel.lower().startswith("u")
    energy = entry.get("energy_upper_2d") if use_upper else entry.get("energy_lower_2d")
    inten = entry.get("intensity_upper") if use_upper else entry.get("intensity_lower")
    
    if energy is None:
        # Fallback to incident energy if emission not available
        energy = entry.get("braggOffset_2d")
    if energy is None or inten is None:
        return np.array([]), np.array([])
    
    return reduce_to_1d(energy, inten)


def xes_from_path(
    path: str,
    channel: str = "upper",
    *,
    type: str = "RXES",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a 1D spectrum from either an I20 .nxs (NeXus) or ASCII file.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".nxs":
        return xes_from_nxs(path, channel=channel, type=type)
    return xes_from_ascii(path)


def available_channels(entry: dict) -> list:
    """
    Return a list of available detector channels in this scan entry.
    """
    ch = []
    if entry.get("energy_upper_2d") is not None and entry.get("intensity_upper") is not None:
        ch.append("upper")
    if entry.get("energy_lower_2d") is not None and entry.get("intensity_lower") is not None:
        ch.append("lower")
    return ch


# ----------------------------- I20 ASCII Format Parsing -----------------------------

def parse_i20_ascii_metadata(path: str) -> dict:
    """
    Parse I20 beamline ASCII file metadata from header.
    
    Extracts:
    - Scan command to determine scan type (RXES vs XES)
    - Outer/inner axis from command structure
    - Column names from header line
    - Detector channel (upper/lower)
    """
    metadata = {
        'command': None,
        'scan_type': None,
        'outer_axis': None,
        'inner_axis': None,
        'columns': [],
        'channel': None,
        'detector': None,
        'sample_name': None,
        'date': None,
    }
    
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            
            if not line:
                continue
            
            if not line.startswith('#'):
                break
            
            content = line[1:].strip()
            
            # Extract command
            if content.startswith('command:'):
                metadata['command'] = content[8:].strip()
            elif 'Sample description:' in content and 'command:' in content:
                match = re.search(r'command:\s*(.+)$', content)
                if match:
                    metadata['command'] = match.group(1).strip()
            
            # Extract sample name
            elif content.startswith('Sample name:'):
                metadata['sample_name'] = content[12:].strip()
            
            # Extract date
            elif content.startswith('Instrument:') and 'Date:' in content:
                match = re.search(r'Date:\s*(.+)$', content)
                if match:
                    metadata['date'] = match.group(1).strip()
            
            # Column headers
            elif 'Energy' in content and 'XESEnergy' in content:
                cols = re.split(r'\t+|\s{2,}', content)
                metadata['columns'] = [c.strip() for c in cols if c.strip()]
    
    # Analyze command to determine scan type
    if metadata['command']:
        cmd = metadata['command']
        
        has_bragg_scan = 'bragg1WithOffset' in cmd and \
                        bool(re.search(r'bragg1WithOffset\s+[\d.]+\s+[\d.]+\s+[\d.]+', cmd))
        has_emission_scan = bool(re.search(r'XESEnergy(?:Upper|Lower)\s+\[Range', cmd))
        
        if has_bragg_scan and has_emission_scan:
            metadata['scan_type'] = 'RXES'
            
            bragg_pos = cmd.find('bragg1WithOffset')
            emission_match = re.search(r'XESEnergy(?:Upper|Lower)', cmd)
            emission_pos = emission_match.start() if emission_match else float('inf')
            
            if bragg_pos < emission_pos:
                metadata['outer_axis'] = 'incident'
                metadata['inner_axis'] = 'emission'
            else:
                metadata['outer_axis'] = 'emission'
                metadata['inner_axis'] = 'incident'
        elif has_emission_scan:
            metadata['scan_type'] = 'XES'
        elif has_bragg_scan:
            metadata['scan_type'] = 'XES'
        else:
            raise ValueError(f"Cannot determine scan type from command: {cmd}")
    
    # Detect channel from column names
    if metadata['columns']:
        if 'XESEnergyUpper' in metadata['columns']:
            metadata['channel'] = 'upper'
            if 'FFI1_medipix1' in metadata['columns']:
                metadata['detector'] = 'medipix1'
        elif 'XESEnergyLower' in metadata['columns']:
            metadata['channel'] = 'lower'
            if 'FFI1_medipix2' in metadata['columns']:
                metadata['detector'] = 'medipix2'
    
    # Validate
    if not metadata['columns']:
        raise ValueError("Could not parse column headers from file")
    if not metadata['channel']:
        raise ValueError("Could not determine detector channel (upper/lower)")
    if not metadata['scan_type']:
        raise ValueError("Could not determine scan type (RXES/XES)")
    
    return metadata


def load_i20_ascii_data(path: str, metadata: dict) -> np.ndarray:
    """Load numeric data from I20 ASCII file."""
    try:
        data = np.genfromtxt(path, comments='#', delimiter='\t', dtype=float, invalid_raise=False)
    except Exception:
        try:
            data = np.genfromtxt(path, comments='#', delimiter=None, dtype=float, invalid_raise=False)
        except Exception as e:
            raise ValueError(f"Failed to load data from {path}: {e}")
    
    data = np.atleast_2d(data)
    
    if data.shape[1] != len(metadata['columns']):
        import warnings
        warnings.warn(f"Column count mismatch: expected {len(metadata['columns'])}, got {data.shape[1]}")
    
    return data


def add_scan_from_i20_ascii(
    scan: Scan,
    path: str,
    scan_number: Optional[Any] = None,
) -> Any:
    """
    Load I20 beamline ASCII file into Scan container.
    """
    # Parse metadata
    metadata = parse_i20_ascii_metadata(path)
    
    # Load data
    data = load_i20_ascii_data(path, metadata)
    
    # Extract columns
    bragg = data[:, 0]  # Energy column
    
    emission_col_name = f"XESEnergy{metadata['channel'].capitalize()}"
    try:
        emission_col_idx = metadata['columns'].index(emission_col_name)
    except ValueError:
        raise ValueError(f"Column {emission_col_name} not found in {metadata['columns']}")
    emission = data[:, emission_col_idx]
    
    intensity_col_name = f"FFI1_{metadata['detector']}"
    try:
        intensity_col_idx = metadata['columns'].index(intensity_col_name)
    except ValueError:
        raise ValueError(f"Column {intensity_col_name} not found in {metadata['columns']}")
    intensity = data[:, intensity_col_idx]
    
    try:
        monitor_col_idx = metadata['columns'].index('I1')
        monitor = data[:, monitor_col_idx]
    except ValueError:
        monitor = None
    
    if scan_number is None:
        scan_number = scan.next_index()
    
    # Validate scan type
    validated_scan_type = validate_scan_type_from_data(
        bragg, emission, metadata['scan_type'], threshold=0.5
    )
    metadata['scan_type'] = validated_scan_type
    
    # Process based on scan type
    if metadata['scan_type'] == 'XES':
        ok = np.isfinite(emission) & np.isfinite(intensity)
        scan.add_scan(scan_number, {
            'energy': emission[ok],
            'intensity': intensity[ok],
            'monitor': monitor[ok] if monitor is not None else None,
            'path': path,
            'scan_type': 'XES',
            'channel': metadata['channel'],
            'source': 'ascii',
        })
    elif metadata['scan_type'] == 'RXES':
        # Validate data
        validate_rxes_data(bragg, emission, intensity, monitor)
        
        # Analyze and normalize grid
        grid_info = analyze_grid_structure(bragg, emission, outer_axis=metadata.get('outer_axis'))
        grids = normalize_grid_to_2d(bragg, emission, intensity, monitor, grid_info=grid_info)
        
        # Create I20-specific entry (with channel-specific keys)
        entry = {
            'path': path,
            'braggOffset_2d': grids['incident_2d'],
            'I1': grids['monitor_2d'],
            'averaged': False,
            'normalised': False,
            'source': 'ascii',
            **metadata
        }
        
        # Add channel-specific data
        if metadata['channel'] == 'upper':
            entry['energy_upper_2d'] = grids['emission_2d']
            entry['intensity_upper'] = grids['intensity_2d']
        else:
            entry['energy_lower_2d'] = grids['emission_2d']
            entry['intensity_lower'] = grids['intensity_2d']
        
        scan.add_scan(scan_number, entry)
    else:
        raise ValueError(f"Unknown scan type: {metadata['scan_type']}")
    
    return scan_number


def detect_scan_type_from_file(path: str) -> str:
    """
    Auto-detect scan type from any I20 file (NeXus or ASCII).
    """
    ext = os.path.splitext(path)[1].lower()
    
    if ext in ('.dat', '.txt', '.csv'):
        metadata = parse_i20_ascii_metadata(path)
        return metadata['scan_type']
    
    elif ext in ('.nxs', '.h5', '.hdf', '.hdf5'):
        if not H5_AVAILABLE:
            raise RuntimeError("h5py required for NeXus files")
        
        with h5py.File(path, 'r') as f:
            try:
                bragg = f[I20_SCHEMA['nexus']['incident_energy_path']][...]
            except KeyError:
                raise ValueError("bragg1WithOffset not found in NeXus file")
            
            emission = None
            for ch_config in I20_SCHEMA['nexus']['channels'].values():
                try:
                    emission = f[ch_config['emission_path']][...]
                    break
                except KeyError:
                    pass
            
            if emission is None:
                raise ValueError("No XESEnergy array found")
            
            bragg = np.asarray(bragg, dtype=float).ravel()
            emission = np.asarray(emission, dtype=float).ravel()
            
            return validate_scan_type_from_data(bragg, emission)
    
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ----------------------------- Legacy Compatibility -----------------------------
# These functions maintain backward compatibility with existing code

def create_rxes_scan_entry(grids: dict, channel: str, path: str, metadata: Optional[dict] = None) -> dict:
    """
    Create I20-specific Scan entry dict for RXES data.
    
    Maintains backward compatibility with existing I20 entry structure.
    """
    use_upper = channel.lower().startswith('u')
    
    entry = {
        'path': path,
        'braggOffset_2d': grids['incident_2d'],
        'averaged': False,
        'normalised': False,
    }
    
    if use_upper:
        entry['energy_upper_2d'] = grids['emission_2d']
        entry['intensity_upper'] = grids['intensity_2d']
    else:
        entry['energy_lower_2d'] = grids['emission_2d']
        entry['intensity_lower'] = grids['intensity_2d']
    
    if grids['monitor_2d'] is not None:
        entry['I1'] = grids['monitor_2d']
    
    if metadata:
        entry.update(metadata)
    
    return entry
