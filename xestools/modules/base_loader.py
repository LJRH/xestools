"""
Base loader module for XES/RXES data processing.

This module contains facility-agnostic functions for loading, validating,
and processing XES (X-ray Emission Spectroscopy) and RXES (Resonant X-ray
Emission Spectroscopy) data.

Other facilities can use these functions by:
1. Defining their own schema dictionary
2. Using the generic loader functions with their schema
3. Or importing individual processing functions as needed

Example schema for a single-channel facility:
    SCHEMA = {
        'name': 'MyFacility',
        'multi_channel': False,
        'nexus': {
            'incident_energy_path': '/entry/mono/energy',
            'emission_energy_path': '/entry/spectrometer/energy',
            'intensity_path': '/entry/detector/counts',
            'monitor_path': '/entry/monitor/I0',  # optional
        },
    }
"""
from __future__ import annotations

import os
from typing import Any, Optional, Tuple, Dict, List

import numpy as np

try:
    import h5py
    H5_AVAILABLE = True
except Exception:
    H5_AVAILABLE = False

from .scan import Scan


# ----------------------------- Schema Definitions -----------------------------

def create_single_channel_schema(
    name: str,
    incident_path: str,
    emission_path: str,
    intensity_path: str,
    monitor_path: Optional[str] = None,
    intensity_fallback: Optional[str] = None,
) -> dict:
    """
    Create a schema for single-channel spectrometer.
    
    Args:
        name: Facility/beamline name
        incident_path: HDF5 path to incident energy array
        emission_path: HDF5 path to emission energy array
        intensity_path: HDF5 path to intensity array
        monitor_path: Optional path to monitor/I0 array
        intensity_fallback: Optional fallback path for intensity
    
    Returns:
        Schema dictionary
    """
    schema = {
        'name': name,
        'multi_channel': False,
        'nexus': {
            'incident_energy_path': incident_path,
            'emission_energy_path': emission_path,
            'intensity_path': intensity_path,
        },
    }
    if monitor_path:
        schema['nexus']['monitor_path'] = monitor_path
    if intensity_fallback:
        schema['nexus']['intensity_fallback'] = intensity_fallback
    return schema


def create_multi_channel_schema(
    name: str,
    incident_path: str,
    channels: Dict[str, Dict[str, str]],
    monitor_path: Optional[str] = None,
) -> dict:
    """
    Create a schema for multi-channel spectrometer (like Diamond I20).
    
    Args:
        name: Facility/beamline name
        incident_path: HDF5 path to incident energy array
        channels: Dict of channel configs, e.g.:
            {
                'upper': {
                    'emission_path': '/entry/XESEnergyUpper',
                    'intensity_path': '/entry/detector1/counts',
                    'intensity_fallback': '/entry/detector1/roi_total',  # optional
                },
                'lower': {...}
            }
        monitor_path: Optional path to monitor/I0 array
    
    Returns:
        Schema dictionary
    """
    schema = {
        'name': name,
        'multi_channel': True,
        'nexus': {
            'incident_energy_path': incident_path,
            'channels': channels,
        },
    }
    if monitor_path:
        schema['nexus']['monitor_path'] = monitor_path
    return schema


# ----------------------------- Data Validation -----------------------------

def validate_rxes_data(
    incident: np.ndarray,
    emission: np.ndarray,
    intensity: np.ndarray,
    monitor: Optional[np.ndarray] = None,
) -> dict:
    """
    Validate RXES data arrays and return metadata.
    
    Works with both 1D (flattened) and 2D (pre-shaped) arrays.
    
    Args:
        incident: Incident energy (Omega) - 1D or 2D
        emission: Emission energy (omega) - 1D or 2D
        intensity: Detector intensity - 1D or 2D
        monitor: Optional monitor signal - 1D or 2D
    
    Returns:
        dict with keys: is_1d, shape, n_points, incident_range, 
                        emission_range, has_finite_data, finite_fraction
    
    Raises:
        ValueError: If arrays incompatible or insufficient data
    """
    incident = np.asarray(incident)
    emission = np.asarray(emission)
    intensity = np.asarray(intensity)
    
    # Check shape compatibility
    if incident.shape != emission.shape or incident.shape != intensity.shape:
        raise ValueError(
            f"Shape mismatch: incident {incident.shape}, "
            f"emission {emission.shape}, intensity {intensity.shape}"
        )
    
    is_1d = (incident.ndim == 1)
    shape = incident.shape
    n_points = incident.size
    
    # Check for finite data
    finite_mask = np.isfinite(incident) & np.isfinite(emission) & np.isfinite(intensity)
    finite_fraction = np.sum(finite_mask) / n_points if n_points > 0 else 0.0
    
    if finite_fraction < 0.5:
        raise ValueError(
            f"Insufficient finite data: only {finite_fraction*100:.1f}% valid"
        )
    
    # Get data ranges
    incident_finite = incident[finite_mask]
    emission_finite = emission[finite_mask]
    
    return {
        'is_1d': is_1d,
        'shape': shape,
        'n_points': n_points,
        'incident_range': (float(np.min(incident_finite)), float(np.max(incident_finite))),
        'emission_range': (float(np.min(emission_finite)), float(np.max(emission_finite))),
        'has_finite_data': finite_fraction > 0,
        'finite_fraction': finite_fraction,
    }


def validate_scan_type_from_data(
    incident: np.ndarray,
    emission: np.ndarray,
    command_scan_type: Optional[str] = None,
    threshold: float = 0.5,
) -> str:
    """
    Validate scan type by analyzing actual data variability.
    
    Uses standard deviation to detect if axes are scanned vs constant:
    - std > threshold: axis is scanned
    - std <= threshold: axis is constant/fixed
    
    Three scan types:
    - RXES: Both incident and emission vary (2D scan)
    - XES: Emission varies, incident fixed (1D emission scan)
    - XANES: Incident varies, emission fixed (NOT SUPPORTED)
    
    Args:
        incident: Incident energy array
        emission: Emission energy array
        command_scan_type: Scan type inferred from command string (optional)
        threshold: Std dev threshold in eV (default 0.5)
    
    Returns:
        Validated scan type: 'RXES' or 'XES'
    
    Raises:
        ValueError: If XANES detected (not supported) or ambiguous
    """
    incident_std = np.nanstd(incident)
    emission_std = np.nanstd(emission)
    
    incident_varies = incident_std > threshold
    emission_varies = emission_std > threshold
    
    # Determine actual scan type from data
    if incident_varies and emission_varies:
        actual_type = 'RXES'
    elif emission_varies and not incident_varies:
        actual_type = 'XES'
    elif incident_varies and not emission_varies:
        actual_type = 'XANES'
    else:
        raise ValueError(
            f"Cannot determine scan type: both axes appear constant "
            f"(incident std={incident_std:.3f}, emission std={emission_std:.3f})"
        )
    
    # XANES not supported
    if actual_type == 'XANES':
        raise ValueError(
            f"XANES data detected (incident scanned, emission fixed). "
            f"This program is for XES/RXES data only. "
            f"Incident std={incident_std:.2f} eV, Emission std={emission_std:.2f} eV"
        )
    
    # Warn if command disagrees with data
    if command_scan_type and command_scan_type != actual_type:
        import warnings
        warnings.warn(
            f"Command suggests '{command_scan_type}' but data indicates '{actual_type}'. "
            f"Using data-based type: {actual_type} "
            f"(incident std={incident_std:.2f}, emission std={emission_std:.2f})"
        )
    
    return actual_type


# ----------------------------- Grid Analysis -----------------------------

def analyze_grid_structure(
    incident: np.ndarray,
    emission: np.ndarray,
    outer_axis: Optional[str] = None,
    precision: float = 0.01,
) -> dict:
    """
    Analyze 2D grid structure of RXES data.
    
    Auto-detects outer/inner axis from repetition patterns.
    
    Args:
        incident: Incident energy (1D or 2D)
        emission: Emission energy (1D or 2D)
        outer_axis: 'incident' or 'emission' or None (auto-detect)
        precision: Rounding precision in eV for unique values
    
    Returns:
        dict with keys: outer_axis, inner_axis, n_outer, n_inner,
                        outer_values, inner_values, outer_counts,
                        is_regular_grid, grid_completeness, needs_reshaping
    """
    incident = np.asarray(incident).ravel()
    emission = np.asarray(emission).ravel()
    
    # Round to precision
    incident_rounded = np.round(incident / precision) * precision
    emission_rounded = np.round(emission / precision) * precision
    
    # Get unique values
    incident_unique = np.unique(incident_rounded[np.isfinite(incident_rounded)])
    emission_unique = np.unique(emission_rounded[np.isfinite(emission_rounded)])
    
    # Auto-detect outer axis if not specified
    if outer_axis is None:
        def count_max_consecutive(arr, unique_vals):
            """Count max consecutive occurrences for first few values."""
            max_counts = []
            for val in unique_vals[:min(5, len(unique_vals))]:
                matches = (arr == val)
                if not matches.any():
                    continue
                # Find runs of consecutive True values
                diff = np.diff(np.concatenate([[False], matches, [False]]).astype(int))
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]
                if len(starts) > 0:
                    max_counts.append(np.max(ends - starts))
            return np.mean(max_counts) if max_counts else 0
        
        incident_consec = count_max_consecutive(incident_rounded, incident_unique)
        emission_consec = count_max_consecutive(emission_rounded, emission_unique)
        
        # More consecutive repetitions = outer axis (stays constant)
        outer_axis = 'incident' if incident_consec > emission_consec else 'emission'
    
    # Set parameters based on outer axis
    if outer_axis == 'incident':
        outer_values = incident_unique
        inner_values = emission_unique
        outer_arr = incident_rounded
    else:
        outer_values = emission_unique
        inner_values = incident_unique
        outer_arr = emission_rounded
    
    n_outer = len(outer_values)
    n_inner = len(inner_values)
    
    # Check regularity by counting outer value occurrences
    outer_counts = [np.sum(outer_arr == val) for val in outer_values]
    is_regular_grid = (len(set(outer_counts)) == 1)
    
    # Grid completeness
    expected_points = n_outer * n_inner
    actual_points = len(incident)
    grid_completeness = actual_points / expected_points if expected_points > 0 else 0.0
    
    return {
        'outer_axis': outer_axis,
        'inner_axis': 'emission' if outer_axis == 'incident' else 'incident',
        'n_outer': n_outer,
        'n_inner': n_inner,
        'outer_values': outer_values,
        'inner_values': inner_values,
        'outer_counts': outer_counts,
        'is_regular_grid': is_regular_grid,
        'grid_completeness': grid_completeness,
        'needs_reshaping': (actual_points == expected_points and is_regular_grid),
    }


def reduce_axes_to_1d(
    emission_2d: np.ndarray,
    incident_2d: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Build 1D axes from 2D meshes so that:
      - y (rows) is emission energy omega
      - x (columns) is incident energy Omega

    Returns (y_emission, x_incident, transposed), where:
      - transposed=True means any Z read from file must be transposed so that
        Z.shape == (len(y_emission), len(x_incident)).
    """
    e = np.asarray(emission_2d)
    b = None if incident_2d is None else np.asarray(incident_2d)
    if e.ndim != 2:
        raise ValueError("emission_2d must be 2D")

    # Heuristic: compare variation across rows vs columns to decide orientation.
    row_var = float(np.nanmean(np.std(e, axis=1)))
    col_var = float(np.nanmean(np.std(e, axis=0)))

    if row_var >= col_var:
        # Emission varies more across columns -> emission currently on columns.
        # Put emission on rows -> take median along columns for omega,
        # and along rows for Omega.
        y_emission = np.nanmedian(e, axis=0)  # length = ncols
        x_incident = (
            np.nanmedian(b, axis=1) if b is not None else np.arange(e.shape[0], dtype=float)
        )  # length = nrows
        transposed = True
    else:
        # Emission varies more across rows -> already on rows.
        y_emission = np.nanmedian(e, axis=1)  # length = nrows
        x_incident = (
            np.nanmedian(b, axis=0) if b is not None else np.arange(e.shape[1], dtype=float)
        )  # length = ncols
        transposed = False

    return y_emission.ravel(), x_incident.ravel(), transposed


def normalize_grid_to_2d(
    incident: np.ndarray,
    emission: np.ndarray,
    intensity: np.ndarray,
    monitor: Optional[np.ndarray] = None,
    grid_info: Optional[dict] = None,
) -> dict:
    """
    Normalize RXES data to standard 2D grid format.
    
    Standard output: emission on rows, incident on columns
    Shape: (n_emission, n_incident)
    
    Handles:
    - Already 2D arrays (NeXus) - validate and transpose if needed
    - 1D arrays (ASCII) - reshape to 2D
    
    Args:
        incident: Incident energy (1D or 2D)
        emission: Emission energy (1D or 2D)
        intensity: Detector intensity (1D or 2D)
        monitor: Optional monitor signal (1D or 2D)
        grid_info: Optional pre-computed from analyze_grid_structure()
    
    Returns:
        dict with keys: incident_2d, emission_2d, intensity_2d, 
                        monitor_2d, method, warnings
    """
    incident = np.asarray(incident)
    emission = np.asarray(emission)
    intensity = np.asarray(intensity)
    if monitor is not None:
        monitor = np.asarray(monitor)
    
    warnings_list = []
    
    # Analyze grid if not provided
    if grid_info is None:
        grid_info = analyze_grid_structure(incident, emission)
    
    # Case 1: Already 2D (from NeXus)
    if incident.ndim == 2:
        # Use existing reduce_axes_to_1d to determine orientation
        y_emission, x_incident, transposed = reduce_axes_to_1d(emission, incident)
        
        if transposed:
            incident_2d = incident.T
            emission_2d = emission.T
            intensity_2d = intensity.T
            monitor_2d = monitor.T if monitor is not None else None
        else:
            incident_2d = incident
            emission_2d = emission
            intensity_2d = intensity
            monitor_2d = monitor
        
        method = 'direct'
    
    # Case 2: 1D arrays - need reshaping (ASCII)
    elif incident.ndim == 1:
        if not grid_info['is_regular_grid']:
            warnings_list.append(
                f"Irregular grid: outer counts vary {set(grid_info['outer_counts'])}"
            )
        
        if grid_info['grid_completeness'] < 0.95:
            warnings_list.append(
                f"Sparse grid: {grid_info['grid_completeness']*100:.1f}% filled"
            )
        
        if not grid_info['needs_reshaping']:
            raise ValueError(
                "Cannot reshape incomplete grid. "
                f"Expected {grid_info['n_outer'] * grid_info['n_inner']} points, "
                f"got {len(incident)}"
            )
        
        # Reshape based on detected structure
        n_outer = grid_info['n_outer']
        n_inner = grid_info['n_inner']
        shape_2d = (n_outer, n_inner)
        
        if grid_info['outer_axis'] == 'incident':
            # incident outer, emission inner
            # Reshape then transpose to standard format
            incident_2d = incident.reshape(shape_2d).T
            emission_2d = emission.reshape(shape_2d).T
            intensity_2d = intensity.reshape(shape_2d).T
            monitor_2d = monitor.reshape(shape_2d).T if monitor is not None else None
        else:
            # emission outer, incident inner (already standard format)
            incident_2d = incident.reshape(shape_2d)
            emission_2d = emission.reshape(shape_2d)
            intensity_2d = intensity.reshape(shape_2d)
            monitor_2d = monitor.reshape(shape_2d) if monitor is not None else None
        
        method = 'reshape'
    
    else:
        raise ValueError(f"Unexpected array dimensions: {incident.ndim}D")
    
    return {
        'incident_2d': incident_2d,
        'emission_2d': emission_2d,
        'intensity_2d': intensity_2d,
        'monitor_2d': monitor_2d,
        'method': method,
        'warnings': warnings_list,
    }


def reduce_to_1d(
    energy: np.ndarray,
    intensity: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduce possibly 2D (mesh-like) energy and intensity to a clean 1D curve Y(X).
    
    Heuristic: detect the varying axis of the energy mesh and sum the intensity
    along the orthogonal axis.
    
    Args:
        energy: Energy array (1D or 2D)
        intensity: Intensity array (1D or 2D)
    
    Returns:
        (x, y) - sorted 1D arrays with NaN values removed
    """
    e = np.asarray(energy, dtype=float)
    y = np.asarray(intensity, dtype=float)

    if e.ndim == 1 and y.ndim == 1:
        x = e
        yi = y
    elif e.ndim == 2 and y.ndim == 2 and e.shape == y.shape:
        row_var = float(np.nanmean(np.std(e, axis=1)))
        col_var = float(np.nanmean(np.std(e, axis=0)))
        if row_var < col_var:
            # energy varies down rows -> collapse rows
            x = np.nanmedian(e, axis=0)
            yi = np.nansum(y, axis=0)
        else:
            # energy varies across columns -> collapse columns
            x = np.nanmedian(e, axis=1)
            yi = np.nansum(y, axis=1)
    else:
        # Fallback: flatten
        x = e.ravel()
        yi = y.ravel()

    order = np.argsort(x)
    x = x[order]
    yi = yi[order]
    ok = np.isfinite(x) & np.isfinite(yi)
    return x[ok], yi[ok]


# ----------------------------- Scan Entry Creation -----------------------------

def create_scan_entry(
    grids: dict,
    path: str,
    channel: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> dict:
    """
    Create standardized Scan entry dict for RXES data.
    
    For single-channel facilities, channel can be None or ignored.
    For multi-channel facilities (like I20), specify 'upper', 'lower', etc.
    
    Args:
        grids: Output from normalize_grid_to_2d()
        path: Source file path
        channel: Optional channel name for multi-channel systems
        metadata: Optional additional metadata
    
    Returns:
        dict suitable for scan.add_scan()
    """
    entry = {
        'path': path,
        'incident_2d': grids['incident_2d'],
        'emission_2d': grids['emission_2d'],
        'intensity': grids['intensity_2d'],
        'averaged': False,
        'normalised': False,
    }
    
    # Add channel info if provided
    if channel:
        entry['channel'] = channel.lower()
    
    # Add monitor if available
    if grids['monitor_2d'] is not None:
        entry['monitor'] = grids['monitor_2d']
    
    # Add optional metadata
    if metadata:
        entry.update(metadata)
    
    return entry


# ----------------------------- Generic NeXus Loader -----------------------------

def load_from_nexus(
    path: str,
    schema: dict,
    channel: Optional[str] = None,
) -> dict:
    """
    Load RXES/XES data from NeXus file using schema.
    
    Args:
        path: Path to NeXus file
        schema: Facility schema dictionary
        channel: For multi-channel systems, which channel to load
    
    Returns:
        dict with: incident, emission, intensity, monitor (optional)
    
    Raises:
        RuntimeError: If h5py not available
        ValueError: If required data not found
    """
    if not H5_AVAILABLE:
        raise RuntimeError("h5py is required to load NeXus files. pip install h5py")
    
    with h5py.File(path, "r") as fh:
        result = {}
        
        # Load incident energy
        incident_path = schema['nexus']['incident_energy_path']
        try:
            result['incident'] = fh[incident_path][...]
        except KeyError:
            raise ValueError(f"Incident energy not found at {incident_path}")
        
        # Load monitor if available
        if 'monitor_path' in schema['nexus']:
            try:
                result['monitor'] = fh[schema['nexus']['monitor_path']][...]
            except KeyError:
                result['monitor'] = None
        else:
            result['monitor'] = None
        
        # Load emission and intensity based on single/multi-channel
        if schema['multi_channel']:
            if channel is None:
                # Try to find first available channel
                for ch_name in schema['nexus']['channels']:
                    try:
                        ch_config = schema['nexus']['channels'][ch_name]
                        result['emission'] = fh[ch_config['emission_path']][...]
                        result['intensity'] = _load_with_fallback(
                            fh, ch_config['intensity_path'],
                            ch_config.get('intensity_fallback')
                        )
                        result['channel'] = ch_name
                        break
                    except (KeyError, ValueError):
                        continue
                else:
                    raise ValueError("No valid channel found in file")
            else:
                # Load specific channel
                ch_config = schema['nexus']['channels'][channel]
                try:
                    result['emission'] = fh[ch_config['emission_path']][...]
                except KeyError:
                    raise ValueError(f"Emission not found at {ch_config['emission_path']}")
                
                result['intensity'] = _load_with_fallback(
                    fh, ch_config['intensity_path'],
                    ch_config.get('intensity_fallback')
                )
                result['channel'] = channel
        else:
            # Single channel
            emission_path = schema['nexus']['emission_energy_path']
            intensity_path = schema['nexus']['intensity_path']
            
            try:
                result['emission'] = fh[emission_path][...]
            except KeyError:
                raise ValueError(f"Emission energy not found at {emission_path}")
            
            result['intensity'] = _load_with_fallback(
                fh, intensity_path,
                schema['nexus'].get('intensity_fallback')
            )
            result['channel'] = None
        
        return result


def _load_with_fallback(fh, primary_path: str, fallback_path: Optional[str] = None):
    """Load dataset from primary path, falling back to secondary if needed."""
    try:
        return fh[primary_path][...]
    except KeyError:
        if fallback_path:
            try:
                return fh[fallback_path][...]
            except KeyError:
                pass
        raise ValueError(f"Intensity not found at {primary_path}")


def add_scan_from_nexus(
    scan: Scan,
    path: str,
    schema: dict,
    channel: Optional[str] = None,
    scan_number: Optional[Any] = None,
) -> Any:
    """
    Load RXES/XES scan from NeXus file and add to Scan container.
    
    Args:
        scan: Scan container
        path: Path to NeXus file
        schema: Facility schema dictionary
        channel: For multi-channel systems, which channel to load
        scan_number: Optional scan number (auto-assigned if None)
    
    Returns:
        scan_number used
    """
    # Load raw data
    data = load_from_nexus(path, schema, channel)
    
    if scan_number is None:
        scan_number = scan.next_index()
    
    # Validate and detect scan type
    validate_rxes_data(data['incident'], data['emission'], data['intensity'], data['monitor'])
    
    scan_type = validate_scan_type_from_data(
        data['incident'].ravel(),
        data['emission'].ravel()
    )
    
    if scan_type == 'XES':
        # 1D XES - reduce and store
        x, y = reduce_to_1d(data['emission'], data['intensity'])
        scan.add_scan(scan_number, {
            'energy': x,
            'intensity': y,
            'monitor': data['monitor'],
            'path': path,
            'scan_type': 'XES',
            'channel': data['channel'],
            'source': 'nexus',
        })
    else:
        # RXES - normalize to 2D grid
        grids = normalize_grid_to_2d(
            data['incident'], data['emission'],
            data['intensity'], data['monitor']
        )
        entry = create_scan_entry(
            grids, path,
            channel=data['channel'],
            metadata={'scan_type': 'RXES', 'source': 'nexus'}
        )
        scan.add_scan(scan_number, entry)
    
    return scan_number


def get_available_channels(schema: dict, path: str) -> List[str]:
    """
    Get list of available channels in a NeXus file.
    
    Args:
        schema: Facility schema dictionary
        path: Path to NeXus file
    
    Returns:
        List of channel names that have valid data
    """
    if not schema['multi_channel']:
        return [None]  # Single channel
    
    if not H5_AVAILABLE:
        return []
    
    channels = []
    with h5py.File(path, "r") as fh:
        for ch_name, ch_config in schema['nexus']['channels'].items():
            try:
                # Check if both emission and intensity exist
                _ = fh[ch_config['emission_path']]
                _ = fh[ch_config['intensity_path']]
                channels.append(ch_name)
            except KeyError:
                # Try fallback for intensity
                if 'intensity_fallback' in ch_config:
                    try:
                        _ = fh[ch_config['emission_path']]
                        _ = fh[ch_config['intensity_fallback']]
                        channels.append(ch_name)
                    except KeyError:
                        pass
    return channels


# ----------------------------- File Detection -----------------------------

def is_probably_detector_hdf(path: str, detector_path: str = "/entry/data/data", scan_path: str = "/entry/data") -> bool:
    """
    Heuristic to detect a raw detector HDF (not the processed scan .nxs).
    
    Returns True if the file contains detector_path but lacks scan_path.
    
    Args:
        path: File path
        detector_path: Path to raw detector data
        scan_path: Path expected in processed scans
    
    Returns:
        True if likely raw detector file
    """
    if not H5_AVAILABLE:
        return False
    try:
        with h5py.File(path, "r") as fh:
            has_detector = isinstance(fh.get(detector_path), h5py.Dataset)
            has_scan = isinstance(fh.get(scan_path), h5py.Dataset)
            return has_detector and not has_scan
    except Exception:
        return False
