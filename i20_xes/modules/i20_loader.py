from __future__ import annotations

import os
from typing import Any, Optional, Tuple

import numpy as np

try:
    import h5py
    H5_AVAILABLE = True
except Exception:  # pragma: no cover
    H5_AVAILABLE = False

from .scan import Scan


# ----------------------------- Heuristics -----------------------------
def is_probably_detector_hdf(path: str) -> bool:
    """
    Heuristic to detect a raw detector HDF (not the RXES scan .nxs).

    Returns True if the file contains /entry/data/data (typical for detector
    frames) and lacks /entry1/I1/I1 (typical for scan .nxs).
    """
    if not H5_AVAILABLE:
        return False
    try:
        with h5py.File(path, "r") as fh:
            has_entry_data = isinstance(fh.get("/entry/data/data"), h5py.Dataset)
            has_scan_i1 = isinstance(fh.get("/entry1/I1/I1"), h5py.Dataset)
            return has_entry_data and not has_scan_i1
    except Exception:
        return False

# ----------------------------- Axes tools -----------------------------
def reduce_axes_for(emission_2d: np.ndarray,bragg_offset_2d: Optional[np.ndarray],) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Build 1D axes from 2D meshes so that:
      - y (rows) is emission energy ω (XESEnergy Upper or Lower)
      - x (columns) is incident energy Ω (bragg1WithOffset)

    Returns (y_omega, x_Omega, transposed), where:
      - transposed=True means any Z read from file must be transposed so that
        Z.shape == (len(y_omega), len(x_Omega)).
    """
    e = np.asarray(emission_2d)
    b = None if bragg_offset_2d is None else np.asarray(bragg_offset_2d)
    if e.ndim != 2:
        raise ValueError("emission_2d must be 2D")

    # Heuristic: compare variation across rows vs columns to decide orientation.
    row_var = float(np.nanmean(np.std(e, axis=1)))
    col_var = float(np.nanmean(np.std(e, axis=0)))

    if row_var >= col_var:
        # Emission varies more across columns -> emission currently on columns.
        # Put emission on rows -> take median along columns for ω (columns -> 1D),
        # and along rows for Ω.
        y_omega = np.nanmedian(e, axis=0)  # length = ncols
        x_Omega = (
            np.nanmedian(b, axis=1) if b is not None else np.arange(e.shape[0], dtype=float)
        )  # length = nrows
        transposed = True
    else:
        # Emission varies more across rows -> already on rows.
        y_omega = np.nanmedian(e, axis=1)  # length = nrows
        x_Omega = (
            np.nanmedian(b, axis=0) if b is not None else np.arange(e.shape[1], dtype=float)
        )  # length = ncols
        transposed = False

    return y_omega.ravel(), x_Omega.ravel(), transposed

# ----------------------------- Shared Grid Processing Functions -----------------------------
def validate_rxes_data(
    bragg: np.ndarray,
    emission: np.ndarray,
    intensity: np.ndarray,
    monitor: Optional[np.ndarray] = None,
) -> dict:
    """
    Validate RXES data arrays and return metadata.
    
    Works with both 1D (flattened) and 2D (pre-shaped) arrays.
    
    Args:
        bragg: Incident energy (Ω) - 1D or 2D
        emission: Emission energy (ω) - 1D or 2D
        intensity: Detector intensity - 1D or 2D
        monitor: Optional I1 monitor - 1D or 2D
    
    Returns:
        dict with keys: is_1d, shape, n_points, bragg_range, 
                        emission_range, has_finite_data, finite_fraction
    
    Raises:
        ValueError: If arrays incompatible or insufficient data
    """
    bragg = np.asarray(bragg)
    emission = np.asarray(emission)
    intensity = np.asarray(intensity)
    
    # Check shape compatibility
    if bragg.shape != emission.shape or bragg.shape != intensity.shape:
        raise ValueError(
            f"Shape mismatch: bragg {bragg.shape}, "
            f"emission {emission.shape}, intensity {intensity.shape}"
        )
    
    is_1d = (bragg.ndim == 1)
    shape = bragg.shape
    n_points = bragg.size
    
    # Check for finite data
    finite_mask = np.isfinite(bragg) & np.isfinite(emission) & np.isfinite(intensity)
    finite_fraction = np.sum(finite_mask) / n_points if n_points > 0 else 0.0
    
    if finite_fraction < 0.5:
        raise ValueError(
            f"Insufficient finite data: only {finite_fraction*100:.1f}% valid"
        )
    
    # Get data ranges
    bragg_finite = bragg[finite_mask]
    emission_finite = emission[finite_mask]
    
    return {
        'is_1d': is_1d,
        'shape': shape,
        'n_points': n_points,
        'bragg_range': (float(np.min(bragg_finite)), float(np.max(bragg_finite))),
        'emission_range': (float(np.min(emission_finite)), float(np.max(emission_finite))),
        'has_finite_data': finite_fraction > 0,
        'finite_fraction': finite_fraction,
    }
def analyze_grid_structure(
    bragg: np.ndarray,
    emission: np.ndarray,
    outer_axis: Optional[str] = None,
    precision: float = 0.01,
) -> dict:
    """
    Analyze 2D grid structure of RXES data.
    
    Auto-detects outer/inner axis from repetition patterns.
    
    Args:
        bragg: Incident energy (1D or 2D)
        emission: Emission energy (1D or 2D)
        outer_axis: 'bragg' or 'emission' or None (auto-detect)
        precision: Rounding precision in eV for unique values
    
    Returns:
        dict with keys: outer_axis, inner_axis, n_outer, n_inner,
                        outer_values, inner_values, outer_counts,
                        is_regular_grid, grid_completeness, needs_reshaping
    """
    bragg = np.asarray(bragg).ravel()
    emission = np.asarray(emission).ravel()
    
    # Round to precision
    bragg_rounded = np.round(bragg / precision) * precision
    emission_rounded = np.round(emission / precision) * precision
    
    # Get unique values
    bragg_unique = np.unique(bragg_rounded[np.isfinite(bragg_rounded)])
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
        
        bragg_consec = count_max_consecutive(bragg_rounded, bragg_unique)
        emission_consec = count_max_consecutive(emission_rounded, emission_unique)
        
        # More consecutive repetitions = outer axis (stays constant)
        outer_axis = 'bragg' if bragg_consec > emission_consec else 'emission'
    
    # Set parameters based on outer axis
    if outer_axis == 'bragg':
        outer_values = bragg_unique
        inner_values = emission_unique
        outer_arr = bragg_rounded
    else:
        outer_values = emission_unique
        inner_values = bragg_unique
        outer_arr = emission_rounded
    
    n_outer = len(outer_values)
    n_inner = len(inner_values)
    
    # Check regularity by counting outer value occurrences
    outer_counts = [np.sum(outer_arr == val) for val in outer_values]
    is_regular_grid = (len(set(outer_counts)) == 1)
    
    # Grid completeness
    expected_points = n_outer * n_inner
    actual_points = len(bragg)
    grid_completeness = actual_points / expected_points if expected_points > 0 else 0.0
    
    return {
        'outer_axis': outer_axis,
        'inner_axis': 'emission' if outer_axis == 'bragg' else 'bragg',
        'n_outer': n_outer,
        'n_inner': n_inner,
        'outer_values': outer_values,
        'inner_values': inner_values,
        'outer_counts': outer_counts,
        'is_regular_grid': is_regular_grid,
        'grid_completeness': grid_completeness,
        'needs_reshaping': (actual_points == expected_points and is_regular_grid),
    }
def normalize_grid_to_2d(
    bragg: np.ndarray,
    emission: np.ndarray,
    intensity: np.ndarray,
    monitor: Optional[np.ndarray] = None,
    grid_info: Optional[dict] = None,
) -> dict:
    """
    Normalize RXES data to standard 2D grid format.
    
    Standard output: emission on rows, bragg on columns
    Shape: (n_emission, n_bragg)
    
    Handles:
    - Already 2D arrays (NeXus) - validate and transpose if needed
    - 1D arrays (ASCII) - reshape to 2D
    
    Args:
        bragg: Incident energy (1D or 2D)
        emission: Emission energy (1D or 2D)
        intensity: Detector intensity (1D or 2D)
        monitor: Optional I1 monitor (1D or 2D)
        grid_info: Optional pre-computed from analyze_grid_structure()
    
    Returns:
        dict with keys: bragg_2d, emission_2d, intensity_2d, 
                        monitor_2d, method, warnings
    """
    bragg = np.asarray(bragg)
    emission = np.asarray(emission)
    intensity = np.asarray(intensity)
    if monitor is not None:
        monitor = np.asarray(monitor)
    
    warnings_list = []
    
    # Analyze grid if not provided
    if grid_info is None:
        grid_info = analyze_grid_structure(bragg, emission)
    
    # Case 1: Already 2D (from NeXus)
    if bragg.ndim == 2:
        # Use existing reduce_axes_for to determine orientation
        y_omega, x_Omega, transposed = reduce_axes_for(emission, bragg)
        
        if transposed:
            bragg_2d = bragg.T
            emission_2d = emission.T
            intensity_2d = intensity.T
            monitor_2d = monitor.T if monitor is not None else None
        else:
            bragg_2d = bragg
            emission_2d = emission
            intensity_2d = intensity
            monitor_2d = monitor
        
        method = 'direct'
    
    # Case 2: 1D arrays - need reshaping (ASCII)
    elif bragg.ndim == 1:
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
                f"got {len(bragg)}"
            )
        
        # Reshape based on detected structure
        n_outer = grid_info['n_outer']
        n_inner = grid_info['n_inner']
        shape_2d = (n_outer, n_inner)
        
        if grid_info['outer_axis'] == 'bragg':
            # bragg outer, emission inner
            # Reshape then transpose to standard format
            bragg_2d = bragg.reshape(shape_2d).T
            emission_2d = emission.reshape(shape_2d).T
            intensity_2d = intensity.reshape(shape_2d).T
            monitor_2d = monitor.reshape(shape_2d).T if monitor is not None else None
        else:
            # emission outer, bragg inner (already standard format)
            bragg_2d = bragg.reshape(shape_2d)
            emission_2d = emission.reshape(shape_2d)
            intensity_2d = intensity.reshape(shape_2d)
            monitor_2d = monitor.reshape(shape_2d) if monitor is not None else None
        
        method = 'reshape'
    
    else:
        raise ValueError(f"Unexpected array dimensions: {bragg.ndim}D")
    
    return {
        'bragg_2d': bragg_2d,
        'emission_2d': emission_2d,
        'intensity_2d': intensity_2d,
        'monitor_2d': monitor_2d,
        'method': method,
        'warnings': warnings_list,
    }
def create_rxes_scan_entry(
    grids: dict,
    channel: str,
    path: str,
    metadata: Optional[dict] = None,
) -> dict:
    """
    Create standardized Scan entry dict for RXES data.
    
    Args:
        grids: Output from normalize_grid_to_2d()
        channel: 'upper' or 'lower'
        path: Source file path
        metadata: Optional additional metadata
    
    Returns:
        dict suitable for scan.add_scan()
    """
    use_upper = channel.lower().startswith('u')
    
    entry = {
        'path': path,
        'braggOffset_2d': grids['bragg_2d'],
        'averaged': False,
        'normalised': False,
    }
    
    # Add channel-specific data
    if use_upper:
        entry['energy_upper_2d'] = grids['emission_2d']
        entry['intensity_upper'] = grids['intensity_2d']
    else:
        entry['energy_lower_2d'] = grids['emission_2d']
        entry['intensity_lower'] = grids['intensity_2d']
    
    # Add monitor if available
    if grids['monitor_2d'] is not None:
        entry['I1'] = grids['monitor_2d']
    
    # Add optional metadata
    if metadata:
        entry.update(metadata)
    
    return entry

# ----------------------------- RXES loader -----------------------------
def add_scan_from_nxs(scan: Scan, path: str, scan_number: Optional[Any] = None) -> Any:
    """
    Load an I20 RXES scan (.nxs) and append into the Scan container.

    Reads from the I20 layout:
      - I1 grid (correction):           /entry1/I1/I1
      - Incident energy Ω (preferred):  /entry1/I1/bragg1WithOffset
      - Emission energy ω (Upper):      /entry1/I1/XESEnergyUpper (if present)
      - Emission energy ω (Lower):      /entry1/I1/XESEnergyLower (if present)
      - Intensities:
          Upper channel: /entry1/instrument/medipix1/FFI1_medipix1
                         (fallback: /entry1/instrument/medipix1/medipix1_roi_total)
          Lower channel: /entry1/instrument/medipix2/FFI1_medipix2
                         (fallback: /entry1/instrument/medipix2/medipix2_roi_total)
    """
    if not H5_AVAILABLE:
        raise RuntimeError("h5py is required to load NeXus files. pip install h5py")

    with h5py.File(path, "r") as fh:
        i1 = fh["/entry1/I1/I1"][...]
        bragg_off = fh["/entry1/I1/bragg1WithOffset"][...]

        energy_upper = None
        energy_lower = None
        try:
            energy_upper = fh["/entry1/I1/XESEnergyUpper"][...]
        except Exception:
            pass
        try:
            energy_lower = fh["/entry1/I1/XESEnergyLower"][...]
        except Exception:
            pass

        def _load_intensity(det_name: str, dset_name: str, fallback: str):
            try:
                return fh[f"/entry1/instrument/{det_name}/{dset_name}"][...]
            except Exception:
                try:
                    return fh[f"/entry1/instrument/{det_name}/{fallback}"][...]
                except Exception:
                    return None

        upper_int = _load_intensity("medipix1", "FFI1_medipix1", "medipix1_roi_total")
        lower_int = _load_intensity("medipix2", "FFI1_medipix2", "medipix2_roi_total")

    if scan_number is None:
        scan_number = scan.next_index()

    scan.add_scan(
        scan_number,
        {
            "path": path,
            "I1": i1,
            "braggOffset_2d": bragg_off,      # Ω mesh
            "energy_upper_2d": energy_upper,  # ω mesh (Upper) or None
            "energy_lower_2d": energy_lower,  # ω mesh (Lower) or None
            "intensity_upper": upper_int,     # counts (Upper)
            "intensity_lower": lower_int,     # counts (Lower)
            "averaged": False,
            "normalised": False,
        },
    )
    return scan_number

# ----------------------------- XES loaders (1D) -----------------------------
def xes_from_nxs(
    path: str,
    channel: str = "upper",
    *,
    type: str = "RXES",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a 1D spectrum from an I20 .nxs.

    type='RXES' (default):
      X = Ω = /entry1/I1/bragg1WithOffset
      Y = intensity from FFI1_medipix1 (upper) or FFI1_medipix2 (lower)
          (fallback to medipix*_roi_total)

    type='XES':
      X = ω = /entry1/I1/XESEnergyUpper (upper) or /entry1/I1/XESEnergyLower (lower)
      Y = intensity from FFI1_medipix1/FFI1_medipix2 (fallback: medipix*_roi_total)
    """
    if not H5_AVAILABLE:
        raise RuntimeError("h5py is required to load NeXus files. pip install h5py")

    det = "medipix1" if channel.lower().startswith("u") else "medipix2"
    ffi = "FFI1_medipix1" if det == "medipix1" else "FFI1_medipix2"
    roi = f"{det}_roi_total"

    with h5py.File(path, "r") as fh:
        if str(type).strip().upper() == "XES":
            # Emission energy (ω) grid
            energy = (
                fh["/entry1/I1/XESEnergyUpper"][...]
                if det == "medipix1"
                else fh["/entry1/I1/XESEnergyLower"][...]
            )
            try:
                inten = fh[f"/entry1/instrument/{det}/{ffi}"][...]
            except Exception:
                inten = fh[f"/entry1/instrument/{det}/{roi}"][...]
            return _reduce_to_1d(energy, inten)
        else:
            # RXES-style projection to 1D: X=Ω, Y=integrated intensity
            omega = fh["/entry1/I1/bragg1WithOffset"][...]
            try:
                inten = fh[f"/entry1/instrument/{det}/{ffi}"][...]
            except Exception:
                inten = fh[f"/entry1/instrument/{det}/{roi}"][...]
            return _reduce_to_1d(omega, inten)
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
            # Use I20-specific loader
            scan = Scan()
            snum = add_scan_from_i20_ascii(scan, path)
            entry = scan[snum]
            return entry['energy'], entry['intensity']
    except Exception:
        pass  # Fall through to simple format
    
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
    Prefers the channel-specific emission mesh; falls back to Ω (braggOffset) if needed
    to produce a 1D projection. Uses _reduce_to_1d for robust 1D reduction.
    """
    use_upper = channel.lower().startswith("u")
    energy = entry.get("energy_upper_2d") if use_upper else entry.get("energy_lower_2d")
    inten = entry.get("intensity_upper") if use_upper else entry.get("intensity_lower")
    if energy is None:
        # RXES-style projection if only Ω is available or emission missing
        energy = entry.get("braggOffset_2d")
    if energy is None or inten is None:
        return np.array([]), np.array([])
    return _reduce_to_1d(energy, inten)
def xes_from_path(
    path: str,
    channel: str = "upper",
    *,
    type: str = "RXES",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a 1D spectrum from either an I20 .nxs (NeXus) or ASCII file.

    ASCII files:
    - I20 beamline format (.dat) - multi-column with metadata
    - Simple 2-column format - energy, intensity
    
    For .nxs, set type='XES' to use emission energy on X; 'RXES' (default) uses incident energy.
    
    Note: ASCII RXES files are automatically handled - they will be loaded as full 2D scans
    then reduced to 1D by xes_from_ascii().
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".nxs":
        return xes_from_nxs(path, channel=channel, type=type)
    return xes_from_ascii(path)

# ----------------------------- Helper Functions -----------------------------
def _reduce_to_1d(energy: np.ndarray, intensity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduce possibly 2D (mesh-like) energy and intensity to a clean 1D curve Y(X).
    Heuristic: detect the varying axis of the energy mesh and sum the intensity
    along the orthogonal axis.
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
def available_channels(entry: dict) -> list[str]:
    """
    Return a list of available detector channels in this scan entry: ['upper'], ['lower'] or ['upper','lower'].
    A channel is available if both its emission-energy mesh and its intensity exist.
    """
    ch = []
    if entry.get("energy_upper_2d") is not None and entry.get("intensity_upper") is not None:
        ch.append("upper")
    if entry.get("energy_lower_2d") is not None and entry.get("intensity_lower") is not None:
        ch.append("lower")
    return ch

# ----------------------------- ASCII Metadata Parsing -----------------------------
def parse_i20_ascii_metadata(path: str) -> dict:
    """
    Parse I20 beamline ASCII file metadata from header.
    
    Extracts:
    - Scan command to determine scan type (RXES vs XES)
    - Outer/inner axis from command structure
    - Column names from header line
    - Detector channel (upper/lower)
    
    Args:
        path: Path to .dat file
    
    Returns:
        dict with keys: command, scan_type, outer_axis, inner_axis,
                        columns, channel, detector, sample_name, date
    
    Raises:
        ValueError: If file format invalid or scan type unclear
    """
    import re
    
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
            
            # Skip empty lines
            if not line:
                continue
            
            # Stop at data section
            if not line.startswith('#'):
                break
            
            # Remove leading '# '
            content = line[1:].strip()
            
            # Extract command
            if content.startswith('command:'):
                metadata['command'] = content[8:].strip()
            
            # Extract sample name
            elif content.startswith('Sample name:'):
                metadata['sample_name'] = content[12:].strip()
            
            # Extract date
            elif content.startswith('Instrument:') and 'Date:' in content:
                match = re.search(r'Date:\s*(.+)$', content)
                if match:
                    metadata['date'] = match.group(1).strip()
            
            # Column headers (tab-delimited, no leading #)
            elif 'Energy' in content and 'XESEnergy' in content:
                # Split on tabs or multiple spaces
                cols = re.split(r'\t+|\s{2,}', content)
                metadata['columns'] = [c.strip() for c in cols if c.strip()]
    
    # Analyze command to determine scan type
    if metadata['command']:
        cmd = metadata['command']
        
        # Check for bragg scan parameters
        has_bragg_scan = 'bragg1WithOffset' in cmd and \
                        bool(re.search(r'bragg1WithOffset\s+[\d.]+\s+[\d.]+\s+[\d.]+', cmd))
        
        # Check for emission scan parameters (nested scan notation)
        has_emission_scan = bool(re.search(r'XESEnergy(?:Upper|Lower)\s+\[Range', cmd))
        
        if has_bragg_scan and has_emission_scan:
            metadata['scan_type'] = 'RXES'
            
            # Determine outer axis from command structure
            # First scanned parameter in command is outer loop
            bragg_pos = cmd.find('bragg1WithOffset')
            emission_match = re.search(r'XESEnergy(?:Upper|Lower)', cmd)
            emission_pos = emission_match.start() if emission_match else float('inf')
            
            if bragg_pos < emission_pos:
                metadata['outer_axis'] = 'bragg'
                metadata['inner_axis'] = 'emission'
            else:
                metadata['outer_axis'] = 'emission'
                metadata['inner_axis'] = 'bragg'
        
        elif has_emission_scan:
            metadata['scan_type'] = 'XES'
        elif has_bragg_scan:
            metadata['scan_type'] = 'XES'  # Only bragg varies
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
    
    # Validate required fields
    if not metadata['columns']:
        raise ValueError("Could not parse column headers from file")
    if not metadata['channel']:
        raise ValueError("Could not determine detector channel (upper/lower)")
    if not metadata['scan_type']:
        raise ValueError("Could not determine scan type (RXES/XES)")
    
    return metadata
def load_i20_ascii_data(path: str, metadata: dict) -> np.ndarray:
    """
    Load numeric data from I20 ASCII file.
    
    Args:
        path: Path to .dat file
        metadata: Output from parse_i20_ascii_metadata()
    
    Returns:
        2D numpy array: shape (n_rows, n_columns)
    
    Raises:
        ValueError: If data loading fails
    """
    try:
        # Try tab-delimited first (I20 standard)
        data = np.genfromtxt(
            path,
            comments='#',
            delimiter='\t',
            dtype=float,
            invalid_raise=False,
        )
    except Exception as e:
        # Fallback to whitespace-delimited
        try:
            data = np.genfromtxt(
                path,
                comments='#',
                delimiter=None,
                dtype=float,
                invalid_raise=False,
            )
        except Exception as e2:
            raise ValueError(f"Failed to load data from {path}: {e2}")
    
    # Ensure 2D
    data = np.atleast_2d(data)
    
    # Validate number of columns
    if data.shape[1] != len(metadata['columns']):
        import warnings
        warnings.warn(
            f"Column count mismatch: expected {len(metadata['columns'])}, "
            f"got {data.shape[1]}"
        )
    
    return data
def add_scan_from_i20_ascii(
    scan: Scan,
    path: str,
    scan_number: Optional[Any] = None,
) -> Any:
    """
    Load I20 beamline ASCII file into Scan container.
    
    Uses shared grid processing pipeline for both RXES and XES.
    
    Args:
        scan: Scan container
        path: Path to .dat file
        scan_number: Optional scan number (auto-assigned if None)
    
    Returns:
        scan_number used
    
    Raises:
        ValueError: If file invalid or processing fails
    """
    # Step 1: Parse metadata
    metadata = parse_i20_ascii_metadata(path)
    
    # Step 2: Load data
    data = load_i20_ascii_data(path, metadata)
    
    # Step 3: Extract columns
    bragg = data[:, 0]  # Energy column (bragg1WithOffset)
    
    # Find emission column
    emission_col_name = f"XESEnergy{metadata['channel'].capitalize()}"
    try:
        emission_col_idx = metadata['columns'].index(emission_col_name)
    except ValueError:
        raise ValueError(f"Column {emission_col_name} not found in {metadata['columns']}")
    emission = data[:, emission_col_idx]
    
    # Find intensity column
    intensity_col_name = f"FFI1_{metadata['detector']}"
    try:
        intensity_col_idx = metadata['columns'].index(intensity_col_name)
    except ValueError:
        raise ValueError(f"Column {intensity_col_name} not found in {metadata['columns']}")
    intensity = data[:, intensity_col_idx]
    
    # Find monitor column
    try:
        monitor_col_idx = metadata['columns'].index('I1')
        monitor = data[:, monitor_col_idx]
    except ValueError:
        monitor = None
    
    # Auto-assign scan number
    if scan_number is None:
        scan_number = scan.next_index()
    
    # Step 4: Process based on scan type
    if metadata['scan_type'] == 'XES':
        # Simple 1D - filter and store
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
        # Use shared pipeline
        
        # Validate
        validate_rxes_data(bragg, emission, intensity, monitor)
        
        # Analyze grid
        grid_info = analyze_grid_structure(
            bragg, emission,
            outer_axis=metadata.get('outer_axis')
        )
        
        # Normalize to 2D
        grids = normalize_grid_to_2d(
            bragg, emission, intensity, monitor,
            grid_info=grid_info
        )
        
        # Create entry
        entry = create_rxes_scan_entry(
            grids,
            channel=metadata['channel'],
            path=path,
            metadata={'source': 'ascii', **metadata}
        )
        
        scan.add_scan(scan_number, entry)
    
    else:
        raise ValueError(f"Unknown scan type: {metadata['scan_type']}")
    
    return scan_number