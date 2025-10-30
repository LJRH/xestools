"""
CLI Export Module for I20 XES/RXES Data

Provides export functions for command-line data analysis with pandas and xarray.
Enables batch processing, statistical analysis, and integration with ML pipelines.

Usage:
    from i20_xes.modules import i20_loader
    from i20_xes.modules.scan import Scan
    from i20_xes.modules.cli_export import scan_to_dataframe
    
    # Load data
    scan = Scan()
    snum = i20_loader.add_scan_from_nxs(scan, 'scan.nxs')
    
    # Export to pandas DataFrame
    df = scan_to_dataframe(scan, snum, channel='upper')
    df.to_csv('data.csv')
"""

import os
import numpy as np
from typing import Optional, Dict, Any

# Optional dependencies with graceful fallback
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False
    xr = None

try:
    import h5py
    H5_AVAILABLE = True
except ImportError:
    H5_AVAILABLE = False
    h5py = None


def available_channels_str(entry: dict) -> str:
    """
    Helper: Get human-readable channel availability string.
    
    Args:
        entry: Scan entry dictionary
    
    Returns:
        String like 'upper', 'lower', or 'upper, lower'
    """
    channels = []
    if 'energy_upper_2d' in entry and entry['energy_upper_2d'] is not None:
        if 'intensity_upper' in entry and entry['intensity_upper'] is not None:
            channels.append('upper')
    if 'energy_lower_2d' in entry and entry['energy_lower_2d'] is not None:
        if 'intensity_lower' in entry and entry['intensity_lower'] is not None:
            channels.append('lower')
    return ', '.join(channels) if channels else 'none'


def scan_to_dataframe(scan, scan_number: int, channel: str = 'upper') -> 'pd.DataFrame':
    """
    Export scan data to pandas DataFrame (flattened for analysis).
    
    Args:
        scan: Scan container object
        scan_number: Which scan to export
        channel: 'upper' or 'lower' for NeXus files
    
    Returns:
        pandas DataFrame with columns:
        - For RXES (2D): bragg, emission, intensity, energy_transfer
        - For XES (1D): energy, intensity
    
    Raises:
        ImportError: If pandas not installed
        KeyError: If scan_number not found
        ValueError: If requested channel not available
    
    Example:
        >>> df = scan_to_dataframe(scan, 1, channel='upper')
        >>> df.head()
           bragg  emission  intensity  energy_transfer
        0  9060.1   9044.5       1234           15.6
        >>> df.groupby('bragg')['intensity'].mean()
    """
    if not PANDAS_AVAILABLE:
        raise ImportError(
            "pandas required for DataFrame export. "
            "Install with: pip install pandas"
        )
    
    entry = scan.get(scan_number)
    if not entry:
        raise KeyError(f"Scan {scan_number} not found")
    
    # Check if RXES (2D) or XES (1D)
    is_rxes = 'braggOffset_2d' in entry
    
    if is_rxes:
        # RXES: 2D data with bragg and emission
        bragg = np.asarray(entry['braggOffset_2d'], dtype=float)
        
        # Get emission and intensity for requested channel
        emission_key = f'energy_{channel}_2d'
        intensity_key = f'intensity_{channel}'
        
        if emission_key not in entry or entry[emission_key] is None:
            avail = available_channels_str(entry)
            raise ValueError(
                f"Channel '{channel}' not available. "
                f"Available channels: {avail}"
            )
        
        if intensity_key not in entry or entry[intensity_key] is None:
            avail = available_channels_str(entry)
            raise ValueError(
                f"Channel '{channel}' not available. "
                f"Available channels: {avail}"
            )
        
        emission = np.asarray(entry[emission_key], dtype=float)
        intensity = np.asarray(entry[intensity_key], dtype=float)
        
        # Flatten 2D arrays to 1D (for DataFrame)
        bragg_flat = bragg.ravel()
        emission_flat = emission.ravel()
        intensity_flat = intensity.ravel()
        
        # Calculate energy transfer (Ω - ω)
        energy_transfer = bragg_flat - emission_flat
        
        # Create DataFrame
        df = pd.DataFrame({
            'bragg': bragg_flat,
            'emission': emission_flat,
            'intensity': intensity_flat,
            'energy_transfer': energy_transfer
        })
        
        # Add metadata
        df.attrs['scan_number'] = scan_number
        df.attrs['channel'] = channel
        df.attrs['scan_type'] = 'RXES'
        df.attrs['path'] = entry.get('path', '')
        
    else:
        # XES: 1D spectrum
        # Try to get emission and intensity (may be ASCII or NeXus)
        if 'energy' in entry and 'intensity' in entry:
            # ASCII loader uses 'energy' key
            energy = np.asarray(entry['energy'], dtype=float)
            intensity = np.asarray(entry['intensity'], dtype=float)
        elif 'emission' in entry and 'intensity' in entry:
            # Already reduced to 1D (other loader)
            energy = np.asarray(entry['emission'], dtype=float)
            intensity = np.asarray(entry['intensity'], dtype=float)
        else:
            # NeXus: extract for channel
            emission_key = f'energy_{channel}_2d'
            intensity_key = f'intensity_{channel}'
            
            if emission_key not in entry or entry[emission_key] is None:
                avail = available_channels_str(entry)
                raise ValueError(
                    f"Channel '{channel}' not available. "
                    f"Available channels: {avail}"
                )
            
            if intensity_key not in entry or entry[intensity_key] is None:
                avail = available_channels_str(entry)
                raise ValueError(
                    f"Channel '{channel}' not available. "
                    f"Available channels: {avail}"
                )
            
            energy = np.asarray(entry[emission_key], dtype=float).ravel()
            intensity = np.asarray(entry[intensity_key], dtype=float).ravel()
        
        # Create DataFrame
        df = pd.DataFrame({
            'energy': energy,
            'intensity': intensity
        })
        
        # Add metadata
        df.attrs['scan_number'] = scan_number
        df.attrs['scan_type'] = 'XES'
        df.attrs['path'] = entry.get('path', '')
    
    return df


def scan_to_xarray(scan, scan_number: int, channel: str = 'upper') -> 'xr.Dataset':
    """
    Export scan data to xarray Dataset (preserves 2D structure).
    
    Args:
        scan: Scan container object
        scan_number: Which scan to export
        channel: 'upper' or 'lower' for NeXus files
    
    Returns:
        xarray Dataset with:
        - Coordinates: bragg, emission (for RXES) or energy (for XES)
        - Variables: intensity
        - Attributes: metadata
    
    Raises:
        ImportError: If xarray not installed
        KeyError: If scan_number not found
        ValueError: If requested channel not available
    
    Example:
        >>> ds = scan_to_xarray(scan, 1, channel='upper')
        >>> ds.intensity.plot()  # Direct plotting
        >>> ds.to_netcdf('data.nc')  # Save to NetCDF
    """
    if not XARRAY_AVAILABLE:
        raise ImportError(
            "xarray required for Dataset export. "
            "Install with: pip install xarray"
        )
    
    entry = scan.get(scan_number)
    if not entry:
        raise KeyError(f"Scan {scan_number} not found")
    
    # Check if RXES (2D) or XES (1D)
    is_rxes = 'braggOffset_2d' in entry
    
    if is_rxes:
        # RXES: 2D data
        bragg = np.asarray(entry['braggOffset_2d'], dtype=float)
        
        # Get emission and intensity for requested channel
        emission_key = f'energy_{channel}_2d'
        intensity_key = f'intensity_{channel}'
        
        if emission_key not in entry or entry[emission_key] is None:
            avail = available_channels_str(entry)
            raise ValueError(
                f"Channel '{channel}' not available. "
                f"Available channels: {avail}"
            )
        
        if intensity_key not in entry or entry[intensity_key] is None:
            avail = available_channels_str(entry)
            raise ValueError(
                f"Channel '{channel}' not available. "
                f"Available channels: {avail}"
            )
        
        emission = np.asarray(entry[emission_key], dtype=float)
        intensity = np.asarray(entry[intensity_key], dtype=float)
        
        # Create 2D Dataset
        ds = xr.Dataset(
            data_vars={
                'intensity': (['point'], intensity.ravel()),
                'energy_transfer': (['point'], (bragg - emission).ravel())
            },
            coords={
                'bragg': (['point'], bragg.ravel()),
                'emission': (['point'], emission.ravel()),
            },
            attrs={
                'scan_number': scan_number,
                'channel': channel,
                'scan_type': 'RXES',
                'path': entry.get('path', '')
            }
        )
        
    else:
        # XES: 1D spectrum
        if 'energy' in entry and 'intensity' in entry:
            # ASCII loader uses 'energy' key
            energy = np.asarray(entry['energy'], dtype=float)
            intensity = np.asarray(entry['intensity'], dtype=float)
        elif 'emission' in entry and 'intensity' in entry:
            # Other loaders use 'emission' key
            energy = np.asarray(entry['emission'], dtype=float)
            intensity = np.asarray(entry['intensity'], dtype=float)
        else:
            emission_key = f'energy_{channel}_2d'
            intensity_key = f'intensity_{channel}'
            
            if emission_key not in entry or entry[emission_key] is None:
                avail = available_channels_str(entry)
                raise ValueError(
                    f"Channel '{channel}' not available. "
                    f"Available channels: {avail}"
                )
            
            if intensity_key not in entry or entry[intensity_key] is None:
                avail = available_channels_str(entry)
                raise ValueError(
                    f"Channel '{channel}' not available. "
                    f"Available channels: {avail}"
                )
            
            energy = np.asarray(entry[emission_key], dtype=float).ravel()
            intensity = np.asarray(entry[intensity_key], dtype=float).ravel()
        
        # Create 1D Dataset
        ds = xr.Dataset(
            data_vars={
                'intensity': (['energy'], intensity)
            },
            coords={
                'energy': energy
            },
            attrs={
                'scan_number': scan_number,
                'scan_type': 'XES',
                'path': entry.get('path', '')
            }
        )
    
    return ds


def export_scan_to_hdf5(scan, output_path: str) -> None:
    """
    Export entire Scan container to HDF5 (full data preservation).
    
    Saves all scan entries with complete metadata and arrays.
    Use import_scan_from_hdf5() for round-trip loading.
    
    Args:
        scan: Scan container object
        output_path: Path to output HDF5 file
    
    Raises:
        ImportError: If h5py not installed
    
    Example:
        >>> export_scan_to_hdf5(scan, 'archive.h5')
        >>> scan2 = import_scan_from_hdf5('archive.h5')
    """
    if not H5_AVAILABLE:
        raise ImportError(
            "h5py required for HDF5 export. "
            "Install with: pip install h5py"
        )
    
    with h5py.File(output_path, 'w') as f:
        # Save each scan entry
        for scan_num, entry in scan.items():
            grp = f.create_group(f'scan_{scan_num}')
            
            # Save arrays
            for key, value in entry.items():
                if isinstance(value, (np.ndarray, list)):
                    grp.create_dataset(key, data=value)
                elif isinstance(value, (str, int, float)):
                    grp.attrs[key] = value
                elif value is None:
                    grp.attrs[key] = 'None'


def import_scan_from_hdf5(path: str):
    """
    Import Scan container from HDF5 file.
    
    Round-trip loading of data saved with export_scan_to_hdf5().
    
    Args:
        path: Path to HDF5 file
    
    Returns:
        Scan container (dictionary) with restored data
    
    Raises:
        ImportError: If h5py not installed
    
    Example:
        >>> scan = import_scan_from_hdf5('archive.h5')
        >>> df = scan_to_dataframe(scan, 1)
    """
    if not H5_AVAILABLE:
        raise ImportError(
            "h5py required for HDF5 import. "
            "Install with: pip install h5py"
        )
    
    from .scan import Scan
    scan = Scan()
    
    with h5py.File(path, 'r') as f:
        # Load each scan entry
        for grp_name in f.keys():
            if not grp_name.startswith('scan_'):
                continue
            
            scan_num = int(grp_name.split('_')[1])
            grp = f[grp_name]
            
            entry = {}
            
            # Load arrays
            for key in grp.keys():
                entry[key] = grp[key][...]
            
            # Load attributes
            for key, value in grp.attrs.items():
                if value == 'None':
                    entry[key] = None
                else:
                    entry[key] = value
            
            scan[scan_num] = entry
    
    return scan
