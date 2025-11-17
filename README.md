# XESTools - I20 XES/RXES Viewer

A lightweight PySide6 GUI to load, view, and process Diamond I20 RXES maps and XES spectra. It supports viewing RXES maps in incident or energy-loss mode, mutliple RXES cuts, multi-scan XES averaging, area-based normalisation, LMFIT-based background extraction of XES, and exporting results as CSV or NeXus.

The application includes comprehensive stability features:
- Segfault prevention across RXES viewer and background subtraction
- Memory monitoring and automatic crash reporting
- Proper matplotlib resource cleanup and error handling
- System-level fault detection and graceful shutdown
- Timestamped logging with automatic rotation
- Qt exception handling and signal management

## Features

- RXES (.nxs) loader with detector channel selection (Upper/Lower)
- 2D RXES map viewing in:
  - Incident energy mode (Ω vs ω)
  - Energy transfer mode (Ω vs Ω−ω)
- Contour overlays with customizable levels, colors, island filling, and labels
- Profile extraction via silx toolbar (horizontal, vertical, diagonal cuts)
- XES multi-scan workflow:
  - Load multiple 1D spectra (.nxs or ASCII: .txt/.dat/.csv)
  - Tick/untick overlays, average selected, and save average
  - Normalise by area using an external XES and save normalised data/average
- Background extraction on average (or single) XES using LMFIT
  - Save fit log and background/residual CSV
- 1D XES plot with built-in silx fitting toolbar
- Save current dataset as ASCII (CSV) or NeXus (HDF5)

## Architecture

XESTools tries to use a modular architecture designed for modification and implementation for other beamlines and datasets (e.g., offline xes):

### Modules (`xestools/modules/`)
- **Core Data**: `scan.py`, `dataset.py` - Data containers -> Scan() holds scan data as dict of dict.
- **I/O Operations**: `io.py`, `io_workflow.py` - File I/O with silx.io.nxdata support, reading .nxs and ascii formats.
- **Loaders**: `base_loader.py` (generic), `i20_loader.py` (DLS I20)
- **Workflows**: `rxes_workflow.py` (2D), `xes_workflow.py` (1D) - Data treatment and logic for 1D and 2D XES datasets
- **CLI Tools**: `cli_export.py` - pandas/xarray export utilities for python scripting away from GUI.

### Widgets (`xestools/widgets/`)
- **RXES Plot**: `rxes_widget.py`, `rxes_plot.py` - 2D map visualisation
- **Controls**: `io_panel.py`, `xes_panel.py` - GUI panels
- **Dialogs**: `background_dialog.py`, `normalise_dialog.py` - Dialogs for background subtraction (1D XES, mainly for VtC) and normalisation respectively.

### Facility Support
Other facilities can create their own nexus format loaders by defining a schema:

```python
# Example: Single-channel facility
from xestools.modules.base_loader import create_single_channel_schema

MY_SCHEMA = create_single_channel_schema(
    name='MyFacility',
    incident_path='/entry/mono/energy',
    emission_path='/entry/spectrometer/energy',
    intensity_path='/entry/detector/counts',
    monitor_path='/entry/monitor/I0',
)
```

## Installation

### Pythonic
Requirements:
- Python 3.9+
- PySide6
- numpy
- silx (fantastic library used for plotting and some nexus data io)
- h5py (nexus/HDF5 load/save)
- lmfit (1D XES background fitting)
- Optional: psutil (memory monitoring)

``` 
python3 ./main.py
```

### Singularity
A build script is a work in progress. A singularity file exists.

## Usage

### GUI Usage

- RXES
  - Load RXES scan: "Load" → "RXES scan (.nxs)"
  - Select channel (Upper/Lower) and view mode (Incident/Transfer)
  - Use silx profile toolbar to extract line cuts (horizontal, vertical, diagonal)
  - Overlay contours with customizable levels, colors, and labels
  - Save profiles as CSV or export current dataset (CSV/NeXus)
  - Normalise RXES by area: "Load XES…" and select an area on the 1D spectrum

- XES
  - Load multiple spectra: "Load Scans…"
  - Tick to include in overlays and averaging
  - "Average Ticked" to build the average; save with "Save Average"
  - Normalise using external XES: "Load XES…" → select area
    - Save the normalised single/average via "Save Normalised" / "Save Normalised Avg."
  - Background:
    - Optionally "Load Wide Scan…"
    - Run "Background Extraction…", then save the fit log and background/residual CSV
  - Use built-in silx fitting toolbar for peak fitting and analysis

## Command-Line Data Analysis

Export data for advanced analysis with pandas, xarray, or other tools:

### Basic Data Loading

```python
from xestools.modules import i20_loader
from xestools.modules.scan import Scan

# Load RXES scan
scan = Scan()
snum = i20_loader.add_scan_from_nxs(scan, 'scan.nxs')
entry = scan[snum]

# Access data arrays
bragg = entry['braggOffset_2d']  # Incident energy mesh
emission = entry['energy_upper_2d']  # Emission energy mesh
intensity = entry['intensity_upper']  # Detector counts

# Get available channels
channels = i20_loader.available_channels(entry)
print(f"Available channels: {channels}")
```

### Using Workflow Classes

```python
from xestools.modules.rxes_workflow import RXESWorkflow
from xestools.modules.xes_workflow import XESWorkflow
from xestools.modules.scan import Scan
from xestools.modules import i20_loader

# RXES Workflow
scan = Scan()
rxes = RXESWorkflow(scan, i20_loader)
snum = rxes.load_rxes('rxes_scan.nxs')
channels = rxes.get_available_channels()
metadata = rxes.get_scan_metadata()
print(f"Scan shape: {metadata['shape']}")

# XES Workflow
xes = XESWorkflow(i20_loader)
xes.load_spectra(['scan1.nxs', 'scan2.nxs'], channel='lower', scan_type='XES')
x_avg, y_avg = xes.average_selected([0, 1])
print(f"Average: {xes.get_average_name()}")

# Background extraction with LMFIT
result = xes.extract_background(x_avg, y_avg, model='polynomial', params={'degree': 3})
print(result['report'])
```

### Export to pandas DataFrame

```python
from xestools.modules import i20_loader
from xestools.modules.scan import Scan
from xestools.modules.cli_export import scan_to_dataframe

# Load data
scan = Scan()
snum = i20_loader.add_scan_from_nxs(scan, 'scan.nxs')

# Export to pandas DataFrame
df = scan_to_dataframe(scan, snum, channel='upper')

# Now use full pandas capabilities
df.groupby('bragg')['intensity'].mean()
df.to_csv('data.csv')
df.to_parquet('data.parquet')  # More efficient
```

### Export to xarray Dataset

```python
from xestools.modules.cli_export import scan_to_xarray

# Load and export to xarray
ds = scan_to_xarray(scan, snum, channel='upper')

# Use xarray operations
ds.intensity.mean(dim='emission')
ds.to_netcdf('data.nc')
```

### I/O Workflow

```python
from xestools.modules.io_workflow import IOWorkflow
from xestools.modules import i20_loader

io = IOWorkflow(i20_loader)

# Validate file
validation = io.validate_file('scan.nxs')
print(f"Valid: {validation['valid']}, Type: {validation['type']}")

# Get file metadata
meta = io.get_file_metadata('scan.nxs')
print(f"Size: {meta['size_human']}")

# Detect scan type
scan_type = io.detect_scan_type('scan.nxs')  # 'RXES' or 'XES'

# Save 1D spectrum
import numpy as np
io.save_1d_spectrum('output.csv', energy, intensity)
```

See `examples/` directory for more usage patterns.

### Export Formats

- **pandas DataFrame**: Flattened data, best for analysis
- **xarray Dataset**: Preserves 2D structure, best for multidimensional ops
- **HDF5/NeXus**: Full data preservation with silx.io.nxdata support
- **CSV/Parquet**: Standard formats for sharing
- **NetCDF**: Scientific standard format (requires xarray)

### Dependencies

CLI export requires:
```bash
pip install pandas  # Required for DataFrame export
pip install xarray  # Optional, for xarray/NetCDF
pip install silx    # Required for GUI and NXdata support
pip install lmfit   # Optional, for background fitting
```

## GUI Usage Notes

- Load data from I20 NeXus files (.nxs) or ASCII files (.dat/.txt/.csv).

- ASCII file support:
  - **I20 beamline format**: Multi-column .dat files with metadata headers (RXES and XES)
  - **Simple format**: Two-column files with energy and intensity
  - The loader automatically detects the format and loads accordingly

- Raw detector HDF files are detected and rejected; load the scan .nxs instead.

- How the transformation from 2D incident energy to energy transfer is performed:
  - Load the 2D meshes from the .nxs file:
    - Incident energy mesh Ω comes from /entry1/I1/bragg1WithOffset.
    - Emission energy mesh ω comes from /entry1/I1/XESEnergyUpper or /entry1/I1/XESEnergyLower (depending on detector).
    - The intensity Z comes from FFI1_medipix1 (Upper) or FFI1_medipix2 (Lower), with ROI-total as fallback.
  - Reduce the 2D meshes to 1D axes (reduce_axes_for):
    - Compare variability across rows vs columns of the ω mesh to detect its orientation.
    - Build y_omega (emission axis) by taking the median along the orthogonal direction.
    - Build x_Omega (incident axis) likewise from the Ω mesh.
    - Return a transposed flag to indicate if Z must be transposed to align rows with ω and columns with Ω.
  - Align the intensity matrix:
    - If transposed is True, transpose Z so that Z.shape == (len(y_omega), len(x_Omega)).
  - Construct the energy-transfer coordinates for plotting:
    - Keep the X axis as incident energy: X2D = broadcast(x_Omega)[None, :].
    - Compute the energy transfer Δ row-wise: Y2D = X2D − broadcast(y_omega)[:, None].
    - This yields a grid where each pixel's Y value is the energy transfer Δ = Ω − ω.

## File formats

- NeXus/HDF5: .nxs/.h5/.hdf5 (requires h5py, uses silx.io.nxdata for standard NXdata)
- ASCII CSV: headers included, numeric data with up to two columns for XES
- Background export: CSV with columns omega_eV, background, residual

## Project structure (high level)

- xestools/modules
  - `base_loader.py`: Generic loader functions (facility-agnostic)
  - `i20_loader.py`: Diamond I20-specific loader with schema
  - `rxes_workflow.py`: 2D RXES workflow operations
  - `xes_workflow.py`: 1D XES workflow operations
  - `io_workflow.py`: High-level I/O orchestration
  - `io.py`: Low-level ASCII/NeXus save helpers with silx.io.nxdata
  - `scan.py`: In-memory Scan container
  - `cli_export.py`: pandas/xarray export utilities
- xestools/widgets
  - panels: IOPanel, XESPanel
  - dialogs: NormaliseDialog, BackgroundDialog
  - RXESWidget: 2D RXES visualization with silx profile toolbar
- main_gui.py: MainWindow, signal wiring, workflows

## Troubleshooting

- Switching XES Upper/Lower doesn't change curves:
  - The app reloads items for the selected channel; if a reload fails, a warning lists problem files.
- "Save NeXus" disabled:
  - Install h5py: `pip install h5py`.
- Loading XES gives an error about 'channel not found':
  - You have to select the correct channel first using the radio button and then load the scans (bug).
- Application crashes or segfaults:
  - Check the logs/ directory for crash reports and diagnostic information
  - The application includes comprehensive stability features to prevent and recover from crashes

## To do
- Allow 1D XES normalisation after background subtraction.
- Improve background subtraction options with more LMFIT models.
- Add multiple function fitting support in silx fitting window.
- Add key metadata to IO panel (start time, total time, experiment number, sample name, crystal cut).
- Continue refactoring main_gui.py (currently 1810 lines, target ~700 lines).

## Recently Fixed
- **Modular Architecture Refactoring** (Nov 17, 2025):
  - Created technique-specific workflow modules (rxes_workflow.py, xes_workflow.py, io_workflow.py)
  - Created base_loader.py with generic facility-agnostic functions
  - Simplified i20_loader.py with schema-based configuration (1040→564 lines)
  - Simplified NeXus I/O using silx.io.nxdata with @long_name support
  - Renamed RIXS to RXES throughout codebase for consistency
  - Removed legacy widgets (matplotlib backend, intermediate silx backend)
  - Removed feature flags (USE_SILX, USE_RXES_PLOT)
  - Enabled silx fitting toolbar for 1D XES analysis
  - Updated contouring to use silx findContours pattern with full options
- ✅ **ASCII Loader Implementation** (Oct 29, 2025) - Complete I20 beamline ASCII file support:
  - Unified grid processing pipeline shared between NeXus and ASCII loaders
  - Automatic detection of RXES vs XES scan types
  - Auto-detection of outer/inner scan axes
  - Proper 2D grid reconstruction from 1D flat data
  - Seamless integration with existing API (xes_from_ascii, xes_from_path)
  - Backward compatible with simple 2-column ASCII format
- ✅ **Comprehensive Segfault Prevention** (Oct 29, 2025) - Complete overhaul of crash prevention across all components:
  - RXES viewer with proper matplotlib resource cleanup
  - Background subtraction dialog crash prevention with memory management
  - System-level fault detection and graceful shutdown handling
  - Memory monitoring and automatic crash reporting capabilities
  - Qt exception handling and signal management
  - Proper resource cleanup in plot widgets and dialogs
- ✅ RXES normalisation (Oct 28, 2025) - Fixed type parameter in XES loader call
- ✅ XES background extraction segfault (Oct 28, 2025) - Improved resource cleanup and error handling

## License

MIT
