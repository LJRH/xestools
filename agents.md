# XESTools Development Progress

## Project Overview
**Project Name**: xestools (I20 XES/RXES Viewer)  
**Purpose**: A lightweight PySide6 GUI to load, view, and process Diamond I20 RXES maps and XES spectra  
**Status**: Semi-working version with known issues

## Current State Analysis (October 28, 2025)

### Environment
- **Python Version**: 3.12.8 (Anaconda)
- **Working Directory**: `/mnt/media_hermes/Work/i20xes`
- **Git Repository**: Yes

### Dependencies Status
- ❌ **PySide6**: Not installed (required for GUI)
- ❌ **h5py**: Not installed (required for NeXus/HDF5 support)
- ✅ **numpy**: Available
- ⚠️ **lmfit**: Unknown (optional, for background fitting)
- ⚠️ **matplotlib**: Unknown (optional, for plotting)

### Project Structure
```
i20_xes/
├── data/
│   ├── rxes/ (279496_1.nxs, 279517_1.nxs)
│   └── vtc/ (280754_1.nxs, 280772_1.nxs, 280782_1.nxs, 280792_1.nxs)
├── modules/
│   ├── dataset.py
│   ├── i20_loader.py
│   ├── init.py
│   ├── io.py
│   └── scan.py
├── widgets/
│   ├── background_dialog.py
│   ├── init.py
│   ├── io_panel.py
│   ├── normalise_dialog.py
│   ├── plot_widget.py
│   ├── roi_panel.py
│   └── xes_panel.py
├── __init__.py (missing)
└── main_gui.py
main.py
README.md
Singularity
```

## Known Issues

### 1. ASCII XES Loading (Critical)
**Location**: `i20_xes/modules/i20_loader.py:186-197` (function `xes_from_ascii`)  
**Issue**: Loading beamline ASCII files DOES NOT WORK  
**Description**: The current implementation uses `np.genfromtxt` with `delimiter=None`, which may not correctly parse beamline-specific ASCII formats  
**Impact**: Cannot load ASCII XES data from beamline instruments  
**Status**: Not fixed

### 2. Channel Selection Bug (High Priority)
**Location**: XES loading workflow  
**Issue**: Must select correct channel (Upper/Lower) BEFORE loading scans  
**Description**: The channel radio button must be set before file loading; changing it afterward doesn't reload properly  
**Impact**: User workflow issue - causes confusion and requires reloading files  
**Status**: Not fixed  
**Note**: README.md line 104 documents this as a known bug

### 3. Code Quality Issues

#### a. Duplicate Code in main_gui.py
**Location**: `i20_xes/main_gui.py:1321-1408`  
**Issue**: Duplicated/orphaned code blocks that appear to be old versions of methods  
**Lines affected**:
- Lines 1321-1324: Duplicate background button enable logic
- Lines 1326-1378: Duplicate `_refresh_xes_plot` method
- Lines 1380-1386: Duplicate `update_status_label` method
- Lines 1388-1408: Duplicate `on_save_nexus` method
**Impact**: Code maintainability and potential bugs  
**Status**: Not fixed

#### b. Missing __init__.py
**Location**: `i20_xes/__init__.py`  
**Issue**: File not found (may be intentionally empty or missing)  
**Impact**: Package imports may not work correctly  
**Status**: Not verified

### 4. Switching XES Upper/Lower Doesn't Change Curves
**Location**: XES panel channel switching  
**Issue**: Documented in README.md (lines 99-100)  
**Description**: "The app reloads items for the selected channel; if a reload fails, a warning lists problem files"  
**Status**: Partially implemented (error handling exists)

## Features Working

### RXES Features ✅
- Load RXES scan (.nxs files)
- Detector channel selection (Upper/Lower)
- 2D RXES map viewing modes:
  - Incident energy mode (Ω vs ω)
  - Energy transfer mode (Ω vs Ω−ω)
- ROI line profiles with adjustable bandwidth
- CSV export of profiles
- RXES normalization by external XES area

### XES Features ✅
- Load multiple 1D XES spectra (.nxs format)
- Tick/untick overlays
- Average selected spectra
- Save average spectrum
- Normalize by area using external XES
- Background extraction (with lmfit)
- Save fit log and background/residual CSV

### Save/Export Features ✅
- Save current dataset as ASCII (CSV)
- Save as NeXus (HDF5) - when h5py installed

## Code Architecture Notes

### Data Flow
1. **RXES Loading**: `i20_loader.add_scan_from_nxs()` → `Scan` container → `refresh_rxes_view()` → `DataSet` → `PlotWidget`
2. **XES Loading**: `on_xes_load_scans()` → `_xes_items` list → `_refresh_xes_plot()` → `PlotWidget`
3. **Axis Reduction**: `reduce_axes_for()` and `_reduce_to_1d()` convert 2D meshes to plottable axes

### Key Components
- **Scan**: In-memory container for loaded scan data
- **DataSet**: Plotting data structure (1D or 2D)
- **PlotWidget**: Main visualization component
- **IOPanel**: File loading/saving controls
- **RXESPanel**: RXES-specific controls (channel, mode, extraction)
- **XESPanel**: XES-specific controls (multi-scan workflow)

### State Management
- `self.scan`: Scan container (can hold multiple scans)
- `self.dataset`: Currently displayed DataSet
- `self._xes_items`: List of loaded XES items
- `self._xes_avg`: Averaged XES spectrum
- `self.current_scan_number`: Active RXES scan

## Potential Improvements Identified

### High Priority
1. Fix ASCII loader to handle beamline-specific formats
2. Fix channel selection to auto-reload when changed
3. Clean up duplicate code in main_gui.py

### Medium Priority
4. Better error messages for missing channels
5. Auto-detect and suggest correct channel on load
6. Improve ASCII format detection/parsing

### Low Priority
7. Add unit tests
8. Improve code documentation
9. Refactor long methods in main_gui.py
10. Add more robust error handling throughout

## Next Steps
**Awaiting user input on priorities:**
1. Which specific issues to fix first?
2. What functionality is most broken/urgent?
3. Any additional issues not yet documented?

## Development Notes
- Main entry point: `main.py` (launches `MainWindow`)
- Verbose logging enabled for debugging
- Fault handler enabled for crash diagnostics
- Singularity container build script exists (WIP)

---
**Last Updated**: October 28, 2025  
**Agent**: OpenCode Assistant
