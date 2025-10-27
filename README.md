# I20 XES/RXES Viewer

A lightweight PySide6 GUI to load, view, and process Diamond I20 RXES maps and XES spectra. It supports viewing RXES maps in incident or energy-loss mode, mutliple RXES cuts, multi-scan XES averaging, area-based normalisation, LMFIT-based background extraction of XES, and exporting results as CSV or NeXus.

## Features

- RXES (.nxs) loader with detector channel selection (Upper/Lower)
- 2D RXES map viewing in:
  - Incident energy mode (Ω vs ω)
  - Energy transfer mode (Ω vs Ω−ω)
- ROI line profiles with adjustable bandwidth and CSV export
- XES multi-scan workflow:
  - Load multiple 1D spectra (.nxs or ASCII: .txt/.dat/.csv)
  - Tick/untick overlays, average selected, and save average
  - Normalise by area using an external XES and save normalised data/average
- Background extraction on average (or single) XES, optional wide scan input
  - Save fit log and background/residual CSV
- Save current dataset as ASCII (CSV) or NeXus (HDF5)


## Installation

### Pythonic
Requirements:
- Python 3.9+
- PySide6
- numpy
- Optional (recommended): h5py (NeXus/HDF5 load/save)
- Optional: lmfit (background fitting), matplotlib (plotting, if used by PlotWidget)

``` python3 ./main.py```

### Singularity
A build script is a work in progress. A singularity file exists.

## Usage

- RXES
  - Load RXES scan: “Load” → “RXES scan (.nxs)”
  - Select channel (Upper/Lower) and view mode (Incident/Transfer)
  - Add/move ROI lines to extract profiles; adjust bandwidth
  - Save profiles as CSV or export current dataset (CSV/NeXus)
  - Normalise RXES by area: “Load XES…” and select an area on the 1D spectrum

- XES
  - Load multiple spectra: “Load Scans…”
  - Tick to include in overlays and averaging
  - “Average Ticked” to build the average; save with “Save Average”
  - Normalise using external XES: “Load XES…” → select area
    - Save the normalised single/average via “Save Normalised” / “Save Normalised Avg.”
  - Background:
    - Optionally “Load Wide Scan…”
    - Run “Background Extraction…”, then save the fit log and background/residual CSV

Notes:
- Load data from the I20 nexus files.

- ASCII XES expects two numeric columns: energy, intensity (.txt/.dat/.csv). Currently, loading beamline ascii files DOES NOT WORK.

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
    - This yields a grid where each pixel’s Y value is the energy transfer Δ = Ω − ω.

## File formats

- NeXus/HDF5: .nxs/.h5/.hdf5 (requires h5py)
- ASCII CSV: headers included, numeric data with up to two columns for XES
- Background export: CSV with columns omega_eV, background, residual

## Project structure (high level)

- i20_xes/modules
  - loader (i20_loader): RXES/XES readers, axis reduction
  - io: ASCII/NeXus save helpers
  - scan: in-memory Scan container
- i20_xes/widgets
  - panels: IOPanel, RXESPanel, XESPanel
  - dialogs: NormaliseDialog, BackgroundDialog
  - PlotWidget: 1D/2D plotting and ROI tools
- main_gui.py: MainWindow, signal wiring, workflows

## Troubleshooting

- Switching XES Upper/Lower doesn’t change curves:
  - The app reloads items for the selected channel; if a reload fails, a warning lists problem files.
- “Save NeXus” disabled:
  - Install h5py: `pip install h5py`.
- Loading XES gives an error about 'channel not found':
  - You have to select the correct channel first using the radio button and then load the scans (bug).

## License

MIT
     