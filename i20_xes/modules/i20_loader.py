from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import numpy as np

try:
    import h5py
    H5_AVAILABLE = True
except Exception:
    H5_AVAILABLE = False

from .scan import Scan

# ---------- File heuristics ----------

def is_probably_detector_hdf(path: str) -> bool:
    """
    Heuristic to detect raw detector HDF (not the scan .nxs).
    Returns True if file contains /entry/data/data and lacks /entry1/I1/I1.
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

# ---------- Axes helper ----------

def reduce_axes_for(emission_2d: np.ndarray,
                    bragg_offset_2d: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Compute 1D axes ensuring:
      - Y (rows) is emission energy ω (from emission_2d)
      - X (cols) is incident energy Ω (from bragg_offset_2d)
    Returns (y_emission, x_incident, transposed) and indicates if Z must be transposed.
    """
    e = np.asarray(emission_2d)
    b = np.asarray(bragg_offset_2d) if bragg_offset_2d is not None else None
    if e.ndim != 2:
        raise ValueError("emission_2d must be 2D")
    n0, n1 = e.shape

    # Variation across rows (per row across columns) vs across columns (per column across rows)
    row_var = float(np.nanmean(np.std(e, axis=1)))
    col_var = float(np.nanmean(np.std(e, axis=0)))

    if row_var >= col_var:
        # Emission varies across columns -> emission dimension is columns.
        # To make emission on Y (rows), transpose.
        y = np.nanmedian(e, axis=0)  # len n1 (columns)
        x = np.nanmedian(b, axis=1) if b is not None else np.arange(n0, dtype=float)  # len n0 (rows)
        transposed = True
    else:
        # Emission varies across rows -> already on Y (rows).
        y = np.nanmedian(e, axis=1)  # len n0 (rows)
        x = np.nanmedian(b, axis=0) if b is not None else np.arange(n1, dtype=float)  # len n1 (columns)
        transposed = False

    return y.ravel(), x.ravel(), transposed

# ---------- I20 scan loader (.nxs) ----------

def add_scan_from_nxs(scan: Scan, path: str, scan_number: Optional[Any] = None) -> Any:
    """
    Load an I20 scan .nxs file (entry1 layout) and append to Scan.

    Reads:
      - I1 grid: /entry1/I1/I1
      - Emission grids: /entry1/I1/XESEnergyUpper and/or XESEnergyLower
      - Incident (Ω) grid: /entry1/I1/bragg1WithOffset
      - Intensities:
          Upper:  /entry1/instrument/medipix1/FFI1_medipix1 (fallback: medipix1_roi_total)
          Lower:  /entry1/instrument/medipix2/FFI1_medipix2 (fallback: medipix2_roi_total)
    """
    if not H5_AVAILABLE:
        raise RuntimeError("h5py is required to load NeXus files. pip install h5py")

    with h5py.File(path, "r") as fh:
        # Grids under entry1/I1
        i1 = fh["/entry1/I1/I1"][...]
        bragg_off = fh["/entry1/I1/bragg1WithOffset"][...]
        # Emission grids
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

        # Intensities
        upper_int = None
        lower_int = None
        try:
            upper_int = fh["/entry1/instrument/medipix1/FFI1_medipix1"][...]
        except Exception:
            try:
                upper_int = fh["/entry1/instrument/medipix1/medipix1_roi_total"][...]
            except Exception:
                pass
        try:
            lower_int = fh["/entry1/instrument/medipix2/FFI1_medipix2"][...]
        except Exception:
            try:
                lower_int = fh["/entry1/instrument/medipix2/medipix2_roi_total"][...]
            except Exception:
                pass

    if scan_number is None:
        scan_number = scan.next_index()

    scan.add_scan(scan_number, {
        "path": path,
        "I1": i1,                                 # shape (n0, n1)
        "braggOffset_2d": bragg_off,              # Ω grid, shape (n0, n1)
        "energy_upper_2d": energy_upper,          # ω grid (Upper) or None
        "energy_lower_2d": energy_lower,          # ω grid (Lower) or None
        "intensity_upper": upper_int,             # FFI1_medipix1 or ROI
        "intensity_lower": lower_int,             # FFI1_medipix2 or ROI
        "averaged": False,
        "normalised": False,
    })
    return scan_number