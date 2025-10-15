from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import os
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
    row_var = float(np.nanmean(np.std(e, axis=1)))
    col_var = float(np.nanmean(np.std(e, axis=0)))

    if row_var >= col_var:
        # Emission varies across columns -> transpose to put emission on rows
        y = np.nanmedian(e, axis=0)  # len n1
        x = np.nanmedian(b, axis=1) if b is not None else np.arange(e.shape[0], dtype=float)
        transposed = True
    else:
        # Emission varies across rows -> already on rows
        y = np.nanmedian(e, axis=1)  # len n0
        x = np.nanmedian(b, axis=0) if b is not None else np.arange(e.shape[1], dtype=float)
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
            val = None
            try:
                val = fh[f"/entry1/instrument/{det_name}/{dset_name}"][...]
            except Exception:
                try:
                    val = fh[f"/entry1/instrument/{det_name}/{fallback}"][...]
                except Exception:
                    pass
            return val

        upper_int = _load_intensity("medipix1", "FFI1_medipix1", "medipix1_roi_total")
        lower_int = _load_intensity("medipix2", "FFI1_medipix2", "medipix2_roi_total")

    if scan_number is None:
        scan_number = scan.next_index()

    scan.add_scan(scan_number, {
        "path": path,
        "I1": i1,
        "braggOffset_2d": bragg_off,       # Ω grid
        "energy_upper_2d": energy_upper,   # ω grid (Upper)
        "energy_lower_2d": energy_lower,   # ω grid (Lower)
        "intensity_upper": upper_int,      # FFI1_medipix1 (or ROI)
        "intensity_lower": lower_int,      # FFI1_medipix2 (or ROI)
        "averaged": False,
        "normalised": False,
    })
    return scan_number

# ---------- XES (Ω vs intensity) from I20 sources ----------

def _reduce_omega_and_intensity_to_1d(omega: np.ndarray, inten: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduce Ω-grid and intensity to a 1D XES curve Y(Ω).
    - If both are 1D: align and return.
    - If both are 2D of same shape: collapse along emission dimension.
    - If shapes differ but compatible: attempt collapse by matching the Ω dimension.
    - Otherwise flatten and sort by Ω.
    """
    om = np.asarray(omega, dtype=float)
    y = np.asarray(inten, dtype=float)

    # Mask invalids
    def _finite(a): return np.isfinite(a)
    if om.ndim == 1 and y.ndim == 1:
        n = min(om.size, y.size)
        x = om[:n]
        yi = y[:n]
    elif om.ndim == 2 and y.ndim == 2 and om.shape == y.shape:
        row_var = float(np.nanmean(np.std(om, axis=1)))
        col_var = float(np.nanmean(np.std(om, axis=0)))
        if row_var < col_var:
            # Ω varies along rows (downwards): make x from median across rows -> columns
            x = np.nanmedian(om, axis=0)
            yi = np.nansum(y, axis=0)
        else:
            x = np.nanmedian(om, axis=1)
            yi = np.nansum(y, axis=1)
    else:
        # Try align by matching one dimension
        if om.ndim == 1 and y.ndim == 2:
            if y.shape[0] == om.size:
                x = om
                yi = np.nansum(y, axis=1)
            elif y.shape[1] == om.size:
                x = om
                yi = np.nansum(y, axis=0)
            else:
                x = om.ravel()
                yi = np.nansum(y, axis=tuple(range(y.ndim)))
        elif om.ndim == 2 and y.ndim == 1:
            if om.shape[0] == y.size:
                x = np.nanmedian(om, axis=1)
                yi = y
            elif om.shape[1] == y.size:
                x = np.nanmedian(om, axis=0)
                yi = y
            else:
                x = om.ravel()
                yi = y.ravel()
        else:
            # Last resort
            x = om.ravel()
            yi = y.ravel()

    # Sort by Ω and drop NaNs
    order = np.argsort(x)
    x = x[order]
    yi = yi[order]
    ok = _finite(x) & _finite(yi)
    return x[ok], yi[ok]

def xes_from_nxs(path: str, channel: str = "upper") -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a 1D XES spectrum from an I20 .nxs:
      X = Ω = /entry1/I1/bragg1WithOffset
      Y = intensity from FFI1_medipix1 (upper) or FFI1_medipix2 (lower)
        fallback to medipix*_roi_total.
    If grids are 2D, integrates intensity across emission to get Y(Ω).
    """
    if not H5_AVAILABLE:
        raise RuntimeError("h5py is required to load NeXus files. pip install h5py")
    det = "medipix1" if channel.lower().startswith("u") else "medipix2"
    dset = "FFI1_medipix1" if det == "medipix1" else "FFI1_medipix2"
    fallback = f"{det}_roi_total"

    with h5py.File(path, "r") as fh:
        omega = fh["/entry1/I1/bragg1WithOffset"][...]
        inten = None
        try:
            inten = fh[f"/entry1/instrument/{det}/{dset}"][...]
        except Exception:
            inten = fh[f"/entry1/instrument/{det}/{fallback}"][...]

    return _reduce_omega_and_intensity_to_1d(omega, inten)

def xes_from_ascii(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    ASCII two-column: X=Ω, Y=intensity.
    """
    arr = np.genfromtxt(path, comments="#", delimiter=None, dtype=float)
    arr = np.atleast_2d(arr)
    if arr.shape[1] < 2:
        raise ValueError("ASCII XES file must have at least two columns (Ω, intensity)")
    x = arr[:, 0]
    y = arr[:, 1]
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]

def xes_from_path(path: str, channel: str = "upper") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load XES from either .nxs (I20) or ASCII two-column.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".nxs":
        return xes_from_nxs(path, channel=channel)
    return xes_from_ascii(path)