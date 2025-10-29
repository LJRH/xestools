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
    ASCII two-column XES: X=energy (ω or Ω), Y=intensity.
    """
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
    Load a 1D spectrum from either an I20 .nxs (NeXus) or an ASCII two-column file.

    For .nxs, set type='XES' to use emission energy on X; 'RXES' (default) uses incident energy.
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