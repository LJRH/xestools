import os
from datetime import datetime
from typing import Optional
import numpy as np
from .dataset import DataSet

try:
    import h5py
    H5_AVAILABLE = True
except Exception:
    H5_AVAILABLE = False

def load_path(path: str) -> DataSet:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".nxs", ".nx5", ".h5", ".hdf", ".hdf5"):
        if not H5_AVAILABLE:
            raise RuntimeError("h5py is required to load NeXus/HDF5 files. Install with: pip install h5py")
        return load_nexus(path)
    return load_ascii(path)

def load_ascii(path: str) -> DataSet:
    data = np.genfromtxt(path, comments="#", delimiter=None, dtype=float)
    if data.ndim == 1:
        data = np.atleast_2d(data)
    if data.ndim != 2 or data.size == 0:
        raise ValueError("ASCII file does not contain a 1D or 2D numeric array")

    rows, cols = data.shape

    if cols == 2:
        x = data[:, 0]
        y = data[:, 1]
        return DataSet("1D", x=x, y=y, xlabel="Energy", ylabel="Intensity", source=path)

    if cols == 3:
        xs = np.unique(data[:, 0])
        ys = np.unique(data[:, 1])
        if xs.size * ys.size != data.shape[0]:
            raise ValueError("Triplet ASCII does not form a rectangular grid")
        x_sorted = np.sort(xs)
        y_sorted = np.sort(ys)
        Z = np.empty((y_sorted.size, x_sorted.size), dtype=float)
        Z[:] = np.nan
        m = {(round(r[0], 12), round(r[1], 12)): r[2] for r in data}
        for iy, yv in enumerate(y_sorted):
            for ix, xv in enumerate(x_sorted):
                Z[iy, ix] = m.get((round(xv, 12), round(yv, 12)), np.nan)
        return DataSet("2D", x=x_sorted, y=y_sorted, z=Z,
                       xlabel="Emission", ylabel="Incident", zlabel="Intensity",
                       source=path)

    if rows >= 2 and cols >= 2:
        Z = data
        x = np.arange(Z.shape[1], dtype=float)
        y = np.arange(Z.shape[0], dtype=float)
        return DataSet("2D", x=x, y=y, z=Z, xlabel="X", ylabel="Y", zlabel="Intensity", source=path)

    raise ValueError("Unrecognized ASCII layout")

def load_nexus(path: str) -> DataSet:
    with h5py.File(path, "r") as f:
        g = _find_first_nxdata(f)
        if g is None:
            raise ValueError("No NXdata group found in file")

        signal_name: Optional[str] = None
        if "signal" in g.attrs:
            attr = g.attrs["signal"]
            signal_name = attr.decode() if isinstance(attr, (bytes, np.bytes_)) else str(attr)
        elif "data" in g:
            signal_name = "data"
        else:
            for k, v in g.items():
                if isinstance(v, h5py.Dataset):
                    signal_name = k
                    break
        if not signal_name or signal_name not in g:
            raise ValueError("No signal dataset found in NXdata")

        signal = g[signal_name][()]
        axes_names = []
        if "axes" in g.attrs:
            axes_attr = g.attrs["axes"]
            if isinstance(axes_attr, (bytes, np.bytes_)):
                axes_names = [axes_attr.decode()]
            elif isinstance(axes_attr, (list, tuple, np.ndarray)):
                axes_names = [a.decode() if isinstance(a, (bytes, np.bytes_)) else str(a) for a in axes_attr]
            else:
                axes_names = [str(axes_attr)]

        if signal.ndim == 1:
            x = None
            if axes_names:
                axn = axes_names[0]
                if axn in g:
                    try:
                        x = g[axn][()]
                    except Exception:
                        x = None
            return DataSet("1D",
                           x=x if x is not None and x.ndim == 1 and x.size == signal.size else np.arange(signal.size),
                           y=signal,
                           xlabel=axes_names[0] if axes_names else "Energy",
                           ylabel="Intensity",
                           source=path)
        if signal.ndim == 2:
            x = None
            y = None
            if len(axes_names) >= 2:
                ax0, ax1 = axes_names[:2]
                if ax0 in g:
                    try:
                        a0 = g[ax0][()]
                        if a0.ndim == 1 and a0.size == signal.shape[0]:
                            y = a0
                    except Exception:
                        pass
                if ax1 in g:
                    try:
                        a1 = g[ax1][()]
                        if a1.ndim == 1 and a1.size == signal.shape[1]:
                            x = a1
                    except Exception:
                        pass
            return DataSet("2D",
                           x=x if x is not None else np.arange(signal.shape[1]),
                           y=y if y is not None else np.arange(signal.shape[0]),
                           z=signal,
                           xlabel=axes_names[1] if len(axes_names) >= 2 else "X",
                           ylabel=axes_names[0] if len(axes_names) >= 1 else "Y",
                           zlabel="Intensity",
                           source=path)
        raise ValueError(f"Only 1D/2D datasets are supported (got {signal.ndim}D)")

def _find_first_nxdata(h5: "h5py.File"):
    def is_nxdata(obj):
        try:
            return isinstance(obj, h5py.Group) and obj.attrs.get("NX_class", b"").decode() == "NXdata"
        except Exception:
            val = obj.attrs.get("NX_class")
            if isinstance(val, str):
                return val == "NXdata"
            if isinstance(val, (bytes, np.bytes_)):
                return val.decode() == "NXdata"
            return False

    def dfs(group):
        for k in group:
            obj = group[k]
            if is_nxdata(obj):
                return obj
            if isinstance(obj, h5py.Group):
                found = dfs(obj)
                if found is not None:
                    return found
        return None

    return dfs(h5)

def save_ascii(path: str, ds: DataSet):
    """
    Save dataset to ASCII (CSV), UTF-8 encoded.
    - 1D: two columns X,Y
    - 2D grid (x,y 1D): matrix with header row (x) and first column (y)
    - 2D curvilinear (x2d,y2d): triplets X,Y,Z per row
    """
    if ds.kind == "1D":
        x = ds.x if ds.x is not None else np.arange(len(ds.y))
        arr = np.column_stack([x, ds.y])
        header = f"{ds.xlabel or 'x'},{ds.ylabel or 'y'}"
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(header + "\n")
            np.savetxt(fh, arr, delimiter=",", comments="", fmt="%.10g")
        return

    # 2D
    Z = np.asarray(ds.z)
    if ds.x2d is not None and ds.y2d is not None:
        X = np.asarray(ds.x2d)
        Y = np.asarray(ds.y2d)
        # Flatten valid points
        m = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
        arr = np.column_stack([X[m], Y[m], Z[m]])
        header = f"{ds.xlabel or 'X'},{ds.ylabel or 'Y'},{ds.zlabel or 'Z'}"
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(header + "\n")
            np.savetxt(fh, arr, delimiter=",", comments="", fmt="%.10g")
        return

    # Grid case with 1D x,y
    if ds.x is not None and ds.y is not None:
        x = np.asarray(ds.x)
        y = np.asarray(ds.y)
        out = np.empty((y.size + 1, x.size + 1), dtype=float)
        out[:] = np.nan
        out[0, 1:] = x
        out[1:, 0] = y
        out[1:, 1:] = Z
        header = f"top row: {ds.xlabel or 'x'}, first col: {ds.ylabel or 'y'}"
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(f"# {header}\n")
            np.savetxt(fh, out, delimiter=",", comments="", fmt="%.10g")
        return

    # Fallback: write Z as a matrix
    with open(path, "w", encoding="utf-8", newline="") as fh:
        np.savetxt(fh, Z, delimiter=",", comments="", fmt="%.10g")
        
def save_nexus(path: str, ds: DataSet):
    if not H5_AVAILABLE:
        raise RuntimeError("h5py is required to save NeXus files. Install with: pip install h5py")

    with h5py.File(path, "w") as f:
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        nxdata = entry.create_group("data")
        nxdata.attrs["NX_class"] = "NXdata"

        if ds.kind == "1D":
            x = ds.x if ds.x is not None else np.arange(len(ds.y))
            dsy = nxdata.create_dataset("y", data=np.asarray(ds.y))
            dsx = nxdata.create_dataset("x", data=np.asarray(x))
            nxdata.attrs["signal"] = "y"
            nxdata.attrs["axes"] = "x"
            dsy.attrs["long_name"] = ds.ylabel or "Intensity"
            dsx.attrs["long_name"] = ds.xlabel or "Energy"
        else:
            Z = np.asarray(ds.z)
            x = ds.x if ds.x is not None else np.arange(Z.shape[1])
            y = ds.y if ds.y is not None else np.arange(Z.shape[0])
            dsz = nxdata.create_dataset("z", data=Z)
            dsx = nxdata.create_dataset("x", data=np.asarray(x))
            dsy = nxdata.create_dataset("y", data=np.asarray(y))
            nxdata.attrs["signal"] = "z"
            nxdata.attrs["axes"] = np.array(["y", "x"], dtype="S")
            dsz.attrs["long_name"] = ds.zlabel or "Intensity"
            dsx.attrs["long_name"] = ds.xlabel or "Emission"
            dsy.attrs["long_name"] = ds.ylabel or "Incident"

        f.attrs["file_time"] = datetime.now().isoformat(timespec="seconds")