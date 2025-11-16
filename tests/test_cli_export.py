#!/usr/bin/env python3
"""Test CLI export functionality."""

import sys
sys.path.insert(0, '.')

from xestools.modules import i20_loader
from xestools.modules.scan import Scan
from xestools.modules.cli_export import (
    scan_to_dataframe,
    scan_to_xarray,
    export_scan_to_hdf5,
    import_scan_from_hdf5,
)

print("Testing CLI Export Module")
print("=" * 70)

# Test 1: RXES to DataFrame
print("\n1. RXES → DataFrame")
scan = Scan()
snum = i20_loader.add_scan_from_nxs(scan, 'xestools/data/rxes/279517_1.nxs')
df = scan_to_dataframe(scan, snum, channel='upper')
print(f"   ✓ Shape: {df.shape}")
print(f"   ✓ Columns: {list(df.columns)}")
assert 'bragg' in df.columns
assert 'emission' in df.columns
assert 'intensity' in df.columns
assert 'energy_transfer' in df.columns
print("   ✓ PASSED")

# Test 2: XES to DataFrame (from ASCII)
print("\n2. XES (ASCII) → DataFrame")
scan2 = Scan()
snum2 = i20_loader.add_scan_from_i20_ascii(
    scan2, 'xestools/data/vtc/ZnO_standard_280754_1.dat'
)
df2 = scan_to_dataframe(scan2, snum2)
print(f"   ✓ Shape: {df2.shape}")
print(f"   ✓ Columns: {list(df2.columns)}")
assert 'energy' in df2.columns
assert 'intensity' in df2.columns
print("   ✓ PASSED")

# Test 3: RXES to xarray
print("\n3. RXES → xarray")
try:
    ds = scan_to_xarray(scan, snum, channel='upper')
    print(f"   ✓ Dimensions: {dict(ds.dims)}")
    print(f"   ✓ Variables: {list(ds.data_vars)}")
    assert 'intensity' in ds.data_vars
    assert 'emission' in ds.coords
    assert 'bragg' in ds.coords
    print("   ✓ PASSED")
except ImportError:
    print("   ⊘ SKIPPED (xarray not installed)")

# Test 4: Export/Import HDF5
print("\n4. Export → Import HDF5")
export_scan_to_hdf5(scan, '/tmp/test_scan.h5')
scan_loaded = import_scan_from_hdf5('/tmp/test_scan.h5')
assert snum in scan_loaded
assert scan_loaded[snum]['path'] == scan[snum]['path']
print("   ✓ PASSED")

# Test 5: CSV Export
print("\n5. CSV Export")
df.to_csv('/tmp/test_rxes.csv', index=False)
import os
size = os.path.getsize('/tmp/test_rxes.csv')
print(f"   ✓ File size: {size} bytes")
print("   ✓ PASSED")

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED")
