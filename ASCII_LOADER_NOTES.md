# ASCII Loader Implementation - Issues and Rollback Notes

**Date**: October 28, 2025  
**Status**: ROLLED BACK - Issues identified, implementation incomplete  
**Rollback Point**: Commit `9386d5d` (after RXES normalisation and segfault fixes)

---

## What Was Attempted

### Goal
Implement full ASCII loading support for both:
1. **1D XES spectra** - Simple and beamline format
2. **2D RXES scans** - Reconstruct meshes from scan points

### What Was Implemented

#### 1. XES ASCII Loader (`xes_from_ascii()`)
- **Commits**: d1d4ecc, e460a68, 76c59be
- **Features Added**:
  - Column header detection from comment lines
  - Channel-aware column selection (Upper/Lower)
  - Priority system: XESEnergyUpper/Lower > Energy
  - Intensity priority: FFI1_medipix1/2 > FF_medipix > I1
  - Backward compatible with simple two-column format

#### 2. RXES ASCII Loader (`add_scan_from_ascii()`)
- **Commits**: 6e40457, 942c836
- **Features Added**:
  - Automatic RXES detection (both Ω and ω varying)
  - 2D mesh reconstruction from sparse scan points
  - Support for both detector channels
  - GUI integration (file dialog filters updated)

---

## Issues Identified

### 🔴 Issue 1: XES Loader Can Load RXES Files (No Guard Rails)
**Problem**: The `xes_from_ascii()` function doesn't check if BOTH energies are varying. It will happily load an RXES scan as a 1D XES spectrum, taking only the first two columns.

**Impact**: User confusion, incorrect data interpretation, workflow errors

**Example**:
```python
# This should fail but doesn't:
x, y = xes_from_ascii("RXES_scan.dat", channel="upper")
# Returns: Energy and XESEnergyUpper as 1D arrays
# Problem: Both are varying! This is 2D data being forced into 1D
```

**Required Fix**:
- Add detection logic to check if BOTH columns vary significantly
- Raise clear error: "This appears to be an RXES scan (2D data). Please use the RXES loader."
- Threshold: If both energy columns vary by >0.5 eV, it's RXES

---

### 🔴 Issue 2: RXES Reconstruction Creates Pixelated/Incorrect Plot
**Problem**: The 2D mesh reconstruction algorithm produces incorrect results. Plot is very pixelated and doesn't show proper data structure.

**Possible Causes**:
1. **Mesh sizing issue**: Creating 51×87 mesh for 867 points leaves many NaN values
2. **Index mapping problem**: `argmin(abs(unique - value))` may map multiple points to same mesh position
3. **Transpose issue**: Mesh orientation may be incorrect (rows vs columns)
4. **Interpolation needed**: Sparse data needs interpolation, not just nearest-neighbor mapping
5. **Scan pattern misunderstood**: May not be simple raster scan

**Algorithm Used** (flawed):
```python
# For each scan point:
omega_idx = np.argmin(np.abs(unique_omega - omega[i]))
emission_idx = np.argmin(np.abs(unique_emission - emission[i]))
intensity_mesh[emission_idx, omega_idx] = intensity[i]
```

**Problem**: This creates a sparse mesh with many gaps (NaN values), which plotting interprets as pixelation.

---

## Sample File Analysis

**File**: `i20_xes/data/rxes/XES_ZnZSM5_air500_279192_1.dat`

**Structure**:
- 867 data points (not a complete mesh)
- Columns: Energy (Ω), XESEnergyUpper (ω), XESBraggUpper, I1, FF_medipix1, FFI1_medipix1, Time
- Ω range: 9658.00 - 9662.80 eV (4.8 eV span, 87 unique values)
- ω range: 8632.00 - 8646.99 eV (15 eV span, 51 unique values)

**Scan Pattern**:
- Fixed Ω, scan ω (51 points per Ω)
- Step Ω, scan ω again
- ~17 Ω steps total
- **This is NOT a complete rectangular mesh!**

**Expected Mesh**: 87×51 = 4437 points  
**Actual Points**: 867 (~20% filled)  
**Conclusion**: This is a sparse/diagonal RXES scan, not a full map

---

## Why It Failed

### Root Cause Analysis

1. **Assumption Error**: Code assumed scan points would naturally fill a complete mesh
2. **No Interpolation**: Gaps in mesh were left as NaN, causing pixelation
3. **No Validation**: Didn't check mesh fill percentage or warn about sparse data
4. **Guard Rails Missing**: XES loader can accidentally load RXES data
5. **Incomplete Testing**: Only tested with programmatic loader, not GUI visualization

### What Would Be Needed for Success

1. **Proper interpolation** for sparse meshes (scipy.interpolate.griddata)
2. **Mesh quality validation** (warn if <50% filled)
3. **Guard rails in xes_from_ascii()** (detect and reject 2D data)
4. **Better scan pattern detection** (detect diagonal/ROI scans vs full maps)
5. **GUI testing** with actual visualization before committing

---

## Lessons Learned

### Technical Lessons
1. ✓ Always test with GUI visualization, not just programmatic loading
2. ✓ Sparse data requires interpolation, not just nearest-neighbor mapping
3. ✓ Add guard rails to prevent misuse of loaders
4. ✓ Validate mesh quality before accepting reconstruction
5. ✓ Check transpose/orientation issues early

### Process Lessons
1. ✓ Test edge cases (sparse scans, incomplete meshes)
2. ✓ Verify assumptions about data structure with user
3. ✓ Start with simple case (complete mesh) before handling sparse data
4. ✓ Get user feedback earlier in development cycle

---

## Recommended Next Steps

### For Next Session - RXES ASCII Loading (Priority Order)

#### Phase 1: Add Guard Rails to XES Loader (Essential)
```python
def xes_from_ascii(path: str, channel: str = "upper"):
    # ... parse header and load data ...
    
    # NEW: Check if this is RXES data
    if header_line:
        has_omega = 'Energy' in col_map
        has_xes = 'XESEnergyUpper' in col_map or 'XESEnergyLower' in col_map
        
        if has_omega and has_xes:
            # Check if BOTH vary significantly
            omega_range = data[:, col_map['Energy']].max() - data[:, col_map['Energy']].min()
            xes_col = 'XESEnergyUpper' if use_upper else 'XESEnergyLower'
            if xes_col in col_map:
                xes_range = data[:, col_map[xes_col]].max() - data[:, col_map[xes_col]].min()
                
                if omega_range > 0.5 and xes_range > 0.5:
                    raise ValueError(
                        f"This appears to be an RXES scan (both Ω and ω vary). "
                        f"Ranges: Ω={omega_range:.2f} eV, ω={xes_range:.2f} eV. "
                        f"Please use 'Load RXES scan' instead of 'Load XES spectra'."
                    )
    
    # ... continue with 1D loading ...
```

#### Phase 2: Fix RXES Mesh Reconstruction (Complex)

**Option A: Interpolate Sparse Data** (Recommended)
```python
from scipy.interpolate import griddata

def reconstruct_mesh(omega, emission, intensity):
    # Get unique sorted values
    unique_omega = np.unique(omega)
    unique_emission = np.unique(emission)
    
    # Create regular grid
    omega_grid, emission_grid = np.meshgrid(unique_omega, unique_emission)
    
    # Interpolate scan points onto regular grid
    points = np.column_stack([omega, emission])
    intensity_grid = griddata(
        points, intensity, 
        (omega_grid, emission_grid),
        method='cubic',  # or 'linear'
        fill_value=np.nan
    )
    
    return emission_grid, omega_grid, intensity_grid
```

**Option B: Keep Sparse, Add Warning** (Simpler)
```python
# After creating mesh, check fill percentage
fill_pct = np.sum(~np.isnan(intensity_mesh)) / intensity_mesh.size * 100

if fill_pct < 50:
    logging.warning(
        f"RXES mesh only {fill_pct:.1f}% filled. "
        f"Plot may be pixelated. Consider interpolation."
    )
```

**Option C: Support Multiple Formats**
- Detect if scan is complete rectangular mesh (>80% filled) → use direct mapping
- Detect if scan is sparse/diagonal (<50% filled) → use interpolation
- Provide user option in dialog: "Interpolate sparse data? [Yes/No]"

#### Phase 3: Testing Strategy
1. Test with simple complete mesh first (create test file)
2. Test with sparse diagonal scan (the provided file)
3. Test XES loader rejects RXES data
4. Test RXES loader with both channels
5. **Visualize in GUI** before committing
6. Compare with NeXus-loaded RXES for validation

---

## Files Modified (Need Rollback)

### To Be Reverted
- `i20_xes/modules/i20_loader.py` - xes_from_ascii() and add_scan_from_ascii()
- `i20_xes/main_gui.py` - Load dialog filters and routing
- `README.md` - ASCII documentation
- `PLAN.md` - Status updates

### To Be Kept
- `PLAN.md` - General structure (will update status)
- Sample file: `i20_xes/data/rxes/XES_ZnZSM5_air500_279192_1.dat`

---

## Safe Rollback Procedure

```bash
# 1. Create this documentation file (done)
# 2. Reset to last good commit
git reset --hard 9386d5d

# 3. Keep sample file (it's useful for testing)
git checkout dev-i20xes -- i20_xes/data/rxes/XES_ZnZSM5_air500_279192_1.dat

# 4. Add documentation files
git add ASCII_LOADER_NOTES.md

# 5. Commit rollback
git commit -m "Rollback ASCII loader implementation - issues identified
..."
```

---

## Current Working State (After Rollback)

### ✅ Confirmed Working
- RXES normalisation (commit 42d6178)
- XES background extraction segfault fix (commit ad1ec79)
- RXES loading from NeXus files
- XES loading from NeXus files
- All basic RXES/XES functionality

### ❌ Not Working
- ASCII RXES loading
- ASCII XES loading (reverted to broken state)

### 📝 Status
- Branch: dev-i20xes
- Safe state: Commit 9386d5d
- 2 critical bugs fixed, documented, tested
- Ready for next session to tackle ASCII properly

---

## Success Criteria for Next Attempt

Before considering ASCII loading "done":
1. ✅ XES loader REJECTS RXES files with clear error message
2. ✅ RXES ASCII produces non-pixelated plot
3. ✅ Visual comparison with NeXus-loaded RXES shows same structure
4. ✅ Sparse scans handled gracefully (interpolation or warning)
5. ✅ Both channels work for both formats
6. ✅ **Tested in GUI with visualization before commit**

---

**End of Notes - Ready for Next Session**
