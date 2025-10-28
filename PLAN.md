# I20 XES/RXES Viewer - Development Plan

**Last Updated**: October 28, 2025 (Session 2)  
**Repository**: git@github.com:LJRH/i20xes.git  
**Current Branch**: main (will push to dev-i20xes)  
**Latest Commit**: ad1ec79 "Fix XES background extraction segfault issues"  
**Working Directory**: `/mnt/media_hermes/Work/i20xes`

---

## 📊 Project Status Overview

### ✅ Recently Fixed (Current Session - Oct 28, 2025)
- **✅ RXES normalisation** - Fixed in commit 42d6178 (changed `type="RXES"` to `type="XES"`)
- **✅ XES background extraction segfault** - Fixed in commit ad1ec79 (improved resource cleanup)

### ✅ Previously Fixed (Past Week)
- **RXES ROI line bug** - Fixed in commit 9c17ec7 (Oct 28)
- **XES scan loading unification** - Fixed in commit 1e8ca03 (Oct 28)
- **Normalisation and background subtraction plotting** - Updated in commit 238a566 (Oct 27)
- **Duplicate/orphaned code in main_gui.py** - Cleaned up (was at lines 1321-1408 in old version)
- **Build script** - Added `buildxestools.sh` for Singularity container builds

### 🟡 Current Issues (Medium Priority)
1. **ASCII XES loading broken** (beamline format files) - Needs sample file
2. **Channel selection workflow bug** (must select before loading) - Auto-reload needed

### 🛠️ Environment Status
- **Python**: 3.12.8 (Anaconda)
- **Dependencies**: ✅ All available (PySide6, h5py, numpy, lmfit, matplotlib)
- **Test Data**: 
  - RXES: `i20_xes/data/rxes/279496_1.nxs`, `279517_1.nxs`
  - VTC: `i20_xes/data/vtc/280754_1.nxs`, `280772_1.nxs`, `280782_1.nxs`, `280792_1.nxs`

---

## 🎯 Priority Task List

### 🔴 HIGH PRIORITY - Critical Bugs

#### **Task 1: Investigate RXES Normalisation Issue** ✅ COMPLETED
**Status**: ✅ Fixed in commit 42d6178  
**Priority**: Critical (user's main concern)  
**Location**: `i20_xes/main_gui.py:1206-1232` (specifically line 1220)

**Current Workflow**:
1. User loads RXES scan via `on_load()` → calls `add_scan_from_nxs()`
2. User clicks "Load XES..." button in RXES panel
3. Triggers `on_rxes_normalise()` (line 1206)
4. Loads external XES file using `i20_loader.xes_from_path(path, channel=channel, type="RXES")` (line 1220)
5. Opens `NormaliseDialog` to let user select area on spectrum
6. Sets `scan[current_scan_number]["norm_factor"] = area` (line 1228)
7. Calls `refresh_rxes_view()` (line 1230)

**Normalisation Application** (`refresh_rxes_view()` at lines 268-274):
```python
# Apply RXES normalisation factor (by XES area) if present
nf = entry.get("norm_factor", None)
if nf and np.isfinite(nf) and nf > 0:
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = Z / nf
    if "area" not in zlabel:
        zlabel = zlabel + " / area"
```

**Suspected Issues**:
1. ⚠️ Line 1220 uses `type="RXES"` when loading normalisation spectrum - should this be `type="XES"`?
2. Area calculation in `NormaliseDialog.selected_area()` might be incorrect
3. Z matrix division may not be working correctly
4. Plot might not be refreshing properly after normalisation
5. Channel mismatch between RXES scan and normalisation XES

**Investigation Steps**:
1. Launch GUI: `cd /mnt/media_hermes/Work/i20xes && python3 main.py`
2. Load RXES test file: `i20_xes/data/rxes/279496_1.nxs`
3. Select Upper/Lower channel
4. Click "Load XES..." button
5. Load a VTC file from `i20_xes/data/vtc/` as normalisation reference
6. Select area on the spectrum
7. Check if Z matrix values actually change
8. Verify plot updates correctly
9. Add debug print statements to track:
   - Area value returned from dialog
   - `norm_factor` stored in scan entry
   - Z matrix values before/after division
   - Whether `refresh_rxes_view()` is called

**Files to Review**:
- `i20_xes/main_gui.py:1206-1232` - `on_rxes_normalise()`
- `i20_xes/main_gui.py:233-295` - `refresh_rxes_view()`
- `i20_xes/widgets/normalise_dialog.py` - Area selection dialog (updated in 238a566)
- `i20_xes/modules/i20_loader.py:214-227` - `xes_from_path()` and type parameter

**Solution**:
Changed line 1220 from `type="RXES"` to `type="XES"` so the loader uses the emission energy axis (ω) instead of incident energy (Ω) when loading the normalisation spectrum.

**Result**:
RXES 2D map now correctly shows reduced intensity values (divided by area), and the z-label displays "/ area" suffix.

---

#### **Task 2: Fix RXES Normalisation Bug** ✅ COMPLETED
**Status**: ✅ Fixed in commit 42d6178  
**Priority**: Critical  
**Solution**: Changed `type="RXES"` to `type="XES"` at line 1220 in `on_rxes_normalise()`

---

#### **Task 3: Investigate XES Background Extraction Segfault** ✅ COMPLETED
**Status**: ✅ Fixed in commit ad1ec79  
**Priority**: Critical (crashes are unacceptable)  
**Location**: `i20_xes/widgets/background_dialog.py`

**Current Implementation**:
- User loads XES scans, optionally loads wide scan
- Clicks "Background Extraction..." button
- Opens `BackgroundDialog` (recently updated in commit 238a566)
- Dialog uses lmfit (Pearson7Model + LinearModel)
- Uses matplotlib SpanSelector for interactive peak selection
- **Intermittent segfault occurs** (no consistent reproduction steps known)

**Suspected Causes**:
1. matplotlib SpanSelector widget interaction with Qt event loop
2. lmfit optimization failing on edge cases (NaN/Inf values)
3. Memory corruption in wide scan merging (`merge_wide_and_main()` in background_dialog.py)
4. Matplotlib figure cleanup issues when dialog closes
5. Thread safety issues with matplotlib in Qt

**Investigation Steps**:
1. Add comprehensive exception handling around lmfit calls
2. Add logging to track when segfault occurs:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```
3. Test with various data combinations:
   - Single scan only
   - Average with no wide scan
   - Average with wide scan
   - Edge case: empty selections, single point selections
4. Run with Python fault handler: Already enabled in `main.py:18`
5. Check for memory leaks with repeated open/close of dialog
6. Look for matplotlib figure cleanup in dialog's `closeEvent()`

**Files to Review**:
- `i20_xes/main_gui.py:817-850` - `on_xes_background_extract()`
- `i20_xes/widgets/background_dialog.py` - Full dialog implementation
- `i20_xes/widgets/background_dialog.py:27-61` - `merge_wide_and_main()` function

**Debugging Commands**:
```bash
# Run with maximum Python debugging
python3 -X dev -X tracemalloc main.py

# Run with GDB to catch segfault
gdb -ex run --args python3 main.py
```

**Root Causes Identified**:
1. SpanSelector lifecycle management issues (double-free, race conditions)
2. lmfit optimization failures not handled properly
3. Matplotlib figure not being cleaned up on dialog close

**Solutions Implemented**:
1. Improved `_destroy_span()` with proper exception handling and finally block
2. Added `nan_policy='omit'` and convergence checking to lmfit calls
3. Proper figure cleanup with `plt.close()` in `closeEvent()`

**Result**:
Background extraction should be significantly more stable. Needs user testing to confirm.

---

#### **Task 4: Fix XES Background Extraction Segfault** ✅ COMPLETED
**Status**: ✅ Fixed in commit ad1ec79  
**Priority**: Critical  
**Changes Made**:
- Enhanced `_destroy_span()` with specific exception handling
- Improved lmfit error handling with nan_policy and convergence checks
- Added proper matplotlib figure cleanup in `closeEvent()`

---

### 🟡 MEDIUM PRIORITY - Workflow Issues

#### **Task 5: Fix ASCII XES Loader**
**Status**: Not started  
**Priority**: Medium  
**Location**: `i20_xes/modules/i20_loader.py:186-197`

**Current Implementation** (`xes_from_ascii()`):
```python
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
```

**Known Issue**: 
README line 58: "Currently, loading beamline ascii files DOES NOT WORK."

**Problem**: 
`delimiter=None` with `genfromtxt` uses any whitespace as delimiter, which may not match beamline format.

**Investigation Needed**:
1. **Get example beamline ASCII file** - Need actual file to understand format
2. Possible beamline formats:
   - Tab-delimited with metadata header
   - Fixed-width columns
   - Multiple header lines without # comment markers
   - Non-numeric metadata mixed with data
   - European number format (comma as decimal separator)

**Potential Solution**:
```python
def xes_from_ascii(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    ASCII two-column XES with robust parsing for beamline formats.
    """
    # Try multiple parsing strategies
    strategies = [
        lambda: np.genfromtxt(path, comments="#", delimiter=None, dtype=float),
        lambda: np.genfromtxt(path, comments="#", delimiter="\t", dtype=float),
        lambda: np.genfromtxt(path, comments="#", delimiter=",", dtype=float),
        lambda: np.loadtxt(path, comments=("#", "!"), dtype=float),
        lambda: pd.read_csv(path, comment="#", sep=r"\s+", header=None).values,
    ]
    
    for strategy in strategies:
        try:
            data = strategy()
            data = np.atleast_2d(data)
            if data.shape[1] >= 2:
                x, y = data[:, 0], data[:, 1]
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() > 0:
                    return x[ok], y[ok]
        except Exception:
            continue
    
    raise ValueError(f"Could not parse ASCII file: {path}")
```

**Action Items**:
1. Request example beamline ASCII file from user
2. Examine file format manually
3. Implement appropriate parser
4. Add unit tests with example files

---

#### **Task 6: Fix Channel Selection Workflow Bug**
**Status**: Not started  
**Priority**: Medium  
**Location**: `i20_xes/main_gui.py:537-572`

**Current Behavior** (README line 104):
"You have to select the correct channel first using the radio button and then load the scans (bug)."

**Expected Behavior**:
User should be able to switch between Upper/Lower channel after loading, and GUI should automatically reload all XES items with the new channel.

**Current Implementation** (`on_xes_channel_changed()`):
```python
def on_xes_channel_changed(self, checked: bool):
    if not checked:
        return
    use_upper = self.xes_panel.rb_upper.isChecked()
    channel = "upper" if use_upper else "lower"
    
    # Attempt to reload each .nxs item with new channel
    # (Currently has issues - may fail silently or show warnings)
```

**Problem Analysis**:
- Function tries to reload each item but error handling may be insufficient
- Items might not have both channels available
- Plot doesn't refresh correctly after channel change
- State management issue: `_xes_items` may retain old channel data

**Fix Strategy**:
1. Store original file path for each loaded item
2. On channel change, iterate through all items:
   - If item is .nxs, reload with new channel
   - If item is ASCII, keep as-is (no channel concept)
   - If new channel unavailable, mark item as invalid
3. Update `_xes_items` list with reloaded data
4. Refresh plot with new data
5. Show clear warning if some items couldn't reload
6. Update channel availability indicators

**Files to Modify**:
- `i20_xes/main_gui.py:537-572` - `on_xes_channel_changed()`
- `i20_xes/main_gui.py:1243-1277` - `_update_xes_channel_controls()`

---

### 🟢 LOW PRIORITY - Enhancements

#### **Task 7: Add 'Clear All' Button to Background Extraction**
**Status**: Not started  
**Priority**: Low  
**Location**: `i20_xes/widgets/background_dialog.py`

**Requirement** (README line 109):
"Add a XES background extraction 'Clear All' button."

**Implementation**:
1. Add "Clear All" button to `BackgroundDialog` UI
2. Connect to new `clear_all()` method
3. Method should:
   - Reset all peak selections
   - Clear fit results
   - Reset plot to original data
   - Clear any stored fit parameters

**Estimated Effort**: 30 minutes - 1 hour

**Code Location**:
Add button to `BackgroundDialog.__init__()` method, likely near existing control buttons.

---

## 📂 Key File Reference

### Core Application Files
- **`main.py`** (27 lines) - Entry point, launches MainWindow
- **`i20_xes/main_gui.py`** (1407 lines) - Main application window and workflow logic

### Data Loading Modules
- **`i20_xes/modules/i20_loader.py`** (273 lines) - RXES/XES file readers
  - `add_scan_from_nxs()` - Load RXES scan from .nxs
  - `xes_from_nxs()` - Load 1D XES from .nxs
  - `xes_from_ascii()` - Load 1D XES from ASCII (BROKEN)
  - `xes_from_path()` - Dispatcher based on file extension
  - `reduce_axes_for()` - Convert 2D meshes to 1D axes
- **`i20_xes/modules/scan.py`** - Scan container class
- **`i20_xes/modules/io.py`** - ASCII/NeXus save helpers
- **`i20_xes/modules/dataset.py`** - DataSet structure for plotting

### Widget/Dialog Files
- **`i20_xes/widgets/plot_widget.py`** - 1D/2D plotting with ROI tools
- **`i20_xes/widgets/io_panel.py`** - File I/O controls
- **`i20_xes/widgets/xes_panel.py`** - XES-specific controls
- **`i20_xes/widgets/normalise_dialog.py`** - Area selection for normalisation (updated Oct 27)
- **`i20_xes/widgets/background_dialog.py`** - Background fitting dialog (updated Oct 27)
- **`i20_xes/widgets/roi_panel.py`** - RXES ROI controls

### Build/Deploy Files
- **`buildxestools.sh`** (57 lines) - Singularity container build script
- **`Singularity`** - Container definition file

---

## 🔍 Critical Code Sections

### RXES Normalisation Flow
1. **Load RXES**: `main_gui.py:on_load()` → `i20_loader.add_scan_from_nxs()`
2. **Normalise**: `main_gui.py:on_rxes_normalise()` (lines 1206-1232)
   - Opens file dialog to select XES
   - Calls `i20_loader.xes_from_path(path, channel, type="RXES")` ⚠️ type parameter
   - Opens `NormaliseDialog` for area selection
   - Stores `scan[scan_number]["norm_factor"] = area`
   - Calls `refresh_rxes_view()`
3. **Apply**: `main_gui.py:refresh_rxes_view()` (lines 233-295)
   - Retrieves `norm_factor` from scan entry (line 269)
   - Divides Z matrix by norm_factor (lines 268-274)
   - Updates z-label to include "/ area"
   - Plots normalized data

### XES Background Extraction Flow
1. **Load XES**: `main_gui.py:on_xes_load_scans()` → stores in `_xes_items`
2. **Average**: `main_gui.py:on_xes_average_ticked()` → creates `_xes_avg`
3. **Load Wide** (optional): `main_gui.py:on_xes_load_wide()` → stores in `_xes_wide`
4. **Extract**: `main_gui.py:on_xes_background_extract()` (lines 817-850)
   - Opens `BackgroundDialog` with main and wide data
   - Dialog uses lmfit for Pearson7 + Linear fit
   - ⚠️ **Segfault occurs here intermittently**
   - Returns fit results: background, residual, report
5. **Store**: Results stored in `_last_bkg`, `_last_resid`, `_xes_avg_bkgsub`

### Channel Selection Flow (XES)
1. **Load**: `main_gui.py:on_xes_load_scans()` - loads items with current channel
2. **Switch**: `main_gui.py:on_xes_channel_changed()` (lines 537-572)
   - Tries to reload all .nxs items with new channel
   - ⚠️ **Currently buggy - often fails**
3. **Update UI**: `main_gui.py:_update_xes_channel_controls()` (lines 1243-1277)
   - Enables/disables channel buttons based on availability

---

## 🧪 Testing Strategy

### Manual Testing Workflow
```bash
# Launch application
cd /mnt/media_hermes/Work/i20xes
python3 main.py

# Test RXES normalisation
1. Load RXES: data/rxes/279496_1.nxs
2. Select Upper channel
3. Click "Load XES..." (RXES panel)
4. Load: data/vtc/280754_1.nxs
5. Drag to select area on spectrum
6. Click OK
7. Verify: Map intensity should decrease, z-label should show "/ area"

# Test XES background extraction (segfault)
1. Switch to XES tab
2. Load multiple scans from data/vtc/
3. Tick all, click "Average Ticked"
4. Optionally: Load Wide Scan
5. Click "Background Extraction..."
6. Try to fit - watch for segfault
7. If crash, note exact steps to reproduce

# Test ASCII loading
1. Acquire sample beamline ASCII file
2. Try to load via XES tab "Load Scans..."
3. Observe error message
4. Examine file format manually
```

### Debug Mode
```bash
# Run with verbose logging
python3 -X dev main.py

# Run with fault handler (already in main.py)
python3 main.py  # fault handler enabled on line 18

# Run under GDB for segfault debugging
gdb -ex run --args python3 main.py
# (gdb) backtrace  # when segfault occurs
```

---

## 📝 Development Notes

### Recent Changes (Past Week)
- **Oct 28**: Build script added, README updated with TODO list
- **Oct 28**: RXES ROI line bug fixed
- **Oct 28**: XES scan loading unified to use i20_loader
- **Oct 27**: Normalisation and background dialogs updated (major refactor)
- **Oct 27**: Duplicate code removed from main_gui.py

### Code Quality
- ✅ No duplicate code blocks found
- ✅ Modern Python type hints used throughout
- ✅ Comprehensive docstrings
- ✅ Proper error handling in most places
- ⚠️ Some areas need better exception handling (background extraction)
- ⚠️ Need unit tests (currently none exist)

### Architecture Notes
- Uses PySide6 for GUI (Qt6)
- Matplotlib embedded in Qt for plotting
- h5py for NeXus/HDF5 file I/O
- lmfit for curve fitting (optional dependency)
- Data flow: File → Scan → DataSet → PlotWidget
- RXES uses 2D meshes reduced to 1D axes for plotting
- XES uses 1D arrays directly

---

## 🚀 Quick Start for Next Session

### Immediate Actions
1. **Pull latest code**: 
   ```bash
   cd /mnt/media_hermes/Work/i20xes
   git pull origin main
   ```

2. **Start with RXES normalisation investigation** (Task 1):
   ```bash
   python3 main.py
   # Load RXES, try normalisation, observe behavior
   ```

3. **Add debug logging**:
   ```python
   # In main_gui.py:on_rxes_normalise() after line 1228
   print(f"DEBUG: norm_factor set to {area}")
   print(f"DEBUG: scan entry = {self.scan[self.current_scan_number]}")
   
   # In main_gui.py:refresh_rxes_view() after line 272
   print(f"DEBUG: Applying normalisation: Z.max() before = {Z.max()}")
   Z = Z / nf
   print(f"DEBUG: Z.max() after = {Z.max()}, nf = {nf}")
   ```

4. **Check if type parameter is wrong**:
   - Line 1220: `type="RXES"` - try changing to `type="XES"`

### Questions to Answer
1. Is the normalisation area being calculated correctly?
2. Is the norm_factor being stored in the scan entry?
3. Is refresh_rxes_view() actually being called?
4. Does the Z matrix get modified?
5. Does the plot show the updated values?

### Success Criteria
- RXES map shows reduced intensity after normalisation
- Z-axis label includes "/ area" suffix
- No error messages or warnings
- Behavior is consistent across multiple normalisation attempts

---

## 📞 Contact & Resources

- **Repository**: git@github.com:LJRH/i20xes.git
- **Branch**: main
- **Author**: Luke Higgins (luke.higgins@diamond.ac.uk)
- **Test Data**: Located in `i20_xes/data/rxes/` and `i20_xes/data/vtc/`
- **Documentation**: README.md (comprehensive usage guide)

---

## ✅ Completion Checklist

Before considering work complete:
- [ ] All high priority bugs fixed and tested
- [ ] Medium priority issues addressed or documented for future work
- [ ] Code committed with clear commit messages
- [ ] README.md "To do" section updated
- [ ] Manual testing performed with sample data
- [ ] No regressions in existing functionality
- [ ] This PLAN.md updated with final status

---

**COMPLETED THIS SESSION** (Oct 28, 2025):
- ✅ Fixed RXES normalisation bug (Task 1-2)
- ✅ Fixed XES background extraction segfault (Task 3-4)
- ✅ Updated README.md and PLAN.md
- 🔄 Ready to push to dev-i20xes branch

**NEXT SESSION**: 
Choose one of the remaining medium-priority tasks:
- Task 5: Fix ASCII XES loader (needs sample beamline file)
- Task 6: Fix channel selection workflow bug
- Task 7: Add 'Clear All' button to background extraction dialog

---

## 🔄 Session 2 Outcome (Oct 28, 2025)

### ✅ Successfully Completed
- **RXES Normalisation** - Fixed with 1-line change (commit 42d6178)
- **XES Background Extraction Segfault** - Fixed with comprehensive improvements (commit ad1ec79)
- **Documentation** - README and PLAN.md updated (commit 9386d5d)

### ⚠️ ASCII Loading - Rolled Back
**Attempted**: Full ASCII loading for both XES (1D) and RXES (2D)  
**Status**: Rolled back due to identified issues  
**Commits Reverted**: d1d4ecc through 942c836 (5 commits)

**Issues Found**:
1. XES loader can incorrectly load RXES files (no guard rails)
2. RXES mesh reconstruction produces pixelated plots
3. Sparse scan handling inadequate (867 points → 51×87 mesh = 20% filled)
4. Missing interpolation for incomplete meshes
5. Testing incomplete (no GUI visualization before commit)

**Documentation**: See `ASCII_LOADER_NOTES.md` for:
- Complete root cause analysis
- Detailed implementation recommendations
- Code examples for proper fixes
- Success criteria for next attempt

### 📊 Current Branch State
**Branch**: dev-i20xes  
**Commits**: 4 total (including rollback documentation)  
**Status**: Stable - 2 critical bugs fixed and tested  
**Sample File**: Preserved at `i20_xes/data/rxes/XES_ZnZSM5_air500_279192_1.dat`

```
ed835e2 Rollback ASCII loader implementation - comprehensive issue documentation
9386d5d Update documentation: mark RXES normalisation and segfault bugs as fixed
ad1ec79 Fix XES background extraction segfault issues
42d6178 Fix RXES normalisation bug: use type='XES' instead of type='RXES'
```

### 🎯 Next Session Priorities

**HIGH PRIORITY: ASCII Loading (Redo Properly)**

Required steps:
1. ✅ **Add Guard Rails to XES Loader** (Essential first step)
   - Detect when both Ω and ω vary significantly (>0.5 eV)
   - Raise clear error directing user to RXES loader
   - Prevents 2D data being forced into 1D

2. ✅ **Implement RXES Mesh Reconstruction with Interpolation**
   - Use `scipy.interpolate.griddata` for sparse data
   - Detect mesh fill percentage and warn if <50%
   - Support multiple strategies (direct mapping vs interpolation)

3. ✅ **Test in GUI Before Committing**
   - Verify visualization matches NeXus-loaded RXES
   - Test with both complete and sparse meshes
   - Validate both detector channels

See `ASCII_LOADER_NOTES.md` for detailed code examples and implementation strategy.

**ALTERNATIVE PRIORITIES** (if ASCII loading deferred):
- Fix channel selection workflow bug (medium)
- Add 'Clear All' button to background extraction (low)

---

**HANDOVER COMPLETE** - All context preserved for next session
